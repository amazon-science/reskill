"""Version-level A/B tracker with Beta-Binomial Thompson Sampling.

Tests a bundle of skill operations (add/modify/delete) as a unit.
Per-rollout Thompson Sampling: each rollout gets either the new
skill version or the old version. See VersionABTracker.
"""

import math
import numpy as np
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VersionTestState:
    """State for a version-level A/B test (bundle of operations)."""
    version_id: str
    operations: List[Dict] = field(default_factory=list)
    snapshot: Dict = field(default_factory=dict)  # For revert on rejection
    # Beta posteriors: new version vs old version
    alpha_new: float = 1.0     # successes with new version + prior
    beta_new: float = 1.0      # failures with new version + prior
    alpha_old: float = 1.0     # successes with old version + prior
    beta_old: float = 1.0      # failures with old version + prior
    # Counters
    total_episodes: int = 0
    new_count: int = 0         # Rollouts assigned to new version
    old_count: int = 0         # Rollouts assigned to old version
    steps_elapsed: int = 0
    started_at_step: int = 0
    # Per-skill diagnostics (not for A/B decision, for optimizer context)
    per_skill_trigger_counts: Dict[str, int] = field(default_factory=dict)
    per_skill_episode_counts: Dict[str, int] = field(default_factory=dict)
    # Per-step logging
    per_step_records: List[Dict] = field(default_factory=list)
    # Per-step raw counts for M estimation
    step_data: List[Dict] = field(default_factory=list)
    # Pending buffer: accumulates outcomes within a step before flushing
    pending_new_successes: int = 0
    pending_new_total: int = 0
    pending_old_successes: int = 0
    pending_old_total: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class VersionABTracker:
    """Version-level Thompson Sampling A/B tracker.

    Tests a bundle of skill operations as a unit. Per-rollout assignment:
    each rollout independently gets new or old version via Thompson Sampling.

    Within each goal's rollouts (e.g., 8 rollouts per goal), the assignment
    is independent — giving a within-goal paired comparison that controls
    for goal difficulty.
    """

    def __init__(
        self,
        test_steps: int = 5,
        min_episodes: int = 50,
        prob_clamp_min: float = 0.15,
        prob_clamp_max: float = 0.85,
        n_thompson_samples: int = 10000,
        uniform_sampling: bool = False,
        weighting_strategy: str = 'none',
    ):
        self.test_steps = test_steps
        self.min_episodes = min_episodes
        self.prob_clamp_min = prob_clamp_min
        self.prob_clamp_max = prob_clamp_max
        self.n_thompson_samples = n_thompson_samples
        self.uniform_sampling = uniform_sampling
        self.weighting_strategy = weighting_strategy

        self._current_test: Optional[VersionTestState] = None
        self.history: List[Dict] = []

        # M estimation state (for 'dynamic' strategy)
        self._M_grid_size = 2000
        self._M_log_likelihoods = [0.0] * self._M_grid_size
        self._M_hat = float('inf')
        self._completed_test_count = 0

    @property
    def is_testing(self) -> bool:
        return self._current_test is not None

    @property
    def active_test(self) -> Optional[str]:
        return self._current_test.version_id if self._current_test else None

    def start_version_test(
        self, version_id: str, operations: List[Dict],
        snapshot: Dict, step: int = 0,
    ):
        """Begin A/B testing for a skill version."""
        if self._current_test is not None:
            logger.warning(
                f"[VersionABTracker] Already testing {self._current_test.version_id}")
            return
        self._current_test = VersionTestState(
            version_id=version_id,
            operations=operations,
            snapshot=snapshot,
            started_at_step=step,
        )
        logger.info(f"[VersionABTracker] Started version test '{version_id}' "
                     f"({len(operations)} ops) at step {step}")

    def sample_rollout_decision(self) -> bool:
        """Thompson sample: should this rollout use the new version?

        Returns True (new version) or False (old version).
        Called once per rollout before the episode begins.
        """
        if self._current_test is None:
            return True  # Not testing -> use current (new) version

        p_new = self._get_new_prob()
        return np.random.random() < p_new

    def _get_new_prob(self) -> float:
        """Compute P(new version is better) via Thompson Sampling."""
        if self._current_test is None:
            return 0.5

        if self.uniform_sampling:
            return 0.5

        st = self._current_test
        theta_new = np.random.beta(
            st.alpha_new, st.beta_new, size=self.n_thompson_samples)
        theta_old = np.random.beta(
            st.alpha_old, st.beta_old, size=self.n_thompson_samples)

        p_new = float(np.mean(theta_new > theta_old))
        return max(self.prob_clamp_min, min(self.prob_clamp_max, p_new))

    def record_episode_outcome(self, is_new_version: bool, success: float):
        """Record one episode's outcome into the pending buffer.

        Outcomes accumulate until record_step() flushes them into
        alpha/beta with the appropriate discount.
        """
        if self._current_test is None:
            return

        st = self._current_test
        st.total_episodes += 1
        is_success = 1 if success > 0.5 else 0

        if is_new_version:
            st.new_count += 1
            st.pending_new_successes += is_success
            st.pending_new_total += 1
        else:
            st.old_count += 1
            st.pending_old_successes += is_success
            st.pending_old_total += 1

    def record_skill_trigger(self, skill_name: str):
        """Record that a skill triggered in an episode (diagnostic only)."""
        if self._current_test is None:
            return
        st = self._current_test
        st.per_skill_trigger_counts[skill_name] = \
            st.per_skill_trigger_counts.get(skill_name, 0) + 1

    def _flush_pending(self) -> Tuple[float, float]:
        """Flush pending episode outcomes into alpha/beta with discount.

        Returns (w_new, w_old) discount weights applied.
        """
        st = self._current_test
        if st is None:
            return 1.0, 1.0

        new_m = st.pending_new_successes
        new_n = st.pending_new_total
        old_m = st.pending_old_successes
        old_n = st.pending_old_total

        if self.weighting_strategy == 'last_step':
            w_new, w_old = 0.0, 0.0
            st.alpha_new = 1.0 + new_m
            st.beta_new = 1.0 + (new_n - new_m)
            st.alpha_old = 1.0 + old_m
            st.beta_old = 1.0 + (old_n - old_m)
        elif self.weighting_strategy == 'dynamic':
            M = self._M_hat
            w_new = M / (M + new_n) if new_n > 0 and M != float('inf') else 1.0
            w_old = M / (M + old_n) if old_n > 0 and M != float('inf') else 1.0
            st.alpha_new = w_new * st.alpha_new + new_m
            st.beta_new = w_new * st.beta_new + (new_n - new_m)
            st.alpha_old = w_old * st.alpha_old + old_m
            st.beta_old = w_old * st.beta_old + (old_n - old_m)
        else:
            w_new, w_old = 1.0, 1.0
            st.alpha_new += new_m
            st.beta_new += (new_n - new_m)
            st.alpha_old += old_m
            st.beta_old += (old_n - old_m)

        st.step_data.append({
            'new_m': new_m, 'new_n': new_n,
            'old_m': old_m, 'old_n': old_n,
            'w_new': w_new, 'w_old': w_old,
            'M_hat': self._M_hat if self._M_hat != float('inf') else None,
        })

        st.pending_new_successes = 0
        st.pending_new_total = 0
        st.pending_old_successes = 0
        st.pending_old_total = 0

        return w_new, w_old

    def record_step(self):
        """Flush pending outcomes with discount, then increment step counter."""
        if self._current_test is None:
            return
        st = self._current_test

        w_new, w_old = self._flush_pending()
        st.steps_elapsed += 1

        new_mean = st.alpha_new / (st.alpha_new + st.beta_new)
        old_mean = st.alpha_old / (st.alpha_old + st.beta_old)
        latest_sd = st.step_data[-1] if st.step_data else {}

        st.per_step_records.append({
            'step': st.steps_elapsed,
            'total_episodes': st.total_episodes,
            'new_count': st.new_count,
            'old_count': st.old_count,
            'new_mean': new_mean,
            'old_mean': old_mean,
            'p_new': self._get_new_prob(),
            'w_new': w_new,
            'w_old': w_old,
            'M_hat': latest_sd.get('M_hat'),
            'step_new_m': latest_sd.get('new_m', 0),
            'step_new_n': latest_sd.get('new_n', 0),
            'step_old_m': latest_sd.get('old_m', 0),
            'step_old_n': latest_sd.get('old_n', 0),
        })

    def should_decide(self) -> bool:
        """True if enough steps and episodes have been collected."""
        if self._current_test is None:
            return False
        st = self._current_test
        return (st.steps_elapsed >= self.test_steps
                and st.total_episodes >= self.min_episodes)

    def decide(self) -> Tuple[bool, Dict]:
        """Accept/reject the version based on posterior means.

        Accepts if new version posterior mean > old version posterior mean.

        Returns:
            (accepted, stats) tuple. Resets state to None.
        """
        if self._current_test is None:
            return False, {}

        st = self._current_test
        new_mean = st.alpha_new / (st.alpha_new + st.beta_new)
        old_mean = st.alpha_old / (st.alpha_old + st.beta_old)

        accepted = new_mean > old_mean

        # Update M estimate for dynamic strategy
        if self.weighting_strategy == 'dynamic' and st.step_data:
            self._score_test(st.step_data)
            self._update_M_estimate()

        stats = {
            'version_id': st.version_id,
            'accepted': accepted,
            'new_mean': new_mean,
            'old_mean': old_mean,
            'new_count': st.new_count,
            'old_count': st.old_count,
            'total_episodes': st.total_episodes,
            'steps_elapsed': st.steps_elapsed,
            'new_posterior': f'Beta({st.alpha_new:.1f}, {st.beta_new:.1f})',
            'old_posterior': f'Beta({st.alpha_old:.1f}, {st.beta_old:.1f})',
            'p_new_final': self._get_new_prob(),
            'operations': st.operations,
            'snapshot': st.snapshot,
            'per_skill_trigger_counts': st.per_skill_trigger_counts,
            'per_step_records': st.per_step_records,
            'weighting_strategy': self.weighting_strategy,
            'M_hat': self._M_hat if self._M_hat != float('inf') else None,
            'step_data': st.step_data,
        }

        self.history.append(stats)
        logger.info(
            f"[VersionABTracker] Decision for {st.version_id}: "
            f"{'ACCEPTED' if accepted else 'REJECTED'} "
            f"(new={new_mean:.3f} vs old={old_mean:.3f}, "
            f"episodes={st.total_episodes}, "
            f"weighting={self.weighting_strategy})")

        self._current_test = None
        return accepted, stats

    # ------------------------------------------------------------------
    # M estimation (dynamic weighting)
    # ------------------------------------------------------------------

    @staticmethod
    def _betabinom_logpmf(k: int, n: int, a: float, b: float) -> float:
        """Log PMF of BetaBinomial(n, a, b) at k."""
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                + math.lgamma(k + a) + math.lgamma(n - k + b)
                - math.lgamma(n + a + b)
                - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))

    def _score_test(self, step_data_list: List[Dict]):
        """Score one completed test against the M grid and accumulate."""
        for i in range(self._M_grid_size):
            M = float(i + 1)  # M grid is [1, 2000]
            a_new, b_new = 1.0, 1.0
            a_old, b_old = 1.0, 1.0
            ll = 0.0

            for sd in step_data_list:
                new_m, new_n = sd['new_m'], sd['new_n']
                old_m, old_n = sd['old_m'], sd['old_n']

                w_new = M / (M + new_n) if new_n > 0 else 1.0
                w_old = M / (M + old_n) if old_n > 0 else 1.0

                a_new_d = max(w_new * a_new, 1e-6)
                b_new_d = max(w_new * b_new, 1e-6)
                a_old_d = max(w_old * a_old, 1e-6)
                b_old_d = max(w_old * b_old, 1e-6)

                if new_n > 0:
                    ll += self._betabinom_logpmf(new_m, new_n, a_new_d, b_new_d)
                if old_n > 0:
                    ll += self._betabinom_logpmf(old_m, old_n, a_old_d, b_old_d)

                a_new = a_new_d + new_m
                b_new = b_new_d + (new_n - new_m)
                a_old = a_old_d + old_m
                b_old = b_old_d + (old_n - old_m)

            self._M_log_likelihoods[i] += ll

        self._completed_test_count += 1

    def _update_M_estimate(self):
        """Pick best M from cumulative log-likelihoods. Needs >= 2 tests."""
        if self._completed_test_count < 2:
            return

        best_idx = 0
        best_ll = self._M_log_likelihoods[0]
        for i in range(1, self._M_grid_size):
            if self._M_log_likelihoods[i] > best_ll:
                best_ll = self._M_log_likelihoods[i]
                best_idx = i

        self._M_hat = float(best_idx + 1)
        logger.info(f"[VersionABTracker] M_hat updated to {self._M_hat:.0f} "
                    f"(from {self._completed_test_count} tests)")

    def get_snapshot(self) -> Optional[Dict]:
        """Get the revert snapshot for the current test."""
        if self._current_test is None:
            return None
        return self._current_test.snapshot

    def cancel_test(self):
        """Cancel current test without deciding."""
        if self._current_test:
            logger.info(f"[VersionABTracker] Cancelled test for "
                        f"{self._current_test.version_id}")
            self._current_test = None

    def get_test_metrics(self) -> Dict:
        """Return current test metrics for W&B logging."""
        if self._current_test is None:
            return {
                'skills/version_ab/is_testing': 0,
                'skills/version_ab/version_id': '',
            }
        st = self._current_test
        new_mean = st.alpha_new / (st.alpha_new + st.beta_new)
        old_mean = st.alpha_old / (st.alpha_old + st.beta_old)
        metrics = {
            'skills/version_ab/is_testing': 1,
            'skills/version_ab/version_id': st.version_id,
            'skills/version_ab/steps_elapsed': st.steps_elapsed,
            'skills/version_ab/total_episodes': st.total_episodes,
            'skills/version_ab/new_count': st.new_count,
            'skills/version_ab/old_count': st.old_count,
            'skills/version_ab/new_mean': new_mean,
            'skills/version_ab/old_mean': old_mean,
            'skills/version_ab/p_new': self._get_new_prob(),
            'skills/version_ab/M_hat': self._M_hat if self._M_hat != float('inf') else -1.0,
        }
        if st.step_data:
            latest = st.step_data[-1]
            metrics['skills/version_ab/w_new'] = latest.get('w_new', 1.0)
            metrics['skills/version_ab/w_old'] = latest.get('w_old', 1.0)
        return metrics

    def save(self, path: str):
        """Save tracker state to JSON."""
        import json
        data = {
            'test_steps': self.test_steps,
            'min_episodes': self.min_episodes,
            'current_test': self._current_test.to_dict() if self._current_test else None,
            'history': self.history,
            'weighting_strategy': self.weighting_strategy,
            'M_hat': self._M_hat if self._M_hat != float('inf') else None,
            'M_log_likelihoods': self._M_log_likelihoods,
            'completed_test_count': self._completed_test_count,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[VersionABTracker] Saved state to {path}")

    def load(self, path: str):
        """Load tracker state from JSON."""
        import json
        with open(path) as f:
            data = json.load(f)
        self.history = data.get('history', [])

        # Load M estimation state (backward-compat: missing = fresh)
        m_hat = data.get('M_hat')
        self._M_hat = float('inf') if m_hat is None else float(m_hat)
        saved_ll = data.get('M_log_likelihoods')
        if saved_ll and len(saved_ll) == self._M_grid_size:
            self._M_log_likelihoods = saved_ll
        self._completed_test_count = data.get('completed_test_count', 0)

        ct = data.get('current_test')
        if ct:
            ct.pop('per_step_records', None)
            self._current_test = VersionTestState(**{
                k: v for k, v in ct.items()
                if k in VersionTestState.__dataclass_fields__
            })
        else:
            self._current_test = None
        logger.info(f"[VersionABTracker] Loaded state from {path} "
                    f"(testing={self.active_test}, M_hat={self._M_hat})")

    def __repr__(self) -> str:
        if self._current_test:
            st = self._current_test
            return (f"VersionABTracker(testing={st.version_id}, "
                    f"step={st.steps_elapsed}/{self.test_steps}, "
                    f"episodes={st.total_episodes}/{self.min_episodes})")
        return f"VersionABTracker(idle, history={len(self.history)} tests)"
