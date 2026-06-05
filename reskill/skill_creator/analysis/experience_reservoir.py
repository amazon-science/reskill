"""
Bounded reservoir of training trajectories with diversity guarantees.

For each unique (task, ab_condition, outcome) combination, stores at most
one episode.  This ensures the reservoir contains diverse tasks rather than
many copies of the same trajectory from repeated rollout groups.

Keys:
  task_key      - normalized task description (lowercase, stripped)
  ab_condition  - 'new' (with proposed skill changes), 'old' (baseline / no A/B test)
  outcome       - 'success' (score >= 1.0), 'failure' (score < 1.0)

Each task_key has at most 4 entries:
  (task_key, new, success), (task_key, new, failure),
  (task_key, old, success), (task_key, old, failure).

When the reservoir exceeds max_size, the oldest entries are evicted (FIFO on
insertion order via OrderedDict).

Sampling returns stratified results (e.g. 7 failures + 3 successes out of 10).
"""

import logging
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TaskGroup:
    """A group of episodes sharing the same task_key (for per-episode analysis)."""
    task_key: str
    episodes: List[Dict[str, Any]]
    has_success: bool
    has_failure: bool
    ab_conditions: Set[str] = field(default_factory=set)


def _extract_task_key(episode: Dict[str, Any]) -> str:
    """Extract a normalized task key for deduplication."""
    desc = episode.get('task_description', '')
    if not desc:
        from reskill.utils.trajectory import extract_task_description
        desc = extract_task_description(episode.get('task', ''))
    return desc.strip().lower()


class ExperienceReservoir:

    def __init__(self, max_size: int = 200, success_ratio: float = 0.3,
                 dedup_by: str = 'task'):
        """Initialize diversity-based reservoir.

        Args:
            max_size: Maximum total entries.  Each unique key holds at most
                      1 episode.
            success_ratio: Default fraction allocated to successes when
                           calling sample_stratified().
            dedup_by: Keying strategy for deduplication.
                      'task' (default) - one entry per (task_description,
                          ab_condition, outcome). Best for multi-task where
                          diversity across task types matters.
                      'episode' - one entry per traj_uid (no dedup). Best
                          for per-task training where a single task type
                          would otherwise cap the reservoir at ~2 entries.
        """
        self.max_size = max_size
        self.success_ratio = success_ratio
        self.dedup_by = dedup_by
        # OrderedDict preserves insertion order for FIFO eviction.
        self._store: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_episodes(self, episodes: List[Dict[str, Any]],
                     prompt_version: Optional[str] = None):
        """Add episodes with deduplication based on self.dedup_by.

        Evicts oldest entries when over max_size.
        """
        for ep in episodes:
            ab_condition = ep.get('ab_condition', 'old')
            success_val = ep.get('success', 0.0)
            outcome = 'success' if success_val >= 1.0 else 'failure'
            task_key = _extract_task_key(ep)

            entry = {
                'traj_uid': ep.get('traj_uid'),
                'task': ep['task'],
                'task_description': ep.get('task_description') or task_key,
                'trajectory': ep['trajectory'],
                'task_type': ep.get('task_type', ep.get('data_source', 'unknown')),
                'score': ep.get('score', 0.0),
                'success': success_val,
                'prompt_version': prompt_version,
                'ab_condition': ab_condition,
            }

            if self.dedup_by == 'episode':
                key = ep.get('traj_uid', id(ep))
            else:
                key = (task_key, ab_condition, outcome)

            # Replace existing entry (move to end) or insert new
            if key in self._store:
                del self._store[key]
            self._store[key] = entry

        # Evict oldest entries if over capacity
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def buffer(self):
        """All episodes (for backward compat with code that iterates the full reservoir)."""
        return list(self._store.values())

    def sample(self, n: int = 10,
               prompt_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return up to *n* failures, prioritizing those from *prompt_version*."""
        pool = [e for e in self._store.values() if e.get('success', 0.0) < 1.0]
        if not pool:
            return []

        if prompt_version is not None:
            matched = [e for e in pool if e.get('prompt_version') == prompt_version]
            others  = [e for e in pool if e.get('prompt_version') != prompt_version]

            if len(matched) >= n:
                return random.sample(matched, n)

            if matched:
                logger.warning(
                    "ExperienceReservoir: only %d/%d failures from prompt_version=%s, "
                    "backfilling with %d from older prompts",
                    len(matched), n, prompt_version, n - len(matched),
                )
            need = n - len(matched)
            backfill = random.sample(others, min(need, len(others)))
            return matched + backfill

        return random.sample(pool, min(n, len(pool)))

    def sample_stratified(self, n: int = 10,
                          prompt_version: Optional[str] = None,
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (successes, failures) with stratified sampling.

        Allocates slots proportional to success_ratio (default 30/70).
        Returns (success_list, failure_list).
        """
        n_success = max(0, int(n * self.success_ratio))
        n_failure = n - n_success

        # Sample failures
        failures = self.sample(n=n_failure, prompt_version=prompt_version)

        # Sample successes
        success_pool = [e for e in self._store.values()
                        if e.get('success', 0.0) >= 1.0]
        if success_pool:
            successes = random.sample(success_pool,
                                      min(n_success, len(success_pool)))
        else:
            successes = []
            # Backfill with more failures if no successes available
            extra_pool = [e for e in self._store.values()
                          if e.get('success', 0.0) < 1.0 and e not in failures]
            if extra_pool:
                extra = random.sample(extra_pool,
                                      min(n_success, len(extra_pool)))
                failures.extend(extra)

        return successes, failures

    # ------------------------------------------------------------------
    # Task-key group sampling
    # ------------------------------------------------------------------

    def sample_task_groups(
        self,
        n_groups: int = 30,
        min_success_ratio: float = 0.3,
    ) -> List[TaskGroup]:
        """Sample task_key groups for per-episode analysis.

        Groups all reservoir entries by task_key. Samples n_groups task_keys,
        prioritizing those with at least one success episode (target:
        min_success_ratio fraction of groups contain a success).

        Args:
            n_groups: Number of task_key groups to sample.
            min_success_ratio: Target fraction of groups with >=1 success.

        Returns:
            List of TaskGroup, each containing 1-6 episodes sharing a task_key.
        """
        groups_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in self._store.values():
            tk = _extract_task_key(entry)
            groups_by_key[tk].append(entry)

        if not groups_by_key:
            logger.warning("ExperienceReservoir.sample_task_groups: reservoir is empty")
            return []

        mixed_keys = []
        success_only_keys = []
        failure_only_keys = []
        for tk, eps in groups_by_key.items():
            has_s = any(e.get('success', 0.0) >= 1.0 for e in eps)
            has_f = any(e.get('success', 0.0) < 1.0 for e in eps)
            if has_s and has_f:
                mixed_keys.append(tk)
            elif has_s:
                success_only_keys.append(tk)
            else:
                failure_only_keys.append(tk)

        # Prioritize mixed keys (both outcomes) for contrastive signal
        sampled_mixed = random.sample(
            mixed_keys, min(n_groups, len(mixed_keys)))

        remaining_needed = n_groups - len(sampled_mixed)
        if remaining_needed > 0:
            # Fill remainder: prefer failure-only over success-only
            remaining_pool = failure_only_keys + success_only_keys
            sampled_rest = random.sample(
                remaining_pool, min(remaining_needed, len(remaining_pool)))
        else:
            sampled_rest = []

        sampled_keys = sampled_mixed + sampled_rest

        result = []
        for tk in sampled_keys:
            eps = groups_by_key[tk]
            result.append(TaskGroup(
                task_key=tk,
                episodes=eps,
                has_success=any(e.get('success', 0.0) >= 1.0 for e in eps),
                has_failure=any(e.get('success', 0.0) < 1.0 for e in eps),
                ab_conditions={e.get('ab_condition', 'old') for e in eps},
            ))

        logger.info(
            "ExperienceReservoir.sample_task_groups: sampled %d groups "
            "(%d mixed, %d failure-only, %d success-only, %d total episodes)",
            len(result),
            sum(1 for g in result if g.has_success and g.has_failure),
            sum(1 for g in result if not g.has_success),
            sum(1 for g in result if g.has_success and not g.has_failure),
            sum(len(g.episodes) for g in result),
        )
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def num_failures(self) -> int:
        return sum(1 for e in self._store.values() if e.get('success', 0.0) < 1.0)

    def num_successes(self) -> int:
        return sum(1 for e in self._store.values() if e.get('success', 0.0) >= 1.0)

    def num_unique_tasks(self) -> int:
        """Number of unique tasks in the reservoir."""
        return len({_extract_task_key(e) for e in self._store.values()})

    def __repr__(self) -> str:
        return (f"ExperienceReservoir(entries={len(self._store)}, "
                f"failures={self.num_failures()}, "
                f"successes={self.num_successes()}, "
                f"unique_tasks={self.num_unique_tasks()}, "
                f"max={self.max_size}, dedup={self.dedup_by})")
