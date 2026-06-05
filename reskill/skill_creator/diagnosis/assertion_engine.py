"""
Rule-based assertion engine for agent trajectories.

Assertions start empty and grow via the BatchDiagnoser LLM. Each assertion is a
rule-based check (regex/string ops) that evaluates to (passed, evidence)
per trajectory — O(n) per trajectory, no LLM calls.
"""

import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check type implementations
# ---------------------------------------------------------------------------

def _first_match_idx(trajectory: List[dict], pattern: str) -> int:
    """Return index of first step whose action matches regex, or len(traj)."""
    for i, step in enumerate(trajectory):
        if re.search(pattern, step.get('action', '')):
            return i
    return len(trajectory)


def _check_action_exists(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    pattern = params['pattern']
    for i, step in enumerate(trajectory):
        if re.search(pattern, step.get('action', '')):
            return True, f"Step {i}: action matches '{pattern}'"
    return False, f"No action matches '{pattern}'"


def _check_action_not_exists(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    pattern = params['pattern']
    for i, step in enumerate(trajectory):
        if re.search(pattern, step.get('action', '')):
            return False, f"Step {i}: action matches '{pattern}' (should not exist)"
    return True, f"No action matches '{pattern}'"


def _check_action_before(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    first_idx = _first_match_idx(trajectory, params['first'])
    second_idx = _first_match_idx(trajectory, params['second'])
    if first_idx >= len(trajectory):
        return False, f"Pattern '{params['first']}' not found"
    if second_idx >= len(trajectory):
        return False, f"Pattern '{params['second']}' not found"
    if first_idx < second_idx:
        return True, f"'{params['first']}' at step {first_idx} before '{params['second']}' at step {second_idx}"
    return False, f"'{params['first']}' at step {first_idx} NOT before '{params['second']}' at step {second_idx}"


def _check_no_consecutive_repeat(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    prefix = params.get('prefix', '')
    prev_action = None
    for i, step in enumerate(trajectory):
        action = step.get('action', '')
        if prefix and not action.startswith(prefix):
            prev_action = None
            continue
        if action == prev_action:
            return False, f"Steps {i-1} and {i}: repeated '{action[:60]}...'"
        prev_action = action
    return True, "No consecutive repeated actions"


def _check_final_action(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    if not trajectory:
        return False, "Empty trajectory"
    pattern = params['pattern']
    last_action = trajectory[-1].get('action', '')
    if re.search(pattern, last_action):
        return True, f"Final action matches '{pattern}'"
    return False, f"Final action '{last_action[:60]}...' does not match '{pattern}'"


def _check_min_steps(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    n = params['n']
    actual = len(trajectory)
    if actual >= n:
        return True, f"{actual} steps >= {n}"
    return False, f"{actual} steps < {n}"


def _check_max_steps(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    n = params['n']
    actual = len(trajectory)
    if actual <= n:
        return True, f"{actual} steps <= {n}"
    return False, f"{actual} steps > {n}"


def _check_action_count(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    pattern = params['pattern']
    min_count = params.get('min', 0)
    max_count = params.get('max', 999)
    count = sum(1 for step in trajectory if re.search(pattern, step.get('action', '')))
    if min_count <= count <= max_count:
        return True, f"{count} actions match '{pattern}' (range [{min_count}, {max_count}])"
    return False, f"{count} actions match '{pattern}' (outside [{min_count}, {max_count}])"


def _check_observation_contains(trajectory: List[dict], params: dict) -> Tuple[bool, str]:
    pattern = params['pattern']
    after_action = params.get('after_action')
    look = False if after_action else True
    for i, step in enumerate(trajectory):
        if after_action and re.search(after_action, step.get('action', '')):
            look = True
            continue
        if look and re.search(pattern, step.get('observation', '')):
            return True, f"Step {i}: observation matches '{pattern}'"
    return False, f"No observation matches '{pattern}'"


CHECK_TYPES = {
    'action_exists': _check_action_exists,
    'action_not_exists': _check_action_not_exists,
    'action_before': _check_action_before,
    'no_consecutive_repeat': _check_no_consecutive_repeat,
    'final_action': _check_final_action,
    'min_steps': _check_min_steps,
    'max_steps': _check_max_steps,
    'action_count': _check_action_count,
    'observation_contains': _check_observation_contains,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AssertionEngine:
    """Rule-based assertion engine: manages and grades behavioral assertions.

    Assertions start empty and are added/modified/deleted by the BatchDiagnoser
    LLM. Grading is O(n) per trajectory — no LLM calls.
    """

    def __init__(self):
        self.assertions: List[dict] = []
        self._version = 0
        # Running pass rate tracking (reset after each save)
        self._pass_counts: Dict[str, int] = {}
        self._eval_counts: Dict[str, int] = {}

    def grade_episode(self, episode: dict) -> dict:
        """Grade one episode against all current assertions.

        Args:
            episode: Dict with at least 'trajectory' (list of {action, observation})
                     and 'traj_uid'.

        Returns:
            EpisodeGrade dict with assertion_results and pass_rate.
        """
        trajectory = episode.get('trajectory', [])
        traj_uid = episode.get('traj_uid', 'unknown')
        results = []

        for assertion in self.assertions:
            check_fn = CHECK_TYPES.get(assertion['check_type'])
            if check_fn is None:
                continue
            try:
                passed, evidence = check_fn(trajectory, assertion['params'])
            except Exception as e:
                passed, evidence = False, f"Error: {e}"

            aid = assertion['id']
            results.append({
                'assertion_id': aid,
                'passed': passed,
                'evidence': evidence,
            })

            # Update running counts
            self._eval_counts[aid] = self._eval_counts.get(aid, 0) + 1
            if passed:
                self._pass_counts[aid] = self._pass_counts.get(aid, 0) + 1

        pass_rate = None
        if results:
            pass_rate = sum(1 for r in results if r['passed']) / len(results)

        return {
            'traj_uid': traj_uid,
            'assertion_results': results,
            'pass_rate': pass_rate,
        }

    def grade_batch(self, episodes: List[dict]) -> List[dict]:
        """Grade a batch of episodes."""
        return [self.grade_episode(ep) for ep in episodes]

    def get_pass_rates(self) -> Dict[str, float]:
        """Get aggregate pass rate per assertion from recent grading."""
        rates = {}
        for assertion in self.assertions:
            aid = assertion['id']
            evaluated = self._eval_counts.get(aid, 0)
            passed = self._pass_counts.get(aid, 0)
            rates[aid] = passed / evaluated if evaluated > 0 else None
        return rates

    def get_assertions_summary(self) -> str:
        """Format current assertions + pass rates + params for LLM context."""
        if not self.assertions:
            return "No assertions defined yet."
        rates = self.get_pass_rates()
        lines = []
        for a in self.assertions:
            aid = a['id']
            rate = rates.get(aid)
            rate_str = f"{rate:.2f}" if rate is not None else "N/A"
            evaluated = self._eval_counts.get(aid, 0)
            params_str = json.dumps(a.get('params', {}))
            lines.append(
                f"- **{aid}** (type={a['check_type']}, params={params_str}, "
                f"pass_rate={rate_str}, n={evaluated}): {a['description']}"
            )
        return "\n".join(lines)

    def reset_pass_rates(self):
        """Reset running pass rate counters (call after each analyzer cycle)."""
        self._pass_counts.clear()
        self._eval_counts.clear()

    # ------------------------------------------------------------------
    # CRUD operations (driven by BatchDiagnoser output)
    # ------------------------------------------------------------------

    def apply_crud(self, operations: List[dict], current_step: int = 0) -> dict:
        """Apply add/delete/modify operations from the BatchDiagnoser.

        Args:
            operations: List of {"op": "add"|"delete"|"modify", ...}
            current_step: Training step for tagging created_at_step.

        Returns:
            Summary dict with counts of added/deleted/modified.
        """
        added, deleted, modified = 0, 0, 0

        for op in operations:
            op_type = op.get('op', '')

            if op_type == 'add':
                check_type = op.get('check_type', '')
                if check_type not in CHECK_TYPES:
                    logger.warning(f"[AssertionEngine] Unknown check_type '{check_type}', skipping add")
                    continue
                if len(self.assertions) >= 15:
                    logger.warning(f"[AssertionEngine] Max assertions (15) reached, skipping add")
                    continue
                new_id = op.get('id', f"{check_type}_{uuid.uuid4().hex[:6]}")
                self.assertions.append({
                    'id': new_id,
                    'check_type': check_type,
                    'params': op.get('params', {}),
                    'description': op.get('description', ''),
                    'created_at_step': current_step,
                    'created_by': 'analyzer',
                    'version': 1,
                })
                added += 1

            elif op_type == 'delete':
                target_id = op.get('assertion_id', '')
                before = len(self.assertions)
                self.assertions = [a for a in self.assertions if a['id'] != target_id]
                if len(self.assertions) < before:
                    deleted += 1

            elif op_type == 'modify':
                target_id = op.get('assertion_id', '')
                for a in self.assertions:
                    if a['id'] == target_id:
                        if 'params' in op:
                            a['params'] = op['params']
                        if 'description' in op:
                            a['description'] = op['description']
                        if 'check_type' in op and op['check_type'] in CHECK_TYPES:
                            a['check_type'] = op['check_type']
                        a['version'] = a.get('version', 1) + 1
                        modified += 1
                        break

        self._version += 1
        summary = {'added': added, 'deleted': deleted, 'modified': modified}
        logger.info(f"[AssertionEngine] CRUD applied: {summary}, "
                    f"total assertions: {len(self.assertions)}")
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str, step: int = 0):
        """Save assertions to versioned JSON file."""
        os.makedirs(save_dir, exist_ok=True)
        data = {
            'version': self._version,
            'step': step,
            'assertions': self.assertions,
        }
        versioned_path = os.path.join(
            save_dir, f'assertions_v{self._version}_step{step}.json')
        latest_path = os.path.join(save_dir, 'assertions_latest.json')
        for path in [versioned_path, latest_path]:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        logger.info(f"[AssertionEngine] Saved {len(self.assertions)} assertions to {versioned_path}")

    def load(self, path: str):
        """Load assertions from JSON file."""
        if not os.path.exists(path):
            logger.info(f"[AssertionEngine] No assertions file at {path}, starting empty")
            return
        with open(path, 'r') as f:
            data = json.load(f)
        self.assertions = data.get('assertions', [])
        self._version = data.get('version', 0)
        logger.info(f"[AssertionEngine] Loaded {len(self.assertions)} assertions "
                    f"(v{self._version}) from {path}")
