"""Stateless trigger evaluation for skill modules."""

import re
import logging
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .skill_registry import SkillModule

logger = logging.getLogger(__name__)

# Pre-compile cache to avoid re-compiling regexes every step
_PATTERN_CACHE = {}


def _get_compiled(pattern: str):
    """Get or compile a regex pattern."""
    if pattern not in _PATTERN_CACHE:
        try:
            _PATTERN_CACHE[pattern] = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"[TriggerMatcher] Invalid regex '{pattern}': {e}")
            _PATTERN_CACHE[pattern] = None
    return _PATTERN_CACHE[pattern]


class TriggerMatcher:
    """Evaluates which skills should fire given current step state."""

    @staticmethod
    def should_trigger(skill: 'SkillModule', step_num: int,
                       last_action: Optional[str]) -> bool:
        """Check if a skill's trigger condition is met.

        Args:
            skill: The skill module to check.
            step_num: Current step number (0-indexed).
            last_action: The last action taken (None if step 0 / no prior action).

        Returns:
            True if the skill should be included in the prompt.
        """
        if skill.trigger_type == "general":
            return True
        if skill.trigger_type == "beginning":
            return step_num == 0
        if skill.trigger_type == "action_pattern":
            if not last_action or not skill.trigger_pattern:
                return False
            compiled = _get_compiled(skill.trigger_pattern)
            if compiled is None:
                return False
            return bool(compiled.search(last_action))
        logger.warning(f"[TriggerMatcher] Unknown trigger_type: {skill.trigger_type}")
        return False

    @staticmethod
    def get_triggered_skill_ids(skills: List['SkillModule'], step_num: int,
                                last_action: Optional[str]) -> List[str]:
        """Return IDs of all skills whose triggers fire."""
        return [
            s.skill_id for s in skills
            if TriggerMatcher.should_trigger(s, step_num, last_action)
        ]

    @staticmethod
    def estimate_trigger_rate(trigger_type: str, trigger_pattern: Optional[str],
                              trajectories: List[dict]) -> float:
        """Estimate what fraction of episodes would trigger a proposed skill.

        Simulates trigger evaluation against stored trajectories. An episode
        "triggers" if any step in its trajectory would fire the condition.

        Args:
            trigger_type: "general", "beginning", or "action_pattern".
            trigger_pattern: Regex string (for action_pattern type).
            trajectories: List of episode dicts with 'trajectory' key,
                where trajectory is [{action, observation}, ...].

        Returns:
            Fraction of episodes that would trigger (0.0 to 1.0).
        """
        if not trajectories:
            return 0.0

        if trigger_type == "general":
            return 1.0
        if trigger_type == "beginning":
            return 1.0  # Every episode has a step 0

        if trigger_type != "action_pattern" or not trigger_pattern:
            return 0.0

        compiled = _get_compiled(trigger_pattern)
        if compiled is None:
            return 0.0

        from reskill.utils.trajectory import extract_action_tag

        triggered_count = 0
        for ep in trajectories:
            steps = ep.get('trajectory', [])
            # Exclude the last action: it's the terminal action (e.g. <answer>),
            # so there is no step i+1 where the skill would actually fire.
            for i, step in enumerate(steps[:-1] if len(steps) > 1 else []):
                raw_action = step.get('action', '')
                # Parse the action the same way runtime does — extract from
                # <action>...</action> tags so regex matches the clean action
                # string, not reasoning text.
                action = extract_action_tag(raw_action) if raw_action else ''
                if action and compiled.search(action):
                    triggered_count += 1
                    break  # One trigger per episode is enough

        return triggered_count / len(trajectories)
