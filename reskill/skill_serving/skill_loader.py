"""SkillLoader: runtime skill loading and prompt assembly.

Wraps a SkillRegistry and assembles prompts dynamically at each agent step
based on trigger conditions. The format_for_prompt() method takes step context
(step_num, last_action) to determine which skills fire at the current step.
"""

import json
import logging
from typing import Optional, Dict, Any, List

from reskill.skill_serving.skill_registry import SkillRegistry, SkillModule

logger = logging.getLogger(__name__)


class SkillLoader:
    """Runtime skill loader: fires triggers and assembles per-step prompt text."""

    def __init__(self, registry: Optional[SkillRegistry] = None,
                 initial_skills_path: Optional[str] = None):
        """Initialize with a registry and optionally load seed skills.

        Args:
            registry: Existing SkillRegistry. Created empty if None.
            initial_skills_path: Path to JSON file with seed skills to load.
        """
        self.registry = registry if registry is not None else SkillRegistry()
        self._version = 0

        if initial_skills_path:
            self._load_initial_skills(initial_skills_path)

    def _load_initial_skills(self, path: str):
        """Load seed skills from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            skills = data if isinstance(data, list) else data.get('skills', [])
            for s in skills:
                self.registry.add_skill(
                    name=s['name'],
                    content=s['content'],
                    trigger_type=s.get('trigger_type', 'general'),
                    trigger_pattern=s.get('trigger_pattern'),
                    status=s.get('status', 'active'),
                    step=0,
                )
            logger.info(f"[SkillLoader] Loaded {len(skills)} seed skills "
                        f"from {path}")
        except Exception as e:
            logger.warning(f"[SkillLoader] Failed to load seed skills "
                           f"from {path}: {e}")

    def retrieve(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """Return registry reference (no per-task routing).

        Compatible with the env_manager retrieve() interface.
        """
        return {"registry": self.registry, "version": self._version}

    def format_for_prompt(self, retrieved: Dict[str, Any],
                          step_num: int = 0,
                          last_action: Optional[str] = None,
                          testing_decisions: Optional[Dict[str, bool]] = None
                          ) -> str:
        """Assemble triggered skills into prompt text.

        Args:
            retrieved: Output from retrieve() (unused, registry accessed directly).
            step_num: Current step number (0-indexed).
            last_action: Last action taken (None if step 0).
            testing_decisions: For testing skills, skill_id -> include (True/False).

        Returns:
            Formatted skill text for injection into the prompt template.
            Empty string if no skills trigger.
        """
        triggered = self.registry.get_triggered_skills(
            step_num=step_num,
            last_action=last_action,
            testing_decisions=testing_decisions,
        )
        return self._format_skills(triggered)

    @staticmethod
    def _format_skills(skills: List[SkillModule]) -> str:
        """Format a list of triggered skills into prompt text.

        Fields: when_to_use, action, examples — all explicitly labeled.
        Handles both structured content (dict) and legacy flat string content.
        """
        if not skills:
            return ""

        lines = []
        for skill in skills:
            lines.append(f"### skill: {skill.name}")
            content = skill.content
            if isinstance(content, dict):
                when_to_use = content.get("when_to_use", "")
                action = content.get("action", "")
                examples = content.get("examples", [])

                if when_to_use:
                    lines.append(f"  when_to_use: \"{when_to_use}\"")
                if action:
                    lines.append(f"  action: \"{action}\"")
                if examples:
                    lines.append("  examples:")
                    for ex in examples:
                        lines.append(f"    - \"{ex.strip()}\"")
            else:
                # Legacy flat string
                lines.append(str(content))
            lines.append("")
        return "\n".join(lines).strip()

    @property
    def version(self) -> int:
        return self._version

    def bump_version(self):
        """Increment version after a skill is accepted/rejected."""
        self._version += 1

    @property
    def prompt(self) -> str:
        """Reconstruct full prompt text from all active skills (for logging)."""
        active = self.registry.get_active_skills()
        if not active:
            return "(no active skills)"
        return self._format_skills(active)

    def __repr__(self):
        return (f"SkillLoader(v{self._version}, "
                f"registry={self.registry})")
