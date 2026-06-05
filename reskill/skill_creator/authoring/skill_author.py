"""Stage 4: Skill content writing from recommendations (~2-3K tok input).

Takes the recommender's operations and writes actual skill content. Input is
scoped: only referenced skills + referenced insights with action snippets.

Output format matches the expected schema for downstream Verify and A/B Test
phases (version_reasoning + operations list).
"""

import json
import logging
import os
from typing import Dict, List, Optional

from reskill.utils.llm_client import LLMClient, extract_json

logger = logging.getLogger(__name__)

_PROMPTS_V2_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts"
)

_TRIGGER_TYPES_SECTION = (
    "## Trigger Types\n\n"
    "1. **general** -- Always included at every step. For broad principles.\n"
    "2. **beginning** -- Included only at step 0. For initialization strategies.\n"
    "3. **action_pattern** -- Included only after the agent takes an action "
    "matching a regex pattern. The regex is matched against the agent's "
    "**raw action string**, NOT reasoning text. Design your regex based on "
    "the environment's action vocabulary and the "
    "`trigger_relevant_actions` in referenced insights."
)

_GENERAL_ONLY_SECTION = (
    "## Trigger Types\n\n"
    "All skills use **general** trigger type -- they are included at every "
    "step. Set `\"trigger_type\": \"general\"` and omit `trigger_pattern` "
    "for every skill."
)


def _load_prompt(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"[SkillAuthor] Prompt not found: {path}")
        return ""


def _normalize_content(content) -> dict:
    """Ensure content is a structured dict with when_to_use/action/examples.

    Accepts either:
    - A dict with the structured fields (pass-through)
    - A plain string (wrap as action-only)
    - Legacy dict with "scenario"/"instruction"/"description" keys (migrate)
    """
    if isinstance(content, dict):
        # Migrate legacy fields -> new format
        if "scenario" in content and "when_to_use" not in content:
            content["when_to_use"] = content.pop("scenario")
        if "instruction" in content and "when_to_use" not in content:
            content["when_to_use"] = content.pop("instruction")
        # Drop description — redundant with action
        content.pop("description", None)
        content.setdefault("when_to_use", "")
        content.setdefault("action", "")
        content.setdefault("examples", [])
        # Ensure list
        if isinstance(content["examples"], str):
            content["examples"] = [content["examples"]]
        return content
    # Legacy flat string — wrap as action-only
    return {
        "when_to_use": "",
        "action": str(content),
        "examples": [],
    }


class SkillAuthor:
    """Stage 4: Skill content writing from recommendations."""

    def __init__(
        self,
        client: LLMClient,
        prompt_path: Optional[str] = None,
        force_general_trigger: bool = False,
    ):
        self.client = client
        self.force_general_trigger = force_general_trigger
        self._system_prompt = _load_prompt(
            prompt_path or os.path.join(_PROMPTS_V2_DIR, "skill_author.md")
        )
        if force_general_trigger:
            self._system_prompt = self._system_prompt.replace(
                _TRIGGER_TYPES_SECTION, _GENERAL_ONLY_SECTION)

    def author(
        self,
        recommendations: dict,
        insights: List[dict],
        skill_registry=None,
        skill_library_summary: str = "",
        save_dir: Optional[str] = None,
        step: int = 0,
        retry_feedback: Optional[List[dict]] = None,
        action_vocabulary: str = "",
    ) -> Optional[dict]:
        """Write full skill definitions from recommendations.

        Args:
            recommendations: Output from SkillRecommender.recommend().
            insights: All insights from EpisodeAnalyzer.
            skill_registry: For resolving skill names and getting content.
            skill_library_summary: Skill library with trigger rates/ages.
            save_dir: Optional directory to save prompts/responses.
            step: Training step number.
            retry_feedback: List of dicts from previous failed verify attempt,
                each with {op_type, name, reason, trigger_pattern, measured_rate,
                threshold}. Tells the author what went wrong so it can fix it.
            action_vocabulary: Environment-specific action template description.
                Helps the author write trigger patterns that match real actions.

        Returns:
            Dict with version_reasoning + operations,
            or None on failure.
        """
        prompt = self._build_prompt(
            recommendations, insights, skill_registry, skill_library_summary,
            retry_feedback=retry_feedback,
            action_vocabulary=action_vocabulary)

        response = self.client.generate(
            user_prompt=prompt,
            system_prompt=self._system_prompt,
            max_tokens=4096,
            temperature=0.3,
        )

        if save_dir:
            self._save(save_dir, step, prompt, response)

        if response is None:
            logger.warning("[SkillAuthor] LLM returned no response")
            return None

        result = self._parse_response(response, skill_registry)
        if result is not None:
            logger.info(
                f"[SkillAuthor] Produced {len(result.get('operations', []))} operations")
        return result

    def _build_prompt(
        self,
        recommendations: dict,
        insights: List[dict],
        skill_registry=None,
        skill_library_summary: str = "",
        retry_feedback: Optional[List[dict]] = None,
        action_vocabulary: str = "",
    ) -> str:
        sections = []

        # Retry feedback from previous failed verify attempt
        if retry_feedback:
            sections.append("## Previous Attempt Failed\n")
            sections.append(
                "Your previous proposal was rejected. Fix the issues below "
                "and resubmit. Write BROADER trigger patterns that match "
                "common action classes, not specific objects.\n")
            for fb in retry_feedback:
                reason = fb.get("reason", "unknown")
                if reason == "trigger_rate_too_low":
                    pat = fb.get("trigger_pattern", "?")
                    rate = fb.get("measured_rate", 0)
                    threshold = fb.get("threshold", 0.5)
                    name = fb.get("name", "?")
                    sections.append(
                        f"- **{fb.get('op_type', '?').upper()} '{name}'**: "
                        f"pattern `{pat}` matched only "
                        f"{rate:.0%} of episodes (need {threshold:.0%}). "
                        f"Write a broader pattern.")
                elif reason == "content_too_long":
                    name = fb.get("name", "?")
                    chars = fb.get("char_count", 0)
                    limit = fb.get("char_limit", 500)
                    sections.append(
                        f"- **{fb.get('op_type', '?').upper()} '{name}'**: "
                        f"content is {chars} chars (limit {limit}). "
                        f"Shorten the content.")
            sections.append("")

        # Recommendations
        sections.append("## Recommendations from Skill Recommender\n")
        reasoning = recommendations.get("reasoning", "")
        if reasoning:
            sections.append(f"Strategy: {reasoning}\n")

        for i, op in enumerate(recommendations.get("operations", [])):
            sections.append(f"### Operation {i + 1}: {op.get('op', '?').upper()}")
            if op.get("target"):
                sections.append(f"Target: {op['target']}")
            sections.append(f"Intent: {op.get('intent', '')}")
            if op.get("evidence"):
                sections.append(f"Evidence: {op['evidence']}")
            if op.get("trigger_suggestion"):
                sections.append(f"Trigger type suggestion: {op['trigger_suggestion']}")
            sections.append("")

        # Referenced skills (for modify/delete)
        if skill_registry:
            referenced_skills = []
            for op in recommendations.get("operations", []):
                if op.get("op") in ("modify", "delete") and op.get("target"):
                    skill = skill_registry.get_skill_by_name(op["target"])
                    if skill:
                        referenced_skills.append(skill)

            if referenced_skills:
                sections.append("## Referenced Skills (full content)\n")
                for skill in referenced_skills:
                    sections.append(f"### {skill.name}")
                    sections.append(f"Trigger: {skill.trigger_type}")
                    if skill.trigger_pattern:
                        sections.append(f"Pattern: `{skill.trigger_pattern}`")
                    content = skill.content
                    if isinstance(content, dict):
                        sections.append(f"Content: {json.dumps(content, indent=2)}")
                    else:
                        sections.append(f"Content: {content}")
                    sections.append("")

        # Referenced insights with action snippets (cap at 3 per operation)
        all_referenced = set()
        for op in recommendations.get("operations", []):
            refs = op.get("referenced_insights", [])
            for idx in refs[:3]:
                if isinstance(idx, int) and 0 <= idx < len(insights):
                    all_referenced.add(idx)

        if all_referenced:
            sections.append("## Referenced Insights (with action snippets)\n")
            for idx in sorted(all_referenced):
                ins = insights[idx]
                sections.append(f"### Insight {idx}")
                task_query = ins.get("task_query")
                if task_query:
                    sections.append(f"Task: {task_query}")
                sections.append(f"Finding: {ins.get('insight', '')}")

                f_snippet = ins.get("failure_snippet")
                if f_snippet and isinstance(f_snippet, list):
                    sections.append(f"Failure actions: {' -> '.join(str(a) for a in f_snippet[:6])}")
                f_point = ins.get("failure_point")
                if f_point:
                    sections.append(f"Failure point: {f_point}")

                s_snippet = ins.get("success_snippet")
                if s_snippet and isinstance(s_snippet, list):
                    sections.append(f"Success actions: {' -> '.join(str(a) for a in s_snippet[:6])}")
                s_pattern = ins.get("success_pattern")
                if s_pattern:
                    sections.append(f"Success pattern: {s_pattern}")


                sections.append("")

        # Skill library with trigger rates
        if skill_library_summary:
            sections.append("## Current Skill Library (with trigger rates)\n")
            sections.append(skill_library_summary)
            sections.append("")

        # Action vocabulary (environment-specific)
        if action_vocabulary:
            sections.append("## Environment Action Vocabulary\n")
            sections.append(action_vocabulary)
            sections.append("")

        # Format spec
        sections.append(
            "## Skill Format Spec\n\n"
            "Each skill has:\n"
            "- `name`: short-kebab-case\n"
            "- `trigger_type`: general | beginning | action_pattern\n"
            "- `trigger_pattern`: regex string (only for action_pattern)\n"
            "- `content`: {when_to_use, action, examples}\n"
            "  - when_to_use: 1 sentence, at most 25 words\n"
            "  - action: 1-2 sentences, at most 50 words\n"
            "  - examples: at most 1 DO/DON'T example, maximum 2 lines\n\n"
            "**Length limit**: Total content (when_to_use + action + examples) "
            "must be at most 500 characters. Be concise.\n"
        )

        sections.append(
            "## Your Task\n\n"
            "Execute each recommended operation. Write full skill definitions "
            "grounded in the referenced insights. "
            "Return JSON as specified in your instructions."
        )

        return "\n".join(sections)

    def _parse_response(self, text: str, skill_registry=None) -> Optional[dict]:
        """Parse author response into version proposal."""
        result = extract_json(text)
        if result is None:
            logger.warning("[SkillAuthor] Failed to parse JSON")
            return None

        if "operations" not in result:
            logger.warning("[SkillAuthor] Missing 'operations' key")
            return None

        valid_ops = []
        for op in result["operations"]:
            op_type = op.get("type")

            if op_type == "add":
                skill = op.get("skill", {})
                if not skill.get("name") or not skill.get("content"):
                    logger.warning("[SkillAuthor] ADD missing name or content")
                    continue
                trigger_type = skill.get("trigger_type", "general")
                if self.force_general_trigger:
                    skill["trigger_type"] = "general"
                    skill["trigger_pattern"] = None
                elif trigger_type not in ("general", "beginning", "action_pattern"):
                    logger.warning(
                        f"[SkillAuthor] Invalid trigger_type '{trigger_type}', "
                        f"defaulting to general")
                    skill["trigger_type"] = "general"
                elif trigger_type == "action_pattern" and not skill.get("trigger_pattern"):
                    skill["trigger_type"] = "general"
                    skill["trigger_pattern"] = None
                skill["content"] = _normalize_content(skill["content"])
                valid_ops.append(op)

            elif op_type == "modify":
                target = op.get("target_skill_name")
                if not target:
                    logger.warning("[SkillAuthor] MODIFY missing target_skill_name")
                    continue
                if skill_registry and not skill_registry.get_skill_by_name(target):
                    logger.warning(
                        f"[SkillAuthor] MODIFY target '{target}' not found")
                    continue
                changes = op.get("changes", {})
                if not changes:
                    logger.warning("[SkillAuthor] MODIFY with empty changes")
                    continue
                if "content" in changes:
                    changes["content"] = _normalize_content(
                        changes["content"])
                valid_ops.append(op)

            elif op_type == "delete":
                target = op.get("target_skill_name")
                if not target:
                    logger.warning("[SkillAuthor] DELETE missing target_skill_name")
                    continue
                if skill_registry and not skill_registry.get_skill_by_name(target):
                    logger.warning(
                        f"[SkillAuthor] DELETE target '{target}' not found")
                    continue
                valid_ops.append(op)

        if not valid_ops:
            logger.warning("[SkillAuthor] No valid operations after validation")
            return None

        result["operations"] = valid_ops
        return result

    def _save(self, save_dir: str, step: int, prompt: str, response: Optional[str]):
        try:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"_step{step}"
            with open(os.path.join(save_dir, f"author_prompt{suffix}.md"), "w") as f:
                f.write(prompt)
            if response:
                with open(os.path.join(save_dir, f"author_response{suffix}.md"), "w") as f:
                    f.write(response)
        except Exception as e:
            logger.warning(f"[SkillAuthor] Failed to save: {e}")
