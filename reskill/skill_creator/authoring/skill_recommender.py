"""Stage 3: Strategic skill recommendations (~3-4K tok input).

Receives insight groups (from diagnoser), diagnosis, skill library,
and version history. Produces operation recommendations (what to change and
why) but does NOT write skill content.

Input now uses insight_groups from the diagnoser (LLM-clustered) rather than
the old deterministic aggregation summary. Only representative insights per
group are expanded, keeping input compact.
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


def _load_prompt(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"[SkillRecommender] Prompt not found: {path}")
        return ""


def _compress_version_history(
    history: List[dict],
    n_last: int = 5,
) -> str:
    """Compress version history to one-liners."""
    if not history:
        return "No version history."

    recent = history[-n_last:]
    lines = [f"Version History (last {len(recent)} of {len(history)}):"]
    for vh in recent:
        step = vh.get("step", "?")
        accepted = "ACCEPTED" if vh.get("accepted") else "REJECTED"
        new_mean = vh.get("new_mean", 0)
        old_mean = vh.get("old_mean", 0)
        ops = vh.get("operations", [])
        op_summary = ", ".join(
            f"{op.get('type', '?').upper()}: "
            f"{op.get('target_skill_name') or op.get('skill', {}).get('name', '?')}"
            for op in ops
        )
        lines.append(
            f"  Step {step}: {accepted} "
            f"(new={new_mean:.3f} vs old={old_mean:.3f}) "
            f"-- {op_summary}"
        )
    return "\n".join(lines)


def _format_insight_groups(
    insight_groups: List[dict],
    insights: List[dict],
) -> str:
    """Format insight groups with representative detail for the recommender.

    Only representative insights are expanded (full snippets/patterns).
    Other group members are listed as indices only.
    """
    if not insight_groups:
        return "No insight groups available."

    lines = [f"FAILURE PATTERN GROUPS ({len(insight_groups)} clusters):", ""]

    for rank, group in enumerate(insight_groups, 1):
        label = group.get("label", "unknown")
        indices = group.get("insight_indices", [])
        rep_idx = group.get("representative_index")

        lines.append(
            f"{rank}. {label} ({len(indices)} insights: "
            f"indices {indices})"
        )

        if rep_idx is not None and isinstance(rep_idx, int) and 0 <= rep_idx < len(insights):
            rep = insights[rep_idx]
            lines.append(f"   Representative (#{rep_idx}):")

            task_type = rep.get("task_type")
            if task_type and task_type != "unknown":
                lines.append(f"   Task type: {task_type}")

            task_query = rep.get("task_query")
            if task_query:
                lines.append(f"   Task query: {task_query}")

            lines.append(f"   Insight: {rep.get('insight', '')}")

            f_mode = rep.get("failure_mode")
            if f_mode:
                lines.append(f"   Failure mode: {f_mode}")

            f_snippet = rep.get("failure_snippet")
            if f_snippet and isinstance(f_snippet, list):
                lines.append(f"   Failure actions: {' -> '.join(str(a) for a in f_snippet[:8])}")

            f_point = rep.get("failure_point")
            if f_point:
                lines.append(f"   Failure point: {f_point}")

            s_snippet = rep.get("success_snippet")
            if s_snippet and isinstance(s_snippet, list):
                lines.append(f"   Success actions: {' -> '.join(str(a) for a in s_snippet[:8])}")

            s_pattern = rep.get("success_pattern")
            if s_pattern:
                lines.append(f"   Success pattern: {s_pattern}")

            ab_delta = rep.get("ab_delta", "")
            if ab_delta and ab_delta.lower() != "n/a":
                lines.append(f"   AB delta: {ab_delta}")

            confidence = rep.get("confidence")
            if confidence:
                lines.append(f"   Confidence: {confidence}")
        lines.append("")

    return "\n".join(lines)


class SkillRecommender:
    """Stage 3: Strategic skill recommendations."""

    def __init__(
        self,
        client: LLMClient,
        prompt_path: Optional[str] = None,
        n_version_history: int = 5,
    ):
        self.client = client
        self.n_version_history = n_version_history
        self._system_prompt = _load_prompt(
            prompt_path or os.path.join(_PROMPTS_V2_DIR, "skill_recommender.md")
        )

    def recommend(
        self,
        diagnosis: dict,
        skill_library_summary: str,
        version_history: List[dict],
        insights: List[dict],
        save_dir: Optional[str] = None,
        step: int = 0,
        max_retries: int = 3,
        action_vocabulary: str = "",
    ) -> Optional[dict]:
        """Produce skill operation recommendations.

        Args:
            diagnosis: Output from BatchDiagnoser (contains
                insight_groups, diagnosis text, priority_modes).
            skill_library_summary: From SkillRegistry.to_diagnostic_summary().
            version_history: Full version history list.
            insights: All insights from EpisodeAnalyzer.
            save_dir: Optional directory to save prompts/responses.
            step: Training step number.
            max_retries: Number of attempts on JSON parse failure.
            action_vocabulary: Environment-specific action template description.

        Returns:
            Dict with reasoning + operations list, or None on failure.
        """
        prompt = self._build_prompt(
            diagnosis, skill_library_summary, version_history, insights,
            action_vocabulary=action_vocabulary,
        )

        for attempt in range(max_retries):
            response = self.client.generate(
                user_prompt=prompt,
                system_prompt=self._system_prompt,
                max_tokens=4096,
                temperature=0.3,
            )

            if save_dir:
                self._save(save_dir, step, prompt, response)

            if response is None:
                logger.warning(
                    f"[SkillRecommender] No response "
                    f"(attempt {attempt + 1}/{max_retries})")
                continue

            parsed = extract_json(response)
            if parsed is None:
                logger.warning(
                    f"[SkillRecommender] JSON parse failed "
                    f"(attempt {attempt + 1}/{max_retries}). "
                    f"Response: {response[:200]}")
                continue

            ops = parsed.get("operations", [])
            valid_ops = [op for op in ops if op.get("op") in ("add", "modify", "delete")]
            if len(valid_ops) != len(ops):
                logger.warning(
                    f"[SkillRecommender] Filtered {len(ops) - len(valid_ops)} invalid ops")
            parsed["operations"] = valid_ops

            if not valid_ops:
                logger.warning(
                    f"[SkillRecommender] No valid operations "
                    f"(attempt {attempt + 1}/{max_retries})")
                continue

            logger.info(
                f"[SkillRecommender] {len(valid_ops)} recommendations: "
                + ", ".join(f"{op['op'].upper()}:{op.get('target', 'new')}" for op in valid_ops)
            )
            return parsed

        logger.warning(
            f"[SkillRecommender] Failed after {max_retries} attempts")
        return None

    def _build_prompt(
        self,
        diagnosis: dict,
        skill_library_summary: str,
        version_history: List[dict],
        insights: List[dict],
        action_vocabulary: str = "",
    ) -> str:
        sections = []

        # Action vocabulary (environment-specific)
        if action_vocabulary:
            sections.append("## Environment Action Vocabulary\n")
            sections.append(action_vocabulary)
            sections.append("")

        # Insight groups (expanded representative per group)
        insight_groups = diagnosis.get("insight_groups", [])
        sections.append(_format_insight_groups(insight_groups, insights))
        sections.append("")

        # Assertion diagnosis
        sections.append("## Assertion Diagnosis\n")
        diag_text = diagnosis.get("diagnosis", "")
        priority = diagnosis.get("priority_modes", [])
        if diag_text:
            sections.append(diag_text)
        else:
            sections.append("No assertion diagnosis available.")
        if priority:
            sections.append(f"Priority modes: {', '.join(str(m) for m in priority)}")
        sections.append("")

        # Skill library
        sections.append("## Current Skill Library\n")
        sections.append(skill_library_summary or "(empty -- no skills registered)")
        sections.append("")

        # Version history (compressed)
        sections.append("## " + _compress_version_history(
            version_history, self.n_version_history))
        sections.append("")

        # Constraints
        sections.append(
            "## Constraints\n"
            "- Max 8 active skills total\n"
            "- Valid trigger types: general, beginning, action_pattern\n"
            "- Max 1-3 operations per version\n"
            "- Do NOT write skill content -- only recommend operations\n"
        )

        sections.append(
            "## Your Task\n\n"
            "Based on the failure pattern groups, diagnosis, skill library, and history, "
            "recommend 1-3 skill operations. Return JSON as specified in your instructions."
        )

        return "\n".join(sections)

    def _save(self, save_dir: str, step: int, prompt: str, response: Optional[str]):
        try:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"_step{step}"
            with open(os.path.join(save_dir, f"recommender_prompt{suffix}.md"), "w") as f:
                f.write(prompt)
            if response:
                with open(os.path.join(save_dir, f"recommender_response{suffix}.md"), "w") as f:
                    f.write(response)
        except Exception as e:
            logger.warning(f"[SkillRecommender] Failed to save: {e}")
