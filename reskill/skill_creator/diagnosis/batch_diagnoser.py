"""Stage 2: Cross-episode diagnosis and insight grouping.

Receives insight summaries (one-liners from EpisodeAnalyzer.summarize()) plus
assertion pass rates (from AssertionEngine). Produces:
1. Assertion CRUD ops (compatible with AssertionEngine.apply_crud())
2. Systemic diagnosis
3. Semantic grouping of the insight summaries → insight_groups
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
        logger.warning(f"[BatchDiagnoser] Prompt not found: {path}")
        return ""


class BatchDiagnoser:
    """Stage 2: Cross-episode LLM diagnosis, assertion updates, and insight grouping."""

    def __init__(
        self,
        client: LLMClient,
        prompt_path: Optional[str] = None,
    ):
        self.client = client
        self._system_prompt = _load_prompt(
            prompt_path or os.path.join(_PROMPTS_V2_DIR, "batch_diagnoser.md")
        )

    def diagnose(
        self,
        assertions_summary: str,
        insight_summaries_text: str,
        current_assertions: List[dict],
        current_skills_summary: str = "",
        save_dir: Optional[str] = None,
        step: int = 0,
        max_retries: int = 3,
    ) -> Optional[dict]:
        """Run cross-episode diagnosis and insight grouping.

        Args:
            assertions_summary: Pass rates from AssertionEngine.get_assertions_summary().
            insight_summaries_text: Formatted diagnoser_summary from EpisodeAnalyzer.summarize().
            current_assertions: List of assertion dicts for context.
            current_skills_summary: Active skills summary from SkillRegistry.
            save_dir: Optional directory to save prompts/responses.
            step: Training step number.
            max_retries: Number of attempts on JSON parse failure.

        Returns:
            Dict with assertion_operations, diagnosis, priority_modes,
            insight_groups. Or None on failure.
        """
        prompt = self._build_prompt(
            assertions_summary, insight_summaries_text, current_assertions,
            current_skills_summary)

        for attempt in range(max_retries):
            response = self.client.generate(
                user_prompt=prompt,
                system_prompt=self._system_prompt,
                max_tokens=4096,
                temperature=0.2,
            )

            if save_dir:
                self._save(save_dir, step, prompt, response)

            if response is None:
                logger.warning(
                    f"[BatchDiagnoser] No response "
                    f"(attempt {attempt + 1}/{max_retries})")
                continue

            parsed = extract_json(response)
            if parsed is None:
                logger.warning(
                    f"[BatchDiagnoser] JSON parse failed "
                    f"(attempt {attempt + 1}/{max_retries}). "
                    f"Response: {response[:200]}")
                continue

            if "assertion_operations" not in parsed:
                parsed["assertion_operations"] = []
            if "diagnosis" not in parsed:
                parsed["diagnosis"] = ""
            if "priority_modes" not in parsed:
                parsed["priority_modes"] = []
            if "insight_groups" not in parsed:
                parsed["insight_groups"] = []

            n_groups = len(parsed["insight_groups"])
            n_indices = sum(
                len(g.get("insight_indices", []))
                for g in parsed["insight_groups"]
            )
            logger.info(
                f"[BatchDiagnoser] {len(parsed['assertion_operations'])} assertion ops, "
                f"{n_groups} insight groups covering {n_indices} indices, "
                f"priority: {parsed['priority_modes'][:3]}")
            return parsed

        logger.warning(
            f"[BatchDiagnoser] Failed after {max_retries} attempts")
        return None

    def _build_prompt(
        self,
        assertions_summary: str,
        insight_summaries_text: str,
        current_assertions: List[dict],
        current_skills_summary: str = "",
    ) -> str:
        sections = []

        sections.append("## Assertion Pass Rates (full reservoir)\n")
        if assertions_summary:
            sections.append(assertions_summary)
        else:
            sections.append("No assertions configured yet.")
        sections.append("")

        sections.append("## Insight Summaries\n")
        sections.append(insight_summaries_text)
        sections.append("")

        sections.append(f"## Current Assertions ({len(current_assertions)} active)\n")
        if current_assertions:
            for a in current_assertions:
                name = a.get("name", a.get("assertion_id", "?"))
                check_type = a.get("check_type", "?")
                desc = a.get("description", "")
                sections.append(f"- {name} ({check_type}): {desc}")
        else:
            sections.append("No assertions configured.")

        sections.append("")

        sections.append("## Current Active Skills\n")
        if current_skills_summary:
            sections.append(current_skills_summary)
        else:
            sections.append("(none)")
        sections.append("")

        sections.append(
            "## Your Task\n\n"
            "1. Analyze assertion pass rates and insight summaries. "
            "Propose assertion CRUD operations and provide a quantitative diagnosis.\n"
            "2. Group the insight summaries into 3-8 semantic clusters. "
            "Pick one representative per group.\n"
            "3. Return JSON as specified in your instructions."
        )

        return "\n".join(sections)

    def _save(self, save_dir: str, step: int, prompt: str, response: Optional[str]):
        try:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"_step{step}"
            with open(os.path.join(save_dir, f"diagnoser_prompt{suffix}.md"), "w") as f:
                f.write(prompt)
            if response:
                with open(os.path.join(save_dir, f"diagnoser_response{suffix}.md"), "w") as f:
                    f.write(response)
        except Exception as e:
            logger.warning(f"[BatchDiagnoser] Failed to save: {e}")
