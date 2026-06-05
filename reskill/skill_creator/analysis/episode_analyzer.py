"""Stage 1: Per-episode analysis + batch summarization.

Two responsibilities handled here:

1. EpisodeAnalyzer — runs one small LLM call
   per task group, comparing success vs failure and new_skills vs old_skills to
   produce a structured insight dict. ~30 groups are processed in parallel.

2. EpisodeAnalyzer.summarize() — deterministic bridge to Stage 2. Indexes the
   full insights and produces lightweight one-liner summaries for BatchDiagnoser
   while preserving the full insight list for the downstream recommender/author.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from reskill.utils.llm_client import LLMClient, extract_json
from reskill.skill_creator.analysis.experience_reservoir import TaskGroup

logger = logging.getLogger(__name__)

_PROMPTS_V2_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts"
)

_EMPTY_INSIGHT = {
    "task_type": "unknown",
    "task_query": "",
    "insight": "",
    "failure_mode": "unknown",
    "failure_snippet": None,
    "failure_point": None,
    "success_snippet": None,
    "success_pattern": None,
    "trigger_relevant_actions": [],
    "skill_impact": "N/A",
    "ab_delta": "N/A",
    "confidence": "low",
}


def _load_prompt(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"[EpisodeAnalyzer] Prompt not found: {path}")
        return ""


def _heuristic_fallback(group: TaskGroup) -> dict:
    """Extract a minimal insight when LLM response is unparseable."""
    insight = dict(_EMPTY_INSIGHT)
    insight["task_type"] = group.episodes[0].get("task_type", "unknown") if group.episodes else "unknown"
    insight["task_query"] = group.task_key or ""

    failures = [e for e in group.episodes if e.get("success", 0.0) < 1.0]
    successes = [e for e in group.episodes if e.get("success", 0.0) >= 1.0]

    if failures:
        traj = failures[0].get("trajectory", [])
        n = len(traj)
        insight["failure_mode"] = "timeout" if n >= 45 else "unknown"
        insight["insight"] = f"Episode failed after {n} steps"
        insight["failure_point"] = f"Step {n}: episode ended"
        insight["confidence"] = "low"

    if successes:
        traj = successes[0].get("trajectory", [])
        insight["success_pattern"] = f"Succeeded in {len(traj)} steps"

    return insight


class EpisodeAnalyzer:
    """Stage 1: Per-episode LLM analysis and batch summarization."""

    def __init__(
        self,
        client: LLMClient,
        prompt_path: Optional[str] = None,
    ):
        self.client = client
        self._system_prompt = _load_prompt(
            prompt_path or os.path.join(_PROMPTS_V2_DIR, "episode_analyzer.md")
        )

    def analyze_groups(
        self,
        task_groups: List[TaskGroup],
        skill_registry=None,
        max_workers: int = 16,
        save_dir: Optional[str] = None,
        step: int = 0,
    ) -> List[dict]:
        """Run per-episode analysis on all groups in parallel.

        Args:
            task_groups: List of TaskGroup from sample_task_groups().
            skill_registry: For skill activation annotations.
            max_workers: ThreadPoolExecutor concurrency.
            save_dir: Optional directory to save prompts/responses.
            step: Training step number.

        Returns:
            List of insight dicts (one per group).
        """
        from reskill.utils.trajectory import format_contrastive_group

        # Build prompts
        user_prompts = []
        for group in task_groups:
            formatted = format_contrastive_group(
                group, skill_registry=skill_registry)
            user_prompts.append(formatted)

        # Parallel LLM calls
        batch_prompts = [
            (self._system_prompt, up) for up in user_prompts
        ]
        responses = self.client.generate_batch(
            prompts=batch_prompts,
            max_tokens=4096,
            temperature=0.2,
            max_workers=max_workers,
        )

        # Parse responses
        insights = []
        for i, resp in enumerate(responses):
            if resp is None:
                logger.warning(f"[EpisodeAnalyzer] Group {i} got no response")
                insights.append(_heuristic_fallback(task_groups[i]))
                continue

            parsed = extract_json(resp)
            if parsed is None:
                logger.warning(
                    f"[EpisodeAnalyzer] Group {i} JSON parse failed, "
                    f"using heuristic fallback. Response: {resp[:200]}")
                insights.append(_heuristic_fallback(task_groups[i]))
            else:
                # Fill missing fields with defaults
                full = dict(_EMPTY_INSIGHT)
                full.update(parsed)
                full["task_query"] = task_groups[i].task_key or ""
                insights.append(full)

        # Save artifacts
        if save_dir:
            self._save(save_dir, step, user_prompts, responses, insights)

        n_high = sum(1 for ins in insights if ins.get("confidence") == "high")
        logger.info(
            f"[EpisodeAnalyzer] {len(insights)} insights "
            f"({n_high} high confidence)")
        return insights

    @staticmethod
    def summarize(insights: List[dict]) -> dict:
        """Index insights and produce lightweight summaries for BatchDiagnoser.

        The full insights (with all 10 fields from per-episode analysis) are
        preserved in the output for downstream stages that need comprehensive
        detail (recommender, author).

        Args:
            insights: List of insight dicts from analyze_groups().
                Each has: insight, failure_mode, failure_snippet,
                failure_point, success_snippet, success_pattern,
                trigger_relevant_actions, skill_impact, ab_delta,
                confidence, task_type.

        Returns:
            Dict with:
            - insights: Full comprehensive insight list (passthrough)
            - diagnoser_summary: Formatted one-liner list for BatchDiagnoser
            - total_insights: Number of insights processed
        """
        if not insights:
            return {
                "insights": [],
                "diagnoser_summary": "No insights available.",
                "total_insights": 0,
            }

        lines = [f"INSIGHT SUMMARIES ({len(insights)} episode analyses):", ""]
        for i, ins in enumerate(insights):
            line = (ins.get("insight") or "").strip()
            if not line:
                line = ins.get("failure_mode") or "unknown"
            lines.append(f"{i}. {line}")

        diagnoser_summary = "\n".join(lines)

        logger.info(f"[EpisodeAnalyzer] {len(insights)} insights indexed for batch diagnosis")

        return {
            "insights": insights,
            "diagnoser_summary": diagnoser_summary,
            "total_insights": len(insights),
        }

    def _save(
        self,
        save_dir: str,
        step: int,
        prompts: List[str],
        responses: List[Optional[str]],
        insights: List[dict],
    ):
        try:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"_step{step}"

            # Save all insights as single JSON
            with open(os.path.join(save_dir, f"episode_insights{suffix}.json"), "w") as f:
                json.dump(insights, f, indent=2)

            # Save prompts + responses for debugging
            debug_data = []
            for i in range(len(prompts)):
                debug_data.append({
                    "group_index": i,
                    "prompt": prompts[i],
                    "response": responses[i],
                })
            with open(os.path.join(save_dir, f"episode_debug{suffix}.json"), "w") as f:
                json.dump(debug_data, f, indent=2)

        except Exception as e:
            logger.warning(f"[EpisodeAnalyzer] Failed to save artifacts: {e}")
