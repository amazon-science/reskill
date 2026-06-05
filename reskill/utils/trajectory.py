"""
Shared trajectory formatting utilities for analyzer, prompt optimizer, and reward optimizer.

All three LLM-based components receive the same trajectory data and should format it
consistently. This module provides the shared formatting logic.

Format:
  - Task: extracted description (not system message preamble), up to 500 chars
  - All steps: parsed action only (extracted from <action> tags)
  - Last 3 steps: full model reasoning + action, up to 600 chars
  - Environment observations: omitted to save token budget
    (available in step['observation'] for future re-addition)
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_task_description(obs_text: str, max_chars: int = 500) -> str:
    """Extract just the task description from the full observation prompt.

    Handles common task markers such as "Your task is to:" and "Your question:".
    """
    # CodeSearch: extract content between <issue_description> tags
    issue_start_tag = "<issue_description>"
    issue_end_tag = "</issue_description>"
    issue_idx = obs_text.find(issue_start_tag)
    if issue_idx != -1:
        content_start = issue_idx + len(issue_start_tag)
        end_idx = obs_text.find(issue_end_tag, content_start)
        if end_idx != -1:
            return obs_text[content_start:end_idx].strip()[:max_chars]
        return obs_text[content_start:].strip()[:max_chars]

    for marker in ["Your task is to: ", "Your question: "]:
        idx = obs_text.find(marker)
        if idx != -1:
            task_start = idx + len(marker)
            for end_marker in [
                ".\nYour current observation",
                ".\n\n## Retrieved",
                ".\nPrior to",
                "\n\n## Available Skills",
                "\n\n## Current Progress",
                "\n\nNow it's your turn",
            ]:
                end_idx = obs_text.find(end_marker, task_start)
                if end_idx != -1:
                    return obs_text[task_start:end_idx][:max_chars]
            end_idx = obs_text.find("\n\n", task_start)
            if end_idx != -1:
                return obs_text[task_start:end_idx][:max_chars]
            return obs_text[task_start:][:max_chars]
    return obs_text[:max_chars]


def extract_action_tag(raw_response: str) -> str:
    """Extract the action from response tags.

    Handles action, search, and answer tags.
    """
    for pattern in [
        r'<action>(.*?)</action>',
        r'(<search>.*?</search>)',
        r'(<answer>.*?</answer>)',
    ]:
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw_response[-100:].strip() if raw_response else "(empty)"


def _compute_step_skill_annotations(
    all_steps: list,
    skill_registry=None,
) -> list:
    """Compute which skills would trigger at each step.

    Uses the current skill registry to retroactively determine which active
    skills would have been shown to the agent at each step, based on trigger
    conditions (step number and previous action).

    Args:
        all_steps: List of step dicts with 'action' key.
        skill_registry: Optional SkillRegistry instance. If None, returns
            empty annotations.

    Returns:
        List of lists of skill names, one per step.
    """
    if skill_registry is None:
        return [[] for _ in all_steps]

    from reskill.skill_serving.trigger_matcher import TriggerMatcher

    active_skills = skill_registry.get_active_skills()
    annotations = []
    for j, step in enumerate(all_steps):
        step_num = j
        # Last action is from the previous step (None at step 0)
        last_action = extract_action_tag(all_steps[j - 1].get('action', '')) if j > 0 else None
        triggered_names = []
        for skill in active_skills:
            if TriggerMatcher.should_trigger(skill, step_num, last_action):
                triggered_names.append(skill.name)
        annotations.append(triggered_names)
    return annotations


def format_trajectory_for_llm(
    traj: Dict,
    max_episodes: int = 10,
    recent_steps_full: int = 3,
    full_step_chars: int = 600,
    task_chars: int = 500,
    skill_registry=None,
) -> str:
    """Format a single failed trajectory for LLM consumption.

    Args:
        traj: Dict with keys: task, trajectory (list of {action, observation}), task_type.
              Optional: task_description (pre-extracted clean task string).
        max_episodes: Unused (for caller to limit before calling this).
        recent_steps_full: Number of recent steps to show full reasoning.
        full_step_chars: Max chars for full reasoning steps.
        task_chars: Max chars for task description.
        skill_registry: Optional SkillRegistry for skill activation annotations.

    Returns:
        Formatted string for one episode.
    """
    task_desc = traj.get('task_description')
    if not task_desc:
        logger.warning("[trajectory_utils] 'task_description' missing from trajectory, "
                       "falling back to parsing full prompt — this may produce garbled output")
        task_desc = extract_task_description(traj.get('task', ''), max_chars=task_chars)
    all_steps = traj.get('trajectory', [])

    # Compute skill activation annotations if registry available
    skill_annotations = _compute_step_skill_annotations(all_steps, skill_registry)

    # A/B condition label (only meaningful during version testing)
    ab_label = ""
    ab_cond = traj.get('ab_condition')
    if ab_cond == 'new':
        ab_label = "  [with proposed skill changes]"
    elif ab_cond == 'old':
        ab_label = "  [with existing skills]"

    lines = [f"**Task:** {task_desc}{ab_label}"]

    step_lines = []
    for j, step in enumerate(all_steps):
        raw_action = step.get('action', '')
        is_recent = j >= len(all_steps) - recent_steps_full

        # Build skill annotation suffix
        skill_suffix = ""
        if skill_annotations[j]:
            skill_suffix = f" | active_skills=[{', '.join(skill_annotations[j])}]"

        if is_recent:
            step_lines.append(f"  Step {j+1} (full): {raw_action[:full_step_chars]}{skill_suffix}")
        else:
            parsed = extract_action_tag(raw_action)
            step_lines.append(f"  Step {j+1}: {parsed}{skill_suffix}")

    if step_lines:
        lines.append(
            f"**Action Trace ({len(all_steps)} steps, "
            f"last {recent_steps_full} with reasoning):**"
        )
        lines.extend(step_lines)

    return "\n".join(lines)


def format_condensed_trajectory(
    traj: Dict, task_chars: int = 300, skill_registry=None,
) -> str:
    """Format a single trajectory as one condensed block: task + action-only sequence.

    No reasoning, no observations — just parsed actions. Used by the analyzer
    to survey many episodes quickly.
    """
    task_desc = traj.get('task_description')
    if not task_desc:
        logger.warning("[trajectory_utils] 'task_description' missing from trajectory, "
                       "falling back to parsing full prompt — this may produce garbled output")
        task_desc = extract_task_description(traj.get('task', ''), max_chars=task_chars)
    all_steps = traj.get('trajectory', [])

    # Compute skill activation annotations if registry available
    skill_annotations = _compute_step_skill_annotations(all_steps, skill_registry)

    actions = []
    for j, step in enumerate(all_steps):
        parsed = extract_action_tag(step.get('action', ''))
        skill_suffix = ""
        if skill_annotations[j]:
            skill_suffix = f" | skills=[{', '.join(skill_annotations[j])}]"
        actions.append(f"  {j+1}. {parsed}{skill_suffix}")
    score = traj.get('task_score', 0.0)
    ab_label = ""
    ab_cond = traj.get('ab_condition')
    if ab_cond == 'new':
        ab_label = "  [with proposed skill changes]"
    elif ab_cond == 'old':
        ab_label = "  [with existing skills]"
    header = f"**Task:** {task_desc}  [score={score:.2f}]{ab_label}"
    return f"{header}\n" + "\n".join(actions) if actions else header


def format_condensed_episodes(
    success_trajectories: List[Dict],
    failed_trajectories: List[Dict],
    max_successes: int = 10,
    max_failures: int = 20,
    task_chars: int = 300,
    skill_registry=None,
) -> str:
    """Format many episodes in condensed action-only format for the analyzer survey."""
    blocks = []

    if success_trajectories:
        blocks.append("## Successful Episodes\n")
        for i, traj in enumerate(success_trajectories[:max_successes]):
            blocks.append(f"--- Success {i+1} ---")
            blocks.append(format_condensed_trajectory(
                traj, task_chars=task_chars, skill_registry=skill_registry))
            blocks.append("")

    if failed_trajectories:
        blocks.append("## Failed Episodes\n")
        for i, traj in enumerate(failed_trajectories[:max_failures]):
            blocks.append(f"--- Failure {i+1} ---")
            blocks.append(format_condensed_trajectory(
                traj, task_chars=task_chars, skill_registry=skill_registry))
            blocks.append("")

    return "\n".join(blocks) if blocks else "No trajectory data available."


def format_failed_trajectories(
    failed_trajectories: List[Dict],
    max_episodes: int = 10,
    recent_steps_full: int = 3,
    full_step_chars: int = 600,
    task_chars: int = 500,
) -> str:
    """Format multiple failed trajectories for LLM consumption.

    NOTE: Environment observations are omitted to save token budget.
    The full observation is available in step['observation'] if needed in
    the future — it would require extracting the env delta from the
    accumulated prompt context.

    Args:
        failed_trajectories: List of trajectory dicts.
        max_episodes: Max episodes to include.
        recent_steps_full: Number of recent steps to show full reasoning.
        full_step_chars: Max chars for full reasoning steps.
        task_chars: Max chars for task description.

    Returns:
        Formatted string for all episodes.
    """
    if not failed_trajectories:
        return "No failed trajectories available."

    blocks = []
    for i, traj in enumerate(failed_trajectories[:max_episodes]):
        header = f"--- Failure {i+1} [{traj.get('task_type', 'unknown')}] ---"
        body = format_trajectory_for_llm(
            traj,
            recent_steps_full=recent_steps_full,
            full_step_chars=full_step_chars,
            task_chars=task_chars,
        )
        blocks.append(f"{header}\n{body}")

    result = "\n\n".join(blocks)

    if len(failed_trajectories) > max_episodes:
        result += f"\n\n... ({len(failed_trajectories) - max_episodes} more failed episodes not shown)"

    return result


def format_trajectories(
    success_trajectories: List[Dict],
    failed_trajectories: List[Dict],
    max_successes: int = 3,
    max_failures: int = 7,
    recent_steps_full: int = 3,
    full_step_chars: int = 600,
    task_chars: int = 500,
    skill_registry=None,
) -> str:
    """Format success + failure trajectories for LLM components.

    Includes task_score in headers so the LLM knows proximity to success.
    Successes shown first for contrast, then failures sorted by task_score descending.

    Args:
        success_trajectories: Successful episodes (task_score == 1.0).
        failed_trajectories: Failed episodes with task_score for proximity.
        max_successes: Max success episodes to show.
        max_failures: Max failure episodes to show.
        recent_steps_full: Recent steps to show full reasoning.
        full_step_chars: Max chars for full reasoning steps.
        task_chars: Max chars for task description.
        skill_registry: Optional SkillRegistry for skill activation annotations.

    Returns:
        Formatted string with success and failure sections.
    """
    blocks = []

    # Success section
    if success_trajectories:
        blocks.append("## Successful Episodes\n")
        for i, traj in enumerate(success_trajectories[:max_successes]):
            score = traj.get('task_score', 1.0)
            ab_tag = ""
            ab_cond = traj.get('ab_condition')
            if ab_cond == 'new':
                ab_tag = " [with proposed skill changes]"
            elif ab_cond == 'old':
                ab_tag = " [with existing skills]"
            header = f"--- Success {i+1} [{traj.get('task_type', 'unknown')}] [task_score={score:.2f}]{ab_tag} ---"
            body = format_trajectory_for_llm(
                traj, recent_steps_full=recent_steps_full,
                full_step_chars=full_step_chars, task_chars=task_chars,
                skill_registry=skill_registry,
            )
            blocks.append(f"{header}\n{body}")

        if len(success_trajectories) > max_successes:
            blocks.append(f"... ({len(success_trajectories) - max_successes} more successes not shown)")
        blocks.append("")

    # Failure section
    if failed_trajectories:
        blocks.append("## Failed Episodes (sorted by task_score, highest first)\n")
        for i, traj in enumerate(failed_trajectories[:max_failures]):
            score = traj.get('task_score', 0.0)
            ab_tag = ""
            ab_cond = traj.get('ab_condition')
            if ab_cond == 'new':
                ab_tag = " [with proposed skill changes]"
            elif ab_cond == 'old':
                ab_tag = " [with existing skills]"
            header = f"--- Failure {i+1} [{traj.get('task_type', 'unknown')}] [task_score={score:.2f}]{ab_tag} ---"
            body = format_trajectory_for_llm(
                traj, recent_steps_full=recent_steps_full,
                full_step_chars=full_step_chars, task_chars=task_chars,
                skill_registry=skill_registry,
            )
            blocks.append(f"{header}\n{body}")

        if len(failed_trajectories) > max_failures:
            blocks.append(f"... ({len(failed_trajectories) - max_failures} more failures not shown)")

    if not blocks:
        return "No trajectory data available."

    return "\n\n".join(blocks)


# ------------------------------------------------------------------
# Contrastive group formatting
# ------------------------------------------------------------------

def _detect_loops(actions: List[str], window: int = 4) -> Optional[str]:
    """Detect repeating action subsequences.

    Returns a description like 'loop: "go to X, open X" repeated 5 times'
    or None if no significant loop found.
    """
    if len(actions) < window * 2:
        return None
    for size in range(2, window + 1):
        for start in range(len(actions) - size * 2 + 1):
            pattern = actions[start:start + size]
            repeats = 1
            pos = start + size
            while pos + size <= len(actions) and actions[pos:pos + size] == pattern:
                repeats += 1
                pos += size
            if repeats >= 3:
                pat_str = ", ".join(pattern[:3])
                if len(pattern) > 3:
                    pat_str += ", ..."
                return f'loop: "{pat_str}" repeated {repeats} times'
    return None


def format_contrastive_group(
    task_group,
    max_actions: int = 30,
    last_n_reasoning: int = 2,
    reasoning_chars: int = 400,
    skill_registry=None,
) -> str:
    """Format a task_key group for contrastive analysis.

    Each episode is formatted as:
    - Header with AB condition, outcome, score, step count
    - All steps: parsed action only (compact)
    - Long traces (>max_actions): first 5 + loop info + last 5
    - Last last_n_reasoning steps: full reasoning text
    - Skill activation annotations

    Args:
        task_group: TaskGroup from experience_reservoir.sample_task_groups().
        max_actions: Compress traces longer than this.
        last_n_reasoning: Show full reasoning for these many final steps.
        reasoning_chars: Max chars per reasoning step.
        skill_registry: Optional SkillRegistry for skill annotations.

    Returns:
        Formatted string for one contrastive group (~800-1200 tokens).
    """
    episodes = task_group.episodes
    if not episodes:
        return ""

    task_desc = task_group.task_key
    task_type = episodes[0].get('task_type', 'unknown')

    lines = [f"TASK: {task_desc}", f"TYPE: {task_type}", ""]

    ep_labels = "ABCDEF"
    for ep_idx, ep in enumerate(episodes):
        label = ep_labels[ep_idx] if ep_idx < len(ep_labels) else str(ep_idx + 1)
        ab_cond = ep.get('ab_condition', 'old')
        success_val = ep.get('success', 0.0)
        outcome = "SUCCESS" if success_val >= 1.0 else "FAIL"
        score = ep.get('score', 0.0)
        all_steps = ep.get('trajectory', [])
        n_steps = len(all_steps)

        lines.append(
            f"EPISODE {label} [{ab_cond}, {outcome}, score={score:.1f}, "
            f"{n_steps} steps]:"
        )

        # Compute skill annotations
        skill_annotations = _compute_step_skill_annotations(all_steps, skill_registry)

        # Parse all actions
        parsed_actions = [extract_action_tag(s.get('action', '')) for s in all_steps]

        def _format_step(j):
            """Format a single step, with optional reasoning context."""
            skill_suffix = ""
            if skill_annotations[j]:
                skill_suffix = f"  [skills: {', '.join(skill_annotations[j])}]"

            show_reasoning = j >= n_steps - last_n_reasoning
            if show_reasoning:
                raw = all_steps[j].get('action', '')
                action_str = parsed_actions[j]
                # Extract reasoning only (text before the action tag)
                for tag in ['<action>', '<search>', '<answer>']:
                    idx = raw.find(tag)
                    if idx > 0:
                        reasoning = raw[:idx].strip()[:reasoning_chars]
                        break
                else:
                    reasoning = raw[:reasoning_chars]
                return (
                    f"  {j+1}. ACTION: {action_str}{skill_suffix}\n"
                    f"       REASONING: {reasoning}"
                )
            else:
                return f"  {j+1}. {parsed_actions[j]}{skill_suffix}"

        if n_steps <= max_actions:
            for j in range(n_steps):
                lines.append(_format_step(j))
        else:
            for j in range(min(5, n_steps)):
                lines.append(_format_step(j))

            loop_info = _detect_loops(parsed_actions)
            if loop_info:
                lines.append(f"  ... ({n_steps - 10} steps, {loop_info}) ...")
            else:
                lines.append(f"  ... ({n_steps - 10} more steps) ...")

            start_last = max(5, n_steps - 5)
            for j in range(start_last, n_steps):
                lines.append(_format_step(j))

        # Skill activation summary for this episode
        all_triggered = set()
        for ann in skill_annotations:
            all_triggered.update(ann)
        if all_triggered:
            lines.append(f"  Skills active: [{', '.join(sorted(all_triggered))}]")

        lines.append("")

    return "\n".join(lines)
