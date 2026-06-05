from typing import List
import re


def scienceworld_projection(actions: List[str], action_pools: List[List[str]], require_think=True):
    """Process raw LLM outputs into environment actions for ScienceWorld.

    Extracts action text from <action>...</action> tags and validates format.
    ScienceWorld handles invalid action strings gracefully (returns error obs),
    so we only validate tag structure, not semantic content.

    Args:
        actions: Raw text outputs from the LLM.
        action_pools: Admissible action lists per environment (unused for
            semantic validation but kept for interface compatibility).
        require_think: Whether <think> tags are required.

    Returns:
        (actions, valids): Cleaned action strings and validity flags (0/1).
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        actions[i] = actions[i].lower()

        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                actions[i] = actions[i][-30:]
                continue

            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip()
            actions[i] = extracted_action
            valids[i] = 1

        except Exception:
            actions[i] = actions[i][-30:]

        # Check <think>...</think> if required
        if require_think:
            think_start_idx = original_str.find("<think>")
            think_end_idx = original_str.find("</think>")
            if think_start_idx == -1 or think_end_idx == -1:
                valids[i] = 0

        # Reject outputs with Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids
