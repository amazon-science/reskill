"""Search prompt templates used by ReSkill experiments.

Supported policy-prompt modes:
- GRPO reason baseline: one-sentence reasoning, then search or answer.
- ReSkill reason mode: same action format with an injected skills section.
"""


SEARCH_GRPO_REASON_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
First, briefly explain your reasoning in one sentence. Then, choose only one of the following actions:
(1) If you lack needed knowledge, call a search engine using format: <search> your query </search>.
(2) If you have enough knowledge to answer confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""


SEARCH_GRPO_REASON = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
First, briefly explain your reasoning in one sentence. Then, choose only one of the following actions:
(1) If you lack needed knowledge, call a search engine using format: <search> your query </search>.
(2) If you have enough knowledge to answer confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""


_SKILL_INSTRUCTION_ACTIVE = (
    "Check the Available Skills section by reading each `when_to_use` field. "
    "If a skill matches your situation, state which skill you are applying "
    "and follow its `action` guidance; otherwise, proceed on your own."
)


SEARCH_RESKILL_REASON_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}
{triggered_skills_section}
Now it's your turn to respond for the current step.
""" + _SKILL_INSTRUCTION_ACTIVE + """
First, briefly explain your reasoning in one sentence (including which skill you applied, if any). Then, choose only one of the following actions:
(1) If you lack needed knowledge, call a search engine using format: <search> your query </search>.
(2) If you have enough knowledge to answer confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""


SEARCH_RESKILL_REASON = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}
{triggered_skills_section}
Now it's your turn to respond for the current step.
""" + _SKILL_INSTRUCTION_ACTIVE + """
First, briefly explain your reasoning in one sentence (including which skill you applied, if any). Then, choose only one of the following actions:
(1) If you lack needed knowledge, call a search engine using format: <search> your query </search>.
(2) If you have enough knowledge to answer confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""


def format_skills_section(skill_text: str) -> str:
    """Wrap skill text in the Available Skills section. Always present."""
    if not skill_text:
        return "\n## Available Skills\n\n(no skills available)\n"
    return f"\n## Available Skills\n\n{skill_text}\n"
