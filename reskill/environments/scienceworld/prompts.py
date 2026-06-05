"""ScienceWorld prompt templates used by ReSkill experiments.

Supported policy-prompt modes:
- GRPO reason baseline: one-sentence reasoning, then <action>.
- ReSkill reason mode: same action format with an injected skills section.
"""


SCIENCEWORLD_GRPO_REASON_NO_HIS = """
You are an expert agent operating in the ScienceWorld Environment, a simulated science laboratory where you perform experiments and solve science tasks.
Your task is to: {task_description}
Your current observation is: {current_observation}

Current available actions:
{admissible_actions}

Now it's your turn to take an action.
First, briefly explain your reasoning in one sentence. Then, provide your chosen action enclosed in <action> </action> tags.
"""


SCIENCEWORLD_GRPO_REASON = """
You are an expert agent operating in the ScienceWorld Environment, a simulated science laboratory where you perform experiments and solve science tasks.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Current available actions:
{admissible_actions}

Now it's your turn to take an action.
First, briefly explain your reasoning in one sentence. Then, provide your chosen action enclosed in <action> </action> tags.
"""


_SKILL_INSTRUCTION_ACTIVE = (
    "Check the Available Skills section by reading each `when_to_use` field. "
    "If a skill matches your situation, state which skill you are applying "
    "and follow its `action` guidance; otherwise, proceed on your own."
)


SCIENCEWORLD_RESKILL_REASON_NO_HIS = """
You are an expert agent operating in the ScienceWorld Environment, a simulated science laboratory where you perform experiments and solve science tasks.
Your task is to: {task_description}
Your current observation is: {current_observation}

Current available actions:
{admissible_actions}
{triggered_skills_section}
Now it's your turn to take an action.
""" + _SKILL_INSTRUCTION_ACTIVE + """
First, briefly explain your reasoning in one sentence (including which skill you applied, if any). Then, provide your chosen action enclosed in <action> </action> tags.
"""


SCIENCEWORLD_RESKILL_REASON = """
You are an expert agent operating in the ScienceWorld Environment, a simulated science laboratory where you perform experiments and solve science tasks.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Current available actions:
{admissible_actions}
{triggered_skills_section}
Now it's your turn to take an action.
""" + _SKILL_INSTRUCTION_ACTIVE + """
First, briefly explain your reasoning in one sentence (including which skill you applied, if any). Then, provide your chosen action enclosed in <action> </action> tags.
"""


def format_skills_section(skill_text: str) -> str:
    """Wrap skill text in the Available Skills section. Always present."""
    if not skill_text:
        return "\n## Available Skills\n\n(no skills available)\n"
    return f"\n## Available Skills\n\n{skill_text}\n"
