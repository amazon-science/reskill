"""Skill-aware manager for the registered search environment."""

import logging
from typing import Optional, Dict, List, Set

from reskill.environments.base import ReSkillEnvManagerBase
from reskill.environments.search.env_manager import SearchEnvironmentManager
from reskill.environments.search.prompts import (
    SEARCH_RESKILL_REASON_NO_HIS,
    SEARCH_RESKILL_REASON,
    format_skills_section,
)
from reskill.skill_serving.trigger_matcher import TriggerMatcher
from reskill.skill_serving.skill_loader import SkillLoader
from reskill.skill_serving.skill_ab_tracker import VersionABTracker

logger = logging.getLogger(__name__)


class ReSkillSearchEnvManager(ReSkillEnvManagerBase, SearchEnvironmentManager):
    """Adds per-step skill trigger matching and A/B testing."""

    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

        # Search prompts are token-dense; use a tighter char budget.
        max_prompt_tokens = getattr(config.data, 'max_prompt_length', 4096)
        obs_char_ratio = config.env.get('obs_char_ratio', 2.5)
        self.max_obs_chars = int(max_prompt_tokens * obs_char_ratio)

    def build_text_obs(self, text_obs: List[str],
                       init: bool = False) -> List[str]:
        if init and not self._decisions_sampled:
            self._num_slots = len(text_obs)
            self._sample_testing_decisions()
            self._decisions_sampled = True

        postprocess_text_obs = []

        memory_contexts = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search")

        for i in range(len(text_obs)):
            if init:
                step_num = 0
                last_action = None
            else:
                step_num = len(self.memory[i]) if self.memory._data else 0
                last_action = None
                if self.memory._data and self.memory[i]:
                    last_action = self.memory[i][-1].get('search', '')

            skill_text = ""
            if self.skill_loader is not None:
                slot_registry = self._get_slot_registry(i)
                if slot_registry is not self.skill_loader.registry:
                    triggered = slot_registry.get_triggered_skills(
                        step_num=step_num, last_action=last_action)
                    skill_text = SkillLoader._format_skills(triggered)
                else:
                    skill_text = self.skill_loader.format_for_prompt(
                        retrieved=None,
                        step_num=step_num,
                        last_action=last_action,
                        testing_decisions=self._testing_decisions.get(i, {}),
                    )
                all_skills = slot_registry.get_all_skills()
                triggered_ids = TriggerMatcher.get_triggered_skill_ids(
                    all_skills, step_num, last_action)
                self._episode_trigger_log.setdefault(i, set()).update(triggered_ids)

            skills_section = format_skills_section(skill_text)

            tpl_no_his = SEARCH_RESKILL_REASON_NO_HIS
            tpl_with_his = SEARCH_RESKILL_REASON

            if init or self.config.env.history_length <= 0:
                obs = tpl_no_his.format(
                    task_description=self.tasks[i],
                    triggered_skills_section=skills_section,
                )
            else:
                obs = tpl_with_his.format(
                    task_description=self.tasks[i],
                    triggered_skills_section=skills_section,
                    step_count=len(self.memory[i]),
                    memory_context=memory_contexts[i],
                )
                if len(obs) > self.max_obs_chars:
                    obs = self._trim_search_observation(
                        obs, self.tasks[i], skills_section,
                        memory_contexts[i], len(self.memory[i]),
                        tpl_with_his, tpl_no_his)

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _trim_search_observation(self, obs: str, task: str,
                                 skills_section: str,
                                 memory_context: str,
                                 step_count: int,
                                 tpl_with_his: str = None,
                                 tpl_no_his: str = None) -> str:
        if tpl_with_his is None:
            tpl_with_his = SEARCH_RESKILL_REASON

        history_lines = memory_context.split("\n") if memory_context else []

        while len(obs) > self.max_obs_chars and len(history_lines) > 1:
            history_lines = history_lines[1:]
            obs = tpl_with_his.format(
                task_description=task,
                triggered_skills_section=skills_section,
                step_count=step_count,
                memory_context="\n".join(history_lines),
            )

        if len(obs) > self.max_obs_chars:
            obs = tpl_with_his.format(
                task_description=task,
                triggered_skills_section=skills_section,
                step_count=step_count,
                memory_context="(history trimmed to fit prompt budget)",
            )

        if len(obs) > self.max_obs_chars:
            overflow = len(obs) - self.max_obs_chars
            if len(skills_section) > overflow + 50:
                trimmed_skills = skills_section[:len(skills_section) - overflow - 30] + "\n(skills trimmed)\n"
            else:
                trimmed_skills = ""
            obs = tpl_with_his.format(
                task_description=task,
                triggered_skills_section=trimmed_skills,
                step_count=step_count,
                memory_context="(history trimmed to fit prompt budget)",
            )

        if len(obs) > self.max_obs_chars:
            logger.warning(
                f"[ReSkillSearchEnvManager] Prompt still {len(obs)} chars "
                f"(budget {self.max_obs_chars}) after all trimming. "
                f"Falling back to no-history template.")
            if tpl_no_his is None:
                tpl_no_his = SEARCH_RESKILL_REASON_NO_HIS
            obs = tpl_no_his.format(
                task_description=task,
                triggered_skills_section=skills_section,
            )

        return obs
