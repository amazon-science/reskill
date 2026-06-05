"""Skill-aware manager for the registered ScienceWorld environment."""

import logging
from typing import Optional, List

from reskill.environments.base import ReSkillEnvManagerBase
from reskill.environments.scienceworld.env_manager import ScienceWorldEnvironmentManager
from reskill.environments.scienceworld.prompts import (
    SCIENCEWORLD_RESKILL_REASON_NO_HIS,
    SCIENCEWORLD_RESKILL_REASON,
    format_skills_section,
)
from reskill.skill_serving.trigger_matcher import TriggerMatcher
from reskill.skill_serving.skill_loader import SkillLoader
from reskill.skill_serving.skill_ab_tracker import VersionABTracker

logger = logging.getLogger(__name__)


class ReSkillScienceWorldEnvManager(ReSkillEnvManagerBase, ScienceWorldEnvironmentManager):
    """Adds per-step skill trigger matching and A/B testing."""

    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def set_skill_components(self, skill_loader: SkillLoader,
                             version_ab_tracker: Optional[VersionABTracker] = None):
        super().set_skill_components(skill_loader, version_ab_tracker)
        self.retrieval_memory = skill_loader
        self._memory_type = 'prompt'

    def build_text_obs(self, text_obs: List[str],
                       admissible_actions: List[List[str]],
                       init: bool = False) -> List[str]:
        if init and not self._decisions_sampled:
            self._num_slots = len(text_obs)
            self._sample_testing_decisions()
            self._decisions_sampled = True

        postprocess_text_obs = []

        memory_contexts = None
        valid_lens = None
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs", action_key="action")

        tpl_no_his = SCIENCEWORLD_RESKILL_REASON_NO_HIS
        tpl_with_his = SCIENCEWORLD_RESKILL_REASON

        for i in range(len(text_obs)):
            actions_hint = admissible_actions[i] if admissible_actions[i] else ''

            if init:
                step_num = 0
                last_action = None
            else:
                step_num = len(self.memory[i]) if self.memory._data else 0
                last_action = None
                if self.memory._data and self.memory[i]:
                    last_action = self.memory[i][-1].get('action', '')

            skill_text = ""
            if self.skill_loader is not None:
                slot_registry = self._get_slot_registry(i)
                if slot_registry is not self.skill_loader.registry:
                    triggered = slot_registry.get_triggered_skills(
                        step_num=step_num, last_action=last_action)
                    skill_text = SkillLoader._format_skills(triggered)
                else:
                    skill_text = self.skill_loader.format_for_prompt(
                        retrieved=None, step_num=step_num,
                        last_action=last_action,
                        testing_decisions=self._testing_decisions.get(i, {}))
                all_skills = slot_registry.get_all_skills()
                triggered_ids = TriggerMatcher.get_triggered_skill_ids(
                    all_skills, step_num, last_action)
                self._episode_trigger_log.setdefault(i, set()).update(triggered_ids)

            skills_section = format_skills_section(skill_text)

            if init or self.config.env.history_length <= 0:
                obs = tpl_no_his.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    admissible_actions=actions_hint,
                    triggered_skills_section=skills_section)
            else:
                obs = tpl_with_his.format(
                    task_description=self.tasks[i],
                    triggered_skills_section=skills_section,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=actions_hint)

            if (len(obs) > self.max_obs_chars and not init
                    and self.config.env.history_length > 0):
                obs = self._trim_scienceworld_observation(
                    obs, text_obs[i], actions_hint, skills_section, i,
                    memory_contexts[i], valid_lens[i],
                    tpl_with_his, tpl_no_his)

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _trim_scienceworld_observation(self, obs, raw_obs, actions_hint,
                                       skills_section, slot_idx,
                                       memory_context, valid_len,
                                       template, tpl_no_his=None):
        history_lines = memory_context.split("\n") if memory_context else []
        while len(obs) > self.max_obs_chars and len(history_lines) > 1:
            history_lines = history_lines[1:]
            obs = template.format(
                task_description=self.tasks[slot_idx],
                triggered_skills_section=skills_section,
                step_count=len(self.memory[slot_idx]),
                history_length=len(history_lines),
                action_history="\n".join(history_lines),
                current_step=len(self.memory[slot_idx]) + 1,
                current_observation=raw_obs,
                admissible_actions=actions_hint)
        if len(obs) > self.max_obs_chars:
            overflow = len(obs) - self.max_obs_chars
            truncated = raw_obs[:len(raw_obs) - overflow - 50] + "... (truncated)"
            obs = template.format(
                task_description=self.tasks[slot_idx],
                triggered_skills_section=skills_section,
                step_count=len(self.memory[slot_idx]),
                history_length=0, action_history="(trimmed)",
                current_step=len(self.memory[slot_idx]) + 1,
                current_observation=truncated,
                admissible_actions=actions_hint)
        if len(obs) > self.max_obs_chars and tpl_no_his is not None:
            logger.warning(
                f"[ReSkillScienceWorldEnvManager] Prompt still {len(obs)} chars "
                f"(budget {self.max_obs_chars}). Falling back to no-history.")
            obs = tpl_no_his.format(
                task_description=self.tasks[slot_idx],
                current_observation=raw_obs,
                admissible_actions=actions_hint,
                triggered_skills_section=skills_section)
        return obs
