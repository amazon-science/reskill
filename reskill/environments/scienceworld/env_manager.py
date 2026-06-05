"""Environment manager for the registered ScienceWorld environment.

This manager owns observation formatting, prompt template selection, history
memory, and task/topic metrics for science-experiment tasks.
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict

from reskill.environments.base import (
    EnvironmentManagerBase, SimpleMemory, to_numpy,
)
from reskill.environments.scienceworld.prompts import (
    SCIENCEWORLD_GRPO_REASON_NO_HIS,
    SCIENCEWORLD_GRPO_REASON,
)

logger = logging.getLogger(__name__)

# Map task names to their topic for grouped logging.
TASK_NAME_TO_TOPIC = {
    'boil': 'Matter',
    'melt': 'Matter',
    'freeze': 'Matter',
    'change-the-state-of-matter-of': 'Matter',
    'use-thermometer': 'Measurement',
    'measure-melting-point-known-substance': 'Measurement',
    'measure-melting-point-unknown-substance': 'Measurement',
    'power-component': 'Electricity',
    'power-component-renewable-vs-nonrenewable-energy': 'Electricity',
    'test-conductivity': 'Electricity',
    'test-conductivity-of-unknown-substances': 'Electricity',
    'find-living-thing': 'Classification',
    'find-non-living-thing': 'Classification',
    'find-plant': 'Classification',
    'find-animal': 'Classification',
    'grow-plant': 'Biology',
    'grow-fruit': 'Biology',
    'lifespan-longest-lived': 'Biology',
    'lifespan-shortest-lived': 'Biology',
    'lifespan-longest-lived-then-shortest-lived': 'Biology',
    'identify-life-stages-1': 'Biology',
    'identify-life-stages-2': 'Biology',
    'chemistry-mix': 'Chemistry',
    'chemistry-mix-paint-secondary-color': 'Chemistry',
    'chemistry-mix-paint-tertiary-color': 'Chemistry',
    'inclined-plane-determine-angle': 'Forces',
    'inclined-plane-friction-named-surfaces': 'Forces',
    'inclined-plane-friction-unnamed-surfaces': 'Forces',
    'mendelian-genetics-known-plant': 'Biology',
    'mendelian-genetics-unknown-plant': 'Biology',
}

TASK_TOPICS = sorted(set(TASK_NAME_TO_TOPIC.values()))


class ScienceWorldEnvironmentManager(EnvironmentManagerBase):
    """Base rollout manager for science-experiment ReSkill environments."""

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()

        max_prompt_tokens = getattr(config.data, 'max_prompt_length', 4096)
        self.max_obs_chars = int(max_prompt_tokens * 1.8)

        self.retrieval_memory = None
        self._memory_type = None

        # Episode ID tracking
        self._next_episode_id = 0
        self._active_episode_ids = {}

        super().__init__(envs, projection_f, config)

    def _assign_episode_id(self):
        eid = self._next_episode_id
        self._next_episode_id += 1
        return eid

    def reset(self, kwargs) -> Dict[str, Any]:
        text_obs, image_obs, infos = self.envs.reset()

        self.memory.reset(batch_size=len(text_obs))

        self.tasks = []
        self.task_names_per_slot = []
        self.extract_task(infos)
        self.pre_text_obs = text_obs

        self._active_episode_ids = {
            i: self._assign_episode_id() for i in range(len(infos))
        }

        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands, init=True)

        observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(
            text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
        }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, infos: List[Dict]):
        """Extract task descriptions from info dicts.

        The underlying environment provides task descriptions in its info dict.
        """
        for info in infos:
            task_desc = info.get('taskDesc', '')
            if not task_desc:
                logger.warning("No taskDesc found in ScienceWorld info dict")
                task_desc = "Complete the science task."
            self.tasks.append(task_desc)
            self.task_names_per_slot.append(info.get('taskName', 'unknown'))

    def build_text_obs(self, text_obs: List[str],
                       admissible_actions,
                       init: bool = False) -> List[str]:
        """Build formatted text observations with history.

        admissible_actions is a list of hint strings containing action
        templates and observable objects.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action")

        tpl_no_his = SCIENCEWORLD_GRPO_REASON_NO_HIS
        tpl_with_his = SCIENCEWORLD_GRPO_REASON

        for i in range(len(text_obs)):
            actions_hint = admissible_actions[i] if admissible_actions[i] else ''

            if init or self.config.env.history_length <= 0:
                obs = tpl_no_his.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    admissible_actions=actions_hint,
                )
            else:
                obs = tpl_with_his.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=actions_hint,
                )

            # Trim if too long
            if (len(obs) > self.max_obs_chars and not init
                    and self.config.env.history_length > 0):
                history_lines = (memory_contexts[i].split("\n")
                                 if memory_contexts[i] else [])
                while len(obs) > self.max_obs_chars and len(history_lines) > 1:
                    history_lines = history_lines[1:]
                    trimmed_history = "\n".join(history_lines)
                    obs = tpl_with_his.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=len(history_lines),
                        action_history=trimmed_history,
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        admissible_actions=actions_hint,
                    )
                if len(obs) > self.max_obs_chars:
                    overflow = len(obs) - self.max_obs_chars
                    truncated_obs = (
                        text_obs[i][:len(text_obs[i]) - overflow - 50]
                        + "... (truncated)")
                    obs = tpl_with_his.format(
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=0,
                        action_history="(trimmed)",
                        current_step=len(self.memory[i]) + 1,
                        current_observation=truncated_obs,
                        admissible_actions=actions_hint,
                    )
                # Final fallback: no-history template
                if len(obs) > self.max_obs_chars:
                    logger.warning(
                        f"[ScienceWorldEnvManager] Prompt still {len(obs)} chars "
                        f"(budget {self.max_obs_chars}) after all trimming. "
                        f"Falling back to no-history template.")
                    obs = tpl_no_his.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        admissible_actions=actions_hint,
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """Evaluate success for logging. Overrides base to use ScienceWorld scoring."""
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                score = float(info.get('score', 0))
                won_value = float(score >= 100)
                success['success_rate'].append(won_value)
                success['scienceworld_score'].append(score / 100.0)

                # Per-task-type and per-topic metrics
                task_name = info.get('taskName', 'unknown')
                self._process_task_type(task_name, won_value, score, success)
                return

    def _process_task_type(self, task_name, won_value, score, success):
        """Track per-topic success rates."""
        topic = TASK_NAME_TO_TOPIC.get(task_name, 'Unknown')
        for t in TASK_TOPICS:
            key = f"{t}_success_rate"
            if t == topic:
                success[key].append(won_value)
            else:
                success[key].append(float('nan'))
