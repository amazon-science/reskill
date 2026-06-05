"""Environment manager for the registered ALFWorld environment.

This manager owns observation formatting, prompt template selection, and
history memory for embodied text tasks.
"""

import logging
from typing import List, Dict, Any

from reskill.environments.base import (
    EnvironmentManagerBase, SimpleMemory, to_numpy,
)
from reskill.environments.alfworld.prompts import (
    ALFWORLD_GRPO_REASON_NO_HIS,
    ALFWORLD_GRPO_REASON,
)

logger = logging.getLogger(__name__)


def parse_gamefile(infos):
    """Extract gamefile paths from info dicts."""
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile


def set_gamefile(infos, gamefile):
    """Set gamefile paths into info dicts if missing."""
    for i in range(len(infos)):
        if 'extra.gamefile' not in infos[i] or infos[i]['extra.gamefile'] is None:
            infos[i]['extra.gamefile'] = gamefile[i]
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    """Base rollout manager for embodied text ReSkill environments."""

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()

        max_prompt_tokens = getattr(config.data, 'max_prompt_length', 4096)
        obs_char_ratio = config.env.get('obs_char_ratio', 1.8)
        self.max_obs_chars = int(max_prompt_tokens * obs_char_ratio)

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
        self.gamefile = parse_gamefile(infos)

        # Initialize memory before building the first prompted observation.
        self.memory.reset(batch_size=len(text_obs))

        self.tasks = []
        self.extract_task(text_obs)
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

        # Ensure gamefile is set in infos
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

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

    def extract_task(self, text_obs: List[str]):
        """Parse task description from initial observations."""
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            if task_start != -1:
                self.tasks.append(
                    obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError(
                    "Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str],
                       admissible_actions: List[List[str]],
                       init: bool = False) -> List[str]:
        """Build formatted text observations with history."""
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action")

        tpl_no_his = ALFWORLD_GRPO_REASON_NO_HIS
        tpl_with_his = ALFWORLD_GRPO_REASON

        for i in range(len(text_obs)):
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = tpl_no_his.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                )
            else:
                obs = tpl_with_his.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
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
                        admissible_actions=reformatted_admissible_actions,
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
                        admissible_actions=reformatted_admissible_actions,
                    )
                # Final hard fallback: rebuild with no-history template
                if len(obs) > self.max_obs_chars:
                    logger.warning(
                        f"[ALFWorldEnvManager] Prompt still {len(obs)} chars "
                        f"(budget {self.max_obs_chars}) after all trimming. "
                        f"Falling back to no-history template.")
                    obs = tpl_no_his.format(
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                # Process game file for per-task-type metrics
                gamefile = info.get("extra.gamefile", "")
                self._process_gamefile(gamefile, won_value, success)
                return

    TASK_TYPES = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]

    def _process_gamefile(self, gamefile, won_value, success):
        """Track per-task-type success rates. Append nan for non-matching types
        so all task arrays stay the same length as success_rate."""
        matched = None
        for task in self.TASK_TYPES:
            if task in gamefile:
                matched = task
                break
        for task in self.TASK_TYPES:
            key = f"{task}_success_rate"
            if task == matched:
                success[key].append(won_value)
            else:
                success[key].append(float('nan'))
