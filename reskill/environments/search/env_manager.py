"""Environment manager for the registered search environment.

This manager owns observation formatting, prompt template selection, and
history memory for search-style tasks.
"""

import logging
from typing import List, Tuple, Dict, Any

from reskill.environments.base import (
    EnvironmentManagerBase, SearchMemory, to_numpy,
)
from reskill.environments.search.prompts import (
    SEARCH_GRPO_REASON_NO_HIS,
    SEARCH_GRPO_REASON,
)

logger = logging.getLogger(__name__)


class SearchEnvironmentManager(EnvironmentManagerBase):
    """Base rollout manager for search-style ReSkill environments."""

    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()

        # Character budget for prompt trimming.
        # Search defaults to a tighter multiplier than other text environments
        # because retrieval passages have dense entities/dates that tokenize
        # at ~1.5 chars/token, plus the chat template adds ~100 tokens overhead.
        max_prompt_tokens = getattr(config.data, 'max_prompt_length', 4096)
        obs_char_ratio = config.env.get('obs_char_ratio', 1.2)
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

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        self.kwargs = kwargs
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))

        self._active_episode_ids = {
            i: self._assign_episode_id() for i in range(len(obs))
        }

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False,
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search",
            )

        tpl_no_his = SEARCH_GRPO_REASON_NO_HIS
        tpl_with_his = SEARCH_GRPO_REASON

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = tpl_no_his.format(
                    task_description=self.tasks[i],
                )
            else:
                obs_i = tpl_with_his.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )

                # Trim if too long — progressively remove oldest history
                if len(obs_i) > self.max_obs_chars:
                    obs_i = self._trim_observation(
                        obs_i, self.tasks[i], memory_ctx[i],
                        len(self.memory[i]), tpl_with_his, tpl_no_his)

            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _trim_observation(self, obs: str, task: str, memory_context: str,
                          step_count: int, tpl_with_his: str,
                          tpl_no_his: str) -> str:
        """Progressively trim search history to fit within max_obs_chars.

        Search results can be very long (multi-paragraph passages), so
        history is the main source of prompt bloat. Trimming priority:
          1. Task description — always kept
          2. Recent history — most relevant
          3. Old history — trimmed first
        """
        history_lines = memory_context.split("\n") if memory_context else []

        # Stage 1: Remove oldest history lines one at a time
        while len(obs) > self.max_obs_chars and len(history_lines) > 1:
            history_lines = history_lines[1:]
            trimmed_context = "\n".join(history_lines)
            obs = tpl_with_his.format(
                task_description=task,
                memory_context=trimmed_context,
                step_count=step_count,
            )

        # Stage 2: All history trimmed to 1 line, still too long — drop entirely
        if len(obs) > self.max_obs_chars:
            obs = tpl_with_his.format(
                task_description=task,
                memory_context="(history trimmed to fit prompt budget)",
                step_count=step_count,
            )

        # Stage 3: Final hard fallback — no-history template
        if len(obs) > self.max_obs_chars:
            logger.warning(
                f"[SearchEnvManager] Prompt still {len(obs)} chars "
                f"(budget {self.max_obs_chars}) after all trimming. "
                f"Falling back to no-history template.")
            obs = tpl_no_his.format(
                task_description=task,
            )

        return obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return
