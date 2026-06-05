"""Standalone base classes for reskill.

Pure utility classes with no skill loading logic.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Any, Optional, Set
from collections import defaultdict

import torch
import numpy as np
import os


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


# ---------------------------------------------------------------------------
# Memory classes
# ---------------------------------------------------------------------------

class BaseMemory(ABC):
    """Base class for memory management."""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def reset(self, batch_size: int):
        pass

    @abstractmethod
    def store(self, record: Dict[str, List[Any]]):
        pass

    @abstractmethod
    def fetch(self, step: int):
        pass


class SimpleMemory(BaseMemory):
    """Memory manager: stores & fetches per-environment history records."""

    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths


class SearchMemory(BaseMemory):
    """Memory manager for search tasks."""

    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "information",
        action_key: str = "search",
    ) -> Tuple[List[str], List[int]]:
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"Step {step_num}:{act} {obs}\n"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths


# ---------------------------------------------------------------------------
# Environment manager base
# ---------------------------------------------------------------------------

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None,
            'image': next_obs,
            'anchor': None,
        }
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self) -> List[str]:
        pass

    def close(self) -> None:
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)

        success = defaultdict(list)

        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)

        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def save_image(self, image, step):
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")

        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)

        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)


# ---------------------------------------------------------------------------
# ReSkill mixin — skill injection, A/B testing, trigger tracking
# ---------------------------------------------------------------------------

class ReSkillEnvManagerBase:
    """Mixin providing shared ReSkill logic for all env-specific managers.

    Use via multiple inheritance, placing this before the environment manager:
        class ReSkillFooEnvManager(ReSkillEnvManagerBase, FooEnvironmentManager)

    Subclass responsibilities:
        - __init__: call super().__init__() then set any env-specific config
          (e.g. max_obs_chars with an environment-specific obs_char_ratio)
        - set_skill_components: call super() then add retrieval_memory if needed
        - build_text_obs: fully environment-specific, must be implemented
        - _trim_*_observation: fully environment-specific, must be implemented
    """

    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

        # Set by trainer via set_skill_components()
        self.skill_loader = None
        self.version_ab_tracker = None
        self._version_old_registry = None

        # Per-episode state — reset at the start of each rollout
        self._version_assignments: Dict[int, bool] = {}
        self._testing_decisions: Dict[int, Dict[str, bool]] = {}
        self._episode_trigger_log: Dict[int, Set[str]] = {}
        self._decisions_sampled: bool = False
        self._num_slots: int = 0

    def set_skill_components(self, skill_loader, version_ab_tracker=None):
        """Inject skill components from the trainer."""
        self.skill_loader = skill_loader
        self.version_ab_tracker = version_ab_tracker

    def set_version_old_registry(self, old_registry):
        """Set/clear the old registry snapshot for version A/B testing."""
        self._version_old_registry = old_registry

    def reset(self, kwargs) -> Dict[str, Any]:
        """Reset environments and pre-sample A/B testing decisions.

        Clears per-episode state BEFORE super().reset() so that build_text_obs
        (called at init step inside reset) sees a clean slate.
        """
        self._episode_trigger_log = {}
        self._testing_decisions = {}
        self._version_assignments = {}
        self._decisions_sampled = False

        result = super().reset(kwargs)
        self._num_slots = len(self.tasks)

        if not self._decisions_sampled:
            self._sample_testing_decisions()

        return result

    def _sample_testing_decisions(self):
        """Thompson-sample new vs. old version assignment per slot."""
        self._testing_decisions = {}
        self._version_assignments = {}

        if (self.version_ab_tracker is not None
                and self.version_ab_tracker.is_testing):
            for i in range(self._num_slots):
                self._version_assignments[i] = \
                    self.version_ab_tracker.sample_rollout_decision()

    def _get_slot_registry(self, slot_idx: int):
        """Return the skill registry for a slot (old version if A/B assigned)."""
        if (self._version_old_registry is not None
                and not self._version_assignments.get(slot_idx, True)):
            return self._version_old_registry
        return self.skill_loader.registry

    def get_episode_trigger_data(self) -> Dict[int, Tuple[Set[str], Dict[str, bool]]]:
        """Return per-slot (triggered_skill_ids, testing_decisions) after rollout."""
        return {
            i: (self._episode_trigger_log.get(i, set()),
                self._testing_decisions.get(i, {}))
            for i in range(self._num_slots)
        }

    def get_version_assignments(self) -> Dict[int, bool]:
        """Return per-slot version assignments (slot_idx -> is_new_version)."""
        return dict(self._version_assignments)

    def get_trigger_frequency_stats(self) -> Dict[str, float]:
        """Fraction of slots where each skill triggered this rollout."""
        if not self._episode_trigger_log or self._num_slots == 0:
            return {}
        skill_counts: Dict[str, int] = {}
        for slot_triggers in self._episode_trigger_log.values():
            for skill_id in slot_triggers:
                skill_counts[skill_id] = skill_counts.get(skill_id, 0) + 1
        return {sid: count / self._num_slots
                for sid, count in skill_counts.items()}
