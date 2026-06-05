"""ScienceWorld environment wrapper with Ray-based parallelism.

- ScienceWorldWorker: Ray remote actor holding one ScienceWorldEnv instance
- ScienceWorldEnvs: gym.Env that manages a pool of workers
- build_scienceworld_envs: factory function used by the environment spec

Task pool is a flat list of (taskName, variationIdx) pairs, matching
agent-lightning's parquet-based approach. Workers randomly sample from
this pool using a seeded RNG (matching DataLoader + RandomSampler).
Tasks with more variations appear more often (variation-weighted).
Workers in the same GRPO group share a seed for group consistency.
"""

import logging
import random

import gymnasium as gym
import numpy as np
import ray

from scienceworld import ScienceWorldEnv

logger = logging.getLogger(__name__)


def _build_task_pool(split: str, task_names=None, max_variations=None):
    """Build a flat task pool: list of (taskName, variationIdx) pairs for the split.

    Creates a temporary ScienceWorldEnv to query available variations.
    Returns a flat list where each (task, variation) pair appears once,
    matching agent-lightning's parquet-based approach. Tasks with more
    variations naturally appear more often when sampling from this list
    (variation-weighted sampling).

    Args:
        split: One of 'train', 'dev', 'test'.
        task_names: Optional list of task names to include. If None, all 30 tasks.
        max_variations: Cap per task. None means use all available.

    Returns:
        List of (taskName, variationIdx) tuples, sorted for determinism.
    """
    env = ScienceWorldEnv(envStepLimit=10)
    all_task_names = env.get_task_names()

    if task_names is not None:
        all_task_names = [t for t in all_task_names if t in task_names]
        if not all_task_names:
            raise ValueError(f"No matching tasks found. Available: {env.get_task_names()}")

    task_pool = []
    task_counts = {}
    for task_name in sorted(all_task_names):
        env.load(task_name, 0, "")
        if split == 'train':
            variations = env.get_variations_train()
        elif split == 'dev':
            variations = env.get_variations_dev()
        elif split == 'test':
            variations = env.get_variations_test()
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'dev', or 'test'.")

        var_list = [int(v) for v in variations]
        if max_variations is not None:
            var_list = var_list[:max_variations]
        for v in var_list:
            task_pool.append((task_name, v))
        if var_list:
            task_counts[task_name] = len(var_list)

    env.close()

    # Sort for deterministic ordering across workers.
    # Randomness comes from the worker's seeded RNG at sample time,
    # not from list order.
    task_pool.sort()

    logger.info(f"[ScienceWorld] Built task pool: split={split}, "
                f"tasks={len(task_counts)}, total_pairs={len(task_pool)}")
    for tn, cnt in sorted(task_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {tn}: {cnt} variations ({cnt*100/len(task_pool):.1f}%)")
    return task_pool


# Per-task max steps, capped at 50 to prevent OOM from long episodes.
# Agent-lightning uses higher limits (up to 120) but with a smaller model
# and more GPU memory. We cap at 50 for memory safety.
MAX_STEPS_PER_TASK = {
    "boil": 50,
    "change-the-state-of-matter-of": 50,
    "freeze": 50,
    "melt": 50,
    "measure-melting-point-known-substance": 50,
    "measure-melting-point-unknown-substance": 50,
    "use-thermometer": 30,
    "power-component": 30,
    "power-component-renewable-vs-nonrenewable-energy": 30,
    "test-conductivity": 30,
    "test-conductivity-of-unknown-substances": 30,
    "find-animal": 30,
    "find-living-thing": 30,
    "find-non-living-thing": 30,
    "find-plant": 30,
    "grow-fruit": 50,
    "grow-plant": 30,
    "chemistry-mix": 50,
    "chemistry-mix-paint-secondary-color": 30,
    "chemistry-mix-paint-tertiary-color": 30,
    "lifespan-longest-lived": 30,
    "lifespan-longest-lived-then-shortest-lived": 30,
    "lifespan-shortest-lived": 30,
    "identify-life-stages-1": 30,
    "identify-life-stages-2": 30,
    "inclined-plane-determine-angle": 30,
    "inclined-plane-friction-named-surfaces": 30,
    "inclined-plane-friction-unnamed-surfaces": 30,
    "mendelian-genetics-known-plant": 30,
    "mendelian-genetics-unknown-plant": 30,
}
DEFAULT_MAX_STEPS = 50


def compute_reward(info, done):
    """Compute reward from ScienceWorld step info.

    Binary reward: 10.0 if score > 70 at done, else 0.0.
    Matches agent-lightning's approach (use_success_rate=True with
    success threshold at score > 70). The 70% threshold is forgiving —
    many tasks award 70+ for completing the main subgoal.

    Returns 0.0 at intermediate steps to match ALFWorld's pattern.
    """
    if done:
        return 10.0 if info.get('score', 0) > 70 else 0.0
    return 0.0


def _build_actions_hint(env):
    """Build available actions hint: action templates + observable objects.

    Matches agent-lightning's format from agl_envs/scienceworld/base.py:69-71.
    Shows the verbs and the nouns the model can combine them with.
    """
    valid_actions = env.get_possible_actions()
    valid_objs = env.get_possible_objects()
    return (f"Valid_actions: {valid_actions}, OBJ needs to be replaced "
            f"with one of the following objects: {valid_objs}\n "
            f"example: focus on door")


class ScienceWorldWorker:
    """Ray remote actor that holds one ScienceWorld environment instance.

    Uses random sampling from a flat list of (taskName, variationIdx) pairs,
    matching agent-lightning's DataLoader with RandomSampler. Workers in the
    same GRPO group share the same RNG seed so they pick the same task at
    each reset. Tasks with more variations appear more often (variation-weighted).
    """

    def __init__(self, task_pool, group_seed, env_step_limit, simplification_str):
        """
        Args:
            task_pool: List of (taskName, variationIdx) tuples.
            group_seed: Shared seed for all workers in the same GRPO group.
            env_step_limit: Fallback max steps (used if task not in MAX_STEPS_PER_TASK).
            simplification_str: ScienceWorld simplification string.
        """
        self.task_pool = task_pool  # flat list of (taskName, varIdx)
        self.group_seed = group_seed
        self.default_env_step_limit = env_step_limit
        self.simplification_str = simplification_str
        # Seeded RNG for random sampling — workers in the same GRPO group
        # share the same group_seed so they draw the same sequence.
        self._rng = random.Random(group_seed)

        self.env = ScienceWorldEnv(envStepLimit=env_step_limit)

    def step(self, action):
        """Execute one step in the environment."""
        obs, reward, done, info = self.env.step(action)
        # Add actions hint (templates + objects) for prompt building
        info['actions_hint'] = _build_actions_hint(self.env)
        return obs, reward, done, info

    def reset(self):
        """Reset with a new task+variation using random sampling.

        Randomly selects a (taskName, variationIdx) pair from the pool.
        Workers in the same GRPO group share an RNG seed so they draw
        the same pair, ensuring group consistency.
        """
        # Random sample from the flat pool (variation-weighted)
        task_name, var_idx = self._rng.choice(self.task_pool)

        # Set per-task step limit
        self.env.envStepLimit = MAX_STEPS_PER_TASK.get(
            task_name, self.default_env_step_limit)

        self.env.load(task_name, var_idx, self.simplification_str)
        obs, info = self.env.reset()
        # Add actions hint (templates + objects) for prompt building
        info['actions_hint'] = _build_actions_hint(self.env)
        return obs, info

    def close(self):
        """Shut down the JVM."""
        try:
            self.env.close()
        except Exception:
            pass


class ScienceWorldEnvs(gym.Env):
    """Pool of ScienceWorld environments managed via Ray remote actors.

    Interface matches AlfworldEnvs for compatibility with the env manager layer.
    """

    def __init__(self, seed, env_num, group_n, resources_per_worker,
                 is_train=True, env_kwargs=None, val_splits=None):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        env_kwargs = env_kwargs or {}
        task_names = env_kwargs.get('task_names', None)
        env_step_limit = env_kwargs.get('env_step_limit', 100)
        simplification_str = env_kwargs.get('simplification_str', 'easy')
        max_variations = env_kwargs.get('max_variations', None)

        self.group_n = group_n
        self.num_processes = env_num * group_n

        env_worker = ray.remote(**resources_per_worker)(ScienceWorldWorker)
        self.workers = []

        if val_splits and not is_train:
            # Multi-split validation: distribute workers evenly across splits
            n_splits = len(val_splits)
            assert self.num_processes % n_splits == 0, (
                f"Total workers ({self.num_processes}) must be divisible by "
                f"number of val_splits ({n_splits})")
            workers_per_split = self.num_processes // n_splits

            for split_idx, split_name in enumerate(val_splits):
                task_pool = _build_task_pool(split_name, task_names, max_variations)
                offset = split_idx * workers_per_split
                for i in range(workers_per_split):
                    group_seed = seed + ((offset + i) // self.group_n)
                    worker = env_worker.remote(
                        task_pool, group_seed, env_step_limit, simplification_str)
                    self.workers.append(worker)

            logger.info(f"[ScienceWorldEnvs] Multi-split val: {val_splits}, "
                        f"{workers_per_split} workers each, {self.num_processes} total")
        else:
            split = 'train' if is_train else 'dev'
            task_pool = _build_task_pool(split, task_names, max_variations)
            for i in range(self.num_processes):
                group_seed = seed + (i // self.group_n)
                worker = env_worker.remote(
                    task_pool, group_seed, env_step_limit, simplification_str)
                self.workers.append(worker)

            logger.info(f"[ScienceWorldEnvs] split={split}, "
                        f"pool_size={len(task_pool)}, workers={self.num_processes}")

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            f"Expected {self.num_processes} actions, got {len(actions)}"

        futures = []
        for i, worker in enumerate(self.workers):
            futures.append(worker.step.remote(actions[i]))

        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, reward, done, info) in enumerate(results):
            text_obs_list.append(obs)
            dones_list.append(done)

            # 'won' = score > 70 (matches agent-lightning threshold)
            info['won'] = bool(done and info.get('score', 0) > 70)

            info_list.append(info)

            # Cache actions hint (templates + objects) for prompt building
            self.prev_admissible_commands[i] = info.get('actions_hint', '')

            # Binary reward: 10.0 if score > 70 at done, else 0.0
            rewards_list.append(compute_reward(info, done))

        return text_obs_list, None, rewards_list, dones_list, info_list

    def reset(self):
        """Reset all workers and collect initial observations."""
        text_obs_list = []
        info_list = []

        futures = []
        for worker in self.workers:
            futures.append(worker.reset.remote())

        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            text_obs_list.append(obs)
            info['won'] = False
            self.prev_admissible_commands[i] = info.get('actions_hint', '')
            info_list.append(info)

        return text_obs_list, None, info_list

    @property
    def get_admissible_commands(self):
        """Return cached admissible commands for all workers."""
        return self.prev_admissible_commands

    def close(self):
        """Terminate all Ray actors."""
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                pass


def build_scienceworld_envs(seed, env_num, group_n, resources_per_worker,
                            is_train=True, env_kwargs=None, val_splits=None):
    """Factory function for creating ScienceWorld environments."""
    return ScienceWorldEnvs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        val_splits=val_splits,
    )
