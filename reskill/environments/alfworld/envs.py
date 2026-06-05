# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from reskill.environments.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker,
                 is_train=True, env_kwargs={}, val_splits=None,
                 sequential_eval=False):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.group_n = group_n

        self.num_processes = env_num * group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []

        if sequential_eval and not is_train:
            import copy
            base_env = get_environment(env_type)(
                config, train_eval=eval_dataset)
            custom_gf = os.environ.get("CUSTOM_GAME_FILES", "")
            if custom_gf and os.path.exists(custom_gf):
                with open(custom_gf) as _f:
                    all_game_files = [l.strip() for l in _f if l.strip()]
                print(f"[AlfworldEnvs] Using {len(all_game_files)} custom game files from {custom_gf}")
            else:
                all_game_files = list(base_env.game_files)
            num_games = len(all_game_files)
            if self.num_processes != num_games:
                raise ValueError(
                    f"sequential_eval requires val_batch_size * group_n "
                    f"({self.num_processes}) == num_games ({num_games}) "
                    f"for split={eval_dataset}. Set data.val_batch_size={num_games}.")
            for i in range(num_games):
                worker_env = copy.copy(base_env)
                worker_env.game_files = [all_game_files[i]]
                worker_env.num_games = 1
                worker = env_worker.remote(config, seed + i, worker_env)
                self.workers.append(worker)
            print(f"[AlfworldEnvs] sequential_eval: {num_games} workers, "
                  f"1 unique game each, split={eval_dataset}")
        elif val_splits and not is_train:
            # Multi-split validation: distribute env_num*group_n workers evenly
            # across splits.
            n_splits = len(val_splits)
            assert self.num_processes % n_splits == 0, (
                f"Total workers ({self.num_processes}) must be divisible by "
                f"number of val_splits ({n_splits})")
            workers_per_split = self.num_processes // n_splits
            for split_idx, split_name in enumerate(val_splits):
                base_env = get_environment(env_type)(config, train_eval=split_name)
                offset = split_idx * workers_per_split
                for i in range(workers_per_split):
                    worker = env_worker.remote(
                        config, seed + offset + (i // self.group_n), base_env)
                    self.workers.append(worker)
            print(f"[AlfworldEnvs] Multi-split val: {val_splits}, "
                  f"{workers_per_split} workers each, {self.num_processes} total")
        else:
            # Single split (train or single eval)
            base_env = get_environment(env_type)(
                config, train_eval='train' if is_train else eval_dataset)
            for i in range(self.num_processes):
                worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
                self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker,
                        is_train=True, env_kwargs={}, val_splits=None,
                        sequential_eval=False):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker,
                        is_train, env_kwargs, val_splits=val_splits,
                        sequential_eval=sequential_eval)
