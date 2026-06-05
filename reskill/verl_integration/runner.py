from pprint import pprint

import ray
from omegaconf import OmegaConf


def _validate_supported_config(config) -> None:
    adv_estimator = config.algorithm.get("adv_estimator")
    rollout_name = config.actor_rollout_ref.rollout.get("name")
    actor_strategy = config.actor_rollout_ref.actor.get("strategy")

    if adv_estimator != "grpo":
        raise ValueError(f"ReSkill currently supports algorithm.adv_estimator=grpo, got {adv_estimator!r}")
    if rollout_name != "vllm":
        raise ValueError(f"ReSkill currently supports actor_rollout_ref.rollout.name=vllm, got {rollout_name!r}")
    if actor_strategy != "fsdp":
        raise ValueError(f"ReSkill currently supports actor_rollout_ref.actor.strategy=fsdp, got {actor_strategy!r}")
    if config.get("critic", {}).get("strategy", "fsdp") != "fsdp":
        raise ValueError("ReSkill currently supports critic.strategy=fsdp")


def run_reskill_ppo(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(include_dashboard=False, **OmegaConf.to_container(ray_init_kwargs))

    runner_cls = ray.remote(num_cpus=1)(_ReSkillTaskRunner)
    runner = runner_cls.remote()
    ray.get(runner.run.remote(config))


class _ReSkillTaskRunner:
    def run(self, config):
        from reskill.environments.make_envs import make_envs
        from reskill.reskill_trainer import ReSkillTrainer
        from reskill.verl_integration.reward_manager import EpisodeRewardManager
        from reskill.verl_integration.rollout import TrajectoryCollector
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        _validate_supported_config(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        envs, val_envs = make_envs(config)

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        if config.actor_rollout_ref.rollout.get("mode", "sync") == "async":
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            from verl.workers.fsdp_workers import RewardModelWorker

            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name != "episode":
            raise ValueError(f"ReSkill currently supports reward_model.reward_manager=episode, got {reward_manager_name!r}")

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.actor_rollout_ref.rollout.n != 1:
            raise ValueError("Use env.rollout.n for ReSkill/GRPO grouping; actor_rollout_ref.rollout.n must stay 1")

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = ReSkillTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.get("device", "cuda"),
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()
