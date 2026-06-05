"""Environment factory for registry-backed ReSkill experiments."""

from omegaconf import OmegaConf

from reskill.environments.registry import resolve_environment_spec


def make_envs(config):
    """Create environments for ReSkill or pure GRPO baselines."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(
        config.env.resources_per_worker, resolve=True)

    spec = resolve_environment_spec(config.env)
    train_envs, val_raw_envs = spec.build_train_val_envs(
        config, group_n, resources_per_worker)
    projection_f = spec.projection_factory(config)

    skills_enabled = config.env.get('trigger_skills', {}).get('enable', False)
    manager_cls = (
        spec.reskill_manager_cls() if skills_enabled else spec.base_manager_cls()
    )
    envs = manager_cls(train_envs, projection_f, config)
    val_envs = manager_cls(val_raw_envs, projection_f, config)

    mode = "ReSkill" if skills_enabled else "base"
    print(f"[make_envs] Using {manager_cls.__name__} ({mode}, key={spec.key})")

    return envs, val_envs
