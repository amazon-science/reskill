from .projection import scienceworld_projection


def build_scienceworld_envs(*args, **kwargs):
    from .envs import build_scienceworld_envs as _build_scienceworld_envs
    return _build_scienceworld_envs(*args, **kwargs)


def _build_train_val_envs(config, group_n, resources_per_worker):
    env_kwargs = {
        'task_names': config.env.scienceworld.get('task_names', None),
        'env_step_limit': config.env.get('max_steps', 100),
        'simplification_str': config.env.scienceworld.get(
            'simplification_str', 'easy'),
        'max_variations': config.env.scienceworld.get('max_variations', None),
    }
    train_envs = build_scienceworld_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=True,
        env_kwargs=env_kwargs)

    val_splits = config.env.scienceworld.get('val_splits', None)
    if val_splits:
        val_splits = list(val_splits)
    val_task_names = config.env.scienceworld.get('val_task_names', None)
    if val_task_names:
        val_env_kwargs = dict(env_kwargs, task_names=list(val_task_names))
    else:
        val_env_kwargs = env_kwargs

    val_envs = build_scienceworld_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        resources_per_worker=resources_per_worker,
        is_train=False,
        env_kwargs=val_env_kwargs,
        val_splits=val_splits)
    return train_envs, val_envs


def get_environment_spec():
    """Return the registry spec for ScienceWorld."""
    from functools import partial

    from reskill.environments.registry import EnvironmentSpec

    def _base_manager_cls():
        from .env_manager import ScienceWorldEnvironmentManager
        return ScienceWorldEnvironmentManager

    def _reskill_manager_cls():
        from .skill_env_manager import ReSkillScienceWorldEnvManager
        return ReSkillScienceWorldEnvManager

    def _action_vocabulary():
        from .action_vocabulary import ACTION_VOCABULARY
        return ACTION_VOCABULARY

    return EnvironmentSpec(
        key="scienceworld",
        aliases=("science-world",),
        build_train_val_envs=_build_train_val_envs,
        projection_factory=lambda config: partial(
            scienceworld_projection, require_think=False),
        base_manager_cls=_base_manager_cls,
        reskill_manager_cls=_reskill_manager_cls,
        action_vocabulary=_action_vocabulary,
    )
