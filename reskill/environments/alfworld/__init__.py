from .projection import alfworld_projection


def build_alfworld_envs(*args, **kwargs):
    from .envs import build_alfworld_envs as _build_alfworld_envs
    return _build_alfworld_envs(*args, **kwargs)


def _build_train_val_envs(config, group_n, resources_per_worker):
    import os

    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        'configs/config_tw.yaml')
    env_kwargs = {
        'eval_dataset': config.env.alfworld.eval_dataset,
    }
    train_envs = build_alfworld_envs(
        alf_config_path,
        config.env.seed,
        config.data.train_batch_size,
        group_n,
        is_train=True,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker)

    sequential_eval = config.env.alfworld.get('sequential_eval', False)
    val_splits = config.env.alfworld.get('val_splits', None)
    if val_splits and not sequential_eval:
        val_splits = list(val_splits)
    else:
        val_splits = None

    val_envs = build_alfworld_envs(
        alf_config_path,
        config.env.seed + 1000,
        config.data.val_batch_size,
        1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
        val_splits=val_splits,
        sequential_eval=sequential_eval)
    return train_envs, val_envs


def get_environment_spec():
    """Return the registry spec for ALFWorld."""
    from functools import partial

    from reskill.environments.registry import EnvironmentSpec

    def _base_manager_cls():
        from .env_manager import AlfWorldEnvironmentManager
        return AlfWorldEnvironmentManager

    def _reskill_manager_cls():
        from .skill_env_manager import ReSkillAlfWorldEnvManager
        return ReSkillAlfWorldEnvManager

    def _action_vocabulary():
        from .action_vocabulary import ACTION_VOCABULARY
        return ACTION_VOCABULARY

    return EnvironmentSpec(
        key="alfworld",
        aliases=("alfred", "alfworld/alfredtwenv"),
        build_train_val_envs=_build_train_val_envs,
        projection_factory=lambda config: partial(
            alfworld_projection, require_think=False),
        base_manager_cls=_base_manager_cls,
        reskill_manager_cls=_reskill_manager_cls,
        action_vocabulary=_action_vocabulary,
    )
