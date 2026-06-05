from .projection import search_projection


def build_search_envs(*args, **kwargs):
    from .envs import build_search_envs as _build_search_envs
    return _build_search_envs(*args, **kwargs)


def _build_train_val_envs(config, group_n, resources_per_worker):
    del resources_per_worker
    train_envs = build_search_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_config=config.env)
    val_envs = build_search_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        env_config=config.env)
    return train_envs, val_envs


def get_environment_spec():
    """Return the registry spec for the search environment."""
    from functools import partial

    from reskill.environments.registry import EnvironmentSpec

    def _base_manager_cls():
        from .env_manager import SearchEnvironmentManager
        return SearchEnvironmentManager

    def _reskill_manager_cls():
        from .skill_env_manager import ReSkillSearchEnvManager
        return ReSkillSearchEnvManager

    def _action_vocabulary():
        from .action_vocabulary import ACTION_VOCABULARY
        return ACTION_VOCABULARY

    return EnvironmentSpec(
        key="search",
        aliases=("search-r1", "qa"),
        build_train_val_envs=_build_train_val_envs,
        projection_factory=lambda config: partial(search_projection),
        base_manager_cls=_base_manager_cls,
        reskill_manager_cls=_reskill_manager_cls,
        action_vocabulary=_action_vocabulary,
    )
