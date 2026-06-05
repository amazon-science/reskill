"""Registry API for pluggable ReSkill environments."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Iterable, Optional


BuildTrainValFn = Callable[[Any, int, Any], tuple[Any, Any]]
ProjectionFactory = Callable[[Any], Callable[..., Any]]
ManagerFactory = Callable[[], type]
ActionVocabularyProvider = Callable[[], str]


@dataclass(frozen=True)
class EnvironmentSpec:
    """Complete integration contract for one ReSkill environment."""

    key: str
    aliases: tuple[str, ...]
    build_train_val_envs: BuildTrainValFn
    projection_factory: ProjectionFactory
    base_manager_cls: ManagerFactory
    reskill_manager_cls: ManagerFactory
    action_vocabulary: ActionVocabularyProvider

    def matches(self, name: str) -> bool:
        """Return True when a legacy env name points at this spec."""
        normalized = name.lower()
        candidates = (self.key, *self.aliases)
        return any(candidate.lower() in normalized for candidate in candidates)


_BUILTIN_SPEC_MODULES = (
    "reskill.environments.alfworld",
    "reskill.environments.search",
    "reskill.environments.scienceworld",
)


def _load_spec(spec_path: str) -> EnvironmentSpec:
    """Load a spec from ``module`` or ``module:factory``."""
    module_path, sep, factory_name = spec_path.partition(":")
    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name or "get_environment_spec")
    spec = factory()
    if not isinstance(spec, EnvironmentSpec):
        raise TypeError(
            f"{spec_path} returned {type(spec)!r}, expected EnvironmentSpec")
    return spec


def registered_environment_specs() -> tuple[EnvironmentSpec, ...]:
    """Return specs for environments shipped with ReSkill."""
    return tuple(_load_spec(module) for module in _BUILTIN_SPEC_MODULES)


def registered_environment_keys() -> tuple[str, ...]:
    """Return sorted keys for built-in environments."""
    return tuple(sorted(spec.key for spec in registered_environment_specs()))


def _get_optional(env_config: Any, key: str) -> Optional[Any]:
    if hasattr(env_config, "get"):
        return env_config.get(key, None)
    return getattr(env_config, key, None)


def resolve_environment_spec(env_config: Any) -> EnvironmentSpec:
    """Resolve an environment spec from config.

    Resolution order:
      1. ``env.spec_module`` / ``env.registry_spec`` for out-of-tree plugins.
      2. Exact ``env.registry_key`` match against built-ins.
      3. Legacy substring match against ``env.env_name``.
    """
    external_spec = (
        _get_optional(env_config, "spec_module")
        or _get_optional(env_config, "registry_spec")
    )
    if external_spec:
        return _load_spec(str(external_spec))

    specs = registered_environment_specs()
    registry_key = _get_optional(env_config, "registry_key")
    if registry_key:
        normalized_key = str(registry_key).lower()
        for spec in specs:
            if normalized_key == spec.key.lower():
                return spec
        raise ValueError(
            f"[environment_registry] Unsupported env.registry_key: "
            f"{registry_key}. Registered keys: {', '.join(registered_environment_keys())}")

    env_name = str(_get_optional(env_config, "env_name") or "")
    for spec in specs:
        if spec.matches(env_name):
            return spec

    raise ValueError(
        f"[environment_registry] Unsupported environment: {env_name}. "
        f"Registered keys: {', '.join(registered_environment_keys())}. "
        "For external environments, set env.spec_module to "
        "'your_package.your_env:get_environment_spec'.")


def get_action_vocabulary(env_config: Any) -> str:
    """Return the env-owned action vocabulary used by skill authoring."""
    return resolve_environment_spec(env_config).action_vocabulary().strip()
