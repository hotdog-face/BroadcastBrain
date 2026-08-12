"""Analyzer registry.

Analyzers register themselves at import time. Adding one means writing a class
and adding an import to :mod:`broadcastbrain.analyze` -- no wiring, no
configuration change, no schema change. The pipeline notices the new analyzer,
sees that no clip has a result for it, and backfills the library on the next
run.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, TypeVar

from ..config import AnalysisSettings
from .base import Analyzer

log = logging.getLogger(__name__)

_REGISTRY: dict[str, Analyzer] = {}

A = TypeVar("A", bound=Analyzer)


def register(analyzer_cls: type[A]) -> type[A]:
    """Class decorator that instantiates and registers an analyzer."""
    instance = analyzer_cls()
    if not instance.name or instance.name == "analyzer":
        raise ValueError(f"{analyzer_cls.__name__} must define a unique `name`")
    if instance.name in _REGISTRY:
        raise ValueError(f"duplicate analyzer name: {instance.name}")
    _REGISTRY[instance.name] = instance
    return analyzer_cls


def unregister(name: str) -> None:
    """Remove an analyzer (tests, and disabling a misbehaving model at runtime)."""
    _REGISTRY.pop(name, None)


def get(name: str) -> Optional[Analyzer]:
    return _REGISTRY.get(name)


def all_analyzers() -> list[Analyzer]:
    """Every registered analyzer, in execution order."""
    return sorted(_REGISTRY.values(), key=lambda a: (a.order, a.name))


def enabled_analyzers(settings: AnalysisSettings) -> list[Analyzer]:
    """Apply the config's enable/disable lists.

    ``enable_analyzers`` empty means "everything registered", which is the
    behaviour that makes newly added analyzers automatically backfill.
    """
    selected = all_analyzers()
    if settings.enable_analyzers:
        wanted = {name.strip() for name in settings.enable_analyzers}
        unknown = wanted - {a.name for a in selected}
        if unknown:
            log.warning("config enables unknown analyzers: %s", ", ".join(sorted(unknown)))
        selected = [a for a in selected if a.name in wanted]
    if settings.disable_analyzers:
        blocked = {name.strip() for name in settings.disable_analyzers}
        selected = [a for a in selected if a.name not in blocked]
    return selected


def available_analyzers(settings: AnalysisSettings) -> list[Analyzer]:
    """Enabled analyzers whose backends are actually installed right now."""
    return [analyzer for analyzer in enabled_analyzers(settings) if analyzer.available()]


def describe_all() -> list[dict[str, object]]:
    return [analyzer.describe() for analyzer in all_analyzers()]


def version_map() -> dict[str, str]:
    """``{analyzer_name: version}`` -- the cache key for stored results."""
    return {analyzer.name: analyzer.version for analyzer in _REGISTRY.values()}


def filter_by(predicate: Callable[[Analyzer], bool]) -> list[Analyzer]:
    return [analyzer for analyzer in all_analyzers() if predicate(analyzer)]


def names(analyzers: Optional[Iterable[Analyzer]] = None) -> list[str]:
    return [a.name for a in (analyzers if analyzers is not None else all_analyzers())]
