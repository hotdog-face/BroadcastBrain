"""AI analysis: the analyzer contract, the registry, and the built-in suite.

Importing this package registers every built-in analyzer. Adding a new one is
a new module plus one import line here -- the pipeline discovers it, notices
that no clip has a result for it, and backfills the library on the next run.
"""

from __future__ import annotations

# Importing these modules is what registers the built-in analyzers. Order here
# is irrelevant -- execution order comes from each analyzer's `order`.
from . import audio as _audio  # noqa: F401
from . import embed as _embed  # noqa: F401
from . import inference as _inference  # noqa: F401
from . import speech as _speech  # noqa: F401
from . import summarize as _summarize  # noqa: F401
from . import technical as _technical  # noqa: F401
from . import text as _text  # noqa: F401
from . import vision as _vision  # noqa: F401
from .base import AnalysisContext, Analyzer, AnalyzerResult, ClipView, EntityMention
from .registry import (
    all_analyzers,
    available_analyzers,
    describe_all,
    enabled_analyzers,
    get,
    register,
    unregister,
    version_map,
)
from .runner import AnalysisOutcome, AnalysisRunner

__all__ = [
    "AnalysisContext",
    "AnalysisOutcome",
    "AnalysisRunner",
    "Analyzer",
    "AnalyzerResult",
    "ClipView",
    "EntityMention",
    "all_analyzers",
    "available_analyzers",
    "describe_all",
    "enabled_analyzers",
    "get",
    "register",
    "unregister",
    "version_map",
]
