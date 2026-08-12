"""BroadcastBrain -- autonomous television ingestion for Fieldstation 42.

Drop recordings into ``Incoming/``, run ``bb run``, wake up to a segmented,
analyzed, deduplicated, categorized and exported library.

The public surface is deliberately small; everything else is reachable through
the CLI or the subpackages:

* :mod:`broadcastbrain.config` -- typed, layered settings
* :mod:`broadcastbrain.db` -- models, queue, repositories
* :mod:`broadcastbrain.pipeline` -- stages, orchestrator, workers, watcher
* :mod:`broadcastbrain.segment` -- multi-signal segmentation
* :mod:`broadcastbrain.analyze` -- the analyzer registry
* :mod:`broadcastbrain.search` -- hybrid keyword + semantic search
* :mod:`broadcastbrain.export` -- Fieldstation 42 export
"""

from __future__ import annotations

from .config import Settings, get_settings, load_settings
from .constants import APP_NAME

__version__ = "1.0.0"
__all__ = ["APP_NAME", "Settings", "__version__", "get_settings", "load_settings"]
