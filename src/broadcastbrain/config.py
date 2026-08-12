"""Typed, layered configuration.

Precedence, lowest to highest:

1. Defaults from :mod:`broadcastbrain.constants`
2. A TOML file (``./broadcastbrain.toml``, or ``$BB_CONFIG``)
3. Environment variables (``BB_`` prefix, ``__`` for nesting)

So ``BB_WORKERS__CPU_WORKERS=4`` overrides ``[workers] cpu_workers`` in the
file, which overrides the built-in default. Nothing in the codebase reads a
path or a tunable from anywhere else.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from . import constants as C  # noqa: N812 - `C.` keeps the default table readable
from .errors import ConfigError

if sys.version_info >= (3, 11):  # pragma: no cover - version branch
    import tomllib
else:  # pragma: no cover - version branch
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


#: Directory name for each derived path, relative to ``paths.root``.
#: This is the canonical on-disk layout of a BroadcastBrain data root.
DEFAULT_DIRECTORY_NAMES: dict[str, str] = {
    "incoming": "Incoming",
    "processing": "Processing",
    "library": "Library",
    "duplicates": "Duplicates",
    "database": "Database",
    "logs": "Logs",
    "rejected": "Rejected",
    "thumbnails": "Thumbnails",
    "transcripts": "Transcripts",
    "metadata": "Metadata",
    "export": "Fieldstation Export",
    "downloads": "Downloads",
    "cache": "Cache",
}


class PathsSettings(BaseModel):
    """The on-disk layout. Only ``root`` is normally set; the rest derive."""

    root: Path = Field(default=Path("./BroadcastBrainData"))
    incoming: Optional[Path] = None
    processing: Optional[Path] = None
    library: Optional[Path] = None
    duplicates: Optional[Path] = None
    database: Optional[Path] = None
    logs: Optional[Path] = None
    rejected: Optional[Path] = None
    thumbnails: Optional[Path] = None
    transcripts: Optional[Path] = None
    metadata: Optional[Path] = None
    export: Optional[Path] = None
    downloads: Optional[Path] = None
    cache: Optional[Path] = None

    @field_validator("*", mode="before")
    @classmethod
    def _expand(cls, value: Any) -> Any:
        if isinstance(value, (str, Path)):
            return Path(os.path.expandvars(str(value))).expanduser()
        return value

    @model_validator(mode="after")
    def _derive(self) -> PathsSettings:
        self.root = self.root.expanduser()
        for attr, name in DEFAULT_DIRECTORY_NAMES.items():
            if getattr(self, attr) is None:
                setattr(self, attr, self.root / name)
        return self

    def _dirs(self) -> list[Path]:
        return [self.root] + [
            p for p in (getattr(self, a) for a in DEFAULT_DIRECTORY_NAMES) if p is not None
        ]

    def ensure(self) -> None:
        """Create every managed directory. Idempotent and safe to call often."""
        for directory in self._dirs():
            directory.mkdir(parents=True, exist_ok=True)

    def require(self, name: str) -> Path:
        """Return a derived path, raising a clear error if it is unset."""
        value = getattr(self, name, None)
        if value is None:
            raise ConfigError(f"paths.{name} is not configured")
        return Path(value)


class DatabaseSettings(BaseModel):
    """SQLite today, PostgreSQL later.

    Leave ``url`` unset to get ``sqlite:///<paths.database>/broadcastbrain.db``.
    Set it to a ``postgresql+psycopg://`` URL and nothing else has to change:
    the ORM layer is dialect-neutral and the only SQLite-specific code lives
    behind the search backend protocol.
    """

    url: Optional[str] = None
    filename: str = "broadcastbrain.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    #: SQLite only. WAL keeps readers (dashboard) from blocking writers.
    sqlite_wal: bool = True
    sqlite_busy_timeout_ms: int = 30_000
    sqlite_synchronous: str = "NORMAL"
    sqlite_cache_size_kb: int = 262_144

    def resolve_url(self, database_dir: Path) -> str:
        if self.url:
            return self.url
        return f"sqlite:///{(database_dir / self.filename).as_posix()}"


class SegmentationSettings(BaseModel):
    """Multi-signal segmentation tunables (see ``constants`` for rationale)."""

    enabled: bool = True
    black_min_duration_s: float = C.BLACK_MIN_DURATION_S
    black_pixel_threshold: float = C.BLACK_PIXEL_THRESHOLD
    black_picture_ratio: float = C.BLACK_PICTURE_RATIO
    silence_noise_db: float = C.SILENCE_NOISE_DB
    silence_min_duration_s: float = C.SILENCE_MIN_DURATION_S
    scene_threshold: float = C.SCENE_THRESHOLD
    freeze_noise_db: float = C.FREEZE_NOISE_DB
    freeze_min_duration_s: float = C.FREEZE_MIN_DURATION_S
    analysis_width: int = C.SIGNAL_ANALYSIS_WIDTH
    analysis_fps: float = C.SIGNAL_ANALYSIS_FPS
    cluster_window_s: float = C.BOUNDARY_CLUSTER_WINDOW_S
    min_boundary_score: float = C.BOUNDARY_MIN_SCORE
    min_segment_s: float = C.MIN_SEGMENT_SECONDS
    max_segment_s: float = C.MAX_SEGMENT_SECONDS
    ad_grid_bonus: float = C.AD_GRID_BONUS
    signal_weights: dict[str, float] = Field(default_factory=lambda: dict(C.SIGNAL_WEIGHTS))

    # -- optional signals (all collected in the same single decode pass) -----
    #: EBU R128 loudness. Ad breaks in this era were mixed noticeably hotter
    #: than programming, so a loudness jump is real evidence of a boundary.
    loudness_enabled: bool = True
    loudness_jump_lu: float = 3.0
    #: cropdetect: pillarbox/letterbox changes mark 4:3 -> 16:9 transitions.
    crop_enabled: bool = True
    #: Watch a screen corner for the network logo appearing/disappearing --
    #: on most cable networks the bug drops during commercials.
    logo_enabled: bool = True
    logo_corner: str = "top_right"  # key of constants.LOGO_REGIONS
    logo_scene_threshold: float = 18.0

    #: Re-encode cut clips instead of stream-copying. Slower, frame-accurate.
    reencode: bool = False
    #: Video encoder used when ``reencode`` is on or stream copy is impossible.
    video_encoder: str = "libx264"
    audio_encoder: str = "aac"
    encoder_crf: int = 20
    encoder_preset: str = "veryfast"


class AnalysisSettings(BaseModel):
    """Which analyzers run, and how hard they work."""

    enabled: bool = True
    #: Empty means "every registered analyzer". Names come from the registry.
    enable_analyzers: list[str] = Field(default_factory=list)
    disable_analyzers: list[str] = Field(default_factory=list)
    frames_per_clip: int = C.VISION_FRAMES_PER_CLIP
    thumbnail_width: int = C.THUMBNAIL_WIDTH
    whisper_model: str = C.WHISPER_MODEL
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute_type: str = "auto"
    whisper_language: Optional[str] = None
    embedding_model: str = C.EMBEDDING_MODEL
    embedding_dim: int = C.EMBEDDING_DIM
    object_model: str = "yolov8n.pt"
    face_model: str = "buffalo_l"
    #: Directory of labelled reference portraits for celebrity recognition,
    #: laid out as ``<gallery>/<Person Name>/*.jpg``. Face *detection* works
    #: without it; naming a face requires a reference to compare against.
    faces_gallery: Optional[Path] = None
    face_match_threshold: float = 0.42
    min_entity_confidence: float = C.MIN_ENTITY_CONFIDENCE
    #: Skip transcription for clips shorter than this (bumpers, IDs).
    min_transcribe_seconds: float = 2.0
    #: Hard ceiling on per-clip analysis wall time before the job is retried.
    per_clip_timeout_s: int = 1800
    #: Keep extracted frames/audio for re-analysis instead of deleting them.
    keep_intermediates: bool = False


class DedupeSettings(BaseModel):
    enabled: bool = True
    phash_frames: int = C.PHASH_FRAME_COUNT
    max_normalized_distance: float = C.PHASH_MAX_NORMALIZED_DISTANCE
    duration_tolerance_s: float = C.DUP_DURATION_TOLERANCE_S
    chromaprint_min_similarity: float = C.CHROMAPRINT_MIN_SIMILARITY
    quality_weights: dict[str, float] = Field(default_factory=lambda: dict(C.QUALITY_WEIGHTS))
    #: Move alternates into Duplicates/ (hardlinked, so no space is wasted).
    move_alternates: bool = True


class WorkerSettings(BaseModel):
    """Concurrency. Defaults are deliberately conservative for overnight runs."""

    cpu_workers: int = 0  # 0 => auto (cpu_count - 1, min 1)
    gpu_workers: int = 1
    io_workers: int = 2
    max_attempts: int = C.DEFAULT_MAX_ATTEMPTS
    lease_seconds: int = C.JOB_LEASE_SECONDS
    #: Cap resident memory by processing fewer frames and streaming audio.
    low_memory: bool = False
    #: Prefer hardware decode/encode when ffmpeg reports support.
    use_gpu: bool = True
    hwaccel: str = "auto"  # auto | none | cuda | videotoolbox | qsv | vaapi
    #: Stop the run when free space on the library volume drops below this.
    min_free_gb: float = 25.0
    #: Nice value applied to ffmpeg children so the box stays usable.
    subprocess_nice: int = 5

    def resolved_cpu_workers(self) -> int:
        if self.cpu_workers > 0:
            return self.cpu_workers
        return max(1, (os.cpu_count() or 2) - 1)


class ExportSettings(BaseModel):
    enabled: bool = True
    break_interval_s: float = C.BREAK_INTERVAL_SECONDS
    pod_target_s: float = C.POD_TARGET_SECONDS
    pod_min_spots: int = C.POD_SIZE_RANGE[0]
    pod_max_spots: int = C.POD_SIZE_RANGE[1]
    return_bumper_probability: float = C.POD_RETURN_BUMPER_PROBABILITY
    #: Deterministic scheduling: same library in, same schedule out.
    random_seed: int = 42
    #: Copy media into the export dir instead of referencing library paths.
    copy_media: bool = False
    #: Only export clips whose category confidence clears this bar.
    min_confidence: float = 0.0


class SearchSettings(BaseModel):
    enabled: bool = True
    default_limit: int = C.SEARCH_DEFAULT_LIMIT
    rrf_k: float = C.RRF_K
    similarity_min_score: float = C.SIMILARITY_MIN_SCORE
    #: Rebuild the FTS index from scratch on next start.
    rebuild_index: bool = False


class DashboardSettings(BaseModel):
    enabled: bool = True
    host: str = C.DASHBOARD_HOST
    port: int = C.DASHBOARD_PORT
    refresh_ms: int = C.DASHBOARD_REFRESH_MS
    #: Serve clip previews. Localhost-only by default.
    allow_preview: bool = True


class ArchiveSettings(BaseModel):
    """Optional Internet Archive (and future source) downloader."""

    enabled: bool = False
    queries: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)
    max_items_per_query: int = 25
    max_total_gb: float = 500.0
    min_item_mb: float = 20.0
    #: Only these formats are pulled from an item's file list.
    accept_formats: list[str] = Field(
        default_factory=lambda: [
            "MPEG4", "h.264", "512Kb MPEG4", "MPEG2", "Ogg Video", "Matroska",
        ]
    )
    date_range: Optional[str] = None  # e.g. "[2002-01-01 TO 2006-12-31]"
    concurrency: int = 2
    #: Auto-enqueue downloads into the pipeline when they finish.
    auto_queue: bool = True


class LoggingSettings(BaseModel):
    level: str = "INFO"
    file_level: str = "DEBUG"
    json_logs: bool = False
    max_bytes: int = C.LOG_MAX_BYTES
    backup_count: int = C.LOG_BACKUP_COUNT
    #: Log every external command line. Verbose but priceless when debugging.
    log_subprocess_commands: bool = True


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class _TomlSettingsSource(PydanticBaseSettingsSource):
    """Lowest-priority source that reads the TOML config file, if present."""

    def __init__(self, settings_cls: type[BaseSettings], path: Optional[Path]) -> None:
        super().__init__(settings_cls)
        self._path = path
        self._data: dict[str, Any] = {}
        if path and path.is_file():
            try:
                with path.open("rb") as handle:
                    self._data = tomllib.load(handle)
            except Exception as exc:
                raise ConfigError(f"could not parse config file {path}: {exc}") from exc

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class Settings(BaseSettings):
    """The complete, validated configuration for one BroadcastBrain instance."""

    model_config = SettingsConfigDict(
        env_prefix=C.ENV_PREFIX,
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )

    #: Human label; shows up in the dashboard and in export metadata.
    instance_name: str = "broadcastbrain"
    #: Path to the config file this instance was loaded from (informational).
    config_file: Optional[Path] = None
    #: Target broadcast era, inclusive. Biases year estimation and reporting.
    era_start_year: int = C.TARGET_ERA[0]
    era_end_year: int = C.TARGET_ERA[1]
    #: External binaries. Absolute paths allowed; bare names are looked up.
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    fpcalc_binary: str = "fpcalc"
    #: Delete the copy in Processing/ once a clip is safely in Library/.
    cleanup_processing: bool = True
    #: Move finished source files out of Incoming/ into Processing/completed.
    move_completed_sources: bool = True
    #: Global dry run: plan everything, write nothing.
    dry_run: bool = False

    paths: PathsSettings = Field(default_factory=PathsSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    dedupe: DedupeSettings = Field(default_factory=DedupeSettings)
    workers: WorkerSettings = Field(default_factory=WorkerSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    archive: ArchiveSettings = Field(default_factory=ArchiveSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # -- validation ---------------------------------------------------------

    @model_validator(mode="after")
    def _check_era(self) -> Settings:
        if self.era_start_year > self.era_end_year:
            raise ConfigError(
                f"era_start_year ({self.era_start_year}) is after "
                f"era_end_year ({self.era_end_year})"
            )
        return self

    # -- source ordering ----------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_path = _pending_config_path.get()
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _TomlSettingsSource(settings_cls, toml_path),
            file_secret_settings,
        )

    # -- helpers ------------------------------------------------------------

    @property
    def database_url(self) -> str:
        return self.database.resolve_url(self.paths.require("database"))

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def era_range(self) -> tuple[int, int]:
        return (self.era_start_year, self.era_end_year)

    def describe(self) -> dict[str, Any]:
        """Redaction-free summary for logs and the dashboard's About panel."""
        return {
            "instance_name": self.instance_name,
            "config_file": str(self.config_file) if self.config_file else None,
            "root": str(self.paths.root),
            "database_url": self.database_url,
            "era": list(self.era_range()),
            "cpu_workers": self.workers.resolved_cpu_workers(),
            "gpu_workers": self.workers.gpu_workers,
            "dry_run": self.dry_run,
        }


class _ContextVarLike:
    """Tiny holder so `settings_customise_sources` can see the chosen path.

    pydantic-settings gives the source factory no access to constructor
    arguments, so the path is stashed here for the duration of the call.
    """

    def __init__(self) -> None:
        self._value: Optional[Path] = None

    def set(self, value: Optional[Path]) -> None:
        self._value = value

    def get(self) -> Optional[Path]:
        return self._value


_pending_config_path = _ContextVarLike()


def find_config_file(explicit: Optional[Path] = None) -> Optional[Path]:
    """Locate the config file: explicit arg, then ``$BB_CONFIG``, then cwd."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path
    env_value = os.environ.get(C.CONFIG_ENV_VAR)
    if env_value:
        path = Path(env_value).expanduser()
        if not path.is_file():
            raise ConfigError(f"{C.CONFIG_ENV_VAR} points at a missing file: {path}")
        return path
    candidate = Path.cwd() / C.DEFAULT_CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def load_settings(config_path: Optional[Path] = None, **overrides: Any) -> Settings:
    """Build a :class:`Settings` from file + environment + explicit overrides."""
    path = find_config_file(config_path)
    _pending_config_path.set(path)
    try:
        settings = Settings(**overrides)
    finally:
        _pending_config_path.set(None)
    if path is not None:
        object.__setattr__(settings, "config_file", path)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (used by the dashboard and workers)."""
    return load_settings()


def reset_settings_cache() -> None:
    """Drop the singleton. Tests and ``bb config reload`` use this."""
    get_settings.cache_clear()
