"""Every tunable default in one place.

Nothing in this project hardcodes a number inline. Modules import a named
constant from here, and :mod:`broadcastbrain.config` exposes the ones an
operator is likely to want to change as configuration keys. Changing a
threshold should mean editing one line here or one line of TOML -- never
hunting through the pipeline.
"""

from __future__ import annotations

from typing import Final

# --- Identity ----------------------------------------------------------------
APP_NAME: Final = "BroadcastBrain"
APP_SLUG: Final = "broadcastbrain"
ENV_PREFIX: Final = "BB_"
CONFIG_ENV_VAR: Final = "BB_CONFIG"
DEFAULT_CONFIG_FILENAME: Final = "broadcastbrain.toml"

# --- Container / codec support -----------------------------------------------
#: Extensions accepted from Incoming. Anything else is moved to Rejected with a
#: reason -- never deleted.
VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".mp4", ".m4v", ".mkv", ".avi", ".mpeg", ".mpg", ".m2p", ".m2ts", ".mts",
        ".ts", ".tp", ".trp", ".vob", ".iso", ".img", ".wmv", ".mov", ".flv",
        ".divx", ".rm", ".rmvb", ".ogv", ".webm", ".dv", ".3gp", ".asf", ".nuv",
    }
)
#: Disc-image containers that must be expanded to titles before segmentation.
DISC_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".iso", ".img"})
#: Directory-style DVD rips are detected by these marker directories.
DVD_MARKER_DIRS: Final[tuple[str, ...]] = ("VIDEO_TS", "video_ts", "BDMV")

# --- Discovery ---------------------------------------------------------------
#: A file must stop changing for this long before we consider it fully written.
FILE_STABILITY_SECONDS: Final = 20.0
#: How often the watcher rescans Incoming (seconds).
WATCH_POLL_SECONDS: Final = 30.0
#: Files below this size are almost certainly truncated downloads.
MIN_SOURCE_BYTES: Final = 128 * 1024
#: Only hash the head/tail of huge files for the fast pre-check.
QUICK_HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024
HASH_READ_CHUNK_BYTES: Final = 4 * 1024 * 1024

# --- Segmentation signal thresholds ------------------------------------------
#: ffmpeg blackdetect: minimum black duration (s) and pixel/picture thresholds.
BLACK_MIN_DURATION_S: Final = 0.08
BLACK_PIXEL_THRESHOLD: Final = 0.10
BLACK_PICTURE_RATIO: Final = 0.98
#: ffmpeg silencedetect.
SILENCE_NOISE_DB: Final = -38.0
SILENCE_MIN_DURATION_S: Final = 0.25
#: ffmpeg scdet scene-change score threshold (0-100).
SCENE_THRESHOLD: Final = 12.0
#: ffmpeg freezedetect (still frames -- slates, TV Guide pages, test cards).
FREEZE_NOISE_DB: Final = -60.0
FREEZE_MIN_DURATION_S: Final = 2.0
#: Detectors run on a downscaled copy; this is the analysis width in pixels.
SIGNAL_ANALYSIS_WIDTH: Final = 320
#: Analyze at this framerate for detector passes (full rate is wasteful).
SIGNAL_ANALYSIS_FPS: Final = 8.0

# --- Boundary fusion ---------------------------------------------------------
#: Weight each signal contributes to a boundary's score.
SIGNAL_WEIGHTS: Final[dict[str, float]] = {
    "black": 1.00,
    "silence": 0.65,
    "scene": 0.45,
    "freeze": 0.35,
    "audio_change": 0.40,
    "logo_change": 0.85,
    "aspect_change": 0.70,
    "loudness_jump": 0.50,
    "credits": 0.60,
    "fade": 0.55,
}
#: Candidate signals within this many seconds collapse into one boundary.
BOUNDARY_CLUSTER_WINDOW_S: Final = 1.2
#: A fused boundary needs at least this score to cut.
BOUNDARY_MIN_SCORE: Final = 1.05
#: A boundary backed by black + silence together is always trusted.
BOUNDARY_STRONG_SCORE: Final = 1.60
#: A detector firing more often than this many times per minute is not marking
#: programme boundaries -- it is reacting to the content itself. Animation is
#: the case that forces this: hard cel cuts fire `scdet` ~20x/minute and the
#: corner-logo watcher ~14x/minute, which together swamp the black and silence
#: that mark the real transitions. Above this rate a signal's weight is scaled
#: down in proportion to how far past it went. Measured against a real
#: two-hour anime broadcast: 312 over-cut fragments become 66 clips with the
#: programme/commercial structure intact.
SIGNAL_NOISE_RATE_PER_MINUTE: Final = 3.0
#: A noisy detector is never silenced completely -- it can still break a tie.
SIGNAL_NOISE_WEIGHT_FLOOR: Final = 0.15
#: Only content-reactive detectors are adapted. Black and silence are physical
#: events in the broadcast, not reactions to what is on screen, so a high rate
#: from them is real information and is left alone.
ADAPTIVE_SIGNAL_KINDS: Final[tuple[str, ...]] = (
    "scene", "logo_change", "aspect_change", "freeze",
)

#: Segments shorter than this are merged into a neighbour.
MIN_SEGMENT_SECONDS: Final = 3.0
#: Segments longer than this are re-split with relaxed thresholds.
MAX_SEGMENT_SECONDS: Final = 5400.0
#: 2000s US cable ran on a hard 15/30/60s ad grid; boundaries landing near a
#: grid multiple get a small bonus. This is what makes pod detection work.
AD_GRID_SECONDS: Final[tuple[float, ...]] = (15.0, 30.0, 60.0)
AD_GRID_TOLERANCE_S: Final = 0.9
AD_GRID_BONUS: Final = 0.25

# --- Duration priors (seconds) for first-pass categorization -----------------
#: (low, high) duration windows used as evidence -- never as the sole decision.
#:
#: Only formats with a *characteristic* length appear here. TV Guide pages,
#: weather segments, news breaks and PPV previews were deliberately removed:
#: they run for however long the station needs, so a prior for them spans
#: nearly every duration and quietly competes with every other category. A
#: 20-900s "tv_guide" prior was outvoting real programme segments and pushing
#: them into Unknown. Those formats are identified by their on-screen and
#: spoken markers in `analyze/knowledge.py`, which is genuinely discriminative.
#:
#: What remains is a deliberate limit rather than a gap: a four-minute
#: programme act and a music video are the same length, and no duration rule
#: can separate them. That needs speech or music identification, and until one
#: is installed the honest answer is Unknown.
DURATION_PRIORS: Final[dict[str, tuple[float, float]]] = {
    "station_id": (1.0, 6.0),
    "network_bumper": (2.0, 12.0),
    "commercial": (9.0, 65.0),
    "promo": (8.0, 45.0),
    "music_video": (110.0, 420.0),
    "infomercial": (1500.0, 2100.0),
    "tv_episode": (1080.0, 3900.0),
    "movie": (3900.0, 14400.0),
}
#: A continuous run at least this long, with no ad markers, is programme
#: content rather than an interstitial. Broadcast is cut into acts by its own
#: commercial breaks, so a captured "episode" arrives as several 5-20 minute
#: segments -- none of which resemble a whole episode by duration alone.
PROGRAMME_SEGMENT_MIN_S: Final = 240.0
#: Above this a clip is more likely a whole episode or a film, which the
#: existing priors already cover.
PROGRAMME_SEGMENT_MAX_S: Final = 3900.0

#: A clip this short that carries network branding is that network's own
#: interstitial, not an advertisement. Nothing else on television is four
#: seconds long and says the channel's name.
STATION_ID_MAX_S: Final = 6.5
NETWORK_BUMPER_MAX_S: Final = 14.0

#: Canonical US ad-break slot lengths; a clip within tolerance of one of these
#: is very likely a commercial.
COMMERCIAL_SLOT_SECONDS: Final[tuple[float, ...]] = (10.0, 15.0, 20.0, 30.0, 45.0, 60.0)
COMMERCIAL_SLOT_TOLERANCE_S: Final = 1.5

# --- Analysis ----------------------------------------------------------------
#: Frames sampled per clip for vision analyzers (OCR, logo, faces, objects).
VISION_FRAMES_PER_CLIP: Final = 12
#: Never sample more densely than this (seconds between frames).
VISION_MIN_FRAME_SPACING_S: Final = 0.75
#: Thumbnail is taken at this fraction into the clip (avoids fade-in black).
THUMBNAIL_POSITION_RATIO: Final = 0.35
THUMBNAIL_WIDTH: Final = 640
THUMBNAIL_QUALITY: Final = 3  # ffmpeg -q:v, lower is better
#: Audio extracted for speech/fingerprinting.
ANALYSIS_AUDIO_SAMPLE_RATE: Final = 16000
ANALYSIS_AUDIO_CHANNELS: Final = 1
#: Corner boxes searched for a network logo, as (x, y, w, h) fractions.
LOGO_REGIONS: Final[dict[str, tuple[float, float, float, float]]] = {
    "top_left": (0.02, 0.03, 0.22, 0.16),
    "top_right": (0.76, 0.03, 0.22, 0.16),
    "bottom_left": (0.02, 0.81, 0.22, 0.16),
    "bottom_right": (0.76, 0.81, 0.22, 0.16),
}
#: Minimum confidence for an analyzer assertion to become a stored entity.
MIN_ENTITY_CONFIDENCE: Final = 0.35
#: Whisper defaults.
WHISPER_MODEL: Final = "base.en"
WHISPER_BEAM_SIZE: Final = 5
#: Text embedding model + dimension.
EMBEDDING_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: Final = 384

# --- Era estimation ----------------------------------------------------------
#: The target era. Used to bias year estimates and to flag out-of-era material.
TARGET_ERA: Final[tuple[int, int]] = (2002, 2006)
#: Clips estimated outside this wider window are flagged, never deleted.
PLAUSIBLE_YEAR_RANGE: Final[tuple[int, int]] = (1940, 2030)

# --- Deduplication -----------------------------------------------------------
#: Number of evenly spaced frames hashed for the perceptual video signature.
PHASH_FRAME_COUNT: Final = 32
PHASH_BITS: Final = 64
#: Two clips are near-duplicates when their normalized Hamming distance over
#: the aligned hash sequence is at or below this.
PHASH_MAX_NORMALIZED_DISTANCE: Final = 0.14
#: Candidate blocking: only compare clips whose durations differ by less.
DUP_DURATION_TOLERANCE_S: Final = 2.5
#: Chromaprint match threshold (0-1).
CHROMAPRINT_MIN_SIMILARITY: Final = 0.90
#: Quality scoring weights when picking which duplicate to keep.
QUALITY_WEIGHTS: Final[dict[str, float]] = {
    "pixels": 0.32,
    "bitrate": 0.28,
    "fps": 0.10,
    "audio_bitrate": 0.10,
    "duration_completeness": 0.14,
    "source_trust": 0.06,
}

# --- Queue / workers ---------------------------------------------------------
DEFAULT_MAX_ATTEMPTS: Final = 3
JOB_LEASE_SECONDS: Final = 3600
JOB_CLAIM_BACKOFF_S: Final = 0.4
RETRY_BASE_DELAY_S: Final = 5.0
RETRY_MAX_DELAY_S: Final = 900.0
RETRY_JITTER_RATIO: Final = 0.25
#: Workers poll this often when the queue is empty.
QUEUE_IDLE_POLL_S: Final = 2.0
#: Priorities -- higher runs first. Keeps the library usable while huge sources
#: are still being chewed on.
JOB_PRIORITY: Final[dict[str, int]] = {
    "ingest_source": 50,
    "segment_source": 40,
    "analyze_clip": 30,
    "fingerprint_clip": 35,
    "dedupe_clip": 25,
    "categorize_clip": 20,
    "materialize_clip": 15,
    "export_library": 5,
    "download_item": 60,
}

# --- Subprocess --------------------------------------------------------------
FFPROBE_TIMEOUT_S: Final = 300
FFMPEG_SEGMENT_TIMEOUT_S: Final = 7200
FFMPEG_SIGNAL_TIMEOUT_S: Final = 21600
SUBPROCESS_KILL_GRACE_S: Final = 10.0

# --- Export ------------------------------------------------------------------
EXPORT_SCHEMA_VERSION: Final = 1
#: Fieldstation scheduling: how often to break for commercials inside a show.
BREAK_INTERVAL_SECONDS: Final = 480.0
#: How many spots make up one commercial pod.
POD_SIZE_RANGE: Final[tuple[int, int]] = (2, 4)
#: Target pod duration; pods are filled until they reach this.
POD_TARGET_SECONDS: Final = 120.0
#: A bumper is played coming back from break this fraction of the time.
POD_RETURN_BUMPER_PROBABILITY: Final = 0.7

# --- Search ------------------------------------------------------------------
SEARCH_DEFAULT_LIMIT: Final = 50
SEARCH_MAX_LIMIT: Final = 1000
#: Reciprocal-rank-fusion constant for blending keyword + vector results.
RRF_K: Final = 60.0
#: Minimum cosine similarity for "clips similar to this one".
SIMILARITY_MIN_SCORE: Final = 0.55

# --- Dashboard ---------------------------------------------------------------
DASHBOARD_HOST: Final = "127.0.0.1"
DASHBOARD_PORT: Final = 8842
DASHBOARD_REFRESH_MS: Final = 2000
DASHBOARD_RECENT_LIMIT: Final = 25

# --- Internet Archive --------------------------------------------------------
IA_SEARCH_URL: Final = "https://archive.org/advancedsearch.php"
IA_METADATA_URL: Final = "https://archive.org/metadata"
IA_DOWNLOAD_URL: Final = "https://archive.org/download"
IA_PAGE_SIZE: Final = 100
IA_MAX_RESULTS_PER_QUERY: Final = 500
IA_REQUEST_TIMEOUT_S: Final = 60.0
IA_DOWNLOAD_TIMEOUT_S: Final = 21600.0
IA_USER_AGENT: Final = f"{APP_NAME}/1.0 (+https://github.com/shane-mason/FieldStation42)"
IA_RETRY_ATTEMPTS: Final = 4

# --- Logging -----------------------------------------------------------------
LOG_FILENAME: Final = "broadcastbrain.log"
LOG_MAX_BYTES: Final = 64 * 1024 * 1024
LOG_BACKUP_COUNT: Final = 10
