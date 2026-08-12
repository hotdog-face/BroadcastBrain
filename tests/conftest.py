"""Shared fixtures.

The interesting one is :func:`fake_ffmpeg`, which puts stub ``ffmpeg`` and
``ffprobe`` executables on ``PATH``. That lets the whole pipeline be exercised
end to end -- discovery, segmentation, cutting, analysis, dedupe,
categorization, materialization, export -- on a machine with no ffmpeg and no
real video, in under a second. Integration tests that need actual decoding are
marked ``requires_ffmpeg`` and skipped when it is absent.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from broadcastbrain import capabilities
from broadcastbrain.config import Settings, load_settings
from broadcastbrain.db.base import Database
from broadcastbrain.db.queue import JobQueue
from broadcastbrain.db.schema import initialize

#: A synthetic ffprobe response: 10 minutes of 4:3 480i MPEG-2 with audio --
#: i.e. exactly what a 2000s cable capture looks like.
FAKE_PROBE = {
    "format": {
        "format_name": "mpegts",
        "duration": "600.000000",
        "size": "1048576",
        "bit_rate": "4500000",
        "tags": {"title": "fake capture"},
    },
    "streams": [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "mpeg2video",
            "codec_long_name": "MPEG-2 video",
            "width": 720,
            "height": 480,
            "avg_frame_rate": "30000/1001",
            "r_frame_rate": "30000/1001",
            "pix_fmt": "yuv420p",
            "field_order": "tt",
            "display_aspect_ratio": "4:3",
            "bit_rate": "4000000",
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "ac3",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
            "bit_rate": "192000",
        },
    ],
    "chapters": [],
}

#: Detector output the fake ffmpeg prints, describing a show, a four-spot
#: commercial pod on the 30-second grid, and a return to programming.
_FAKE_DETECTOR_LINES = "\n".join(
    [
        f"[blackdetect @ 0x1] black_start:{t:.2f} black_end:{t + 0.4:.2f} black_duration:0.40"
        for t in (240.0, 270.0, 300.0, 330.0, 360.0)
    ]
    + [
        f"[silencedetect @ 0x1] silence_start: {t:.2f}\n"
        f"[silencedetect @ 0x1] silence_end: {t + 0.5:.2f} | silence_duration: 0.50"
        for t in (240.0, 270.0, 300.0, 330.0, 360.0)
    ]
)

_FFPROBE_STUB = """#!/usr/bin/env python3
import json, sys
PROBE = {probe}
print(json.dumps(PROBE))
"""

_FFMPEG_STUB = '''#!/usr/bin/env python3
"""Stub ffmpeg: writes plausible output files and prints detector lines."""
import sys
from pathlib import Path

args = sys.argv[1:]
DETECTOR_OUTPUT = """{detector}"""

if "-version" in args:
    print("ffmpeg version 0.0-fake")
    sys.exit(0)

# Detector pass: `-f null -`. Emit signals on stderr and write nothing.
if "null" in args:
    sys.stderr.write(DETECTOR_OUTPUT + "\\n")
    sys.exit(0)

# Otherwise the last argument is the output path (or a %04d pattern).
target = Path(args[-1])
target.parent.mkdir(parents=True, exist_ok=True)

if "%04d" in target.name:
    for index in range(1, 5):
        frame = target.parent / target.name.replace("%04d", f"{{index:04d}}")
        frame.write_bytes(b"\\xff\\xd8\\xff" + b"f" * 6000)
elif target.suffix == ".gray":
    # 9x8 grayscale frames for perceptual hashing: a deterministic ramp.
    target.write_bytes(bytes((i * 7 + j) % 256 for i in range(32) for j in range(72)))
else:
    target.write_bytes(b"\\x00" * 200000)
sys.exit(0)
'''


@pytest.fixture()
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put stub ffmpeg/ffprobe binaries first on PATH."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()

    ffprobe = bin_dir / "ffprobe"
    ffprobe.write_text(_FFPROBE_STUB.format(probe=json.dumps(FAKE_PROBE)))
    ffmpeg = bin_dir / "ffmpeg"
    ffmpeg.write_text(_FFMPEG_STUB.format(detector=_FAKE_DETECTOR_LINES))

    for script in (ffprobe, ffmpeg):
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    capabilities.reset()
    yield bin_dir
    capabilities.reset()


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A fully isolated instance rooted in a temp directory."""
    # Environment variables would otherwise leak the developer's own config in.
    for key in list(os.environ):
        if key.startswith("BB_"):
            monkeypatch.delenv(key, raising=False)

    resolved = load_settings(None, paths={"root": str(tmp_path / "data")})
    resolved.paths.ensure()
    resolved.workers.cpu_workers = 1
    resolved.workers.gpu_workers = 0
    resolved.logging.level = "CRITICAL"
    resolved.analysis.frames_per_clip = 4
    resolved.dedupe.phash_frames = 8
    return resolved


@pytest.fixture()
def db(settings: Settings) -> Iterator[Database]:
    database = Database(settings)
    initialize(database)
    yield database
    database.dispose()


@pytest.fixture()
def queue(db: Database) -> JobQueue:
    return JobQueue(db)


@pytest.fixture()
def sample_source(settings: Settings) -> Path:
    """A file in Incoming that the fake ffprobe will happily describe."""
    path = settings.paths.require("incoming") / "capture_2004.ts"
    path.write_bytes(b"\x47" + b"\x00" * (2 * 1024 * 1024))
    # Backdate it so the stability check passes without sleeping.
    old = os.stat(path).st_atime - 3600
    os.utime(path, (old, old))
    return path
