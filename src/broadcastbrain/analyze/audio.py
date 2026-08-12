"""Audio fingerprinting and music identification.

Chromaprint produces an acoustic fingerprint that survives re-encoding, so two
copies of the same commercial ripped from different sources fingerprint alike.
That serves two purposes here: it is a deduplication signal, and -- when an
AcoustID API key is configured -- it identifies the music in a music video or
the licensed track under a spot.

Identification is optional and network-dependent. Fingerprinting is not, and
the pipeline only needs the fingerprint to do its job.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..constants import IA_REQUEST_TIMEOUT_S
from ..errors import AnalyzerError
from ..types import Category, EntityKind
from ..util.proc import run, which
from .base import AnalysisContext, Analyzer, AnalyzerResult
from .registry import register

log = logging.getLogger(__name__)

#: fpcalc's default fingerprint window. 120s covers any clip we care about.
FINGERPRINT_LENGTH_S = 120
ACOUSTID_URL = "https://api.acoustid.org/v2/lookup"


@register
class AudioFingerprintAnalyzer(Analyzer):
    """Compute a Chromaprint fingerprint for dedupe and music lookup."""

    name = "audio_fingerprint"
    version = "1"
    requires = ("chromaprint",)
    order = 50
    needs_audio = True
    description = "Chromaprint acoustic fingerprint (dedupe + music identification)"

    def applicable(self, ctx: AnalysisContext) -> bool:
        return ctx.audio_path is not None

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        if ctx.audio_path is None:
            raise AnalyzerError(self.name, "no analysis audio available")

        binary = which(ctx.settings.fpcalc_binary) or which("fpcalc")
        if binary is None:  # pragma: no cover - guarded by check_backends
            raise AnalyzerError(self.name, "fpcalc not found")

        try:
            completed = run(
                [binary, "-length", str(FINGERPRINT_LENGTH_S), "-json", str(ctx.audio_path)],
                timeout_s=300,
                check=True,
                stderr_tail_lines=20,
            )
        except Exception as exc:
            raise AnalyzerError(self.name, f"fpcalc failed: {exc}") from exc

        fingerprint, duration = _parse_fpcalc(completed.stdout)
        if not fingerprint:
            raise AnalyzerError(self.name, "fpcalc produced no fingerprint")

        result = AnalyzerResult(backend="chromaprint")
        result.payload = {
            "fingerprint": fingerprint,
            "fingerprint_duration_s": duration,
            "length": len(fingerprint),
        }
        result.confidence = 1.0
        result.fields["chromaprint"] = fingerprint
        return result


@register
class MusicAnalyzer(Analyzer):
    """Identify music via AcoustID, when a key is configured.

    Without a key this reports ``skipped``: no key, no lookup, no pretending.
    The fingerprint is still stored by :class:`AudioFingerprintAnalyzer`, so
    enabling identification later re-runs only this analyzer.
    """

    name = "music"
    version = "1"
    requires = ("chromaprint",)
    order = 55
    description = "Music identification via AcoustID (requires an API key)"

    def applicable(self, ctx: AnalysisContext) -> bool:
        return bool(_acoustid_key())

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        key = _acoustid_key()
        if not key:
            from ..errors import BackendUnavailableError

            raise BackendUnavailableError(
                self.name, "set BB_ACOUSTID_API_KEY to enable music identification"
            )

        fingerprint_payload = ctx.payload_of("audio_fingerprint")
        fingerprint = fingerprint_payload.get("fingerprint")
        duration = fingerprint_payload.get("fingerprint_duration_s")
        if not fingerprint:
            raise AnalyzerError(self.name, "no fingerprint available to look up")

        recordings = _acoustid_lookup(key, str(fingerprint), float(duration or ctx.clip.duration_s))
        result = AnalyzerResult(backend="acoustid")
        result.payload = {"recordings": recordings[:10], "match_count": len(recordings)}

        if not recordings:
            result.confidence = 0.0
            return result

        best = recordings[0]
        result.confidence = float(best.get("score", 0.0))
        title = best.get("title")
        artists = best.get("artists") or []

        if title:
            result.add_entity(EntityKind.SONG, str(title), result.confidence)
            result.add_tag(str(title).lower(), result.confidence)
        for artist in artists:
            result.add_entity(EntityKind.ARTIST, str(artist), result.confidence * 0.95)
            result.add_tag(str(artist).lower(), result.confidence * 0.95)

        if result.confidence >= 0.8 and ctx.clip.duration_s >= 100:
            result.vote(Category.MUSIC_VIDEO, 0.5, f"identified music: {title}")
        return result


def _acoustid_key() -> Optional[str]:
    """API key from the environment. Deliberately not stored in the config file."""
    import os

    return os.environ.get("BB_ACOUSTID_API_KEY") or None


def _parse_fpcalc(stdout: str) -> tuple[Optional[str], float]:
    import json

    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None, 0.0
    fingerprint = payload.get("fingerprint")
    duration = float(payload.get("duration") or 0.0)
    return (str(fingerprint) if fingerprint else None), duration


def _acoustid_lookup(api_key: str, fingerprint: str, duration_s: float) -> list[dict[str, Any]]:
    """Query AcoustID. Network failures degrade to "no match", never to an error."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a core dependency
        return []

    params = {
        "client": api_key,
        "format": "json",
        "duration": str(int(duration_s)),
        "fingerprint": fingerprint,
        "meta": "recordings",
    }
    try:
        response = httpx.get(ACOUSTID_URL, params=params, timeout=IA_REQUEST_TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        log.debug("acoustid lookup failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for entry in payload.get("results") or []:
        score = float(entry.get("score") or 0.0)
        for recording in entry.get("recordings") or []:
            results.append(
                {
                    "score": round(score, 3),
                    "id": recording.get("id"),
                    "title": recording.get("title"),
                    "artists": [a.get("name") for a in recording.get("artists") or [] if a.get("name")],
                    "duration": recording.get("duration"),
                }
            )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
