"""Speech transcription with Whisper.

Uses ``faster-whisper`` (CTranslate2) rather than the reference implementation:
it is several times quicker on CPU, which matters when the library is tens of
thousands of clips and most of them are thirty seconds long.

The model is loaded once per process and reused. Loading a Whisper model takes
seconds; doing it per clip would dominate the runtime of an overnight pass over
short commercials.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..errors import AnalyzerError
from ..types import Category, EntityKind
from . import knowledge
from .base import AnalysisContext, Analyzer, AnalyzerResult
from .registry import register

log = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model_cache: dict[tuple[str, str, str], Any] = {}

#: Whisper hallucinates confident nonsense on pure music or silence. Segments
#: below this average log-probability are dropped.
MIN_SEGMENT_LOGPROB = -1.0
#: A very high no-speech probability means exactly that.
MAX_NO_SPEECH_PROB = 0.75

#: Word count at which a format marker is considered to characterize the whole
#: clip. Interstitials are short and their markers dominate what is said; the
#: same phrase inside a long episode transcript is incidental. Vote weight is
#: scaled by this ratio so one "stay tuned" in 1700 words of dialogue cannot
#: outvote the clip's actual structure.
MARKER_REFERENCE_WORDS = 60


def marker_density(word_count: int, *, floor: float = 0.08) -> float:
    """How much of the clip's speech a matched marker phrase represents."""
    if word_count <= MARKER_REFERENCE_WORDS:
        return 1.0
    return max(floor, MARKER_REFERENCE_WORDS / word_count)


def _load_model(model_name: str, device: str, compute_type: str) -> Any:
    """Load (and cache) a Whisper model for this process."""
    key = (model_name, device, compute_type)
    with _model_lock:
        model = _model_cache.get(key)
        if model is not None:
            return model
        from faster_whisper import WhisperModel

        log.info("loading whisper model %s on %s (%s)", model_name, device, compute_type)
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _model_cache[key] = model
        return model


def _resolve_device(configured: str) -> tuple[str, str]:
    """Pick a device and compute type that will actually work here.

    faster-whisper does not support Apple's MPS backend, so an Apple Silicon
    box runs int8 on CPU -- which is genuinely fast -- instead of failing.
    """
    from .. import capabilities

    device = configured
    if configured == "auto":
        gpu = capabilities.detect_gpu()
        device = "cuda" if gpu.vendor == "nvidia" else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return device, compute


@register
class SpeechAnalyzer(Analyzer):
    """Transcribe dialogue, narration and jingles."""

    name = "speech"
    version = "2"
    requires = ("whisper", "ffmpeg")
    needs_gpu = True
    order = 25
    needs_audio = True
    description = "Whisper speech-to-text with word timings"

    def applicable(self, ctx: AnalysisContext) -> bool:
        if ctx.audio_path is None:
            return False
        return ctx.clip.duration_s >= ctx.settings.analysis.min_transcribe_seconds

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        if ctx.audio_path is None:
            raise AnalyzerError(self.name, "no analysis audio was extracted")

        cfg = ctx.settings.analysis
        device, compute = _resolve_device(cfg.whisper_device)
        if cfg.whisper_compute_type != "auto":
            compute = cfg.whisper_compute_type

        try:
            model = _load_model(cfg.whisper_model, device, compute)
            segments, info = model.transcribe(
                str(ctx.audio_path),
                language=cfg.whisper_language,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,  # stops runaway repetition
            )
            collected = list(segments)
        except Exception as exc:
            raise AnalyzerError(self.name, f"transcription failed: {exc}") from exc

        kept: list[dict[str, Any]] = []
        for segment in collected:
            logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
            no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
            text = (segment.text or "").strip()
            if not text or logprob < MIN_SEGMENT_LOGPROB or no_speech > MAX_NO_SPEECH_PROB:
                continue
            kept.append(
                {
                    "start_s": round(float(segment.start), 3),
                    "end_s": round(float(segment.end), 3),
                    "text": text,
                    "logprob": round(logprob, 3),
                }
            )

        transcript = " ".join(item["text"] for item in kept).strip()
        language = getattr(info, "language", None)
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)

        result = AnalyzerResult(backend=f"faster-whisper/{cfg.whisper_model}@{device}")
        result.payload = {
            "text": transcript,
            "language": language,
            "language_probability": round(language_probability, 3),
            "segment_count": len(kept),
            "dropped_segments": len(collected) - len(kept),
            "words": len(transcript.split()),
            "speech_ratio": _speech_ratio(kept, ctx.clip.duration_s),
        }
        result.confidence = language_probability if transcript else 0.0
        result.texts["transcript"] = (transcript, {"segments": kept, "language": language})
        result.fields["language"] = language

        if not transcript:
            # Silence or pure music is itself informative.
            result.add_tag("no-speech", 0.8)
            result.vote(Category.MUSIC_VIDEO, 0.15, "no intelligible speech")
            return result

        self._mine_transcript(transcript, result)
        return result

    def _mine_transcript(self, transcript: str, result: AnalyzerResult) -> None:
        """Same knowledge base as OCR, applied to what was said."""
        for match in knowledge.match_networks(transcript):
            result.add_entity(EntityKind.NETWORK, match.label, match.confidence * 0.85)
        for match in knowledge.match_brands(transcript):
            result.add_entity(match.kind or EntityKind.BRAND, match.label, match.confidence)
            result.add_tag(match.label.lower(), match.confidence)
            result.vote(
                Category.COMMERCIAL, 0.3 * match.confidence, f"brand mentioned: {match.label}"
            )
        for match in knowledge.match_people(transcript):
            result.add_entity(EntityKind.PERSON, match.label, match.confidence)
        for match in knowledge.match_shows(transcript):
            result.add_entity(EntityKind.SHOW, match.label, match.confidence * 0.9)

        formats = knowledge.match_formats(transcript)
        if formats:
            result.payload["format_markers"] = formats
        weights: dict[str, tuple[Category, float]] = {
            "commercial": (Category.COMMERCIAL, 0.6),
            "promo": (Category.PROMO, 0.55),
            "infomercial": (Category.INFOMERCIAL, 0.75),
            "station_id": (Category.STATION_ID, 0.45),
            "news_break": (Category.NEWS_BREAK, 0.55),
            "weather": (Category.WEATHER, 0.6),
            "ppv_preview": (Category.PPV_PREVIEW, 0.6),
        }
        density = marker_density(len(transcript.split()))
        for label, phrases in formats.items():
            entry = weights.get(label)
            if entry is None:
                continue
            category, base = entry
            weight = base * min(1.5, 1.0 + 0.25 * (len(phrases) - 1)) * density
            result.vote(
                category,
                weight,
                f"spoken: {', '.join(phrases[:3])}"
                + (f" (diluted x{density:.2f} over {len(transcript.split())} words)"
                   if density < 1.0 else ""),
            )


def _speech_ratio(segments: list[dict[str, Any]], duration_s: float) -> float:
    """Fraction of the clip that contains speech.

    A dense, wall-to-wall voiceover is characteristic of a 30-second spot; a
    sparse one is characteristic of a music video or a show.
    """
    if duration_s <= 0 or not segments:
        return 0.0
    spoken = sum(max(0.0, float(s["end_s"]) - float(s["start_s"])) for s in segments)
    return round(min(1.0, spoken / duration_s), 3)


def clear_model_cache() -> None:
    """Release loaded Whisper models (used when shutting a worker down)."""
    with _model_lock:
        _model_cache.clear()
