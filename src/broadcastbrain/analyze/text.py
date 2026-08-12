"""On-screen text: OCR over sampled frames.

OCR is the single highest-value analyzer for this material. Broadcast graphics
are *full* of ground truth -- the network bug, the phone number in a spot, the
"NEXT ON" promo card, the copyright year in an end credit. A ten-second clip
with no dialogue at all can still be dated, attributed and categorized purely
from what is printed on it.

Frames are pre-extracted by the runner and shared with the other vision
analyzers, so this costs inference time only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..types import Category, EntityKind
from ..util.text import clean_ocr, dedupe_preserving_order, extract_copyright_year, extract_years
from . import knowledge
from .base import AnalysisContext, Analyzer, AnalyzerResult, sample_frames
from .registry import register

log = logging.getLogger(__name__)

#: Frames per clip to OCR. Text lingers on screen for seconds, so sampling
#: more densely than this mostly re-reads the same card.
OCR_MAX_FRAMES = 10
#: Tesseract page-segmentation mode 11 = "sparse text", which is what broadcast
#: overlays are: a few words scattered over a picture, not a document.
TESSERACT_CONFIG = "--oem 1 --psm 11"
#: Below this per-word confidence, tesseract output on video noise is garbage.
MIN_WORD_CONFIDENCE = 45.0


@register
class OcrAnalyzer(Analyzer):
    """Read on-screen text and mine it for dates, networks, brands and format."""

    name = "ocr"
    version = "1"
    requires = ("pytesseract", "tesseract", "pillow")
    order = 20
    needs_frames = True
    description = "On-screen text via Tesseract, plus year/network/brand extraction"

    def applicable(self, ctx: AnalysisContext) -> bool:
        return bool(ctx.frames)

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        import pytesseract
        from PIL import Image

        frames = sample_frames(ctx.frames, OCR_MAX_FRAMES)
        per_frame: list[dict[str, Any]] = []
        lines: list[str] = []

        for index, frame in enumerate(frames):
            text, confidence = _ocr_frame(pytesseract, Image, frame)
            if not text:
                continue
            at_s = _frame_time(ctx, index, len(frames))
            per_frame.append({"frame": frame.name, "at_s": round(at_s, 2),
                              "confidence": round(confidence, 1), "text": text})
            lines.extend(text.splitlines())

        combined = "\n".join(dedupe_preserving_order(lines))
        result = AnalyzerResult(backend="tesseract")
        result.payload = {
            "text": combined,
            "frames_read": len(per_frame),
            "frames_sampled": len(frames),
            "characters": len(combined),
        }
        result.confidence = min(1.0, len(combined) / 200.0) if combined else 0.0
        result.texts["ocr"] = (combined, {"frames": per_frame})

        if not combined:
            return result

        self._mine_text(combined, result, ctx)
        return result

    # -- evidence extraction ------------------------------------------------

    def _mine_text(self, text: str, result: AnalyzerResult, ctx: AnalysisContext) -> None:
        """Turn raw OCR into entities, tags, dates and classification votes."""
        # Copyright lines are the strongest dating evidence in broadcast media.
        copyright_year = extract_copyright_year(text)
        years = extract_years(text)
        if copyright_year:
            result.payload["copyright_year"] = copyright_year
        if years:
            result.payload["years_seen"] = sorted(set(years))

        for match in knowledge.match_networks(text):
            result.add_entity(EntityKind.NETWORK, match.label, match.confidence)
            result.payload.setdefault("networks", []).append(
                {"name": match.label, "confidence": match.confidence, "evidence": match.evidence}
            )

        for match in knowledge.match_brands(text):
            kind = match.kind or EntityKind.BRAND
            result.add_entity(kind, match.label, match.confidence)
            result.add_tag(match.label.lower(), match.confidence)
            # A brand name printed on screen is overwhelmingly an ad.
            result.vote(Category.COMMERCIAL, 0.35 * match.confidence, f"brand on screen: {match.label}")

        for match in knowledge.match_people(text):
            result.add_entity(EntityKind.PERSON, match.label, match.confidence * 0.8)

        for match in knowledge.match_shows(text):
            result.add_entity(EntityKind.SHOW, match.label, match.confidence)
            network = str(match.extra.get("network") or "")
            if network:
                result.add_entity(EntityKind.NETWORK, network, match.confidence * 0.7)

        formats = knowledge.match_formats(text)
        if formats:
            result.payload["format_markers"] = formats
        _vote_from_formats(formats, result)

        # A phone number on screen is close to definitive for direct response.
        if any(marker in text.lower() for marker in ("1-800", "1 800", "1-888", "call now")):
            result.add_tag("direct-response", 0.9)


def _vote_from_formats(formats: dict[str, list[str]], result: AnalyzerResult) -> None:
    """Convert format markers into weighted category votes."""
    votes: dict[str, tuple[Category, float]] = {
        "commercial": (Category.COMMERCIAL, 0.55),
        "promo": (Category.PROMO, 0.5),
        "station_id": (Category.STATION_ID, 0.4),
        "infomercial": (Category.INFOMERCIAL, 0.7),
        "news_break": (Category.NEWS_BREAK, 0.5),
        "weather": (Category.WEATHER, 0.55),
        "tv_guide": (Category.TV_GUIDE, 0.6),
        "ppv_preview": (Category.PPV_PREVIEW, 0.6),
        "credits": (Category.CREDITS, 0.35),
    }
    for label, phrases in formats.items():
        entry = votes.get(label)
        if entry is None:
            continue
        category, base = entry
        # More distinct marker phrases means more confidence, with diminishing
        # returns -- three "call now" variants is not three times the evidence.
        weight = base * min(1.6, 1.0 + 0.3 * (len(phrases) - 1))
        result.vote(category, weight, f"on-screen text: {', '.join(phrases[:3])}")
        if label == "credits":
            result.fields["has_credits"] = True
        if label == "rating":
            result.add_tag("rated", 0.6)


def _ocr_frame(pytesseract: Any, image_module: Any, frame: Path) -> tuple[str, float]:
    """OCR one frame, filtering out low-confidence noise words.

    Video is a hostile OCR target: interlacing combs, compression mosquitoes
    and moving backgrounds all produce confident-looking nonsense. Dropping
    words below a confidence floor is what keeps the transcript index clean.
    """
    try:
        with image_module.open(frame) as image:
            data = pytesseract.image_to_data(
                image, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
            )
    except Exception as exc:
        log.debug("OCR failed on %s: %s", frame.name, exc)
        return "", 0.0

    words: list[str] = []
    confidences: list[float] = []
    line_map: dict[tuple[int, int, int], list[str]] = {}

    for index, raw_word in enumerate(data.get("text", [])):
        word = (raw_word or "").strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence = 0.0
        if confidence < MIN_WORD_CONFIDENCE:
            continue
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        line_map.setdefault(key, []).append(word)
        words.append(word)
        confidences.append(confidence)

    if not words:
        return "", 0.0
    text = "\n".join(" ".join(parts) for parts in line_map.values())
    mean_confidence = sum(confidences) / len(confidences)
    return clean_ocr(text), mean_confidence


def _frame_time(ctx: AnalysisContext, index: int, total: int) -> float:
    """Approximate clip-relative timestamp of an evenly sampled frame."""
    if total <= 1:
        return ctx.clip.midpoint_s
    return ctx.clip.duration_s * (index / (total - 1))


def ocr_text_for(ctx: AnalysisContext) -> Optional[str]:
    """Convenience accessor used by later analyzers."""
    return ctx.text_of("ocr", "text") or None
