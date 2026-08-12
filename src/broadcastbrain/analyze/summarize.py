"""Summaries and searchable tags.

The summary is extractive, not generative. For thirty-second commercials an
extractive summary made from the strongest available evidence -- the brand, the
network, the spoken hook, the on-screen call to action -- is more useful and
far more truthful than a paraphrase, and it costs no model and no API budget.
An LLM backend can be slotted in behind the same interface later; because
analyzer results are versioned, doing so re-summarizes the library
automatically.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..types import Category, EntityKind
from ..util.text import dedupe_preserving_order, format_duration, slugify, truncate
from .base import AnalysisContext, Analyzer, AnalyzerResult
from .registry import register

log = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 400
MAX_TAGS = 40
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: Words too generic to be worth indexing as tags.
_STOP_TAGS = frozenset(
    {
        "the", "and", "you", "your", "for", "with", "this", "that", "from", "have", "get",
        "new", "now", "all", "out", "one", "our", "its", "was", "are", "but", "not", "can",
        "will", "just", "more", "than", "then", "them", "they", "what", "when", "here",
    }
)


@register
class SummaryAnalyzer(Analyzer):
    """Compose a human-readable summary and a tag set from all the evidence."""

    name = "summary"
    version = "1"
    requires = ()
    order = 80
    description = "Extractive summary and searchable tag generation"

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        result = AnalyzerResult(backend="extractive")

        estimation = ctx.payload_of("estimation")
        network = _estimate_value(estimation, "network")
        year = _estimate_value(estimation, "air_year")
        series = _estimate_value(estimation, "series")
        genre = _estimate_value(estimation, "genre")

        transcript = ctx.text_of("speech", "text")
        ocr_text = ctx.text_of("ocr", "text")
        brands = _brands_from_context(ctx)

        title = self._make_title(ctx, network, series, brands)
        summary = self._make_summary(
            ctx, network=network, year=year, series=series, genre=genre,
            brands=brands, transcript=transcript, ocr_text=ocr_text,
        )

        tags = self._make_tags(
            ctx, network=network, year=year, genre=genre, brands=brands,
            transcript=transcript, ocr_text=ocr_text,
        )

        result.payload = {
            "title": title,
            "summary": summary,
            "tag_count": len(tags),
            "evidence_used": {
                "transcript_words": len(transcript.split()),
                "ocr_characters": len(ocr_text),
                "brands": brands,
            },
        }
        result.confidence = 0.8 if (transcript or ocr_text) else 0.3
        result.fields = {"summary": summary, "title": title}
        for tag, confidence in tags:
            result.add_tag(tag, confidence)
        return result

    # -- title --------------------------------------------------------------

    def _make_title(
        self,
        ctx: AnalysisContext,
        network: Optional[str],
        series: Optional[str],
        brands: list[str],
    ) -> str:
        """A short, filename-safe, human-recognizable label.

        Titles are what an operator scans in the dashboard and what ends up in
        the library filename, so specificity beats completeness.
        """
        if series:
            return truncate(str(series), 120)
        if brands:
            lead = brands[0]
            return truncate(f"{lead} spot", 120)

        for line in ctx.text_of("ocr", "text").splitlines():
            cleaned = line.strip()
            # A short, title-cased on-screen line is usually the actual title.
            if 4 <= len(cleaned) <= 60 and sum(c.isalpha() for c in cleaned) >= 4:
                return truncate(cleaned, 120)

        if network:
            return truncate(f"{network} clip ({format_duration(ctx.clip.duration_s)})", 120)
        return f"Unidentified clip ({format_duration(ctx.clip.duration_s)})"

    # -- summary ------------------------------------------------------------

    def _make_summary(
        self,
        ctx: AnalysisContext,
        *,
        network: Optional[str],
        year: Optional[int],
        series: Optional[str],
        genre: Optional[str],
        brands: list[str],
        transcript: str,
        ocr_text: str,
    ) -> str:
        parts: list[str] = []

        descriptor = format_duration(ctx.clip.duration_s)
        attribution = " on ".join(filter(None, [series or "", network or ""])) or network or ""
        if attribution and year:
            parts.append(f"{descriptor} clip from {attribution}, around {year}.")
        elif attribution:
            parts.append(f"{descriptor} clip from {attribution}.")
        elif year:
            parts.append(f"{descriptor} clip, estimated {year}.")
        else:
            parts.append(f"{descriptor} clip of unidentified origin.")

        if brands:
            listed = ", ".join(brands[:3])
            parts.append(f"Features {listed}.")

        hook = _best_sentence(transcript)
        if hook:
            parts.append(f'Spoken: "{truncate(hook, 160)}"')
        elif ocr_text:
            first_line = next((line for line in ocr_text.splitlines() if len(line) > 6), "")
            if first_line:
                parts.append(f"On screen: {truncate(first_line, 120)}")

        if genre:
            parts.append(f"Genre: {genre}.")

        scene = ctx.payload_of("scene")
        if scene.get("animation_score", 0) >= 0.65:
            parts.append("Animated.")
        if scene.get("letterboxed"):
            parts.append("Letterboxed.")

        return truncate(" ".join(parts), SUMMARY_MAX_CHARS)

    # -- tags ---------------------------------------------------------------

    def _make_tags(
        self,
        ctx: AnalysisContext,
        *,
        network: Optional[str],
        year: Optional[int],
        genre: Optional[str],
        brands: list[str],
        transcript: str,
        ocr_text: str,
    ) -> list[tuple[str, float]]:
        """Build the tag set that free-text search will hit."""
        tags: list[tuple[str, float]] = []

        def add(value: Optional[str], confidence: float) -> None:
            if not value:
                return
            slug = slugify(str(value), max_length=40)
            if slug and slug not in _STOP_TAGS:
                tags.append((slug, confidence))

        add(network, 0.9)
        add(genre, 0.7)
        if year:
            add(str(year), 0.7)
            add(f"{(year // 10) * 10}s", 0.6)
        for brand in brands:
            add(brand, 0.85)

        for entity_kind in (EntityKind.PERSON, EntityKind.SHOW, EntityKind.SONG):
            for name in _entities_of_kind(ctx, entity_kind):
                add(name, 0.75)

        # Salient words from the text: long enough to be meaningful, seen more
        # than once so a single OCR misread does not become a tag.
        counts: dict[str, int] = {}
        for word in re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", f"{transcript}\n{ocr_text}".lower()):
            if word in _STOP_TAGS:
                continue
            counts[word] = counts.get(word, 0) + 1
        for word, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]:
            if count >= 2:
                add(word, min(0.6, 0.3 + 0.05 * count))

        deduped = dedupe_preserving_order([t for t, _c in tags])
        weights = dict(tags)
        return [(tag, weights.get(tag, 0.5)) for tag in deduped][:MAX_TAGS]


def _estimate_value(payload: dict[str, Any], key: str) -> Optional[Any]:
    entry = payload.get(key)
    if isinstance(entry, dict):
        return entry.get("value")
    return None


def _brands_from_context(ctx: AnalysisContext) -> list[str]:
    from . import knowledge

    matches = knowledge.match_brands(ctx.all_text())
    return dedupe_preserving_order([m.label for m in matches])


def _entities_of_kind(ctx: AnalysisContext, kind: EntityKind) -> list[str]:
    """Entity names of one kind that earlier analyzers asserted."""
    from . import knowledge

    text = ctx.all_text()
    if kind is EntityKind.PERSON:
        return [m.label for m in knowledge.match_people(text)]
    if kind is EntityKind.SHOW:
        return [m.label for m in knowledge.match_shows(text)]
    if kind is EntityKind.SONG:
        recordings = ctx.payload_of("music").get("recordings") or []
        return [str(r.get("title")) for r in recordings[:2] if r.get("title")]
    return []


def _best_sentence(text: str) -> str:
    """The most quotable sentence: the longest one that is not a fragment."""
    if not text:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 12]
    if not sentences:
        return truncate(text.strip(), 160)
    return max(sentences[:12], key=len)


def category_hint(payload: dict[str, Any]) -> Optional[Category]:
    """Best-effort category from a stored summary payload (used by reports)."""
    value = payload.get("category")
    try:
        return Category(str(value)) if value else None
    except ValueError:
        return None
