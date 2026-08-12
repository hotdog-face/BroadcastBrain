"""Natural-language query parsing.

Turns the way a person actually asks -- "find every Taco Bell commercial",
"all MTV clips from 2004", "music videos by Green Day" -- into structured
filters plus the leftover free text.

This is deliberately a small, legible rule set rather than an LLM call. The
queries this system needs to answer are highly patterned (a type, a network, a
year, a name), the rules are inspectable when a query surprises you, and it
costs nothing per search. The free-text remainder still goes to keyword and
vector search, so anything the rules do not understand is not lost -- it just
falls through to the general case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..analyze import knowledge
from ..constants import PLAUSIBLE_YEAR_RANGE, SEARCH_DEFAULT_LIMIT
from ..types import Category, EntityKind
from ..util.text import canonical_key, normalize_whitespace

#: Phrases that map onto a category. Ordered longest-first at match time so
#: "music video" wins over "video".
CATEGORY_PHRASES: dict[str, Category] = {
    "commercial": Category.COMMERCIAL,
    "commercials": Category.COMMERCIAL,
    "advert": Category.COMMERCIAL,
    "adverts": Category.COMMERCIAL,
    "advertisement": Category.COMMERCIAL,
    "ad": Category.COMMERCIAL,
    "ads": Category.COMMERCIAL,
    "spot": Category.COMMERCIAL,
    "spots": Category.COMMERCIAL,
    "episode": Category.TV_EPISODE,
    "episodes": Category.TV_EPISODE,
    "show": Category.TV_EPISODE,
    "shows": Category.TV_EPISODE,
    "movie": Category.MOVIE,
    "movies": Category.MOVIE,
    "film": Category.MOVIE,
    "films": Category.MOVIE,
    "promo": Category.PROMO,
    "promos": Category.PROMO,
    "trailer": Category.PROMO,
    "bumper": Category.NETWORK_BUMPER,
    "bumpers": Category.NETWORK_BUMPER,
    "station id": Category.STATION_ID,
    "station ids": Category.STATION_ID,
    "ident": Category.STATION_ID,
    "idents": Category.STATION_ID,
    "music video": Category.MUSIC_VIDEO,
    "music videos": Category.MUSIC_VIDEO,
    "infomercial": Category.INFOMERCIAL,
    "infomercials": Category.INFOMERCIAL,
    "weather": Category.WEATHER,
    "news": Category.NEWS_BREAK,
    "news break": Category.NEWS_BREAK,
    "tv guide": Category.TV_GUIDE,
    "ppv": Category.PPV_PREVIEW,
    "pay per view": Category.PPV_PREVIEW,
    "unknown": Category.UNKNOWN,
}

#: Filler that carries no search meaning.
_STOPWORDS = frozenset(
    {
        "find", "show", "me", "all", "every", "any", "the", "a", "an", "of", "please",
        "get", "list", "search", "for", "clips", "clip", "with", "featuring", "that",
        "which", "some", "give", "from", "by", "in", "on", "at", "about", "appearance",
        "appearances", "there", "is", "are", "was", "were", "and",
    }
)

_YEAR_RE = re.compile(r"\b(19[3-9]\d|20[0-4]\d)\b")
_YEAR_RANGE_RE = re.compile(r"\b(19[3-9]\d|20[0-4]\d)\s*(?:-|to|through|until)\s*(19[3-9]\d|20[0-4]\d)\b")
_DECADE_RE = re.compile(r"\b((?:19|20)\d0)s\b")
_DURATION_RE = re.compile(r"\b(under|over|longer than|shorter than)\s+(\d+)\s*(s|sec|secs|seconds|m|min|mins|minutes)\b")
_BY_RE = re.compile(r"\bby\s+([A-Za-z0-9'&.\- ]{2,40})")
_SIMILAR_RE = re.compile(r"\b(?:similar to|like)\s+([0-9a-f-]{8,36})\b", re.IGNORECASE)
_MENTION_RE = re.compile(r"\bmention(?:ing|s)?\s+([A-Za-z0-9'&.\- ]{2,40})")


@dataclass
class ParsedQuery:
    """A user question decomposed into filters plus leftover text."""

    raw: str
    text: str = ""
    category: Optional[Category] = None
    network: Optional[str] = None
    years: list[int] = field(default_factory=list)
    year_range: Optional[tuple[int, int]] = None
    entities: list[tuple[EntityKind, str]] = field(default_factory=list)
    min_duration_s: Optional[float] = None
    max_duration_s: Optional[float] = None
    similar_to_clip_id: Optional[str] = None
    limit: int = SEARCH_DEFAULT_LIMIT
    include_alternates: bool = False

    @property
    def has_filters(self) -> bool:
        return any(
            [
                self.category,
                self.network,
                self.years,
                self.year_range,
                self.entities,
                self.min_duration_s,
                self.max_duration_s,
                self.similar_to_clip_id,
            ]
        )

    def describe(self) -> str:
        """Plain-English readback, so the user can see how they were understood."""
        parts: list[str] = []
        if self.category:
            parts.append(f"category={self.category.value}")
        if self.network:
            parts.append(f"network={self.network}")
        if self.year_range:
            parts.append(f"years={self.year_range[0]}-{self.year_range[1]}")
        elif self.years:
            parts.append(f"year={','.join(str(y) for y in self.years)}")
        for kind, name in self.entities:
            parts.append(f"{kind.value}={name}")
        if self.min_duration_s:
            parts.append(f"duration>={self.min_duration_s:.0f}s")
        if self.max_duration_s:
            parts.append(f"duration<={self.max_duration_s:.0f}s")
        if self.similar_to_clip_id:
            parts.append(f"similar-to={self.similar_to_clip_id[:8]}")
        if self.text:
            parts.append(f'text="{self.text}"')
        return ", ".join(parts) or "everything"

    def as_dict(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "text": self.text,
            "category": self.category.value if self.category else None,
            "network": self.network,
            "years": self.years,
            "year_range": list(self.year_range) if self.year_range else None,
            "entities": [[k.value, n] for k, n in self.entities],
            "min_duration_s": self.min_duration_s,
            "max_duration_s": self.max_duration_s,
            "similar_to_clip_id": self.similar_to_clip_id,
            "limit": self.limit,
        }


def parse(query: str, *, limit: int = SEARCH_DEFAULT_LIMIT) -> ParsedQuery:
    """Parse a natural-language query into filters and free text."""
    raw = normalize_whitespace(query or "")
    parsed = ParsedQuery(raw=raw, limit=limit)
    if not raw:
        return parsed

    remaining = raw.lower()

    remaining = _extract_similar(parsed, remaining)
    remaining = _extract_years(parsed, remaining)
    remaining = _extract_duration(parsed, remaining)
    remaining = _extract_category(parsed, remaining)
    remaining = _extract_network(parsed, remaining)
    remaining = _extract_entities(parsed, remaining)

    # `_extract_entities` may already have set explicit search text (from a
    # "mentioning X" phrase); the leftover words are appended, never dropped.
    leftover = _clean_remainder(remaining)
    parsed.text = normalize_whitespace(" ".join(part for part in (parsed.text, leftover) if part))
    return parsed


# ---------------------------------------------------------------------------
# Extractors. Each removes what it consumed so later rules see less noise.
# ---------------------------------------------------------------------------


def _extract_similar(parsed: ParsedQuery, text: str) -> str:
    match = _SIMILAR_RE.search(text)
    if match:
        parsed.similar_to_clip_id = match.group(1)
        return text.replace(match.group(0), " ")
    return text


def _extract_years(parsed: ParsedQuery, text: str) -> str:
    low, high = PLAUSIBLE_YEAR_RANGE

    range_match = _YEAR_RANGE_RE.search(text)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        parsed.year_range = (min(start, end), max(start, end))
        return text.replace(range_match.group(0), " ")

    decade_match = _DECADE_RE.search(text)
    if decade_match:
        decade = int(decade_match.group(1))
        parsed.year_range = (decade, decade + 9)
        text = text.replace(decade_match.group(0), " ")

    years = [int(m.group(0)) for m in _YEAR_RE.finditer(text)]
    years = [y for y in years if low <= y <= high]
    if years:
        parsed.years = sorted(set(years))
        for year in parsed.years:
            text = text.replace(str(year), " ")
    return text


def _extract_duration(parsed: ParsedQuery, text: str) -> str:
    match = _DURATION_RE.search(text)
    if not match:
        return text
    comparator, amount, unit = match.group(1), float(match.group(2)), match.group(3)
    seconds = amount * (60.0 if unit.startswith("m") else 1.0)
    if comparator in ("under", "shorter than"):
        parsed.max_duration_s = seconds
    else:
        parsed.min_duration_s = seconds
    return text.replace(match.group(0), " ")


def _extract_category(parsed: ParsedQuery, text: str) -> str:
    # Longest phrase first, so "music video" beats "video".
    for phrase in sorted(CATEGORY_PHRASES, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(phrase)}\b")
        if pattern.search(text):
            parsed.category = CATEGORY_PHRASES[phrase]
            return pattern.sub(" ", text, count=1)
    return text


def _extract_network(parsed: ParsedQuery, text: str) -> str:
    matches = knowledge.match_networks(text)
    if not matches:
        return text
    best = matches[0]
    parsed.network = best.label
    if best.evidence:
        text = re.sub(rf"\b{re.escape(best.evidence)}\b", " ", text, count=1)
    return text


def _extract_entities(parsed: ParsedQuery, text: str) -> str:
    """Pull brands, people and shows out of what is left."""
    for match in knowledge.match_brands(text):
        kind = match.kind or EntityKind.BRAND
        parsed.entities.append((kind, match.label))
        text = re.sub(rf"\b{re.escape(match.evidence)}\b", " ", text, count=1)

    for match in knowledge.match_people(text):
        parsed.entities.append((EntityKind.PERSON, match.label))
        text = re.sub(rf"\b{re.escape(match.evidence)}\b", " ", text, count=1)

    for match in knowledge.match_shows(text):
        parsed.entities.append((EntityKind.SHOW, match.label))
        text = re.sub(rf"\b{re.escape(match.evidence)}\b", " ", text, count=1)

    # "by <name>" names an artist or director we may not have in the knowledge
    # base yet -- keep it as an entity hint rather than dropping it.
    by_match = _BY_RE.search(text)
    if by_match:
        candidate = by_match.group(1).strip()
        if candidate and canonical_key(candidate):
            known = {canonical_key(name) for _kind, name in parsed.entities}
            if canonical_key(candidate) not in known:
                parsed.entities.append((EntityKind.ARTIST, candidate))
            text = text.replace(by_match.group(0), " ")

    mention_match = _MENTION_RE.search(text)
    if mention_match:
        # "mentioning Star Wars" is a text search, not an entity filter: the
        # phrase should hit transcripts and OCR wherever it appears.
        parsed.text = normalize_whitespace(mention_match.group(1))
        text = text.replace(mention_match.group(0), " ")
    return text


def _clean_remainder(text: str) -> str:
    words = [word for word in re.split(r"[^\w'&-]+", text) if word and word not in _STOPWORDS]
    return normalize_whitespace(" ".join(words))


def to_fts_expression(text: str) -> str:
    """Build an FTS5 MATCH expression from free text.

    Terms are OR-ed with a prefix wildcard, which behaves the way people expect
    from a search box: "taco bell" finds documents containing either word, and
    ranking (BM25) sorts the ones containing both to the top.
    """
    terms = [t for t in re.split(r"\s+", text.strip()) if t]
    if not terms:
        return ""
    escaped = [f'"{t}"*' if not t.isalnum() else f"{t}*" for t in terms]
    return " OR ".join(escaped)
