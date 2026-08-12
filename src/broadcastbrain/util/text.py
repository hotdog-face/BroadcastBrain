"""Text normalization, timecode parsing and small linguistic helpers.

Shared by OCR cleanup, transcript post-processing, entity matching and the
natural-language search parser.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_YEAR = re.compile(r"\b(19[3-9]\d|20[0-4]\d)\b")
_COPYRIGHT_YEAR = re.compile(r"(?:©|\(c\)|copyright|mcm|\bc\b)\s*([12][0-9]{3})", re.IGNORECASE)

#: OCR routinely confuses these on the low-resolution, interlaced text that
#: 2000s broadcast graphics are made of.
_OCR_CONFUSIONS = str.maketrans(
    {"|": "I", "\u00a6": "I", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
)


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def clean_ocr(text: str) -> str:
    """Tidy raw OCR output without destroying evidence.

    Deliberately conservative: it fixes encoding artifacts and collapses
    whitespace but never "corrects" words, because brand names and station
    call signs are exactly the strings a spellchecker would ruin.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_OCR_CONFUSIONS)
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    kept = [line for line in lines if len(line) >= 2]
    return "\n".join(kept)


def slugify(text: str, *, max_length: int = 80) -> str:
    """ASCII slug for filenames, tags and entity keys."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _NON_ALNUM.sub("-", ascii_text.lower()).strip("-")
    return slug[:max_length].strip("-")


def canonical_key(text: str) -> str:
    """Aggressive normalization used to match entities across analyzers.

    ``"Taco Bell"``, ``"TACO BELL!"`` and ``"taco  bell"`` all collapse to
    ``tacobell`` so they resolve to a single entity row.
    """
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub("", ascii_text.lower())


def parse_timecode(value: str) -> float:
    """Parse ``HH:MM:SS.mmm``, ``MM:SS``, or a bare seconds value."""
    value = value.strip()
    if not value:
        raise ValueError("empty timecode")
    if ":" not in value:
        return float(value)
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"unparseable timecode: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def format_timecode(seconds: float, *, millis: bool = True) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` for logs, reports and filenames."""
    if seconds < 0:
        seconds = 0.0
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if millis:
        frac = seconds - int(seconds)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{int(frac * 1000):03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """Human-friendly duration: ``1h 04m``, ``3m 12s``, ``28s``."""
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        hours, rem = divmod(int(seconds), 3600)
        return f"{hours}h {rem // 60:02d}m"
    if seconds >= 60:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.0f}s"


def extract_years(text: str, *, valid_range: tuple[int, int] = (1940, 2030)) -> list[int]:
    """All plausible four-digit years in a blob of OCR or transcript text."""
    low, high = valid_range
    found = [int(m.group(0)) for m in _YEAR.finditer(text or "")]
    return [y for y in found if low <= y <= high]


def extract_copyright_year(
    text: str, *, valid_range: tuple[int, int] = (1940, 2030)
) -> Optional[int]:
    """The year from a copyright notice -- the strongest dating signal there is.

    Broadcast material almost always carries a copyright line in its end card,
    which is why credits detection and year estimation are wired together.

    The range check is not paranoia. This is the highest-confidence date source
    in the system, and it reads OCR of interlaced broadcast text: a misread
    price or phone number matched the pattern and produced "copyright 2804" at
    0.9 confidence on real material.
    """
    low, high = valid_range
    for match in _COPYRIGHT_YEAR.finditer(text or ""):
        year = int(match.group(1))
        if low <= year <= high:
            return year
    return None


def truncate(text: str, limit: int, *, suffix: str = "...") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def join_nonempty(parts: Iterable[Optional[str]], separator: str = " ") -> str:
    return separator.join(p.strip() for p in parts if p and p.strip())


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out
