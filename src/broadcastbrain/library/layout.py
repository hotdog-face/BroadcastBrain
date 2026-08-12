"""Library layout: where a clip lives on disk and what it is called.

The layout is browsable by a human with a file manager and stable enough that
Fieldstation (or anything else) can point at a folder and get sensible
content. Two rules govern it:

* **The path encodes the classification.** ``Library/Commercials/2004/Taco
  Bell/...`` tells you what a file is without opening a database.
* **The filename is stable and unique.** Names are derived from the clip's
  identity, not from a counter, so re-materializing a clip after a re-encode
  overwrites the same path instead of accumulating copies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..db.models import Clip
from ..types import Category
from ..util.atomic import safe_filename
from ..util.text import slugify

#: Category -> folder name under ``Library/``. The first eight are the folders
#: named in the project brief; the rest keep material that would otherwise be
#: dumped into Unknown properly separated.
CATEGORY_DIRECTORIES: dict[Category, str] = {
    Category.COMMERCIAL: "Commercials",
    Category.TV_EPISODE: "TV Shows",
    Category.MOVIE: "Movies",
    Category.PROMO: "Promos",
    Category.STATION_ID: "Station IDs",
    Category.NETWORK_BUMPER: "Network Bumpers",
    Category.MUSIC_VIDEO: "Music Videos",
    Category.UNKNOWN: "Unknown",
    Category.INFOMERCIAL: "Infomercials",
    Category.PPV_PREVIEW: "PPV Previews",
    Category.WEATHER: "Weather",
    Category.NEWS_BREAK: "News Breaks",
    Category.TV_GUIDE: "TV Guide",
    Category.INTRO: "Intros",
    Category.CREDITS: "Credits",
}

#: Every folder the library creates, in the order they are listed to the user.
LIBRARY_DIRECTORIES: tuple[str, ...] = tuple(
    dict.fromkeys(CATEGORY_DIRECTORIES[category] for category in Category)
)

#: Maximum characters of title used in a filename before truncation.
TITLE_LENGTH = 60


def category_directory(category: Category) -> str:
    return CATEGORY_DIRECTORIES.get(category, CATEGORY_DIRECTORIES[Category.UNKNOWN])


def ensure_library_tree(library_root: Path) -> None:
    """Create every category folder up front.

    An empty ``Movies/`` folder is information: it says the pipeline knows about
    movies and has not found one yet, rather than leaving the operator to
    wonder whether the feature exists.
    """
    for name in LIBRARY_DIRECTORIES:
        (library_root / name).mkdir(parents=True, exist_ok=True)


def clip_relative_path(clip: Clip, *, extension: Optional[str] = None) -> Path:
    """Where a clip belongs, relative to ``Library/``.

    Grouping inside a category is chosen per category to match how someone
    actually browses: commercials by year then brand, shows by series, music
    videos by artist.
    """
    category = _category_of(clip)
    parts: list[str] = [category_directory(category)]

    if category is Category.COMMERCIAL:
        parts.append(str(clip.air_year) if clip.air_year else "Undated")
        brand = _primary_label(clip)
        if brand:
            parts.append(safe_filename(brand, max_bytes=80))
    elif category in (Category.TV_EPISODE, Category.MOVIE):
        parts.append(safe_filename(clip.series or clip.title or "Unidentified", max_bytes=80))
    elif category in (Category.PROMO, Category.STATION_ID, Category.NETWORK_BUMPER, Category.PPV_PREVIEW):
        parts.append(safe_filename(clip.network or "Unidentified Network", max_bytes=80))
    elif category is Category.MUSIC_VIDEO:
        parts.append(safe_filename(clip.series or clip.network or "Unidentified", max_bytes=80))
    elif clip.air_year:
        parts.append(str(clip.air_year))

    parts.append(clip_filename(clip, extension=extension))
    return Path(*parts)


def clip_filename(clip: Clip, *, extension: Optional[str] = None) -> str:
    """A descriptive, unique, filesystem-safe filename.

    Shape: ``<title>__<network>-<year>__<id8>.<ext>``. The 8-character id
    suffix guarantees uniqueness without an existence check, which matters when
    many workers materialize clips concurrently.
    """
    suffix = extension or _extension_of(clip)
    title = clip.title or clip.series or clip.category or "clip"
    pieces = [safe_filename(title, max_bytes=TITLE_LENGTH)]

    context = "-".join(
        part for part in (clip.network or "", str(clip.air_year) if clip.air_year else "") if part
    )
    if context:
        pieces.append(safe_filename(context, max_bytes=40))
    pieces.append(clip.id[:8])
    return "__".join(pieces) + suffix


def thumbnail_filename(clip: Clip) -> str:
    return f"{clip.id}.jpg"


def metadata_filename(clip: Clip) -> str:
    return f"{clip.id}.json"


def export_slug(clip: Clip) -> str:
    """Stable, lowercase identifier used in the Fieldstation export."""
    base = slugify(clip.title or clip.series or clip.category or "clip", max_length=48)
    return f"{base or 'clip'}-{clip.id[:8]}"


def _category_of(clip: Clip) -> Category:
    try:
        return Category(clip.category)
    except ValueError:
        return Category.UNKNOWN


def _primary_label(clip: Clip) -> Optional[str]:
    """The brand or series a clip is filed under, if the title reveals one."""
    if clip.series:
        return clip.series
    title = clip.title or ""
    # Titles produced by the summarizer look like "Taco Bell spot".
    if title.endswith(" spot"):
        return title[: -len(" spot")]
    return None


def _extension_of(clip: Clip) -> str:
    for candidate in (clip.processing_path, clip.file_path):
        if candidate:
            suffix = Path(candidate).suffix
            if suffix:
                return suffix
    return ".mkv"
