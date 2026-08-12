"""A real FieldStation42 station: content tree plus ``station_conf`` JSON.

FieldStation42 is **directory-driven**, not manifest-driven. It does not read a
metadata file describing your library; it reads a folder whose *subdirectory
names are the tags* its scheduler refers to:

    catalog/cartoon_network/
        commercial/          <- ad clips
        bump/                <- station IDs, network bumpers, promos
        toonami/             <- programming, referenced as "tags": "toonami"
        toonami/dragon_ball_z/

and a JSON file in ``confs/`` containing a ``station_conf`` object that names
the channel, points at that directory, and lays out a weekly schedule of
``{"tags": ...}`` slots per hour.

So the adapter's job is not to describe the library -- it is to *build that
directory* and write a config whose tags provably exist inside it.

Files are **hardlinked**, never copied. A 100 TB library must not be
duplicated to be scheduled; the fallback to a copy only happens across
filesystem boundaries.

Reference: https://fieldstation42.com/docs/reference/station-config/
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..db.base import Database
from ..db.models import Clip
from ..db.repo import ClipRepository
from ..errors import ExportError
from ..types import Category
from ..util.atomic import write_json
from ..util.text import slugify

log = logging.getLogger(__name__)

#: FieldStation42's defaults for the two special directories. Kept as its
#: names, not ours, because the scheduler looks for exactly these.
FS42_COMMERCIAL_DIR = "commercial"
FS42_BUMP_DIR = "bump"

#: Categories that become FieldStation's ad pool.
COMMERCIAL_CATEGORIES = frozenset({Category.COMMERCIAL, Category.INFOMERCIAL})
#: Categories that become the bump pool -- the short things between programmes.
BUMP_CATEGORIES = frozenset(
    {Category.STATION_ID, Category.NETWORK_BUMPER, Category.PROMO, Category.PPV_PREVIEW}
)
#: Programming, in the order the day is built from it.
PROGRAMME_TAG_FOR_CATEGORY: dict[Category, str] = {
    Category.TV_EPISODE: "shows",
    Category.MOVIE: "movies",
    Category.MUSIC_VIDEO: "music",
    Category.NEWS_BREAK: "news",
    Category.WEATHER: "weather",
    Category.TV_GUIDE: "guide",
}
#: Everything else is catalogued but not scheduled: present for the operator to
#: promote by hand, invisible to the scheduler until they do.
UNSORTED_TAG = "unsorted"

#: Channel numbers are assigned from here upward, in a stable order.
FIRST_CHANNEL_NUMBER = 3

#: A station with fewer than this many minutes of programming cannot fill a
#: schedule; it is written but flagged, rather than silently producing a
#: channel that plays the same clip all day.
MIN_PROGRAMME_MINUTES = 20.0


@dataclass
class StationPlan:
    """One channel: its inventory, its config, and whether it can actually run."""

    network: str
    slug: str
    channel_number: int
    content_dir: Path
    programme_tags: dict[str, list[Clip]] = field(default_factory=dict)
    commercials: list[Clip] = field(default_factory=list)
    bumps: list[Clip] = field(default_factory=list)
    unsorted: list[Clip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def programme_seconds(self) -> float:
        return sum(c.duration_s for clips in self.programme_tags.values() for c in clips)

    @property
    def schedulable(self) -> bool:
        return bool(self.programme_tags) and self.programme_seconds >= MIN_PROGRAMME_MINUTES * 60

    def summary(self) -> dict[str, Any]:
        return {
            "network": self.network,
            "channel_number": self.channel_number,
            "content_dir": str(self.content_dir),
            "programme_minutes": round(self.programme_seconds / 60.0, 1),
            "tags": {tag: len(clips) for tag, clips in self.programme_tags.items()},
            "commercials": len(self.commercials),
            "bumps": len(self.bumps),
            "unsorted": len(self.unsorted),
            "schedulable": self.schedulable,
            "warnings": self.warnings,
        }


@dataclass
class Fs42ExportResult:
    root: Path
    stations: list[StationPlan]
    files_linked: int
    files_copied: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "stations": [s.summary() for s in self.stations],
            "schedulable_stations": sum(1 for s in self.stations if s.schedulable),
            "files_linked": self.files_linked,
            "files_copied": self.files_copied,
        }


class Fieldstation42Exporter:
    """Builds a FieldStation42 catalog tree and station configs."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        output_root: Optional[Path] = None,
        content_root: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_root: Where to build the export. Defaults to the configured
                export directory; point it at removable media to hand the
                catalog to another machine.
            content_root: What to write into each config's ``content_dir``,
                instead of this machine's absolute path. The catalog is useless
                on the target machine if the path in the config does not exist
                there -- and FieldStation42 resolves relative paths against its
                own root, so ``"catalog/<slug>"`` travels correctly once the
                catalog folder is copied into the FieldStation42 directory.
        """
        self.settings = settings
        self.db = db
        self.root = output_root or (settings.paths.require("export") / "fieldstation42")
        self.content_root = content_root
        self._linked = 0
        self._copied = 0

    # -- public API ---------------------------------------------------------

    def export(self, *, first_channel: int = FIRST_CHANNEL_NUMBER) -> Fs42ExportResult:
        """Regenerate the whole FieldStation42 handoff."""
        clips = self._collect()
        if not clips:
            raise ExportError("no clips are ready to export")

        plans = self._plan(clips, first_channel=first_channel)
        self._reset_root()

        for plan in plans:
            self._build_content_tree(plan)
            self._write_station_conf(plan)

        self._write_readme(plans)
        result = Fs42ExportResult(
            root=self.root, stations=plans, files_linked=self._linked, files_copied=self._copied
        )
        write_json(self.root / "broadcastbrain_export.json", result.as_dict())
        self._strip_apple_double()
        self._verify(plans)

        schedulable = [p for p in plans if p.schedulable]
        log.info(
            "FieldStation42 export: %d station(s), %d schedulable, %d file(s) linked",
            len(plans), len(schedulable), self._linked,
        )
        return result

    # -- planning -----------------------------------------------------------

    def _collect(self) -> list[Clip]:
        with self.db.session() as session:
            clips = ClipRepository.ready_for_export(
                session, min_confidence=self.settings.export.min_confidence
            )
            usable = [c for c in clips if c.file_path and Path(c.file_path).exists()]
            session.expunge_all()
        return usable

    def _plan(self, clips: list[Clip], *, first_channel: int) -> list[StationPlan]:
        """Group clips into stations, one per network.

        Clips with no identified network go to a single "General" channel
        rather than being dropped -- an unattributed commercial is still a
        perfectly good commercial.
        """
        by_network: dict[str, list[Clip]] = {}
        for clip in clips:
            by_network.setdefault(clip.network or "General", []).append(clip)

        plans: list[StationPlan] = []
        for index, (network, network_clips) in enumerate(sorted(by_network.items())):
            slug = slugify(network) or f"channel{index}"
            plan = StationPlan(
                network=network,
                slug=slug,
                channel_number=first_channel + index,
                content_dir=self.root / "catalog" / slug,
            )
            for clip in network_clips:
                self._assign(plan, clip)
            self._check(plan)
            plans.append(plan)
        return plans

    def _assign(self, plan: StationPlan, clip: Clip) -> None:
        """Route one clip to the ad pool, the bump pool, or a programming tag."""
        try:
            category = Category(clip.category)
        except ValueError:
            category = Category.UNKNOWN

        if category in COMMERCIAL_CATEGORIES:
            plan.commercials.append(clip)
            return
        if category in BUMP_CATEGORIES:
            plan.bumps.append(clip)
            return

        tag = PROGRAMME_TAG_FOR_CATEGORY.get(category)
        if tag is None:
            plan.unsorted.append(clip)
            return

        # A known series becomes a nested tag, which is exactly how
        # FieldStation42 expects hierarchy: "shows/dragon_ball_z".
        if clip.series:
            tag = f"{tag}/{slugify(clip.series)}"
        plan.programme_tags.setdefault(tag, []).append(clip)

    def _check(self, plan: StationPlan) -> None:
        if not plan.programme_tags:
            plan.warnings.append(
                "no programming -- this channel would be off-air all day; "
                "its commercials and bumps are still available to other stations"
            )
        elif plan.programme_seconds < MIN_PROGRAMME_MINUTES * 60:
            plan.warnings.append(
                f"only {plan.programme_seconds / 60:.0f} minutes of programming; "
                "the schedule will repeat heavily"
            )
        if not plan.commercials:
            plan.warnings.append("no commercials -- config sets commercial_free")
        if not plan.bumps:
            plan.warnings.append("no bumps -- breaks will cut straight back to programming")

    # -- content tree -------------------------------------------------------

    def _reset_root(self) -> None:
        """Rebuild from scratch. The tree is derived, so it is never patched.

        The error handler is not defensive padding. On exFAT -- which is what
        removable media is formatted as -- deleting ``name.json`` also removes
        the AppleDouble ``._name.json`` beside it, so ``rmtree``'s directory
        listing goes stale mid-walk and it raises ``FileNotFoundError``. That
        aborted the export and left a half-deleted tree on the card: configs
        for some channels, media for others, and no error the operator would
        notice until FieldStation42 failed to load it.
        """

        def _tolerate_vanished(
            _func: Any, _path: str, exc_info: Any
        ) -> None:  # pragma: no cover - filesystem specific
            error = exc_info[1]
            if isinstance(error, FileNotFoundError):
                return  # already gone, which is the outcome we wanted
            raise error

        if self.root.exists():
            shutil.rmtree(self.root, onerror=_tolerate_vanished)
        (self.root / "confs").mkdir(parents=True, exist_ok=True)
        (self.root / "catalog").mkdir(parents=True, exist_ok=True)

    def _build_content_tree(self, plan: StationPlan) -> None:
        """Create the tag directories and link every clip into place."""
        for tag, clips in plan.programme_tags.items():
            for clip in clips:
                self._place(clip, plan.content_dir / tag)
        for clip in plan.commercials:
            self._place(clip, plan.content_dir / FS42_COMMERCIAL_DIR)
        for clip in plan.bumps:
            self._place(clip, plan.content_dir / FS42_BUMP_DIR)
        for clip in plan.unsorted:
            self._place(clip, plan.content_dir / UNSORTED_TAG)

    def _place(self, clip: Clip, directory: Path) -> None:
        """Hardlink one clip into the catalog.

        Hardlinks matter here: the same media has to appear under a tag name
        FieldStation42 understands while still living in the library. Copying
        would double a 100 TB archive to schedule it.
        """
        source = Path(str(clip.file_path))
        if not source.exists():
            return
        directory.mkdir(parents=True, exist_ok=True)
        # ASCII-only names on the way out. A non-ASCII character forces exFAT
        # to store an extended attribute, which macOS materializes as a "._"
        # sidecar that cannot be deleted (the driver regenerates it) and that
        # FieldStation42 would catalog as a four-kilobyte programme. Clip
        # titles come from OCR, so stray accented characters are routine.
        target = directory / _ascii_name(source.name)

        if target.exists():
            return
        try:
            os.link(source, target)
            self._linked += 1
        except OSError:
            # Different filesystem, or one without hardlinks -- removable media.
            # `copy2` would carry extended attributes across, and on exFAT/FAT
            # macOS materializes those as AppleDouble "._name" sidecars. They
            # look like media files to a catalog scanner, so the plain copy is
            # the correct one here.
            try:
                shutil.copy(source, target)
                self._copied += 1
            except OSError as exc:  # pragma: no cover - disk failure
                log.warning("could not place %s: %s", source.name, exc)

    def _verify(self, plans: list[StationPlan]) -> None:
        """Confirm the export on disk matches what was planned.

        Exporting to removable media can fail halfway and leave something that
        looks finished. Every planned channel must have ended up with a config;
        anything less is raised rather than reported, because the failure mode
        is carrying a broken card to another machine and finding out there.
        """
        missing = [
            plan.network
            for plan in plans
            if not (self.root / "confs" / f"{plan.slug}.json").is_file()
        ]
        if missing:
            raise ExportError(
                "export is incomplete -- no config was written for: "
                + ", ".join(sorted(missing))
                + f". The directory at {self.root} should not be used."
            )

    def _strip_apple_double(self) -> None:
        """Remove macOS "._" sidecars from the export.

        Writing to exFAT or FAT removable media makes macOS spill an
        AppleDouble file beside every real one. FieldStation42 scans
        directories for content, so "._episode.mkv" would be catalogued as a
        four-kilobyte programme and scheduled to air.
        """
        if shutil.which("dot_clean"):
            # The supported tool: merges the sidecars back into the files.
            subprocess.run(
                ["dot_clean", "-m", str(self.root)],
                check=False, capture_output=True, timeout=600,
            )
        removed = 0
        for sidecar in self.root.rglob("._*"):
            if sidecar.is_file():
                try:
                    sidecar.unlink()
                    removed += 1
                except OSError:  # pragma: no cover - media disconnected
                    pass
        if removed:
            log.info("removed %d AppleDouble sidecar(s) from the export", removed)

    # -- station config -----------------------------------------------------

    def _write_station_conf(self, plan: StationPlan) -> None:
        """Emit ``confs/<slug>.json`` in FieldStation42's own format."""
        # An absolute path is right for a local install and wrong for removable
        # media; `content_root` lets the caller write the path the *target*
        # machine will see.
        if self.content_root:
            content_dir = f"{self.content_root.rstrip('/')}/{plan.slug}"
        else:
            content_dir = str(plan.content_dir)

        conf: dict[str, Any] = {
            "network_name": plan.network,
            "channel_number": plan.channel_number,
            "content_dir": content_dir,
            "commercial_dir": FS42_COMMERCIAL_DIR,
            "bump_dir": FS42_BUMP_DIR,
            "schedule_increment": 30,
            "break_strategy": "standard",
            "break_duration": int(self.settings.export.pod_target_s),
            # A channel with no ad inventory must say so, or the scheduler
            # will look for breaks it cannot fill.
            "commercial_free": not plan.commercials,
        }

        tags = sorted(plan.programme_tags)
        if tags:
            conf["fallback_tag"] = tags[0]

        template = self._day_template(tags)
        conf["day_templates"] = {"daily": template}
        # All seven days must be present; an omitted day is off-air.
        for day in (
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        ):
            conf[day] = "daily" if template else {}

        write_json(self.root / "confs" / f"{plan.slug}.json", {"station_conf": conf})

    def _day_template(self, tags: list[str]) -> dict[str, dict[str, str]]:
        """Lay the available tags across a 24-hour day.

        Deliberately simple and even: with one recording's worth of material
        there is no basis for claiming a daypart structure. Tags rotate hour by
        hour so everything gets air time, and the operator edits from there --
        which is far easier than unpicking an invented schedule.
        """
        if not tags:
            return {}
        return {str(hour): {"tags": tags[hour % len(tags)]} for hour in range(24)}

    # -- documentation ------------------------------------------------------

    def _write_readme(self, plans: list[StationPlan]) -> None:
        lines = [
            "# FieldStation42 handoff",
            "",
            "Generated by BroadcastBrain. This directory is in FieldStation42's own",
            "format: a content tree whose subdirectory names are the scheduler's tags,",
            "plus one station config per channel.",
            "",
            "## Install",
            "",
            "```bash",
            "cp confs/*.json  <FieldStation42>/confs/",
            "python3 validate_configs.py          # from the FieldStation42 root",
            "python3 station_42.py --rebuild_catalog",
            "```",
            "",
            "`content_dir` in each config is an absolute path into this export, so the",
            "catalog does not need to be moved. Media is hardlinked from the library:",
            "it occupies no additional disk, and deleting this directory does not touch",
            "your clips.",
            "",
            "## Channels",
            "",
            "| Ch | Network | Programming | Ads | Bumps | Ready |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
        for plan in plans:
            lines.append(
                f"| {plan.channel_number} | {plan.network} | "
                f"{plan.programme_seconds / 60:.0f} min | {len(plan.commercials)} | "
                f"{len(plan.bumps)} | {'yes' if plan.schedulable else 'no'} |"
            )

        flagged = [p for p in plans if p.warnings]
        if flagged:
            lines += ["", "## Warnings", ""]
            for plan in flagged:
                for warning in plan.warnings:
                    lines.append(f"- **{plan.network}**: {warning}")

        lines += [
            "",
            "## Tags",
            "",
            "Each channel's schedule refers to these by directory name:",
            "",
        ]
        for plan in plans:
            if plan.programme_tags:
                listed = ", ".join(f"`{t}`" for t in sorted(plan.programme_tags))
                lines.append(f"- **{plan.network}**: {listed}")
            if plan.unsorted:
                lines.append(
                    f"  - plus `{UNSORTED_TAG}/` ({len(plan.unsorted)} clip(s)) — catalogued "
                    "but not scheduled; move them into a tag folder to put them on air"
                )

        lines += [
            "",
            "## The generated schedule",
            "",
            "Tags rotate hour by hour across all seven days. That is a starting point,",
            "not a claim about how the channel should run — with a small library there",
            "is no basis for inventing dayparts. Edit `day_templates` in the config to",
            "shape it.",
            "",
        ]
        (self.root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _ascii_name(name: str) -> str:
    """Transliterate a filename to ASCII for removable media."""
    import unicodedata

    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(ch if ch.isprintable() else "_" for ch in folded).strip()
    return cleaned or "clip.mkv"


def validate_station_conf(conf: dict[str, Any]) -> list[str]:
    """Check a config against FieldStation42's documented requirements.

    Cheap insurance: catching a missing day here is far better than finding out
    when a channel is silently off-air on Wednesdays.
    """
    problems: list[str] = []
    station = conf.get("station_conf")
    if not isinstance(station, dict):
        return ["missing top-level 'station_conf' object"]

    if not station.get("network_name"):
        problems.append("network_name is required")
    if not isinstance(station.get("channel_number"), int):
        problems.append("channel_number is required and must be an integer")

    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    for day in days:
        if day not in station:
            problems.append(f"missing day: {day} (an absent day is off-air)")

    templates = station.get("day_templates") or {}
    for day in days:
        value = station.get(day)
        if isinstance(value, str) and value not in templates:
            problems.append(f"{day} references undefined day_template '{value}'")
        elif isinstance(value, dict):
            for hour in value:
                if not (isinstance(hour, str) and hour.isdigit() and 0 <= int(hour) <= 23):
                    problems.append(f"{day} has invalid hour key {hour!r}; expected '0'-'23'")

    for name, template in templates.items():
        if not isinstance(template, dict):
            problems.append(f"day_template '{name}' must be an object")
            continue
        for hour, slot in template.items():
            if not (isinstance(hour, str) and hour.isdigit() and 0 <= int(hour) <= 23):
                problems.append(f"day_template '{name}' has invalid hour key {hour!r}")
            if isinstance(slot, dict) and "tags" not in slot and "event" not in slot:
                problems.append(f"day_template '{name}' hour {hour} has neither tags nor event")
    return problems


def validate_export(root: Path) -> dict[str, list[str]]:
    """Validate every config in an export, and that its tags exist on disk."""
    from ..util.atomic import read_json

    results: dict[str, list[str]] = {}
    confs = root / "confs"
    if not confs.is_dir():
        return {"__export__": ["no confs/ directory"]}

    for path in sorted(confs.glob("*.json")):
        if path.name.startswith("._"):
            continue  # macOS AppleDouble sidecar, not a config
        problems = []
        try:
            conf = read_json(path)
        except (OSError, ValueError) as exc:
            results[path.name] = [f"unreadable: {exc}"]
            continue

        problems.extend(validate_station_conf(conf))
        station = conf.get("station_conf") or {}
        content_dir = Path(str(station.get("content_dir", "")))
        if not content_dir.is_absolute():
            # Relative paths are resolved by FieldStation42 against its own
            # root; here the equivalent anchor is the export directory.
            content_dir = root / content_dir

        # Every tag the schedule names must be a real directory with media in
        # it, or the channel goes off-air at that hour with no explanation.
        for template in (station.get("day_templates") or {}).values():
            for hour, slot in (template or {}).items():
                tag = (slot or {}).get("tags")
                if not isinstance(tag, str):
                    continue
                directory = content_dir / tag
                if not directory.is_dir():
                    problems.append(f"hour {hour}: tag '{tag}' has no directory")
                elif not any(p.is_file() for p in directory.rglob("*")):
                    problems.append(f"hour {hour}: tag '{tag}' directory is empty")
        results[path.name] = problems
    return results
