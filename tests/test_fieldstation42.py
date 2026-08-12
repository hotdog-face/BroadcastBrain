"""The FieldStation42 adapter: does it produce something FS42 can actually load?

FieldStation42 is directory-driven -- subdirectory names *are* the scheduler's
tags -- so the tests that matter are about the shape of the tree and the
agreement between the config and what is on disk, not about a document format.

Schema per https://fieldstation42.com/docs/reference/station-config/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from broadcastbrain.config import Settings
from broadcastbrain.db.base import Database
from broadcastbrain.db.models import Clip, Source
from broadcastbrain.export.fieldstation42 import (
    FS42_BUMP_DIR,
    FS42_COMMERCIAL_DIR,
    Fieldstation42Exporter,
    validate_export,
    validate_station_conf,
)
from broadcastbrain.types import Category, ClipStatus

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@pytest.fixture()
def library(settings: Settings, db: Database) -> Settings:
    """A small but realistic library: one network, shows, ads and bumps."""
    media_root = settings.paths.require("library") / "media"
    media_root.mkdir(parents=True, exist_ok=True)

    def make(clip_id: str, category: Category, duration: float, **kwargs: object) -> Clip:
        path = media_root / f"{clip_id}.mkv"
        path.write_bytes(b"\x00" * 2048)
        return Clip(
            id=clip_id, source_id="s1", seq=int(clip_id[-2:]),
            start_s=0.0, end_s=duration, duration_s=duration,
            category=category.value, category_confidence=0.8,
            status=ClipStatus.READY.value, file_path=str(path),
            **kwargs,  # type: ignore[arg-type]
        )

    with db.write_session() as session:
        session.add(Source(id="s1", path="/a", original_path="/a", filename="a", size_bytes=1))
        session.flush()
        session.add_all(
            [
                make("ep01", Category.TV_EPISODE, 1300, network="Cartoon Network",
                     series="Dragon Ball Z"),
                make("ep02", Category.TV_EPISODE, 1250, network="Cartoon Network",
                     series="Dragon Ball Z"),
                make("ep03", Category.TV_EPISODE, 1400, network="Cartoon Network",
                     series="Samurai Jack"),
                make("ad04", Category.COMMERCIAL, 30, network="Cartoon Network"),
                make("ad05", Category.COMMERCIAL, 15, network="Cartoon Network"),
                make("bm06", Category.NETWORK_BUMPER, 5, network="Cartoon Network"),
                make("id07", Category.STATION_ID, 4, network="Cartoon Network"),
                make("un08", Category.UNKNOWN, 200, network="Cartoon Network"),
                make("ad09", Category.COMMERCIAL, 30),  # no network
            ]
        )
    return settings


class TestContentTree:
    def test_tags_become_directories(self, library: Settings, db: Database) -> None:
        result = Fieldstation42Exporter(library, db).export()
        station = next(s for s in result.stations if s.network == "Cartoon Network")

        assert (station.content_dir / "shows" / "dragon-ball-z").is_dir()
        assert (station.content_dir / "shows" / "samurai-jack").is_dir()
        assert (station.content_dir / FS42_COMMERCIAL_DIR).is_dir()
        assert (station.content_dir / FS42_BUMP_DIR).is_dir()

    def test_special_directories_use_fieldstations_names(
        self, library: Settings, db: Database
    ) -> None:
        """FS42 looks for 'commercial' and 'bump' specifically."""
        result = Fieldstation42Exporter(library, db).export()
        station = result.stations[0]
        assert FS42_COMMERCIAL_DIR == "commercial"
        assert FS42_BUMP_DIR == "bump"
        assert any(station.content_dir.rglob("commercial/*"))

    def test_bumps_pool_ids_bumpers_and_promos_together(
        self, library: Settings, db: Database
    ) -> None:
        result = Fieldstation42Exporter(library, db).export()
        station = next(s for s in result.stations if s.network == "Cartoon Network")
        assert len(list((station.content_dir / FS42_BUMP_DIR).glob("*.mkv"))) == 2

    def test_media_is_hardlinked_not_copied(self, library: Settings, db: Database) -> None:
        """A 100 TB library must not be duplicated in order to be scheduled."""
        result = Fieldstation42Exporter(library, db).export()
        assert result.files_linked > 0
        assert result.files_copied == 0

        station = result.stations[0]
        linked = next(station.content_dir.rglob("*.mkv"))
        assert linked.stat().st_nlink >= 2, "file should share an inode with the library"

    def test_unknown_clips_are_catalogued_but_not_scheduled(
        self, library: Settings, db: Database
    ) -> None:
        """Nothing is discarded, but nothing unclassified goes to air either."""
        result = Fieldstation42Exporter(library, db).export()
        station = next(s for s in result.stations if s.network == "Cartoon Network")

        assert (station.content_dir / "unsorted").is_dir()
        conf = json.loads(
            (result.root / "confs" / f"{station.slug}.json").read_text()
        )["station_conf"]
        scheduled = {
            slot["tags"]
            for template in conf["day_templates"].values()
            for slot in template.values()
        }
        assert "unsorted" not in scheduled

    def test_clips_without_a_network_get_their_own_channel(
        self, library: Settings, db: Database
    ) -> None:
        result = Fieldstation42Exporter(library, db).export()
        assert any(s.network == "General" for s in result.stations)

    def test_export_is_regenerated_not_patched(self, library: Settings, db: Database) -> None:
        exporter = Fieldstation42Exporter(library, db)
        exporter.export()
        stale = exporter.root / "catalog" / "cartoon-network" / "shows" / "stale.mkv"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"x")

        exporter = Fieldstation42Exporter(library, db)
        exporter.export()
        assert not stale.exists(), "a rebuild must not leave orphaned media behind"


class TestStationConfig:
    def _conf(self, library: Settings, db: Database) -> dict:
        result = Fieldstation42Exporter(library, db).export()
        station = next(s for s in result.stations if s.network == "Cartoon Network")
        return json.loads((result.root / "confs" / f"{station.slug}.json").read_text())

    def test_has_the_required_wrapper_and_fields(self, library: Settings, db: Database) -> None:
        conf = self._conf(library, db)
        assert "station_conf" in conf
        station = conf["station_conf"]
        assert station["network_name"] == "Cartoon Network"
        assert isinstance(station["channel_number"], int)

    def test_all_seven_days_are_defined(self, library: Settings, db: Database) -> None:
        """An omitted day is off-air, which would be a silent dead channel."""
        station = self._conf(library, db)["station_conf"]
        for day in DAYS:
            assert day in station, f"{day} missing -- the channel would be off-air"

    def test_hour_keys_are_strings_zero_to_twentythree(
        self, library: Settings, db: Database
    ) -> None:
        station = self._conf(library, db)["station_conf"]
        for template in station["day_templates"].values():
            for hour in template:
                assert isinstance(hour, str)
                assert 0 <= int(hour) <= 23

    def test_every_scheduled_tag_exists_on_disk(self, library: Settings, db: Database) -> None:
        """The failure this prevents: a channel off-air at 3pm with no reason given."""
        result = Fieldstation42Exporter(library, db).export()
        station = next(s for s in result.stations if s.network == "Cartoon Network")
        conf = json.loads(
            (result.root / "confs" / f"{station.slug}.json").read_text()
        )["station_conf"]

        for template in conf["day_templates"].values():
            for slot in template.values():
                directory = Path(conf["content_dir"]) / slot["tags"]
                assert directory.is_dir(), f"tag {slot['tags']} has no directory"
                assert any(directory.rglob("*.mkv")), f"tag {slot['tags']} is empty"

    def test_commercial_free_is_set_when_there_are_no_ads(
        self, settings: Settings, db: Database
    ) -> None:
        media = settings.paths.require("library") / "only.mkv"
        media.write_bytes(b"\x00" * 2048)
        with db.write_session() as session:
            session.add(Source(id="s2", path="/b", original_path="/b", filename="b", size_bytes=1))
            session.flush()
            session.add(
                Clip(
                    id="solo", source_id="s2", seq=0, start_s=0, end_s=1300, duration_s=1300,
                    category=Category.TV_EPISODE.value, status=ClipStatus.READY.value,
                    file_path=str(media), network="Lonely TV",
                )
            )
        result = Fieldstation42Exporter(settings, db).export()
        station = next(s for s in result.stations if s.network == "Lonely TV")
        conf = json.loads(
            (result.root / "confs" / f"{station.slug}.json").read_text()
        )["station_conf"]
        assert conf["commercial_free"] is True
        assert any("no commercials" in w for w in station.warnings)

    def test_channel_numbers_are_unique(self, library: Settings, db: Database) -> None:
        result = Fieldstation42Exporter(library, db).export()
        numbers = [s.channel_number for s in result.stations]
        assert len(numbers) == len(set(numbers))


class TestValidation:
    def test_a_generated_export_validates(self, library: Settings, db: Database) -> None:
        exporter = Fieldstation42Exporter(library, db)
        exporter.export()
        problems = validate_export(exporter.root)
        assert problems, "validation should have inspected at least one config"
        for name, issues in problems.items():
            assert issues == [], f"{name}: {issues}"

    def test_missing_day_is_caught(self) -> None:
        conf = {"station_conf": {"network_name": "X", "channel_number": 3, "monday": {}}}
        problems = validate_station_conf(conf)
        assert any("tuesday" in p for p in problems)

    def test_missing_required_fields_are_caught(self) -> None:
        problems = validate_station_conf({"station_conf": {}})
        assert any("network_name" in p for p in problems)
        assert any("channel_number" in p for p in problems)

    def test_undefined_day_template_reference_is_caught(self) -> None:
        conf = {
            "station_conf": {
                "network_name": "X", "channel_number": 3,
                "day_templates": {"weekday": {"8": {"tags": "shows"}}},
                **dict.fromkeys(DAYS, "nonexistent"),
            }
        }
        assert any("undefined day_template" in p for p in validate_station_conf(conf))

    def test_bad_hour_key_is_caught(self) -> None:
        conf = {
            "station_conf": {
                "network_name": "X", "channel_number": 3,
                "day_templates": {"d": {"25": {"tags": "shows"}}},
                **dict.fromkeys(DAYS, "d"),
            }
        }
        assert any("invalid hour key" in p for p in validate_station_conf(conf))

    def test_missing_wrapper_is_caught(self) -> None:
        assert validate_station_conf({"network_name": "X"}) == [
            "missing top-level 'station_conf' object"
        ]


class TestRemovableMedia:
    """Exporting to an SD card is the normal way this reaches another machine.

    Both behaviours here were found doing exactly that: exFAT has no hardlinks
    and no extended attributes, so macOS spills an AppleDouble "._name" sidecar
    beside every file. FieldStation42 scans directories for content, so those
    sidecars would be catalogued as four-kilobyte programmes.
    """

    def test_apple_double_sidecars_are_removed(
        self, library: Settings, db: Database, tmp_path: Path
    ) -> None:
        exporter = Fieldstation42Exporter(library, db, output_root=tmp_path / "card")
        exporter.export()
        # Simulate what the filesystem does, then confirm a rebuild sweeps it.
        sidecar = exporter.root / "confs" / "._cartoon-network.json"
        sidecar.write_bytes(b"\x00\x05\x16\x07")
        exporter = Fieldstation42Exporter(library, db, output_root=tmp_path / "card")
        exporter.export()
        assert not sidecar.exists()

    def test_validator_ignores_sidecars(
        self, library: Settings, db: Database, tmp_path: Path
    ) -> None:
        exporter = Fieldstation42Exporter(library, db, output_root=tmp_path / "card")
        exporter.export()
        (exporter.root / "confs" / "._ghost.json").write_bytes(b"\xb0\xff not json")

        problems = validate_export(exporter.root)
        assert "._ghost.json" not in problems, "a sidecar was mistaken for a config"
        assert all(issues == [] for issues in problems.values())

    def test_content_root_makes_paths_portable(
        self, library: Settings, db: Database, tmp_path: Path
    ) -> None:
        """An absolute path to this Mac is useless on the target machine."""
        exporter = Fieldstation42Exporter(
            library, db, output_root=tmp_path / "card", content_root="catalog"
        )
        result = exporter.export()
        station = next(s for s in result.stations if s.network == "Cartoon Network")
        conf = json.loads(
            (result.root / "confs" / f"{station.slug}.json").read_text()
        )["station_conf"]

        assert conf["content_dir"] == "catalog/cartoon-network"
        assert not Path(conf["content_dir"]).is_absolute()

    def test_relative_paths_still_validate(
        self, library: Settings, db: Database, tmp_path: Path
    ) -> None:
        exporter = Fieldstation42Exporter(
            library, db, output_root=tmp_path / "card", content_root="catalog"
        )
        exporter.export()
        for name, issues in validate_export(exporter.root).items():
            assert issues == [], f"{name}: {issues}"

    def test_export_to_an_alternate_root(
        self, library: Settings, db: Database, tmp_path: Path
    ) -> None:
        target = tmp_path / "elsewhere"
        result = Fieldstation42Exporter(library, db, output_root=target).export()
        assert result.root == target
        assert (target / "confs").is_dir()
        assert (target / "catalog").is_dir()

    def test_a_vanishing_file_does_not_abort_the_rebuild(
        self, library: Settings, db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """exFAT deletes a file's AppleDouble sidecar along with the file.

        `rmtree`'s directory listing then goes stale and it raises
        FileNotFoundError halfway through, leaving a partial export behind.
        """
        import shutil as shutil_module

        exporter = Fieldstation42Exporter(library, db, output_root=tmp_path / "card")
        exporter.export()

        real_rmtree = shutil_module.rmtree
        calls = {"n": 0}

        def flaky(path, ignore_errors=False, onerror=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if onerror is not None:
                onerror(os.unlink, str(path) + "/._gone.json",
                        (FileNotFoundError, FileNotFoundError(2, "No such file"), None))
            return real_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)

        monkeypatch.setattr(shutil_module, "rmtree", flaky)
        exporter = Fieldstation42Exporter(library, db, output_root=tmp_path / "card")
        result = exporter.export()  # must not raise

        assert calls["n"] == 1
        assert len(result.stations) > 0

    def test_an_incomplete_export_is_raised_not_reported(
        self, library: Settings, db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A half-written card must fail loudly, not look finished."""
        from broadcastbrain.errors import ExportError

        exporter = Fieldstation42Exporter(library, db, output_root=tmp_path / "card")
        original = exporter._write_station_conf
        written = {"n": 0}

        def skip_one(plan):  # type: ignore[no-untyped-def]
            written["n"] += 1
            if written["n"] == 1:
                return  # simulate a config that silently failed to land
            original(plan)

        monkeypatch.setattr(exporter, "_write_station_conf", skip_one)
        with pytest.raises(ExportError, match="incomplete"):
            exporter.export()

    def test_filenames_are_ascii_on_removable_media(
        self, library: Settings, db: Database, tmp_path: Path
    ) -> None:
        """A non-ASCII name forces an exFAT extended attribute, which macOS
        materializes as an undeletable "._" sidecar that FieldStation42 would
        catalog as a four-kilobyte programme. Clip titles come from OCR, so
        stray accented characters are routine."""
        from broadcastbrain.db.models import Clip, Source
        from broadcastbrain.export.fieldstation42 import _ascii_name

        assert _ascii_name("écc,.com__ABC__1234.mp4") == "ecc,.com__ABC__1234.mp4"
        assert _ascii_name("DOUDI Lé UP__2010__ab.mp4") == "DOUDI Le UP__2010__ab.mp4"

        media = library.paths.require("library") / "média-ü.mkv"
        media.write_bytes(b"\x00" * 2048)
        with db.write_session() as session:
            session.add(Source(id="s9", path="/z", original_path="/z", filename="z", size_bytes=1))
            session.flush()
            session.add(Clip(
                id="accent", source_id="s9", seq=0, start_s=0, end_s=30, duration_s=30,
                category=Category.COMMERCIAL.value, status=ClipStatus.READY.value,
                file_path=str(media), network="Accent TV",
            ))

        result = Fieldstation42Exporter(library, db, output_root=tmp_path / "card").export()
        for path in result.root.rglob("*.mkv"):
            assert path.name.isascii(), f"non-ASCII name reached the export: {path.name}"
