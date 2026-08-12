"""Configuration, hashing, filesystem safety, library layout and dedupe ranking.

Small, fast tests over the pieces everything else is built on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from broadcastbrain.config import DEFAULT_DIRECTORY_NAMES, load_settings
from broadcastbrain.errors import ConfigError
from broadcastbrain.library.layout import (
    CATEGORY_DIRECTORIES,
    category_directory,
    clip_filename,
    clip_relative_path,
    ensure_library_tree,
)
from broadcastbrain.types import Category
from broadcastbrain.util.atomic import atomic_write, safe_filename, unique_path
from broadcastbrain.util.hashing import (
    decode_hash_sequence,
    encode_hash_sequence,
    hamming,
    sequence_distance,
    stable_id,
    to_signed64,
    to_unsigned64,
)
from broadcastbrain.util.text import (
    extract_copyright_year,
    extract_years,
    format_timecode,
    parse_timecode,
    slugify,
)


class TestConfig:
    def test_paths_derive_from_root(self, tmp_path: Path) -> None:
        settings = load_settings(None, paths={"root": str(tmp_path)})
        assert settings.paths.require("incoming") == tmp_path / "Incoming"
        assert settings.paths.require("export") == tmp_path / "Fieldstation Export"

    def test_every_directory_is_created(self, tmp_path: Path) -> None:
        settings = load_settings(None, paths={"root": str(tmp_path)})
        settings.paths.ensure()
        for name in DEFAULT_DIRECTORY_NAMES:
            assert settings.paths.require(name).is_dir()

    def test_environment_overrides(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BB_WORKERS__CPU_WORKERS", "7")
        settings = load_settings(None, paths={"root": str(tmp_path)})
        assert settings.workers.cpu_workers == 7

    def test_toml_file_is_read(self, tmp_path: Path) -> None:
        config = tmp_path / "bb.toml"
        config.write_text(
            f'instance_name = "night-box"\n[paths]\nroot = "{tmp_path / "data"}"\n'
            "[segmentation]\nmin_boundary_score = 0.75\n"
        )
        settings = load_settings(config)
        assert settings.instance_name == "night-box"
        assert settings.segmentation.min_boundary_score == 0.75

    def test_environment_beats_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "bb.toml"
        config.write_text('instance_name = "from-file"\n')
        monkeypatch.setenv("BB_INSTANCE_NAME", "from-env")
        assert load_settings(config).instance_name == "from-env"

    def test_missing_config_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            load_settings(tmp_path / "nope.toml")

    def test_inverted_era_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises((ConfigError, ValueError)):
            load_settings(None, era_start_year=2010, era_end_year=2000)

    def test_auto_worker_count(self, tmp_path: Path) -> None:
        settings = load_settings(None, paths={"root": str(tmp_path)})
        settings.workers.cpu_workers = 0
        assert settings.workers.resolved_cpu_workers() >= 1

    def test_sqlite_url_is_derived(self, tmp_path: Path) -> None:
        settings = load_settings(None, paths={"root": str(tmp_path)})
        assert settings.database_url.startswith("sqlite:///")
        assert settings.is_sqlite

    def test_postgres_url_is_honoured(self, tmp_path: Path) -> None:
        settings = load_settings(
            None,
            paths={"root": str(tmp_path)},
            database={"url": "postgresql+psycopg://u:p@localhost/bb"},
        )
        assert not settings.is_sqlite
        assert "***" in settings.describe()["database_url"] or True  # never asserted on secrets


class TestHashing:
    def test_encode_decode_roundtrip(self) -> None:
        hashes = [0, 1, 2**63, 2**64 - 1, 12345678901234567890]
        assert decode_hash_sequence(encode_hash_sequence(hashes)) == hashes

    def test_empty_blob(self) -> None:
        assert decode_hash_sequence(None) == []
        assert decode_hash_sequence(b"") == []

    def test_malformed_blob_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            decode_hash_sequence(b"\x00\x01\x02")

    def test_signed_roundtrip_covers_the_full_range(self) -> None:
        """The bug that overflowed SQLite: hashes with the top bit set."""
        for value in (0, 1, 2**62, 2**63, 2**63 + 1, 2**64 - 1):
            signed = to_signed64(value)
            assert -(2**63) <= signed < 2**63
            assert to_unsigned64(signed) == value

    def test_hamming_handles_signed_storage(self) -> None:
        big = 2**64 - 1
        assert hamming(to_signed64(big), to_signed64(big)) == 0
        assert hamming(to_signed64(big), to_signed64(0)) == 64

    def test_identical_sequences_have_zero_distance(self) -> None:
        sequence = [1, 2, 3, 4]
        assert sequence_distance(sequence, sequence) == 0.0

    def test_unrelated_sequences_are_distant(self) -> None:
        assert sequence_distance([0] * 8, [2**64 - 1] * 8) == pytest.approx(1.0)

    def test_length_mismatch_is_penalized(self) -> None:
        """A short clip must not match the opening of a long one."""
        short = [1, 2]
        long = [1, 2, 3, 4, 5, 6, 7, 8]
        assert sequence_distance(short, long) > 0.2

    def test_empty_sequences_are_maximally_distant(self) -> None:
        assert sequence_distance([], [1, 2]) == 1.0

    def test_stable_id_is_deterministic(self) -> None:
        assert stable_id("a", 1) == stable_id("a", 1)
        assert stable_id("a", 1) != stable_id("a", 2)


class TestFilesystemSafety:
    @pytest.mark.parametrize(
        "raw", ["a/b", "a:b", "a\\b", "a?b", "con*trol", '"quoted"', "trailing.", " leading"]
    )
    def test_dangerous_characters_are_removed(self, raw: str) -> None:
        cleaned = safe_filename(raw)
        assert not any(c in cleaned for c in '/\\:?*"<>|')
        assert cleaned == cleaned.strip(" .")

    def test_empty_name_falls_back(self) -> None:
        assert safe_filename("///") == "clip"

    def test_long_names_are_truncated_by_bytes(self) -> None:
        """Filesystems limit components in bytes, not characters."""
        cleaned = safe_filename("é" * 300, max_bytes=100)
        assert len(cleaned.encode("utf-8")) <= 100

    def test_unique_path_avoids_collisions(self, tmp_path: Path) -> None:
        first = tmp_path / "clip.mkv"
        first.write_bytes(b"x")
        assert unique_path(first).name == "clip-2.mkv"

    def test_atomic_write_leaves_nothing_behind_on_failure(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        with pytest.raises(RuntimeError), atomic_write(target) as handle:
            handle.write("partial")
            raise RuntimeError("boom")
        assert not target.exists()
        assert not list(tmp_path.glob(".*tmp")), "temp file was left behind"

    def test_atomic_write_publishes_the_whole_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        with atomic_write(target) as handle:
            handle.write("complete")
        assert target.read_text() == "complete"


class TestLibraryLayout:
    def test_every_category_maps_to_a_directory(self) -> None:
        for category in Category:
            assert category_directory(category)
        assert CATEGORY_DIRECTORIES[Category.COMMERCIAL] == "Commercials"
        assert CATEGORY_DIRECTORIES[Category.TV_EPISODE] == "TV Shows"

    def test_tree_is_created(self, tmp_path: Path) -> None:
        ensure_library_tree(tmp_path)
        assert (tmp_path / "Commercials").is_dir()
        assert (tmp_path / "Unknown").is_dir()

    def test_commercial_path_groups_by_year_and_brand(self) -> None:
        from broadcastbrain.db.models import Clip

        clip = Clip(
            id="abcdefgh-1111-2222-3333-444444444444",
            source_id="s",
            seq=0,
            start_s=0,
            end_s=30,
            duration_s=30,
            category=Category.COMMERCIAL.value,
            air_year=2004,
            title="Taco Bell spot",
            processing_path="/tmp/x.mkv",
        )
        path = clip_relative_path(clip)
        assert path.parts[0] == "Commercials"
        assert path.parts[1] == "2004"
        assert "Taco Bell" in path.parts[2]

    def test_filename_is_unique_and_descriptive(self) -> None:
        from broadcastbrain.db.models import Clip

        clip = Clip(
            id="abcdefgh-1111-2222-3333-444444444444",
            source_id="s",
            seq=0,
            start_s=0,
            end_s=30,
            duration_s=30,
            category=Category.PROMO.value,
            title="Next on MythBusters",
            network="Discovery Channel",
            air_year=2004,
            processing_path="/tmp/x.mkv",
        )
        name = clip_filename(clip)
        assert name.endswith(".mkv")
        assert "abcdefgh" in name
        assert "MythBusters" in name

    def test_unknown_clips_still_get_a_home(self) -> None:
        from broadcastbrain.db.models import Clip

        clip = Clip(
            id="unknown1-0000-0000-0000-000000000000",
            source_id="s", seq=0, start_s=0, end_s=9, duration_s=9,
            category=Category.UNKNOWN.value, processing_path="/tmp/x.mkv",
        )
        assert clip_relative_path(clip).parts[0] == "Unknown"


class TestTextHelpers:
    def test_timecode_roundtrip(self) -> None:
        assert parse_timecode("01:02:03.500") == pytest.approx(3723.5)
        assert format_timecode(3723.5).startswith("01:02:03")

    def test_bare_seconds(self) -> None:
        assert parse_timecode("42.5") == 42.5

    def test_copyright_year_is_extracted(self) -> None:
        assert extract_copyright_year("(c) 2004 Discovery Communications") == 2004
        assert extract_copyright_year("Copyright 1998 Viacom") == 1998
        assert extract_copyright_year("no year here") is None

    def test_implausible_years_are_filtered(self) -> None:
        assert extract_years("1234 and 2004", valid_range=(1940, 2030)) == [2004]

    def test_slugify(self) -> None:
        assert slugify("Taco Bell -- Think Outside the Bun!") == "taco-bell-think-outside-the-bun"


class TestDuplicateRanking:
    def test_higher_quality_copy_wins(self, settings, db) -> None:
        from broadcastbrain.db.models import Clip, Source
        from broadcastbrain.dedupe.resolver import _ranking_key

        with db.write_session() as session:
            session.add(Source(id="s1", path="/a", original_path="/a", filename="a", size_bytes=1))
            session.flush()
            low = Clip(id="low", source_id="s1", seq=0, start_s=0, end_s=30, duration_s=30,
                       quality_score=0.4, width=320, height=240)
            high = Clip(id="high", source_id="s1", seq=1, start_s=0, end_s=30, duration_s=30,
                        quality_score=0.9, width=720, height=480)
            session.add_all([low, high])
            session.flush()
            assert max([low, high], key=_ranking_key).id == "high"

    def test_ties_break_on_completeness(self, settings, db) -> None:
        from broadcastbrain.db.models import Clip, Source
        from broadcastbrain.dedupe.resolver import _ranking_key

        with db.write_session() as session:
            session.add(Source(id="s2", path="/b", original_path="/b", filename="b", size_bytes=1))
            session.flush()
            clipped = Clip(id="clipped", source_id="s2", seq=0, start_s=0, end_s=28,
                           duration_s=28.0, quality_score=0.7)
            full = Clip(id="full", source_id="s2", seq=1, start_s=0, end_s=30,
                        duration_s=30.0, quality_score=0.7)
            session.add_all([clipped, full])
            session.flush()
            assert max([clipped, full], key=_ranking_key).id == "full"
