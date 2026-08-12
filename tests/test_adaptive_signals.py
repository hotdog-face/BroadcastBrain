"""Adaptive signal weighting, and the source-completion race.

Both behaviours in this file were found by running the pipeline against a real
two-hour off-air anime recording, not by reasoning about it. The numbers in
:class:`TestRealWorldAnimation` are the actual signal counts that recording
produced.
"""

from __future__ import annotations

import threading
from typing import ClassVar

import pytest

from broadcastbrain.constants import SIGNAL_WEIGHTS
from broadcastbrain.segment.fusion import FusionConfig, adapt_weights, fuse
from broadcastbrain.types import Signal, SignalKind


class TestAdaptWeights:
    def test_normal_broadcast_is_left_alone(self) -> None:
        """Live-action rates are plausible, so nothing should change."""
        counts = {"scene": 90, "black": 60, "silence": 70, "aspect_change": 12}
        weights, notes = adapt_weights(dict(SIGNAL_WEIGHTS), counts, duration_s=3600)
        assert notes == []
        assert weights == SIGNAL_WEIGHTS

    def test_runaway_detector_is_scaled_down(self) -> None:
        counts = {"scene": 1200}  # 20/min over an hour
        weights, notes = adapt_weights(dict(SIGNAL_WEIGHTS), counts, duration_s=3600)
        assert weights["scene"] < SIGNAL_WEIGHTS["scene"]
        assert "scene" in notes[0]

    def test_scaling_is_proportional(self) -> None:
        base = dict(SIGNAL_WEIGHTS)
        twice, _ = adapt_weights(base, {"scene": 360}, duration_s=3600)   # 6/min
        tenfold, _ = adapt_weights(base, {"scene": 1800}, duration_s=3600)  # 30/min
        assert twice["scene"] == pytest.approx(SIGNAL_WEIGHTS["scene"] * 0.5)
        assert tenfold["scene"] < twice["scene"]

    def test_a_noisy_detector_is_never_silenced_completely(self) -> None:
        """It should still be able to break a tie."""
        weights, _ = adapt_weights(dict(SIGNAL_WEIGHTS), {"scene": 100_000}, duration_s=3600)
        assert weights["scene"] > 0.0

    def test_black_and_silence_are_never_adapted(self) -> None:
        """They are physical broadcast events, not reactions to content."""
        counts = {"black": 5000, "silence": 5000}
        weights, notes = adapt_weights(dict(SIGNAL_WEIGHTS), counts, duration_s=3600)
        assert weights["black"] == SIGNAL_WEIGHTS["black"]
        assert weights["silence"] == SIGNAL_WEIGHTS["silence"]
        assert notes == []

    def test_zero_duration_is_safe(self) -> None:
        weights, notes = adapt_weights(dict(SIGNAL_WEIGHTS), {"scene": 10}, duration_s=0)
        assert weights == SIGNAL_WEIGHTS
        assert notes == []


class TestRealWorldAnimation:
    """The exact signal counts from a real 122-minute anime broadcast.

    Animation defeats scene, corner-logo and crop detection: hard cel cuts fire
    them constantly, drowning the black frames and silences that mark the
    actual programme and commercial boundaries.
    """

    DURATION_S = 7319.0
    COUNTS: ClassVar[dict[str, int]] = {
        "scene": 2618, "logo_change": 1694, "aspect_change": 329,
        "black": 97, "silence": 143, "freeze": 2,
    }

    def _signals(self) -> list[Signal]:
        """Reconstruct a representative signal set from the observed rates.

        Programme boundaries carry black + silence together; the content-driven
        detectors fire densely and independently all the way through.
        """
        signals: list[Signal] = []
        # Real structure: a programme boundary roughly every 7 minutes.
        for index in range(1, 18):
            t = index * 420.0
            signals.append(Signal(SignalKind.BLACK, t, 0.9, 0.4))
            signals.append(Signal(SignalKind.SILENCE, t + 0.1, 0.8, 0.5))
        # Content noise: scene changes every ~2.8s, logo flicker every ~4.3s.
        step = self.DURATION_S / self.COUNTS["scene"]
        for index in range(self.COUNTS["scene"]):
            signals.append(Signal(SignalKind.SCENE, index * step, 0.5))
        step = self.DURATION_S / self.COUNTS["logo_change"]
        for index in range(self.COUNTS["logo_change"]):
            signals.append(Signal(SignalKind.LOGO_CHANGE, index * step, 0.5))
        step = self.DURATION_S / self.COUNTS["aspect_change"]
        for index in range(self.COUNTS["aspect_change"]):
            signals.append(Signal(SignalKind.ASPECT_CHANGE, index * step, 0.9))
        return signals

    def _config(self, weights: dict[str, float]) -> FusionConfig:
        return FusionConfig(
            cluster_window_s=1.2, min_boundary_score=1.05, min_segment_s=3.0,
            max_segment_s=5400.0, weights=weights, grid_bonus=0.25,
        )

    def test_without_adaptation_the_recording_is_shredded(self) -> None:
        segments, _ = fuse(
            self._signals(), self.DURATION_S, self._config(dict(SIGNAL_WEIGHTS))
        )
        # Content-driven detectors overwhelm the real boundaries.
        assert len(segments) > 150

    def test_adaptation_recovers_the_programme_structure(self) -> None:
        weights, notes = adapt_weights(dict(SIGNAL_WEIGHTS), self.COUNTS, self.DURATION_S)
        assert any("scene" in note for note in notes)
        assert any("logo_change" in note for note in notes)

        segments, _ = fuse(self._signals(), self.DURATION_S, self._config(weights))

        assert len(segments) < 40, "should cut on real boundaries, not on every shot"
        long_segments = [s for s in segments if s.duration_s >= 240]
        assert len(long_segments) >= 8, "programme segments should survive intact"

    def test_aspect_change_alone_never_cuts(self) -> None:
        """Crop detection flickers on animation; it must not cut by itself.

        Measured over a span shorter than ``max_segment_s`` so the oversized
        segment repair -- which deliberately does accept a sub-threshold
        candidate rather than emit one enormous clip -- stays out of the way.
        """
        weights, _ = adapt_weights(dict(SIGNAL_WEIGHTS), self.COUNTS, self.DURATION_S)
        config = self._config(weights)
        lone = [Signal(SignalKind.ASPECT_CHANGE, 500.0, 1.0)]
        segments, boundaries = fuse(lone, 1200.0, config)
        assert boundaries == [], "a lone crop flicker must not clear the bar"
        assert len(segments) == 1


class TestSourceCompletionIsSingleShot:
    """Several workers finish a source's last clips at the same instant.

    Before this was guarded, both archived the file and the second move renamed
    it to ``<name>-2`` by moving it onto itself.
    """

    def test_only_one_worker_archives_the_source(self, settings, db, tmp_path) -> None:
        from broadcastbrain.db.models import Source
        from broadcastbrain.pipeline.discovery import mark_source_complete

        incoming = settings.paths.require("incoming")
        media = incoming / "capture.ts"
        media.write_bytes(b"\x47" * 4096)

        with db.write_session() as session:
            source = Source(
                id="race-test", path=str(media), original_path=str(media),
                filename="capture.ts", size_bytes=4096,
            )
            session.add(source)

        barrier = threading.Barrier(6)

        def finish() -> None:
            barrier.wait()
            mark_source_complete(db, settings, "race-test")

        threads = [threading.Thread(target=finish) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        archived = list((settings.paths.require("processing") / "completed").glob("*"))
        assert len(archived) == 1, f"source archived more than once: {archived}"
        assert archived[0].name == "capture.ts", "no -2 suffix should appear"

    def test_calling_again_after_completion_is_a_no_op(self, settings, db) -> None:
        from broadcastbrain.db.models import Source
        from broadcastbrain.pipeline.discovery import mark_source_complete

        media = settings.paths.require("incoming") / "second.ts"
        media.write_bytes(b"\x47" * 4096)
        with db.write_session() as session:
            session.add(
                Source(
                    id="idempotent", path=str(media), original_path=str(media),
                    filename="second.ts", size_bytes=4096,
                )
            )

        mark_source_complete(db, settings, "idempotent")
        mark_source_complete(db, settings, "idempotent")

        archived = list((settings.paths.require("processing") / "completed").glob("*"))
        assert len(archived) == 1


class TestCorrectionsCanClearAField:
    """A re-run must be able to *retract* a value, not only overwrite it.

    Found when a network attribution survived the removal of the bad marker
    that produced it: the analyzer stopped asserting a network, wrote nothing,
    and the stale value stayed on the clip with its evidence long gone.
    """

    def test_estimation_writes_none_when_it_has_no_answer(self, settings, db) -> None:
        from broadcastbrain.analyze.base import AnalysisContext, ClipView
        from broadcastbrain.analyze.inference import EstimationAnalyzer
        from broadcastbrain.media.ffmpeg import FFmpeg

        ctx = AnalysisContext(
            clip=ClipView(
                id="x", source_id="s", path=settings.paths.require("cache") / "x.mkv",
                start_s=0.0, end_s=30.0, duration_s=30.0,
            ),
            settings=settings,
            ffmpeg=FFmpeg(settings),
            workdir=settings.paths.require("cache"),
        )
        result = EstimationAnalyzer().run(ctx)

        # Present-and-None, not absent: absent would leave a stale value behind.
        assert "network" in result.fields
        assert result.fields["network"] is None
        assert "series" in result.fields
        assert result.fields["series"] is None

    def test_a_stale_value_is_actually_cleared(self, settings, db) -> None:
        from broadcastbrain.db.models import Clip, Source
        from broadcastbrain.db.repo import ClipRepository

        with db.write_session() as session:
            session.add(Source(id="s1", path="/a", original_path="/a", filename="a", size_bytes=1))
            session.flush()
            session.add(
                Clip(
                    id="c1", source_id="s1", seq=0, start_s=0, end_s=30, duration_s=30,
                    network="Adult Swim",
                )
            )

        with db.write_session() as session:
            ClipRepository.update_fields(session, "c1", network=None)

        with db.session() as session:
            assert ClipRepository.get(session, "c1").network is None


class TestYearFromFolder:
    """An operator who files a clip under "2004/" has stated its year.

    This archive arrived as 1,997 commercials in year-named folders -- ground
    truth no analyzer could recover from the pixels.
    """

    def test_exact_year_folder_is_read(self) -> None:
        from broadcastbrain.analyze.inference import _year_from_path

        assert _year_from_path("/media/Commercials/2004/taco-bell.mp4") == 2004

    def test_a_title_containing_digits_is_not_a_year(self) -> None:
        """"Top 2004 Ads Vol 3" is a title, not a date."""
        from broadcastbrain.analyze.inference import _year_from_path

        assert _year_from_path("/media/Top 2004 Ads Vol 3/a.mp4") is None

    def test_the_nearest_folder_wins(self) -> None:
        from broadcastbrain.analyze.inference import _year_from_path

        assert _year_from_path("/media/2004/2005/b.mp4") == 2005

    def test_no_year_anywhere(self) -> None:
        from broadcastbrain.analyze.inference import _year_from_path

        assert _year_from_path("/media/misc/c.mp4") is None
        assert _year_from_path("") is None

    def test_a_year_outside_the_target_era_is_still_honoured(self) -> None:
        """Filing under 1998 means 1998, even when hunting 2002-2006."""
        from broadcastbrain.analyze.inference import _year_from_path

        assert _year_from_path("/media/1998/d.mp4") == 1998


class TestYearIsNeverInvented:
    """Dating must reflect evidence, or record nothing.

    On a real 2,384-clip library the era-midpoint fallback produced 288 clips
    all claiming 2004 -- sitting in the same histogram bucket as the 143 that
    were genuinely filed under a 2004 folder. And the copyright reader, the
    highest-confidence source in the system, accepted "copyright 2804".
    """

    def test_a_copyright_year_must_be_plausible(self) -> None:
        from broadcastbrain.util.text import extract_copyright_year

        assert extract_copyright_year("(c) 2004 Viacom") == 2004
        assert extract_copyright_year("copyright 2804") is None
        assert extract_copyright_year("c 1383") is None
        assert extract_copyright_year("© 2086") is None

    def test_a_plausible_year_later_in_the_text_still_wins_over_junk(self) -> None:
        from broadcastbrain.util.text import extract_copyright_year

        assert extract_copyright_year("c 2804 ... copyright 2003 CBS") == 2003

    def test_no_evidence_yields_no_year(self, settings, db) -> None:
        from broadcastbrain.analyze.base import AnalysisContext, ClipView
        from broadcastbrain.analyze.inference import EstimationAnalyzer
        from broadcastbrain.media.ffmpeg import FFmpeg

        ctx = AnalysisContext(
            clip=ClipView(
                id="x", source_id="s", path=settings.paths.require("cache") / "x.mkv",
                start_s=0.0, end_s=30.0, duration_s=30.0, source_path="/media/misc/x.mkv",
            ),
            settings=settings, ffmpeg=FFmpeg(settings),
            workdir=settings.paths.require("cache"),
        )
        result = EstimationAnalyzer().run(ctx)
        assert result.fields["air_year"] is None
        assert result.fields["air_year_confidence"] == 0.0

    def test_an_implausible_year_is_rejected_whatever_its_source(
        self, settings, db, monkeypatch
    ) -> None:
        from broadcastbrain.analyze.base import AnalysisContext, ClipView
        from broadcastbrain.analyze.inference import Estimate, EstimationAnalyzer
        from broadcastbrain.media.ffmpeg import FFmpeg

        analyzer = EstimationAnalyzer()
        monkeypatch.setattr(
            analyzer, "_estimate_year_inner",
            lambda _ctx, _network: Estimate(2804, 0.9, ["copyright:2804"]),
        )
        ctx = AnalysisContext(
            clip=ClipView(
                id="x", source_id="s", path=settings.paths.require("cache") / "x.mkv",
                start_s=0.0, end_s=30.0, duration_s=30.0,
            ),
            settings=settings, ffmpeg=FFmpeg(settings),
            workdir=settings.paths.require("cache"),
        )
        assert analyzer.run(ctx).fields["air_year"] is None

    def test_a_folder_year_is_still_used(self, settings, db) -> None:
        from broadcastbrain.analyze.base import AnalysisContext, ClipView
        from broadcastbrain.analyze.inference import EstimationAnalyzer
        from broadcastbrain.media.ffmpeg import FFmpeg

        ctx = AnalysisContext(
            clip=ClipView(
                id="x", source_id="s", path=settings.paths.require("cache") / "x.mkv",
                start_s=0.0, end_s=30.0, duration_s=30.0,
                source_path="/media/Commercials/2003/ad.mp4",
            ),
            settings=settings, ffmpeg=FFmpeg(settings),
            workdir=settings.paths.require("cache"),
        )
        result = EstimationAnalyzer().run(ctx)
        assert result.fields["air_year"] == 2003
        assert result.fields["air_year_confidence"] >= 0.8

    def test_a_network_lifetime_does_not_date_a_clip(self, settings, db) -> None:
        """"MTV existed 2002-2006" is a range; 2004 is not its answer."""
        from broadcastbrain.analyze.base import AnalysisContext, ClipView
        from broadcastbrain.analyze.inference import EstimationAnalyzer
        from broadcastbrain.media.ffmpeg import FFmpeg

        ctx = AnalysisContext(
            clip=ClipView(
                id="x", source_id="s", path=settings.paths.require("cache") / "x.mkv",
                start_s=0.0, end_s=30.0, duration_s=30.0,
            ),
            settings=settings, ffmpeg=FFmpeg(settings),
            workdir=settings.paths.require("cache"),
        )
        # OCR saw the network name and nothing else datable.
        ctx.previous["ocr"] = {"text": "you're watching MTV"}

        result = EstimationAnalyzer().run(ctx)
        assert result.fields["network"] == "MTV", "the network itself is still recorded"
        assert result.fields["air_year"] is None, "but it must not imply a year"


class TestShortInterstitials:
    """A station ID is a few seconds of the network's own branding.

    Real WOC ("with original commercials") tape produced 48 such clips sitting
    in Unknown: four to seven seconds, no speech for Whisper to find, the
    network name read off the screen by OCR, and nothing being advertised.
    Duration alone cannot tell one from the tail of a spot.
    """

    def _ctx(self, settings, *, duration: float, text: str):
        from broadcastbrain.analyze.base import AnalysisContext, ClipView
        from broadcastbrain.media.ffmpeg import FFmpeg

        ctx = AnalysisContext(
            clip=ClipView(
                id="x", source_id="s", path=settings.paths.require("cache") / "x.mkv",
                start_s=0.0, end_s=duration, duration_s=duration,
            ),
            settings=settings, ffmpeg=FFmpeg(settings),
            workdir=settings.paths.require("cache"),
        )
        ctx.previous["ocr"] = {"text": text}
        return ctx

    def _decide(self, ctx):
        from broadcastbrain.analyze.inference import EstimationAnalyzer
        from broadcastbrain.categorize.rules import decide

        result = EstimationAnalyzer().run(ctx)
        return decide([(c, w, r) for c, w, r in result.category_votes])

    def test_five_seconds_of_network_branding_is_a_station_id(self, settings, db) -> None:
        from broadcastbrain.types import Category

        ctx = self._ctx(settings, duration=5.0, text="Nickelodeon")
        assert self._decide(ctx).category is Category.STATION_ID

    def test_ten_seconds_of_network_branding_is_a_bumper(self, settings, db) -> None:
        from broadcastbrain.types import Category

        ctx = self._ctx(settings, duration=10.0, text="you're watching Cartoon Network")
        assert self._decide(ctx).category is Category.NETWORK_BUMPER

    def test_a_short_clip_advertising_something_stays_a_commercial(
        self, settings, db
    ) -> None:
        """A brand on screen means an advertisement, however short."""
        from broadcastbrain.types import Category

        ctx = self._ctx(settings, duration=6.0, text="Taco Bell on Nickelodeon")
        assert self._decide(ctx).category is not Category.STATION_ID

    def test_a_short_clip_with_no_branding_is_not_promoted(self, settings, db) -> None:
        """Absence of evidence must not become a station ID either."""
        from broadcastbrain.types import Category

        decision = self._decide(self._ctx(settings, duration=5.0, text=""))
        assert decision.category in (Category.UNKNOWN, Category.STATION_ID)
        if decision.category is Category.STATION_ID:
            assert decision.confidence < 0.5, "bare duration must not be decisive"

    def test_a_thirty_second_branded_clip_is_not_an_interstitial(
        self, settings, db
    ) -> None:
        """Networks advertise themselves at 30 seconds too -- that is a promo."""
        from broadcastbrain.types import Category

        ctx = self._ctx(settings, duration=30.0, text="Nickelodeon")
        assert self._decide(ctx).category not in (
            Category.STATION_ID, Category.NETWORK_BUMPER,
        )
