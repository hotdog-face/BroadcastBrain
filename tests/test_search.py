"""Search: the natural-language queries from the brief must actually work."""

from __future__ import annotations

from broadcastbrain.search.query import parse, to_fts_expression
from broadcastbrain.types import Category, EntityKind


class TestBriefExamples:
    """Every example query the project brief asked to be answerable."""

    def test_find_every_taco_bell_commercial(self) -> None:
        parsed = parse("Find every Taco Bell commercial")
        assert parsed.category is Category.COMMERCIAL
        assert (EntityKind.BRAND, "Taco Bell") in parsed.entities

    def test_find_all_mtv_clips_from_2004(self) -> None:
        parsed = parse("Find all MTV clips from 2004")
        assert parsed.network == "MTV"
        assert parsed.years == [2004]

    def test_find_every_appearance_of_billy_mays(self) -> None:
        parsed = parse("Find every appearance of Billy Mays")
        assert (EntityKind.PERSON, "Billy Mays") in parsed.entities

    def test_find_every_discovery_bumper(self) -> None:
        parsed = parse("Find every Discovery bumper")
        assert parsed.category is Category.NETWORK_BUMPER
        assert parsed.network == "Discovery Channel"

    def test_find_all_xbox_commercials(self) -> None:
        parsed = parse("Find all Xbox commercials")
        assert parsed.category is Category.COMMERCIAL
        assert (EntityKind.PRODUCT, "Xbox") in parsed.entities

    def test_find_every_clip_mentioning_star_wars(self) -> None:
        parsed = parse("Find every clip mentioning Star Wars")
        assert "star wars" in parsed.text.lower()

    def test_find_music_videos_by_green_day(self) -> None:
        parsed = parse("Find music videos by Green Day")
        assert parsed.category is Category.MUSIC_VIDEO
        assert any(name == "Green Day" for _kind, name in parsed.entities)

    def test_find_all_clips_similar_to_this_one(self) -> None:
        parsed = parse("find all clips similar to 3f2a91bc-1111-2222-3333-444455556666")
        assert parsed.similar_to_clip_id is not None


class TestParsing:
    def test_decade_becomes_a_range(self) -> None:
        assert parse("promos from the 2000s").year_range == (2000, 2009)

    def test_explicit_year_range(self) -> None:
        assert parse("clips from 2002 to 2006").year_range == (2002, 2006)

    def test_duration_filters(self) -> None:
        assert parse("commercials under 20 seconds").max_duration_s == 20.0
        assert parse("shows over 30 minutes").min_duration_s == 1800.0

    def test_longest_category_phrase_wins(self) -> None:
        assert parse("music videos").category is Category.MUSIC_VIDEO

    def test_filler_words_are_dropped(self) -> None:
        assert parse("find me all the clips").text == ""

    def test_empty_query_is_harmless(self) -> None:
        parsed = parse("")
        assert parsed.has_filters is False
        assert parsed.text == ""

    def test_unknown_terms_survive_as_free_text(self) -> None:
        """Anything the rules do not understand still reaches the index."""
        parsed = parse("clips about zeppelins")
        assert "zeppelins" in parsed.text

    def test_describe_is_human_readable(self) -> None:
        assert "commercial" in parse("Taco Bell commercials").describe()


class TestFtsExpression:
    def test_terms_are_prefix_matched(self) -> None:
        assert to_fts_expression("taco bell") == "taco* OR bell*"

    def test_empty_input(self) -> None:
        assert to_fts_expression("   ") == ""

    def test_punctuation_is_quoted(self) -> None:
        """FTS5 would otherwise treat punctuation as syntax and error."""
        expression = to_fts_expression("1-800")
        assert expression.startswith('"')


class TestSearchService:
    """The service against a real (empty) database."""

    def test_search_on_an_empty_library_returns_nothing_gracefully(self, settings, db) -> None:
        from broadcastbrain.search.service import SearchService

        response = SearchService(settings, db).search("Taco Bell commercials")
        assert response.hits == []
        assert "commercial" in response.explanation

    def test_similar_without_an_embedding_explains_itself(self, settings, db) -> None:
        from broadcastbrain.search.service import SearchService

        response = SearchService(settings, db).similar("does-not-exist")
        assert response.hits == []
        assert "reanalyze" in response.explanation

    def test_facets_work_on_an_empty_library(self, settings, db) -> None:
        from broadcastbrain.search.service import SearchService

        facets = SearchService(settings, db).facets()
        assert "categories" in facets
        assert Category.COMMERCIAL.value in facets["valid_categories"]
