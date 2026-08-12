"""Era knowledge base: the 2002-2006 American cable dial.

This module is what makes searches like "every Taco Bell commercial" or "all
MTV clips from 2004" work on day one, without an LLM and without labelled
training data. OCR and speech produce text; this maps that text onto the
networks, brands, shows and people that actually populated basic cable in this
window, with an explicit confidence for each match.

It is plain data on purpose. Adding a network or a brand is a one-line edit
that immediately improves every future clip -- and, because analyzer versions
drive re-analysis, bumping :data:`KNOWLEDGE_VERSION` re-labels the existing
library too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..types import EntityKind
from ..util.text import canonical_key

#: Bump when the tables below change materially. The inference analyzer folds
#: this into its version so the whole library gets re-labelled.
KNOWLEDGE_VERSION = "5"


@dataclass(frozen=True)
class NetworkProfile:
    """One television network and how to recognize it."""

    name: str
    #: Strings that, seen in OCR or heard in speech, indicate this network.
    #:
    #: These are matched against lowercased speech as well as on-screen text,
    #: so a marker that is also an ordinary English word is a false-positive
    #: machine. "as" for Adult Swim matched almost every transcript in a real
    #: library and made it the most-detected network on the dial. Two- and
    #: three-letter abbreviations belong here only when they are not words.
    markers: tuple[str, ...]
    #: On-air slogans and ident lines -- very high confidence when matched.
    slogans: tuple[str, ...] = ()
    #: Programming genres this network is known for, used for genre priors.
    genres: tuple[str, ...] = ()
    #: Shows strongly associated with the network.
    shows: tuple[str, ...] = ()
    #: Years the network carried this identity (helps date a clip).
    era: tuple[int, int] = (1990, 2010)


NETWORKS: tuple[NetworkProfile, ...] = (
    NetworkProfile(
        "MTV",
        ("mtv", "music television", "total request live", "trl"),
        ("i want my mtv",),
        ("music", "reality", "youth"),
        ("TRL", "Pimp My Ride", "Punk'd", "The Real World", "Cribs", "Jackass"),
        (1981, 2010),
    ),
    NetworkProfile(
        "VH1",
        ("vh1", "vh-1", "video hits one"),
        ("music first",),
        ("music", "nostalgia", "countdown"),
        ("Behind the Music", "I Love the 80s", "Best Week Ever", "Pop-Up Video"),
    ),
    NetworkProfile(
        "Discovery Channel",
        ("discovery channel", "discovery", "dsc"),
        ("explore your world",),
        ("documentary", "science", "reality"),
        ("MythBusters", "Dirty Jobs", "American Chopper", "Deadliest Catch", "Monster Garage"),
    ),
    NetworkProfile(
        "TLC",
        ("tlc", "the learning channel"),
        (),
        ("documentary", "lifestyle", "reality"),
        ("Trading Spaces", "What Not to Wear", "A Baby Story", "Junkyard Wars"),
    ),
    NetworkProfile(
        "Cartoon Network",
        ("cartoon network", "toonami", "cartoonnetwork.com"),
        ("the best reason to get out of bed",),
        ("animation", "kids"),
        (
            "Dexter's Laboratory", "Powerpuff Girls", "Samurai Jack", "Teen Titans",
            "Ed Edd n Eddy", "Courage the Cowardly Dog", "Johnny Bravo",
            "Codename: Kids Next Door", "Foster's Home for Imaginary Friends",
            # Toonami -- Cartoon Network's action block, and where imported
            # anime aired in this era. These are what an off-air Cartoon
            # Network tape is most likely to actually contain.
            "Dragon Ball Z", "Dragon Ball GT", "Dragon Ball", "Gundam Wing",
            "Sailor Moon", "Yu Yu Hakusho", "Rurouni Kenshin", "Tenchi Muyo",
            "Naruto", "One Piece", "Megas XLR", "IGPX", "Zatch Bell",
            "Bobobo-bo Bo-bobo", "Cardcaptors", "Outlaw Star", "Big O",
            "Batman Beyond", "Justice League", "Static Shock", "ThunderCats",
        ),
    ),
    NetworkProfile(
        "Adult Swim",
        ("adult swim", "[adult swim]"),
        (),
        ("animation", "comedy", "late night"),
        ("Aqua Teen Hunger Force", "Family Guy", "Futurama", "Sealab 2021", "Robot Chicken"),
        (2001, 2010),
    ),
    NetworkProfile(
        "Comedy Central",
        ("comedy central", "comedycentral"),
        (),
        ("comedy", "animation", "news satire"),
        ("South Park", "The Daily Show", "Chappelle's Show", "Reno 911", "Crank Yankers"),
    ),
    NetworkProfile(
        "Nickelodeon",
        ("nickelodeon", "nicktoons", "nick jr", "nick.com"),
        ("you're watching nickelodeon", "nick is kids"),
        ("kids", "animation"),
        (
            # Nicktoons
            "SpongeBob SquarePants", "The Fairly OddParents", "Rugrats",
            "Hey Arnold!", "The Wild Thornberrys", "Rocket Power",
            "As Told by Ginger", "Danny Phantom", "Invader Zim", "CatDog",
            "ChalkZone", "My Life as a Teenage Robot", "All Grown Up!",
            "The Adventures of Jimmy Neutron", "Avatar: The Last Airbender",
            "Catscratch", "The X's", "Tak and the Power of Juju",
            # Live action
            "Drake & Josh", "Zoey 101", "Ned's Declassified School Survival Guide",
            "The Amanda Show", "All That", "Kenan & Kel", "Unfabulous",
            "Romeo!", "Taina", "The Brothers Garcia",
            # Nick Jr / preschool
            "Dora the Explorer", "Blue's Clues", "Little Bill", "Oswald",
            "LazyTown", "The Backyardigans", "Go, Diego, Go!",
        ),
    ),
    NetworkProfile(
        "ESPN",
        ("espn", "espn2", "sportscenter"),
        ("the worldwide leader in sports",),
        ("sports",),
        ("SportsCenter", "Pardon the Interruption", "Baseball Tonight"),
    ),
    NetworkProfile(
        "CNN",
        ("cnn", "cnn headline news", "headline news"),
        ("the most trusted name in news",),
        ("news",),
        ("Larry King Live", "Anderson Cooper 360"),
    ),
    NetworkProfile(
        "Fox News",
        ("fox news", "foxnews", "fox news channel"),
        ("fair and balanced", "we report you decide"),
        ("news",),
        ("The O'Reilly Factor", "Hannity & Colmes"),
    ),
    NetworkProfile(
        "TNT",
        ("tnt", "turner network television"),
        ("we know drama",),
        ("drama", "movies", "sports"),
        ("Law & Order", "Charmed", "NBA on TNT"),
    ),
    NetworkProfile(
        "TBS",
        ("tbs", "superstation"),
        ("very funny",),
        ("comedy", "movies", "sports"),
        ("Friends", "Seinfeld", "Braves baseball"),
    ),
    NetworkProfile(
        "USA Network",
        ("usa network", "usanetwork.com"),
        ("characters welcome",),
        ("drama", "wrestling", "movies"),
        (
            "Monk", "Law & Order: SVU", "Law & Order: Criminal Intent",
            "The Dead Zone", "The 4400", "Psych", "Kojak", "Touching Evil",
            "Peacemakers", "Nash Bridges", "Walker, Texas Ranger", "JAG",
            "WWE Raw", "WWE SmackDown", "Silk Stalkings", "Strike Force",
        ),
        (1980, 2010),
    ),
    NetworkProfile(
        "SciFi Channel",
        ("sci fi channel", "scifi", "sci-fi channel"),
        (),
        ("science fiction", "drama"),
        ("Stargate SG-1", "Battlestar Galactica", "Ghost Hunters"),
    ),
    NetworkProfile(
        "History Channel",
        ("history channel", "the history channel", "hist"),
        ("where the past comes alive",),
        ("documentary", "history"),
        ("Modern Marvels", "Mail Call", "UFO Files"),
    ),
    NetworkProfile(
        "A&E",
        ("a&e", "arts and entertainment", "a and e"),
        (),
        ("documentary", "drama", "reality"),
        ("Biography", "Cold Case Files", "Dog the Bounty Hunter"),
    ),
    NetworkProfile(
        "Food Network",
        ("food network", "foodtv"),
        (),
        ("cooking", "lifestyle"),
        ("Iron Chef America", "Good Eats", "Emeril Live", "Unwrapped"),
    ),
    NetworkProfile(
        "HGTV",
        ("hgtv", "home and garden television"),
        ("start at home",),
        ("lifestyle", "home"),
        ("House Hunters", "Designed to Sell", "Curb Appeal"),
    ),
    NetworkProfile(
        "Spike TV",
        ("spike tv", "spike", "tnn"),
        ("the first network for men",),
        ("men", "action", "wrestling"),
        ("CSI", "Trucks!", "MXC", "Star Trek: The Next Generation"),
        (2003, 2010),
    ),
    NetworkProfile(
        "Nick at Nite",
        ("nick at nite",),
        (),
        ("classic sitcom",),
        ("The Cosby Show", "Full House", "Roseanne"),
    ),
    NetworkProfile(
        "CBS",
        ("cbs", "cbs.com", "the eye network"),
        ("welcome home",),
        ("game show", "drama", "news", "sports"),
        (
            "The Price Is Right", "The Young and the Restless", "As the World Turns",
            "The Bold and the Beautiful", "Guiding Light", "60 Minutes",
            "CSI: Crime Scene Investigation", "Survivor", "Everybody Loves Raymond",
            "Late Show with David Letterman", "The Late Late Show", "Jeopardy!",
            "Wheel of Fortune",
        ),
        (1941, 2010),
    ),
    NetworkProfile(
        "ABC",
        ("abc", "abc.com", "abc sports"),
        ("america's broadcasting company",),
        ("sports", "drama", "news"),
        (
            "Monday Night Football", "Wide World of Sports", "World News Tonight",
            "Good Morning America", "Lost", "Desperate Housewives",
            "The Bachelor", "Who Wants to Be a Millionaire", "20/20", "Nightline",
        ),
        (1948, 2010),
    ),
    NetworkProfile(
        "The WB",
        ("the wb", "wb network", "warner bros television"),
        (),
        ("drama", "teen"),
        ("Smallville", "Gilmore Girls", "7th Heaven", "One Tree Hill"),
        (1995, 2006),
    ),
    NetworkProfile(
        "UPN",
        ("upn", "united paramount network"),
        (),
        ("drama", "comedy"),
        ("Veronica Mars", "America's Next Top Model", "Star Trek: Enterprise"),
        (1995, 2006),
    ),
    NetworkProfile(
        "FX",
        ("fx networks", "fx"),
        ("there is no box",),
        ("drama",),
        ("The Shield", "Nip/Tuck", "Rescue Me"),
    ),
    NetworkProfile(
        "Bravo",
        ("bravo",),
        (),
        ("lifestyle", "reality"),
        ("Queer Eye for the Straight Guy", "Inside the Actors Studio", "Project Runway"),
    ),
    NetworkProfile(
        "G4",
        ("g4", "g4techtv", "techtv"),
        (),
        ("gaming", "technology"),
        ("X-Play", "Attack of the Show", "Cheat!"),
        (2002, 2010),
    ),
    NetworkProfile(
        "Weather Channel",
        ("the weather channel", "weather channel", "twc", "local on the 8s"),
        (),
        ("weather",),
        ("Local on the 8s", "Storm Stories"),
    ),
)


#: Brand -> (kind, aliases). Ad-heavy categories for this era: fast food,
#: telecoms, soft drinks, consoles, cars, retail, pharma.
BRANDS: dict[str, tuple[EntityKind, tuple[str, ...]]] = {
    "Taco Bell": (EntityKind.BRAND, ("taco bell", "yo quiero", "think outside the bun")),
    "McDonald's": (EntityKind.BRAND, ("mcdonald's", "mcdonalds", "i'm lovin' it", "big mac", "mcnuggets")),
    "Burger King": (EntityKind.BRAND, ("burger king", "whopper", "have it your way", "the king")),
    "Wendy's": (EntityKind.BRAND, ("wendy's", "wendys", "frosty")),
    "Subway": (EntityKind.BRAND, ("subway", "eat fresh", "jared")),
    "KFC": (EntityKind.BRAND, ("kfc", "kentucky fried chicken")),
    "Pizza Hut": (EntityKind.BRAND, ("pizza hut",)),
    "Domino's": (EntityKind.BRAND, ("domino's pizza", "dominos pizza", "domino's")),
    "Coca-Cola": (EntityKind.BRAND, ("coca cola", "coca-cola", "coke")),
    "Pepsi": (EntityKind.BRAND, ("pepsi", "pepsi cola")),
    "Mountain Dew": (EntityKind.BRAND, ("mountain dew", "mtn dew", "do the dew")),
    "Gatorade": (EntityKind.BRAND, ("gatorade", "is it in you")),
    "Budweiser": (EntityKind.BRAND, ("budweiser", "bud light", "whassup")),
    "Xbox": (EntityKind.PRODUCT, ("xbox", "xbox 360", "xbox live")),
    "PlayStation": (EntityKind.PRODUCT, ("playstation", "playstation 2", "ps2", "sony computer entertainment")),
    "Nintendo": (EntityKind.BRAND, ("nintendo", "gamecube", "game boy advance", "nintendo ds")),
    "Sega": (EntityKind.BRAND, ("sega", "dreamcast")),
    "Verizon": (EntityKind.BRAND, ("verizon", "can you hear me now")),
    "Cingular": (EntityKind.BRAND, ("cingular", "cingular wireless")),
    "Sprint": (EntityKind.BRAND, ("sprint pcs", "sprint")),
    "T-Mobile": (EntityKind.BRAND, ("t-mobile", "t mobile")),
    "Nokia": (EntityKind.BRAND, ("nokia",)),
    "Motorola": (EntityKind.BRAND, ("motorola", "razr")),
    "AOL": (EntityKind.BRAND, ("america online", "aol", "you've got mail")),
    "Dell": (EntityKind.BRAND, ("dell", "dude you're getting a dell")),
    "Gap": (EntityKind.BRAND, ("gap",)),
    "Old Navy": (EntityKind.BRAND, ("old navy",)),
    "Wal-Mart": (EntityKind.BRAND, ("wal-mart", "walmart", "always low prices")),
    "Target": (EntityKind.BRAND, ("target stores", "expect more pay less")),
    "Best Buy": (EntityKind.BRAND, ("best buy",)),
    "Circuit City": (EntityKind.BRAND, ("circuit city",)),
    "Blockbuster": (EntityKind.BRAND, ("blockbuster video", "blockbuster")),
    "Ford": (EntityKind.BRAND, ("ford motor", "built ford tough", "f-150")),
    "Chevrolet": (EntityKind.BRAND, ("chevrolet", "chevy", "like a rock", "an american revolution")),
    "Toyota": (EntityKind.BRAND, ("toyota", "get the feeling")),
    "Honda": (EntityKind.BRAND, ("honda",)),
    "Nike": (EntityKind.BRAND, ("nike", "just do it")),
    "Adidas": (EntityKind.BRAND, ("adidas", "impossible is nothing")),
    "OxiClean": (EntityKind.PRODUCT, ("oxiclean", "oxi clean")),
    "ShamWow": (EntityKind.PRODUCT, ("shamwow", "sham wow")),
    "Girls Gone Wild": (EntityKind.PRODUCT, ("girls gone wild",)),
    "Head-On": (EntityKind.PRODUCT, ("head on apply directly", "head-on")),
    "Ronco": (EntityKind.BRAND, ("ronco", "set it and forget it")),
    "Slap Chop": (EntityKind.PRODUCT, ("slap chop",)),
    "Geico": (EntityKind.BRAND, ("geico", "15 minutes could save you")),
    "Progressive": (EntityKind.BRAND, ("progressive insurance",)),
    "State Farm": (EntityKind.BRAND, ("state farm", "like a good neighbor")),
    "Netflix": (EntityKind.BRAND, ("netflix",)),
    "Apple": (EntityKind.BRAND, ("apple computer", "ipod", "itunes", "powerbook")),
}


#: Shows that go by more than one name on air. Matched in addition to the
#: canonical titles above, and resolved back to the canonical form so a search
#: for "Dragon Ball Z" also finds a clip where the announcer only said "DBZ".
SHOW_ALIASES: dict[str, tuple[str, ...]] = {
    "Dragon Ball Z": ("dragon ball z", "dragonball z", "dbz", "kakarot", "super saiyan"),
    "Dragon Ball GT": ("dragon ball gt", "dragonball gt"),
    "Yu Yu Hakusho": ("yu yu hakusho", "yuyu hakusho"),
    "Codename: Kids Next Door": ("kids next door", "codename kids next door"),
    "Foster's Home for Imaginary Friends": ("foster's home", "fosters home"),
    "The Powerpuff Girls": ("powerpuff girls", "powerpuff"),
    "SpongeBob SquarePants": ("spongebob", "spongebob squarepants", "bikini bottom"),
    "Law & Order: SVU": ("special victims unit", "law and order svu", "svu"),
    "Law & Order: Criminal Intent": ("criminal intent", "law and order criminal intent"),
    "The Price Is Right": ("the price is right", "price is right", "come on down"),
    "Monday Night Football": ("monday night football", "mnf"),
    "The Adventures of Jimmy Neutron": ("jimmy neutron", "jimmy neutron boy genius"),
    "Avatar: The Last Airbender": ("avatar the last airbender", "the last airbender"),
    "Ned's Declassified School Survival Guide": ("ned's declassified", "neds declassified"),
    "Who Wants to Be a Millionaire": ("who wants to be a millionaire", "final answer"),
    "The 4400": ("the 4400", "4400"),
    "Walker, Texas Ranger": ("walker texas ranger", "walker, texas ranger"),
    "Hey Arnold!": ("hey arnold",),
    "All Grown Up!": ("all grown up",),
    "Go, Diego, Go!": ("go diego go", "diego"),
    "The Fairly OddParents": ("fairly oddparents", "fairly odd parents"),
    "Aqua Teen Hunger Force": ("aqua teen hunger force", "aqua teen", "athf"),
    "Teen Titans": ("teen titans",),
    "Samurai Jack": ("samurai jack",),
}

#: Programming blocks -- not networks in their own right, but strong evidence
#: of the network that carried them, and of the era.
BLOCKS: dict[str, tuple[str, tuple[str, ...], tuple[int, int]]] = {
    "Toonami": ("Cartoon Network", ("toonami", "tom and sara", "absolution"), (1997, 2008)),
    "Adult Swim": ("Adult Swim", ("adult swim",), (2001, 2010)),
    "Miguzi": ("Cartoon Network", ("miguzi",), (2004, 2007)),
    "SNICK": ("Nickelodeon", ("snick",), (1992, 2004)),
    "TGIF": ("ABC", ("tgif",), (1989, 2005)),
}


#: On-camera personalities whose presence identifies a clip. Pitchmen matter
#: as much as celebrities here -- an infomercial is often *defined* by its host.
PEOPLE: dict[str, tuple[str, ...]] = {
    "Billy Mays": ("billy mays", "billy mays here"),
    "Vince Offer": ("vince offer", "vince shlomi"),
    "Ron Popeil": ("ron popeil", "popeil"),
    "Adam Savage": ("adam savage",),
    "Jamie Hyneman": ("jamie hyneman",),
    "Mike Rowe": ("mike rowe",),
    "Alton Brown": ("alton brown",),
    "Emeril Lagasse": ("emeril lagasse", "emeril", "bam!"),
    "Jon Stewart": ("jon stewart",),
    "Dave Chappelle": ("dave chappelle", "chappelle"),
    "Carson Daly": ("carson daly",),
    "Xzibit": ("xzibit",),
    "Ashton Kutcher": ("ashton kutcher",),
    "Johnny Knoxville": ("johnny knoxville",),
    "Ryan Seacrest": ("ryan seacrest",),
    "Simon Cowell": ("simon cowell",),
    "Bob Barker": ("bob barker",),
    "Regis Philbin": ("regis philbin",),
    "Oprah Winfrey": ("oprah winfrey", "oprah"),
    "Larry King": ("larry king",),
    "Bill O'Reilly": ("bill o'reilly", "o'reilly factor"),
    "Anderson Cooper": ("anderson cooper",),
    "Tony Hawk": ("tony hawk",),
    "Green Day": ("green day", "billie joe armstrong"),
    "Linkin Park": ("linkin park",),
    "Eminem": ("eminem", "slim shady"),
    "Britney Spears": ("britney spears",),
    "Usher": ("usher raymond",),
    "OutKast": ("outkast", "hey ya"),
}


#: Phrases that identify a *kind* of clip regardless of who made it.
FORMAT_MARKERS: dict[str, tuple[str, ...]] = {
    "commercial": (
        "call now", "order now", "available at", "offer expires", "while supplies last",
        "for a limited time", "see store for details", "participating locations",
        "operators are standing by", "plus shipping and handling", "not available in stores",
        "1-800", "1 800", "visit us online", "taxes extra", "results not typical",
    ),
    "promo": (
        "next on", "coming up next", "tonight at", "only on", "all new episode",
        "new episode", "premieres", "don't miss", "this week on", "stay tuned",
        "right after this", "up next on",
    ),
    # NB: every phrase here must be one that does not occur in ordinary
    # dialogue. "this is" was in this list and matched essentially every
    # transcript in the library, voting station-ID on 19-minute episodes.
    "station_id": ("you're watching", "brought to you by", "you are watching"),
    "infomercial": (
        "paid programming", "this program is a paid advertisement", "call within the next",
        "but wait there's more", "money back guarantee", "three easy payments",
    ),
    "news_break": ("breaking news", "this just in", "we now return", "back to you", "live report"),
    "weather": ("forecast", "doppler", "high of", "low of", "chance of rain", "local on the 8s"),
    "tv_guide": ("prevue", "tv guide channel", "coming up on", "channel guide"),
    "ppv_preview": ("pay per view", "order today", "call your cable operator", "ppv"),
    "credits": (
        "executive producer", "directed by", "produced by", "written by", "all rights reserved",
        "copyright", "in memory of", "filmed on location",
    ),
    "rating": ("tv-14", "tv-ma", "tv-pg", "tv-g", "tv-y7", "rated pg", "rated r"),
}


@dataclass
class Match:
    """One knowledge-base hit, with why it fired."""

    label: str
    confidence: float
    kind: Optional[EntityKind] = None
    evidence: str = ""
    extra: dict[str, object] = field(default_factory=dict)


def _compile(phrases: Iterable[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Word-boundary patterns, so 'as' never matches inside 'has'."""
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for phrase in phrases:
        escaped = re.escape(phrase.lower())
        # Only anchor boundaries where the phrase actually starts/ends with a
        # word character; "[adult swim]" and "1-800" must still match.
        prefix = r"\b" if phrase[:1].isalnum() else ""
        suffix = r"\b" if phrase[-1:].isalnum() else ""
        compiled.append((phrase, re.compile(prefix + escaped + suffix)))
    return compiled


_NETWORK_PATTERNS = {
    profile.name: (_compile(profile.markers), _compile(profile.slogans)) for profile in NETWORKS
}
_BRAND_PATTERNS = {name: _compile(aliases) for name, (_kind, aliases) in BRANDS.items()}
_PEOPLE_PATTERNS = {name: _compile(aliases) for name, aliases in PEOPLE.items()}
_FORMAT_PATTERNS = {label: _compile(phrases) for label, phrases in FORMAT_MARKERS.items()}
#: Canonical title plus every alias it is known by on air.
_SHOW_PATTERNS = {
    show: _compile((show, *SHOW_ALIASES.get(show, ())))
    for profile in NETWORKS
    for show in profile.shows
}
_BLOCK_PATTERNS = {name: _compile(markers) for name, (_net, markers, _era) in BLOCKS.items()}
_SHOW_TO_NETWORK = {
    canonical_key(show): profile.name for profile in NETWORKS for show in profile.shows
}


def match_networks(text: str) -> list[Match]:
    """Networks referenced in ``text``.

    A slogan is worth far more than a bare call-sign: "MTV" appears in plenty
    of unrelated text, while "i want my mtv" does not.
    """
    lowered = (text or "").lower()
    if not lowered:
        return []
    matches: list[Match] = []
    for name, (markers, slogans) in _NETWORK_PATTERNS.items():
        best = 0.0
        evidence = ""
        for phrase, pattern in markers:
            if pattern.search(lowered):
                # A long, specific marker is stronger than a two-letter one.
                score = 0.55 if len(phrase) <= 4 else 0.75
                if score > best:
                    best, evidence = score, phrase
        for phrase, pattern in slogans:
            if pattern.search(lowered):
                best, evidence = max(best, 0.95), phrase
        if best:
            matches.append(Match(label=name, confidence=best, kind=EntityKind.NETWORK, evidence=evidence))
    return sorted(matches, key=lambda m: m.confidence, reverse=True)


def match_brands(text: str) -> list[Match]:
    lowered = (text or "").lower()
    if not lowered:
        return []
    matches: list[Match] = []
    for name, patterns in _BRAND_PATTERNS.items():
        for phrase, pattern in patterns:
            if pattern.search(lowered):
                kind = BRANDS[name][0]
                confidence = 0.9 if len(phrase) > 8 else 0.72
                matches.append(Match(label=name, confidence=confidence, kind=kind, evidence=phrase))
                break
    return matches


def match_people(text: str) -> list[Match]:
    lowered = (text or "").lower()
    if not lowered:
        return []
    matches: list[Match] = []
    for name, patterns in _PEOPLE_PATTERNS.items():
        for phrase, pattern in patterns:
            if pattern.search(lowered):
                matches.append(
                    Match(label=name, confidence=0.8, kind=EntityKind.PERSON, evidence=phrase)
                )
                break
    return matches


def match_shows(text: str) -> list[Match]:
    lowered = (text or "").lower()
    if not lowered:
        return []
    matches: list[Match] = []
    for show, patterns in _SHOW_PATTERNS.items():
        for phrase, pattern in patterns:
            if pattern.search(lowered):
                matches.append(
                    Match(
                        label=show,
                        confidence=0.85,
                        kind=EntityKind.SHOW,
                        evidence=phrase,
                        extra={"network": _SHOW_TO_NETWORK.get(canonical_key(show), "")},
                    )
                )
                break
    return _longest_wins(matches)


def _longest_wins(matches: list[Match]) -> list[Match]:
    """Drop a match whose title is contained in a longer one that also matched.

    Franchises nest: "Dragon Ball Z" contains "Dragon Ball", and reporting both
    would tag a Z episode with two different series and split the entity count
    across them. The more specific title is always the right answer.
    """
    keys = {canonical_key(m.label): m for m in matches}
    return [
        match
        for key, match in keys.items()
        if not any(other != key and key in other for other in keys)
    ]


def match_blocks(text: str) -> list[Match]:
    """Programming blocks named in the text.

    A block is strong evidence of its parent network -- "Toonami" means
    Cartoon Network as surely as the bug in the corner does -- and it narrows
    the date range, because blocks ran for known spans.
    """
    lowered = (text or "").lower()
    if not lowered:
        return []
    matches: list[Match] = []
    for name, patterns in _BLOCK_PATTERNS.items():
        for phrase, pattern in patterns:
            if pattern.search(lowered):
                network, _markers, era = BLOCKS[name]
                matches.append(
                    Match(
                        label=name,
                        confidence=0.85,
                        kind=EntityKind.NETWORK,
                        evidence=phrase,
                        extra={"network": network, "era": era},
                    )
                )
                break
    return matches


def match_formats(text: str) -> dict[str, list[str]]:
    """Format markers found, grouped by the format they imply.

    Returns:
        ``{"commercial": ["call now", "1-800"], ...}`` -- the raw phrases are
        kept so a clip's classification can be explained to the operator.
    """
    lowered = (text or "").lower()
    if not lowered:
        return {}
    found: dict[str, list[str]] = {}
    for label, patterns in _FORMAT_PATTERNS.items():
        hits = [phrase for phrase, pattern in patterns if pattern.search(lowered)]
        if hits:
            found[label] = hits
    return found


def network_for_show(show: str) -> Optional[str]:
    return _SHOW_TO_NETWORK.get(canonical_key(show))


def genres_for_network(network: str) -> tuple[str, ...]:
    for profile in NETWORKS:
        if profile.name.lower() == network.lower():
            return profile.genres
    return ()


def network_era(network: str) -> Optional[tuple[int, int]]:
    """The years a network carried this identity -- a dating constraint."""
    for profile in NETWORKS:
        if profile.name.lower() == network.lower():
            return profile.era
    return None


def all_network_names() -> list[str]:
    return [profile.name for profile in NETWORKS]
