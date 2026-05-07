import pytest
from jazz_graph.clean.data_normalization import normalize_title, clean_remasters


@pytest.mark.parametrize('title,expected', [
    ('El Barrio - Remastered 2004/Rudy Van Gelder Edition', 'El Barrio - /'),
    ('El Barrio - Remastered 2004/Rudy Van Gelder Edition'.lower(), 'El Barrio - /'.lower()),
    ('Footprints - Remastered', 'Footprints - '),
    ("Adam's Apple - Remastered 2000 / Rudy Van Gelder Edition", "Adam's Apple -  / "),
    ("'Round Midnight - Rudy Van Gelder Remaster", "'Round Midnight - "),
    ("Blue 7 - Rudy Van Gelder Remaster 1956", "Blue 7 - "),
    ("Human Nature (2022 Remaster)", "Human Nature ()"),
    ("Teru - Rudy Van Gelder Edition/2000 Digital Remaster/24 Bit Mastering", "Teru - //"),
    ("Sunday At The Village Vanguard [Keepnews Collection]", "Sunday At The Village Vanguard []")
])
def test_clean_remasters(title, expected):
    assert clean_remasters(title) == expected, f"Got {clean_remasters(title)}"

@pytest.mark.parametrize('title,expected', [
    ('So What (Miles Davis feat. John Coltrane, Cannonball Adderley)', 'so what'),
    ('Genius of Modern Music, Vol. 1', 'genius of modern music volume 1'),
    ('All Blues (Take 1)', 'all blues take 1'),
    ('Moritat (Mack the Knife)', 'moritat mack the knife'),
    ('So What (5.0 Mix)', 'so what'),
    # ('Gloria\'s Step (Live at the Village Vanguard 1961)', 'gloria s step live at the village vanguard 1961'),  # deprecated: no matching instances in data, use tests below instead.
    ('Freddie Freeloader (Pitch Corrected)', 'freddie freeloader'),
    ('Nuit sur les Champs‐Élysées (take 3) (Générique)', 'nuit sur les champs‐élysées take 3 générique'),
    ('El Barrio - Remastered 2004/Rudy Van Gelder Edition', 'el barrio'),
    ('Footprints - Remastered', 'footprints'),
    ("Adam's Apple - Remastered 2000 / Rudy Van Gelder Edition", "adams apple"),
    ("Human Nature (2022 Remaster)", "human nature"), # Miles Davis plays Michael Jackson
    ("Teru - Rudy Van Gelder Edition/2000 Digital Remaster/24 Bit Mastering", "teru"),
    ("Kind Of Blue (Legacy Edition)", "kind of blue"),
    ("Legacy Edition Blues", "legacy edition blues"),
    ("'Feio (feat. Wayne Shorter, John McLaughlin, Chick Corea, Joe Zawinul & Dave Holland)'", 'feio'),
    ('The Bill Evans Trio', 'bill evans trio'),
    ('Mr. PC - Mono', 'mr pc'),
    ('A Song with Mono in the Title', 'a song with mono in the title'),
        # A Love Supreme special cases
    ("A Love Supreme, Part I: Acknowledgement", "acknowledgement"),
    ("A Love Supreme, Part II - Resolution", "resolution"),
    ("A Love Supreme, Part III: Pursuance", "pursuance"),
    ("A Love Supreme, Part IV: Psalm & A Love Supreme", "psalm and a love supreme"),
    ("A Love Supreme", "a love supreme"),          # album title — untouched
    ("A Love Supreme (Deluxe Edition)", "a love supreme"),  # album — untouched
    ("a love supreme, part i: acknowledgement", "acknowledgement"),  # already-lowercased input
    ('A Love Supreme, Part 3: Pursuance / A Love Supreme, Part 4: Psalm', 'pursuance psalm'),
    ('A Love Supreme, Pt. IV - Psalm', 'psalm'),
    ("The", 'the'), # dumb case, but I accidentally wrote code where this one failed.,
    ("'Gloria’s Step (take 2)'", "glorias step take 2"),  # curly apostrophe.
    ("Gloria's Step - Take 2 / Live At The Village Vanguard, NYC; 6/25/1961", "glorias step take 2"),
    ("Coltrane: Live at the Village Vanguard", 'coltrane live at the village vanguard'),  # Hypothetical case: should not trim "Live at ... when no delimiter."
    ("Teru - Rudy Van Gelder Edition/2000 Digital Remaster/24 Bit Mastering", "teru"),
])
def test_normalize_title(title, expected):
    assert normalize_title(title) == expected, f"Got {normalize_title(title)}"