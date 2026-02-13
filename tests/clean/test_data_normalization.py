import pytest
from jazz_graph.clean.data_normalization import normalize_title


@pytest.mark.parametrize('title,expected', [
    ('So What (Miles Davis feat. John Coltrane, Cannonball Adderley)', 'so what'),
    ('Genius of Modern Music, Vol. 1', 'genius of modern music volume 1'),
    ('All Blues (Take 1)', 'all blues take 1'),
    ('Moritat (Mack the Knife)', 'moritat mack the knife'),
    ('So What (5.0 Mix)', 'so what'),
    ('Gloria\'s Step (Live at the Village Vanguard 1961)', 'gloria s step live at the village vanguard 1961'),
    ('Freddie Freeloader (Pitch Corrected)', 'freddie freeloader'),
    ('Nuit sur les Champs‐Élysées (take 3) (Générique)', 'nuit sur les champs‐élysées take 3 générique')
])
def test_normalize_title(title, expected):
    assert normalize_title(title) == expected, f"Got {normalize_title(title)}"