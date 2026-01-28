from unicodedata import normalize
import re


def remove_parentheticals(title: str) -> str:
    """Remove parenteticals that are not part of the title.

    Input title is expected to be lower case, normalized unicode.
    """
    title_junk_strings = [
        r'\(.*?featuring.*?\)',
        r'\(.*?feat\..*?\)',
        r'\(\d\.\d\smix\)',
        r'\(pitch corrected\)',
        # remaster, stereo, live
    ]
    for reg in title_junk_strings:
        if re.search(reg, title):
            title = re.sub(reg, '', title)
    return title

def tokenize_title(title) -> list[str]:
    return [token.strip() for token in title.strip().split(' ') if token]

def remove_punctuation(tokens: list[str]) -> list[str]:
    punct = r'[:;.,\'"`<>\[\]\(\)-]'
    processed_tokens = []
    for token in tokens:
        while re.search(punct, token):
            token = re.sub(punct, ' ', token).strip()
        if token:
            processed_tokens.append(token)
    return processed_tokens

def expand_abbreviations(tokens) -> list[str]:
    abbreviations = {'vol.': 'volume', 'e.p.': 'ep'}
    processed_tokens = []
    for token in tokens:
        if (result := re.search(r'\w+\.', token)):
            if abbreviations.get(result.string):
                token = token.replace(result.string, abbreviations[result.string])
        processed_tokens.append(token)
    return processed_tokens

def remove_stop_words(tokens: list[str]) -> list[str]:
    return tokens

def normalize_title(title: str) -> str:
    title = normalize('NFD', title).lower()
    title = remove_parentheticals(title)
    title_tokens = tokenize_title(title)
    title_tokens = expand_abbreviations(title_tokens)
    title_tokens = remove_punctuation(title_tokens)
    title_tokens = remove_stop_words(title_tokens)
    title = ' '.join(title_tokens)
    return title

print(__name__)
if __name__ == '__main__':
    expand_abbreviations(['vol.'])
    cases = [
        ('So What (Miles Davis feat. John Coltrane, Cannonball Adderley)', 'so what'),
        ('Genius of Modern Music, Vol. 1', 'genius of modern music volume 1'),
        ('All Blues (Take 1)', 'all blues take 1'),
        ('Moritat (Mack the Knife)', 'moritat mack the knife'),
        ('So What (5.0 Mix)', 'so what'),
        ('Gloria\'s Step (Live at the Village Vanguard 1961)', 'gloria s step live at the village vanguard 1961'),
        ('Freddie Freeloader (Pitch Corrected)', 'freddie freeloader'),
        ('Nuit sur les Champs‐Élysées (take 3) (Générique)', 'nuit sur les champs‐élysées take 3 générique')
    ]
    for case, expected in cases:
        assert normalize_title(case) == expected, f"Got {normalize_title(case)}"