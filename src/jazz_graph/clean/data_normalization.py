from unicodedata import normalize
import re
from functools import lru_cache


# Strip "A Love Supreme, Part N - " or "A Love Supreme, Part N" prefix from track titles
# but leave "A Love Supreme" alone as an album/song title
_ALS_PREFIX = re.compile(
    r'^a love supreme,\s*(part|pt\.)\s+\w+[\s\-–—]*',
    re.IGNORECASE
)

def strip_als_prefix(title: str) -> str:
    parts = title.split('/')
    parts = [_ALS_PREFIX.sub('', p.strip()) for p in parts]
    return ' '.join(parts)

def punctuation():
    return r'[:;.,\'"`<>\[\]\(\)-\/\\]'

def remove_parentheticals(title: str) -> str:
    """Remove parenteticals that are not part of the title.

    Also handles cases where a punctuation mark clearly delimits metadata.

    Input title is expected to be lower case, normalized unicode.
    """
    title_junk_strings = [
        r'\(.*?featuring.*?\)',
        r'\(.*?feat\..*?\)',
        r'\(\d\.\d\smix\)',
        r'\(pitch corrected\)',
        r'\((legacy|deluxe) edition\)',
        punctuation() + r'\s?mono$'
        # remaster, stereo, live
    ]
    for reg in title_junk_strings:
        title = re.sub(reg, '', title)
    return title

def tokenize_title(title) -> list[str]:
    return [token.strip() for token in title.strip().split(' ') if token]

def remove_punctuation(tokens: list[str]) -> list[str]:
    punct = punctuation()
    processed_tokens = []
    for token in tokens:
        while re.search(punct, token):
            token = re.sub(punct, ' ', token).strip()
        if token == '&':
            processed_tokens.append('and')
        elif token:
            processed_tokens.append(token)
    return processed_tokens

def expand_abbreviations(tokens) -> list[str]:
    abbreviations = {'vol.': 'volume', 'e.p.': 'ep', 'pt.': 'part'}
    processed_tokens = []
    for token in tokens:
        if (result := re.search(r'\w+\.', token)):
            if abbreviations.get(result.string):
                token = token.replace(result.string, abbreviations[result.string])
        processed_tokens.append(token)
    return processed_tokens

def remove_stop_words(tokens: list[str]) -> list[str]:
    if len(tokens) > 1 and tokens[0] == 'the':
        return tokens[1:]
    return tokens

def clean_remasters(title):
    title = re.sub('Rudy Van Gelder (Edition|Remaster)( \d\d\d\d)?', '', title, flags=re.IGNORECASE)
    title = re.sub('(\d\d\d\d )?(Digital )?Remaster(ed)?( \d\d\d\d)?', '', title, flags=re.IGNORECASE)
    title = re.sub("\d\d Bit Master(ing)?", '', title, flags=re.IGNORECASE)
    return title

@lru_cache(maxsize=128)
def normalize_title(title: str) -> str:
    title = normalize('NFD', title).lower()
    title = strip_als_prefix(title)
    title = remove_parentheticals(title)
    title: str = clean_remasters(title)
    title_tokens: list[str] = tokenize_title(title)
    title_tokens = expand_abbreviations(title_tokens)
    title_tokens = remove_punctuation(title_tokens)
    title_tokens = remove_stop_words(title_tokens)
    title = ' '.join(title_tokens)
    return title


if __name__ == '__main__':
    expand_abbreviations(['vol.'])
