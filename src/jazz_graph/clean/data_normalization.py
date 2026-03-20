from unicodedata import normalize
import re
from functools import lru_cache


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
    punct = r'[:;.,\'"`<>\[\]\(\)-\/\\]'
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

def clean_remasters(title):
    title = re.sub('Rudy Van Gelder (Edition|Remaster)( \d\d\d\d)?', '', title, flags=re.IGNORECASE)
    title = re.sub('(\d\d\d\d )?(Digital )?Remaster(ed)?( \d\d\d\d)?', '', title, flags=re.IGNORECASE)
    title = re.sub("\d\d Bit Master(ing)?", '', title, flags=re.IGNORECASE)
    return title

@lru_cache(maxsize=128)
def normalize_title(title: str) -> str:
    title = normalize('NFD', title).lower()
    title = remove_parentheticals(title)
    title = clean_remasters(title)
    title_tokens = tokenize_title(title)
    title_tokens = expand_abbreviations(title_tokens)
    title_tokens = remove_punctuation(title_tokens)
    title_tokens = remove_stop_words(title_tokens)
    title = ' '.join(title_tokens)
    return title


if __name__ == '__main__':
    expand_abbreviations(['vol.'])
