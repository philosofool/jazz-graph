from functools import cache
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from .legacy import prune_isolated_nodes, torch_values


class JazzDataStore:
    """Responsible only for loading and caching raw dataframes."""
    def __init__(self, directory):
        self.directory = directory
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.read_parquet(Path(self.directory) / name)
        return self._cache[name]


class PerformanceFeatures:
    style_columns = [
        'Swing', 'Big Band', 'Contemporary Jazz',
        'Bop', 'Easy Listening', 'Post Bop',
        'Free Jazz', 'Dixieland', 'Fusion',
        'Cool Jazz', 'Hard Bop', 'Free Improvisation',
        'Soul-Jazz', 'Vocal', 'Jazz-Funk',
        'Avant-garde Jazz', 'Smooth Jazz', 'Modal',
        'Latin Jazz', 'Jazz-Rock'
    ]
    def __init__(self, store: JazzDataStore):
        self.store = store

    def data(self):
        return self.store.load('performance_nodes.parquet').copy()

    def base(self) -> torch.Tensor:
        df = self.data()
        df['release_date'] = df.release_date.astype('datetime64[ms]').dt.year
        return torch.tensor(df[['release_date', 'recording_id']].to_numpy(), dtype=torch.float)

    def style(self) -> torch.Tensor:
        df = self.data()
        return torch.tensor(df[self.style_columns].to_numpy(), dtype=torch.float)

    def album_ids(self) -> torch.Tensor:
        data = self.data()
        return torch.tensor(data.release_group_id.values)

class SongFeatures:
    def __init__(self, store: JazzDataStore):
        self.store = store

    def data(self):
        return self.store.load('song_nodes.parquet').copy()

    def base(self) -> torch.Tensor:
        return torch.tensor(self.data().to_numpy())

class ArtistFeatures:
    def __init__(self, store: JazzDataStore):
        self.store = store

    def data(self):
        return self.store.load('artist_nodes.parquet').copy()

    def base(self):
        return torch.tensor(self.data().to_numpy())

class EdgeFeatures:
    def __init__(self, store: JazzDataStore):
        self.store = store

    def instruments(self) -> torch.Tensor:
        data = self.store.load('artist_performance_edges.parquet').copy()
        instruments = data['instruments']
        encoded_instruments = instruments.map(encode_instrument).to_numpy()
        return torch.tensor(encoded_instruments.reshape(-1, 1), dtype=torch.long)


@cache
def encode_instrument(instrument: str):
    if pd.isna(instrument):
        # Zero is value for an unknown.
        return 0
    x = INSTRUMENT_CATEGORIES.get(instrument, 'other/world')
    return INDEXED_INSTRUMENTS.get(x, 0)


def _make_jazz_graph(store: JazzDataStore, with_style, with_edges) -> HeteroData:
    data = HeteroData()
    perf = PerformanceFeatures(store)
    data['performance'].x = perf.base()
    data['performance'].album_id = PerformanceFeatures(store).album_ids()
    data['artist'].x = ArtistFeatures(store).base()
    data['song'].x = SongFeatures(store).base()
    _add_edge_indexes(data, store)

    if with_style:
        data['performance'].style = perf.style()
    if with_edges:
        data['artist', 'performs', 'performance'].edge_attrs = EdgeFeatures(store).instruments()


    data = prune_isolated_nodes(data)
    data = ToUndirected()(data)
    data.validate()
    return data

def make_jazz_graph(store: JazzDataStore) -> HeteroData:
    return _make_jazz_graph(store, False, False)

def make_jazz_graph_with_styles(store: JazzDataStore) -> HeteroData:
    return _make_jazz_graph(store, True, False)

def make_jazz_graph_with_edges(store: JazzDataStore) -> HeteroData:
    return _make_jazz_graph(store, False, True)

def make_jazz_graph_with_style_and_edges(store: JazzDataStore) -> HeteroData:
    return _make_jazz_graph(store, True, True)

def load_edge_data(store: JazzDataStore, edge_file: str, cols: list[str]) -> torch.Tensor:
    data = store.load(edge_file)
    return torch_values(data[cols]).T

def _add_edge_indexes(data: HeteroData, store: JazzDataStore):
    """Add edge indexes to jazz data."""
    artist_performance = load_edge_data(store, 'artist_performance_edges.parquet', ['artist_id', 'recording_id'])
    artist_song = load_edge_data(store, 'artist_song_edges.parquet', ['artist_id', 'work_id'])
    performance_song = load_edge_data(store, 'performance_song_edges.parquet', ['recording_id', 'work_id'])

    data['artist', 'composed', 'song'].edge_index = artist_song
    data['artist', 'performs', 'performance'].edge_index = artist_performance
    data['performance', 'performing', 'song'].edge_index = performance_song

INSTRUMENT_CATEGORIES = {
    # Drums
    "drums (drum set)": "drums",
    "electronic drum set": "drums (electronic)",
    "drum machine": "drums (electronic)",

    # Acoustic Piano
    "piano": "piano (acoustic)",
    "grand piano": "piano (acoustic)",
    "upright piano": "piano (acoustic)",
    "prepared piano": "piano (acoustic)",
    "fortepiano": "piano (acoustic)",
    "tack piano": "piano (acoustic)",
    "piano spinet": "piano (acoustic)",

    # Electric Piano / Keys (non-organ)
    "electric piano": "piano (electric)",
    "Rhodes piano": "piano (electric)",
    "Wurlitzer electric piano": "piano (electric)",
    "electric grand piano": "piano (electric)",
    "clavinet": "piano (electric)",
    "celesta": "piano (electric)",
    "harpsichord": "piano (electric)",
    "synthesizer": "piano (electric)",
    "analog synthesizer": "piano (electric)",
    "bass synthesizer": "piano (electric)",
    "string synthesizer": "piano (electric)",
    "keyboard": "piano (electric)",
    "mellotron": "piano (electric)",
    "Moog": "piano (electric)",
    "Minimoog": "piano (electric)",
    "synclavier": "piano (electric)",

    # Organ (distinct jazz tradition)
    "organ": "organ",
    "Hammond organ": "organ",
    "electronic organ": "organ",
    "pipe organ": "organ",
    "theatre organ": "organ",
    "harmonium": "organ",
    "barrel organ": "organ",

    # Double Bass (acoustic)
    "double bass": "bass (acoustic)",
    "acoustic bass guitar": "bass (acoustic)",
    "bass violin": "bass (acoustic)",
    "bass viol": "bass (acoustic)",
    "contrabass saxophone": "bass (acoustic)",  # functional bass role

    # Electric Bass
    "electric bass guitar": "bass (electric)",
    "bass guitar": "bass (electric)",
    "fretless bass": "bass (electric)",
    "electric bass guitar": "bass (electric)",
    "electric upright bass": "bass (electric)",
    "bass pedals": "bass (electric)",
    "keyboard bass": "bass (electric)",
    "washtub bass": "bass (electric)",

    # Acoustic Guitar
    "guitar": "guitar (acoustic)",
    "acoustic guitar": "guitar (acoustic)",
    "classical guitar": "guitar (acoustic)",
    "steel-string acoustic guitar": "guitar (acoustic)",
    "archtop guitar": "guitar (acoustic)",
    "resonator guitar": "guitar (acoustic)",
    "12 string guitar": "guitar (acoustic)",
    "acoustic fretless guitar": "guitar (acoustic)",

    # Electric Guitar
    "electric guitar": "guitar (electric)",
    "guitar synthesizer": "guitar (electric)",
    "pedal steel guitar": "guitar (electric)",
    "lap steel guitar": "guitar (electric)",
    "steel guitar": "guitar (electric)",
    "baritone guitar": "guitar (electric)",
    "electric fretless guitar": "guitar (electric)",
    "ebow": "guitar (electric)",
    "Chapman stick": "guitar (electric)",
    "slide guitar": "guitar (electric)",

    # Trumpet family (keep together — flugelhorn/cornet are same acoustic tradition)
    "trumpet": "trumpet",
    "flugelhorn": "trumpet",
    "cornet": "trumpet",
    "pocket trumpet": "trumpet",
    "piccolo trumpet": "trumpet",
    "bass trumpet": "trumpet",
    "bugle": "trumpet",

    # Trombone
    "trombone": "trombone",
    "bass trombone": "trombone",
    "valve trombone": "trombone",
    "tenor trombone": "trombone",

    # Saxophones — keep all voices distinct
    "soprano saxophone": "saxophone (soprano)",
    "sopranino saxophone": "saxophone (soprano)",
    "alto saxophone": "saxophone (alto)",
    "tenor saxophone": "saxophone (tenor)",
    "baritone saxophone": "saxophone (baritone)",
    "bass saxophone": "saxophone (baritone)",  # close enough
    "saxophone": "saxophone (tenor)",  # generic, tenor is the default assumption in jazz

    # Clarinet
    "clarinet": "clarinet",
    "bass clarinet": "clarinet",
    "alto clarinet": "clarinet",
    "contrabass clarinet": "clarinet",
    "basset clarinet": "clarinet",

    # Flute
    "flute": "flute",
    "alto flute": "flute",
    "bass flute": "flute",
    "piccolo": "flute",
    "recorder": "flute",

    # Vibraphone (signature jazz melodic percussion)
    "vibraphone": "vibraphone",
    "marimba": "vibraphone",  # close enough functionally

    # Violin
    "violin": "violin",
    "electric violin": "violin",
    "fiddle": "violin",
    "viola": "violin",
    "cello": "violin",
    "electric cello": "violin",
    "string quartet": "violin",
    "strings": "violin",
    "bass violin": "violin",
    "tenor violin": "violin",

    # Vocals
    "lead vocals": "vocals (lead)",
    "background vocals": "vocals (other)",
    "choir vocals": "vocals (other)",
    "spoken vocals": "vocals (other)",
    "tenor vocals": "vocals (other)",
    "soprano vocals": "vocals (other)",
    "baritone vocals": "vocals (other)",
    "alto vocals": "vocals (other)",
    "bass vocals": "vocals (other)",
    "vocal": "vocals (other)",
    "other vocals": "vocals (other)",
    "mezzo-soprano vocals": "vocals (other)",
    "countertenor vocals": "vocals (other)",
    "whistling": "vocals (other)",

    # Other brass
    "French horn": "other brass",
    "tuba": "other brass",
    "sousaphone": "other brass",
    "mellophone": "other brass",
    "euphonium": "other brass",
    "baritone horn": "other brass",
    "horn": "other brass",
    "tenor horn / alto horn": "other brass",
    "brass": "other brass",
    "valved brass instruments": "other brass",
    "cornett": "other brass",
    "sackbut": "other brass",

    # Other wind
    "bassoon": "other wind",
    "contrabassoon": "other wind",
    "oboe": "other wind",
    "cor anglais": "other wind",
    "woodwind": "other wind",
    "reeds": "other wind",
    "wind instruments": "other wind",
    "double reed": "other wind",
    "harmonica": "other wind",
    "melodica": "other wind",
    "accordion": "other wind",
    "bandoneón": "other wind",
    "bagpipe": "other wind",
    "concertina": "other wind",

    # Percussion (not drum set)
    "percussion": "percussion",
    "membranophone": "percussion",
    "congas": "percussion",
    "bongos": "percussion",
    "timbales": "percussion",
    "snare drum": "percussion",
    "bass drum": "percussion",
    "tambourine": "percussion",
    "cowbell": "percussion",
    "djembe": "percussion",
    "tabla": "percussion",
    "shekere": "percussion",
    "talking drum": "percussion",
    "gong": "percussion",
    "maracas": "percussion",
    "triangle": "percussion",
    "timpani": "percussion",
    "tubular bells": "percussion",
    "xylophone": "percussion",
    "glockenspiel": "percussion",
    "bell": "percussion",
    "cymbal": "percussion",
    "chimes": "percussion",
    "handclaps": "percussion",
    "shakers": "percussion",
    "cimbalom": "percussion",
    "vibraslap": "percussion",
    "finger snaps": "percussion",
    "foot stomps": "percussion",
    "castanets": "percussion",
    "washboard": "percussion",
    "steelpan": "percussion",
    "marimba": "percussion",
    "finger cymbals": "percussion",
    "hi-hat": "percussion",
    "tom-tom": "percussion",
    "frame drum": "percussion",
    "slit drum": "percussion",
    "Batá drum": "percussion",
    "dunun": "percussion",
    "bendir": "percussion",
    "surdo": "percussion",
    "timbales": "percussion",
    "claves": "percussion",
    "cabasa": "percussion",
    "bongos": "percussion",
    "cuíca": "percussion",
    "berimbau": "percussion",
    "wood block": "percussion",
    "gankogui": "percussion",
    "agogô": "percussion",
    "handbell": "percussion",
    "crotales": "percussion",
    "ganzá": "percussion",
    "ocean drum": "percussion",
    "rainstick": "percussion",
    "rhythm sticks": "percussion",
    "bones": "percussion",
    "darbuka": "percussion",
    "goblet drum": "percussion",
    "sabar": "percussion",
    "zabumba": "percussion",
    "ashiko": "percussion",
    "quinto": "percussion",
    "dohol": "percussion",
    "junjung": "percussion",
    "water drum": "percussion",
    "friction drum": "percussion",
    "mridangam": "percussion",
    "kanjira": "percussion",
    "ghatam": "percussion",

    # Electronic / effects
    "effects": "electronic/effects",
    "sampler": "electronic/effects",
    "vocoder": "electronic/effects",
    "turntable": "electronic/effects",
    "tape": "electronic/effects",
    "EWI": "electronic/effects",
    "Lyricon": "electronic/effects",
    "wind synthesizer": "electronic/effects",
    "talkbox": "electronic/effects",
    "voice synthesizer": "electronic/effects",
    "theremin": "electronic/effects",

    # World / other instruments (too sparse or too niche to signal anything)
    "banjo": "other/world",
    "tenor banjo": "other/world",
    "mandolin": "other/world",
    "ukulele": "other/world",
    "harp": "other/world",
    "electric harp": "other/world",
    "harp guitar": "other/world",
    "oud": "other/world",
    "sitar": "other/world",
    "koto": "other/world",
    "lute": "other/world",
    "zither": "other/world",
    "kora": "other/world",
    "bouzouki": "other/world",
    "charango": "other/world",
    "banjo": "other/world",
    "mbira": "other/world",
    "shakuhachi": "other/world",
    "didgeridoo": "other/world",
    "kazoo": "other/world",
    "whistle": "other/world",
    "tin whistle": "other/world",
    "autoharp": "other/world",
    "guitar family": "other/world",
    "violin family": "other/world",
    "trumpet family": "other/world",
    "bowed string instruments": "other/world",
    "slide brass instruments": "other/world",
    "lamellaphone": "other/world",
    "idiophone": "other/world",
    "shruti box": "other/world",
    "tambura": "other/world",
    "harmonium": "other/world",
    "accordion": "other/world",
    "mandola": "other/world",
    "mandocello": "other/world",
    "cavaquinho": "other/world",
    "fiddle": "other/world",
    "rebab": "other/world",
    "hardingfele": "other/world",
    "bandura": "other/world",
    "Chapman stick": "other/world",
    "guzheng": "other/world",
    "pipa": "other/world",
    "khamak": "other/world",
    "sarod": "other/world",
    "santoor": "other/world",
    "kamancheh": "other/world",
    "setar": "other/world",
    "tar": "other/world",
    "sitar": "other/world",
    "tabla": "other/world",
    "tanpura": "other/world",
    "valiha": "other/world",
    "ngɔni": "other/world",
    "xalam": "other/world",
    "gumbri": "other/world",
    "balafon": "other/world",
    "musical saw": "other/world",
    "waterphone": "other/world",
    "singing bowl": "other/world",
    "glass harp": "other/world",
    "toy piano": "other/world",
    "clavichord": "other/world",
    "harpsichord": "other/world",
    "psaltery": "other/world",
    "dulcimer": "other/world",
    "hammered dulcimer": "other/world",
    "omnichord": "other/world",
    "marímbula": "other/world",
    "taragot": "other/world",
    "duduk": "other/world",
    "suona": "other/world",
    "shehnai": "other/world",
    "kaval": "other/world",
    "ney": "other/world",
    "bansuri": "other/world",
    "quena": "other/world",
    "pan flute": "other/world",
    "shakuhachi": "other/world",
    "transverse flute": "other/world",
    "alto flute": "other/world",
    "sheng": "other/world",
    "fife": "other/world",
    "farfisa": "other/world",
    "tubax": "other/world",
    "tiple": "other/world",
    "charango": "other/world",
    "cuatro": "other/world",
    "berimbau": "other/world",
    "udu": "other/world",
    "caxixi": "other/world",
    "güiro": "other/world",
    "shofar": "other/world",

    # Other personnel (not instruments)
    "guest": "other personnel",
    "assistant": "other personnel",
    "executive": "other personnel",
    "associate": "other personnel",
    "additional": "other personnel",
    "co": "other personnel",
    "task": "other personnel",
    "solo": "other personnel",
    "other instruments": "other personnel",
}

INDEXED_INSTRUMENTS = {
    'unknown': 0,
    'bass (acoustic)': 1,
    'bass (electric)': 2,
    'clarinet': 3,
    'drums': 4,
    'drums (electronic)': 5,
    'electronic/effects': 6,
    'flute': 7,
    'guitar (acoustic)': 8,
    'guitar (electric)': 9,
    'organ': 10,
    'other brass': 11,
    'other personnel': 12,
    'other wind': 13,
    'other/world': 14,
    'percussion': 15,
    'piano (acoustic)': 16,
    'piano (electric)': 17,
    'saxophone (alto)': 18,
    'saxophone (baritone)': 19,
    'saxophone (soprano)': 20,
    'saxophone (tenor)': 21,
    'trombone': 22,
    'trumpet': 23,
    'vibraphone': 24,
    'violin': 25,
    'vocals (lead)': 26,
    'vocals (other)': 27
}