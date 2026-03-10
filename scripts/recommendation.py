# %%
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from pathlib import Path

import json
import pandas as pd
import numpy as np
from collections.abc import Callable, Iterable

from jazz_graph.data.graph_builder import CreateTensors
from jazz_graph.data.fetch import fetch_recording_traits, fetch_discogs_to_recording_id
from jazz_graph.model.model import LinkPredictionModel, UnsupervisedJazzModel
from jazz_graph.recommendation.recommender import Recommender, LookupRecordings, PredictLinkRecommender
from jazz_graph.training.logging import ExperimentLogger, load_embeddings, load_model, find_most_recent_run

checkpoint_path = str(find_most_recent_run('/workspace/experiments', 'simCLR'))

with open(Path(checkpoint_path) / 'config.json', 'r') as f:
    run_config = json.loads(f.read())
    nodes_data_path = run_config['data_config'].get('dataset')

experiment_config = {
    'gnn': checkpoint_path,
    'nodes_data': nodes_data_path,
}

from jazz_graph.metrics.ranking import map_at_k
from jazz_graph.recommendation.recommender import LookupRecordings

class RecommenderExperiment:
    def __init__(self, recommender: Recommender, recording_traits: pd.DataFrame):
        self.recommender = recommender
        self.recording_traits = recording_traits

    def _extract_recording_ids(self, artist, album) -> tuple:
        album_recordings = self.recording_traits.query(f"artist == '{artist}'").query(f"album == '{album}'").index.to_numpy()
        assert len(album_recordings) > 1, f"Unable to identify mulitple recordings with {artist}, {album}"
        n_inputs = len(album_recordings) // 2
        input_ids = album_recordings[:n_inputs]
        expected_recs = album_recordings[n_inputs:]
        return input_ids, expected_recs

    def b_side_precision(self, artist: str, album: str):
        """Do a b-side experiment for the input artist's album."""
        input_ids, expected_recs = self._extract_recording_ids(artist, album)
        k = len(expected_recs) + 20
        recs, *_ = self.recommender.get_recommendations(input_ids.tolist())
        map_k = map_at_k(recs, expected_recs, k)
        return {
            'artist': artist,
            'album': album,
            'n_inputs': len(input_ids),
            'n_expected_recs': len(expected_recs),
            'k': k,
            f'MAP_at_k': float(map_k),
            'recommended_ids': recs.tolist()[:k]}

    def b_side_experiment(self, experiment_config: dict, album_experiments: list[tuple[str, str]]):
        """Perform a b-side experiment for each artist-album pair.

        The second side of a record, also known as the 'B-side' should probably be
        recommended given only the A-side as inputs. This experiment simulates that
        as a basic sanity check for the recommender.
        """
        experiment_log = ExperimentLogger(root='/workspace/experiments', run_name='recommendations', config=experiment_config)
        results = []
        for artist, album in album_experiments:
            result = self.b_side_precision(artist, album)
            experiment_log.log_metrics(None, result, "b_side_experiment")
            results.append(result)
        return results


def lookup_from_dataset(data: HeteroData) -> LookupRecordings:
    rec_ids = data['performance'].x[:, 1].numpy()
    ids = np.arange(rec_ids.size)
    df = pd.DataFrame({'recording_ids': rec_ids, 'ids': ids}).set_index('recording_ids')
    return LookupRecordings(df)


class UnsupervisedModelAdapter(torch.nn.Module):
    def __init__(self, model: UnsupervisedJazzModel):
        super().__init__()
        self.model = model

    def __call__(self, x_dict, edge_index_dict, batch):
        return self.model(batch)
        return self.model.encode(batch)



# %%
## Inductive Graph Recommender
from jazz_graph.data.graph_builder import make_jazz_data
from jazz_graph.model.model import JazzModel
from jazz_graph.recommendation.recommender import InferenceRecommender


graph_data: HeteroData = make_jazz_data(CreateTensors(nodes_data_path))
model_state = load_model(checkpoint_path)
model = UnsupervisedJazzModel.from_config(run_config)
model.load_state_dict(model_state)
model = UnsupervisedModelAdapter(model)
lookup = lookup_from_dataset(graph_data)
link_prediction_recommender = InferenceRecommender(model, graph_data, lookup)
recording_traits = fetch_recording_traits(use_proto=nodes_data_path.endswith('proto')).set_index('recording_id')
experimenter = RecommenderExperiment(link_prediction_recommender, recording_traits)

# %%

album_experiments = [
    ('Miles Davis', 'Kind of Blue'),
    ('Miles Davis', 'Sketches of Spain'),
    ('Art Blakey & The Jazz Messengers', 'Mosaic'),
    ('Charles Mingus', "Mingus Ah Um"),  # lots of songs, should have some.
    ('The Dave Brubeck Quartet', "Time Out"),
    ('Ornette Coleman', 'The Shape of Jazz to Come')  # very unusual music--should probably be easy.
]

experiment = album_experiments[1]
print(experiment)
result = experimenter.b_side_precision(*experiment)

recs = recording_traits.loc[result['recommended_ids']].head(20)


results = experimenter.b_side_experiment(experiment_config, album_experiments)
print([{k: v for k, v in e.items() if k != 'recommended_ids'} for e in results])
for rec in results:
    print(rec['recommended_ids'])

raise Exception("Finish here.")

# %%
def analyze_album_similarity(album: str, artist: str, experimenter: RecommenderExperiment, recommender: Recommender):
    rec_ids = np.concat(experimenter._extract_recording_ids(artist, album))
    rand_ids = torch.randint(0, len(recommender.embeddings.weight), (len(rec_ids),)).numpy()
    rec_embeddings = recommender.lookup_recordings.lookup_node_index(rec_ids)

    rec_embeddings = recommender.embeddings(torch.tensor(rec_embeddings))
    rand_embeddings = recommender.embeddings(torch.tensor(rand_ids))
    within_album = torch.cdist(rec_embeddings, rec_embeddings)
    between_albums = torch.cdist(rec_embeddings, rand_embeddings)

    print(f"Album {album}:")
    print(f"  Within-album distance: {within_album.mean():.3f}")
    print(f"  Between-album distance: {between_albums.mean():.3f}")
    print(f"  Ratio: {between_albums.mean() / within_album.mean():.2f}")


# %%
torch.identiy()

# %%
model.model.base_model.performance_embed
artist, album = album_experiments[0]
ids = recording_traits.query(f"artist == '{artist}'").query(f"album == '{album}'").index.to_numpy()
mask = torch.isin(graph_data['performance'].x[:, 1], torch.tensor(ids))

def zero_diag(tensor):
    return tensor * (torch.ones_like(tensor) - torch.eye(tensor.size(0)))

graph_data['performance'].x[mask]
model.model.base_model.performance_embed.weight[mask].sum(axis=1)    # pyright: ignore
enc = model.model(graph_data)['performance']
kob = enc[mask]
rand_selection = torch.randint(0, mask.size(0), (5,))
other = enc[rand_selection]
zero_diag((kob @ kob.T)).mean().item(), (kob @ other.T).mean().item()

# %%
(kob @ other.T)

# %%
(kob @ kob.T)

# %%
performance_embeddings = load_embeddings(checkpoint_path)[0]['performance']
recording_lookup = lookup_without_islands(nodes_data_path)
print(performance_embeddings, recording_lookup.data.shape)

recommender = Recommender(
    performance_embeddings,
    recording_lookup
)


# %%
for artist, album in album_experiments:
    analyze_album_similarity(album, artist, experimenter, recommender)

# %%
results = experimenter.b_side_experiment(experiment_config, album_experiments)
pd.DataFrame.from_records(results).drop(columns=['recommended_ids'])

# %%
from jazz_graph.recommendation.recommender import cosine_similarity


def test_score_similarity():
    user_embed = torch.tensor([1., -1, 0])
    performance_embed = torch.tensor([
        [0, 0, 0],
        [1, -1, 0],
        [-1, 1, 0.]
    ])
    result: torch.Tensor = cosine_similarity(user_embed, performance_embed)
    print(result)
    assert result.size(0) == 3
    assert result.dim() == 1, result.dim

test_score_similarity()

# %% [markdown]
# ## Spotify Data

# %%
spotify = 'local_data'
import os
import json
spotify_sample = '../local_data/my_spotify_data/spotify_extended_streaming_history/Streaming_History_Audio_2025-2026_5.json'
with open(spotify_sample, 'r') as f:
    spotify_data = json.loads(f.read())

def spotify_details(record: dict):
    details = [
        'ts', 'master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name',
        'reason_start', 'reason_end', 'shuffle', 'skipped'
    ]
    return {key: record.get(key) for key in details}

spotify_data[-100:-90]
spotify_data[-10:]
ten_recent = [spotify_details(record) for record in spotify_data[-20:-10]]

# %%
from jazz_graph.clean.data_normalization import normalize_title

seen = set()
i = 0
for rec in spotify_data:
    title = rec['master_metadata_track_name']
    norm_title = normalize_title(title)
    if norm_title in seen:
        continue
    seen.add(norm_title)
    if 'digital' in title.lower():
        i += 1
        print(norm_title)
        print(title)
    # print(norm_title)
print(i)

# %%
from jazz_graph.etl.extract_discogs import InMemDiscogs, MatchDiscogs, is_jazz_album
discogs = MatchDiscogs(InMemDiscogs('/workspace/local_data/jazz_releases.jsonl', is_jazz_album))


# %%
len(spotify_data)

# %%
class SpotifyListens:
    def __init__(self):
        self.discogs = MatchDiscogs(InMemDiscogs('/workspace/local_data/jazz_releases.jsonl', is_jazz_album))
        self._discogs_to_recording_id = fetch_discogs_to_recording_id().set_index('discogs_id')

    def get_spotify_jazz(self, spotify_data: list[dict], unique=True) -> Iterable[tuple[dict, dict]]:
        """Return a generator for jazz records in spotify data."""
        seen = set()
        for record in spotify_data:
            song = record['master_metadata_track_name']
            album = record['master_metadata_album_album_name']
            artist = record['master_metadata_album_artist_name']
            spot_id = record['spotify_track_uri']
            if unique and spot_id in seen:
                continue
            seen.add(spot_id)
            matching = discogs.matching_discog((None, None, song, album, artist))
            if not matching:
                continue
            yield record, matching

    def get_listen_ids(self, spotify_data: list[dict], unique=True):
        ids = []
        for _, matching in self.get_spotify_jazz(spotify_data, unique):
            discogs_id = matching['id']
            ids.append(discogs_id)
        return self.discogs_to_recording_id(ids)

    def discogs_to_recording_id(self, discogs_ids: list[int]):
        valid_ids = self._discogs_to_recording_id.index.intersection(discogs_ids)
        return self._discogs_to_recording_id.loc[valid_ids].recording_id

# %%
spotify_listens = SpotifyListens()
my_listens = spotify_listens.get_listen_ids(spotify_data)
recommendations, scores = recommender.get_recommendations(my_listens)


# %%
recording_traits.loc[recommendations].tail(20)

# %% [markdown]
# Sanity Check:
# - Scores: do obviously similar performances (e.g. same album) cluster?
# - Provide half of the songs from an album. Are the other half highly rated? If not, indicates bug or surprising learning by the model.
# Pick an album where the compositions are mostly novel ones. See Sketches of Spain above; that seemed a little off to me.
#
# Bad candidates indicate that cosine similarity between the mean performance vectors of the listener combined with the performance embeddings don't produce a reliable score.
# Bad embeddings: If performance embeddings don't cluster much, we would expect this. Could also be that what he GNN is learning isn't very relevant. (we would still expect songs from side B of an album to be high given songs from side A.) Indicates issue with GNN.
# Bad aggregation: It could be that there are issues with taking the mean of the listener performance. Indicates that we need better user model.
# Bad similarity score: It could be that dot product rather than similarity is the way to go. Switch similarity metric, possibly as alpha(CS) + (1  - alpha)(DP) Question: what does the embedding norm mean? That's not an easy question to answer...
# Bug: this is the most annoying possibility: what if there's a problem in your set up and your not aligning embeddings, etc. correctly?
# you would expect random seeming recommendation in that case. <- This is unlikely. The album precision experiments show high relevance for the input albums but much lower relevance to the B-sides.
#
