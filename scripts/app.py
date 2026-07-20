import gradio as gr
import pandas as pd

from pathlib import Path
import json
import torch
from jazz_graph.model.model import UnsupervisedJazzModel
from jazz_graph.data.fetch import fetch_recording_traits
from jazz_graph.training.logging import load_model

from jazz_graph.data.graph_builder.make_jazz import make_jazz_graph_with_style_and_edges, JazzDataStore


from jazz_graph.clean.data_normalization import normalize_title
from jazz_graph.recommendation.recommender import InferenceRecommender, Recommender, LookupRecordings, Recommender

JAZZ_GRAPH_DESCRIPTION = """Generate Jazz Musical Recommendations.

JazzGraph generates jazz musical recommendations using the social network of musical collaborations
among Jazz artists. Using over 100,000 jazz performances, nearly 25,000 artists and greater than 500,000
connections between them, JazzGraph learns musical similarity from artists decisions about whom to
collaborate with.

This is a light-weight demonstration based on version 1.1 of the JazzGraph system. The complete write up
of version 1.0.1 can be found at http://philosofool.github.io/jazz-graph
"""

class LookupInput:
    def __init__(self, recording_traits: pd.DataFrame):
        # recording traits should be indexed on recording_id

        self.recording_traits = recording_traits.copy()
        self.recording_traits['norm_album'] = self.recording_traits.album.apply(normalize_title)
        self.recording_traits['norm_song'] = self.recording_traits.title.apply(normalize_title)
        self.recording_traits['norm_artist'] = self.recording_traits.artist.apply(normalize_title)

    def _match_recording(self, record: dict) -> pd.DataFrame:
        fields = 'album', 'artist', 'song'
        df = self.recording_traits
        valid_filter = False
        for field in fields:
            data = record.get(field)
            if not data:
                continue
            data = normalize_title(data)
            norm_field = 'norm_' + field
            mask = df[norm_field] == data
            if not mask.any():
                continue
            valid_filter = True
            df = df[mask]
        if len(df) == len(self.recording_traits):
            return pd.DataFrame({}, columns=self.recording_traits.columns)
        return df

    def match_recordings(self, records: dict | list[dict]) -> pd.DataFrame:
        if isinstance(records, dict):
            result = self._match_recording(records)
            if result.empty:
                return self._match_recording({'album': "Kind of Blue", 'artist': "Miles Davis"})
            return result
        dfs = [self._match_recording(e) for e in records if not e.empty]
        if not dfs:
            return self._match_recording({'album': "Kind of Blue", 'artist': "Miles Davis"})
        return pd.concat(dfs)


class Recommend:
    def __init__(self, recommender: InferenceRecommender|Recommender, recording_traits: pd.DataFrame):
        self.lookup = LookupInput(recording_traits)
        self.recording_traits = self.lookup.recording_traits[['artist', 'title', 'album', 'release_date']]
        self.recommender = recommender

    def recommend(self, artist, song, album):
        record = {'artist': artist, 'song': song, 'album': album}
        df = self.lookup._match_recording(record)
        # TODO: handle no matches, all matches.
        if len(df) == len(self.lookup.recording_traits):
            ...
        elif len(df) == 0:
            ...
        rec_ids, *_ = self.recommender.get_recommendations(list(df.index))
        return self.recording_traits.loc[rec_ids]

def filter_recommendations(recs: pd.DataFrame, n_recommendations=6, albums_only=True):
    seen = set()
    n_albums = 0
    n_rows = 0
    for row in recs.itertuples():
        n_rows += 1
        album = row.album
        if album not in seen:
            n_albums += 1
            seen.add(album)
        if n_albums >= n_recommendations:
            break
    out = recs[:n_rows]
    if albums_only:
        return out.drop_duplicates(subset=['album']).drop(columns=['title'])
    return out


class UnsupervisedModelAdapter(torch.nn.Module):
    def __init__(self, model: UnsupervisedJazzModel):
        super().__init__()
        self.model = model

    def __call__(self, x_dict, edge_index_dict, batch):
        # NOTE: Code smell here.
        return self.model(batch)


def _get_usupervised_model(model_path) -> UnsupervisedModelAdapter:
    with open(Path(model_path) / 'config.json', 'r') as f:
        run_config = json.loads(f.read())
    model_state = load_model(model_path)
    model_state = model_state.get('model_state_dict', model_state)
    model = UnsupervisedJazzModel.from_config(run_config)
    model.load_state_dict(model_state)
    model = UnsupervisedModelAdapter(model)
    return model

def _get_metapath_model(model_path) -> Recommender:
    return Recommender.from_path(model_path, '/workspace/local_data/graph_parquet')

def get_model(model_path):
    if 'gnn_simCLR' in model_path:
        return _get_usupervised_model(model_path)
    if 'metapath' in model_path:
        return _get_metapath_model(model_path)
    raise ValueError("Unable to determine model type from path.")

def build_recommend(model_path):
    print("Fetching data.")
    recording_traits = fetch_recording_traits().set_index('recording_id')
    print("Loading model.")
    model = get_model(model_path)
    if isinstance(model, Recommender):
        # Metapath2Vec: Recommender.from_path already produced a full recommender
        # from cached embeddings, so there's no GNN forward pass to wrap.
        recommender = model
    else:
        print("Building data...")
        graph_data = make_jazz_graph_with_style_and_edges(JazzDataStore('/workspace/local_data/graph_parquet'))
        recommender = InferenceRecommender(model, graph_data, 'softmax')
    recommend = Recommend(recommender, recording_traits)
    return recommend

def build_demo(recommend: Recommend):
    print("Constructing demo.")
    demo = gr.Interface(
        fn=(lambda artist, song, album: filter_recommendations(recommend.recommend(artist, song, album))),
        inputs=[
            gr.Textbox(label="artist", info="Name of the artist who published the perfromance. Does not currently support filters by sidemen, i.e., use 'Miles Davis' for Kind of Blue, not 'Bill Evans'."),
            gr.Textbox(label="song", info="Name of the song. Many jazz songs are performed by several artists. For best results, also include artist or album name."),
            gr.Textbox(label="album", info="Name of the album the song was first on."),
        ],
        outputs=["dataframe"],
        examples=[
            ['Joe Henderson', 'Isotope', 'Inner Urge'],
            ['Miles Davis', 'So What', 'Kind of Blue'],
        ],
        title="JazzGraph",
        description=JAZZ_GRAPH_DESCRIPTION,
        api_name="jazz_graph"
    )
    print("Finished building demo.")
    return demo

if __name__ == '__main__':
    model_path = '/workspace/experiments/2026-07-19_18-24-03_metapath2vec_graph_parquet'
    recommend = build_recommend(model_path)

    recs = recommend.recommend(*['Joe Henderson', 'Isotope', 'Inner Urge'])
    recs = recommend.recommend('John Coltrane', 'Psalm', 'A Love Supreme')
    # print(recs)
    demo = build_demo(recommend)
    print("Launching demo...")
    demo.launch(server_port=7861)