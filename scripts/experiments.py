from collections.abc import Iterable
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

from jazz_graph.data.fetch import fetch_recording_traits
from jazz_graph.model.model import UnsupervisedJazzModel
from jazz_graph.recommendation.recommender import LookupRecordings, RandomWalkRecommender, ArtistWeightedRecommender
from jazz_graph.recommendation.experiment import BSideExperiment, SpotifyExperiement
from jazz_graph.recommendation.recommender import InferenceRecommender
from jazz_graph.training.logging import load_model

RANDOM_SEED = 51342
SPOTIFY_DATA_PATH = '/workspace/local_data/spotify_dataset'

def make_jazz_graph_from_config(config) -> HeteroData:
    graph_data_function = config.get('graph_data_function')
    nodes_data_path = config['data_config']['dataset']
    if graph_data_function is None:
        from jazz_graph.data.graph_builder.graph_builder import make_jazz_data, CreateTensors
        return make_jazz_data(CreateTensors(nodes_data_path))
    from jazz_graph.data.graph_builder import make_jazz
    make_jazz_graph = getattr(make_jazz, graph_data_function)
    store = make_jazz.JazzDataStore(nodes_data_path)
    return make_jazz_graph(store)

def lookup_from_dataset(data: HeteroData) -> LookupRecordings:
    # But why? Answer: the graph data may include pruning that is not part of the
    # parquet datasets or recording traits.
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


def get_experiment_config(path: str):
    with open(Path(path) / 'config.json', 'r') as f:
        run_config = json.loads(f.read())

    nodes_data_path = run_config['data_config'].get('dataset')

    experiment_config = {
        'gnn': path,
        'run_configuration': run_config,
        'nodes_data': nodes_data_path,
    }
    return experiment_config

def b_side_experiment(recommender, recording_traits, experiment_config, log_dir):
    album_experiments = [
        ('Miles Davis', 'Kind of Blue'),
        ('Miles Davis', 'Sketches of Spain'),
        ('Art Blakey & The Jazz Messengers', 'Mosaic'),
        ('Charles Mingus', "Mingus Ah Um"),  # lots of songs, should have some.
        ('The Dave Brubeck Quartet', "Time Out"),
        ('Ornette Coleman', 'The Shape of Jazz to Come')  # very unusual music--should probably be easy.
    ]
    bside = BSideExperiment(recommender, recording_traits, log_dir)
    bside.b_side_experiment(experiment_config, album_experiments)

def create_spotify_datasets(directory) -> Iterable:
    dir_path = Path(directory)
    for path in dir_path.iterdir():
        if not path.is_dir():
            continue
        datadir = next(path.iterdir())
        print(datadir)

        def file_name_is_history(name: str):
            x = name.startswith('StreamingHistory_musi')
            y = name.startswith('Streaming_History_Audio')
            return y
            return x or y

        histories = [file for file in datadir.iterdir() if file_name_is_history(file.name)]
        spotify_data = []
        for history in histories:
            with open(history, 'r') as f:
                data = json.load(f)
            spotify_data.extend(data)
        yield path.name, spotify_data


def get_run_name(experiment_config: dict) -> str:
    run_config = experiment_config['run_configuration']
    task = run_config.get('training_task')
    if task is None:
        return "official_match_album_no_features"
    return 'official_' + task + '_with_features'


def run_experiments():
    models: list[str] = [
        # Commented out are completed.
        '/workspace/experiments/2026-04-03_21-14-25_gnn_simCLR_graph_parquet',  # dual loss task.
        '/workspace/experiments/2026-04-03_16-48-18_gnn_simCLR_graph_parquet',  # match album task, has edges. CHECK COMMIT 04cf388055613913f7
        '/workspace/experiments/2026-04-07_18-07-32_gnn_simCLR_graph_parquet',  # edge ablation loss, has edges.
        '/workspace/experiments/2026-03-31_17-39-18_gnn_simCLR_graph_parquet',  # match album task, no edges. "f361e3e6bc750eef35de9898103a06eb8a0966af"
    ]

    recording_traits = fetch_recording_traits(use_proto=False).set_index('recording_id')
    spotify_datasets = list(create_spotify_datasets(SPOTIFY_DATA_PATH))
    for model_path in models:
        print(f"Running model {model_path}")
        experiment_config = get_experiment_config(model_path)
        experiment_config['random_seed'] = RANDOM_SEED
        run_config = experiment_config['run_configuration']

        graph_data = make_jazz_graph_from_config(run_config)
        model_state = load_model(model_path)
        model_state = model_state.get('model_state_dict', model_state)
        model = UnsupervisedJazzModel.from_config(run_config)
        model.load_state_dict(model_state)
        model = UnsupervisedModelAdapter(model)

        for pooling in ['sum', 'max', 'softmax']:
            experiment_config['recommender_pooling'] = pooling
            recommender = InferenceRecommender(model, graph_data, pooling=experiment_config['recommender_pooling'])    # pyright: ignore [reportArgumentType]
            # lookup = recommender.lookup_recordings
            model_name = get_run_name(experiment_config)
            log_dir = f'/workspace/experiments/{model_name}'
            b_side_experiment(recommender, recording_traits, experiment_config, log_dir)
            for spotify_path, spotify_data in spotify_datasets:
                if not spotify_data:
                    continue
                spotify_experiment = SpotifyExperiement(recording_traits, spotify_data, seed=RANDOM_SEED, log_dir=log_dir)
                experiment_config['spotify_data'] = spotify_path
                spotify_experiment.run_experiment(recommender, experiment_config, k=20)
            experiment_config.pop('spotify_data')
        experiment_config.pop('recommender_pooling')


def baseline_experiments():
    from time import sleep
    experiment_config = get_experiment_config('/workspace/experiments/2026-03-31_17-39-18_gnn_simCLR_graph_parquet')
    run_config = experiment_config['run_configuration']

    graph_data = make_jazz_graph_from_config(run_config)
    recording_traits = fetch_recording_traits(use_proto=False).set_index('recording_id')

    baseline_recommender = RandomWalkRecommender(graph_data, RANDOM_SEED)
    rand_walk_config = {'model': "RandomWalkRecommender", "seed": RANDOM_SEED}
    b_side_experiment(baseline_recommender, recording_traits, rand_walk_config, log_dir='/workspace/experiments/official_random_walk')

    simple_artist_recommender = ArtistWeightedRecommender(recording_traits)
    simple_artist_config = {'model': "ArtistWeightedRecommender"}
    b_side_experiment(simple_artist_recommender, recording_traits, simple_artist_config, log_dir='/workspace/experiments/official_simple_artist')

    for spotify_path, spotify_data in create_spotify_datasets(SPOTIFY_DATA_PATH):
        if not spotify_data:
            continue
        spotify_experiment = SpotifyExperiement(recording_traits, spotify_data, seed=RANDOM_SEED, log_dir='/workspace/experiments/official_spotify_baselines')
        spotify_experiment.run_experiment(simple_artist_recommender, simple_artist_config)
        sleep(1.1)  # you can get name collisions in the experiment logging without this.
        spotify_experiment.run_experiment(baseline_recommender, rand_walk_config)


if __name__ == '__main__':
    print("Running experiments.")
    run_experiments()
    baseline_experiments()
