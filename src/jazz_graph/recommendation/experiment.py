from __future__ import annotations
from collections.abc import Iterable
import pandas as pd
import numpy as np

import random

from typing import TYPE_CHECKING
from jazz_graph.training.logging import ExperimentLogger
from jazz_graph.recommendation.playlist import SpotifyListens
from jazz_graph.metrics.ranking import map_at_k

if TYPE_CHECKING:
    from jazz_graph.recommendation.recommender import Recommender, LookupRecordings


class RandomAlbumSplit:
    def __init__(self, frac: float = .5, seed: int|None = None):
        self.frac = frac
        self.split_a = set()
        self.split_b = set()
        self.recordings_a = []
        self.recordings_b = []
        self._rng = random.Random(seed)

    def make_splits(self, data: Iterable):
        for record in data:
            self.add_to_splits(record)

    def add_to_splits(self, record):
        album = record.release_group_id
        recording_id = record.Index
        if album in self.split_a:
            self.recordings_a.append(recording_id)
        elif album in self.split_b:
            self.recordings_b.append(recording_id)
        else:
            if self._rng.random() > self.frac:
                self.recordings_a.append(recording_id)
                self.split_a.add(album)
            else:
                self.recordings_b.append(recording_id)
                self.split_b.add(album)


class BSideExperiment:
    """Do a b-side experiment.

    Given an input of an artist-album pair and asplit of songs as seed inputs
    measure the number of remaining songs on the album being recommended.
    """
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
        recs, _, mask = self.recommender.get_recommendations(input_ids.tolist())
        recs = recs[~mask]
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
        experiment_log = ExperimentLogger(root='../experiments', run_name='recommendations', config=experiment_config)
        results = []
        for artist, album in album_experiments:
            result = self.b_side_precision(artist, album)
            experiment_log.log_metrics(None, result, "b_side_experiment")
            results.append(result)
        return results


class SpotifyExperiement:
    def __init__(self, recording_traits: pd.DataFrame, listening_history: list[dict], seed=None):
        self.listening_history = listening_history
        self.spotify = SpotifyListens(recording_traits)
        self.album_split = RandomAlbumSplit(seed=seed)
        self.album_split.make_splits(self.spotify.get_listen_data(listening_history))

    # @classmethod
    # def from_graph_data(cls, graph_data: HeteroData, listening_history):
    #     recording_traits = LookupRecordings.from_hetero_data(graph_data)
    #     return cls(recording_traits, listening_history)

    def run_experiment(self, recommender: Recommender, experiment_config):
        recommendations, scores, mask = recommender.get_recommendations(self.in_samples)
        metrics = self.experiment_metrics(recommendations, mask)
        self.log_experiment(experiment_config, metrics)

    @property
    def in_samples(self) -> np.ndarray:
        """One side of the experiment split, used for recommender seeding."""
        return np.array(self.album_split.recordings_a)

    @property
    def out_samples(self) -> np.ndarray:
        """The complement of in_samples, used for experiment metrics."""
        return np.array(self.album_split.recordings_b)

    def _make_logger(self, experiment_config):
        self._experiment_log = ExperimentLogger(root='../experiments', run_name='spotify_recommendations', config=experiment_config)

    def log_experiment(self, experiment_config, metrics):
        if not hasattr(self, '_experiment_log'):
            self._make_logger(experiment_config)
        else:
            config = self._experiment_log.config()
            config.pop('commit')    # pyright: ignore [reportOptionalMemberAccess]  <- None should never happen here, so fail if it does, but not typing error.
            if experiment_config != config:
                self._make_logger(experiment_config)
        experiment_log = self._experiment_log
        experiment_log.log_metrics(None, metrics, "spotify_recommendations")

    def experiment_metrics(self, recommendations, mask):
        # We are mainly interested in recall of out-of-sample splits from the album spliter.
        # Recall of albums from the in-sample cases are less interesting, since no matter
        # how the model scores those, we can always filter--the familiar candiates are availbale
        # at inference time.
        in_samples = self.in_samples
        out_samples = self.out_samples
        novel_recommendations = recommendations[~mask]
        familiar_recommendations = recommendations[mask]

        n = max(20, out_samples.size * 2)
        positives_novel = novel_recommendations[:n]
        positives_familiar = familiar_recommendations[:n]

        novel_tp = np.intersect1d(novel_recommendations, out_samples)
        familiar_tp = np.intersect1d(familiar_recommendations, in_samples)
        novel_fn = np.setdiff1d(out_samples, novel_tp)
        familiar_fn = np.setdiff1d(in_samples, familiar_tp)

        recall_novel = novel_tp.size / (novel_tp.size + novel_fn.size)
        recall_familiar = familiar_tp.size / (familiar_tp.size + familiar_fn.size)

        metrics = {
            'recall_novel': float(recall_novel),
            'recall_familiar': float(recall_familiar),
            'recall_tp': float(novel_tp.size),
            'known_tp': float(familiar_tp.size),
            'samples': {
                f'top_{n}_novel_recommendations': positives_novel.tolist(),
                'true_positive_in_novel_recommendation': novel_tp.tolist(),
                'false_negative_in_novel_sample': novel_fn.tolist(),
                f'top_{n}_familiar_recommendations': positives_familiar.tolist(),
                'true_positive_in_familiar_recommendation': familiar_tp.tolist(),
                'false_negative_in_familiar_sample': familiar_fn.tolist(),
            }

        }
        return metrics
