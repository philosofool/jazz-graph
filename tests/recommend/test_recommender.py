from jazz_graph.recommendation.recommender import LookupRecordings
import pandas as pd
import numpy as np
from tempfile import TemporaryFile

class TestLookupRecordings:
    # def test_from_path(self):
    #     ...

    def test_lookup_node_index(self):
        data = pd.DataFrame({'ids': [1, 2]}, index=[101, 102])
        lookup = LookupRecordings(data)
        ids = [101, 102]
        result = lookup.lookup_node_index(ids)
        np.testing.assert_array_equal(result, np.array([1, 2]))

        result = lookup.lookup_node_index([101, 103])
        np.testing.assert_array_equal(result, [1])

        with np.testing.assert_raises(KeyError):
            result = lookup.lookup_node_index([101, 103], missing='raise')

    def test_lookup_recording_ids(self):
        data = pd.DataFrame({'ids': [1, 2]}, index=[101, 102])
        lookup = LookupRecordings(data)
        np.testing.assert_array_equal(lookup.lookup_recording_ids(np.array([1])), [102])
