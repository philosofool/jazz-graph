from jazz_graph.recommendation.experiment import RandomAlbumSplit

from collections import namedtuple

# Mock tuples from recording_traits.itertuples
Record = namedtuple('Record', ['Index', 'release_group_id'])

class TestRandomAlbumSplit:

    def test_add_to_splits(self):
        album_split = RandomAlbumSplit(seed=2)
        record = {'recording_id': 1, 'album_id': 10}
        record = Record(1, 10)
        album_split.add_to_splits(record)
        assert 10 in album_split.split_a or 10 in album_split.split_b
        if 10 in album_split.split_a:
            assert 1 in album_split.recordings_a
        else:
            assert 1 in album_split.recordings_b

        record = Record(2, 10)
        album_split.add_to_splits(record)
        if 10 in album_split.split_a:
            assert 2 in album_split.recordings_a
        else:
            assert 2 in album_split.recordings_b

    def test_make_splits(self):
        album_split = RandomAlbumSplit(seed=2)
        data = [Record(x * 1000, x % 100) for x in range(100)]
        album_split.make_splits(data)
        for x in album_split.recordings_a:
            x_root = x // 1000
            assert x_root % 100 in album_split.split_a
        for x in album_split.recordings_b:
            x_root = x // 1000
            assert x_root % 100 in album_split.split_b

        assert album_split.split_a, "There should be some albums in a."
        assert album_split.split_b, "There should be some albums in b."
        assert not album_split.split_a.intersection(album_split.split_b), "They should not overlap."