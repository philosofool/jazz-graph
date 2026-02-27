from jazz_graph.clean.data_normalization import normalize_title
from jazz_graph.etl.extract_discogs import InMemDiscogs, MatchDiscogs


class TestInMemDiscogs:
    path = '/workspace/tests/test_releases.jsonl'
    def test_tracklist(self):
        in_mem_discogs = InMemDiscogs(self.path)
        tracklist = in_mem_discogs.tracklist()
        assert isinstance(in_mem_discogs.tracklist(), dict), "Expect a dictionary from the tracklist."
        assert 'head hunters' in tracklist
        assert 'sly' in tracklist['head hunters']
        assert normalize_title("It's Herbie Hancock") in tracklist['head hunters'], "The data contains an album by the same name, with a title that's not on the famous Hancock record."

    def test_get_albums_matching_title(self):
        in_mem_discogs = InMemDiscogs(self.path)
        albums = in_mem_discogs.get_albums_matching_title('Head Hunters')
        assert len(albums) == 2
        for album in albums:
            artist = album['artists'][0]
            name = artist['name']
            assert (name == 'Chris Farley' or name == 'Herbie Hancock'), "There are two artists with an album entitle 'Head Hunters'"

    def test_tracklist__filtering(self):
        def filter_farley(result):
            artist = result['artists'][0]
            return artist['name'] != 'Chris Farley'

        in_mem_discogs = InMemDiscogs(self.path, filter_farley)
        tracklist = in_mem_discogs.tracklist()
        assert isinstance(in_mem_discogs.tracklist(), dict), "Expect a dictionary from the tracklist."
        assert 'head hunters' in tracklist
        assert 'sly' in tracklist['head hunters']
        assert normalize_title("It's Herbie Hancock") not in tracklist['head hunters'], "The filter should prevent this from being included."


class TestMatchDiscogs:
    path = '/workspace/tests/test_releases.jsonl'

    def test_match_discogs(self):
        match_discogs = MatchDiscogs(InMemDiscogs(self.path))
        row = (1, 2, "Sly", "Head Hunters", "Herbie Hancock")
        result = match_discogs.matching_discog(row)
        assert result['id'] == 31381, "This should match Herbie Hancock's artist id in the data."

        row = (1, 2, "It's Herbie Hancock", "Head Hunters", "Chris Farley")
        result = match_discogs.matching_discog(row)
        assert result['id'] == 111111, "This should match the surprise entry by Chris Farley"

        row = (1, 2, "Sly", "Head Hunters", "Someone I've never heard of")
        result = match_discogs.matching_discog(row)
        assert result == {}, "An album with a known name and known song by an unknown artist should be an empty record."

        row = (1, 2, "Superunknown", "Superunknown", "Soundgarden")
        assert match_discogs.matching_discog(row) == {}, "A row that matches nothing in the data should return an empty record."
