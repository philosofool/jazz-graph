"""There are a few starts her"""

import time
import threading
import discogs_client
from jazz_graph.serialize import DiscogsCache

## Several code for working through the discogs API.
#  This is too slow for working with large batch processing.
#  Could be very useful in the future for monthly releases
#  and other updating. Concept: show me stuff on discogs of this
#  master release.

class RateLimitedClient(discogs_client.Client):
    def __init__(self, user_agent: str, per_second: float = 0.41, *args, **kwargs):
        super().__init__(user_agent, *args, **kwargs)
        self._lock = threading.Lock()
        self._interval = 1.0 / per_second # default is ~24.9 per second, which is the limit.
        self._last_call = 0.0

    def _throttled_request(self, method, url, *args, **kwargs):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.time()
        return super()._request(method, url, *args, **kwargs)

    # override the discogs_client request mechanism
    def _request(self, method, url, *args, **kwargs):
        return self._throttled_request(method, url, *args, **kwargs)


PERFORMER_ROLES: set[str] = {
    # A very broad list of roles which involve performance of a song.
    "Vocals",
    "Lead Vocals",
    "Backing Vocals",
    "Acoustic Guitar",
    "Electric Guitar",
    "Bass",
    "Synthesizer",
    "Keyboards",
    "Piano",
    "Drums",
    "Percussion",
    "Flute",
    "Saxophone",
    "Trumpet",
    "Trombone",
    "Violin",
    "Cello",
    "Mandolin",
    "Banjo",
    "Harmonica",
    "Organ",
    "Hammond Organ",
    "Electric Piano",
    "Lead Guitar",
    "Rhythm Guitar",
    "Bass Guitar",
    "Double Bass",
    "Synth Bass",
    # "Programming",  # borderline – often part of modern performance
    # "Sampler",      # likewise
}


def double_dfs(tree_1, tree_2):
    """This is just an express of the process."""
    seen_1 = set()
    seen_2 = set()
    while tree_1 or tree_2:
        elem_1 = tree_1.pop()
        if elem_1 not in seen_1:
            for new_elem in add_tree_elems(elem_1, seen_2):
                tree_2.append(new_elem)
        elem_2 = tree_2.pop()
        if elem_2 not in seen_2:
            for new_elem in add_tree_elems(elem_2, seen_1):
                tree_1.append(new_elem)

def add_tree_elems(elem, seen):
    """Helper."""
    for elem_of_tree_type in elem:
        if elem_of_tree_type in seen:
            continue
        yield elem_of_tree_type
        seen.add(elem_of_tree_type)

def extract_jazz(artists: list, releases: list, cache: DiscogsCache, max_iter=10) -> tuple[list, list]:
    """Given seed list of artists and releases, traverse discogs to find other releases and artists."""
    seen_artists = set()
    seen_releases = set()
    all_artists = []
    all_releases = []
    n_iter = 0
    while artists or releases:
        if artists and artists[-1].id not in seen_artists:
            artist = artists.pop()
            all_artists.append(artist)
            for new_release in add_releases(artist, seen_artists, seen_releases):
                releases.append(new_release)
            # once all the artist's releases have been delivered to the releases, we don't need to traverse the artist again.
            seen_artists.add(artist.id)

        if releases and releases[-1].id not in seen_releases:
            release = releases.pop()
            all_releases.append(release)
            for new_artist in add_artists(release, seen_artists, seen_releases):
                artists.append(new_artist)
            # once all the release's artists have been delivered to the artists, we don't need to traverse the release again.
            seen_releases.add(release.id)
        n_iter += 1
        cache.save_artists_batch(all_artists)
        cache.save_releases_batch(all_releases)
        if n_iter == max_iter:
            break
    return all_artists, all_releases

def add_releases(artist, seen_artists, seen_releases):
    """Yield any jazz musician associated with the release."""
    for release in artist.releases:
        if release.id in seen_releases:
            continue
        try:
            main_release = get_main_release(release)
        except MasterlessRecording:
            seen_releases.add(release.id)
            continue
        if main_release.id not in seen_releases:
            if is_jazz_release(main_release):
                yield main_release
        if release.id != main_release.id:
            # the main release, which is what we care about is yielded to the main body function.
            # this is safely ignored once it has been, since it's "proper children" are
            # handled when the main release is.
            seen_releases.add(release.id)

def add_artists(release, seen_artists, seen_releases):
    """Yeild any jazz release associated with the artist."""
    seen_releases.add(release.id)
    for artist in release.credits:
        if artist.id in seen_artists:
            continue
        if is_jazz_artist(artist):
            yield artist

def is_jazz_artist(artist):
    """Return true if the artist released mostly jazz albums.

    # NOTE: this is NOT the best test of whether an artist is a jazz musician.
    #       Main purpose is to help traverse releases which are likely jazz. If most of the artists
    #       releases are Jazz, then we should look at them.
    """
    n_jazz_releases = 0
    n_releases = 0
    if not plays_jazz_instrument(artist):
        # Filters artist credits for Rudy van Gelder, etc.
        # I'm not denying the credit here, but these aren't going to find releases.
        return False
    for release in artist.releases:
        n_jazz_releases += is_jazz_release(release)
        n_releases += 1
        # depends on sort order, is it okay?
        if n_releases >= 20:
            break
    if n_releases == 0:
        return False
    return n_jazz_releases / n_releases >= .5

def plays_jazz_instrument(artist):
    return artist.role in PERFORMER_ROLES

def is_jazz_release(release):
    return 'Jazz' in release.genres

class MasterlessRecording(Exception):
    ...

def get_main_release(release):
    """Get the main release associated with the release.

    Raises MasterlessRecording if it is a release with no master.
    """

    try:
        # in this case, the relase is a master
        return release.main_release
    except AttributeError:
        # release is not a master, so get the master.
        master = release.master
    if master is not None:
        return master.main_release
    # singles, EPs, may not have a master.
    # we are limited to main releases to reduce overhead.
    raise MasterlessRecording()


## Parse discogs from their xml dumps.
#  This is very useful for processing the entire
#  discog monthly dump, but not for extracting 1 artist.

import gzip
import xml.etree.ElementTree as ET
from lxml import etree    # pyright: ignore [reportAttributeAccessIssue]
import jsonlines
from pathlib import Path

class DiscogsXMLParser:
    def __init__(self, data_dir='local_data/discogs_dumps'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def parse_masters(self, xml_path, output_jsonl='local_data/masters.jsonl', n_records=None):
        """
        Extract main_release from each master
        Master = canonical version of an album
        Main_release = the "primary" release to use
        """
        print(f"Parsing {xml_path}...")
        # Open gzipped XML
        with gzip.open(xml_path, 'rb') as f:
            context = etree.iterparse(f, events=('end',), tag='master')

            with jsonlines.open(output_jsonl, 'w') as out:
                n_records_parse = 0
                for event, elem in context:
                    master_data = self._parse_master(elem)
                    if master_data:
                        out.write(master_data)
                    n_records_parse += 1
                    if n_records is not None and n_records_parse == n_records:
                        break
                    # Clear element to free memory
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

                    if master_data and master_data['id'] % 10000 == 0:
                        print(f"  Processed {master_data['id']} masters...")

        print(f"Done! Output: {output_jsonl}")

    def _parse_master(self, elem):
        """Extract relevant fields from master XML element"""
        master = {
            'id': int(elem.get('id')),
            'main_release': None,
            'artists': [],
            'genres': [],
            'styles': [],
            'year': None,
            'title': None,
        }

        # Main release ID (this is what we want!)
        main_release = elem.find('main_release')
        if main_release is not None:
            master['main_release'] = int(main_release.text)

        # Title
        title = elem.find('title')
        if title is not None:
            master['title'] = title.text

        # Year
        year = elem.find('year')
        if year is not None and year.text:
            try:
                master['year'] = int(year.text)
            except ValueError:
                pass

        # Artists
        artists = elem.find('artists')
        if artists is not None:
            for artist in artists.findall('artist'):
                name = artist.find('name')
                artist_id = artist.find('id')
                if name is not None and artist_id is not None:
                    master['artists'].append({
                        'id': int(artist_id.text),
                        'name': name.text
                    })

        # Genres
        genres = elem.find('genres')
        if genres is not None:
            master['genres'] = [g.text for g in genres.findall('genre')]

        # Styles
        styles = elem.find('styles')
        if styles is not None:
            master['styles'] = [s.text for s in styles.findall('style')]

        return master

    def parse_releases(self, xml_path, release_ids, output_jsonl='local_data/releases.jsonl'):
        """
        Extract specific releases by ID
        release_ids: set of release IDs to extract (from masters)
        """
        print(f"Parsing {xml_path} for {len(release_ids)} specific releases...")
        release_ids = set(release_ids)  # Fast lookup
        found = 0

        with gzip.open(xml_path, 'rb') as f:
            context = etree.iterparse(f, events=('end',), tag='release')

            with jsonlines.open(output_jsonl, 'w') as out:
                for event, elem in context:
                    release_id = int(elem.get('id'))

                    # Only process if this is a main_release we want
                    if release_id in release_ids:
                        release_data = self._parse_release(elem)
                        out.write(release_data)
                        found += 1

                        if found % 1000 == 0:
                            print(f"  Found {found}/{len(release_ids)} releases...")

                    # Clear memory
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

        print(f"Done! Found {found} releases. Output: {output_jsonl}")

    def _parse_release(self, elem):
        """Extract detailed release info"""
        release = {
            'id': int(elem.get('id')),
            'title': None,
            'artists': [],
            'labels': [],
            'formats': [],
            'genres': [],
            'styles': [],
            'tracklist': [],
            'credits': [],
            'released': None,
        }

        # Title
        title = elem.find('title')
        if title is not None:
            release['title'] = title.text

        # Released date
        released = elem.find('released')
        if released is not None:
            release['released'] = released.text

        # Artists
        artists = elem.find('artists')
        if artists is not None:
            for artist in artists.findall('artist'):
                name = artist.find('name')
                artist_id = artist.find('id')
                if name is not None:
                    release['artists'].append({
                        'id': int(artist_id.text) if artist_id is not None else None,
                        'name': name.text
                    })

        # Labels
        labels = elem.find('labels')
        if labels is not None:
            for label in labels.findall('label'):
                release['labels'].append({
                    'name': label.get('name'),
                    'catno': label.get('catno')
                })

        # Genres
        genres = elem.find('genres')
        if genres is not None:
            release['genres'] = [g.text for g in genres.findall('genre')]

        # Styles
        styles = elem.find('styles')
        if styles is not None:
            release['styles'] = [s.text for s in styles.findall('style')]

        # Tracklist
        tracklist = elem.find('tracklist')
        if tracklist is not None:
            for track in tracklist.findall('track'):
                position = track.find('position')
                title = track.find('title')
                duration = track.find('duration')

                release['tracklist'].append({
                    'position': position.text if position is not None else None,
                    'title': title.text if title is not None else None,
                    'duration': duration.text if duration is not None else None,
                })

        # Credits (musicians!)
        extraartists = elem.find('extraartists')
        if extraartists is not None:
            for artist in extraartists.findall('artist'):
                name = artist.find('name')
                role = artist.find('role')
                if name is not None:
                    release['credits'].append({
                        'name': name.text,
                        'role': role.text if role is not None else None
                    })

        return release

    def parse_artists(self, xml_path, artist_ids, output_jsonl='data/artists.jsonl'):
        """Extract specific artists by ID"""
        print(f"Parsing {xml_path} for {len(artist_ids)} artists...")
        artist_ids = set(artist_ids)
        found = 0

        with gzip.open(xml_path, 'rb') as f:
            context = etree.iterparse(f, events=('end',), tag='artist')

            with jsonlines.open(output_jsonl, 'w') as out:
                for event, elem in context:
                    # ID is a child element, not an attribute
                    id_elem = elem.find('id')
                    if id_elem is not None and id_elem.text:
                        artist_id = int(id_elem.text)

                        if artist_id in artist_ids:
                            artist_data = self._parse_artist(elem)
                            out.write(artist_data)
                            found += 1

                            if found % 1000 == 0:
                                print(f"  Found {found}/{len(artist_ids)} artists...")

                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

        print(f"Done! Found {found} artists.")

    def _parse_artist(self, elem):
        """Extract artist info"""
        id_elem = elem.find('id')
        artist_id = int(id_elem.text) if id_elem is not None and id_elem.text else None

        name_elem = elem.find('name')
        realname_elem = elem.find('realname')
        profile_elem = elem.find('profile')

        return {
            'id': artist_id,
            'name': name_elem.text if name_elem is not None else None,
            'realname': realname_elem.text if realname_elem is not None else None,
            'profile': profile_elem.text if profile_elem is not None else None,
        }

def parse_artists_debug(self, xml_path, artist_ids, output_jsonl='data/artists.jsonl'):
    """Extract specific artists by ID - with debugging"""
    print(f"Parsing {xml_path} for {len(artist_ids)} artists...")
    artist_ids = set(artist_ids)
    found = 0

    with gzip.open(xml_path, 'rb') as f:
        context = etree.iterparse(f, events=('end',), tag='artist')

        with jsonlines.open(output_jsonl, 'w') as out:
            for i, (event, elem) in enumerate(context):
                # DEBUG: Print first few elements
                if i < 3:
                    print(f"\n=== Artist {i} ===")
                    print(f"Element tag: {elem.tag}")
                    print(f"Element attrib: {elem.attrib}")
                    print(f"Element keys: {elem.keys()}")
                    print(f"Direct children: {[child.tag for child in elem]}")

                    # Try to find ID
                    for child in elem:
                        print(f"  {child.tag}: {child.text}")

                # Try getting ID different ways
                artist_id = elem.get('id')
                if artist_id is None:
                    # Maybe ID is a child element, not an attribute
                    id_elem = elem.find('id')
                    if id_elem is not None:
                        artist_id = id_elem.text

                if artist_id:
                    artist_id = int(artist_id)

                    if artist_id in artist_ids:
                        artist_data = self._parse_artist(elem)
                        out.write(artist_data)
                        found += 1
                else:
                    if i < 3:
                        print("WARNING: No ID found!")

                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

                if i >= 10:  # Stop after 10 for debugging
                    break

    print(f"Done! Found {found} artists.")
