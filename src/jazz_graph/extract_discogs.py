"""There are a few starts her"""

import time
import threading
import discogs_client
import gzip
import xml.etree.ElementTree as ET
from lxml import etree    # pyright: ignore [reportAttributeAccessIssue]
import jsonlines
from pathlib import Path


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


## Parse discogs from their xml dumps.
#  This is very useful for processing the entire
#  discog monthly dump, but not for extracting 1 artist.
# There's a commit, 682ec2d046e6b311b2798eaa1893f7e06e85499e which has draft
# code for using the API. I will probably remove that code,
# so, grab it with git if you decide you want to try something like that.

class DiscogsXMLParser:
    # CREDIT: This was written by ChatGPT with minor edits after debugging.
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
        release_ids = set(release_ids)
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


if __name__ == '__main__':
    discogs_parser = DiscogsXMLParser()
    discogs_parser.parse_masters('/workspace/local_data/discogs_20251101_masters.xml.gz', 'local_data/test_master_parse.jsonl')