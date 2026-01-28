from pathlib import Path
from jazz_graph.extract_discogs import DiscogsXMLParser


def main():
    """Extract discogs data to jsonlines format, which is much easier to work with.

    Create three files:
    1. jazz_masters.jsonl
        This is a listing of the master recordings and should not contain duplicates (e.g., reissues.)
    2. jazz_releases.jsonl
        This is the actual data associated with a master recording, there is one data entry per master.
    3. jazz_artists.jsonl
        This is a list of performers who appear in jazz releases, and should be deduplicated.
    Note that the deduplication in these depends on the cleanliness of the discogs data,
    not data cleaning steps. Discogs appears to be very clean, but duplication is possible.
    """
    # Step 1: Parse masters.xml to get jazz albums
    parser = DiscogsXMLParser()
    masters_path = 'local_data/jazz_masters.jsonl'
    if not Path(masters_path).exists():
        parser.parse_masters('local_data/discogs_20251101_masters.xml.gz', masters_path)

    # Step 2: Filter for jazz
    import jsonlines

    jazz_masters = []
    main_release_ids = set()

    with jsonlines.open(masters_path, 'r') as f:
        for master in f:
            # Filter: has 'Jazz' genre
            if 'Jazz' in master.get('genres', []):
                jazz_masters.append(master)
                if master['main_release']:
                    main_release_ids.add(master['main_release'])

    print(f"Found {len(jazz_masters)} jazz masters")
    print(f"Need to fetch {len(main_release_ids)} main releases")

    # Save filtered jazz masters
    with jsonlines.open('local_data/jazz_masters_filtered.jsonl', 'w') as f:
        for master in jazz_masters:
            f.write(master)

    # Step 3: Extract those specific releases from releases.xml
    if not Path('local_data/jazz_releases.jsonl').exists():
        parser.parse_releases(
            'local_data/discogs_20251101_releases.xml.gz',
            release_ids=main_release_ids,
            output_jsonl='local_data/jazz_releases.jsonl'
        )

    # Step 4: Extract all jazz artists mentioned
    artist_ids = set()
    with jsonlines.open('local_data/jazz_releases.jsonl', 'r') as f:
        for release in f:
            for artist in release.get('artists', []):
                if artist['id']:
                    artist_ids.add(artist['id'])
            for credit in release.get('credits', []):
                # Credits don't have IDs in XML, unfortunately
                # Would need to match by name later
                pass

    print(f"Found {len(artist_ids)} unique artists")

    ## Step 5: Extract those artists
    parser.parse_artists(
        'local_data/discogs_20251101_artists.xml.gz',
        artist_ids=artist_ids,
        output_jsonl='local_data/jazz_artists.jsonl'
    )


if __name__ == '__main__':
    main()
    # import os
    # from jazz_graph.serialize import DiscogsCache
    # cache = DiscogsCache()
    # discog_api_key = os.environ.get('DISCOG_API_KEY')
    # if discog_api_key is None:
    #     raise Exception("Unable to load DISCOG_API_KEY from environment.")
    # client = RateLimitedClient('JazzWork-nn/0.1', user_token=discog_api_key)
    # john_coltrane = client.artist(97545)
    # blue_train = client.release(3022494)

    # assert john_coltrane.name == "John Coltrane"
    # assert blue_train.title == "Blue Train"
    # artists, releases = extract_jazz([], [blue_train], cache)

    # artists, releases = extract_jazz([john_coltrane], [])

    # discog_parser = DiscogsXMLParser()
    # discog_parser.parse_masters('local_data/discogs_20251101_masters.xml.gz', n_records=10)
