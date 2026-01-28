# TODO: rename this file.

from collections.abc import Callable

from functools import cache
import re
from unittest.main import MAIN_EXAMPLES
from discogs_client import Client
import functools
import os

@functools.cache
def get_discogs_client():
    discog_api_key = os.environ.get('DISCOG_API_KEY')
    if discog_api_key is None:
        raise Exception("Unable to load DISCOG_API_KEY from environment.")
    client = Client('YourAppName/0.1', user_token=discog_api_key)
    return client


def get_colaborators(artist):
    ...

def is_master(release):
    ...

def get_seed_associated_releases() -> list[int]:
    """Return all main releases, by id, associated with Miles Davis."""
    client = get_discogs_client()
    seed = 'Miles Davis'
    results = client.search(seed, type='artist')
    miles = results.page(0)[0]
    assert miles.name == "Miles Davis", "Miles Davis should the result we start with."
    # not sure this will work. Might have to do while until something or other...
    i = 0
    main_releases = []
    while True:
        releases = miles.releases.page(i)
        i += 1
        for release in releases:
            if not is is_master(release):
                continue
                main_releases.append(release.main_release.id)
    return main_releases

def get_associated_artists(release_ids, filter: Callable) -> list[int]:
    client = get_discogs_client()
    associated_artists = []
    for release_id in release_ids:
        release = client.release(release_id)
        for artist in release.credits:
            if not filter(artist):
                continue
            associated_artists.append(artist.id)
    return associated_artists

def get_main_releases(releases) -> list[int]:
    """Return main releases from any master in releases."""
    main_releases = []
    for release in releases:
        if 'Jazz' not in release.genres:
            continue
        if is_master(release):
            main_releases.append(release.main_release.id)
    return main_releases

def get_associated_releases(artist_id):
    client = get_discogs_client()
    artist = client.artist(artist_id)
    main_releases = []
    while True:
        try:
            releases = artist.releases.page(i)
        except Exception:
            # we're out of pages or gaia error?
            break
        i += 1
        main_releases.extend(get_main_releases(releases))

    return main_releases


class ExtractDiscogsJazzArtists:
    def __init__(self):
        self._client = get_discogs_client()
        self.artists = {}
        self.releases = {}
        self.artist_credits = defaultdict(set)

    def update_data(self, artist_id: int | None, release_id: int | None):
        if artist_id is None and release_id is not None:
            self._update_from_release(release_id)
        elif release_id is None:
            self._update_from_artist(artist_id)
        else:
            raise ValueError("At least one must be not ")

    def _update_from_release(self, release_id):
        release = self._client.release(release_id)
        if (master_id := release.master.main_release.id) != release_id:
            release = self._client.release(master_id)
            release_id = release.id
        if self.is_jazz_release(release):
            self.releases[release_id] = release
        current_artists = set(self.artists)
        self.add_associated_artists(release_id)
        new_artist


    def _update_from_artist(self, artist_id):
        ...


    def add_associated_artists(self, release_id):
        release = self._client.release(release_id)
        if not self.is_jazz_release(release):
            return
        for artist in release.credits:
            self.artist_credits[artist.id].add(release.id)
            self._add_if_jazz_artist(artist)

    def add_associated_releases(self, artist_id):
        artist = self._client.artist(artist_id)
        for release in artist.releases:
            # is there an equivalent of artist_credits?
            if self.is_jazz_release(release):
                self.releases[release.id] = release

    def is_jazz_release(self, release) -> bool:
        return 'Jazz' in release.genres

    def _add_if_jazz_artist(self, artist):
        if artist.id in self.artists:
            return

        if len(self.artist_credits[artist.id]) > 10:
            self.artists[artist.id] = artist
            return

        if not self.is_instrumentalist(artist):
            return

        n_jazz_releases = 0
        n_releases = 0
        for release in artist.releases:
            n_releases += 1
            if self.is_jazz_release(release):
                n_jazz_releases += 1
        if n_jazz_releases / n_releases >= .5:
            self.artists[artist.id] = artist
