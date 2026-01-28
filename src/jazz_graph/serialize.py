import json
from pathlib import Path
import jsonlines
from pathlib import Path
from datetime import datetime
import shutil



def serialize_artist(artist):
    """Convert discogs_client.Artist to serializable dict"""
    return {
        'id': artist.id,
        'name': artist.name,
        'real_name': getattr(artist, 'real_name', None),
        'profile': getattr(artist, 'profile', None),
        'urls': getattr(artist, 'urls', []),
        'namevariations': getattr(artist, 'namevariations', []),
    }

def serialize_release(release):
    """Convert discogs_client.Release to serializable dict"""
    return {
        'id': release.id,
        'title': release.title,
        'year': getattr(release, 'year', None),
        'genres': getattr(release, 'genres', []),
        'styles': getattr(release, 'styles', []),
        'formats': [str(f) for f in getattr(release, 'formats', [])],
        'labels': [
            {'name': label.name, 'catno': label.catno}
            for label in getattr(release, 'labels', [])
        ],
        'artists': [
            {'id': artist.id, 'name': artist.name}
            for artist in getattr(release, 'artists', [])
        ],
        'tracklist': [
            {'position': track.position, 'title': track.title, 'duration': track.duration}
            for track in getattr(release, 'tracklist', [])
        ],
        'credits': [
            {'name': credit.name, 'role': credit.role}
            for credit in getattr(release, 'credits', [])
        ] if hasattr(release, 'credits') else [],
    }


class DiscogsCache:
    """Write discogs data to jsonlines format."""
    def __init__(self, cache_dir='data/discogs_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artists_file = self.cache_dir / 'artists.jsonl'
        self.releases_file = self.cache_dir / 'releases.jsonl'

    def _backup_if_exists(self, filepath):
        """Create timestamped backup if file exists"""
        filepath = Path(filepath)
        if filepath.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"back_{timestamp}_{filepath.name}"
            backup_path = filepath.parent / backup_name
            shutil.copy2(filepath, backup_path)
            print(f"Backed up {filepath.name} to {backup_name}")

    def save_artist(self, artist):
        """Append artist to current cache file."""
        # No backup for append mode - we're adding, not replacing
        with jsonlines.open(self.artists_file, 'a') as f:
            f.write(serialize_artist(artist))

    def save_release(self, release):
        """Append release to current cache file."""
        # No backup for append mode - we're adding, not replacing
        with jsonlines.open(self.releases_file, 'a') as f:
            f.write(serialize_release(release))

    def save_artists_batch(self, artists):
        """Save entire list, replacing file (creates backup)"""
        self._backup_if_exists(self.artists_file)
        with jsonlines.open(self.artists_file, 'w') as f:
            for artist in artists:
                f.write(serialize_artist(artist))

    def save_releases_batch(self, releases):
        """Save entire list, replacing file (creates backup)"""
        self._backup_if_exists(self.releases_file)
        with jsonlines.open(self.releases_file, 'w') as f:
            for release in releases:
                f.write(serialize_release(release))

    def load_artists(self):
        if self.artists_file.exists():
            with jsonlines.open(self.artists_file, 'r') as f:
                return list(f)
        return []

    def load_releases(self):
        if self.releases_file.exists():
            with jsonlines.open(self.releases_file, 'r') as f:
                return list(f)
        return []

    def list_backups(self):
        """List all backup files"""
        backups = sorted(self.cache_dir.glob('back_*'))
        for backup in backups:
            print(f"{backup.name} ({backup.stat().st_size / 1024:.1f} KB)")
        return backups

    def restore_backup(self, backup_name):
        """Restore a specific backup"""
        backup_path = self.cache_dir / backup_name
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup {backup_name} not found")

        # Determine target file
        if 'artists' in backup_name:
            target = self.artists_file
        elif 'releases' in backup_name:
            target = self.releases_file
        else:
            raise ValueError("Can't determine target file from backup name")

        # Backup current file before restoring
        self._backup_if_exists(target)

        # Restore
        shutil.copy2(backup_path, target)
        print(f"Restored {backup_name} to {target.name}")