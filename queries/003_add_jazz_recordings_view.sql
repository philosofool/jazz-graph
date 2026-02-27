/*
    A view of unique jazz recordings.
    This loads very quickly, so it's not materialized.
*/
DROP TABLE IF EXISTS jazz_recordings;
CREATE OR REPLACE VIEW jazz_recordings AS
    SELECT discogs_release.id as discogs_id, recording_first_release.*
    FROM discogs_release
    JOIN discogs_release_to_recording AS dr2r ON discogs_release.id = dr2r.discogs_id
    JOIN recording_first_release ON recording_first_release.recording_id = dr2r.recording_id;