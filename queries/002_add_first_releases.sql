CREATE MATERIALIZED VIEW recording_first_release AS
    SELECT DISTINCT ON (recording_id)
    *
    FROM recording_to_album
    ORDER BY recording_id, release_date;
CREATE INDEX recording_first_release_idx ON recording_first_release (recording_id);