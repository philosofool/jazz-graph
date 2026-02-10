/*
	Performance to albums.
	This collects distinct recordings from the data
	grouped by album: one row per recording per album
	on which it has appeared. Albums (which may duplicate
	on release info--e.g. Europena vs. US release,
	rereleases, etc.) are deduplicated on the earliest
	certain date information available, i.e., first release.

	For this project, the value of this table is that
	we can match against discogs title, album, artist
	names to extract valid jazz ablums and metadata
	about them which is unavailable in musicbrainz.
*/
-- DROP MATERIALIZED VIEW IF EXISTS recording_to_album;
-- CREATE MATERIALIZED VIEW IF NOT EXISTS recording_to_album
SELECT DISTINCT ON (recording.id, release_group.id)
	recording.id as recording_id,
	release_group.id as release_group_id,
	recording.name as title,
	release_group.name as album,
	artist_credit.name as artist,
	COALESCE(
	    MAKE_DATE(date_year, date_month, date_day),
	    (
	        MAKE_DATE(
	            COALESCE(date_year, 2026),
	            COALESCE(date_month, 12),
	            1
	        )
	        + INTERVAL '1 month'
	        - INTERVAL '1 day'
	    )::date
	) AS release_date
	-- release_event.date_year as release_year,
	-- release_event.date_month as release_month,
	-- release_event.date_day as release_day
FROM recording
JOIN artist_credit ON artist_credit.id = recording.artist_credit
JOIN track on track.recording = recording.id
JOIN medium on medium.id = track.medium
JOIN release ON release.id = medium.release
JOIN release_group on release_group.id = release.release_group
JOIN release_event ON release_event.release = release.id
-- adding release group assures first release is the one included.
ORDER BY recording_id, release_group_id, release_date
;
-- Drop this index and rename?
-- DROP INDEX IF EXISTS performers_to_album_idx;
-- DROP INDEX IF EXISTS recording_to_album_idx;
-- CREATE INDEX recording_to_album_idx ON recording_to_album (recording_id)
