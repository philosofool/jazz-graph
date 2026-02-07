/*
	Performance to albums.
	This collects distinct recordings from the data
	grouped by album. So, basically every album in the
	data with one row per song on the album.
*/
CREATE MATERIALIZED VIEW IF NOT EXISTS recording_to_album AS
SELECT
	recording.id as recording_id,
	release_group.id as release_group_id,
	recording.name as title,
	release_group.name as album,
	artist_credit.name as artist,
	release_event.date_year as release_year,
	release_event.date_month as release_month,
	release_event.date_day as release_day
FROM recording
JOIN artist_credit ON artist_credit.id = recording.artist_credit
JOIN track on track.recording = recording.id
JOIN medium on medium.id = track.medium
JOIN release ON release.id = medium.release
JOIN release_group on release_group.id = release.release_group
JOIN release_event ON release_event.release = release.id;
CREATE INDEX performers_to_album_idx ON recording_to_album (recording_id)
