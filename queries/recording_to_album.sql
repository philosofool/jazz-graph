/*
	Performance to albums.
	WIP.
	This collects distinct recordings from the data
	grouped by album. So, basically every album in the
	data with one row per song on the album.
	An album is a release group, however, rereleases
	will create album dublicates ("Legacy Edition!")
	We want to deduplicate these.

	What's this for? The main role here is to filter
*/
WITH recording_to_album AS (

	SELECT
		MIN(recording.name) as title,
		MIN(release_group.name) as album,
		recording.id as performance_id,
		release_group.id as album_id,
		MIN(release_event.date_year) as release_year,
		MIN(release_event.date_month) as release_month,
		MIN(release_event.date_day) as release_day
	FROM recording
	JOIN track on track.recording = recording.id
	JOIN medium on medium.id = track.medium -- may cause duplication on medium. There were a comment like this, but groupby should handle.
	JOIN release ON release.id = medium.release
	JOIN release_group on release_group.id = release.release_group
	JOIN release_event ON release_event.release = release.id
	JOIN l_release_group_release_group ON l_release_group_release_group.entity0 = release_group.id
	JOIN medium_format ON medium_format.id = medium.format
	GROUP BY recording.id, release_group.id
)

SELECT * FROM recording_to_album;