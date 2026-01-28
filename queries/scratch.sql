WITH discogs AS (
	SELECT 'Kind of Blue' AS album, 'Miles Davis' AS artist, 1959 AS year UNION ALL
	SELECT 'Genuis of Modern Music' AS album, 'Thelonious Monk' AS artist, 1951 AS year UNION ALL
	SELECT 'Saxophone Colossus' AS album, 'Sonny Rollins' AS artist, 1957 AS year
),
releases AS (
	SELECT 
		MIN(recording.name) as title, 
		MIN(release_group.name) as album,
		recording.id as performance_id, 
		release_group.id as album_id,
		MIN(release_event.date_year) as release_year, 
		MIN(release_event.date_month),
		MIN(release_event.date_day)
		-- medium.format, medium_format.name
	FROM recording
	JOIN track on track.recording = recording.id
	JOIN medium on medium.id = track.medium -- mistake-causes duplication by medium
	JOIN release ON release.id = medium.release
	JOIN release_group on release_group.id = release.release_group
	-- JOIN artist on artist.id = l_artist_recording.entity0
	-- JOIN artist_credit ON artist_credit.id = recording.artist_credit
	-- JOIN release on release.record_group = record_group.id
	-- JOIN release_group on release_group.id = release.release_group
	JOIN release_event ON release_event.release = release.id
	JOIN l_release_group_release_group ON l_release_group_release_group.entity0 = release_group.id
	GROUP BY recording.id, release_group.id
)
SELECT * from discogs
LEFT JOIN releases ON releases.album = discogs.album and releases.release_year = discogs.year
LIMIT 100;
