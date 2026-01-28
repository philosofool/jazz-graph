-- performance_to_album
WITH recording_to_album AS (

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
	-- JOIN medium_format ON medium_format.id = medium.format
	WHERE 
		-- recording.name LIKE '%Freddie Freeloader%'
		-- AND recording.artist_credit = 1954
		recording.artist_credit = 1954

		
		-- AND release_group.name LIKE 'Kind of Blue%'
	-- GROUP BY release_group.id, recording.id
	-- LIMIT 10
	GROUP BY recording.id, release_group.id
)
SELECT
	*
FROM recording_to_album
JOIN l_release_group_release_group as group_links ON 
	group_links.entity0 = recording_to_album.album_id
JOIN link on link.id = group_links.link
JOIN link_type on link_type.id = link.link_type
WHERE performance_id != 14535669;
	

-- SELECT * 
-- from release_group
-- JOIN artist_credit ON artist_credit.id = release_group.artist_credit
-- JOIN recording ON recording.artist_credit = release_group.artist_credit
-- -- JOIN release on artist_credit.id = release.artist_credit
-- WHERE
--   release_group.id = 36073

-- LIMIT 20;

-- select * from l_release_group_release_group limit 10;
-- select * from LINK 
-- JOIN link_type on link_type.id = link.link_type
-- WHERE entity_type0 = 'release_group' and entity_type1 = 'release_group'
-- LIMIT 10;
-- SELECT * from release_event limit 10;
-- SELECT * from l_release_work LIMIT 10;
