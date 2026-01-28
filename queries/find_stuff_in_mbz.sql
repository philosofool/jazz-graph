SELECT * FROM work WHERE name ILIKE 'Ruby%' LIMIT 10;
SELECT * FROM release_group 
JOIN release ON release.release_group = release_group.id
-- JOIN release_group on release_group.id = release.release_group
JOIN release_event ON release_event.release = release.id 
WHERE
	release_group.name ILIKE 'Genius of Mo%'
LIMIT 10;
-- SELECT * from l_artist_work 
-- JOIN work ON work.id = l_artist_work.entity1
-- WHERE 
-- 	-- l_artist_work.entity0 = 23412 
-- 	l_artist_work.entity0 = 1954
-- LIMIT 100;
-- SELECT * from l_recording_work
-- WHERE l_recording_work.entity0 = 159729;
-- SELECT * from work WHERE work.id = 159729;
SELECT * 
FROM
	release_group
JOIN l_recording_release_group on l_recording_release_group.entity1 = release_group.id