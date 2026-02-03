/*
	Recording to performers maps
	a recording to the performers on it, including
	instrument played.
*/

WITH recording_to_performers AS (
	-- SELECT * from link_attribute_credit limit 10;
	
	SELECT 
		artist.name as artist_name, 
		recording.name as song_name, 
		-- l_artist_recording.*, link.*, link_attribute.*, 
		link_attribute_type.name as instrument,
		artist.id as artist_id, 
		recording.id as rec_id
	FROM
		l_artist_recording 
	JOIN recording on recording.id = l_artist_recording.entity1
	JOIN artist on artist.id = l_artist_recording.entity0
	JOIN link on link.id = l_artist_recording.link
	JOIN link_attribute on link_attribute.link = link.id
	JOIN link_attribute_type on link_attribute_type.id = link_attribute.attribute_type
	-- JOIN link_attribute_credit on link_attribute_credit.link = link_attribute.link
	-- WHERE artist.id = 1954 -- Miles Davis
	-- WHERE recording.id = 159726
)
-- SElECT * from recording_to_performers
-- WHERE song_name ILIKE 'Ruby My Dear%';
-- SELECT * FROM recording_to_performers WHERE rec_id = 171985 LIMIT 100;
-- SELECT * FROM track WHERE recording = 171985;
-- SELECT * FROM artist WHERE id = 808120;
SELECT jazz_recordings.artist_name AS act,
    recording_to_perfor
FROM jazz_recordings
JOIN recording_to_performers ON recording_to_performers.rec_id = jazz_recordings.recording;
