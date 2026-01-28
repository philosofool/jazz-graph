-- Recording to performers.

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
	

)
-- SElECT * from recording_to_performers
-- WHERE song_name ILIKE 'Ruby My Dear%';
SELECT * FROM recording_to_performers LIMIT 10;
