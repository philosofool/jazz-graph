-- SELECT * FROM artist LIMIT 10;
WITH compositions AS (
	SELECT
		l_artist_work.entity0 as composer_id,
		l_recording_work.entity0 as recording_id,
		work.name as song,
		recording.name as recording,
		recording.artist_credit as performer_id,
		artist.name as composer
		
	FROM work 
	JOIN l_artist_work ON work.id = l_artist_work.entity1
	JOIN l_recording_work ON work.id = l_recording_work.entity1
	-- JOIN recording ON recording.id = l_recording_work.entity0
	-- JOIN artist ON artist.id = recording.artist_credit
	JOIN artist ON artist.id = l_artist_work.entity0
)

SELECT * FROM compositions 
WHERE
	composer_id = 1954
LIMIT 100;
