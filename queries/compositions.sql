CREATE OR REPLACE VIEW compositions AS
	SELECT
		l_artist_work.entity0 as composer_id,
		work.id as work_id,
		l_recording_work.entity0 as recording_id,
		link.id as link_id,
		link.link_type as link_type,
		recording.artist_credit as performer_id,
		work.name as song_title,
		artist.name as composer,
		recording.name as recording_name

	FROM work
	JOIN l_artist_work ON work.id = l_artist_work.entity1
	JOIN l_recording_work ON work.id = l_recording_work.entity1
	JOIN recording ON recording.id = l_recording_work.entity0
	JOIN artist ON artist.id = l_artist_work.entity0
	JOIN link ON link.id = l_artist_work.link
	WHERE link.link_type = 168;