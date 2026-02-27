/*
	Recording to performers maps a recording to the
	performers on it, including instrument played.

	NOTE: An artist may be represented multiple times
	in the same recording with this query if they
	play multiple instruments on the song. To create
	a per-recording personel list, GROUP BY artist_id,
	recording_id and SELECT MIN(artist_name), MIN(song_name)
*/

CREATE OR REPLACE VIEW recording_to_performer AS
SELECT
	artist.id AS artist_id,
	recording.id AS recording_id,
	artist.name AS artist_name,
	recording.name AS song_name,
	link_attribute_type.name AS instrument
FROM
	l_artist_recording
JOIN recording ON recording.id = l_artist_recording.entity1
JOIN artist ON artist.id = l_artist_recording.entity0
JOIN link ON link.id = l_artist_recording.link
JOIN link_attribute ON link_attribute.link = link.id
JOIN link_attribute_type ON link_attribute_type.id = link_attribute.attribute_type;
