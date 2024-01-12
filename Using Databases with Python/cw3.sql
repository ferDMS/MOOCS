INSERT INTO Artist(name) VALUES ('Led Zepellin'), ('ACDC')
INSERT INTO Genre(name) VALUES ('Rock'), ('Metal')
INSERT INTO Album(artistId, title) VALUES (2, 'Who Made Who'), (1, 'IV')
INSERT INTO Track(title, rating, len, count, albumId, genreId) VALUES
	('Black Dog', 5, 297, 0, 2, 1),
	('Stairway', 5, 482, 0, 2, 1),
	('About to Rock', 5, 313, 0, 1, 2),
	('Who Made Who', 5, 207, 0, 1, 2)
UPDATE Track SET artistId = Album.artistId FROM Album JOIN Track ON Album.albumId = Track.albumId
