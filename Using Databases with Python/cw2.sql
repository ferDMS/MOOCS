-- Drop tables already existing (and their values)
DROP TABLE IF EXISTS Track
DROP TABLE IF EXISTS Album
DROP TABLE IF EXISTS Artist
DROP TABLE IF EXISTS Genre

-- Based on a schema, create the tables and describe relations with keys
CREATE TABLE Genre (
	genreId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	name VARCHAR(50)
)

CREATE TABLE Artist (
	artistId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	name VARCHAR(50)
)

CREATE TABLE Album (
	albumId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	artistId INT FOREIGN KEY REFERENCES Artist(artistId),
	title VARCHAR(50)
)

CREATE TABLE Track (
	trackId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	albumId INT FOREIGN KEY REFERENCES Album(albumId),
	artistId INT FOREIGN KEY REFERENCES Artist(artistId),
	genreId INT FOREIGN KEY REFERENCES Genre(genreId),
	title VARCHAR(100),
	len INT,
	rating INT,
	count INT
)

-- Inserting values into the tables
INSERT INTO Artist(name) VALUES ('Led Zepellin'), ('ACDC')
INSERT INTO Genre(name) VALUES ('Rock'), ('Metal')
INSERT INTO Album(artistId, title) VALUES (2, 'Who Made Who'), (1, 'IV')
INSERT INTO Track(title, rating, len, count, albumId, genreId) VALUES
	('Black Dog', 5, 297, 0, 2, 1),
	('Stairway', 5, 482, 0, 2, 1),
	('About to Rock', 5, 313, 0, 1, 2),
	('Who Made Who', 5, 207, 0, 1, 2)
UPDATE Track SET artistId = Album.artistId FROM Album JOIN Track ON Album.albumId = Track.albumId

-- Selecting through relational SQL with joins and where clauses
SELECT Artist.name, Genre.name FROM Track INNER JOIN Genre ON Track.genreId = Genre.genreId INNER JOIN Album ON Track.albumId = Album.albumId INNER JOIN Artist ON Album.artistId = Artist.artistId;

SELECT Artist.name, Genre.name FROM Track, Genre, Album, Artist WHERE Track.genreId = Genre.genreId AND Track.albumId = Album.albumId AND Album.artistId = Artist.artistId;