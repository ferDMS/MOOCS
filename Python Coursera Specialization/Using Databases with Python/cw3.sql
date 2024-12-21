-- Drop tables already existing (and their values)
DROP TABLE IF EXISTS cw3_Track;
DROP TABLE IF EXISTS cw3_Album;
DROP TABLE IF EXISTS cw3_Artist;
DROP TABLE IF EXISTS cw3_Genre;

-- Based on a schema, create the tables and describe relations with keys
CREATE TABLE cw3_Genre (
	genreId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	name VARCHAR(50)
);

CREATE TABLE cw3_Artist (
	artistId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	name VARCHAR(50)
);

CREATE TABLE cw3_Album (
	albumId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	artistId INT FOREIGN KEY REFERENCES cw3_Artist(artistId),
	title VARCHAR(100)
);

CREATE TABLE cw3_Track (
	trackId INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
	albumId INT FOREIGN KEY REFERENCES cw3_Album(albumId),
	artistId INT FOREIGN KEY REFERENCES cw3_Artist(artistId),
	genreId INT FOREIGN KEY REFERENCES cw3_Genre(genreId),
	title VARCHAR(100),
	len INT,
	rating INT,
	count INT
);