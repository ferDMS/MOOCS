SELECT Artist.name, Genre.name FROM Track INNER JOIN Genre ON Track.genreId = Genre.genreId INNER JOIN Album ON Track.albumId = Album.albumId INNER JOIN Artist ON Album.artistId = Artist.artistId;

SELECT Artist.name, Genre.name FROM Track, Genre, Album, Artist WHERE Track.genreId = Genre.genreId AND Track.albumId = Album.albumId AND Album.artistId = Artist.artistId;