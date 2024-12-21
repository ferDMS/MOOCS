DROP TABLE IF EXISTS Ages;

CREATE TABLE Ages (
	name VARCHAR(128),
	age INTEGER
)

DELETE FROM Ages;
INSERT INTO Ages (name, age) VALUES
	('Tristian', 21),
	('Cilla', 26),
	('Rennie', 32),
	('Brendyn', 14);

-- Get the HEX representation of concat(name, age)
SELECT TOP(1) CONVERT(VARBINARY(MAX) ,(name + CAST(age AS VARCHAR)) ) AS X
FROM Ages
ORDER BY X