-- Drop table if it already exists
IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[Users]') AND type in (N'U'))
DROP TABLE [dbo].[Users]

-- Create database table
CREATE TABLE [dbo].[Users](
	[name] [varchar](128) NULL,
	[email] [varchar](128) NULL
) ON [PRIMARY]

-- Insert some values
INSERT INTO Users(name, email) VALUES
	('John', 'john@doe.edu'),
	('Marie', 'marie@umich.edu'),
	('Chuk', 'chuck@umich.edu'),
	('Ted', 'ted@umich.edu'),
	('Rebecca', 'rebecca@umich.edu')

-- Select all the entries 
SELECT * FROM Users

-- Bye bye ted
DELETE FROM Users WHERE name='Ted'

-- Hey 'Chuck' is spelt incorrectly!
UPDATE Users 
SET name='Chuck' 
WHERE email='chuck@umich.edu'

SELECT * FROM Users
ORDER BY email