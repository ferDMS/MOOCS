"""
Python script to update an Azure database with an XML record of different songs and attributes of them which include details regarding Tracks, Album, Artists, and Genres.
"""

import pyodbc
import xml.etree.ElementTree as ET
import os
import sqlite3

# Function to connect to a database with a small CLI
def connectToDB(server, user, password, db) -> pyodbc.Cursor:

    print(f"Connecting to {server}...")

    # Prepare the ODBC connection string with the information provided
    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=" + server + ";"
        "Port=1433;"
        "Database=" + db + ";"
        "Uid=" + user + ";"
        "Pwd=" + password + ";"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )

    # Attempt connection to database and show error if there is an exception
    print("Attempting to connect to the database...")
    try:
        conn = pyodbc.connect(conn_str)
        print("Successfully connected to the database.")
        # Return the cursor object
        cur = conn.cursor()
        return cur
    except pyodbc.Error as e:
        print("Error:", e)
        raise Exception("Could not connect to database")


# Setup tables in the database through a semicolon-separated sql queries script. Its name should be the same as the python script but with sql extension
def setupDB(conn, cur, file_name):
    print("Setting up database through script...")
    script_file = file_name + '.sql'
    with open(script_file, 'r') as file:
        script = file.read()
    queries = script.split(';')
    for query in queries:
        if len(query) > 0:
            # print(query.strip())
            cur.execute(query.strip())
    conn.commit()
    print("Database ready.")


# Function to search the value of a key from the parsed XML as <key><value>
def search(tree, key):
    found = False
    for child in tree:
        if found:
            return child
        if child.tag == 'key' and child.text == key:
            found = True
    return None


# Create a new SQLite database and cursor
file_name = os.path.basename(__file__).split('.')[0]
conn = sqlite3.connect(f'{file_name}.sqlite')
cur = conn.cursor()

# Setup tables
setupDB(conn, cur, file_name)

# Open the file of records and parse the XML
xml_file = input("\nEnter songs record file name: ")
if len(xml_file) < 1: xml_file = "Library.xml"
tree = ET.parse(xml_file)

# Obtain a list of all tracks in the record
content_tree = tree.find("dict")
tracks_tree = search(content_tree, "Tracks").findall("dict")

# Insert  the track into the Database
print("Inserting tracks into database...")
count = 1
for track in tracks_tree:
    # Obtain each attribute from the parsed element
    info = {}
    valid = True
    if ( search(track, 'Track ID') is None ) : continue
    for attr in ["Album", "Artist", "Genre", "Name", "Play Count", "Total Time", "Rating"]:
        element = search(track, attr)
        try:
            info[attr] = element.text
        except:
            if attr in ["Album", "Artist", "Genre", "Name"]:
                valid = False
            else:
                info[attr] = None

    if not valid: 
        continue

    # Insert the track's relational attributes in the database (if they haven't been inserted yet)
    # Print status
    print(f"{count}. Inserting {info['Name']}...")

    # Genre
    if not cur.execute("SELECT id FROM Genre WHERE name = ? LIMIT 1", (info['Genre'],)).fetchone():
        cur.execute("INSERT INTO Genre(name) VALUES (?)", (info['Genre'],))
    genre_id = cur.execute("SELECT id FROM Genre WHERE name = ? LIMIT 1", (info['Genre'],)).fetchone()[0]

    # Artist
    if not cur.execute("SELECT id FROM Artist WHERE name = ? LIMIT 1", (info['Artist'],)).fetchone():
        cur.execute("INSERT INTO Artist(name) VALUES (?)", (info['Artist'],))
    artist_id = cur.execute("SELECT id FROM Artist WHERE name = ? LIMIT 1", (info['Artist'],)).fetchone()[0]

    # Album
    if not cur.execute("SELECT id FROM Album WHERE title = ? LIMIT 1", (info['Album'],)).fetchone():
        cur.execute("INSERT INTO Album(title, artist_id) VALUES (?, ?)", (info['Album'], artist_id))
    album_id = cur.execute("SELECT id FROM Album WHERE title = ? LIMIT 1", (info['Album'],)).fetchone()[0]

    # Insert the track into the database
    cur.execute("INSERT OR REPLACE INTO Track(album_id, genre_id, title, len, rating, count) VALUES (?, ?, ?, ?, ?, ?)", (
        album_id,
        genre_id,
        info['Name'],
        info['Total Time'],
        info['Rating'],
        info['Play Count']
        )
    )

    # Commit
    conn.commit()

    # Add track to count
    count += 1

# Confirmation message
print(f"Inserted {count} tracks into local database.")

# Test query to confirm correct insertion
rows = cur.execute(
"""
SELECT Track.title, Artist.name, Album.title, Genre.name 
    FROM Track JOIN Genre JOIN Album JOIN Artist 
    ON Track.genre_id = Genre.id and Track.album_id = Album.id 
        and Album.artist_id = Artist.id
    ORDER BY Artist.name, Track.title LIMIT 3
"""
).fetchall()

print("\n\n\n\t\tTest query\n------------------------------------------------------------")
for i in range(len(rows)):
    print(f"{i+1}\t{rows[i][0]},  {rows[i][1]},   {rows[i][2]},   {rows[i][3]}")