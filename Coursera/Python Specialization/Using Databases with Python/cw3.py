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
def setupDB(cur, file_name):
    print("Setting up database through script...")
    script_file = file_name + '.sql'
    with open(script_file, 'r') as file:
        script = file.read()
    queries = script.split(';')
    for query in queries:
        if len(query) > 0:
            # print(query.strip())
            cur.execute(query.strip())
    cur.commit()
    print("Database ready.")


# Function to search the value of a key from the parsed XML as <key><value>
def search(tree, key):
    found = False
    for child in tree:
        if found:
            return child
        if child.tag == 'key' and child.text == key:
            found = True
    raise Exception("Element not found")


# Obtain DB connection information from Local credentials file
login_file = input("\n\nLocal SQL Credentials file name (leave blank for './localDB.secret'): ")
if len(login_file) < 1: login_file = "localDB.secret"
with open(login_file) as file:
    server = file.readline().strip()
    user = file.readline().strip()
    password = file.readline().strip()
    db = file.readline().strip()

# Connect to the Local database
local_cur = connectToDB(server, user, password, db)

# Setup tables
file_name = os.path.basename(__file__).split('.')[0]
setupDB(local_cur, file_name)

# Open the file of records and parse the XML
xml_file = input("\nEnter songs record file name: ")
if len(xml_file) < 1: xml_file = "Library.xml"
tree = ET.parse(xml_file)

# Obtain a list of all tracks in the record
content_tree = tree.find("dict")
tracks_tree = search(content_tree, "Tracks").findall("dict")

# Insert  the track into the Database
print("Inserting tracks into database...")
count = 0
for track in tracks_tree:
    # Obtain each attribute from the parsed element
    info = {}
    for attr in ["Album", "Artist", "Genre", "Name", "Play Count", "Total Time", "Rating"]:
        try:
            info[attr] = search(track, attr).text
        except:
            if attr in ["Play Count", "Total Time", "Rating"]:
                info[attr] = "-1"
            else:
                info[attr] = "None"

    # Insert the track's relational attributes in the database (if they haven't been inserted yet)
        
    # Genre
    if not local_cur.execute(f"SELECT TOP(1) genreId FROM {file_name+'_Genre'} WHERE name = ?", info['Genre']).fetchone():
        local_cur.execute(f"INSERT INTO {file_name+'_Genre'}(name) VALUES (?)", info['Genre'])
    genre_id = local_cur.execute(f"SELECT TOP(1) genreId FROM {file_name+'_Genre'} WHERE name = ?", info['Genre']).fetchone()[0]

    # Artist
    if not local_cur.execute(f"SELECT TOP(1) artistId FROM {file_name+'_Artist'} WHERE name = ?", info['Artist']).fetchone():
        local_cur.execute(f"INSERT INTO {file_name+'_Artist'}(name) VALUES (?)", info['Artist'])
    artist_id = local_cur.execute(f"SELECT TOP(1) artistId FROM {file_name+'_Artist'} WHERE name = ?", info['Artist']).fetchone()[0]

    # Album
    if not local_cur.execute(f"SELECT TOP(1) albumId FROM {file_name+'_Album'} WHERE title = ?", info['Album']).fetchone():
        local_cur.execute(f"INSERT INTO {file_name+'_Album'}(title, artistId) VALUES (?, ?)", info['Album'], artist_id)
    album_id = local_cur.execute(f"SELECT TOP(1) albumId FROM {file_name+'_Album'} WHERE title = ?", info['Album']).fetchone()[0]

    # Insert the track into the database
    local_cur.execute(f"INSERT INTO {file_name+'_Track'}(artistId, albumId, genreId, title, len, rating, count) VALUES (?, ?, ?, ?, ?, ?, ?)", (
        int(artist_id),
        int(album_id),
        int(genre_id),
        info['Name'],
        int(info['Total Time']),
        int(info['Rating']),
        int(info['Play Count'])
        )
    )

    # Commit
    local_cur.commit()

    # Add track to count
    count += 1

    # Print status
    print(f"{count}. Inserted {info['Name']}")

# Confirmation message
print(f"Inserted {count} tracks into local database.")

# Obtain DB connection information from Local credentials file, connect and setup tables
login_file = input("\n\nAzure SQL Credentials file name (leave blank for './azureDB.secret'): ")
if len(login_file) < 1: login_file = "azureDB.secret"
with open(login_file) as file:
    server = file.readline().strip()
    user = file.readline().strip()
    password = file.readline().strip()
    db = file.readline().strip()
azure_cur = connectToDB(server, user, password, db)
setupDB(azure_cur, file_name)
azure_cur.commit()


# Clone Local database tables into Azure database tables
print("Cloning tables...")
# Genre
print("Inserting Genres...")
rows = local_cur.execute(f"SELECT * FROM {file_name+'_Genre'}").fetchall()
for row in rows:
    azure_cur.execute(f"INSERT INTO {file_name+'_Genre'}(name) VALUES (?)", row[1:])
azure_cur.commit()
# Artist
print("Inserting Artists...")
rows = local_cur.execute(f"SELECT * FROM {file_name+'_Artist'}").fetchall()
for row in rows:
    azure_cur.execute(f"INSERT INTO {file_name+'_Artist'}(name) VALUES (?)", row[1:])
azure_cur.commit()
# Album
print("Inserting Albums...")
rows = local_cur.execute(f"SELECT * FROM {file_name+'_Album'}").fetchall()
for row in rows:
    azure_cur.execute(f"INSERT INTO {file_name+'_Album'}(artistId, title) VALUES (?, ?)", row[1:])    
azure_cur.commit()
# Track
print("Inserting Tracks...")
rows = local_cur.execute(f"SELECT * FROM {file_name+'_Track'}").fetchall()
for row in rows:
    azure_cur.execute(f"INSERT INTO {file_name+'_Track'}(albumId, artistId, genreId, title, len, rating, count) VALUES (?, ?, ?, ?, ?, ?, ?)", row[1:])
azure_cur.commit()
print("Cloning completed")