"""
Python script to get information about students in courses from a JSON file and organize the data into an Azure Database with N to N relationships
"""

import pyodbc
import json
import os

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
        "Connection Timeout=60;"
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


# Obtain DB connection information from Local credentials file
login_file = input("\n\nSQL Credentials file name: ")
if len(login_file) < 1: login_file = "localDB.secret"
with open(login_file) as file:
    server = file.readline().strip()
    user = file.readline().strip()
    password = file.readline().strip()
    db = file.readline().strip()

# Connect to the Local database
cur = connectToDB(server, user, password, db)

# Setup tables
file_name = os.path.basename(__file__).split('.')[0]
setupDB(cur, file_name)

# Read and parse JSON file
json_file = input("Enter JSON file name: ")
if len(json_file) < 1: json_file = "roster_data_sample.json"
with open(json_file) as file:
    data = json.loads(file.read())

# For each member of a course obtain the info and insert it into the DB
count = 1
for member in data:
    # Status message
    print(f"{count}. Inserting {member[0]} into {member[1]}...")

    # User
    query = cur.execute(f"SELECT id FROM {file_name}_User WHERE name = ?", (member[0],)).fetchone()
    if not query:
        cur.execute(f"INSERT INTO {file_name}_User(name) VALUES (?)", (member[0],))
    user_id = cur.execute(f"SELECT id FROM {file_name}_User WHERE name = ?", (member[0],)).fetchone()[0]

    # Course
    query = cur.execute(f"SELECT id FROM {file_name}_Course WHERE title = ?", (member[1],)).fetchone()
    if not query:
        cur.execute(f"INSERT INTO {file_name}_Course(title) VALUES (?)", (member[1],))
    course_id = cur.execute(f"SELECT id FROM {file_name}_Course WHERE title = ?", (member[1],)).fetchone()[0]

    # Member
    cur.execute(
        f"INSERT INTO {file_name}_Member(user_id, course_id, role) VALUES (?,?,?)", (
            int(user_id), 
            int(course_id), 
            int(member[2])
        )
    )

    # Update count
    count += 1

# Commit changes
cur.commit()

# Show count
print(f"Inserted {count} members")

# Show top 20 records by user's name
rows = cur.execute(f"""
SELECT TOP(20) [User].name, Course.title, Member.role
    FROM {file_name}_Member AS Member
    INNER JOIN {file_name}_User AS [User] ON Member.user_id = [User].id 
    INNER JOIN {file_name}_Course AS Course ON Member.course_id = Course.id
    ORDER BY [User].name ASC
""").fetchall()

print("\n\n\n\tTop 20 records by User's name\n----------------------------------------")
for i in range(len(rows)):
    print(f"{i+1}\t{rows[i][0]},  {rows[i][1]},   {rows[i][2]},  ")

"""
# Query for hw4
SELECT TOP(1) 'XYZZY' + CONVERT(VARCHAR(MAX), CAST(CAST(Users.name AS varchar(MAX)) + CAST(Course.title AS varchar(MAX)) + CAST(Member.role AS varchar(MAX)) AS varbinary(MAX)), 2) AS X 
FROM cw4_User AS Users
JOIN cw4_Member AS Member ON Users.id = Member.user_id 
JOIN cw4_Course AS Course ON Member.course_id = Course.id
ORDER BY X;
"""