"""
Python script to create a table in an Azure database that contains the
count of emails sent by an organization based on an email log.
"""

import regex
import pyodbc
import sqlite3

# Function to connect to a database with a small CLI
def connectToDB() -> pyodbc.Cursor:
    # Information to execute the connection to the Azure Database
    server = "ferdms.database.windows.net"

    # Obtain user and password from local file
    login_file = input("SQL Credentials file name (leave blank for './credentials.secret'): ")
    if len(login_file) < 1: login_file = "credentials.secret"
    with open(login_file) as file:
        user = file.readline().strip()
        password = file.readline().strip()
    print(f"Connecting to {server}...\n")

    # Prepare the ODBC connection string with the information provided
    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=tcp:" + server + ",1433;"
        "Database=PythonForEverybodyDB;"
        "Uid=" + user + ";"
        "Pwd={" + password + "};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    # Attempt connection to database and show error if there is an exception
    print("Attempting to connect to the database...")
    try:
        conn = pyodbc.connect(conn_str)
        print("Successfully connected to the database.")
    except pyodbc.Error as e:
        print("Failed to connect to the database.")
        print("Error:", e)

    # Return the cursor object
    cur = conn.cursor()
    return cur


# Connect to the Azure database
cur = connectToDB()

# Open file and obtain email logs (default is 'mbox-short.txt')
file_name = input("Enter email logs file name: ")
if len(file_name) < 1: 
    file_name = "mbox.txt"
with open(file_name, 'r') as file: 
    log = file.read()

# Use regular expressions to find all sender addresses
senders = regex.findall(r'From:.*@(.+)\n', log)

# Stablish table name using the email log file name
table_name = file_name.split('.')[0] + "_org_counts"

# Create a fresh table where to save the organizations and their counts if
# a table with relevant values doesn't exist yet in the database
cur.execute(f"DROP TABLE IF EXISTS [{table_name}]")
cur.execute(f"CREATE TABLE [{table_name}] (org VARCHAR(50), count INT)")

# Track the count for emails locally and insert the data in the table only at the end
counts = dict()
for org in senders:
    counts[org] = counts.get(org, 0) + 1
for org, count in counts.items():
    cur.execute(f"INSERT INTO [{table_name}](org, count) VALUES (?, ?)", (org, count))
cur.commit()

# Obtain top 10 organizations by count in descending order and display them
cur.execute(f"SELECT TOP(10) org, count FROM [{table_name}] ORDER BY count DESC")
rows = cur.fetchall()
print("\n\tTop 10 Organizations by Email Count (SQL Server)\n--------------------------------------------------------")
for i in range(len(rows)):
    print(f"{i+1}\t{rows[i].org}: {rows[i].count}")

# Obtain the table from the SQL Server DB
rows = cur.execute(f"SELECT * FROM [{table_name}]").fetchall()

# Create the SQLite DB and the `Counts` table
conn2 = sqlite3.connect(table_name + '.sqlite')
cur2 = conn2.cursor()
cur2.execute('DROP TABLE IF EXISTS Counts')
cur2.execute('CREATE TABLE Counts (org TEXT, count INTEGER)')

# Insert the data from the SQL Server table into the SQLite table
for row in rows:
    cur2.execute("INSERT INTO Counts(org, count) VALUES (?, ?)", (row.org, row.count))
conn2.commit()

# Obtain top 10 organizations by count in descending order and display them
cur2.execute(f"SELECT org, count FROM Counts ORDER BY count DESC LIMIT 10")
sqlite_rows = cur2.fetchall()
print("\n\n\n\tTop 10 Organizations by Email Count (SQLite)\n--------------------------------------------------------")
for i in range(len(sqlite_rows)):
    print(f"{i+1}\t{sqlite_rows[i][0]}: {sqlite_rows[i][1]}")