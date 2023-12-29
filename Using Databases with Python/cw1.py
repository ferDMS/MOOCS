"""
Python script to save the total count of emails received from certain
addresses based on a security log. The script obtains the email addresses using
regular expressions, saves them in a list data structure, and updates the count 
of emails from every sender on an Azure Database. Any new address occurrence is saved
on the DB, while already recorded addresses are only increased by their count.

In the end the code displays the top 5 sender addresses by count.
"""

import pyodbc
import getpass
import regex

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


# Connect to Azure Database
cur = connectToDB()

# Open file and obtain email logs (default is 'mbox-short.txt')
file_name = input("Enter email logs file name: ")
if len(file_name) < 1: file_name = "mbox-short.txt"
with open(file_name, 'r') as file: log = file.read()

# Use regular expressions to find all sender addresses
senders = regex.findall(r'From:\s*(.*)\n', log)

# Stablish database name using the email log file name
db_name = file_name.split('.')[0] + "_counts"

# Create a fresh table where to save the email addresses and their counts if
# a table with relevant values doesn't exist yet in the database
cur.execute(f"DROP TABLE IF EXISTS [{db_name}]")
cur.execute(f"CREATE TABLE [{db_name}] (email VARCHAR(50), count INT)")

# The following section of code works but is not as efficient as it relies
# completely on performing multiple queries to the DB that could be avoided
# in order to save runtime and Azure CPU usage.
"""
# Perform database operations for each sender address retrieved
for email in senders:
    cur.execute(f"SELECT count FROM [{db_name}] WHERE email = ?", email)
    row = cur.fetchone()
    # If there is no entry for the address then add it to the database
    if row is None:
        cur.execute(f"INSERT INTO [{db_name}](email, count) VALUES (?,?)", (email, 1))
    # If there is an entry, then add one to the count
    else:
        cur.execute(f"UPDATE [{db_name}] SET count = ? WHERE email = ?", (row.count+1, email))
    cur.commit()
"""

# More effective code block which tracks the count for emails locally and
# inserts the data in the Azure Database only at the end
counts = dict()
for email in senders:
    counts[email] = counts.get(email, 0) + 1
for email, count in counts.items():
    cur.execute(f"INSERT INTO [{db_name}](email, count) VALUES (?, ?)", (email, count))
cur.commit()

# Obtain the top 5 senders by count and display them
cur.execute(f"SELECT TOP(5) email, count FROM [{db_name}] ORDER BY count DESC")
rows = cur.fetchall()
print("\n\tTop 5 Email Senders\n-----------------------------------")
for i in range(len(rows)):
    print(f"{i+1}\t{rows[i].email}: {rows[i].count}")