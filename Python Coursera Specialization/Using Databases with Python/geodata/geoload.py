import urllib.request, urllib.parse, urllib.error
import http
import sqlite3
import json
import time
import ssl
import sys

# Try to retrieve an API key from a local file containing it
api_file_name = input("Enter the name of the file containing Google Places API key: ")
if len(api_file_name) < 1: api_file_name = "api.secret"
try:
    with open(api_file_name) as file:
        api_key = file.read()
except:
    api_key = False

# If the API key couldn't be obtained, use the course's API instead
if api_key is False:
    api_key = 42
    serviceurl = "http://py4e-data.dr-chuck.net/json?"
# If we have a Google API key, use the Google Places API
else :
    serviceurl = "https://maps.googleapis.com/maps/api/geocode/json?"

# Additional detail for urllib
# http.client.HTTPConnection.debuglevel = 1

# Connect to SQLite DB
conn = sqlite3.connect('geodata.sqlite')
cur = conn.cursor()

# Setup the table if it doesn't exist
cur.execute('''
CREATE TABLE IF NOT EXISTS Locations (address TEXT, geodata TEXT)''')

# Ignore SSL certificate errors (for HTTPS requests)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Get every location from the `where.data` file. If the location is already stored, skip its insertion. If not, insert it.
fh = open("where.data")
count = 0
for line in fh:
    # Limit the script to perform only 200 API calls each run (due to rate limits on the Google Places API or the course's)
    if count > 200 :
        print('Retrieved 200 locations, restart to retrieve more')
        break

    # Get address from file and try to find it in the database
    address = line.strip()
    print('')
    cur.execute("SELECT geodata FROM Locations WHERE address= ?",
        (memoryview(address.encode()), ))

    # If found, then skip its insertion
    try:
        data = cur.fetchone()[0]
        print("Found in database ",address)
        continue

    # If not found, then continue with insertion
    except:
        pass

    # Stablish parameters for the API request as URL encoded values. Here the API key and location are passed.
    parms = dict()
    parms["address"] = address
    if api_key is not False: parms['key'] = api_key
    url = serviceurl + urllib.parse.urlencode(parms)

    # Construct the API request and retrieve the JSON response. Here we retrieve the geocode of the location.
    print('Retrieving', url)
    uh = urllib.request.urlopen(url, context=ctx)
    data = uh.read().decode()
    print('Retrieved', len(data), 'characters', data[:20].replace('\n', ' '))

    # Increment the count of inserted locations
    count = count + 1

    # Parse the JSON response
    try:
        js = json.loads(data)
    except:
        print(data)  # We print in case unicode causes an error
        continue

    # Make sure we have a valid response (OK status) from the server before manipulating the parsed response
    if 'status' not in js or (js['status'] != 'OK' and js['status'] != 'ZERO_RESULTS') :
        print('==== Failure To Retrieve ====')
        print(data)
        break

    # Insert the location and geocode into the database
    cur.execute('''INSERT INTO Locations (address, geodata)
            VALUES ( ?, ? )''', (memoryview(address.encode()), memoryview(data.encode()) ) )
    conn.commit()

    # Make the script pause for a bit each 10 requests to allow for the servers (API and DB) to catch up on the requests
    if count % 10 == 0 :
        print('Pausing for a bit...')
        time.sleep(5)

# Confirmation of execution
print("Run geodump.py to read the data from the database so you can vizualize it on a map.")
