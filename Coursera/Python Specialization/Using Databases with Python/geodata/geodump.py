import sqlite3
import json
import codecs

# Connect to SQLite DB
conn = sqlite3.connect('geodata.sqlite')
cur = conn.cursor()

# Get all locations and JSON geocode data from the DB 
cur.execute('SELECT * FROM Locations')

# Select `where.js` as the file to write. Deletes existing data.
fhand = codecs.open('where.js', 'w', "utf-8")

# Open a list for each location
fhand.write("myData = [\n")

# Parse each location's geocode and format into the `where.js` file
count = 0
for row in cur :
    # Get the DB saved JSON bytes, decode into string and parse it
    data = str(row[1].decode())
    try: js = json.loads(str(data))
    except: continue

    # Check that the status of the JSON response is correct
    if not('status' in js and js['status'] == 'OK') : continue

    # Get the longitude and latitude (geocode) of the location from the parsed JSON
    lat = js["results"][0]["geometry"]["location"]["lat"]
    lng = js["results"][0]["geometry"]["location"]["lng"]

    # If the location wasn't found, don't include it in the `where.js` file
    if lat == 0 or lng == 0 : continue

    # Get the formatted address from the parsed JSON
    where = js['results'][0]['formatted_address']
    where = where.replace("'", "")

    # Display the information, format it and add it into `where.js`
    try :
        print(where, lat, lng)
        count += 1
        if count > 1 : fhand.write(",\n") # Go to next line for every line end
        output = "["+str(lat)+","+str(lng)+", '"+where+"']" # Format the info into `where.js`
        fhand.write(output)
    
    # If we couldn't write the location, skip it
    except:
        continue

# Close list for each location
fhand.write("\n];\n")

# Show confirmation
cur.close()
fhand.close()
print(count, "records written to where.js")
print("Open where.html to view the data in a browser")

