# Get 'place_id' information from Google Maps Geocoding data
# through an API call and parse a JSON response to obtain the desired information

import urllib.request, urllib.error, urllib.parse
import ssl
import json

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Get API key from file path
key_path = input("Enter path to file with API key (blank for './api.secret'): ")
if not key_path: key_path = 'api.secret'
with open(key_path) as file:
    key = file.read()

# Get address to search and construct url
address = input("Enter address: ")
service_url = "https://maps.googleapis.com/maps/api/geocode/json?"
params = urllib.parse.urlencode({
    'address' : address,
    'key' : key
})
url = service_url + params
print(f"Retrieving {url}")

# Send API query and get response from server
response = urllib.request.urlopen(url, context=ctx)
data = response.read().decode()

# Parse JSON formatted data
try: js = json.loads(data)
except: js = None
if not js or 'status' not in js or js['status'] != 'OK':
    print("\n==== Failure parsing JSON ====")
    print(data)
    exit(0)
else:
    print ("\n==== JSON parsed correctly ====")
    print(f"Retrieved {len(data)} characters")

# Get 'place_id' property from parsed data
place_id = js['results'][0]['place_id']
print(f"Place ID: {place_id}")
