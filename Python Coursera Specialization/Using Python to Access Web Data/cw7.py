# Using Google Maps API to obtain information about locations
# Requesting information to APIs and parsing the responses from the web server

import urllib.request, urllib.error, urllib.parse
import ssl
import json

"""
More information about using the up-to-date GeoCoding Google Map's API
can be found in: 
https://developers.google.com/maps/documentation/geocoding/

To start using the API follow the instructions found on:
https://developers.google.com/maps/documentation/geocoding/overview#how-to-use-the-geocoding-api

Using the provided sample HTTPS request we can construct a URL such as:
https://maps.googleapis.com/maps/api/geocode/json?address=1600+Amphitheatre+Parkway,+Mountain+View,+CA&key=YOUR_API_KEY
"""

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Get API key from selected path
key_path = input("Enter file path with API key (leave blank for default path './api.secret'): ")  
if not key_path:
    key_path = "api.secret"
with open(key_path) as file:
    key = file.read()

# Loop while an address is provided for geocoding
address = input("Enter address: ")
while len(address) > 1:
    # Prepare URL from provided address and API key
    params = urllib.parse.urlencode({
        'address' : address,
        'key' : key
    })
    service_url = "https://maps.googleapis.com/maps/api/geocode/json?"
    url = service_url + params
    print(f"URL: {url}")

    # Send the HTTPS request through urllib and get response from Google servers
    response = urllib.request.urlopen(url, context=ctx)
    data = response.read().decode()
    
    # Parse the JSON response and display it
    try: js = json.loads(data)
    except: js = None
    if not js or 'status' not in js or js['status'] != 'OK':
        print('==== Failure to retrieve data ====')
        print(data)
        break
    else:
        print('\n==== Retrieved data ====')
        # print(json.dumps(js, indent=4))
    
    # Get latitude, longitude and formatted address from response
    # See response format at https://developers.google.com/maps/documentation/geocoding/requests-geocoding#GeocodingResponses 
    lat = js['results'][0]['geometry']['location']['lat']
    lng = js['results'][0]['geometry']['location']['lng']
    formatted_address = js['results'][0]['formatted_address']

    # Display information
    print(f"Latitude: {lat}\nLongitude: {lng}")
    print(f"Formatted address: {formatted_address}")

    # Get new address to geocode or exit program
    address = input("\nEnter address (leave blank to exit): ")

print("\n\n")