# Program to obtain a server's response formatted in JSON
# and get the total count from all the comments contained in it

import urllib.request, urllib.parse, urllib.error
import json
import ssl

# Input URL to get JSON from
url = input("Enter URL: ")

# Send request to server and get response
response = urllib.request.urlopen(url)
data = response.read().decode()

# Parse JSON response
try: js = json.loads(data)
except: js = None
if not js:
    print('==== Failure to retrieve data ====')
    print(data)
else:
    print('\n==== Retrieved data ====')
    # print(json.dumps(js, indent=4))

# Sum each value inside of each 'count' property
comments = js['comments']
count = 0
for comment in comments:
    count += int(comment['count'])

# Display the total count of all comments
print(f"Count: {count}\n")
