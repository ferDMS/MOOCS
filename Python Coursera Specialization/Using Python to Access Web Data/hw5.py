# Get number of comments from XML document

import urllib.request, urllib.response, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl

url = input("Enter URL: ")

# Ignore SSL certificate errors from HTTPS connection
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Open URL, get response, and parse it
handle = urllib.request.urlopen(url)
response = handle.read().decode()
tree = ET.fromstring(response)

# Counter for all counts in each comment
count = 0

# Iterate through every comment and add to the count
comments = tree.findall('./comments/comment')
for comment in comments:
    try:
        count += int(comment.find('count').text)
    except:
        print("Count in XML document not an integer")
        continue

# Display the final count
print(f"Count: {count}")