import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input("Enter - ")

# Connect with the URL, obtain the HTML document, and parse it with BeautifulSoup
fhand = urllib.request.urlopen(url, context=ctx)
html_doc = fhand.read().decode()
soup = BeautifulSoup(html_doc, 'lxml')

# Obtain all span tags inside of the html document
tags = soup('span')

# Sum the contents of the tags and count their appearance
sum, count = 0, 0
for tag in tags:
    count += 1
    sum += int(tag.string)

# Print the sum and count
print(f"Count {count}")
print(f"Sum {sum}")