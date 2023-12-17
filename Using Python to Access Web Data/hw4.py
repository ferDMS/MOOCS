import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Obtain starting URL, number of times to follow links
# and the position from where to obtain the link
url = input("Enter URL - ")
count = int(input("Enter count: "))
# Correct position to match with index
pos = int(input("Enter position: "))-1  

# Variable to save the last name found 
# after `count` references and at position `pos`
name = ""

# Follow link `count` times
print(f"Retrieving: {url}")
for i in range(0, count):
    # Connect to the URL, obtain the HTML document, and parse it with BeautifulSoup
    fhand = urllib.request.urlopen(url, context=ctx)
    html_doc = fhand.read().decode()
    soup = BeautifulSoup(html_doc, 'lxml')

    # Obtain all anchor tags inside of the html document
    tags = soup('a')

    # Get the anchor at position `pos`
    sel_tag = tags[pos]

    # Save the person's name of the selected link
    name = sel_tag.string

    # Update the URL to follow from the tag
    url = tags[pos]['href']
    print(f"Retrieving: {url}")

print(f"Last name reached: {name}")