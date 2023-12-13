# Same as cw3.py but can use HTTP and HTTPS

import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter URL: ')
# Adding context parameter to allow HTTPS connection
fhand = urllib.request.urlopen(url, context=ctx)
html_doc = fhand.read().decode()
soup = BeautifulSoup(html_doc, 'lxml')

# To print the HTML raw text (with fixed format and syntax)
source = soup.prettify()

print(soup.title.string)
# URL: https://www.crummy.com/software/BeautifulSoup/
# Beautiful Soup: We called him Tortoise because he taught us.