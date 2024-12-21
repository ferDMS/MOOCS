# Can only scrap data through HTTP
# Dependency: lxml parser library

import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

url = input('Enter URL: ')
fhand = urllib.request.urlopen(url)
html_doc = fhand.read().decode()
soup = BeautifulSoup(html_doc, 'lxml')

print(soup.prettify())