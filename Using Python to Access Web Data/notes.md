## Regular Expressions

They are formatted or "wild card" strings that can match other strings. It is like specifying certain conditions that will make a regular expression match with any other expression / string. It can be used to search through text with certain restrictions that are specified with certain characters. It is like programming with characters:

![](/Users/pez/Library/Application%20Support/marktext/images/2023-12-11-19-21-59-image.png)

Regular expressions can be used through a library for python. By using the search method we can search through the specified string for any substring or expression specified

```python
import re

hand = open('text.txt')
for line in hand:
    line = line.rstrip()
    if re.search('From:', line):
    print(line)
```

| Method                      | Description                                 |
| --------------------------- | ------------------------------------------- |
| .search(*regex*, *source*)  | Get boolean if match found                  |
| .findall(*regex*, *source*) | Get list of all matches, if none then empty |

This would just search for `From: ` at any point in the text, but, for example, if we wanted to search it as startswith() then we can use `^From: `. A more complicated one could be `^X.*:`, which means that we are searching for any string that starts with `X`, followed by any character in any quantity but that has at some point a `:`.

By greedy it means that the result to a regular expression match will be the largest possible in characters. The `*` and `+` operators are greedy by default, but can be changed to non-greedy, which means that they will get the smallest possible match instead. Useful for these type of cases:

**Greedy**

![](/Users/pez/Library/Application%20Support/marktext/images/2023-12-11-19-38-37-image.png)

**Non-greedy**

![](/Users/pez/Library/Application%20Support/marktext/images/2023-12-11-19-38-08-image.png)

## Network Technology

TCPs (Transfer Control Protocols) create pipes for two different devices to communicate between each other. They use an Internet Protocol (IP) address. Generally, the app prepares data through a protocol (transport) and this reaches the internet through an ethernet link. After this, it goes through multiple pathways and eventually reaches the end device. The connection is called the socket.

Stablishing a socket with a device can be done simply with its address and a port

```python
import socket
mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysocket.connect( ('data.pr4e.org', 80) )
```

A protocol is a set of rules for communication. HTTP (HyperText Transfer Protocol) is the most basic protocol that emerged for web interaction.

## HTTP

For example, in an HTTP connection between devices, **GET** requests are submitted from the client to the host device and this last one sends a response with the HTML document to load a webpage. This is the request-response cycle.

The format of the response includes a header (with metadata) and the content, which includes the requested document.

You can perform a get request to a server by encoding the command into bytes

```python
cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\r\n\r\n'.encode()
```

The entire response might take multiple receive commands if it is too long, which is why it might be useful to use a while loop. We receive the information 512 chars at a time.

```python
while True:
    data = mysocket.recv(512)
    if (len(data) < 1):
        break
    print(data.decode())
mysocket.close()
```

The response will be in bytes and should be handled according to the data's information. Since we are returning text we just convert it back to UNICODE

200 is success, 404 is not found 

## Unicode, ASCII and characters

ASCII only considered the most basic characters in the latin alphabet in a single byte. UTF-8 is part of UNICODE and allows characters to be declared with more than 1 byte whenever there is a character specifying so. This means that all ASCII characters are in UTF-8.

In python 2 strings and bytes were the same, and unicode strings different.

In pyhton 3 bytes and strings are different, but all strings are unicode.

`str.encode()` turns the string into bytes and `str.decode()` turns it back to UTF-8

## Easier sockets

Connection with a server and requesting of a document that is saved as a file handle and manipulated as if it were a file.

```python
import urllib.request, urllib.parse, urllib.error
fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
for line in fhand:
    print(line.decode().strip())
```

## Web Scraping and Parsing

To create a script that pretends to be a browser and retrieves data from web pages and specific information. Also called spidering. Could be illegal depending on what you are trying to retrieve because of copyright or terms and conditions of websites.

Parsing is to obtain the individual sets of information wanted from a structured syntax. HTML forgives some syntax errors, which is why it is hard to parse HTML structured information correctly as, for example, some regex could be positive for some pages but not for others, or it could contain more elements not considered on others, etc. 

[Beautiful Soup: We called him Tortoise because he taught us.](https://www.crummy.com/software/BeautifulSoup/)

[Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

Beautiful Soup is a library that can parse an HTML document into an object with attributes that lead to more specific information about the webpage, for example:

```python
soup.title
# <title>The Dormouse's story</title>

soup.title.name
# u'title'

soup.title.string
# u'The Dormouse's story'

soup.title.parent.name
# u'head'

soup.p
# <p class="title"><b>The Dormouse's story</b></p>

soup.p['class']
# u'title'

soup.a
# <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

soup.find_all('a')
# [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
#  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
#  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.find(id="link3")
# <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
```

To use it you have to select a parser.

* lxml: very fast and lenient, external library

* default html python parser: fast and lenient, already included

```python
from bs4 import BeautifulSoup
html_doc = "<html>a web page</html>"

soup = BeautifulSoup(html_doc, 'html.parser')
```


