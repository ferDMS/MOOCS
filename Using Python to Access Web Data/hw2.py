import socket
import re

mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect( ('data.pr4e.org', 80) )

# \r : carriage return character, like typewriter returning to the beginning of a line
# \n : new line character, go to next line
cmd = 'GET http://data.pr4e.org/intro-short.txt HTTP/1.0\r\n\r\n'.encode()

mysock.send(cmd)
text = ""

while True:
    data = mysock.recv(512)
    if (len(data) < 1):
        break
    text += data.decode()
mysock.close()

# Use regular expressions to obtain desired values

try:
    info = {
        'Last-Modified' : re.findall('Last-Modified:\s*(.+)\r\n', text)[0],
        'ETag' : re.findall('ETag:\s*(.+)\r\n', text)[0],
        'Content-Length' : re.findall('Content-Length:\s*(.+)\r\n', text)[0],
        'Cache-Control' : re.findall('Cache-Control:\s*(.+)\r\n', text)[0],
        'Content-Type' : re.findall('Content-Type:\s*(.+)\r\n', text)[0]
    }
    for k, v in info.items():
        print(k, ': ', v)
except:
    print('One of the values was not found correctly')
