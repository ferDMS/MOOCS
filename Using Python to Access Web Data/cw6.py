# Parsing complex type elements from an XML document

import xml.etree.ElementTree as ET

input = """
<stuff>
    <users>
        <user x="2">
            <id>001</id>
            <name>Chuck</name>
        </user>
        <user x="7">
            <id>009</id>
            <name>Brent</name>
        </user>
    </users>
</stuff>
"""

stuff = ET.fromstring(input)
# Get the node 'user' inside of the node 'users'
list = stuff.findall('users/user')
print('User count:', len(list))
for item in list:
        print('Name', item.find('name').text)
        print('Id', item.find('id').text)
        print('Attribute', item.get("x"))