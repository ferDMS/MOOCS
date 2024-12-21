# XML parsing

# Library included in the python installation to parse XML
import xml.etree.ElementTree as ET

# A triple quoted string is a possible multiline string
data = """ 
<person>
    <name>Chuck</name>
    <phone type="intl">
        +1 123 456 7890
    </phone>
    <email hide="yes"/>
</person>"""

tree = ET.fromstring(data)

# The text attribute gets the content of the name tag
print('Name:', tree.find('name').text)
print('Attr:', tree.find('email').get('hide'))