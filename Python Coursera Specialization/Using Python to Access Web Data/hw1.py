import re

# Read and save data
handler = open('regex_sum_1378268.txt', 'r')
text = handler.read()
# Go any set of digits in the string
    # () : extraction start and end
    # [0-9] : match any digit
    # + : one or more times
nums = re.findall('([0-9]+)', text)

count = 0
for num in nums:
    count += int(num)
print(count)