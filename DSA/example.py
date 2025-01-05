original = [0, 0, 0, 0, 0]
n = len(original)

# Start, end, operation
operations = [(0, 3, +10), (1, 4, -5), (2, 4, -10)]

diff = [0] * (n + 1) # n + 1 for left over counter operations at the end
for start, end, op in operations:
    # Save operation onwards
    diff[start] += op
    # Apply counter-operation onwards
    diff[end+1] -= op

# Sweep through diff array
# Accumulating operations in the way
sweep = 0
for i in range(n):
    sweep += diff[i]
    # Apply and accumulate (sweep) operations 
    original[i] += sweep

print(original)