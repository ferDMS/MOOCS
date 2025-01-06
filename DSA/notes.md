Some topics to refresh and practice more often:

- deque in Python
- iterative traversal with BFS

## Line Sweep / Diff Array

Whenever we are working with intervals, instead of saving an operation for each element within each interval ( $O(km)$ ), we can instead save **event points** into a **`diff` array** in order to apply changes in a given order (from left to right) as we **sweep** through this `diff` array.

The way in which an interval is represented in a `diff` array is simple to understand with the right intuition:

- A certain operation needs to be applied with elements inside of the interval, say a simple $+1$.
- To mark the start of the operation, we register the operation starting at a given index `i`.
- The mark above will mean that all elements starting at `start_idx` until the end of the `diff` array ($\rightarrow$ `n`), will apply said operation
- We know that the operation shouldn't be applied no longer than just the end of the interval. To achieve this we do the following:
- We mark an exactly opposite operation, $-1$, starting just after the interval ends, so, at `end_idx + 1`.
- This will counter the original operation for every element after `end_idx` until the end of `diff`.

The following code exemplifies line sweep with a simple example:

```py
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
# [10, 5, -5, -5, -15]
```

## Prefix Sum

It's a basic technique that ressembles DP in some ways, since it breaks down a problem in subproblems that are overlapping and that are built up incrementally. By keeping an array with the result of the calculations for indices `i` and lower (the "sum" for our prefix) we can build the solution for the following index `i+1` by using the latest calculated prefix sum.

It's helpful for cumulative calculations that involve ranges or attributes specific to an element's location (index).

Because of how it works, a prefix sum is like the "sweep" explained above.