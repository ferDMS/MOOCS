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

## Bitwise operations

These are operations that work directly over the binary representation of integers. We take as input the two integers we'll be applying the operation over, and get as output another int.

They are called "bitwise" precisely because we apply the operation over THE BITS of the integers, not over the integers themselves. So, we take each of the bits of each number (by position) and compare them between each other so as to apply the operation.

For example:

```python
# Bitwise AND
5 & 3 # 0101 & 0011 -> 0001 -> int 1

# Bitwise OR
5 | 3 # 0101 | 0011 -> 0111 -> int 7

# Bitwise XOR
5 ^ 3 # 0101 ^ 0011 -> 0110 -> int 6

# Bitwise NOT
~5    # ~0101       -> 1010 -> int 10
```

Apart from these basic boolean operations over bits, we can also manipulate them in another very interesting way.

We perform Bit Shifting to shift ALL THE BITS of an int to the right or left.

The shift does NOT cycle the bits, but the state of the bit we add or remove can change. Specifically, the behavior for shifts will is:

- For shifts to the left, be the number either positive or negative, we add 0s at the rightmost position.
- For shifts to the right things change a bit. For positive numbers we add 0s at the leftmost position. 
- For negative numbers, which are represented in two's complement, we add 1s at the leftmost position in order to preserve the negative value. This because a complement means that bits are inverted "infinitely", so all 0s to the left of positive numbers would acount for 1s.

Remember, to convert a number to and from two's complement we:

- Invert the bits
- Add one

The operator shifts an integer (the one to its left), by X bits (the int to its right).

```python
# Left shift of positive
5 << 1 # 0101 -> 1010 -> int 10

# Left shift of negative
-5 << 1 # 1011 -> 0110 -> int -10

# Right shift of positive
5 >> 1 # 0101 -> 0010 -> int 2

# Right shift of negative
-5 >> 1 # 1011 -> 1101 -> int -3
```

Basic way of efficiently "iterating" over bits using bitwise operations.

Careful: This method "destroys" the number (by shifting bits to "read" them)

```python
# For positive numbers
# Alters the original number
num = 5
temp = num
while num > 0
    print(num & 1) # "Extract" least significant bit
    num = num >> 1 # "Move" to next bit

# For negative numbers (since 1s to the left in a complement are "infinite")
# Doesn't alter the original number
n = -5
num_bits = 8  # Limit to 8 bits
for i in range(num_bits):
    print((n >> i) & 1)
```

A clever way to flip the bit at the ith position:

```python
n = n ^ (1 << i)
# 5 ^ (1 << 2) -> 0101 ^ 0100 -> 0001 -> int 1
```

## XOR operator

One of the most commonly used for leetcode problems because of its interesting properties that make very complex problems reduce to simple abstractions.

The best way to see the operator is as a "bit flip", which can be seen in the operator's truth table:

| $A$ | $B$ | $A \oplus B$ |
|:-:|:-:|:-------:|
| 0 | 0 |    0    |
| 0 | 1 |    1    |
| 1 | 0 |    1    |
| 1 | 1 |    0    |

Viewing the operation as nothing but a "toggle", we can easily identify the following properties:

- Commutative: $a \oplus b = b \oplus a$
- Associative: $a \oplus (b \oplus c) = (a \oplus b) \oplus c$
- Identity: $a \oplus 0 = a$ (doesn't change)
- Self-inverse: $a \oplus a = 0$ (always is 0)

Taking these properties we can create interesting phenomenon to simplify complex calculations:

- Cancellation: When a same element is XORed with itself an even number of times, it cancels to just $0$. Similarly, the element XORed with itself an odd number of times will just leave the original element.
