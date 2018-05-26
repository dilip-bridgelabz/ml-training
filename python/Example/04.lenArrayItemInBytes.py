"""
    @Question : #4. Write a Python program to get the length in bytes of one array item in the internal representation.

"""
import sys

arr = []  # Create an empty array
for i in range(10):  # Appending the new item at the end of array
    arr.append(i + 100)

for index, value in enumerate(bytearray(arr)):  # Printing the array element individual element through indexes
    print("index:", index, "value:", sys.getsizeof(value))  # bytes(value))

print(sys.getsizeof(arr))

class testClass:
    def __init__(self):
        pass

mixArr = [1, b'', 'Python', (), {}, testClass]
for value in mixArr:
    print("Data Type: ", type(value), sys.getsizeof(value))