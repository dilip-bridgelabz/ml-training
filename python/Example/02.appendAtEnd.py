"""
    @Question : #2. Write a Python program to append a new item to the end of the array.

"""

x = []  # Create an empty array
for i in range(10):  # Appending the new item at the end of array
    x.append(i+1)
    print("Appending: ", i + 1)

for index, value in enumerate(x):  # Printing the array element individual element through indexes
    print("index:", index, "value:", value)
