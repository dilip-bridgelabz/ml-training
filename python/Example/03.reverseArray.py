"""
    @Question : #3. Write a Python program to reverse the order of the items in the array.

"""
# Empty list (array)
arr = []  # Create an empty list(array)
for i in range(10):  # appending new items to the list(array)
    arr.append(i+1)

# Option 1: Using the for loop to reverse the individual list(array) element
for index, value in enumerate(arr):  # Printing the array element individual element through indexes
    print("index:", index, "value:", value, ('  And %s is %d' % (value, index)))
    # print('%s is %d' % (value, index))
print('')

# Option 1: Using the reverse function, But notice the arr is not saved.
"""
    - Reverses the list in-place
    - Fast, doesn"t take up extra memory
    - Modifies the original list
"""
arr.reverse()
print('Using reverse() func e.g. arr.reverse() :')
print('- Reverses the list in-place ')
print('- Fast, doesn"t take up extra memory ')
print('- Modifies the original list')
print('Output :', arr)
print('')

# Option 2: Using the “[::-1]” Slicing Trick to Reverse a Python List (For e.g. arr[start:end:step])
arr[::-1]
print('Using the slicing e.g. arr[::-1] :')
print('Output :', arr)
print('')

# Option 3: Using reversed function
arr1 = list(reversed(arr))
print('Using reversed func e.g. list(reversed(arr)) :')
print('Output :', arr1)
print('')

# Option 4: Using the “__reversed__()” function
arr1 = list(arr.__reversed__())
print('Using reversed func e.g. list(arr.__reversed__()) :')
print('Output :', arr1)
print('')
