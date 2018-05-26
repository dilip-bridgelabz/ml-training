"""
    @Question : #1. Write a Python program to create an array of 5 integers and display the array items.
                Access individual element through indexes.
"""
from array import *
arr = array('i', [1, 2, 3, 4, 5])
for i in arr:
    print(i)
print('')
print('Pringing the array using index:')
print(arr[0])
print(arr[1])
print(arr[2])