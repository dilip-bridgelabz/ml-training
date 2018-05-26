"""
    @array: Array class Questions
"""

from array import *


class ArrayClass:
    arr = None
    arr_bytes = None
    list_sample = [10, 20, 30, 40, 50]

    def __init__(self):
        pass

    def create_and_display(self):
        """
            @Question : #1. Write a Python program to create an array of 5 integers and display the array items.
                            Access individual element through indexes.
            :return: True
        """
        self.arr = array('i', [1, 2, 3, 4, 5])
        print('Created new array having items:')
        self.display()
        return True

    def append_at_end_and_display(self):
        """
            @Question : #2. Write a Python program to append a new item to the end of the array.
        """
        self.arr.append(6)
        self.arr.append(7)
        self.arr.append(8)
        self.arr.append(8)
        print('')
        print('Appended 4 new items at end:')
        self.display()
        return True

    def reverse_and_display(self):
        """
            @Question #3. Write a Python program to reverse the order of the items in the array.
        :return:
        """
        self.arr.reverse()
        print('')
        print('Reverse the array:')
        self.display()
        return True

    def len_array_item_in_bytes(self):
        """
            @Question : #4. Write a Python program to get the length in bytes of one array item in the internal representation.
            :return: True
        """
        print('')
        print('Array length in bytes of one array item:')
        print(self.arr.itemsize)
        return True

    def buffer_info_mem_buffer_bytes(self):
        """
            @Question #5. Write a Python program to get the current memory address and the length in elements of the
                            buffer used to hold an array?s contents and also find the size of the memory buffer in bytes.
            :return: True
        """
        print('')
        print('Current memory address & the length in elements of the buffer:')
        print(self.arr.buffer_info())
        print('Size of memory buffer in bytes:')
        print(self.arr.buffer_info()[1] * self.arr.itemsize)
        return True

    def num_element_occur(self):
        """
            @Question #6. Write a Python program to get the number of occurrences of a specified element in an array.
            :return: True
        """
        print('')
        print('Count of element occurs in array')
        for index, value in enumerate(self.arr):
            print(" %d occurs %d " % (value, self.arr.count(value)))
        return True

    def append_array(self):
        """
            @Question #7. Write a Python program to append items from inerrable to the end of the array.
            :return: True
        """
        print('')
        print('Appending array at the end')
        self.arr.extend(self.arr)
        print(self.arr)
        return True

    def machine_value(self):
        """
            @Question #8. Write a Python program to convert an array to an array of machine values and return the bytes representation.
            :return: True
            97-a, 98-b, 99-c, 100-d, 101-e, 102-f, 103-g, 104-h, 105-i, 106-j, 107-k, 108-l, 109-m, 110-n, 111-o
            112-p, 113-q, 114-r, 115-s, 116-t, 117-u, 118-v, 119-w, 120-x, 121-y, 122-z
        """
        print('')
        print('Converting array bytes elements to string')
        self.arr_bytes = array('b', [100, 105, 108, 105, 112])
        print(self.arr_bytes.tobytes())

    def append_items_from_list(self):
        """
            @Question #9. Write a Python program to append items from a specified list.
            :return: True
        """
        print('')
        print('Appending the list elements to array')
        self.arr.fromlist(self.list_sample)
        print(self.arr)

    def append_at_second(self):
        """
            @Question #10. Write a Python program to insert a new item before the second element in an existing array.
            :return: True
        """
        print('')
        print('Append at the second position:')
        self.arr.insert(1, 101)
        print(self.arr)

    def remove_by_index(self):
        """
            @Question #11. Write a Python program to remove a specified item using the index from an array.
            :return: True
        """
        print('')
        print('Removing the element by index, 19 again and again for 5 times')
        self.arr.pop(19)
        self.arr.pop(19)
        self.arr.pop(19)
        self.arr.pop(19)
        self.arr.pop(19)
        print(self.arr)

    def remove_first_item_by_value(self):
        """
            @Question #12. Write a Python program to remove the first occurrence of a specified element from an array.
            :return: True
        """
        print('')
        print('Remove the element by value, Removing 101, 8, 8')
        self.arr.remove(101)
        self.arr.remove(8)
        print(self.arr)

    def convert_array_list(self):
        """
            @Question #13. Write a Python program to convert an array to an ordinary list with the same items.
            :return: True
        """
        print('')
        print('Convert array to list')
        self.arr = self.arr.tolist()
        print(self.arr)

    def display(self, by_index=False):
        """
            Displays the output on console.
            :return:
        """
        if by_index:
            for index, value in enumerate(self.arr):
                print(self.arr[index])
        else:
            print(self.arr)


arrObject = ArrayClass()
arrObject.create_and_display()
arrObject.append_at_end_and_display()
arrObject.reverse_and_display()
arrObject.len_array_item_in_bytes()
arrObject.buffer_info_mem_buffer_bytes()
arrObject.num_element_occur()
arrObject.append_array()
arrObject.machine_value()
arrObject.append_items_from_list()
arrObject.append_at_second()
arrObject.remove_by_index()
arrObject.remove_first_item_by_value()
arrObject.convert_array_list()