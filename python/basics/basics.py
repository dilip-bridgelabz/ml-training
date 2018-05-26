"""
    @basics: Basic Python class Questions
"""
import os
import sys
import platform
import calendar
from datetime import datetime
from math import pi


class BasicsPython():

    redius = 0.0
    first_name = None
    last_name = None

    def __init__(self):
        pass

    def version(self):
        """
            @Question #1. Write a Python program to print version of python being used
            :return: True
        """
        print('')
        print("Printing the python version : ", sys.version)
        print("Version information : ", sys.version_info)

    def using_version(self):
        """
            @Question #2. Write a Python program to get the Python version you are using.
            :return: True
        """
        print('')
        print('Hex version of python:', sys.hexversion)
        print('', platform.python_version())
        print(os.path.dirname(sys.executable))

    def current_datetime(self):
        """
            @Question #3. Write a Python program to display the current date and time.
                Sample Output :
                Current date and time :
                2014-07-05 14:34:14
            :return: True
        """
        print('')
        print('System current date time :', datetime.now())
        print('Current Date & time      :', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def circle_area(self):
        """
            @Question #4. Write a Python program which accepts the radius of a circle from the user and compute the area.
                Sample Output :
                r = 1.1
                Area = 3.8013271108436504
            :return:
        """
        print('')
        self.radius = float(input("Enter the radius of circle :"))
        print("Area of circle is :", (pi * self.radius **2))

    def reverse_order(self):
        """
            @Question #5. Write a Python program which accepts the user's first and last name and print them in
                            reverse order with a space between them.
            :return:
        """
        print('')
        first_name = str(input('Enter first name : '))
        last_name = str(input('Enter last name : '))
        name = [first_name, last_name]
        reverse_name = " ".join(str(x) for x in reversed(name))
        print('Reversed the name: ', reverse_name)

    def gen_list_tuples(self):
        """
            @Question #6. Write a Python program which accepts a sequence of comma-separated numbers from user
                            and generate a list and a tuple with those numbers.
                Sample data : 3, 5, 7, 23
                Output :
                List : ['3', ' 5', ' 7', ' 23']
                Tuple : ('3', ' 5', ' 7', ' 23')
        :return:
        """
        print('')
        sequence = input("Enter some number separated by comma : ")
        list_seq = sequence.split(",")
        tuple_list = tuple(list_seq)
        print('Generated List : ', list_seq)
        print('Generated Tuple : ', tuple_list)

    def file_extenstion(self):
        """
            @Question #7. Write a Python program to accept a filename from the user and print the extension of that.
                Sample filename : abc.java
                Output : java
        :return:
        """
        print('')
        file_name = 'abc.java'
        basename = os.path.basename(file_name)  # os independent
        ext = ' '.join(basename.split('.')[1:])
        print('Extension of file {} is \'{}\''.format(file_name, ext))

    def first_last_list_item(self):
        """
            @Question #8. Write a Python program to display the first and last colors from the following list.
                color_list = ["Red","Green","White" ,"Black"]
        :return:
        """
        print('')
        color_list = ["Red", "Green", "White", "Black"]
        print('First color is {} & last color is {}'.format(color_list[0], color_list[-1]))

    def list_date(self):
        """
            @Question #9. Write a Python program to display the examination schedule. (extract the date from exam_st_date).
                exam_st_date = (11, 12, 2014)
                Sample Output : The examination will start from : 11 / 12 / 2014
        :return:
        """
        print('')
        exam_st_date = (11, 12, 2014)
        print('The examination will start from :', datetime.strptime('%i/%i/%i' % exam_st_date, '%m/%d/%Y'))

    def multiplier(self):
        """
            @Question #10. Write a Python program that accepts an integer (n) and computes the value of n+nn+nnn.
                Sample value of n is 5
                Expected Result : 615
        :return:
        """
        print('')
        n = int(input("Enter an integer : "))
        nth = 1
        for x in [n for i in range(5)]:
            nth *= x

        print('5 Multiplier of integer is {}'. format(nth))

    def get_syntax_details(self):
        """
            @Question #11. Write a Python program to print the documents (syntax, description etc.) of
                            Python built-in function(s).
                Sample function : abs()
                Expected Result :
                abs(number) -> number
                Return the absolute value of the argument.
        :return:
        """
        print('')
        print('Details of abs() built-in function is \'{}\''.format(abs.__doc__))

    def show_calendar(self):
        """
            @Question #12. Write a Python program to print the calendar of a given month and year.
                Note : Use 'calendar' module.
        :return:
        """
        print('')
        y = int(input("Enter the year in YYYY: "))
        m = int(input("Enter the month in MM: "))
        print(calendar.month(y, m))


basicObj = BasicsPython()
basicObj.version()
basicObj.using_version()
basicObj.current_datetime()
basicObj.file_extenstion()
basicObj.first_last_list_item()
basicObj.list_date()
basicObj.show_calendar()
basicObj.multiplier()
basicObj.get_syntax_details()
basicObj.gen_list_tuples()
basicObj.circle_area()
basicObj.reverse_order()