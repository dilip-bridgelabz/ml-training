"""
Class Person Directory
"""

from collections import defaultdict, namedtuple
from datetime import datetime
import pickle
import pickle as pkl
import pandas as pd
import numpy as np
import csv
import traceback
import zlib

"""
Since this is such a popular answer, I'd like touch on a few slightly advanced usage topics.
cPickle (or _pickle) vs pickle

It's almost always preferable to actually use the cPickle module rather than pickle because the former is written in C and is much faster. There are some subtle differences between them, but in most situations they're equivalent and the C version will provide greatly superior performance. Switching to it couldn't be easier, just change the import statement to this:
# import cPickle as pickle
"""
df = pd.DataFrame()


class Person:

    name = None
    dob = None
    p_dict = {}

    def __init__(self):
        global df
        self.data = read_object()

        self.addName()
        self.addDob()
        self.p_list = [self.name, self.dob]
        df = df.append({'Name': self.name, 'Dob': self.dob}, ignore_index=True)
        print(df.head())

    def addName(self):
        try:
            self.name = str(input("Enter the name : "))
            if len(self.name) is 0:
                raise ValueError("Please enter your name again.")
            else:
                print("Adding your name as %s" % self.name)
        except ValueError as e:
            print(e)
            self.addName()

    def addDob(self):
        dob = input("Enter the dob : ")
        try:
            self.dob = datetime.strptime(dob, "%d/%m/%Y").date()
        except ValueError:
            print('Invalid Dob!')
            self.addDob()


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    output.close()
    return


def read_object():
    data = None
    try:
        with open("person.pkl", "rb") as f:
            data = pickle.load(f) #zlib.decompress(f.read()))  #uncompressed_data = zlib.decompress(f.read())
        return data
    except pickle.UnpicklingError as e:
        # normal, somewhat expected
        return
    except (AttributeError, EOFError, ImportError, IndexError) as e:
        # secondary errors
        print(traceback.format_exc(e))
        return
    except Exception as e:
        # everything else, possibly fatal
        print(traceback.format_exc(e))
        return


class Switch:
    """
    Switcher class to define the command
    """
    def __init__(self, value):
        self._val = value

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False # Allows traceback to occur

    def __call__(self, *mconds):
        return self._val in mconds


class BreakIt(Exception): pass


if __name__ == "__main__":
    try:
        while True:
            with Switch(int(input("Enter `3` to list, `2` to search, `1` to feed OR `0` to exit."))) as case:
                if case(1):
                    person = Person()
                    save_object(person, 'person.pkl')
                elif case(0):
                    raise BreakIt
                elif case(2):
                    # This switch also supports multiple conditions (in one line)
                    print("search ")
                elif case(3):
                    p = read_object()
                    print(p)
                    # This switch also supports multiple conditions (in one line)
                    print("List")
                else:
                    # Default would occur here
                    print("Let's enter some valid input `1` or `0`!")
                    break
    except BreakIt:
        print("Exiting...")
        pass

