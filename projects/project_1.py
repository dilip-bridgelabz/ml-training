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


df = pd.DataFrame({'type': list('ABB')
                   # 'image': [np.random.random((64, 64,)) for i in range(3)],
                   # 'temperature_vec': [np.random.random((60,)) for i in range(3)],
                   # 'magnetic_field_vec': [np.random.random((60,)) for i in range(3)],
                  })
print(df)


# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict


# Call lists2dict: rs_fxn
rs_fxn = lists2dict([], [])

# Print rs_fxn
print(rs_fxn)


class Person:

    name = None
    dob = None
    p_dict = {}
    col_names = ['Name', 'Age']
    df = pd.DataFrame(columns=col_names)

    def __init__(self):

        self.addName()
        self.addDob()
        self.p_list = [self.name, self.dob]
        self.p_dict[self.name] = self.p_list

        ser1 = pd.Series(self.p_list)
        ser1.name = self.name
        ser1.head()
        if self.df.empty:
            self.df = pd.DataFrame([ser1], columns=["name", "age"])
        else:
            self.pd.DataFrame(np.insert(df.values, 0, values=ser1, axis=0))

        print(self.df.head())
        # list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]
        #
        # # Turn list of dicts into a dataframe: df
        # df = pd.DataFrame(list_of_dicts)
        #
        # # Print the head of the dataframe
        # print(df.head())

        # Create an empty table of lookup tables for each field name that maps
        # each unique field value to a list of record-list indices of the ones
        # that contain it.
        # self.lookup_tables = defaultdict(lambda: defaultdict(list))

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

    def retrieve(self, **kwargs):
        """ Fetch a list of records with a field name with the value supplied
            as a keyword arg (or return None if there aren't any). """

        with open('person.pkl', 'rb') as f:
            person = pickle.load(f)
            print(person)
        if len(kwargs) != 1: raise ValueError(
            'Exactly one fieldname/keyword argument required for function '
            '(%s specified)' % ', '.join([repr(k) for k in kwargs.keys()]))
        field, value = list(kwargs.items())[0]  # Get only keyword arg and value.
        if field not in self.valid_fieldnames:
            raise ValueError('keyword arg "%s" isn\'t a valid field name' % field)
        if field not in self.lookup_tables:  # Must create field look up table.
            for index, record in enumerate(self.records):
                value = getattr(record, field)
                self.lookup_tables[field][value].append(index)
        matches = [self.records[index]
                    for index in self.lookup_tables[field].get(value, [])]
        return matches if matches else None


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

    # person =
    # p.addName()
    # p.addDob()

