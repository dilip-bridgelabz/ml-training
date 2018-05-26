
import numpy as np


class NumpyArray():

    a = None

    def __init__(self):
        pass

    def version(self):
        print("numpy version using __version__ :", np.__version__)
        print("numpy version using version key :", np.version.version)
        return np.version.version

    def listToNumPyArray(self):
        list_number = [12.23, 13.32, 100, 36.32]
        print("Converting the numeric list to numpy array list:", np.array(list_number).tolist())

    def mat_three(self):
        print("3 diagonaly matrix : \n", np.eye(3))
        print("3 x 3 Matrix : \n", np.full((3, 3), 10))
        print("3 x 3 Matrix having all ones(1): \n", np.ones((3, 3)))
        print("3 x 3 Indices :\n", np.indices((3, 3)))

    def mat_v_2_10(self):
        print("3 x 3 with random 2-9 :\n", np.random.randint(2, 9, size=(3, 3)))

    def vector_update_element(self):
        print("Fill array using np.zero() with 0: \n", np.zeros((10, ), dtype=int))
        print("Fill array using np.zero() with 0.0: \n", np.zeros((10, ), dtype=float))
        print("Fill array using np.repeat(0, 10) with 0 : \n", np.repeat(0, 10))
        a = np.array([x for x in range(10)])
        a.fill(0)
        print("Fill array using np.array() & range :\n", a)
        a[5] = 11
        print("Fill array using np.array() & range :\n", a)
        a = np.array([0])
        a = np.tile(a, 10)
        print("Fill array using np.tile(a, 10) with 0 : \n", a)
        np.put(a, [5], [11])
        print("Updating the 6th element with 11 : \n", a)

    def array_update_range_elements(self):
        self.a = np.arange(12, 38)
        print("Array filled with number ranging 12 to 38: \n", self.a)

    def reverse_array(self):
        print("Reversing the array filled with number ranging 12 to 38: \n", np.flip(self.a, 0))


npArray = NumpyArray()
npArray.version()
npArray.listToNumPyArray()
npArray.mat_three()
npArray.mat_v_2_10()
npArray.vector_update_element()
npArray.array_update_range_elements()
npArray.reverse_array()