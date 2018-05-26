'''
1) Array and pointers(?)
        How are arguments passed in functions (call by value or reference)
        What happens when:
            Def: Foo (a):
                a[3]=10 return
            a = [1,2,3,4,5]
            foo(a)
            Print (a)

			Def foo (a):
				print(id(a))
		Assume variable “a” ultimately contains a number 1.

	Scope of variable
		Def foo():
			#Print (s)
			S = “local variable”
			Print (“in foo”, s)
		S = “global variable
		Print (“in main”, s
		foo()
		Print (“in main”, s)
		What happens if you un-comment 1st line of foo? How can you prevent it?
'''


# def foo(a):
#     a[3] = 10
#     return
# a = [1, 2, 3, 4, 5]
# foo(a)
# print(a)
#
#
# def foo(a):
#     print(id(a))


def foo():
    # print(s)
    s = "local variable"
    print("in foo", s)

s = "global variable"
print("in main", s)
foo()
print("in main", s)

# __init__, __del__

a = [1, 2, 3, 4]
b = ["a", "b", "c", "d"]
sync_index = [(aValue, bValue) for aIndex, aValue in enumerate(a) for bIndex, bValue in enumerate(b) if aIndex is bIndex]
print("List Comprehension - Sync to list keeping same index: \n", sync_index)

sync_lambda = map(lambda x, y: (x, y), a, b)
print("Lambda function    - Sync to list keeping same index: \n", list(sync_lambda))