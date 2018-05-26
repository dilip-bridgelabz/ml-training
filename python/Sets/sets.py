#!/usr/bin/env python

x = set(["Postcard", "Radio", "Telegram"])
y = set(["Radio","Television"])
print( x.difference(y) )
print( y.difference(x) )


x = set(["a","b","c","d"])
y = set(["c","d"])
print( x.issubset(y) )



x = set(["a","b","c","d"])
y = set(["c","d"])
print( x.issuperset(y) )


x = set(["a","b","c","d"])
y = set(["c","d"])
print( x.intersection(y) )