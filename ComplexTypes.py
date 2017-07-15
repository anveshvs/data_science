# -*- coding: utf-8 -*-
"""
Created on Fri May 19 08:11:31 2017

@author: Anvesh
"""

list1 = [10,20,30]
type(list1)
print(list1)
#
#list2 = range(1,10,1)
#print(list2)

##Operators
list1[0]
list1[-1]
list1[0:2]
list1[:2]
list1[2:]

##Api usage
list1.append(10)
list1.insert(3,7)
list1.append("ji")
list1.sort()
print(list1)
len(list1)

###For loop
for x in list1:
    print(x)



