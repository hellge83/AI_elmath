# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:59:05 2020

@author: snetkova
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import itertools
#05.1-----------------------
for i in range(0, 5):
    a = input()
    x = np.random.randint(0, 36)
    print(x)
#05.1-----------------------   

#05.2.1-----------------------
heads,tiles = 0, 0
n = 1000
for i in range(0, n):
    x = np.random.uniform(0, 10)
    if x<5:
        heads +=1
    else:
        tiles +=1
W_head = heads / n
W_tile = tiles / n
print (W_head + W_tile)
#05.2.1-----------------------

#05.2.2----------------------- 
arr = []
for i in range (0, 9):
    arr.append(np.random.rand(10))

rand_sum=[sum(arr[i]) for i in range(0,len(arr))]
num_bins = 5
plt.hist(rand_sum, num_bins)
plt.xlabel('sum')
plt.ylabel('Probability')
plt.title('Histogram')
plt.show()
#05.2.2----------------------- 

#05.3----------------------- 
k, n = 0, 1000
a = np.random.randint(0, 2, n)
b = np.random.randint(0, 2, n)
c = np.random.randint(0, 2, n)
d = np.random.randint(0, 2, n)
x = a + b + c + d
for i in range(0, n):
    if x[i] == 2:
        k = k + 1
p = np.math.factorial(n)/(np.math.factorial(k) * np.math.factorial(n - k)) * ((1/2) ** k) * ((1/2) ** (n - k)) 
# print(x)
# print(k, n, k/n)
print(p)
#05.3----------------------- 


#05.4-----------------------     
for p in itertools.combinations("01234", 4):
    print(''.join(str(x) for x in p))
#05.4----------------------- 

#05.5----------------------- 
n = 100
r = 0.7
x = np.random.rand(n)
y = r*x + (1 - r)*np.random.rand(n)
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

a = (np.sum(x)*np.sum(y) - n*np.sum(x*y))/(np.sum(x)*np.sum(x) - n*np.sum(x*x))
b = (np.sum(y) - a*np.sum(x))/n

A = np.vstack([x, np.ones(len(x))]).T
a1, b1 = np.linalg.lstsq(A, y, rcond=None)[0]
print(f'manual: {a}, {b}')
print(f'lib: {a1}, {b1}')
plt.plot([0, 1], [b, a + b])

xm = sum(x)/n
ym = sum(y)/n
R = sum((x - xm)*(y - ym))/((sum((x - xm) ** 2) * sum((y - ym) ** 2)) ** (1/2))

c = np.corrcoef(x, y)
print(c)
print(R)
plt.show()
#05.5----------------------- 