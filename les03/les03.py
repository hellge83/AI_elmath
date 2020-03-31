import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib import axes
import matplotlib.patches
from mpl_toolkits import mplot3d
# # 2.1.2----------------
# def vect_sum(*args):
#     return [itm1 + itm2 for itm1, itm2 in zip(*args)]
#
# a = [10, 10, 10]
# b = [0, 0, -10]
# c = vect_sum(a, b)
# print(c)
# # 2.1.2----------------

# # 2.2----------------
# x = np.linspace(-5, 5, 21)
# # y = 3 * x + 1
# y2 = (-1/3) * x + 1
# # plt.plot(x, y)
# plt.plot(x, y2)
# plt.xlabel('x')
# plt.xlabel('y')
# plt.show()
# # 2.2----------------

# #2.3.1----------------
# circle1 = pylab.Circle((1, 2), radius=5, fill=False)
# ax=plt.gca()
# ax.add_patch(circle1)
# plt.axis('scaled')
# plt.show()
# #2.3.1----------------

# #2.3.1 v2----------------
# xmin = -20
# xmax = 20
# dx = 0.01
# x0 = 1
# y0 = 2
# r = 5
# xlist = np.around(np.arange(xmin, xmax, dx), decimals=4)
# ylist = y0 + np.sqrt(r ** 2 - (xlist - x0) ** 2)
# ylist2 = y0 + -1 * np.sqrt(r ** 2 - (xlist - x0) ** 2)
# plt.plot(xlist, ylist)
# plt.plot(xlist, ylist2)
# plt.show()
# #2.3.1 v2----------------

# 2.3.2----------------
# xmin = -20
# xmax = 20
# dx = 0.01
# a = 2
# b = 3
# xlist = np.around(np.arange(xmin, xmax, dx), decimals=4)
# ylist = np.sqrt((a ** 2 * b ** 2 - b ** 2 * xlist ** 2) / a ** 2)
# ylist2 =-1 * np.sqrt((a ** 2 * b ** 2 - b ** 2 * xlist ** 2) / a ** 2)
# plt.plot(xlist, ylist)
# plt.plot(xlist, ylist2)
# plt.show()
# # 2.3.3----------------

# # 2.5.1----------------
# a = 2
# b = 3
# c = 2
# d1 = 4
# d2 = 6
# xlist = np.outer(np.linspace(-2, 2, 30), np.ones(30))
# ylist = xlist.copy().T
# zlist1 = -d1 - b * ylist - a * xlist
# zlist2 = -d2 - b * ylist - a * xlist
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(xlist, ylist, zlist1 ,cmap='viridis', edgecolor='none')
# ax.plot_surface(xlist, ylist, zlist2 ,cmap='viridis', edgecolor='none')
# ax.set_title('Surface plot')
# plt.show()
# # 2.5.1----------------

# # 2.5.2----------------
# a = 2
# b = 3

# xlist = np.outer(np.linspace(-10, 10, 300), np.ones(300))
# ylist = xlist.copy().T
# zlist1 = xlist * 2 / a ** 2  + ylist ** 2 / b ** 2
# zlist2 = -1 * (xlist * 2 / a ** 2  - ylist ** 2 / b ** 2)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(xlist, ylist, zlist1 ,cmap='viridis', edgecolor='none')
# ax.plot_surface(xlist, ylist, zlist2 ,cmap='viridis', edgecolor='none')
# ax.set_title('Surface plot')
# plt.show()
# # 2.5.2----------------

# # 3.1----------------
# a1 = 1
# a2 = -1
# b1 = 2
# b2 = -2
# k1 = 3
# k2 = -3
# xlist = np.linspace(-2*np.pi, 3*np.pi, 201)
# ylist = k1 * np.cos(xlist - a1) + b1
# ylist2 = k2 * np.cos(xlist - a2) + b2
# plt.plot(xlist, ylist)
# plt.plot(xlist, ylist2)
# plt.show()
# # 3.1----------------

# # 3.3.1----------------
# def polar_to_ort(r, phi):
#     x = r * np.cos(phi)
#     y = r * np.sin(phi)
#     return x, y

# r = 1
# phi = np.pi / 2
# print(polar_to_ort(r, phi))
# # 3.3.1----------------



# # 3.3.2----------------
# xmin = -20
# xmax = 20
# dx = 0.01
# x0 = 1
# y0 = 2
# r = 5
# xlist = np.around(np.arange(xmin, xmax, dx), decimals=4)
# ylist = y0 + np.sqrt(r ** 2 - (xlist - x0) ** 2)
# ylist2 = y0 + -1 * np.sqrt(r ** 2 - (xlist - x0) ** 2)
# plt.polar(xlist, ylist)
# # 3.3.2----------------

# # 3.3.3----------------
# x = np.linspace(5, 10, 100)
# y = 8 * x
# plt.polar(x, y)
# # 3.3.3----------------

# # 4.1----------------
# x = np.linspace(-2, 5, 201)
# plt.plot(x, 1 - (1 - np.exp(x))/x)
# plt.plot(x, x**2 - 1)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim(-1,10) 
# plt.grid(True)
# plt.show()

# from scipy.optimize import fsolve

# def equations(p):
#     x, y = p
#     return (y - x**2 + 1, 1 - np.exp(x) - x + x*y)

# x1, y1 =  fsolve(equations, (-2, 1))
              
# print (x1, y1)
# # 4.1----------------

# 4.2----------------
x = np.linspace(-2, 5, 201)
plt.plot(x, 1 - (1 - np.exp(x))/x)
plt.plot(x, x**2 - 1)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1,10) 
plt.grid(True)
plt.show()

from scipy.optimize import fsolve

def equations(p):
    x, y = p
    return (y - x**2 + 1, 1 - np.exp(x) - x + x*y)

x1, y1 =  fsolve(equations, (3, 7))
              
print (x1, y1)
# 4.2----------------