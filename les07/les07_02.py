#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from pylab import *
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


#--------6.3-----------
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[12, 2, -8]])
C = np.concatenate((A,B.T), axis=1)
print(np.linalg.det(A))
np.linalg.matrix_rank(A, 0.0001), np.linalg.matrix_rank(C, 0.0001)


# In[3]:


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([12, 2, -8])
#print(np.linalg.solve(A, B))
np.linalg.lstsq(A, B)


# In[4]:


#--------6.4-----------
A = np.array([ [1, 2, 3], [2, 16, 21], [4, 28, 73]])
P, L, U = scipy.linalg.lu(A)

print(P)
print(L)
print(U)
print(np.linalg.det(U))

B = np.array([12, 8, 10])
np.linalg.solve(A, B)


# In[5]:


#--------6.5-----------
def Q(x, y, z):
    return (x**2 + y**2 + z**2)

fig = figure()
ax = Axes3D(fig)
X = np.arange(-5, 5, 0.01)
ax.plot(X, Q(X, 10 * X - 14, 21 * X - 29))
plt.show()


# In[6]:


A = np.array([[1, 2, -1], [8, -5, 2]])
B = np.array([1, 12])
np.linalg.lstsq(A, B)


# In[7]:


#--------6.6-----------
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[2, 5, 11]])
C = np.concatenate((A,B.T), axis=1)
print(np.linalg.det(A))
np.linalg.matrix_rank(A, 0.0001), np.linalg.matrix_rank(C, 0.0001)


# In[8]:


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([2, 5, 11])
Q, R = np.linalg.qr(A)

print(A)
print(Q)
print(R)


# In[9]:


R1 = R[:2, :2]
R1


# In[10]:


B1 = np.dot(np.transpose(Q), B)[:2]
B1


# In[11]:


X1 = np.linalg.solve(R1, B1)
X1


# In[12]:


X = np.append(X1, 0)
print (X)
np.linalg.norm(X)


# In[13]:


np.linalg.norm(np.dot(A, X) - B)


# In[14]:


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([2, 5, 11])
np.linalg.lstsq(A, B)


# In[15]:


X = np.array([1.25, 0.5, -0.25])
np.linalg.norm(X),  np.linalg.norm(np.dot(A, X) - B)


# In[ ]:




