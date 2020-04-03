import numpy as np
from scipy.optimize import brentq

def eq(x, a):

    return (np.sin(a*x))


roots = []
a = 0.01
while a < 0.02:
    try:
        x1 =  brentq(eq, 100, 500, a)
        pair = [a, x1]
        roots.append(pair)
    except ValueError:
        pass
    a+=0.001


print(roots)
