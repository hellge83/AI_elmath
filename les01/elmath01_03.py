# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)

plt.plot(x, np.cos(x))
plt.plot(x, np.cos(2*x))

plt.show()