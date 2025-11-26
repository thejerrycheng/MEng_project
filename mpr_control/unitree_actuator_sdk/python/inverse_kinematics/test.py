import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Just plot a simple helix or something
theta = np.linspace(0, 6*np.pi, 200)
z = np.linspace(0, 2, 200)
r = 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='3D helix')
plt.legend()
plt.show()
