import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

Ref_point = np.load('reference_path.npy')

x = Ref_point[:, 0]
y = Ref_point[:, 1]

N = 20
dt = 0.1
L = 20.0

    
heading = np.arctan2(np.diff(y), np.diff(x))
heading = np.append(heading, heading[-1])

dx = np.diff(x)
dy = np.diff(y)
ddx = np.diff(dx)
ddy = np.diff(dy)
curvature_raw = np.abs((dx[:-1] * ddy - dy[:-1] * ddx) / (dx[:-1]**2 + dy[:-1]**2)**(1.5))
curvature = np.append(curvature_raw, [curvature_raw[-1], curvature_raw[-1]])

max_speed = 30.0
k_curve = 500 #curvature constant
speed_ref = max_speed/ (1 + k_curve * curvature)
curvature = np.convolve(curvature, np.ones(15)/15, mode='same') 


# plt.plot(curvature)
# plt.title('Curvature along reference path')
# plt.xlabel('Waypoint index')
# plt.ylabel('Curvature')
# plt.show()