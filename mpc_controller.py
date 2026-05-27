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
k_curve = 500 
speed_ref = max_speed/ (1 + k_curve * curvature)
curvature = np.convolve(curvature, np.ones(15)/15, mode='same') 

# --- MPC Parameters ---

# plt.plot(curvature)
# plt.title('Curvature along reference path')
# plt.xlabel('Waypoint index')
# plt.ylabel('Curvature')
# plt.show()


Q1 = 10.0 
Q2 = 5.0
Q3 = 1.0
R1 = 0.1
R2 = 0.1
R3 = 0.5
R4 = 0.5

delta_max = np.radians(28)
a_max = 3.0
v_min = 2.0
v_max = max_speed
k_speed = 0.05
lat_accel_max = 2.943 / 0.221 #0.3g (multiplied by 9.81) - Limit for comfortable lateral acceleration
