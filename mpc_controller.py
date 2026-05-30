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


def get_linear_model(state, delta0):
    theta0 = state[2]
    v0 = state[3]

    A = np.array([[1, 0, -v0*np.sin(theta0)*dt, np.cos(theta0)*dt], 
                  [0, 1, v0*np.cos(theta0)*dt, np.sin(theta0)*dt], 
                  [0, 0, 1, (1/L)*np.tan(delta0)*dt],
                  [0, 0, 0, 1]])  
     
    B = np.array([[0, 0], 
                  [0, 0], 
                  [(v0*dt)/(L*(np.cos(delta0))**2), 0],
                  [0, dt]])
    return A, B

def mpc_solve(state, waypoint_idx):

    
    states = cp.Variable((4, N+1))
    inputs = cp.Variable((2, N))

    
    cost = 0

    for t in range(0,N):
        idx = (waypoint_idx + t) % 500

        ref_x     = x[idx]
        ref_y     = y[idx]
        ref_theta = heading[idx]
        ref_v     = speed_ref[idx]

        cost += (Q1 * (cp.square(states[0, t] - ref_x) + cp.square(states[1, t] - ref_y)) + 
                     Q2 * (cp.square(states[2, t] - ref_theta)) + 
                     Q3 * (cp.square(states[3, t] -  ref_v))
                     + R1 * (cp.abs(inputs[0, t])) + R2 * (cp.abs(inputs[1, t])))
            
        if t > 0:
            cost += (R3 * (cp.square(inputs[0, t] - inputs[0, t - 1])) + 
                     R4 * (cp.square(inputs[1, t] - inputs[1, t - 1])) 
                )


    constraints = []

    constraints += [states[:, 0] == state]

    for t in range(N):
        idx = (waypoint_idx + t) % 500
        ref_state = np.array([x[idx], y[idx], heading[idx], speed_ref[idx]])
        A, B = get_linear_model(ref_state, 0.0)
        constraints += [states[:, t+1] == A @ states[:, t] + B @ inputs[:, t]]

    constraints += [cp.abs(inputs[0, :]) <= delta_max]
    constraints += [cp.abs(inputs[1, :]) <= a_max]
        
    constraints += [states[3, :] >= v_min] 
    constraints += [states[3, :] <= v_max]

    for t in range(N):
        idx = (waypoint_idx + t) % 500
        v_ref_t = speed_ref[idx]
        steer_limit = delta_max / (1 + k_speed * v_ref_t)
        constraints += [cp.abs(inputs[0, t]) <= steer_limit]

    for t in range(N):
        idx = (waypoint_idx + t) % 500
        kappa = curvature[idx]
        if kappa > 1e-6:
            max_v = np.sqrt(lat_accel_max / kappa)
            constraints += [states[3, t] <= max_v]
            
    problem = cp.Problem(cp.Minimize(cost), constraints)

    problem.solve(solver=cp.OSQP, warm_start=True)

    return inputs[:,0].value