import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

Ref_point = np.load('reference_path.npy')

x = Ref_point[:, 0]
y = Ref_point[:, 1]

x = x[::-1]
y = y[::-1]

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
curvature_signed_raw = (dx[:-1] * ddy - dy[:-1] * ddx) / (dx[:-1]**2 + dy[:-1]**2)**(1.5)
curvature = np.append(curvature_raw, [curvature_raw[-1], curvature_raw[-1]])

path_diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
s_path = np.insert(np.cumsum(path_diffs), 0, 0.0)
total_s = s_path[-1]
print(f"Total path length: {total_s:.1f} pixels")

max_speed = 30.0
k_curve = 500 
curvature = np.convolve(curvature, np.ones(15)/15, mode='same')
curvature_signed = np.append(curvature_signed_raw, [curvature_signed_raw[-1], curvature_signed_raw[-1]])
curvature_signed = np.convolve(curvature_signed, np.ones(15)/15, mode='same')
speed_ref = max_speed/ (1 + k_curve * curvature) 
print("max curvature =", np.max(curvature))
print("mean curvature =", np.mean(curvature))
# --- MPC Parameters ---

# plt.plot(curvature)
# plt.title('Curvature along reference path')
# plt.xlabel('Waypoint index')
# plt.ylabel('Curvature')
# plt.show()


Q1 = 50.0 
Q2 = 20.0
Q3 = 100.0
R1 = 0.1
R2 = 0.1
R3 = 10.0
R4 = 10.0

delta_max = np.radians(28)
a_max = 3.0
v_min = 2.0
v_max = 8.0
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

def get_reference_at_distance(s_target):
    s_target = s_target % total_s
    ref_x = np.interp(s_target, s_path, x)
    ref_y = np.interp(s_target, s_path, y)
    heading_unwrapped = np.unwrap(heading)
    ref_heading = np.interp(s_target, s_path, heading_unwrapped)
    ref_heading = (ref_heading + np.pi) % (2 * np.pi) - np.pi
    ref_v = np.interp(s_target, s_path, speed_ref)
    ref_kappa = np.interp(s_target, s_path, curvature)
    ref_kappa_signed = np.interp(s_target, s_path, curvature_signed)
    return ref_x, ref_y, ref_heading, ref_v, ref_kappa, ref_kappa_signed

def mpc_solve(state, waypoint_idx):

    
    states = cp.Variable((4, N+1))
    inputs = cp.Variable((2, N))

    
    cost = 0
    constraints = [states[:, 0] == state]
    prev_ref_theta = state[2]
    current_s = s_path[waypoint_idx]


    for t in range(N):
        # Physical distance the car is expected to be at timestep t
        s_target = current_s + t * speed_ref[waypoint_idx] * dt

        # Get interpolated reference at that distance
        ref_x, ref_y, ref_theta, ref_v, ref_kappa, ref_kappa_signed = get_reference_at_distance(s_target)

        # Normalize heading relative to previous step
        while ref_theta - prev_ref_theta > np.pi:
            ref_theta -= 2 * np.pi
        while ref_theta - prev_ref_theta < -np.pi:
            ref_theta += 2 * np.pi
        prev_ref_theta = ref_theta

        # Cost terms
        cost += (Q1 * (cp.square(states[0, t] - ref_x) + cp.square(states[1, t] - ref_y)) +
                 Q2 * cp.square(states[2, t] - ref_theta) +
                 Q3 * cp.square(states[3, t] - ref_v) +
                 R1 * cp.abs(inputs[0, t]) + R2 * cp.abs(inputs[1, t]))

        if t > 0:
            cost += (R3 * cp.square(inputs[0, t] - inputs[0, t-1]) +
                     R4 * cp.square(inputs[1, t] - inputs[1, t-1]))

        # Linearization at correct reference point
        ref_state = np.array([ref_x, ref_y, ref_theta, ref_v])
        delta_ref = np.arctan(L * ref_kappa_signed) if abs(ref_kappa) > 1e-6 else 0.0
        A, B = get_linear_model(ref_state, delta_ref)

        # Affine offset term
        ref_next = np.array([
            ref_state[0] + ref_state[3] * np.cos(ref_state[2]) * dt,
            ref_state[1] + ref_state[3] * np.sin(ref_state[2]) * dt,
            ref_state[2] + (ref_state[3] / L) * np.tan(delta_ref) * dt,
            ref_state[3]
        ])
        c = ref_next - A @ ref_state - B @ np.array([delta_ref, 0.0])
        constraints += [states[:, t+1] == A @ states[:, t] + B @ inputs[:, t] + c]

        # Speed dependent steering limit
        steer_limit = delta_max / (1 + k_speed * ref_v)
        constraints += [cp.abs(inputs[0, t]) <= steer_limit]

        # Lateral acceleration limit
        if ref_kappa > 1e-6:
            max_v = np.sqrt(lat_accel_max / ref_kappa)
            constraints += [states[3, t] <= max_v]

    # Global bounds
    constraints += [cp.abs(inputs[0, :]) <= delta_max]
    constraints += [cp.abs(inputs[1, :]) <= a_max]
    constraints += [states[3, :] >= v_min]
    constraints += [states[3, :] <= v_max]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, warm_start=True)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.SCS)

    return inputs[:, 0].value

car_state = np.array([x[0], y[0], heading[0], 5])

log_x       = []
log_y       = []
log_cte     = []
log_speed   = []
log_steering = []
log_lat_accel = []

for n in range(1000):
    distances = np.sqrt((x - car_state[0])**2 + (y - car_state[1])**2)
    way_p = (np.argmin(distances)) % 500
    

    optimal_input = mpc_solve(car_state, way_p)

    if optimal_input is None:
         print(f"MPC failed at way point {way_p}")
         break
        
    delta = optimal_input[0]
    accel = optimal_input[1]

    
    x_k = car_state[0] + car_state[3] * np.cos(car_state[2]) * dt
    y_k = car_state[1] + car_state[3] * np.sin(car_state[2]) * dt
    theta_k = car_state[2] + (dt * car_state[3] * np.tan(optimal_input[0])) / L
    theta_k = (theta_k + np.pi) % (2 * np.pi) - np.pi
    v_k = car_state[3] + optimal_input[1] * dt

    car_state = np.array([x_k, y_k, theta_k, v_k])
    log_x.append(car_state[0])
    log_y.append(car_state[1])
    log_cte.append(distances[way_p])
    log_speed.append(car_state[3])
    log_steering.append(delta)
    log_lat_accel.append(car_state[3]**2 * curvature[way_p])

# Constraint verification report
log = np.array([log_x, log_y, log_cte, log_speed, log_steering, log_lat_accel])

print("\n=== CONSTRAINT VERIFICATION ===")

# Steering magnitude (Constraint 1)
max_steer = np.max(np.abs(log_steering))
print(f"Max steering angle: {np.degrees(max_steer):.1f}° (limit: {np.degrees(delta_max):.1f}°) {'✓' if max_steer <= delta_max else '✗ VIOLATED'}")

# Speed bounds (Constraints 7 and 8)
max_speed_actual = np.max(log_speed)
min_speed_actual = np.min(log_speed)
print(f"Speed range: {min_speed_actual:.2f} to {max_speed_actual:.2f} px/s (limits: {v_min} to {v_max}) {'✓' if max_speed_actual <= v_max and min_speed_actual >= v_min else '✗ VIOLATED'}")

# Lateral acceleration (Constraint 4)
max_lat = np.max(log_lat_accel)
print(f"Max lateral accel: {max_lat:.3f} px/s² = {max_lat * 0.221:.3f} m/s² = {max_lat * 0.221 / 9.81:.3f}g (limit: 0.3g) {'✓' if max_lat <= lat_accel_max else '✗ VIOLATED'}")

# Steering jerk (Constraint 5)
steering_jerk = np.max(np.abs(np.diff(log_steering)))
print(f"Max steering jerk: {np.degrees(steering_jerk):.1f}°/step {'✓ smooth' if np.degrees(steering_jerk) < 5.0 else '⚠ check jerk weights'}")

# Acceleration jerk (Constraint 6)
speed_changes = np.diff(log_speed)
accel_log = speed_changes / dt
max_accel = np.max(np.abs(accel_log))
print(f"Max acceleration: {max_accel:.2f} px/s² (limit: {a_max}) {'✓' if max_accel <= a_max else '✗ VIOLATED'}")

print("================================\n")

np.save('mpc_log.npy', np.array([log_x, log_y, log_cte, log_speed, log_steering, log_lat_accel]))
print(f"Simulation complete: {len(log_x)} timesteps logged")