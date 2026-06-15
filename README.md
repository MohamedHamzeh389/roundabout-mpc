# Roundabout MPC Trajectory Planner
  
<img width="800" height="600" alt="MPCRoundabout-ezgif com-gif-to-mp4-converter" src="https://github.com/user-attachments/assets/55e57109-98dc-46e4-b1d7-e6df3f7b4f87" />

An autonomous vehicle controller built in Python that navigates a **real satellite image** of a roundabout in LaSalle, Ontario using **Model Predictive Control (MPC)**. The system includes satellite image extraction, reference path generation, a linearized bicycle model MPC controller with 10 automotive safety constraints, gap-acceptance yielding behavior, a circulating obstacle vehicle, and a real-time OpenCV telemetry dashboard.
 
---
 
## Demo
 
 
The animation shows:
- Green car (ego vehicle) approaching the roundabout from an entry road
- Gap acceptance check at the yield line - car stops if circulating vehicle is too close
- Smooth re-entry once gap clears
- Full ring arc navigation and exit onto departure road
- Orange dashed lines showing alternative exit options
- Live telemetry: cross-track error, speed, steering angle, lateral acceleration
---
 
## Project Structure
 
```
roundabout-mpc/
├── circ_log.npy                  # Obstacle vehicle position log
├── circle_path.npy               # Circular path for obstacle vehicle
├── get_map.py                    # Fetches satellite image via Google Maps Static API
├── map_extraction.py             # HSV thresholding + morphological ops → road mask
├── mpc_extraction.py             # Click-based reference path generation (spline fitting)
├── mpc_controller.py             # Full MPC controller with yielding logic
├── visualizer_cv.py              # Real-time OpenCV animation + telemetry dashboard
├── reference_path1.npy           # Main entry-arc-exit reference path (500 waypoints)
├── reference_path2.npy           # Alternative exit option 2
├── Reference_path3.npy           # Alternative exit option 3
├── Reference_path4.npy           # Alternative exit option 4
├── mpc_log.npy                   # Simulation output (x, y, CTE, speed, steering, lat_accel)
├── road_mask.png                 # Filtered Satellite Roundabout picture
├── roundabout_satellite.png      # Google maps - extracted picture
└── roundabout_satellite_cleanup.png  # Google maps - extracted picture (Cars filtered)
```
 
---
 
## Pipeline Overview
 
```
Google Maps API → Satellite Image
        ↓
HSV Thresholding → Road Mask
        ↓
Click-Based Spline → Reference Path (500 waypoints)
        ↓
Arc-Length Parameterization → Distance-Based Reference Interpolation
        ↓
Linearized Bicycle Model → MPC (CVXPY / OSQP)
        ↓
Gap Acceptance Logic → Yielding Constraint
        ↓
OpenCV Real-Time Visualization + Telemetry
```
 
---
 
## Phase 1 - Map Extraction
 
The satellite image is fetched using the **Google Maps Static API** at zoom level 19. The scale conversion uses the standard Web Mercator formula:
 
```
metres_per_pixel = 156543.03392 × cos(latitude × π/180) / 2^zoom
```
 
At latitude 42.2° N (LaSalle, Ontario) and zoom 19, this gives **0.221 metres/pixel**. This conversion factor is used throughout the project to translate pixel distances into real-world measurements for constraints like lateral acceleration limits.
 
The raw image is processed using **OpenCV HSV thresholding** and **morphological operations** (dilation, erosion) to extract a clean binary road mask - white pixels are drivable road, black pixels are everything else.
 
---
 
## Phase 2 - Reference Path Generation
 
A user clicks waypoints on the road mask image. A **SciPy parametric spline** (`splprep` / `splev`) is fitted through those points to generate a smooth 500-waypoint reference path saved as `reference_path.npy`.
 
For the circular obstacle vehicle path, a **least-squares circle fit** (`scipy.optimize.least_squares`) finds the best-fit circle centre and radius from the clicked points.
 
**Arc-length parameterization** is computed once at startup:
 
```python
path_diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
s_path = np.insert(np.cumsum(path_diffs), 0, 0.0)
total_s = s_path[-1]  # ≈ 809.7 pixels ≈ 178.9 metres
```
 
This is the key architectural fix that solved the MPC's biggest failure mode - see the Controller section below.
 
---
 
## Phase 3 - MPC Controller
 
### Bicycle Kinematic Model
 
The car is modelled as a **bicycle kinematic model** - a standard 2-axle vehicle approximation used in automotive control. The four state variables are position (x, y), heading angle (θ), and speed (v). The two inputs are steering angle (δ) and acceleration (a).
 
The discrete-time dynamics are:
 
```
x_{k+1}  = x_k + v_k · cos(θ_k) · dt
y_{k+1}  = y_k + v_k · sin(θ_k) · dt
θ_{k+1}  = θ_k + (v_k / L) · tan(δ_k) · dt
v_{k+1}  = v_k + a_k · dt
```
 
where `L = 20.0 px` (wheelbase ≈ 4.4 metres) and `dt = 0.1 s`.
 
### Why Linearization is Necessary
 
MPC requires solving a **convex quadratic program** at every timestep. The bicycle model is nonlinear - cosine and sine of θ appear in the dynamics. CVXPY's OSQP solver only accepts linear constraints, so the model must be linearized around a reference trajectory at each step.
 
**Jacobian Linearization** is used - the idea that any smooth nonlinear function is approximately linear at a single point (the same concept as a tangent line). The nonlinear dynamics `f(state, input)` are approximated as:
 
```
state_{t+1} ≈ A · state_t + B · input_t + c
```
 
where A and B are the Jacobian matrices (partial derivatives) evaluated at the reference state and reference steering angle, and c is an affine correction term that accounts for the linearization residual - it shifts the linear approximation to match the nonlinear function at the operating point.
 
```python
def get_linear_model(state, delta0):
    theta0 = state[2]
    v0     = state[3]
 
    A = np.array([[1, 0, -v0*sin(θ)*dt,  cos(θ)*dt ],
                  [0, 1,  v0*cos(θ)*dt,  sin(θ)*dt ],
                  [0, 0,  1,             tan(δ)/L*dt],
                  [0, 0,  0,             1          ]])
 
    B = np.array([[0,                    0 ],
                  [0,                    0 ],
                  [v0*dt / (L*cos²(δ)), 0 ],
                  [0,                    dt]])
    return A, B
```
 
**A describes how the current state evolves on its own. B describes how the optimizer's inputs change the next state. Together they are the physics rules the optimizer must obey.**
 
Note that acceleration `a` does not appear in A at all - when you take the partial derivative of the dynamics with respect to `a`, it only appears in the speed row of B. Similarly, x and y have zero columns in B because position is not directly controlled by inputs - it changes only through the dynamics propagated by A.
 
### The Critical Architectural Fix - Distance-Based Reference Interpolation
 
The original implementation indexed the reference path by array index: `idx = (waypoint_idx + t) % 500`. This caused a fatal mismatch:
 
- Car travel per timestep: `v × dt = 8.0 × 0.1 = 0.8 pixels`
- Array index step distance: `circumference / 500 = 738 / 500 = 1.477 pixels`
Over the N=20 horizon the reference was **29.6 pixels ahead** of where the car could physically reach (16.0 pixels). The optimizer tried to close an impossible gap, the linearization gradients pointed the wrong direction, and the solver commanded maximum steering in the opposite direction - the car spiralled off the path every time.
 
The fix advances the reference by actual physical distance traveled:
 
```python
s_target = current_s + t * speed_ref[waypoint_idx] * dt
ref_x, ref_y, ref_theta, ref_v, ... = get_reference_at_distance(s_target)
```
 
This solved the failure mode completely - CTE dropped from 80+ pixels (catastrophic drift) to sub-pixel tracking throughout the entire simulation.
 
### Cost Function
 
The optimizer minimizes a weighted sum of tracking errors and input costs over a horizon of N=20 timesteps (2 seconds lookahead):
 
```
J = Σ_{t=0}^{N} [ Q1·(Δx² + Δy²) + Q2·Δθ² + Q3·Δv² + R1·|δ| + R2·|a| + R3·Δδ² + R4·Δa² ]
```
 
| Weight | Value | Purpose |
|--------|-------|---------|
| Q1 | 100.0 | Position tracking — penalizes lateral deviation from path |
| Q2 | 20.0 | Heading tracking — keeps car pointed along path direction |
| Q3 | 500.0 | Speed tracking — follows curvature-based speed profile |
| R1 | 50.0 | Steering magnitude — discourages large steering angles |
| R2 | 100.0 | Acceleration magnitude — discourages aggressive throttle/brake |
| R3 | 100.0 | Steering jerk — penalizes rapid steering changes (comfort) |
| R4 | 100.0 | Acceleration jerk — penalizes rapid speed changes (comfort) |
 
R3 and R4 are **soft constraints** — they appear in the cost function as quadratic penalties rather than hard inequalities. This allows small jerk when necessary (e.g. during yielding) while still discouraging it in normal driving.
 
### Safety Constraints (10 Total)
 
All hard constraints are enforced as strict mathematical inequalities inside CVXPY:
 
| # | Constraint | Formula | Value |
|---|-----------|---------|-------|
| 1 | Max steering angle | \|δ\| ≤ delta_max | 28° |
| 2 | Speed-dependent steering | \|δ\| ≤ delta_max / (1 + k·v) | k=0.05 |
| 3 | Max acceleration | \|a\| ≤ a_max | 3.0 px/s² |
| 4 | Lateral acceleration | v ≤ √(lat_accel_max / κ) | 0.3g ISO 22737 |
| 5 | Steering jerk | R3·(δ_t − δ_{t-1})² | soft penalty |
| 6 | Acceleration jerk | R4·(a_t − a_{t-1})² | soft penalty |
| 7 | Min speed | v ≥ v_min | 0.0 px/s |
| 8 | Max speed | v ≤ v_max | 8.0 px/s |
| 9 | Bicycle dynamics | state_{t+1} = A·state_t + B·u_t + c | exact |
| 10 | Initial state | state_0 = current car state | exact |
 
**Constraint 2 - Speed-dependent steering limit** is the most physically meaningful safety constraint. At high speed a car cannot safely turn as sharply, mimicking real vehicle dynamics and preventing rollover:
 
```python
steer_limit = delta_max / (1 + k_speed * ref_v)
constraints += [cp.abs(inputs[0, t]) <= steer_limit]
```
 
**Constraint 4 - Lateral acceleration limit** bounds the centripetal force felt by occupants during cornering. 0.3g is the threshold for comfortable cornering as defined in ISO 22737 (the international standard for low-speed autonomous vehicles). It is converted from m/s² to px/s² using the map scale:
 
```python
lat_accel_max = 0.3 × 9.81 / 0.221  # = 13.32 px/s²
```
 
### Curvature-Based Speed Profile
 
Reference speed at each waypoint is automatically reduced in tight curves:
 
```python
speed_ref = max_speed / (1 + k_curve * curvature)
speed_ref = np.clip(speed_ref, v_min, v_max)
```
 
This mimics how human drivers naturally slow before a curve. Curvature κ is computed from the second derivative of the path and smoothed with a 15-point moving average to prevent spikes at waypoint transitions from destabilizing the controller.
 
### Solver
 
The optimization problem is solved at every timestep (every 0.1s) using **OSQP** (Operator Splitting Quadratic Program) with warm-starting from the previous solution. If OSQP returns infeasible, **SCS** (Splitting Conic Solver) is used as fallback. If both fail, the last known good steering angle is held with zero acceleration.
 
---
 
## Phase 4 - Gap Acceptance Yielding
 
### Why Not CVXPY for the Gap Check?
 
The distance between two vehicles involves `x_ego` which is a CVXPY optimization variable. The expression `(x_ego − x_obs)²` is nonlinear and non-convex - CVXPY cannot handle it as a constraint. All gap calculations are done in plain NumPy **before** `mpc_solve` is called. CVXPY only sees the result as a simple speed target adjustment — which is perfectly linear.
 
### Gap Acceptance Logic
 
The obstacle vehicle advances along the circle path at constant speed using arc-length parameterization (same technique as the ego vehicle's reference interpolation):
 
```python
circ_s = (circ_s + circ_speed * dt) % circ_total
circ_idx = np.searchsorted(circ_arc_length, circ_s)
circ_x = cx_circle[circ_idx]
circ_y = cy_circle[circ_idx]
```
 
The gap check at every timestep:
 
```python
circ_dist_to_yield  = (circ_s_yield - circ_s) % circ_total
time_to_conflict    = circ_dist_to_yield / circ_speed
car_approaching     = time_to_conflict < gap_threshold        # 6.0 seconds
car_in_intersection = circ_to_yield < 100.0                  # already nearby
yield_required      = (ego_to_yield < yield_radius) and (car_approaching or car_in_intersection)
```
 
When `yield_required = True`, the MPC target speed is set to 0.0 and the speed tracking weight is raised to Q3=2000, causing the optimizer to plan a smooth full stop within the acceleration limits. Once the gap clears, the target speed returns to the curvature-based profile and the car re-enters smoothly - no hard-coded delays or state machines needed.
 
### Constraint Verification Results
 
```
Max steering angle:  17.2°      (limit: 28.0°)  ✓
Speed range:         0.0–8.0    (limits: 0–8.0) ✓
Max lateral accel:   0.008g     (limit: 0.3g)   ✓
Max steering jerk:   4.3°/step                  ✓ smooth
Max acceleration:    3.00 px/s² (limit: 3.0)    ✓
```
 
---
 
## Phase 5 - Real-Time Visualization
 
The visualizer (`visualizer_cv.py`) renders at ~60fps using OpenCV's direct pixel manipulation on a NumPy array - no Matplotlib rendering overhead. Each frame:
 
1. Copies the pre-rendered base image (satellite + reference paths drawn once before the loop)
2. Draws the growing trajectory trail with `cv2.polylines`
3. Draws the ego vehicle as a rotated rectangle using a 2D rotation matrix applied to the four corner vectors
4. Draws the obstacle vehicle in red using the same function
5. Updates the right panel with four live scrolling telemetry charts (last 200 timesteps)
Alternative exit paths are drawn as dashed orange lines on the base image so they appear without any per-frame computation cost.
 
---
 
## Setup
 
```bash
git clone https://github.com/MohamedHamzeh389/roundabout-mpc
cd roundabout-mpc
pip install numpy scipy matplotlib cvxpy opencv-python
```
 
Run in order:
 
```bash
python get_map.py           # fetch satellite image (requires Google Maps API key in file)
python map_extraction.py    # generate road mask
python mpc_extraction.py    # click reference path waypoints, press Q when done
python mpc_controller.py    # run MPC simulation (~10 min for 1200 timesteps)
python visualizer_cv.py     # open real-time animation
```
 
---
 
## Technical Stack
 
| Component | Library |
|-----------|---------|
| Convex optimization | CVXPY 1.4+ with OSQP / SCS solvers |
| Image processing | OpenCV 4.x |
| Numerical computing | NumPy, SciPy |
| Spline fitting | scipy.interpolate.splprep / splev |
| Least-squares circle fit | scipy.optimize.least_squares |
| Visualization | OpenCV (real-time), Matplotlib (static) |
 
---
 
## Real-World Scale
 
| Parameter | Pixels | Metres |
|-----------|--------|--------|
| Image resolution | 640×640 px | 141.4×141.4 m |
| Scale factor | 1 px | 0.221 m |
| Ring radius | 118 px | 26.1 m |
| Ring circumference | 741 px | 163.8 m |
| Wheelbase L | 20 px | 4.4 m |
| Max speed v_max | 8 px/s | 1.77 m/s (6.4 km/h) |
 
---
 
## Author
 
**Mohamed Hamzeh** - 2nd Year Electrical Engineering Co-op, University of Windsor
 
