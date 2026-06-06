import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import cv2

Ref_point = np.load('reference_path.npy')
x = Ref_point[:, 0]
y = Ref_point[:, 1]

log = np.load('mpc_log.npy')

imag = cv2.imread('roundabout_satellite_cleanup.png')
imag_rgb = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(16,8))
gs = fig.add_gridspec(4, 2, width_ratios=[2, 1])

ax_main = fig.add_subplot(gs[:, 0])      # left column, all 4 rows — the animation
ax_cte  = fig.add_subplot(gs[0, 1])      # right column, row 0 — CTE plot
ax_spd  = fig.add_subplot(gs[1, 1])      # right column, row 1 — speed plot
ax_str  = fig.add_subplot(gs[2, 1])      # right column, row 2 — steering plot
ax_lat  = fig.add_subplot(gs[3, 1])      # right column, row 3 — lateral accel plot

def get_car_corners(cx, cy, theta, width=6, length=12):
    corners = np.array([
        [-length/2, -width/2],
        [ length/2, -width/2],
        [ length/2,  width/2],
        [-length/2,  width/2]
    ])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    rotated = (rot @ corners.T).T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated



#--- Main panel ---
ax_main.imshow(imag_rgb)
ax_main.plot(x, y, 'y--', linewidth=1.5, label='Reference path')
ax_main.plot(log[0], log[1], 'g-', linewidth=1.5, label='MPC trajectory')
ax_main.legend(loc='upper right', fontsize=8)
ax_main.axis('off')
ax_main.set_title('MPC Roundabout Navigation', fontsize=11)

#Create empty line objects that will grow during animation
line_cte, = ax_cte.plot([], [], 'g-', linewidth=1)
ax_cte.set_ylabel('CTE (px)', fontsize=8)
ax_cte.set_title('Cross-Track Error', fontsize=9)
ax_cte.set_xlim(0, len(log[0]))
ax_cte.set_ylim(0, max(log[2]) * 1.2 + 0.1)
ax_cte.grid(True, alpha=0.3)

line_spd, = ax_spd.plot([], [], 'b-', linewidth=1)
ax_spd.set_ylabel('Speed (px/s)', fontsize=8)
ax_spd.set_title('Speed', fontsize=9)
ax_spd.set_xlim(0, len(log[0]))
ax_spd.set_ylim(0, max(log[3]) * 1.2)
ax_spd.grid(True, alpha=0.3)

line_str, = ax_str.plot([], [], 'r-', linewidth=1)
ax_str.set_ylabel('Steering (°)', fontsize=8)
ax_str.set_title('Steering Angle', fontsize=9)
ax_str.set_xlim(0, len(log[0]))
ax_str.set_ylim(min(np.degrees(log[4])) * 1.2, max(np.degrees(log[4])) * 1.2)
ax_str.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
ax_str.grid(True, alpha=0.3)

lat_g = log[5] * 0.221 / 9.81
line_lat, = ax_lat.plot([], [], 'm-', linewidth=1)
ax_lat.axhline(y=0.3, color='r', linestyle='--', linewidth=1.5, label='0.3g limit')
ax_lat.set_ylabel('Lat Accel (g)', fontsize=8)
ax_lat.set_title('Lateral Acceleration', fontsize=9)
ax_lat.set_xlim(0, len(log[0]))
ax_lat.set_ylim(0, 0.35)
ax_lat.legend(fontsize=7)
ax_lat.grid(True, alpha=0.3)
ax_lat.set_xlabel('Timestep', fontsize=8)

plt.tight_layout()

vline_cte = ax_cte.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)
vline_spd = ax_spd.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)
vline_str = ax_str.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)
vline_lat = ax_lat.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)

# Create car patch ONCE before animation
initial_corners = get_car_corners(log[0][0], log[1][0], 0)
car_patch = plt.Polygon(initial_corners, color='lime', zorder=5)
ax_main.add_patch(car_patch)

def update(frame):
    # Update car position
    car_x = log[0][frame]
    car_y = log[1][frame]
    next_frame = min(frame+1, len(log[0])-1)
    car_theta = np.arctan2(
        log[1][next_frame] - log[1][frame],
        log[0][next_frame] - log[0][frame]
    )
    corners = get_car_corners(car_x, car_y, car_theta)
    car_patch.set_xy(corners)
    
    # Update vertical lines
    vline_cte.set_xdata([frame])
    vline_spd.set_xdata([frame])
    vline_str.set_xdata([frame])
    vline_lat.set_xdata([frame])

    timesteps = np.arange(frame+1)
    line_cte.set_data(timesteps, log[2][:frame+1])
    line_spd.set_data(timesteps, log[3][:frame+1])
    line_str.set_data(timesteps, np.degrees(log[4][:frame+1]))
    line_lat.set_data(timesteps, lat_g[:frame+1])
    
    return [car_patch, vline_cte, vline_spd, vline_str, vline_lat,
        line_cte, line_spd, line_str, line_lat]
plt.ion()
fig.canvas.draw()

frame = 0
step = 5  # process every 5th frame
while plt.fignum_exists(fig.number):
    update(frame)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)
    frame = (frame + step) % len(log[0])

