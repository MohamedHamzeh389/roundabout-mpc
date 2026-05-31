import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

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

 #--- Main panel ---
ax_main.imshow(imag_rgb)
ax_main.plot(x, y, 'y--', linewidth=1.5, label='Reference path')
ax_main.plot(log[0], log[1], 'g-', linewidth=1.5, label='MPC trajectory')
ax_main.legend(loc='upper right', fontsize=8)
ax_main.axis('off')
ax_main.set_title('MPC Roundabout Navigation', fontsize=11)

# --- Metric panels ---
timesteps = np.arange(len(log[0]))

ax_cte.plot(timesteps, log[2], 'g-', linewidth=1)
ax_cte.set_ylabel('CTE (px)', fontsize=8)
ax_cte.set_title('Cross-Track Error', fontsize=9)
ax_cte.set_xlim(0, len(log[0]))
ax_cte.set_ylim(0, max(log[2]) * 1.2 + 0.1)
ax_cte.grid(True, alpha=0.3)

ax_spd.plot(timesteps, log[3], 'b-', linewidth=1)
ax_spd.set_ylabel('Speed (px/s)', fontsize=8)
ax_spd.set_title('Speed', fontsize=9)
ax_spd.set_xlim(0, len(log[0]))
ax_spd.set_ylim(0, max(log[3]) * 1.2)
ax_spd.grid(True, alpha=0.3)

ax_str.plot(timesteps, np.degrees(log[4]), 'r-', linewidth=1)
ax_str.set_ylabel('Steering (°)', fontsize=8)
ax_str.set_title('Steering Angle', fontsize=9)
ax_str.set_xlim(0, len(log[0]))
ax_str.set_ylim(min(np.degrees(log[4])) * 1.2, max(np.degrees(log[4])) * 1.2)
ax_str.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
ax_str.grid(True, alpha=0.3)

lat_g = log[5] * 0.221 / 9.81
ax_lat.plot(timesteps, lat_g, 'm-', linewidth=1)
ax_lat.axhline(y=0.3, color='r', linestyle='--', linewidth=1.5, label='0.3g limit')
ax_lat.set_ylabel('Lat Accel (g)', fontsize=8)
ax_lat.set_title('Lateral Acceleration', fontsize=9)
ax_lat.set_xlim(0, len(log[0]))
ax_lat.set_ylim(0, 0.35)
ax_lat.legend(fontsize=7)
ax_lat.grid(True, alpha=0.3)
ax_lat.set_xlabel('Timestep', fontsize=8)

plt.tight_layout()
plt.show()