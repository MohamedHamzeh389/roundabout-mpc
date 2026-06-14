import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# --- SPLINE SECTION (already run, waypoints saved) ---
# Uncomment this section only if you need to re-click the path for roundabout



img = cv2.imread('road_mask.png')
cv2.namedWindow('Reference_Path', cv2.WINDOW_NORMAL)
points = []
cv2.imshow('Reference_Path', img)
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        points.append((x, y))
        
cv2.setMouseCallback('Reference_Path', draw_circle)
while True:
    cv2.imshow('Reference_Path', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):  #Click q to end
        break
cv2.destroyAllWindows()

x_points = [x for x, y in points]
y_points = [y for x, y in points]
tck, u = splprep([x_points, y_points], s=0, per=False)
u_range = np.linspace(0, 1, 500)

x_spline, y_spline = splev(u_range, tck)
for i in range(500):
    cv2.circle(img, (int(x_spline[i]), int(y_spline[i])), 2, (0, 255, 255), -1)
cv2.imshow('Reference_Path', img)
waypoints = np.array([x_spline, y_spline]).T
np.save('Reference_path4.npy', waypoints)
print(f"Saved {len(waypoints)} waypoints")
cv2.waitKey(0)
cv2.destroyAllWindows()


#  -----IF CIRCULAR PATH------- 

# existing = np.load('circle_path.npy')
# x = existing[:, 0]
# y = existing[:, 1]

# def circle_residuals(params, x_pts, y_pts):
#     cx, cy, r = params
#     distances = np.sqrt((x_pts - cx)**2 + (y_pts - cy)**2)
#     return distances - r

# cx0 = np.mean(x)
# cy0 = np.mean(y)
# r0  = np.mean(np.sqrt((x - cx0)**2 + (y - cy0)**2))

# print(f"Initial guess - centre: ({cx0:.1f}, {cy0:.1f}), radius: {r0:.1f}")

# result = least_squares(circle_residuals, [cx0, cy0, r0], args=(x, y))
# cx, cy, r = result.x
# print(f"Fitted circle - centre: ({cx:.1f}, {cy:.1f}), radius: {r:.1f} pixels")


# theta_vals = np.linspace(0, 2 * np.pi, 500, endpoint=False)
# x_circle = cx + r * np.cos(theta_vals)
# y_circle = cy + r * np.sin(theta_vals)

# mask = cv2.imread('road_mask.png', cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(8, 8))
# plt.imshow(mask, cmap='gray')
# plt.plot(x_circle, y_circle, 'y-', linewidth=2, label='Fitted circle')
# plt.plot(cx, cy, 'r+', markersize=15, label='Centre')
# plt.legend()
# plt.title(f'Fitted circle - centre ({cx:.0f}, {cy:.0f}), r={r:.0f}px')
# plt.show()

# new_waypoints = np.array([x_circle, y_circle]).T
# np.save('circle_path.npy', new_waypoints)
# print(f"Saved clean circular path - {len(new_waypoints)} waypoints")