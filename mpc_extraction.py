import numpy as np
import cv2
from scipy.interpolate import splprep, splev

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
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

x_points = [x for x, y in points]
y_points = [y for x, y in points]
tck, u = splprep([x_points, y_points], s=0, per=True)
u_range = np.linspace(0, 1, 500)

x_spline, y_spline = splev(u_range, tck)
for i in range(500):
    cv2.circle(img, (int(x_spline[i]), int(y_spline[i])), 2, (0, 255, 255), -1)
cv2.imshow('Reference_Path', img)
waypoints = np.array([x_spline, y_spline]).T
np.save('reference_path.npy', waypoints)
print(f"Saved {len(waypoints)} waypoints")
cv2.waitKey(0)
cv2.destroyAllWindows()



