import cv2
import numpy as np

img =cv2.imread('roundabout_satellite_cleanup.png')
image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


lower = np.array([79, 0, 92])
upper = np.array([102, 40, 196])
mask = cv2.inRange(image_hsv, lower, upper)

kernel = np.ones((15, 15), np.uint8)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)

cv2.imshow("Opening", opening)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.imwrite('road_mask.png', opening)
        
cv2.destroyAllWindows()
