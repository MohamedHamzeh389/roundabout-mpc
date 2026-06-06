import cv2
import numpy as np

log = np.load('mpc_log.npy')
Ref_point = np.load('reference_path.npy')
x = Ref_point[:, 0]
y = Ref_point[:, 1]



base_img = cv2.imread('roundabout_satellite_cleanup.png')
h, w = base_img.shape[:2]

panel_w = 350
canvas_w = w + panel_w

for i in range(len(x) - 1):
    pt1 = (int(x[i]), int(y[i]))
    pt2 = (int(x[i+1]), int(y[i+1]))
    cv2.line(base_img, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

trajectory = np.array([[int(log[0][i]), int(log[1][i])]
                        for i in range(len(log[0]))], np.int32)

def get_car_corners(cx, cy, theta, width=16, length=32):
    corners = np.array([
        [-length/2, -width/2],
        [ length/2, -width/2],
        [ length/2,  width/2],
        [-length/2,  width/2]
    ], dtype=np.float32)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    rotated = (rot @ corners.T).T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated.astype(np.int32)

def draw_chart(img, data, frame_idx, y_top, height, color, label, max_val, panel_x):
    chart_x = panel_x + 10
    chart_w = panel_w - 20
    cv2.rectangle(img, (chart_x, y_top), (chart_x + chart_w, y_top + height), (50, 50, 50), -1)
    cv2.rectangle(img, (chart_x, y_top), (chart_x + chart_w, y_top + height), (80, 80, 80), 1)
    cv2.putText(img, label, (chart_x, y_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    if frame_idx > 1:
        n = min(frame_idx, 200)
        subset = data[frame_idx - n:frame_idx]
        pts = []
        for i, val in enumerate(subset):
            px = chart_x + int(i * chart_w / n)
            py = y_top + height - int(np.clip(val / max_val, 0, 1) * height)
            pts.append([px, py])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)

frame_idx = 0
step = 2
total_frames = len(log[0])

while True:
    canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :w] = base_img.copy()

    if frame_idx > 1:
        cv2.polylines(canvas, [trajectory[:frame_idx]], False, (0, 200, 0), 2, cv2.LINE_AA)

    car_x = log[0][frame_idx]
    car_y = log[1][frame_idx]
    next_f = min(frame_idx + 1, total_frames - 1)
    car_theta = np.arctan2(log[1][next_f] - log[1][frame_idx],
                           log[0][next_f] - log[0][frame_idx])

    corners = get_car_corners(car_x, car_y, car_theta)
    cv2.fillPoly(canvas, [corners], (0, 255, 0))
    cv2.polylines(canvas, [corners], True, (255, 255, 255), 1)

    cv2.line(canvas, (int(car_x), int(car_y)),
             (int(car_x + 14 * np.cos(car_theta)),
              int(car_y + 14 * np.sin(car_theta))),
             (255, 255, 255), 2, cv2.LINE_AA)

    panel_x = w
    cv2.rectangle(canvas, (panel_x, 0), (canvas_w, h), (30, 30, 30), -1)

    cv2.putText(canvas, 'MPC Telemetry', (panel_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f'Step: {frame_idx}/{total_frames}',
                (panel_x + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    draw_chart(canvas, log[2], frame_idx, 80, 100,
               (0, 255, 100), 'CTE (px)', 2.0, panel_x)
    draw_chart(canvas, log[3], frame_idx, 210, 100,
               (255, 150, 0), 'Speed (px/s)', 10.0, panel_x)
    draw_chart(canvas, np.abs(log[4]), frame_idx, 340, 100,
               (0, 100, 255), 'Steering (rad)', 0.6, panel_x)
    draw_chart(canvas, log[5] * 0.221 / 9.81, frame_idx, 470, 100,
               (200, 0, 255), 'Lat Accel (g)', 0.05, panel_x)

    cv2.putText(canvas, '2.0', (panel_x + 12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.putText(canvas, '0', (panel_x + 12, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.putText(canvas, '10', (panel_x + 12, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.putText(canvas, '0', (panel_x + 12, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.putText(canvas, '0.6', (panel_x + 12, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.putText(canvas, '0', (panel_x + 12, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.putText(canvas, '0.05g', (panel_x + 12, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    cv2.putText(canvas, '0', (panel_x + 12, 568), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    cv2.imshow('MPC Roundabout Navigation', canvas)

    if cv2.waitKey(8) & 0xFF == ord('q'):
        break

    frame_idx = (frame_idx + step) % total_frames

cv2.destroyAllWindows()