import matplotlib.pyplot as plt
import numpy as np

Ref_point = np.load('reference_path.npy')
x = Ref_point[:, 0]
y = Ref_point[:, 1]

log = np.load('mpc_log.npy')
plt.figure(figsize=(8,8))
plt.plot(x, y, 'y--', label='Reference path')
plt.plot(log[0], log[1], 'g-', label='MPC trajectory')
plt.axis('equal')
plt.legend()
plt.title('MPC Trajectory vs Reference Path')
plt.show()