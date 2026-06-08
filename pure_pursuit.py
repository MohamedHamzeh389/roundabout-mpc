import numpy as np
import matplotlib.pyplot as plt

log = np.load('mpc_log_new.npy')
print(f"Shape: {log.shape}")
print(f"X range: {log[0].min():.1f} to {log[0].max():.1f}")
print(f"Y range: {log[1].min():.1f} to {log[1].max():.1f}")
print(f"First point: ({log[0][0]:.1f}, {log[1][0]:.1f})")
print(f"Last point: ({log[0][-1]:.1f}, {log[1][-1]:.1f})")

plt.plot(log[0], log[1], 'b-')
plt.axis('equal')
plt.title('Current mpc_log trajectory')
plt.show()