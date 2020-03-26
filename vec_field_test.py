import matplotlib.pyplot as plt
import numpy as np


n_steps = 15
ub = 3*2*np.pi
step = ub/n_steps
X, Y = np.meshgrid(np.arange(-ub, ub, step), np.arange(-ub, ub, step))


R = (X**2 + Y**2)**0.5
theta = np.arctan2(Y, X)

R_max = np.max(R)
R = np.clip(R, 0.1, R_max)

print(R_max)

theta_rot = 1.0*np.cos(2*np.pi*(R/R_max))
#theta_rot = 0.1*np.pi

jitter = 0.0

x_whorl = 10
y_whorl = 10


whorl_rel_x = X - x_whorl
whorl_rel_y = Y - y_whorl

whorl_rel_r = 0.3*(whorl_rel_x**2 + whorl_rel_y**2).clip(0.1, R_max**2)
theta_whorl = 0.5*np.pi
u_whorl = whorl_rel_x*np.cos(theta_whorl)/whorl_rel_r + whorl_rel_y*np.sin(theta_whorl)/whorl_rel_r
v_whorl = -whorl_rel_x*np.sin(theta_whorl)/whorl_rel_r + whorl_rel_y*np.cos(theta_whorl)/whorl_rel_r

w = 0

U = X*np.cos(theta_rot)/R + Y*np.sin(theta_rot)/R + jitter*np.random.randn(*X.shape) + w*u_whorl
V = -X*np.sin(theta_rot)/R + Y*np.cos(theta_rot)/R + jitter*np.random.randn(*X.shape) + w*v_whorl

print(np.min(U), np.max(U))
print(np.min(V), np.max(V))


plt.figure(figsize=(10,8))
plt.quiver(X, Y, U, V, units='width')
plt.axis('off')
plt.tight_layout()
plt.savefig('vec_field.png')
plt.show()
