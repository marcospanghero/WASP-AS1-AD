import socket
import struct
import binascii
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

UDP_IP = "0.0.0.0"
UDP_PORT = 10000
t = 0

AccX_Variance = 0.0007
AccY_Variance = 0.0007
AccZ_Variance = 0.0007

# time step
dt = 0.01

# transition_matrix
F = [[1, dt, 0.5*dt**2],
     [0,  1,       dt],
     [0,  0,        1]]

# observation_matrix
H = [0, 0, 1]

# transition_covariance
Q = [[0.2,    0,      0],
     [  0,  0.1,      0],
     [  0,    0,  10e-4]]

# observation_covariance
R = AccX_Variance

# initial_state_mean
X0 = [0,
      0,
      0]
Y0 = [0,
      0,
      0]
Z0 = [0,
      0,
      0]

# initial_state_covariance
P0 = [[  0,    0,               0],
      [  0,    0,               0],
      [  0,    0,   AccX_Variance]]

n_timesteps = 1000
n_dim_state = 3
time = np.zeros((n_timesteps, 1))
filtered_state_means_x = np.zeros((n_timesteps, n_dim_state))
filtered_state_means_y = np.zeros((n_timesteps, n_dim_state))
filtered_state_means_z = np.zeros((n_timesteps, n_dim_state))

filtered_state_covariances_x = np.zeros((n_timesteps, n_dim_state, n_dim_state))
filtered_state_covariances_y = np.zeros((n_timesteps, n_dim_state, n_dim_state))
filtered_state_covariances_z = np.zeros((n_timesteps, n_dim_state, n_dim_state))


kf = KalmanFilter(transition_matrices = F,
                  observation_matrices = H,
                  transition_covariance = Q,
                  observation_covariance = R,
                  initial_state_mean = X0,
                  initial_state_covariance = P0)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

acc_3d_meas = np.array([0.0,0.0,0.0])  #[x,y,z]
acc_3d_ts = np.array([0.0,0.0])            #[ts]

print(acc_3d_meas.size)
while t<n_timesteps:
    data, addr = sock.recvfrom(2048) # buffer size is 1024 bytes
    #print("received message:", data)
    block = data.split(b',')
    acc_3d_meas[0] = float(block[2])
    acc_3d_meas[1] = float(block[3])
    acc_3d_meas[2] = float(block[4])
    #print(acc_3d_meas)
    if t == 0:
        acc_3d_ts[0] = float(block[0])
        filtered_state_means_x[t] = X0
        filtered_state_means_y[t] = Y0
        filtered_state_means_z[t] = Z0
        filtered_state_covariances_x[t] = P0
        filtered_state_covariances_y[t] = P0
        filtered_state_covariances_z[t] = P0
    else:
        acc_3d_ts[1] = float(block[0])
        dt =  acc_3d_ts[1] - acc_3d_ts[0]
        print(dt)
        filtered_state_means_x[t], filtered_state_covariances_x[t] = (
            kf.filter_update(
                filtered_state_means_x[t - 1],
                filtered_state_covariances_x[t - 1],
                acc_3d_meas[0]
            )
        )
        filtered_state_means_y[t], filtered_state_covariances_y[t] = (
            kf.filter_update(
                filtered_state_means_y[t - 1],
                filtered_state_covariances_y[t - 1],
                acc_3d_meas[1]
            )
        )
        filtered_state_means_z[t], filtered_state_covariances_z[t] = (
            kf.filter_update(
                filtered_state_means_z[t - 1],
                filtered_state_covariances_z[t - 1],
                acc_3d_meas[2]
            )
        )
    acc_3d_ts[0] = acc_3d_ts[1]
    print((filtered_state_means_x[t, 0], filtered_state_means_y[t, 0], filtered_state_means_z[t, 0]))
    time[t] = t
    t = t + 1


f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(time, filtered_state_means_x[:, 2], linewidth = '0.3', label="AccX")
axarr[0].plot(time, filtered_state_means_y[:, 2], linewidth = '0.3', label="AccX")
axarr[0].plot(time, filtered_state_means_z[:, 2], linewidth = '0.3', label="AccX")
axarr[0].set_title('Acceleration XYZ')
axarr[0].grid()
axarr[0].legend()

axarr[1].plot(time, filtered_state_means_x[:, 1], linewidth = '0.3', label="VelX")
axarr[1].plot(time, filtered_state_means_y[:, 1], linewidth = '0.3', label="VelX")
axarr[1].plot(time, filtered_state_means_z[:, 1], linewidth = '0.3', label="VelX")
axarr[1].set_title('Velocity XYZ')
axarr[1].grid()
axarr[1].legend()

axarr[2].plot(time, filtered_state_means_x[:, 0], linewidth = '0.3', label="PosX")
axarr[2].plot(time, filtered_state_means_y[:, 0], linewidth = '0.3', label="PosY")
axarr[2].plot(time, filtered_state_means_z[:, 0], linewidth = '0.3', label="PosZ")
axarr[2].set_title('Position XYZ')
axarr[2].grid()
axarr[2].legend()

plt.show()