import socket
import numpy as np
import scipy.constants as sciconst
from pykalman import KalmanFilter
import matplotlib.pyplot as plt



import sys
sys.path.append("/home/marco/GITS/madgwick_py")

import madgwickahrs as md



UDP_IP = "0.0.0.0"
UDP_PORT = 10005
t = 1
M_PI = np.pi
DEG2RAD = M_PI / 180.0
RAD2DEG = 180.0 / M_PI
n_timesteps = 100
Ts = 1/100.0


#MEASUREMENTS AND COMPENSATIONS

acc_3d_meas = np.array([0.0,0.0,0.0])  #[x,y,z]
gyr_3d_meas = np.array([0.0,0.0,0.0])
mag_3d_meas = np.array([0.0,0.0,0.0])
acc_3d_ts = np.zeros((1,n_timesteps))            #[ts]
compensate_g =  np.array([0.0,0.0,0.0])
orientation = np.tile([0., 0., 0.], (n_timesteps, 1))

# Kalman matrices
AccX_Variance = 0.07
AccY_Variance = 0.07
AccZ_Variance = 0.07

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
P0 = [[AccX_Variance,    0,               0],
      [0,    AccY_Variance,               0],
      [0,                0,   AccZ_Variance]]

##KALMAN VECTORS
n_dim_state = 3
time = np.zeros((n_timesteps, 1))
filtered_state_means_x = np.zeros((n_timesteps, n_dim_state))
filtered_state_means_y = np.zeros((n_timesteps, n_dim_state))
filtered_state_means_z = np.zeros((n_timesteps, n_dim_state))

filtered_state_covariances_x = np.zeros((n_timesteps, n_dim_state, n_dim_state))
filtered_state_covariances_y = np.zeros((n_timesteps, n_dim_state, n_dim_state))
filtered_state_covariances_z = np.zeros((n_timesteps, n_dim_state, n_dim_state))

#Initialize filter and socket for data recv
attitude = md.MadgwickAHRS(beta=0.5, sampleperiod=Ts)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))
kf = KalmanFilter(transition_matrices = F,
                  observation_matrices = H,
                  transition_covariance = Q,
                  observation_covariance = R,
                  initial_state_mean = X0,
                  initial_state_covariance = P0)


while t<n_timesteps:
    data, addr = sock.recvfrom(2048) # buffer size is 1024 bytes
    #print("received message:", data)
    block = data.split(b',')
    #print(len(block))
    if len(block) == 13:

        acc_3d_ts[0,t] = float(block[0])
        acc_3d_meas[0] = float(block[2])
        acc_3d_meas[1] = float(block[3])
        acc_3d_meas[2] = float(block[4])
        gyr_3d_meas[0] = float(block[6])
        gyr_3d_meas[1] = float(block[7])
        gyr_3d_meas[2] = float(block[8])
        mag_3d_meas[0] = float(block[10])
        mag_3d_meas[1] = float(block[11])
        mag_3d_meas[2] = float(block[12])
        #print(acc_3d_meas)
        attitude.update_imu(gyr_3d_meas * DEG2RAD, acc_3d_meas)
        Qw, Qx, Qy, Qz = np.array(attitude.quaternion)
        compensate_g[0] = 2 * ((Qw * Qx) + (Qy * Qz)) * sciconst.g
        compensate_g[1] = 2 * ((Qx * Qz) + (Qw * Qy)) * sciconst.g
        compensate_g[2] = ((Qw * Qw) - (Qx * Qx) - (Qy * Qy) + (Qz * Qz)) * sciconst.g
        orientation = np.array(attitude.quaternion.to_euler_angles()) * RAD2DEG
        #print(orientation ,'-',compensate_g)
        print(acc_3d_meas, acc_3d_meas - compensate_g)
        ##compensate g
        acc_3d_meas = acc_3d_meas - compensate_g
        if t == 0:
            filtered_state_means_x[t] = X0
            filtered_state_means_y[t] = Y0
            filtered_state_means_z[t] = Z0
            filtered_state_covariances_x[t] = P0
            filtered_state_covariances_y[t] = P0
            filtered_state_covariances_z[t] = P0
        else:
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
        time[t] = t
        t = t + 1

f, axarr = plt.subplots(3, sharex=True)

axarr[0].plot(time, filtered_state_means_x[:, 2], linewidth = '0.3', label="AccX")
axarr[0].plot(time, filtered_state_means_y[:, 2], linewidth = '0.3', label="AccY")
axarr[0].plot(time, filtered_state_means_z[:, 2], linewidth = '0.3', label="AccZ")
axarr[0].set_title('Acceleration XYZ')
axarr[0].grid()
axarr[0].legend()

axarr[1].plot(time, filtered_state_means_x[:, 1], linewidth = '0.3', label="VelX")
axarr[1].plot(time, filtered_state_means_y[:, 1], linewidth = '0.3', label="VelY")
axarr[1].plot(time, filtered_state_means_z[:, 1], linewidth = '0.3', label="VelZ")
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