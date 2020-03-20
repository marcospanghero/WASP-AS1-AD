import socket
import struct
import binascii
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import ahrs


def to_euler_angles(self):
    pitch = np.arcsin(2 * self[1] * self[2] + 2 * self[0] * self[3])
    if np.abs(self[1] * self[2] + self[3] * self[0] - 0.5) < 1e-8:
        roll = 0
        yaw = 2 * np.arctan2(self[1], self[0])
    elif np.abs(self[1] * self[2] + self[3] * self[0] + 0.5) < 1e-8:
        roll = -2 * np.arctan2(self[1], self[0])
        yaw = 0
    else:
        roll = np.arctan2(2 * self[0] * self[1] - 2 * self[2] * self[3], 1 - 2 * self[1] ** 2 - 2 * self[3] ** 2)
        yaw = np.arctan2(2 * self[0] * self[2] - 2 * self[1] * self[3], 1 - 2 * self[2] ** 2 - 2 * self[3] ** 2)
    return roll, pitch, yaw

UDP_IP = "0.0.0.0"
UDP_PORT = 10000
t = 0

AccX_Variance = 0.007
AccY_Variance = 0.007
AccZ_Variance = 0.007

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
gyr_3d_meas = np.array([0.0,0.0,0.0])
mag_3d_meas = np.array([0.0,0.0,0.0])
acc_3d_ts = np.array([0.0,0.0])            #[ts]


#acc_3d_ts_test = np.linspace(0, 8*np.pi, num=n_timesteps)
#acc_3d_meas[0,:] =  0.5 * np.sin(acc_3d_ts_test) + np.random.normal(0,AccX_Variance,n_timesteps)
#acc_3d_meas[1,:] = np.sin(3 * acc_3d_ts_test) + np.ones((1, n_timesteps)) * np.random.normal(0,AccX_Variance,n_timesteps)
#acc_3d_meas[2,:] = 2 * np.sin(5*acc_3d_ts_test) + np.ones((1, n_timesteps)) * np.random.normal(0,AccX_Variance,n_timesteps)


madgwick = ahrs.filters.Madgwick(beta=0.1, frequency=100.0)
Q = np.tile([1., 0., 0., 0.], (n_timesteps, 1))
d2r = ahrs.common.DEG2RAD

print(acc_3d_meas.size)
while t<n_timesteps:
    data, addr = sock.recvfrom(2048) # buffer size is 1024 bytes
    #print("received message:", data)
    block = data.split(b',')
    #print(len(block))
    if len(block) == 13:
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
        if t == 0:
            acc_3d_ts[0] = float(block[0])
            #acc_3d_ts[0] = acc_3d_ts_test[t]
            filtered_state_means_x[t] = X0
            filtered_state_means_y[t] = Y0
            filtered_state_means_z[t] = Z0
            filtered_state_covariances_x[t] = P0
            filtered_state_covariances_y[t] = P0
            filtered_state_covariances_z[t] = P0
            Q[t] = [1., 0., 0., 0.]
        else:
            acc_3d_ts[1] = float(block[0])
            #acc_3d_ts[1] = acc_3d_ts_test[t]
            dt =  acc_3d_ts[1] - acc_3d_ts[0]
            print("Fs = {:.5f}".format(1/dt))
            Q[t] = madgwick.updateMARG(Q[t - 1], d2r * gyr_3d_meas, acc_3d_meas, mag_3d_meas)
            #Compute gravity vector
            Qw, Qx, Qy, Qz = Q[t]
            Xgrav = 2 *((Qw * Qx) + (Qy * Qz)) * 10
            Ygrav = 2 * ((Qx * Qz) + (Qw * Qy)) * 10
            Zgrav = ((Qw * Qw) - (Qx * Qx) - (Qy*Qy) + (Qz * Qz)) * 10
            print("Q: ",Q[t])
            print("Gravity: ", Xgrav, Ygrav, Zgrav)
            print("corrected Gravity: ",  acc_3d_meas[0] - Xgrav, acc_3d_meas[1] - Ygrav, acc_3d_meas[2] - Zgrav)
            #print(dt)
            filtered_state_means_x[t], filtered_state_covariances_x[t] = (
                kf.filter_update(
                    filtered_state_means_x[t - 1],
                    filtered_state_covariances_x[t - 1],
                    acc_3d_meas[0] - Xgrav
                )
            )
            filtered_state_means_y[t], filtered_state_covariances_y[t] = (
                kf.filter_update(
                    filtered_state_means_y[t - 1],
                    filtered_state_covariances_y[t - 1],
                    acc_3d_meas[1] - Ygrav
                )
            )
            filtered_state_means_z[t], filtered_state_covariances_z[t] = (
                kf.filter_update(
                    filtered_state_means_z[t - 1],
                    filtered_state_covariances_z[t - 1],
                    acc_3d_meas[2] - Zgrav
                )
            )
        acc_3d_ts[0] = acc_3d_ts[1]
        #print((filtered_state_means_x[t, 0], filtered_state_means_y[t, 0], filtered_state_means_z[t, 0]))
        time[t] = t
        t = t + 1

ahrs.utils.plot_quaternions(Q)

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