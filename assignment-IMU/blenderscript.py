import socket
import numpy as np
import bge
import mathutils
import math

import sys
sys.path.append("/home/marco/GITS/madgwick_py")

import madgwickahrs as md

#GLOBALS
UDP_IP = "0.0.0.0"
UDP_PORT = 10000

#sensor meas
acc_3d_meas = np.array([0.0,0.0,0.0])  #[x,y,z]
gyr_3d_meas = np.array([0.0,0.0,0.0])
mag_3d_meas = np.array([0.0,0.0,0.0])

#CONVERSION
M_PI = np.pi
DEG2RAD = M_PI / 180.0
RAD2DEG = 180.0 / M_PI

#attitude controller
attitude = md.MadgwickAHRS(beta=0.5, sampleperiod=Ts)
Q = np.tile([1., 0., 0., 0.], (2, 1))

def main():
    #get blender controller and info about the object
    cont = bge.logic.getCurrentController()
    obj = cont.owner

    mainloop = cont.sensors["MainLoop"]
    if not 'init' in obj:
        obj['init'] = 1
        obj['socket'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        obj['socket'].bind((UDP_IP, UDP_PORT))
        mainloop.usePosPulseMode = True

    #Main Loop
    data, addr = obj['socket'].recvfrom(2048)  # buffer size is 1024 bytes
    block = data.split(b',')
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

        #Update orientation
        attitude.update_imu(DEG2RAD * gyr_3d_meas, acc_3d_meas)
        orientation = np.array(attitude.quaternion.to_euler_angles()) * RAD2DEG

        obj.worldOrientation = np.asmatrix(orientation)