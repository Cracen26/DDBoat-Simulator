# control function from DDBOAT_controller.py adapted for the simulation
from sympy.physics.vector import cross

from boat import sawtooth
import numpy as np
from math import atan2

k_heading = 0.7 # heading control gain
# k_speed = 0.7 # speed control gain
# kd_speed = 2*k_speed # derivative gain for the speed control

def heading_regul(speed_d, th_d, th):  # compute the motor control signal
    # speed_d: desired speed (m/s)
    # th_d: desired heading (rad)
    # th: current heading (rad)
    w = k_heading * sawtooth(th_d - th)  # desired angular speed
    vd = np.array([speed_d, w])
    return vd

def follow_pose(pd,pose,dt,v1_old=0,pd_dot=np.array([0,0]),pd_ddot=np.array([0,0]),k_speed=0.7,kd_speed=1.4):
    # feedback linearization to follow a pose
    # pd : desired position [xd,yd]
    # pose : current pose [x,y,theta]
    # dt : time step
    # v1_old : previous speed
    # return : desired speed

    #optional parameters
    # pd_dot : desired speed [xd_dot,yd_dot]
    # pd_ddot : desired acceleration [xd_ddot,yd_ddot]

    ex = k_speed * (pd[0]-pose[0])+ kd_speed * (pd_dot[0]-v1_old*np.cos(pose[2])) + pd_ddot[0]
    ey = k_speed * (pd[1]-pose[1])+ kd_speed * (pd_dot[1]-v1_old*np.sin(pose[2])) + pd_ddot[1]

    v1 = v1_old + dt * (np.cos(pose[2]) * ex + np.sin(pose[2]) * ey)

    if abs(v1) < 0.01:
        # avoid singularity
        v2 = 0
    else:
        v2 = -np.sin(pose[2])/v1 * ex + np.cos(pose[2])/v1 * ey

    if v1<0:
        # the robot can move backward, turn left
        v1 = 0.
        v2 = 100
    return np.array([v1,v2])

def follow_line(a, b, m):
    # compute the desired heading to follow the line [ab)
    # a: beginning of the line
    # b: end of the line
    # m: current position
    r = 4  # gain
    u = (b - a) / np.linalg.norm(b - a)  # unit vector of the line
    e = np.linalg.det(np.vstack([u, m - a]))  # distance error with the line
    phi = np.arctan2(b[1] - a[1], b[0] - a[0])  # orientation of the line
    # print("phi",phi)
    # print("correct",np.arctan(e/r))
    ang = sawtooth(phi - np.arctan(e / r))  # desired heading
    # print("desired ANG (deg)", ang * 180 / np.pi)
    return ang

def lissajou_trajectory(c, R, T, t, i, N):
    # compute the Lissajou trajectory and its derivatives
    # c : center of the curve
    # R : amplitude of the curve
    # T : period of the trajectory
    # t : current time since the beginning of the mission
    # i : id of the robot
    # N : total number of robots
    phi1, phi2 = t * np.pi * 2 / T + 2 * i * np.pi / N, 2 * (t * np.pi * 2 / T + 2 * i * np.pi / N)
    pd = c + R * np.array([np.cos(phi1),
                           np.sin(phi2)])
    pd_dot = R * np.array([-np.pi * 2 / T * np.sin(phi1),
                           np.pi * 4 / T * np.cos(phi2)])
    pd_ddot = R * np.array([-(np.pi * 2 / T) ** 2 * np.cos(phi1),
                            -(np.pi * 4 / T) ** 2 * np.sin(phi2)])
    return pd, pd_dot, pd_ddot