# plot the attainable speeds

import numpy as np
import matplotlib.pyplot as plt

umin = 30
umax = 200
X = np.linspace(umin,umax,30)
Y = np.linspace(umin,umax,30)

k1 = 0.01
k2 = 0.006
M = np.array([[k1,k1],[-k2,k2]])

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim([0,1.1*umax*2*k1])
ax.set_ylim([-2*k2*umax,2*k2*umax])
ax.set_xlabel('speed v1 (m/s)')
ax.set_ylabel('angular speed v2 (rad/s)')
ax.set_title('possible speeds')
for x in X:
    for y in Y:
        u = np.array([x,y])
        v = M@u
        # print(v)

        ax.plot(v[0],v[1],'k',marker='o',markersize=1)

# we need to add the case when one motor is off
for x in X:
    u = np.array([x,0])
    v = M@u
    ax.plot(v[0],v[1],'r',marker='o',markersize=1)

for y in Y:
    u = np.array([0,y])
    v = M@u
    ax.plot(v[0],v[1],'b',marker='o',markersize=1)

# # now let us plot what happen when we saturate the control signal
# from boat import Boat
# boat_test = Boat(np.array([0,0,0]),k1,k2,umin,umax)
# v1max = k1*2*umax
# v1min = k1*umin
# v2max = k2*umax
# X = np.linspace(0,v1max,100)
# Y = np.linspace(-1.2*v2max,1.2*v2max,100)
#
#
#
# for x in X:
#     for y in Y:
#         vd = np.array([x,y])
#         v = boat_test.bound_desired_speed(vd)
#         print(v)
#         vect = v-vd
#         if(np.linalg.norm(vect)>0.01):
#             if(x<v1min and y<0):
#                 # shift the head of the arrow such that the tip of the arrow is at the end of the arrow
#                 ax.arrow(x,y,vect[0],vect[1],head_width=0.03, head_length=0.05, fc='r', ec='r')
#             elif(x>v1min and y>0):
#                 ax.arrow(x, y, vect[0], vect[1], head_width=0.03, head_length=0.05, fc='g', ec='g')
#
#
plt.show()