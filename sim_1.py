# basic simulation of the boat model
# constant input

from boat import *
import time

# initial state
x0 = np.array([0, 5, 0])

# boat parameters
k_prop = 0.01  # (m/s)/(rad/s)
k_rot = 0.006  # (rad/s)/(rad/s)
sat_motor = 200.0  # max motor rotation speed (rad/s)

# create the boat
boat = Boat(x0, k_prop, k_rot, sat_motor,traj_memory_=-1)
boat2 = Boat(x0, 2*k_prop, k_rot, sat_motor,traj_memory_=-1)
boat3 = Boat(x0, k_prop, 2*k_rot, sat_motor,traj_memory_=-1)

# simulation parameters
sim = SimuDisplay(grid=True)
dt = 0.05  # time step
T = 20.0  # simulation duration
N = int(T/dt)  # number of iterations

# simulation
t = 0
while t < T:
    t1 = time.time()
    sim.clear()

    # control input
    u = np.array([200,0])  # (left motor rotation speed, right motor rotation speed)

    # update the state
    boat.motor_update(dt, u)
    boat2.motor_update(dt, u)
    boat3.motor_update(dt, u)

    # display
    draw_tank2(sim.ax,boat.x,"r")
    draw_traj(sim.ax,boat.traj,"r")

    draw_tank2(sim.ax,boat2.x,"b")
    draw_traj(sim.ax,boat2.traj,"b")

    draw_tank2(sim.ax,boat3.x,"g")
    draw_traj(sim.ax,boat3.traj,"g")

    # print time on the figure
    sim.ax.text(0.1, 0.9, 'Time: {:.2f}'.format(t), transform=sim.ax.transAxes)

    # update time
    t += dt
    plt.pause(0.01)


print("done")
plt.show()


