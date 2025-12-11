# sim_2 simulating basic control of a boat model
# red robot follows a speed reference
# blue robot follows a heading reference
# green robot follows a position reference
# black robot follows a line
import numpy as np

from boat import *
import time
from controller import *

# initial state
x_red = np.array([-5., 5, 0])
x_blue = np.array([-5., 0, 0])
x_green = np.array([-5., -5, 0])
x_black = np.array([10., -10, np.pi/2])

# references
vd_red = np.array([1, -0.5])  # (speed, angular speed) for the red boat
thd_blue = np.pi/3 # heading reference for the blue boat
speed_blue = 1.0 # desired speed for the blue boat

# speed_green = 0.5 # desired speed for the green boat
pd_green = np.array([0.5, -0.7]) # desired position for the green boat

speed_black = 2. # desired speed for the black boat
line_a = np.array([10, -10]) # beginning of the line
line_b = np.array([-10, 10]) # end of the line

# create the boat
boat_red = Boat(x_red,traj_memory_=-1)
boat_blue = Boat(x_blue,traj_memory_=-1)
boat_green = Boat(x_green ,traj_memory_=-1)
boat_black = Boat(x_black,traj_memory_=-1)

# simulation parameters
sim = SimuDisplay()
dt = 0.05  # time step
T = 100.0  # simulation duration
N = int(T/dt)  # number of iterations

# simulation
t = 0
vd_green = np.array([0,0])
while t < T:
    t1 = time.time()
    sim.clear()

    # control input
    u_red = boat_red.convert_motor_control_signal(vd_red)
    u_blue = boat_blue.convert_motor_control_signal(heading_regul(1, thd_blue, boat_blue.th()))

    vd_green = follow_pose(pd_green,boat_green.x,dt,v1_old=vd_green[0])
    u_green = boat_green.convert_motor_control_signal(vd_green)

    thd_black = follow_line(line_a, line_b, boat_black.p())
    u_black = boat_black.convert_motor_control_signal(heading_regul(speed_black, thd_black, boat_black.th()))

    # update the state
    v_red = boat_red.motor_update(dt, u_red)
    v_blue = boat_blue.motor_update(dt, u_blue)
    v_green = boat_green.motor_update(dt, u_green)
    v_black = boat_black.motor_update(dt, u_black)

    # display
    draw_tank2(sim.ax,boat_red.x,"r")
    draw_traj(sim.ax,boat_red.traj,"r")

    draw_tank2(sim.ax,boat_blue.x,"b")
    draw_traj(sim.ax,boat_blue.traj,"b")

    draw_tank2(sim.ax,boat_green.x,"g")
    draw_traj(sim.ax,boat_green.traj,"g")
    sim.ax.plot(pd_green[0], pd_green[1], 'go', markersize=10)

    draw_tank2(sim.ax, boat_black.x, "k")
    draw_traj(sim.ax, boat_black.traj, "k")
    sim.ax.plot([line_a[0], line_b[0]], [line_a[1], line_b[1]], 'k-')

    # print time on the figure
    sim.ax.text(0.1, 0.9, 'Time: {:.2f}'.format(t), transform=sim.ax.transAxes)

    # update time
    t += dt
    plt.pause(0.01)

print("done")
plt.show()