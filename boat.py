# define the boat class as well as its display

import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
import json

mqtt_client = mqtt.Client()
mqtt_client.connect("localhost", 1883, 60)  
mqtt_client.loop_start()

def sawtooth(x):
    # return (x + pi) % (2 * pi) - pi  # or equivalently   2*arctan(tan(x/2))
    return 2*np.arctan(np.tan(x/2))

class Boat:  # Debins car model
    def __init__(self,
                 id,
                 x0= np.array([0,0,0]),
                 k_1=0.01,
                 k_2=0.006,
                 u_min = 30.,
                 u_max=200.,
                 traj_memory_=None,
                 motor_limitation=False):
        self.id = id
        self.state = 'Idle'
        self.topic = None
        self.x = x0  # state vector [px,py,th]
        # px : the x-axis position
        # py : the y-axis position
        # th : the heading

        # dynamical parameters
        # k_1 : proportionality constant between the motor rotation speed sum and the boat speed
        # k_2 : proportionality constant between the motor rotation speed difference and the boat angular speed
        
        self.k_1 = k_1 # (m/s)/(rad/s)
        self.k_2 = k_2 # (rad/s)/(rad/s)

        self.r_min = self.k_1 / self.k_2  # minimum turning radius

        self.motor_limitation = motor_limitation # if True, the motor input "u" is bounded between u_min and u_max when not null
        self.u_min = u_min # min motor rotation speed (rad/s)
        self.u_max = u_max # max motor rotation speed (rad/s)

        # relevant bounds
        self.v1max = 2*self.u_max*self.k_1 # max speed (with two motors at full speed)
        self.v1min = self.u_min*self.k_1 # min speed with only one motor
        self.v2max = self.u_max*self.k_2 # max angular speed
        self.v2min = self.u_min*self.k_2 # min angular speed with only one motor

        self.B = np.array([[k_1, k_1], [-k_2, k_2]])  # Matrix conversion from motor to speed matrix
        self.B_inv = np.linalg.inv(self.B)

        # Trajectory of the robot (for display)
        # traj_memory : number of points to keep in memory (if None, no memory / if -1, infinite memory)
        self.traj_memory = traj_memory_ # number of points to keep in memory
        if self.traj_memory:
            self.traj = np.array([x0])


    def __str__(self):
        return (
            f"DDBoat (\n"
            f"  id: {self.id}\n"
            f"  x position: {self.x[0]}\n"
            f"  y position: {self.x[1]}\n"
            f"  th heading: {self.x[2]}\n"
            f"  linear speed: {self.k_1}\n"
            f"  angular speed: {self.k_2}\n"
            f"  state: {self.state}\n"
            f"  topic: {self.topic}\n"
            f")"
        )
    

    def to_dict(self):
        return {"id": self.id, "x": self.x[0], "y": self.x[1], "heading": self.x[2], "k_1": self.k_1, "k_2": self.k_2, "status": self.status}
    
    def kinematic_update(self, dt, v,q=np.zeros(2)):
        # discrete time update of the state with euler method
        # dt : time step
        # v : (speed, angular speed)
        # q is an optional drift (caused by wind or current)

        # kinematic model:
        # x_dot = [ px_dot, py_dot, th_dot]
        # px_dot = speed * cos(th) + q[0]
        # py_dot = speed * sin(th) + q[1]
        # th_dot = angular speed

        x_dot = np.array([v[0]*np.cos(self.x[2])+q[0],v[0]*np.sin(self.x[2])+q[1],v[1]])
        self.x = self.x + dt*x_dot
        self.x[2] = sawtooth(self.x[2]) # keep the angle between -pi and pi

        # update the trajectory
        if self.traj_memory:
            self.traj = np.vstack((self.traj,self.x))
            if self.traj.shape[0] > self.traj_memory > 0:
                self.traj = np.delete(self.traj,0,0)


    def motor_update(self,dt,u, q=np.zeros(2)):
        # discrete time update of the state with euler method
        # dt : time step
        # u : (left motor rotation speed, right motor rotation speed)
        # q is an optional drift (caused by wind or current)

        # v = B @ u, but u may be limited
        if self.motor_limitation:
            v = self.B @ self.bound_motor(u)
        else:
            v = self.B @ u
        self.kinematic_update(dt, v, q)
        return v # return the actual speed vector

    def px(self): return self.x[0]
    def py(self): return self.x[1]
    def th(self): return self.x[2]

    def p(self):
        return self.x[0:2]

    def bound_motor(self, u):
        # bound the motor input (rotation speed of the motors)
        # u can be null or between u_min and u_max
        # u = (left motor rotation speed, right motor rotation speed)

        # bound the motor speed between 0 and u_max
        res = np.zeros(2)
        res[0] = max(0,min(u[0],self.u_max))
        res[1] = max(0,min(u[1],self.u_max))

        # stop the motor if the speed is too low
        if res[0] < self.u_min:
            res[0] = 0
        if res[1] < self.u_min:
            res[1] = 0
        return res

    def bound_desired_speed(self, vd):
        # bound the desired speed to anticipate the motor limitations
        # the speed vector vd is bounded such that the input u = B_inv @ v is unchanged by the "bound_motor" function
        # experimental function, there are other ways to bound the speed
        # vd = (speed, angular speed)

        # step 1
        # square saturation in the box [0,v1max] * [-v2max,v2max]
        res = np.array(vd)
        res[0] = min(max(0.,res[0]),self.v1max)

        if abs(res[1])>self.v2max:
            res[1] = np.sign(res[1])*self.v2max

        # step 2
        # very low speed
        if res[0] <= self.v1min:
            if abs(res[1])>=self.v2min:
                # increase speed to turn with min speed
                res[0] = self.k_1/self.k_2*abs(res[1])
                return res
            else:
                # stop the boat
                res[0] = 0
                res[1] = 0
                return res

        # low speed
        elif res[0] <= 2*self.v1min:
            # low rotation speed
            if abs(res[1]) < self.v2min:
                # dead zone of the motor -> stop the boat
                res[0] = 0
                res[1] = 0
                return res
            else:
                # use only one motor and keep the speed
                res[1] = self.k_2/self.k_1*res[0] * np.sign(res[1])
                return res

        # average speed
        elif res[0] <= 0.5*self.v1max:
            # priority to the speed
            if abs(res[1]) > self.k_2/self.k_1*(res[0]-2*self.v1min):
                # only use on motor and keep the speed
                res[1] = self.k_2/self.k_1*res[0] * np.sign(res[1])
                return res
            else:
                # no saturation
                return res

        # average speed +
        elif res[0] <= 0.5*self.v1max + self.v1min:
            # priority to the speed
            if abs(res[1]) > self.k_2/self.k_1*(res[0]-2*self.v1min):
                # must reduce the angular speed
                res[1] = self.k_2/self.k_1*(res[0]-2*self.v1min)* np.sign(res[1])
                return res
            else:
                # no saturation
                return res

        # high speed saturation
        else:
            if abs(res[1]) > self.k_2*(2*self.u_max - res[0]/self.k_1):
                # priority to the angular speed
                # must reduce the speed
                if abs(res[1]) > self.v2max-self.v2min:
                    res[1] = np.sign(res[1]) * (self.v2max-self.v2min)
                res[0] = self.k_1*(2*self.u_max - abs(res[1])/self.k_2)
                return res
            else:
                # no saturation
                return res

    def convert_motor_control_signal(self, vd):
        # convert desired speed to motor control signal (open loop)
        # vd = (speed (m/s), angular speed (rad/s))
        # return u = (left motor rotation speed (rad/s), right motor rotation speed (rad/s))
        if self.motor_limitation:
            u = self.B_inv @ self.bound_desired_speed(vd)
        else:
            u = self.B_inv @ vd
        return u


    def publish_data(self):
        try:
            mqtt_client.publish(self.topic, json.dumps({"x": self.x[0], "y": self.x[1], "k_1": self.k_1, "k_2": self.k_2, "heading": self.x[2], "state": self.state}, indent=4, ensure_ascii=False))
            print("Asset Data Published")
        except Exception:
            print("Error Publishing Data")

# -------------------------
# Display functions
# -------------------------

def add1(M):
    M = np.array(M)
    return np.vstack((M, np.ones(M.shape[1])))

def tran2H(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

def rot2H(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

def plot2D2(ax, M, col='black', w=1):
    ax.plot(M[0, :], M[1, :], col, linewidth=w)

def draw_tank2(ax,x, col='darkblue', r=0.5, w=2):
    mx, my, θ = list(x[0:3])
    M = r * np.array(
        [[-1, 0, 1, 2,1, 0, -1, -1],
         [-0.5, -0.5, -0.2, 0, 0.2, 0.5, 0.5,-0.5]])
    M = add1(M)
    plot2D2(ax, tran2H(mx, my) @ rot2H(θ) @ M, col, w)

def draw_traj(ax,traj, col='black', w=2):
    position = traj[:,0:2]
    plot2D2(ax, position.T, col, w)

class SimuDisplay:

    def __init__(self,size=(10,10),speed=False,grid=False,
                 v1min=0,v1max=1,v2min = 0, v2max=1):
        self.size_x,self.size_y = size[0], size[1]
        self.fig = plt.figure(figsize=(10,5))
        self.speed = speed
        self.grid = grid
        if not speed:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
            self.ax2.set_aspect('equal')

        self.ax.set_aspect('equal')
        self.v1min = v1min
        self.v1max = v1max
        self.v2min = v2min
        self.v2max = v2max
        self.set_figure()

    def set_figure(self):
        self.ax.set_xlim(-self.size_x, self.size_x)
        self.ax.set_ylim(-self.size_y, self.size_y)
        self.ax.set_title('Boat simulation')
        self.ax.set_xlabel('px (m)')
        self.ax.set_ylabel('py (m)')
        if self.grid:
            self.ax.grid()
            
        if self.speed:
            self.ax2.set_title('Speed')
            self.ax2.set_xlim(-0.1, self.v1max * 1.1)
            self.ax2.set_ylim(-self.v2max * 1.1, self.v2max * 1.1)
            self.ax2.set_xlabel('v1 (m/s)')
            self.ax2.set_ylabel('v2 (rad/s)')

            # plot the speed limits
            self.ax2.plot([self.v1min, 0.5 * self.v1max], [self.v2min, self.v2max], 'k--')
            self.ax2.plot([self.v1min, 0.5 * self.v1max], [-self.v2min, -self.v2max], 'k--')
            self.ax2.plot([2 * self.v1min, 0.5 * self.v1max + self.v1min,
                      self.v1max, 0.5 * self.v1max + self.v1min,
                      2 * self.v1min],
                     [0, self.v2max - self.v2min, 0, -self.v2max + self.v2min, 0], 'k--')

    def clear(self):
        self.ax.clear()
        if self.speed:
            self.ax2.clear()
        self.set_figure()

def clear2(ax,formater='%.0f'):
    try:
        xmin,xmax,ymin,ymax = ax.xmin,ax.xmax,ax.ymin,ax.ymax
        border = True
    except:
        border = False

    if not border:
        try:
            xmin,xmax = ax.get_xlim()
            ymin,ymax = ax.get_ylim()
            border = True
        except:
            border

    try:
        ti = ax.get_title()
        title = True
    except:
        title = False

    xl,yl = ax.get_xlabel(), ax.get_ylabel()

    ax.clear()
    ax.set_xlabel(xl,fontsize= 20)
    ax.set_ylabel(yl,fontsize= 20)
    if border:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    if title:
        ax.set_title(ti, fontsize=20)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(formater))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter(formater))






