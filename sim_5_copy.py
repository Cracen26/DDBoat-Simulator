import numpy as np
import time
import json
import matplotlib.pyplot as plt
from boat import *
from controller import *

# -------------------------------
# Helper function (can be static)
# -------------------------------
def update_reference(L_phi, L_status):
    """Update reference angles based on active boats."""
    N_effective = sum(L_status)
    j = 0
    for i in range(len(L_phi)):
        if L_status[i]:
            L_phi[i] = 2 * -j * np.pi / N_effective
            j += 1
    return L_phi, N_effective

# -------------------------------
# Boat Patrol Simulation Class
# -------------------------------
class BoatPatrolSimulation:
    def __init__(self, N=10, circle_radius=5, w_patrol=0.2, dt=0.05, T=30.0, speed=False):
        self.N = N
        self.circle_radius = circle_radius
        self.w_patrol = w_patrol
        self.dt = dt
        self.T = T
        self.speed = speed

        # Initialize master boat
        self.master = Boat(255, np.array([0., 0, 0]), traj_memory_=-1)

        # Initialize escort boats
        self.escort_boats = []
        self.status = []
        self.L_phi = []
        for i in range(N):
            x0 = np.array([-10, (i-N/2), 0])
            self.escort_boats.append(Boat(i, x0, traj_memory_=20, motor_limitation=False))
            self.status.append(True)
            self.L_phi.append(2 * -i * np.pi / N)

        # Killer plan
        self.killer_plan = np.array([[2,5],[7,10],[0,15],[4,20],[6,25],[8,26],[9,27],[3,28]])

        # Display
        self.sim = SimuDisplay()
        self.L_vd = [np.array([0,0]) for _ in range(N)]
        self.p_dot_master = np.array([0,0])

        # Logs
        self.L_t = []
        self.L_ope = []
        self.logs = []

        #internal simulation time 
        self.t = 0.0

        if self.speed:
            self.fig, self.ax2 = plt.subplots()
            self.ax2.set_xlim(0.1, self.escort_boats[0].v1max*1.1)
            self.ax2.set_ylim(-self.escort_boats[0].v2max*1.1, self.escort_boats[0].v2max*1.1)

    def step(self, dt=None):
        """Advance simulation by dt (or default dt) and return current states of all boats."""
        if dt is None:
            dt = self.dt

        self.sim.clear()
        if self.speed:
            clear2(self.ax2)
            self._plot_speed_limits()

        # Master control input (can be extended)
        u_master = np.array([0,30])
        v_master = self.master.motor_update(dt, u_master)
        p_dot_master_old = np.array(self.p_dot_master)
        self.p_dot_master = v_master * np.array([np.cos(self.master.th()), np.sin(self.master.th())])
        p_ddot_master = (self.p_dot_master - p_dot_master_old) / dt

        # Draw master boat
        self._draw_master()

        # Update references
        self.L_phi, N_effective = update_reference(self.L_phi, self.status)
        operability = 0

        for i, boat in enumerate(self.escort_boats):
            if self.status[i]:
                operability += self._update_boat(boat, i, self.t, N_effective, p_ddot_master)

        # Time display
        self.sim.ax.text(0.1, 0.9, f'Time: {self.t:.2f}', transform=self.sim.ax.transAxes)

        self.L_t.append(self.t)
        self.L_ope.append(operability)

        # Kill boats
        for killer in self.killer_plan:
            if self.t > killer[1] and self.status[killer[0]]:
                self.status[killer[0]] = False
                self.escort_boats[killer[0]].state = 'Attacked'
                print(f"Killing robot {killer[0]}")

        # Periodic logging
        # if int(self.t/self.dt) % 10 == 0:
        #     self._log_status(self.t)

        plt.pause(0.01)
        self.t += dt

        # Return states of all boats
        states = []
        for boat in self.escort_boats:
            states.append({
                "id": boat.id,
                "x": boat.x[0],
                "y": boat.x[1],
                "heading": boat.x[2],
                "linear": boat.k_1,
                "angular": boat.k_2,
            })
        return states
    # -------------------------------
    # Simulation Loop
    # -------------------------------
    def run1(self):
        t = 0
        while t < self.T:
            t_start = time.time()
            self.sim.clear()
            if self.speed:
                clear2(self.ax2)
                self._plot_speed_limits()

            # Master control input (can be extended)
            u_master = np.array([0,30])
            v_master = self.master.motor_update(self.dt, u_master)
            p_dot_master_old = np.array(self.p_dot_master)
            self.p_dot_master = v_master * np.array([np.cos(self.master.th()), np.sin(self.master.th())])
            p_ddot_master = (self.p_dot_master - p_dot_master_old) / self.dt

            # Draw master boat
            self._draw_master()

            # Update references
            self.L_phi, N_effective = update_reference(self.L_phi, self.status)
            operability = 0

            for i, boat in enumerate(self.escort_boats):
                if self.status[i]:
                    operability += self._update_boat(boat, i, t, N_effective, p_ddot_master)
                else:
                    draw_tank2(self.sim.ax, boat.x, "g")

            # Time display
            self.sim.ax.text(0.1, 0.9, f'Time: {t:.2f}', transform=self.sim.ax.transAxes)

            self.L_t.append(t)
            self.L_ope.append(operability)

            # Kill boats
            for killer in self.killer_plan:
                if t > killer[1] and self.status[killer[0]]:
                    self.status[killer[0]] = False
                    self.escort_boats[killer[0]].state = 'Attacked'
                    print(f"Killing robot {killer[0]}")

            # Periodic logging
            if int(t/self.dt) % 10 == 0:
                self._log_status(t)

            plt.pause(0.01)
            t += self.dt

        self._save_logs()
        self._plot_operability()

    def run(self):
        """Run the full simulation by repeatedly stepping through time."""
        i = 0
        while self.t < self.T:
            ivar = self.step(self.dt)  # advance by one time step
            print(ivar)
            if i == 2: break
            i +=1

        # self._save_logs()
        # self._plot_operability()

    # -------------------------------
    # Internal Methods
    # -------------------------------
    def _draw_master(self):
        draw_tank2(self.sim.ax, self.master.x,"r", r=1)
        draw_traj(self.sim.ax, self.master.traj, "r")
        self.sim.ax.add_artist(plt.Circle((self.master.x[0], self.master.x[1]),
                                          self.circle_radius, color='r', fill=False))

    def _update_boat(self, boat, i, t, N_effective, p_ddot_master):
        """Update boat reference and motor."""
        phi = self.L_phi[i]
        pd = self.master.p() + self.circle_radius * np.array([np.cos(phi + self.w_patrol*t),
                                                              np.sin(phi + self.w_patrol*t)])
        pd_dot = self.p_dot_master + self.circle_radius * self.w_patrol * np.array([-np.sin(phi + self.w_patrol*t),
                                                                                   np.cos(phi + self.w_patrol*t)])
        pd_ddot = p_ddot_master + self.circle_radius * self.w_patrol**2 * np.array([-np.cos(phi + self.w_patrol*t),
                                                                                   -np.sin(phi + self.w_patrol*t)])
        # Operability metric
        ope = 1/(1+np.linalg.norm(pd-boat.p())) / N_effective
        self.sim.ax.plot(pd[0], pd[1], 'k*', markersize=5)

        # Control update
        self.L_vd[i] = follow_pose(pd, boat.x, self.dt, v1_old=self.L_vd[i][0], pd_dot=pd_dot, pd_ddot=pd_ddot)
        u = boat.convert_motor_control_signal(self.L_vd[i])
        boat.motor_update(self.dt, u)

        draw_tank2(self.sim.ax, boat.x, "k")
        draw_traj(self.sim.ax, boat.traj, "k")

        if self.speed:
            self.ax2.plot(boat.x[0], boat.x[1], 'ko', markersize=5)
            self.ax2.plot(self.L_vd[i][0], self.L_vd[i][1], 'kx', markersize=10)

        return ope

    def _plot_speed_limits(self):
        ax2 = self.ax2
        boat = self.escort_boats[0]
        ax2.plot([boat.v1min, 0.5 * boat.v1max], [boat.v2min, boat.v2max], 'k--')
        ax2.plot([boat.v1min, 0.5 * boat.v1max], [-boat.v2min, -boat.v2max], 'k--')
        ax2.plot([2 * boat.v1min, 0.5*boat.v1max + boat.v1min, boat.v1max, 0.5*boat.v1max + boat.v1min, 2*boat.v1min],
                 [0, boat.v2max - boat.v2min, 0, -boat.v2max + boat.v2min, 0], 'k--')

    def _log_status(self, t):
        for i, boat in enumerate(self.escort_boats):
            print(f"t={t:.2f}, Boat {i}: x={boat.x[0]:.2f}, y={boat.x[1]:.2f}, heading={boat.x[2]:.2f}, state={boat.state}")

    def _save_logs(self):
        with open("logs.json", "w", encoding="utf-8") as f:
            json.dump([log.to_dict() for log in self.logs], f, indent=4, ensure_ascii=False)

    def _plot_operability(self):
        plt.figure()
        plt.plot(self.L_t, self.L_ope)
        plt.title("Operability")
        plt.show()
        print("Simulation complete.")

# -------------------------------
# Run the simulation
# -------------------------------
if __name__ == "__main__":
    sim = BoatPatrolSimulation()
    sim.run()
