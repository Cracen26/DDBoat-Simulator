import numpy as np
import matplotlib.pyplot as plt
import time
import json

from boat import *
from controller import *

class DisturbanceManager:
    def __init__(self, disturbances):
        self.disturbances = disturbances
        self.active_faults = {}   # boat_id -> fault dict (actuator_fault, comm_loss, cyber_attack)
        self.pose_bias = {}       # boat_id -> biais de pose (pour sensor_drift)

    def apply(self, t, dt, L_escort, L_status):
        """
        t        : temps courant
        dt       : pas de temps
        L_escort : liste de Boat
        L_status : liste de bool (True = en formation / actif)
        """

        # 1) Activation des perturbations qui démarrent à ce pas de temps
        for d in self.disturbances:
            if d.get("_activated", False):
                continue

            if np.isclose(t, d["time"], atol=dt/2):
                boat_id = int(d["target"])
                if not (0 <= boat_id < len(L_escort)):
                    print(f"[t={t:.2f}] WARNING: boat {boat_id} does not exist, skip disturbance.")
                    d["_activated"] = True
                    continue

                boat = L_escort[boat_id]
                d["_activated"] = True
                print(f"[t={t:.2f}] Activating disturbance '{d['type']}' on boat {boat_id}")

                if d["type"] == "shutdown":
                    self._apply_shutdown(boat_id, boat, L_status)

                elif d["type"] == "sensor_drift":
                    # biais fixe sur la pose mesurée
                    bias = np.array(d.get("bias", [0.0, 0.0, 0.0]), dtype=float)
                    self.pose_bias[boat_id] = bias
                    boat.state = "SensorDrift"

                elif d["type"] == "actuator_fault":
                    self._apply_actuator_fault_setup(boat, d)
                    self.active_faults[boat_id] = d

                elif d["type"] == "communication_loss":
                    duration = float(d.get("duration", 3.0))
                    d["end_time"] = t + duration
                    self._apply_comm_loss_start(boat_id, boat, L_status)
                    self.active_faults[boat_id] = d

                elif d["type"] == "cyber_attack":
                    # bruit fort sur la pose mesurée (dans get_pose_measurement)
                    self.active_faults[boat_id] = d
                    boat.state = "CyberAttack"

        # 2) Mise à jour des perturbations actives
        for boat_id, fault in list(self.active_faults.items()):
            boat = L_escort[boat_id]

            if fault["type"] == "communication_loss":
                self._update_comm_loss(boat_id, boat, L_status, fault, t)

    def get_pose_measurement(self, boat_id, true_pose):
        """
        Renvoie la pose mesurée (true_pose + dérives / bruit éventuel).

        true_pose : np.array([x, y, th])
        """
        pose = np.array(true_pose, dtype=float)

        # 1) dérive de capteur (biais constant)
        if boat_id in self.pose_bias:
            bias = self.pose_bias[boat_id]
            pose[:3] += bias  # [dx, dy, dtheta]

        # 2) cyber attack : bruit gaussien fort sur la mesure
        fault = self.active_faults.get(boat_id, None)
        if fault is not None and fault["type"] == "cyber_attack":
            level = float(fault.get("noise_level", 0.2))
            noise = level * np.random.randn(3)
            pose[:3] += noise

        return pose

    def _apply_shutdown(self, boat_id, boat, L_status):
        if L_status[boat_id]:
            L_status[boat_id] = False
            boat.state = "Shutdown"
            print(f" -> Boat {boat_id} permanently shut down.")

    def _apply_actuator_fault_setup(self, boat, fault):
      
        eff = float(fault.get("efficiency", 1.0))

        # Sauvegarder les valeurs nominales si pas déjà fait
        if not hasattr(boat, "_k1_nominal"):
            boat._k1_nominal = boat.k_1
        if not hasattr(boat, "_k2_nominal"):
            boat._k2_nominal = boat.k_2

        boat.k_1 = boat._k1_nominal * eff
        boat.k_2 = boat._k2_nominal * eff

        # Mettre à jour B, B_inv et les vitesses limites
        boat.B = np.array([[boat.k_1, boat.k_1],
                           [-boat.k_2, boat.k_2]])
        boat.B_inv = np.linalg.inv(boat.B)

        boat.v1max = 2 * boat.u_max * boat.k_1
        boat.v1min = boat.u_min * boat.k_1
        boat.v2max = boat.u_max * boat.k_2
        boat.v2min = boat.u_min * boat.k_2

        boat.state = f"ActuatorFault({eff:.2f})"
        print(f" -> Boat {boat.id} actuator efficiency set to {eff:.2f}")

    def _apply_comm_loss_start(self, boat_id, boat, L_status):
        L_status[boat_id] = False
        boat.state = "CommLoss"
        print(f" -> Boat {boat_id} temporarily removed from formation (comm loss).")

    def _update_comm_loss(self, boat_id, boat, L_status, fault, t):
        if t >= fault["end_time"]:
            L_status[boat_id] = True
            boat.state = "OK"
            print(f"[t={t:.2f}] Communication restored for boat {boat_id}")
            del self.active_faults[boat_id]

def update_reference(L_phi, L_status):
    N_active = sum(1 for s in L_status if s)
    j = 0
    for i in range(len(L_phi)):
        if L_status[i]:
            L_phi[i] = 2 * -j * np.pi / max(N_active, 1)
            j += 1
    return L_phi, N_active

def main():
    speed = False       # affichage du diagramme vitesse (False = désactivé)
    N = 10              # nombre de bateaux d'escorte
    circle_radius = 5
    w_patrol = 0.2      # vitesse angulaire de patrouille sur le cercle

    # Etat initial du bateau maître
    x0_master = np.array([0., 0., 0.])

    # Etats initiaux des escortes
    Lx0 = [np.array([-10., (i - N/2), 0.]) for i in range(N)]

    # Angles initiaux de répartition sur le cercle
    L_phi = [2 * -i * np.pi / N for i in range(N)]

    disturbances = [
        # 1) shutdown : bateau 2 meurt à t = 5 s
        {"time": 5.0, "type": "shutdown", "target": 2},

        # 2) sensor_drift : bateau 3 a une dérive capteur à partir de t = 8 s
        #    (biais de +0.02 m en x, -0.01 m en y, +2° sur le heading)
        {"time": 8.0, "type": "sensor_drift", "target": 3,
         "bias": np.array([0.02, -0.01, np.deg2rad(2.0)])},

        # 3) actuator_fault : bateau 5 avec moteurs à 50% à partir de t = 12 s
        {"time": 12.0, "type": "actuator_fault", "target": 5,
         "efficiency": 0.5},

        # 4) communication_loss : bateau 1 quitte la formation entre t = 15 et t = 19 s
        {"time": 15.0, "type": "communication_loss", "target": 1,
         "duration": 4.0},

        # 5) cyber_attack : bateau 0 reçoit des mesures très bruitées à partir de t = 18 s
        {"time": 18.0, "type": "cyber_attack", "target": 0,
         "noise_level": 0.3},
    ]

    disturbance_manager = DisturbanceManager(disturbances)

    boat_master = Boat(255, x0_master, traj_memory_=-1)
    boat_master.state = "OK"

    L_escort = []
    L_status = []
    for i in range(N):
        b = Boat(i, Lx0[i], traj_memory_=200, motor_limitation=False)
        b.state = "OK"
        L_escort.append(b)
        L_status.append(True)   # tous actifs au début

    sim = SimuDisplay()
    dt = 0.05
    T = 30.0

    # Pour le calcul de la cinématique du maître
    p_dot_master = np.array([0.0, 0.0])

    # Vitesse désirée mémorisée pour chaque bateau (pour follow_pose)
    L_vd = [np.array([0.0, 0.0]) for _ in range(N)]

    # Logging
    logs = []
    L_t = []
    L_ope = []

    t = 0.0

    while t < T:
        sim.clear()

        # 1) Appliquer les perturbations pour ce pas de temps
        disturbance_manager.apply(t, dt, L_escort, L_status)

        # 2) Mettre à jour le bateau maître (ici simple commande constante)
        u_master = np.array([0.0, 30.0])  # (left motor, right motor)
        v_master = boat_master.motor_update(dt, u_master)  # [v_lin, v_ang]

        p_dot_master_old = p_dot_master.copy()
        p_dot_master = v_master[0] * np.array([np.cos(boat_master.th()),
                                               np.sin(boat_master.th())])
        p_ddot_master = (p_dot_master - p_dot_master_old) / dt

        # 3) Affichage du maître + cercle de patrouille
        sim.ax.add_artist(
            plt.Circle((boat_master.x[0], boat_master.x[1]),
                       circle_radius, color='r', fill=False)
        )
        draw_tank2(sim.ax, boat_master.x, "r", r=1)
        if hasattr(boat_master, "traj"):
            draw_traj(sim.ax, boat_master.traj, "r")

        # 4) Répartition des escorts sur le cercle
        ope = 0.0
        L_phi, N_effectif = update_reference(L_phi, L_status)

        for i in range(N):
            if L_status[i]:
                # ----- Bateau actif : suit sa position sur le cercle -----
                angle = L_phi[i] + w_patrol * t
                center = boat_master.p()

                pd = center + circle_radius * np.array([
                    np.cos(angle),
                    np.sin(angle)
                ])

                pd_dot = p_dot_master + circle_radius * w_patrol * np.array([
                    -np.sin(angle),
                    np.cos(angle)
                ])

                pd_ddot = p_ddot_master + circle_radius * (w_patrol**2) * np.array([
                    -np.cos(angle),
                    -np.sin(angle)
                ])

                # contribution à l'"operability"
                ope += 1.0 / (1.0 + np.linalg.norm(pd - L_escort[i].p())) / max(N_effectif, 1)

                sim.ax.plot(pd[0], pd[1], 'k*', markersize=5)

                # pose mesurée (avec drift / cyber attack éventuels)
                pose_meas = disturbance_manager.get_pose_measurement(i, L_escort[i].x)

                # contrôle
                L_vd[i] = follow_pose(pd, pose_meas, dt,
                                      v1_old=L_vd[i][0],
                                      pd_dot=pd_dot,
                                      pd_ddot=pd_ddot)

                u = L_escort[i].convert_motor_control_signal(L_vd[i])
                v = L_escort[i].motor_update(dt, u)

                # dessin du bateau (état vrai)
                draw_tank2(sim.ax, L_escort[i].x, "k")
                if hasattr(L_escort[i], "traj"):
                    draw_traj(sim.ax, L_escort[i].traj, "k")
            else:
                # ----- Bateau inactif : on l'affiche en vert, vitesse nulle -----
                v = np.array([0.0, 0.0])
                draw_tank2(sim.ax, L_escort[i].x, "g")

            # ---- Logging pour ce bateau et cet instant ----
            logs.append({
                "time": float(t),
                "boat_id": int(i),
                "x": float(L_escort[i].x[0]),
                "y": float(L_escort[i].x[1]),
                "heading": float(L_escort[i].x[2]),
                "v_linear": float(v[0]),
                "v_angular": float(v[1]),
                "state": str(L_escort[i].state),
            })

        # Affichage du temps
        sim.ax.text(0.02, 0.95, f"Time: {t:.2f} s", transform=sim.ax.transAxes)

        L_t.append(t)
        L_ope.append(ope)

        # Avancer le temps
        t += dt

        # Debug console de temps en temps
        if int(t / dt) % 10 == 0:
            for i in range(N):
                print(
                    f"t={t:.2f}, Boat {i}: x={L_escort[i].x[0]:.2f}, "
                    f"y={L_escort[i].x[1]:.2f}, heading={L_escort[i].x[2]:.2f}, "
                    f"state={L_escort[i].state}"
                )

        plt.pause(0.05)

    with open("logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    plt.figure()
    plt.plot(L_t, L_ope)
    plt.xlabel("t (s)")
    plt.ylabel("Operability")
    plt.title("Operability over time")
    plt.grid(True)
    print("Simulation finished, logs saved to logs.json")
    plt.show()


if __name__ == "__main__":
    main()
