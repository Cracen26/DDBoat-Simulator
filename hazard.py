import numpy as np

class DisturbanceManager:
    """
    Gère les perturbations appliquées aux bateaux d'escorte.

    Types supportés:
        - shutdown           : arrêt définitif
        - sensor_drift       : dérive de capteur sur la pose mesurée
        - actuator_fault     : baisse d'efficacité des moteurs (k_1, k_2)
        - communication_loss : perte de com temporaire (retiré de la formation)
        - cyber_attack       : bruit fort sur la pose mesurée (télémesure falsifiée)
    """
    def __init__(self, disturbances):
        """
        disturbances : liste de dicts contenant au moins :
            "time"   : instant d'activation
            "type"   : type de perturbation (string)
            "target" : id du bateau (index dans L_escort)
        """
        self.disturbances = disturbances
        self.active_faults = {}   # boat_id -> fault dict (actuator_fault, comm_loss, cyber_attack)
        self.pose_bias = {}       # boat_id -> biais de pose (pour sensor_drift)

    # ------------------------------------------------------------------
    #  Appelée à chaque pas de temps
    # ------------------------------------------------------------------
    def apply(self, t, dt, L_escort, L_status):
        """
        t        : temps courant
        dt       : pas de temps
        L_escort : liste de Boat
        L_status : liste de bool (True = en formation / actif)
        """

        # 1) Activation des perturbations qui "démarrent" à ce pas de temps
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

                # ----- types de perturbations -----
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

        # 2) Mise à jour des perturbations actives qui dépendent du temps
        for boat_id, fault in list(self.active_faults.items()):
            boat = L_escort[boat_id]

            if fault["type"] == "communication_loss":
                self._update_comm_loss(boat_id, boat, L_status, fault, t)

            # actuator_fault et cyber_attack n'ont pas besoin de traitement
            # supplémentaire ici : actuator_fault a été appliqué une fois,
            # cyber_attack agit dans get_pose_measurement

    # ------------------------------------------------------------------
    #  Fonctions utilisées par le contrôleur pour la "mesure" de pose
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    #  Implémentation détaillée des perturbations
    # ------------------------------------------------------------------
    def _apply_shutdown(self, boat_id, boat, L_status):
        """Arrêt définitif du bateau."""
        if L_status[boat_id]:
            L_status[boat_id] = False
            boat.state = "Shutdown"
            print(f" -> Boat {boat_id} permanently shut down.")

    def _apply_actuator_fault_setup(self, boat, fault):
        """
        Baisse d'efficacité des moteurs : on réduit k_1 et k_2,
        et on met à jour les matrices et vitesses max.
        """
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
        """Début d'une perte de communication : retiré de la formation."""
        L_status[boat_id] = False
        boat.state = "CommLoss"
        print(f" -> Boat {boat_id} temporarily removed from formation (comm loss).")

    def _update_comm_loss(self, boat_id, boat, L_status, fault, t):
        """Fin de la perte de communication : retour dans la formation."""
        if t >= fault["end_time"]:
            L_status[boat_id] = True
            boat.state = "OK"
            print(f"[t={t:.2f}] Communication restored for boat {boat_id}")
            del self.active_faults[boat_id]
