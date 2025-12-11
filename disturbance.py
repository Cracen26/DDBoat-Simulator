import numpy as np

class DisturbanceManager:
    """
    Handles disturbances on the escort boats.
    Types:
        - shutdown
        - sensor_drift
        - actuator_fault
        - communication_loss
        - cyber_attack
    """
    def __init__(self, disturbances):
        """
        disturbances: list of dicts, each with at least:
            "time": activation time (float)
            "type": "shutdown" | "sensor_drift" | "actuator_fault" | "communication_loss" | "cyber_attack"
            "target": boat index (int)
        plus type-specific parameters (see below).
        """
        self.disturbances = disturbances
        self.active_faults = {}  # boat_id -> fault dict

    def apply(self, t, dt, L_escort, L_status):
        """
        Call this every simulation step.

        t: current sim time
        dt: time step
        L_escort: list of Boat objects
        L_status: list of bool (True = in formation)
        """

        # 1) Activate new disturbances near their scheduled time
        for d in self.disturbances:
            if d.get("_activated", False):
                continue

            if np.isclose(t, d["time"], atol=dt/2):
                boat_id = d["target"]
                if 0 <= boat_id < len(L_escort):
                    d["_activated"] = True
                    print(f"[t={t:.2f}] Activating disturbance {d['type']} on boat {boat_id}")
                    boat = L_escort[boat_id]

                    if d["type"] == "shutdown":
                        self._apply_shutdown(boat_id, boat, L_status)

                    elif d["type"] == "communication_loss":
                        # mark as active fault with end_time
                        d["end_time"] = t + d["duration"]
                        self.active_faults[boat_id] = d
                        self._apply_comm_loss_start(boat_id, boat, L_status)

                    else:
                        # persistent faults stored and applied each step
                        self.active_faults[boat_id] = d
                        # actuator fault needs a one-time setup
                        if d["type"] == "actuator_fault":
                            self._apply_actuator_fault_setup(boat, d)
                else:
                    print(f"[t={t:.2f}] WARNING: boat {boat_id} does not exist, skipping disturbance.")

        # 2) Apply active persistent faults each step
        for boat_id, fault in list(self.active_faults.items()):
            boat = L_escort[boat_id]

            if fault["type"] == "sensor_drift":
                self._apply_sensor_drift(boat, fault)

            elif fault["type"] == "actuator_fault":
                # nothing per-step here if we only adjust efficiency once
                pass

            elif fault["type"] == "cyber_attack":
                self._apply_cyber_attack(boat, fault)

            elif fault["type"] == "communication_loss":
                self._update_comm_loss(boat_id, boat, L_status, fault, t)

    # --------- Disturbance implementations ---------

    def _apply_shutdown(self, boat_id, boat, L_status):
        """Permanent shutdown of the boat."""
        if L_status[boat_id]:
            L_status[boat_id] = False
            boat.state = "Shutdown"
            print(f" -> Boat {boat_id} permanently shut down.")

    def _apply_sensor_drift(self, boat, fault):
        """
        Gradually drifts the position. Each step, we add a small bias.
        bias: np.array([dx, dy]) – interpreted as drift per step.
        """
        bias = fault.get("bias", np.zeros(2))
        boat.x[0] += bias[0]
        boat.x[1] += bias[1]

    def _apply_actuator_fault_setup(self, boat, fault):
        """
        Reduce motor efficiency once.
        efficiency: float in (0, 1]
        """
        eff = fault.get("efficiency", 1.0)
        # store original efficiency if not already
        if not hasattr(boat, "motor_efficiency"):
            boat.motor_efficiency = 1.0
        if not hasattr(boat, "_motor_efficiency_nominal"):
            boat._motor_efficiency_nominal = boat.motor_efficiency

        boat.motor_efficiency = boat._motor_efficiency_nominal * eff
        boat.state = f"ActuatorFault({eff:.2f})"
        print(f" -> Boat {boat.id} motor efficiency set to {boat.motor_efficiency:.2f}")

    def _apply_comm_loss_start(self, boat_id, boat, L_status):
        """Start of a temporary communication loss."""
        L_status[boat_id] = False  # temporarily removed from formation
        boat.state = "CommLoss"
        print(f" -> Boat {boat_id} lost communication (temporarily).")

    def _update_comm_loss(self, boat_id, boat, L_status, fault, t):
        """Check if communication loss interval has ended."""
        if t >= fault["end_time"]:
            L_status[boat_id] = True  # back in formation
            boat.state = "OK"
            print(f"[t={t:.2f}] Communication restored for boat {boat_id}")
            del self.active_faults[boat_id]

    def _apply_cyber_attack(self, boat, fault):
        """
        Inject random noise into state (x, y, heading).
        noise_level: standard deviation of noise.
        """
        level = fault.get("noise_level", 0.1)
        noise = level * np.random.randn(3)
        boat.x[:3] += noise
        boat.state = "CyberAttack"
