import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# --- Configuration ---
XML_NAME = "iris.xml"
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", XML_NAME)

class SequentialController:
    def __init__(self, model):
        # Base position to hold
        self.base_q = np.array([0.0, -30.0, 60.0, 0.0, 45.0, 0.0])
        self.kp = 350.0  
        self.kd = 35.0
        
        # Sequential Logic
        self.current_joint_idx = 0
        self.seconds_per_joint = 4.0
        self.last_transition_time = 0.0

    def get_control(self, model, data):
        t = data.time
        
        # Check if it's time to move to the next joint
        if t - self.last_transition_time > self.seconds_per_joint:
            self.current_joint_idx = (self.current_joint_idx + 1) % 6
            self.last_transition_time = t
            print(f"\n>>> SWITCHING TO JOINT {self.current_joint_idx + 1} <<<")

        # Calculate target: Hold base_q, but add sine wave only to the active joint
        target_q = self.base_q.copy()
        oscillation = 30.0 * np.sin((t - self.last_transition_time) * 2.0)
        target_q[self.current_joint_idx] += oscillation
        
        # Current state
        current_q = data.qpos[:6]
        current_dq = data.qvel[:6]
        
        # PID Logic
        error = target_q - current_q
        torque = self.kp * error - self.kd * current_dq
        
        # Logging
        if int(t * 10) % 5 == 0: # Log every 0.5 seconds of sim time
            self.log_state(t, target_q, current_q, torque)
            
        return torque

    def log_state(self, t, target, current, torque):
        j_idx = self.current_joint_idx
        print(f"Time: {t:5.2f}s | J{j_idx+1} | Target: {target[j_idx]:7.2f}° | "
              f"Actual: {current[j_idx]:7.2f}° | Torque: {torque[j_idx]:7.2f}Nm")

def main():
    if not os.path.exists(XML_PATH):
        print(f"Error: Cannot find {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = SequentialController(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Starting Sequential Joint Test...")
        print("-" * 60)
        
        while viewer.is_running():
            step_start = time.time()

            # 1. PID Control
            data.ctrl[:6] = controller.get_control(model, data)

            # 2. Step physics
            mujoco.mj_step(model, data)

            # 3. Sync viewer
            viewer.sync()

            # 4. Real-time synchronization
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()