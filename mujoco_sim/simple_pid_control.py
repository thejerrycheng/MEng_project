import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def main():
    # Use absolute path to ensure the XML is found
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(curr_dir, "assets", "simple.xml")
    
    if not os.path.exists(xml_path):
        print(f"Error: Could not find XML at {xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Gains tuned for degree-based error
    # Note: Kp 20 in degree-space is much softer than Kp 20 in radian-space
    kp = 40.0  
    kd = 4.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Starting Sine Wave Test: +/- 90 Degrees")
        
        while viewer.is_running():
            step_start = time.time()

            # 1. Target in Degrees
            # 0.5 rad/s frequency = ~12.5 seconds per full cycle
            target_deg = 90.0 * np.sin(data.time * 0.5)
            
            # 2. Convert current state from Radians to Degrees
            current_deg = np.rad2deg(data.qpos[0])
            current_vel_deg = np.rad2deg(data.qvel[0])
            
            # 3. PID calculation in Degree-space
            error = target_deg - current_deg
            torque = (kp * error) - (kd * current_vel_deg)
            
            # 4. Apply Torque
            # MuJoCo applies the raw value in data.ctrl to the motor
            data.ctrl[0] = torque

            # 5. Physics Step
            mujoco.mj_step(model, data)
            viewer.sync()

            # 6. Logging (Every 0.2 seconds)
            if int(data.time * 100) % 20 == 0:
                # actuator_force[0] shows the torque actually applied after XML limits
                actual_torque = data.actuator_force[0]
                print(f"Time: {data.time:5.2f}s | Target: {target_deg:6.1f}° | "
                      f"Actual: {current_deg:6.1f}° | Torque: {actual_torque:6.1f}Nm")

            # 7. Real-time sync
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()