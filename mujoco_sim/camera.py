import mujoco
import mujoco.viewer
import numpy as np
import cv2
import argparse
import time

def main():
    # 1. Setup Argument Parsing
    parser = argparse.ArgumentParser(description="MuJoCo 6-DOF Arm Simulation")
    parser.add_argument('--camera', action='store_true', help="Display the end-effector camera feed")
    args = parser.parse_args()

    # 2. Load the Model
    # Ensure your XML file name matches
    model = mujoco.MjModel.from_xml_path('6dof_arm.xml')
    data = mujoco.MjData(model)

    # 3. Setup Renderer (only if camera flag is active)
    renderer = None
    if args.camera:
        # Match the resolution you want for the pop-up window
        renderer = mujoco.Renderer(model, height=480, width=640)
        print("Camera feed enabled. Press 'q' on the image window to close the feed (but keep sim running).")

    # 4. Launch Main Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation started. Use the viewer to interact with the arm.")
        
        while viewer.is_running():
            step_start = time.time()

            # Physics step
            mujoco.mj_step(model, data)

            # 5. Handle Camera Visualization
            if args.camera and renderer:
                # Update the renderer specifically for the "hand_cam"
                renderer.update_scene(data, camera="hand_cam")
                pixels = renderer.render()
                
                # Convert RGB (MuJoCo) to BGR (OpenCV)
                bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                
                # Display in an OpenCV window
                cv2.imshow("End-Effector Feed (hand_cam)", bgr_pixels)
                
                # Check for 'q' key to stop showing the camera window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    args.camera = False
                    cv2.destroyAllWindows()

            # Sync the passive viewer
            viewer.sync()

            # Maintain real-time frequency
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Cleanup
    if args.camera:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()