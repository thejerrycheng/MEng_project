import mujoco
import mujoco.viewer
import time

# MJCF Model Definition
# Gravity is set in the <option> tag: [x, y, z] -> [0, 0, -9.81]
mjcf_model = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.002"/>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    </asset>

    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="0 0 .05" material="grid" condim="1"/>
        
        <body name="box1" pos="0.5 0 1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1" rgba="1 0.5 0 1" mass="1"/>
        </body>

        <body name="ball1" pos="0 0 2">
            <joint type="free"/>
            <geom type="sphere" size="0.1" rgba="0 0.7 1 1" mass="0.5"/>
        </body>
    </worldbody>
</mujoco>
"""

def main():
    # 1. Load the model and data
    model = mujoco.MjModel.from_xml_string(mjcf_model)
    data = mujoco.MjData(model)

    # 2. Launch the passive viewer (Best for Mac)
    # viewer = mujoco_viewer.MujocoViewer(model, data)
    # print("Simulation started. Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation running in real-time. Gravity: -9.81 m/s^2")
        
        while viewer.is_running():
            step_start = time.time()

            # 3. Advance physics simulation
            mujoco.mj_step(model, data)

            # 4. Sync the viewer with the data
            viewer.sync()

            # 5. Real-time synchronization
            # Calculate how much time the physics step actually took
            elapsed = time.time() - step_start
            
            # If we are running faster than the physics timestep, sleep the difference
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()