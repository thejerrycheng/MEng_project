import mujoco
import mujoco.viewer
import os
import time

def launch_mesh_viewer(mesh_file_path):
    # Get absolute path and file name
    abs_path = os.path.abspath(mesh_file_path)
    mesh_dir = os.path.dirname(abs_path)
    mesh_name = os.path.basename(abs_path)

    # Temporary XML to load the mesh
    # We use a 1-meter grid to help you judge the scale
    mjcf_inspect = f"""
    <mujoco>
        <compiler meshdir="{mesh_dir}" autolimits="true"/>
        <asset>
            <mesh name="preview_mesh" file="{mesh_name}"/>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
        </asset>
        <worldbody>
            <light diffuse=".8 .8 .8" pos="0 0 3" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="2 2 .01" material="grid"/>
            
            <body name="preview_body" pos="0 0 0.5">
                <freejoint/>
                <geom type="mesh" mesh="preview_mesh" rgba="0.8 0.8 0.8 1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Load and launch
    model = mujoco.MjModel.from_xml_string(mjcf_inspect)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"Viewing: {mesh_name}")
        print("Check if the object is too small or too big relative to the 2m floor.")
        
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    # Point this to any of your .obj files
    target_mesh = "assets/meshes/base_link.obj"
    launch_mesh_viewer(target_mesh)