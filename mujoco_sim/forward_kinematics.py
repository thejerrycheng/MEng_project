import mujoco
import numpy as np
import os

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris_with_axis.xml")

class IRISKinematics:
    def __init__(self):
        # Configuration strictly extracted from your XML <body pos="..." euler="...">
        self.link_configs = [
            # Joint 1: shoulder
            {'pos': [0, 0, 0.2487],        'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            # Joint 2: arm_link1
            {'pos': [0.0218, 0, 0.059],    'euler': [0, 90, 180], 'axis': [0, 0, 1]}, 
            # Joint 3: elbow
            {'pos': [0.299774, 0, -0.0218],'euler': [0, 0, 0],    'axis': [0, 0, 1]}, 
            # Joint 4: arm_link2
            {'pos': [0.02, 0, 0],          'euler': [0, 90, 0],   'axis': [0, 0, 1]}, 
            # Joint 5: wrist1
            {'pos': [0, -0.00624, 0.315],  'euler': [0, -90, 0],  'axis': [0, 0, 1]}, 
            # Joint 6: wrist2
            {'pos': [0.042824, 0, 0],      'euler': [0, 90, 0],   'axis': [0, 0, 1]}  
        ]

    def get_local_transform(self, config, q_rad):
        T = np.eye(4)
        T[:3, 3] = config['pos']
        
        # 1. Fixed Rotation (Euler -> Quat -> Mat)
        R_fixed = np.eye(3)
        if any(config['euler']):
            quat_e = np.zeros(4)
            mujoco.mju_euler2Quat(quat_e, np.deg2rad(config['euler']), 'xyz')
            res_e = np.zeros(9)
            mujoco.mju_quat2Mat(res_e, quat_e)
            R_fixed = res_e.reshape(3, 3)
            
        # 2. Joint Rotation (Axis-Angle -> Quat -> Mat)
        quat_j = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat_j, np.array(config['axis']), q_rad)
        res_j = np.zeros(9)
        mujoco.mju_quat2Mat(res_j, quat_j)
        R_joint = res_j.reshape(3, 3)
        
        T[:3, :3] = R_fixed @ R_joint
        return T

    def calculate_all_positions(self, q_deg):
        q_rad = np.deg2rad(q_deg)
        positions = []
        T_accum = np.eye(4)
        for i in range(len(self.link_configs)):
            T_local = self.get_local_transform(self.link_configs[i], q_rad[i])
            T_accum = T_accum @ T_local
            positions.append(T_accum[:3, 3].copy())
        return positions

def run_test_case(name, test_q, model, data, kin):
    """Executes a single FK test case and returns the max error."""
    # Set Sim state
    data.qpos[:6] = np.deg2rad(test_q)
    mujoco.mj_forward(model, data)
    
    # Set Analytical state
    analytical_pos = kin.calculate_all_positions(test_q)
    body_names = ['shoulder', 'arm_link1', 'elbow', 'arm_link2', 'wrist1', 'wrist2']
    
    errors = []
    for i, b_name in enumerate(body_names):
        sim_pos = data.body(b_name).xpos
        ana_pos = analytical_pos[i]
        errors.append(np.linalg.norm(sim_pos - ana_pos))
    
    max_err_mm = np.max(errors) * 1000
    status = "PASS" if max_err_mm < 1e-4 else "FAIL"
    
    print(f" {name:<20} | {str(test_q):<40} | {max_err_mm:10.6f} | {status}")
    return max_err_mm

def main():
    if not os.path.exists(XML_PATH):
        print(f"Error: XML not found at {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISKinematics()

    print("\n" + "="*85)
    print(f"{'FK VERIFICATION SUITE':^85}")
    print("="*85)
    print(f" {'Test Name':<20} | {'Joint Angles (Deg)':^40} | {'Max Err(mm)':^10} | {'Result'}")
    print("-" * 85)

    # Test Case 1: Zero Position (Home)
    run_test_case("Home Position", [0, 0, 0, 0, 0, 0], model, data, kin)

    # Test Case 2: Full Extension
    run_test_case("Full Extension", [0, -90, 0, 0, 0, 0], model, data, kin)

    # Test Case 3: Complex Pose (Random-ish)
    run_test_case("Complex Pose", [45, -30, 15, 90, -45, 180], model, data, kin)

    # Test Case 4: Wrist Only
    run_test_case("Wrist Isolation", [0, 0, 0, 45, 90, 45], model, data, kin)

    # Test Case 5: Negative Quadrants
    run_test_case("Negative Quadrant", [-120, -10, -80, -30, -15, -90], model, data, kin)

    print("="*85 + "\n")

if __name__ == "__main__":
    main()