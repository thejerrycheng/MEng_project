import mujoco
import numpy as np
import os

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris.xml")


class IRISFinalDH:
    def __init__(self):
        # Parameters: [a, alpha, d, theta_offset]
        self.dh_table = [
            {'a': 0.0,      'alpha': 0.0,   'd': 0.2487,  'th_off': 0.0},   # J1: ground -> base 
            {'a': 0.0218,      'alpha': 90.0,  'd': 0.059,   'th_off': 180.0}, # J2: base -> shoulder
            {'a': 0.299774,  'alpha': 0.0, 'd': 0.0,     'th_off': 0.0},  # shoulder -> arm_link1
            {'a': 0.02,      'alpha': 90.0,  'd': 0.0,     'th_off': 0.0},   # J4: arm_link1 -> elbow
            {'a': 0.32512 ,   'alpha': -90.0, 'd': 0.0,     'th_off': 0.0},   # J5: elbow -> wrist1
            {'a': 0.0428,      'alpha': 90.0,  'd': 0,  'th_off': 0.0},    # J6: wrist1 -> wrist2
        ]

    def get_dh_matrix(self, q_rad, dh):
        # Using a standard DH matrix structure is safer for coordinate propagation
        theta = -q_rad + np.deg2rad(dh['th_off'])
        alpha = np.deg2rad(dh['alpha'])
        a, d = dh['a'], dh['d']
        
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        # Standard DH Matrix (Internal X/Y logic is handled by standard indices)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,      ca,    d],
            [0,   0,       0,     1]
        ])

    def calculate_positions(self, q_deg):
        q_rad = -np.deg2rad(q_deg)
        positions = []
        T_accum = np.eye(4)
        
        # Initial Base Rotation to align DH frame with MuJoCo World
        # This fixes the "Reversed X/Y" issue at the source
        for i in range(len(self.dh_table)):
            T_i = self.get_dh_matrix(q_rad[i], self.dh_table[i])
            T_accum = T_accum @ T_i
            positions.append(T_accum[:3, 3].copy())
        return positions
    
def run_validation(model, data, kin, test_name, q_deg):
    data.qpos[:6] = np.deg2rad(q_deg)
    mujoco.mj_forward(model, data)
    
    ana_pos = kin.calculate_positions(q_deg)
    body_names = ['shoulder', 'arm_link1', 'elbow', 'arm_link2', 'wrist1', 'wrist2']

    print(f"\n--- TEST CASE: {test_name} (q={q_deg}) ---")
    print(f"{'Link':<12} | {'Sim (xpos)':^20} | {'DH (pos)':^20} | {'Err (mm)'}")
    print("-" * 75)

    for i, name in enumerate(body_names):
        sim = data.body(name).xpos
        ana = ana_pos[i]
        err = np.linalg.norm(sim - ana) * 1000
        print(f"{name:<12} | {sim[0]:5.2f} {sim[1]:5.2f} {sim[2]:5.2f} | {ana[0]:5.2f} {ana[1]:5.2f} {ana[2]:5.2f} | {err:8.2f}")

def main():
    if not os.path.exists(XML_PATH): return
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISFinalDH()

    # TEST 1: Zero Pose (Vertical Extension)
    run_validation(model, data, kin, "ZERO POSE", [0, 0, 0, 0, 0, 0])

    # TEST 2: Elbow Flex (90 degrees)
    run_validation(model, data, kin, "ELBOW FLEX", [0, 0, 90, 0, 0, 0])

    # TEST 3: Complex Multi-axis rotation
    run_validation(model, data, kin, "COMPLEX POSE", [45, -30, 45, 10, -20, 0])

if __name__ == "__main__":
    main()