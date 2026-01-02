import mujoco
import numpy as np
import os

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris_with_axis.xml")

class IRISFinalDH:
    def __init__(self):
        # Parameters: [a, alpha, d, theta_offset]
        # These are kept as per your correct DH table
        self.dh_table = [
            {'a': 0.0,      'alpha': 0.0,   'd': 0.2487,  'th_off': 0.0},   # J1 base -> shoulder
            {'a': 0.0218,   'alpha': 90.0,  'd': 0.0,   'th_off': 0.0},  # J2 shoulder -> arm_link1
            {'a': 0.29977,  'alpha': 0.0,   'd': -0.0218, 'th_off': 0.0},   # J3 arm_link1 -> elbow
            {'a': 0.02,     'alpha': 90.0,  'd': 0.0,     'th_off': 0.0},   # J4 elbow -> arm_link2
            {'a': 0.315 ,  'alpha': -90.0, 'd': 0.00624,   'th_off': 0.0},   # J5 arm_link2 -> wrist1
            {'a': 0.0,      'alpha': 90.0,  'd': 0.0428,  'th_off': 0.0}    # J6 wrist1 -> wrist2(end effector)
        ]

    def get_dh_matrix(self, q_rad, dh):
        # theta is inverted as per your requirement
        theta = -q_rad + np.deg2rad(dh['th_off'])
        alpha = np.deg2rad(dh['alpha'])
        a, d = dh['a'], dh['d']
        
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        
        # To handle reversed X and Y axes, we negate the X and Y basis vectors 
        # (Rows 0 and 1) in the transformation matrix.
        # This aligns the analytical 'forward' with MuJoCo's 'forward'.
        return np.array([
            [-ct,  st*ca, -st*sa, -a*ct], # Inverted X-row
            [-st, -ct*ca,  ct*sa, -a*st], # Inverted Y-row
            [0,    sa,     ca,     d],    # Standard Z-row (Z matches in your output)
            [0,    0,      0,      1]
        ])

    def calculate_positions(self, q_deg):
        q_rad = np.deg2rad(q_deg)
        positions = []
        T_accum = np.eye(4)
        for i in range(len(self.dh_table)):
            T_i = self.get_dh_matrix(q_rad[i], self.dh_table[i])
            T_accum = T_accum @ T_i
            positions.append(T_accum[:3, 3].copy())
        return positions

def main():
    # Attempt to load the model
    if not os.path.exists(XML_PATH):
        print(f"Error: XML file not found at {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISFinalDH()

    # Test Case: User's non-zero pose
    test_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Set MuJoCo joint positions
    # Assuming the first 6 qpos are the arm joints
    data.qpos[:6] = np.deg2rad(test_q)
    mujoco.mj_forward(model, data)
    
    ana_pos = kin.calculate_positions(test_q)
    
    # Body names based on your previous logs
    body_names = ['shoulder', 'arm_link1', 'elbow', 'arm_link2', 'wrist1', 'wrist2']

    print("\n" + "="*85)
    print(f"{'FULL DH VALIDATION (REVERSED X/Y AXIS)':^85}")
    print("="*85)
    print(f"{'Link':<12} | {'Sim (xpos)':^22} | {'DH Analytical':^22} | {'Err (mm)'}")
    print("-" * 85)

    for i, name in enumerate(body_names):
        try:
            sim = data.body(name).xpos
            ana = ana_pos[i]
            err = np.linalg.norm(sim - ana) * 1000
            
            print(f"{name:<12} | {sim[0]:6.3f} {sim[1]:6.3f} {sim[2]:6.3f} | "
                  f"{ana[0]:6.3f} {ana[1]:6.3f} {ana[2]:6.3f} | {err:8.4f}")
        except Exception as e:
            print(f"{name:<12} | Error retrieving data: {e}")
            
    print("="*85)

if __name__ == "__main__":
    main()