import mujoco
import numpy as np
import os

# --- Configuration ---
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURR_DIR, "assets", "iris_with_axis.xml")

class IRISFinalDH:
    def __init__(self):
        # Parameters derived from your geometric extraction and frame alignment
        # Format: [a, alpha, d, theta_offset] (Units: Meters, Degrees)
        self.dh_table = [
            {'a': 0.0,      'alpha': 0.0,   'd': 0.2487,  'th_off': 0.0},   # J1
            {'a': 0.059,   'alpha': 90.0,  'd': 0.0218,   'th_off': 180.0},  # J2
            {'a': 0.29977,  'alpha': 0.0,   'd': -0.0218, 'th_off': 0.0},   # J3
            {'a': 0.02,     'alpha': 90.0, 'd': 0.0,     'th_off': 0.0},   # J4
            {'a': 0.00624,  'alpha': -90.0,  'd': 0.315,   'th_off': 0.0},   # J5
            {'a': 0.0,      'alpha': 90.0,   'd': 0.0428,  'th_off': 0.0}    # J6
        ]

    def get_dh_matrix(self, q_rad, dh):
        theta = q_rad + np.deg2rad(dh['th_off'])
        alpha = np.deg2rad(dh['alpha'])
        a, d = dh['a'], dh['d']
        
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
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
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    kin = IRISFinalDH()

    # Test Case: Non-zero pose to check cross-axis accuracy
    test_q = np.array([10.0, -20.0, 30.0, 45.0, -10.0, 90.0])
    data.qpos[:6] = np.deg2rad(test_q)
    mujoco.mj_forward(model, data)
    
    ana_pos = kin.calculate_positions(test_q)
    body_names = ['shoulder', 'arm_link1', 'elbow', 'arm_link2', 'wrist1', 'wrist2']

    print("\n" + "="*80)
    print(f"{'FINAL DH VALIDATION':^80}")
    print("="*80)
    print(f"{'Link':<12} | {'Sim (xpos)':^22} | {'DH Analytical':^22} | {'Err (mm)'}")
    print("-" * 80)

    for i, name in enumerate(body_names):
        sim = data.body(name).xpos
        ana = ana_pos[i]
        err = np.linalg.norm(sim - ana) * 1000
        print(f"{name:<12} | {sim[0]:6.3f} {sim[1]:6.3f} {sim[2]:6.3f} | "
              f"{ana[0]:6.3f} {ana[1]:6.3f} {ana[2]:6.3f} | {err:8.4f}")
    print("="*80)

if __name__ == "__main__":
    main()