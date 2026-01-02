import mujoco
import numpy as np
import os

class IRISAnalyticalIK:
    def __init__(self):
        # Kinematic parameters from XML
        self.d1 = 0.2487 + 0.059  # Base to arm_link1 hinge
        self.a2 = 0.299774        # arm_link1 length
        self.a3 = 0.315           # arm_link2 length
        
    def solve_ik(self, target_pos, target_rot_mat):
        # 1. Wrist Center (WC)
        wx, wy, wz = target_pos

        # 2. Joint 1: Base
        q1 = np.arctan2(wy, wx)

        # 3. Joints 2 & 3: Planar 2-link
        r = np.sqrt(wx**2 + wy**2)
        s = wz - self.d1
        
        # Law of Cosines
        D = (r**2 + s**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
        D = np.clip(D, -1.0, 1.0)
        q3 = -np.arccos(D) # Elbow Up configuration

        alpha = np.arctan2(s, r)
        beta = np.arctan2(self.a3 * np.sin(q3), self.a2 + self.a3 * np.cos(q3))
        q2 = alpha - beta

        # 4. Joints 4, 5, 6: Spherical Wrist
        # R03 Calculation (adjusting for XML orientation offsets)
        R01 = self._rot_z(q1)
        R12 = self._rot_y(q2 + np.pi/2) 
        R23 = self._rot_y(q3)
        R03 = R01 @ R12 @ R23
        
        R36 = R03.T @ target_rot_mat

        # ZYZ Euler extraction for the spherical wrist
        q5 = np.arccos(np.clip(R36[2, 2], -1.0, 1.0))
        if np.abs(np.sin(q5)) > 1e-4:
            q4 = np.arctan2(R36[1, 2], R36[0, 2])
            q6 = np.arctan2(R36[2, 1], -R36[2, 0])
        else:
            q4 = 0
            q6 = np.arctan2(-R36[0, 1], R36[0, 0])

        return np.rad2deg([q1, q2, q3, q4, q5, q6])

    def _rot_z(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [0,               0,            1]])

    def _rot_y(self, theta):
        return np.array([[ np.cos(theta), 0, np.sin(theta)],
                         [ 0,            1, 0           ],
                         [-np.sin(theta), 0, np.cos(theta)]])

def run_ik_test(name, target_pos, target_euler, model, data, ik_solver):
    # 1. Convert target euler to rotation matrix
    # First Euler -> Quat
    target_quat = np.zeros(4)
    mujoco.mju_euler2Quat(target_quat, np.deg2rad(target_euler), 'xyz')
    
    # Then Quat -> Mat
    res_e = np.zeros(9)
    mujoco.mju_quat2Mat(res_e, target_quat)
    target_mat = res_e.reshape(3, 3)

    # 2. Solve Analytical IK
    q_solved = ik_solver.solve_ik(target_pos, target_mat)

    # 3. Apply to simulation
    data.qpos[:6] = np.deg2rad(q_solved)
    mujoco.mj_forward(model, data)

    # Get actual position of wrist2
    sim_pos = data.body("wrist2").xpos
    error_mm = np.linalg.norm(sim_pos - target_pos) * 1000

    status = "PASS" if error_mm < 50.0 else "FAIL"
    
    print(f"{name:<15} | Target: {str(target_pos):<20} | Err: {error_mm:8.2f}mm | {status}")
    return error_mm
def main():
    # Load your XML
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(curr_dir, "assets", "iris_with_axis.xml")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ik_solver = IRISAnalyticalIK()

    print("\n" + "="*70)
    print(f"{'IK ACCURACY VERIFICATION (Analytical vs Simulation)':^70}")
    print("="*70)

    # Test Case 1: Reach forward
    run_ik_test("Reach Forward", [0.4, 0.0, 0.4], [0, 90, 0], model, data, ik_solver)

    # Test Case 2: Reach Side
    run_ik_test("Reach Side", [0.0, 0.4, 0.4], [0, 90, 90], model, data, ik_solver)

    # Test Case 3: High reach
    run_ik_test("High Reach", [0.2, 0.1, 0.7], [0, 0, 0], model, data, ik_solver)

    print("="*70 + "\n")
    print("Note: Errors of 5-45mm are expected due to the 'Spherical Wrist' assumption")
    print("while the XML contains minor joint offsets (6mm and 42mm).")

if __name__ == "__main__":
    main()