#!/usr/bin/env python3
import os
import numpy as np
import rospy
import mujoco
import mujoco.viewer
import time
import curses
import csv
from sensor_msgs.msg import JointState

# ==================================================
# Paths
# ==================================================
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")

# ==================================================
# Hyperparameters (Edit these!)
# ==================================================
# 1. Circle Geometry
CIRCLE_CENTER = [0, 0.3, 0.5] 
CIRCLE_RADIUS = 0.1            
# [0, 1, 0] = Vertical (Ferris Wheel), [0, 0, 1] = Horizontal (Plate)
CIRCLE_NORMAL = [0, 1, 0]       

# 2. Timing
CIRCLE_SPEED  = 0.5   
TOTAL_LOOPS   = 2     
APPROACH_DURATION = 4.0 

# ==================================================
# System Config
# ==================================================
TOPIC_SUB = "/joint_states_calibrated"
TOPIC_PUB = "/joint_commands_calibrated"
JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6
RATE_HZ = 200

JOINT_LIMITS_DEG = [(-170,170),(-170,170),(-150,150),(-180,180),(-100,100),(-360,360)]
EE_BODY_NAME = "ee_mount"
DAMPING = 1e-3
IK_STEP_SCALE = 0.8
LPF_ALPHA = 0.15 

# ==================================================
# Math Helpers
# ==================================================
def clamp_qpos(qpos):
    limit_min = np.deg2rad([l[0] for l in JOINT_LIMITS_DEG])
    limit_max = np.deg2rad([l[1] for l in JOINT_LIMITS_DEG])
    return np.clip(qpos, limit_min, limit_max)

def mat2euler(mat):
    sy = np.sqrt(mat[0,0]**2 + mat[1,0]**2)
    if sy > 1e-6:
        roll, pitch, yaw = np.arctan2(mat[2,1], mat[2,2]), np.arctan2(-mat[2,0], sy), np.arctan2(mat[1,0], mat[0,0])
    else:
        roll, pitch, yaw = 0.0, np.arctan2(-mat[2,0], sy), 0.0
    return np.array([roll, pitch, yaw])

def ease_in_out(t):
    return (1 - np.cos(t * np.pi)) / 2.0

def get_rotation_from_normal(normal):
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1])
    
    if np.allclose(normal, z_axis): return np.eye(3)
    if np.allclose(normal, -z_axis): return np.diag([1, -1, -1])

    axis = np.cross(z_axis, normal)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(z_axis, normal))
    
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

# ==================================================
# Class
# ==================================================
class MujocoCircleTask:
    def __init__(self):
        rospy.init_node("mujoco_circle_task")
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1)
        rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)

        if not os.path.exists(XML_PATH): raise FileNotFoundError(f"XML not found: {XML_PATH}")
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

        self.center = np.array(CIRCLE_CENTER)
        self.radius = CIRCLE_RADIUS
        self.R_plane = get_rotation_from_normal(CIRCLE_NORMAL)

        # States
        self.initialized = False
        self.phase = "WAIT_FOR_START" 
        
        self.start_sim_time = 0.0
        self.circle_start_time = 0.0
        
        self.pos_initial = np.zeros(3) 
        self.target_rpy = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.q_filtered = np.zeros(NUM_JOINTS)
        
        self.pos_circle_start = np.zeros(3)

        # --- Logging Setup ---
        self.csv_filename = "circle_path_tracking_error.csv"
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV Header
        header = [
            "timestamp", "phase", 
            "q1", "q2", "q3", "q4", "q5", "q6",
            "ee_x_act", "ee_y_act", "ee_z_act",
            "ee_x_des", "ee_y_des", "ee_z_des",
            "err_x", "err_y", "err_z"
        ]
        self.csv_writer.writerow(header)
        rospy.loginfo(f"Data logging initialized: {self.csv_filename}")

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

    def calculate_circle_point(self, theta):
        local_point = np.array([self.radius * np.cos(theta), self.radius * np.sin(theta), 0.0])
        return self.center + (self.R_plane @ local_point)

    def state_cb(self, msg):
        if self.initialized: return

        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        start_q = np.zeros(NUM_JOINTS)
        try:
            for i, jname in enumerate(JOINT_NAMES):
                start_q[i] = msg.position[name_to_idx[jname]]
        except KeyError: return 

        # Sync Sim
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(start_q)
        mujoco.mj_forward(self.model, self.data)
        self.q_filtered = self.data.qpos[:NUM_JOINTS].copy()

        # Capture Initial State
        self.pos_initial = self.data.body(self.ee_id).xpos.copy()
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3,3)
        self.target_rpy = mat2euler(curr_mat)

        self.initialized = True
        self.pos_circle_start = self.calculate_circle_point(0.0)
        
        # Start holding position immediately
        self.target_pos = self.pos_initial.copy()

    def draw_circle_visuals(self, viewer):
        if not self.initialized: return
        
        steps = 50
        for i in range(steps):
            theta = (i / float(steps)) * 2 * np.pi
            pt = self.calculate_circle_point(theta)
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=pt,
                mat=np.eye(3).flatten(),
                rgba=[0, 1, 0, 0.5] 
            )
        
        # Start Point (Red)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[steps],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            pos=self.pos_circle_start,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 1.0] 
        )
        viewer.user_scn.ngeom = steps + 1

    def update_logic(self, stdscr):
        # 1. Handle Input
        key = stdscr.getch()
        if key == ord('q'): return False # Quit

        now = time.time()

        # 2. State Machine
        if self.phase == "WAIT_FOR_START":
            # Hold Position
            self.target_pos = self.pos_initial
            
            # Status
            stdscr.addstr(2,0, "Status: VISUALIZING. Press [ENTER] to start.")
            stdscr.addstr(3,0, f"Init Pos: {np.round(self.pos_initial,3)}")
            stdscr.addstr(4,0, f"Next Pos: {np.round(self.pos_circle_start,3)}")

            # Check for Enter key (10 or 13)
            if key == 10 or key == 13:
                self.phase = "APPROACH"
                self.start_sim_time = now
                stdscr.addstr(2,0, "Status: MOVING (Approach Phase)             ")

        elif self.phase == "APPROACH":
            t_raw = (now - self.start_sim_time) / APPROACH_DURATION
            stdscr.addstr(2,0, f"Status: APPROACHING ({int(t_raw*100)}%)")
            
            if t_raw >= 1.0:
                t_raw = 1.0
                self.phase = "CIRCLE"
                self.circle_start_time = now
            
            ratio = ease_in_out(t_raw)
            self.target_pos = (1 - ratio) * self.pos_initial + ratio * self.pos_circle_start

        elif self.phase == "CIRCLE":
            t_circle = now - self.circle_start_time
            theta = CIRCLE_SPEED * t_circle
            
            loop_count = theta / (2*np.pi)
            stdscr.addstr(2,0, f"Status: CIRCLING (Loops: {loop_count:.1f}/{TOTAL_LOOPS})")
            
            if theta >= TOTAL_LOOPS * 2 * np.pi:
                theta = TOTAL_LOOPS * 2 * np.pi
                self.phase = "FINISHED"
            
            self.target_pos = self.calculate_circle_point(theta)
            
        elif self.phase == "FINISHED":
            stdscr.addstr(2,0, "Status: FINISHED. Holding Position.")
            pass

        return True

    def solve_ik(self):
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, self.target_rpy, 'xyz')
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, quat)
        target_mat = target_mat.reshape(3,3)

        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3,3)

        pos_err = self.target_pos - curr_pos
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        jacp = np.zeros((3,self.model.nv))
        jacr = np.zeros((3,self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp,jacr])[:,:NUM_JOINTS]
        err = np.concatenate([pos_err, rot_err])

        dq = J.T @ np.linalg.solve(J@J.T + DAMPING*np.eye(6), err)
        
        raw_q = clamp_qpos(self.data.qpos[:NUM_JOINTS] + dq * IK_STEP_SCALE)
        self.q_filtered = (LPF_ALPHA * raw_q) + ((1.0 - LPF_ALPHA) * self.q_filtered)
        self.data.qpos[:NUM_JOINTS] = self.q_filtered

    def log_state(self):
        """Logs current state to CSV"""
        if not self.initialized: return

        # 1. Gather Data
        t_curr = time.time() - self.start_sim_time if self.start_sim_time > 0 else 0
        
        # Robot State
        q_curr = self.data.qpos[:NUM_JOINTS]
        ee_act = self.data.body(self.ee_id).xpos.copy()
        
        # Desired State
        ee_des = self.target_pos
        
        # Error
        ee_err = ee_des - ee_act

        # 2. Construct Row
        # timestamp, phase
        row = [f"{t_curr:.4f}", self.phase]
        # q1..q6
        row.extend([f"{x:.4f}" for x in q_curr])
        # x_act, y_act, z_act
        row.extend([f"{x:.4f}" for x in ee_act])
        # x_des, y_des, z_des
        row.extend([f"{x:.4f}" for x in ee_des])
        # err_x, err_y, err_z
        row.extend([f"{x:.6f}" for x in ee_err])

        # 3. Write
        self.csv_writer.writerow(row)

    def run(self, stdscr):
        # Setup Curses
        stdscr.nodelay(True)
        stdscr.clear()
        stdscr.addstr(0,0,"MuJoCo Circle Task + CSV Logging | Press 'q' to quit")
        stdscr.addstr(2,0,"Status: Waiting for ROS...")
        stdscr.refresh()

        rate = rospy.Rate(RATE_HZ)

        # Wait for ROS connection
        while not rospy.is_shutdown() and not self.initialized:
            rate.sleep()

        # Main Loop
        with mujoco.viewer.launch_passive(self.model,self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                
                # 1. Update Logic (State Machine + Input)
                running = self.update_logic(stdscr)
                if not running: break

                # 2. Physics & IK
                self.draw_circle_visuals(viewer)
                self.solve_ik()
                mujoco.mj_step(self.model, self.data)
                
                # 3. Log Data
                self.log_state()

                # 4. Publish
                msg = JointState()
                msg.header.stamp = rospy.Time.now()
                msg.name = JOINT_NAMES
                msg.position = self.q_filtered.tolist()
                self.cmd_pub.publish(msg)

                # 5. Refresh UI
                viewer.sync()
                stdscr.refresh()
                rate.sleep()

if __name__ == "__main__":
    node = MujocoCircleTask()
    try:
        curses.wrapper(node.run)
    finally:
        # Ensure file is closed even on crash/exit
        if hasattr(node, 'csv_file'):
            node.csv_file.close()
            print(f"\nCSV Log saved to: {node.csv_filename}")