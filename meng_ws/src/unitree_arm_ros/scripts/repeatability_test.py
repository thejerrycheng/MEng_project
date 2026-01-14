#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
import rospy
import mujoco
import mujoco.viewer
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")

# Topics
TOPIC_SUB = "/joint_states_calibrated"
TOPIC_PUB = "/joint_commands_calibrated"

# Logging
LOG_FILE = "repeatability_log.csv"

# Robot Config
JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6
EE_BODY_NAME = "ee_mount"
RATE_HZ = 200  
DT = 1.0 / RATE_HZ

# Trajectory Settings
LINEAR_VELOCITY = 0.25    # m/s
ANGULAR_VELOCITY = 0.5    # rad/s
ARRIVAL_TOLERANCE = 0.005 # 5mm
IK_DAMPING = 1e-4

# Joint Limits
JOINT_LIMITS_DEG = [(-170,170),(-170,170),(-150,150),(-180,180),(-100,100),(-360,360)]

# --- TEST POINTS [X, Y, Z, Roll, Pitch, Yaw] ---
# HOME is captured dynamically at startup
POINT_A    = [0.3, -0.2, 0.3, 1.57, 0.0, 0.0] # Right
POINT_B    = [0.3,  0.2, 0.3, 0.0, 0.7, 0.0]  # Left

NUM_CYCLES = 10

# ==================================================
# Helpers
# ==================================================
def clamp_qpos(qpos):
    limit_min = np.deg2rad([l[0] for l in JOINT_LIMITS_DEG])
    limit_max = np.deg2rad([l[1] for l in JOINT_LIMITS_DEG])
    return np.clip(qpos, limit_min, limit_max)

# ==================================================
# Main Node
# ==================================================
class RepeatabilityTestNode:
    def __init__(self):
        rospy.init_node("repeatability_test_node")
        
        # ROS
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1)
        self.state_sub = rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)
        
        # MuJoCo - Simulation (Control)
        if not os.path.exists(XML_PATH):
            rospy.logerr(f"XML not found at {XML_PATH}")
            sys.exit(1)
            
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        
        # MuJoCo - Logging (Forward Kinematics)
        # We use a separate data structure so we don't mess up the control loop
        self.data_log = mujoco.MjData(self.model)
        
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

        # State
        self.state_ready = False
        self.curr_real_joints = np.zeros(NUM_JOINTS) 
        
        # Targets
        self.curr_target_pos = None 
        self.curr_target_quat = None 
        self.goal_pos = None
        self.goal_quat = None
        
        # Dynamic Home Point
        self.point_home_pos = None
        self.point_home_quat = None

        # Test Logic
        self.state_machine = "INIT_WAIT" 
        self.cycle_count = 0
        self.target_name = "HOME"

        # CSV Setup
        self.log_file = open(LOG_FILE, 'w', newline='')
        self.writer = csv.writer(self.log_file)
        
        # Updated Header
        header = [
            "timestamp", "cycle", "target_name", 
            "pos_x", "pos_y", "pos_z",           # Actual EE Position
            "j1", "j2", "j3", "j4", "j5", "j6"   # Actual Joints
        ]
        self.writer.writerow(header)
        rospy.loginfo(f"Logging data to {LOG_FILE}")

        rospy.loginfo("Waiting for robot state...")

    def state_cb(self, msg):
        """Sync Sim to Reality ONCE at startup to define HOME"""
        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        q_current = np.zeros(NUM_JOINTS)
        
        for i,j in enumerate(JOINT_NAMES):
            if j not in name_to_idx: return
            q_current[i] = msg.position[name_to_idx[j]]

        # Always update buffer
        self.curr_real_joints = q_current

        # One-time initialization
        if not self.state_ready:
            # 1. Update Sim to Real
            self.data.qpos[:NUM_JOINTS] = clamp_qpos(q_current)
            mujoco.mj_forward(self.model, self.data)

            # 2. Capture Current Pose as HOME
            curr_pos = self.data.body(self.ee_id).xpos.copy()
            curr_mat = self.data.body(self.ee_id).xmat.reshape(3,3)
            
            # Store HOME permanently
            self.point_home_pos = curr_pos
            self.point_home_quat = np.zeros(4)
            mujoco.mju_mat2Quat(self.point_home_quat, self.data.body(self.ee_id).xmat)
            
            # 3. Initialize Targets
            self.curr_target_pos = curr_pos.copy()
            self.curr_target_quat = self.point_home_quat.copy()
            
            # Set initial goal to Point A
            rospy.loginfo(f"HOME Captured at: {np.round(curr_pos, 3)}")
            rospy.loginfo("Starting Cycles...")
            
            self.state_machine = "CYCLE_A"
            self.set_goal(POINT_A, "Point A")
            
            self.publish_command()
            self.state_ready = True

    def set_goal(self, wp, name):
        """Set goal from fixed list or dynamic home"""
        self.target_name = name
        
        if name == "HOME":
            self.goal_pos = self.point_home_pos.copy()
            self.goal_quat = self.point_home_quat.copy()
        else:
            self.goal_pos = np.array(wp[:3])
            self.goal_quat = np.zeros(4)
            mujoco.mju_euler2Quat(self.goal_quat, wp[3:], 'xyz')

    def log_data(self):
        """
        Calculate Forward Kinematics on ACTUAL joints
        and write to CSV.
        """
        # 1. Update Logging Data Structure
        self.data_log.qpos[:NUM_JOINTS] = self.curr_real_joints
        
        # 2. Compute Forward Kinematics
        mujoco.mj_kinematics(self.model, self.data_log)
        
        # 3. Read EE Position
        ee_pos = self.data_log.body(self.ee_id).xpos
        
        # 4. Write Row
        row = [
            rospy.Time.now().to_sec(), 
            self.cycle_count, 
            self.target_name,
            ee_pos[0], ee_pos[1], ee_pos[2] # Actual X, Y, Z
        ]
        row.extend(self.curr_real_joints.tolist())
        
        self.writer.writerow(row)
        self.log_file.flush() 

    def publish_command(self):
        q_cmd = self.data.qpos[:NUM_JOINTS]
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_cmd.tolist()
        self.cmd_pub.publish(msg)

    def solve_ik(self):
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        pos_err = self.curr_target_pos - curr_pos
        
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, self.curr_target_quat)
        target_mat = target_mat.reshape(3,3)
        
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :NUM_JOINTS]
        
        error = np.concatenate([pos_err, rot_err_vec])
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), error)
        
        self.data.qpos[:NUM_JOINTS] += dq
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(self.data.qpos[:NUM_JOINTS])

    def update_trajectory(self):
        if self.state_machine == "DONE" or self.state_machine == "INIT_WAIT":
            return

        # 1. Interp Virtual Target -> Goal
        pos_diff = self.goal_pos - self.curr_target_pos
        dist = np.linalg.norm(pos_diff)
        
        if dist > 0.0001:
            step = min(dist, LINEAR_VELOCITY * DT)
            self.curr_target_pos += (pos_diff / dist) * step
            
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, self.curr_target_quat)
        q_diff = np.zeros(4)
        mujoco.mju_mulQuat(q_diff, self.goal_quat, q_inv)
        
        vel_vec = np.zeros(3)
        mujoco.mju_quat2Vel(vel_vec, q_diff, 1.0)
        vel_norm = np.linalg.norm(vel_vec)
        
        if vel_norm > 1e-4:
            step_vel = (vel_vec / vel_norm) * min(vel_norm, ANGULAR_VELOCITY)
            mujoco.mju_quatIntegrate(self.curr_target_quat, step_vel, DT)
            mujoco.mju_normalize4(self.curr_target_quat)

        # 2. Check Arrival
        if dist < ARRIVAL_TOLERANCE and vel_norm < 0.1:
            
            # --- STATE MACHINE LOGIC ---
            if self.state_machine == "CYCLE_A":
                self.log_data() 
                rospy.sleep(0.2)
                self.state_machine = "CYCLE_B"
                self.set_goal(POINT_B, "Point B")

            elif self.state_machine == "CYCLE_B":
                self.log_data() 
                self.cycle_count += 1
                rospy.loginfo(f"Completed Cycle {self.cycle_count}/{NUM_CYCLES}")
                
                if self.cycle_count >= NUM_CYCLES:
                    self.state_machine = "FINISH_HOME"
                    self.set_goal(None, "HOME")
                else:
                    rospy.sleep(0.2)
                    self.state_machine = "CYCLE_A"
                    self.set_goal(POINT_A, "Point A")

            elif self.state_machine == "FINISH_HOME":
                rospy.loginfo("Test Complete. Returned to Start.")
                self.log_file.close()
                self.state_machine = "DONE"

    def run(self):
        rate = rospy.Rate(RATE_HZ)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                if not self.state_ready:
                    rate.sleep()
                    continue

                if self.state_machine == "DONE":
                    self.publish_command() 
                else:
                    self.update_trajectory()
                    self.solve_ik()
                    mujoco.mj_step(self.model, self.data)
                    self.publish_command()

                # --- VISUALIZATION ---
                viewer.user_scn.ngeom = 0
                
                # Active Goal (RED)
                if self.goal_pos is not None:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0],
                        pos=self.goal_pos,
                        mat=np.eye(3).flatten(),
                        rgba=[1, 0, 0, 0.6]
                    )
                    viewer.user_scn.ngeom += 1

                # Virtual Path (GREEN)
                if self.curr_target_pos is not None:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.015, 0, 0],
                        pos=self.curr_target_pos,
                        mat=np.eye(3).flatten(),
                        rgba=[0, 1, 0, 1]
                    )
                    viewer.user_scn.ngeom += 1
                
                viewer.sync()
                rate.sleep()

if __name__ == "__main__":
    try:
        RepeatabilityTestNode().run()
    except rospy.ROSInterruptException:
        pass