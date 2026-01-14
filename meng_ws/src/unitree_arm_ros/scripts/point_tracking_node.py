#!/usr/bin/env python3
import os
import sys
import yaml
import time
import numpy as np
import rospy
import mujoco
import mujoco.viewer
from sensor_msgs.msg import JointState

# ==================================================
# Configuration
# ==================================================
# Robot & Simulation Paths
MUJOCO_SIM_DIR = "/home/jerry/Desktop/MEng_project/mujoco_sim"
ASSETS_DIR = os.path.join(MUJOCO_SIM_DIR, "assets")
XML_PATH = os.path.join(ASSETS_DIR, "iris.xml")
CALIB_PATH = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config/calibration.yaml"

# ROS / Hardware Config
JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6
EE_BODY_NAME = "ee_mount"
RATE_HZ = 200  # Control loop frequency
DT = 1.0 / RATE_HZ

# Trajectory Parameters
LINEAR_VELOCITY = 0.08    # m/s
ANGULAR_VELOCITY = 0.3    # rad/s
ARRIVAL_TOLERANCE = 0.005 # 5mm tolerance to switch to next waypoint
IK_DAMPING = 1e-4

# Joint Limits (Deg) - Used for safety clamping
JOINT_LIMITS_DEG = [(-170,170),(-170,170),(-150,150),(-180,180),(-100,100),(-360,360)]

# Waypoints: [X, Y, Z, Roll, Pitch, Yaw] (in World Frame)
WAYPOINTS = [
    [0.4, 0.2, 0.4, 0.0, 0.0, 0.0],
    [0.2,  0.1, 0.5, 1.57, 0.0, 0.0],
    [0.4,  0.3, 0.3, 0.0, 1.57, 0.0],
    [0.3,  0.0, 0.6, 0.0, 0.0, 1.57],
]

# ==================================================
# Helpers
# ==================================================
def load_calibration(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)["joint_offsets"]

def clamp_qpos(qpos):
    limit_min = np.deg2rad([l[0] for l in JOINT_LIMITS_DEG])
    limit_max = np.deg2rad([l[1] for l in JOINT_LIMITS_DEG])
    return np.clip(qpos, limit_min, limit_max)

# ==================================================
# Main Node Class
# ==================================================
class TrackPointNode:
    def __init__(self):
        rospy.init_node("track_point_node")
        
        # --- ROS Setup ---
        self.cmd_pub = rospy.Publisher("/arm/command", JointState, queue_size=1)
        self.state_sub = rospy.Subscriber("/joint_states", JointState, self.state_cb, queue_size=1)
        
        # --- MuJoCo Setup ---
        if not os.path.exists(XML_PATH):
            rospy.logerr(f"XML not found at {XML_PATH}")
            sys.exit(1)
            
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

        # --- State Management ---
        self.offsets = load_calibration(CALIB_PATH)
        self.state_ready = False
        self.mission_complete = False
        
        # --- Path Planning State ---
        self.waypoint_idx = 0
        
        # "Current Target" - The virtual point we are dragging the robot towards
        # Initially None, will be set to robot's actual position on startup
        self.curr_target_pos = None 
        self.curr_target_quat = None 

        rospy.loginfo("Waiting for robot joint states to initialize...")

    def state_cb(self, msg):
        """
        Initialize the simulation EXACTLY where the real robot is.
        This runs only once at startup.
        """
        if self.state_ready:
            return

        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        q_robot = np.zeros(NUM_JOINTS)
        
        # check if all joints present
        for j in JOINT_NAMES:
            if j not in name_to_idx: return

        # Read Real Robot Joints
        for i,j in enumerate(JOINT_NAMES):
            q_robot[i] = msg.position[name_to_idx[j]]

        # Apply Calibration (Real -> Sim)
        q_mujoco = np.zeros(NUM_JOINTS)
        for i,j in enumerate(JOINT_NAMES):
            q_mujoco[i] = q_robot[i] - self.offsets.get(j, 0.0)

        # Update MuJoCo
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(q_mujoco)
        mujoco.mj_forward(self.model, self.data)

        # Initialize Virtual Target at Current EE Pose
        # This prevents the robot from jumping; it will interpolate from here to WP[0]
        self.curr_target_pos = self.data.body(self.ee_id).xpos.copy()
        self.curr_target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.curr_target_quat, self.data.body(self.ee_id).xmat)

        # Send 'Hold' Command to hardware to lock servos
        self.publish_command()
        
        self.state_ready = True
        rospy.loginfo("Initialized! Starting Trajectory Tracking...")

    def publish_command(self):
        """Send Sim state + Calibration Offsets -> Real Robot"""
        q_cmd = np.zeros(NUM_JOINTS)
        for i,j in enumerate(JOINT_NAMES):
            q_cmd[i] = self.data.qpos[i] + self.offsets.get(j, 0.0)

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_cmd.tolist()
        self.cmd_pub.publish(msg)

    def solve_ik(self):
        """Jacobian Damped Least Squares to follow self.curr_target_pos/quat"""
        curr_pos = self.data.body(self.ee_id).xpos
        curr_mat = self.data.body(self.ee_id).xmat.reshape(3, 3)
        
        # Position Error
        pos_err = self.curr_target_pos - curr_pos
        
        # Orientation Error
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, self.curr_target_quat)
        target_mat = target_mat.reshape(3,3)
        
        rot_err_mat = target_mat @ curr_mat.T
        rot_err_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rot_err_quat, rot_err_mat.flatten())
        
        # Rotation vector (axis * angle)
        rot_err_vec = rot_err_quat[1:] * np.sign(rot_err_quat[0])

        # Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        J = np.vstack([jacp, jacr])[:, :NUM_JOINTS]
        
        error = np.concatenate([pos_err, rot_err_vec])
        
        # DLS Solve
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), error)
        
        # Update Simulation State (Full step 1.0 because we are tracking a smooth interpolation)
        self.data.qpos[:NUM_JOINTS] += dq
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(self.data.qpos[:NUM_JOINTS])

    def update_trajectory(self):
        """Interpolates virtual target towards the current waypoint"""
        if self.mission_complete or self.waypoint_idx >= len(WAYPOINTS):
            self.mission_complete = True
            return

        # Get Goal Waypoint
        wp = WAYPOINTS[self.waypoint_idx]
        dest_pos = np.array(wp[:3])
        dest_quat = np.zeros(4)
        mujoco.mju_euler2Quat(dest_quat, wp[3:], 'xyz')

        # 1. Linear Interpolation
        pos_diff = dest_pos - self.curr_target_pos
        dist = np.linalg.norm(pos_diff)
        
        if dist > 0.0001:
            step = min(dist, LINEAR_VELOCITY * DT)
            self.curr_target_pos += (pos_diff / dist) * step
            
        # 2. Spherical Interpolation (Angular Velocity limit)
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, self.curr_target_quat)
        q_diff = np.zeros(4)
        mujoco.mju_mulQuat(q_diff, dest_quat, q_inv)
        
        vel_vec = np.zeros(3)
        mujoco.mju_quat2Vel(vel_vec, q_diff, 1.0)
        vel_norm = np.linalg.norm(vel_vec)
        
        if vel_norm > 1e-4:
            # Integrate with constant angular velocity cap
            step_vel = (vel_vec / vel_norm) * min(vel_norm, ANGULAR_VELOCITY)
            mujoco.mju_quatIntegrate(self.curr_target_quat, step_vel, DT)
            # Normalize to prevent drift
            mujoco.mju_normalize4(self.curr_target_quat)

        # 3. Check Arrival
        if dist < ARRIVAL_TOLERANCE and vel_norm < 0.1:
            rospy.loginfo(f"Reached Waypoint {self.waypoint_idx}")
            self.waypoint_idx += 1

    def run(self):
        rate = rospy.Rate(RATE_HZ)
        
        # Launch Visualizer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                
                # Wait for initial robot state
                if not self.state_ready:
                    rate.sleep()
                    continue

                # Control Logic
                self.update_trajectory()
                self.solve_ik() # Move Sim Robot
                mujoco.mj_step(self.model, self.data) # Physics Step
                self.publish_command() # Send to Real Robot

                # Visualization Sync
                viewer.sync()
                
                # Dashboard
                if self.waypoint_idx < len(WAYPOINTS):
                    curr_real_pos = self.data.body(self.ee_id).xpos
                    goal_pos = WAYPOINTS[self.waypoint_idx][:3]
                    dist_remain = np.linalg.norm(curr_real_pos - goal_pos)
                    
                    sys.stdout.write("\r\033[K") # Clear line
                    print(f"Tracking WP {self.waypoint_idx}/{len(WAYPOINTS)} | Dist: {dist_remain:.4f}m", end="", flush=True)
                elif self.mission_complete:
                     sys.stdout.write("\r\033[K")
                     print("Mission Complete.", end="", flush=True)

                rate.sleep()

if __name__ == "__main__":
    try:
        node = TrackPointNode()
        node.run()
    except rospy.ROSInterruptException:
        pass