#!/usr/bin/env python3
import os
import sys
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

JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
NUM_JOINTS = 6
EE_BODY_NAME = "ee_mount"
RATE_HZ = 200  
DT = 1.0 / RATE_HZ

# Trajectory Settings
LINEAR_VELOCITY = 0.08    # m/s
ANGULAR_VELOCITY = 0.3    # rad/s
ARRIVAL_TOLERANCE = 0.01  # 1cm
IK_DAMPING = 1e-4

# Joint Limits
JOINT_LIMITS_DEG = [(-170,170),(-170,170),(-150,150),(-180,180),(-100,100),(-360,360)]

# Waypoints [X, Y, Z, Roll, Pitch, Yaw]
WAYPOINTS = [
    [0.0, 0.3, 0.5, 0.0, 0.0, 0.0],       # Start High
    [-0.2, 0.3, 0.4, 1.57, 0.0, 0.0],     # Right
    [0.2,  0.3, 0.4, 0.0, 0.7, 0.0],      # Left + Pitch
    [0.0,  0.4, 0.35, 0.0, 0.0, 0.0],     # Forward Low
]

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
class TrackPointNode:
    def __init__(self):
        rospy.init_node("track_point_node")
        
        # ROS
        self.cmd_pub = rospy.Publisher(TOPIC_PUB, JointState, queue_size=1)
        self.state_sub = rospy.Subscriber(TOPIC_SUB, JointState, self.state_cb, queue_size=1)
        
        # MuJoCo
        if not os.path.exists(XML_PATH):
            rospy.logerr(f"XML not found at {XML_PATH}")
            sys.exit(1)
            
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data  = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

        # State
        self.state_ready = False
        self.mission_complete = False
        self.waypoint_idx = 0
        
        # Targets
        self.curr_target_pos = None 
        self.curr_target_quat = None 

        rospy.loginfo("Waiting for /joint_states_calibrated...")

    def state_cb(self, msg):
        """Sync Sim to Reality ONCE at startup"""
        if self.state_ready: return

        name_to_idx = {n:i for i,n in enumerate(msg.name)}
        q_start = np.zeros(NUM_JOINTS)
        
        for i,j in enumerate(JOINT_NAMES):
            if j not in name_to_idx: return
            q_start[i] = msg.position[name_to_idx[j]]

        # Set Sim State
        self.data.qpos[:NUM_JOINTS] = clamp_qpos(q_start)
        mujoco.mj_forward(self.model, self.data)

        # Initialize Virtual Target at Current Real EE
        self.curr_target_pos = self.data.body(self.ee_id).xpos.copy()
        self.curr_target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.curr_target_quat, self.data.body(self.ee_id).xmat)

        self.publish_command()
        self.state_ready = True
        rospy.loginfo(f"Initialized. Moving to WP 1/{len(WAYPOINTS)}...")

    def publish_command(self):
        q_cmd = self.data.qpos[:NUM_JOINTS]
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q_cmd.tolist()
        self.cmd_pub.publish(msg)

    def solve_ik(self):
        """Standard Jacobian IK to follow curr_target_pos"""
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
        """Interpolate Green Sphere towards Red Sphere"""
        if self.mission_complete or self.waypoint_idx >= len(WAYPOINTS):
            self.mission_complete = True
            return

        wp = WAYPOINTS[self.waypoint_idx]
        dest_pos = np.array(wp[:3])
        dest_quat = np.zeros(4)
        mujoco.mju_euler2Quat(dest_quat, wp[3:], 'xyz')

        # Linear Interp
        pos_diff = dest_pos - self.curr_target_pos
        dist = np.linalg.norm(pos_diff)
        
        if dist > 0.0001:
            step = min(dist, LINEAR_VELOCITY * DT)
            self.curr_target_pos += (pos_diff / dist) * step
            
        # Spherical Interp
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, self.curr_target_quat)
        q_diff = np.zeros(4)
        mujoco.mju_mulQuat(q_diff, dest_quat, q_inv)
        
        vel_vec = np.zeros(3)
        mujoco.mju_quat2Vel(vel_vec, q_diff, 1.0)
        vel_norm = np.linalg.norm(vel_vec)
        
        if vel_norm > 1e-4:
            step_vel = (vel_vec / vel_norm) * min(vel_norm, ANGULAR_VELOCITY)
            mujoco.mju_quatIntegrate(self.curr_target_quat, step_vel, DT)
            mujoco.mju_normalize4(self.curr_target_quat)

        if dist < ARRIVAL_TOLERANCE and vel_norm < 0.1:
            self.waypoint_idx += 1

    def run(self):
        rate = rospy.Rate(RATE_HZ)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while not rospy.is_shutdown() and viewer.is_running():
                if not self.state_ready:
                    rate.sleep()
                    continue

                if self.mission_complete:
                    self.publish_command()
                else:
                    self.update_trajectory()
                    self.solve_ik()
                    mujoco.mj_step(self.model, self.data)
                    self.publish_command()

                # --- VISUALIZATION MARKERS ---
                # 1. Reset Scene Markers
                viewer.user_scn.ngeom = 0
                
                # 2. Draw Active Goal Waypoint (RED)
                if self.waypoint_idx < len(WAYPOINTS):
                    wp = WAYPOINTS[self.waypoint_idx]
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0],
                        pos=np.array(wp[:3]),
                        mat=np.eye(3).flatten(),
                        rgba=[1, 0, 0, 0.6] # Red (Current Goal)
                    )
                    viewer.user_scn.ngeom += 1

                # 3. Draw Active Virtual Target (GREEN)
                if self.curr_target_pos is not None:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.015, 0, 0],
                        pos=self.curr_target_pos,
                        mat=np.eye(3).flatten(),
                        rgba=[0, 1, 0, 1] # Green (Current Path)
                    )
                    viewer.user_scn.ngeom += 1
                # -----------------------------

                viewer.sync()
                
                # Dashboard
                if not self.mission_complete:
                    curr_real_pos = self.data.body(self.ee_id).xpos
                    goal_pos = WAYPOINTS[self.waypoint_idx][:3]
                    dist_remain = np.linalg.norm(curr_real_pos - goal_pos)
                    sys.stdout.write("\r\033[K") 
                    print(f"WP {self.waypoint_idx+1}/{len(WAYPOINTS)} | Dist: {dist_remain:.3f}m", end="", flush=True)

                rate.sleep()

if __name__ == "__main__":
    try:
        TrackPointNode().run()
    except rospy.ROSInterruptException:
        pass