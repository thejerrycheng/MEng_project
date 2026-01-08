#!/usr/bin/env python3
import os
import yaml
import rospy
import threading
from sensor_msgs.msg import JointState

# ==================================================
# Output path
# ==================================================
CALIB_DIR = "/home/jerry/Desktop/MEng_project/meng_ws/src/unitree_arm_ros/config"
CALIB_PATH = os.path.join(CALIB_DIR, "calibration.yaml")

# Joint names expected from /joint_states
REAL_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",   # wrist motor A
    "joint_6",   # wrist motor B
]

# ==================================================
class HomeCalibrationNode:
    def __init__(self):
        rospy.init_node("home_calibration_node")

        self.latest_joint_state = None
        self.lock = threading.Lock()

        self.sub = rospy.Subscriber(
            "/joint_states",
            JointState,
            self.joint_state_callback,
            queue_size=1
        )

        rospy.loginfo("==============================================")
        rospy.loginfo(" HOME CALIBRATION NODE ")
        rospy.loginfo("==============================================")
        rospy.loginfo("1) Manually move the robot to HOME position")
        rospy.loginfo("   → Arm fully extended upward")
        rospy.loginfo("   → Wrist pitch = 0, roll = 0")
        rospy.loginfo("2) Make sure robot is stable")
        rospy.loginfo("3) Press ENTER in this terminal to save calibration")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def joint_state_callback(self, msg):
        with self.lock:
            self.latest_joint_state = msg

    # ------------------------------------------------
    def wait_for_user(self):
        input("\n>>> Press ENTER to capture HOME calibration <<<\n")

    # ------------------------------------------------
    def save_calibration(self):
        with self.lock:
            if self.latest_joint_state is None:
                rospy.logerr("No joint state received yet. Aborting calibration.")
                return

            name_to_idx = {n: i for i, n in enumerate(self.latest_joint_state.name)}
            offsets = {}

            for joint in REAL_JOINT_NAMES:
                if joint not in name_to_idx:
                    rospy.logerr(f"Missing joint in /joint_states: {joint}")
                    return
                offsets[joint] = float(
                    self.latest_joint_state.position[name_to_idx[joint]]
                )

        calib_data = {
            "description": "Home calibration: arm fully extended upward",
            "joint_offsets": offsets
        }

        os.makedirs(CALIB_DIR, exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            yaml.safe_dump(calib_data, f)

        rospy.loginfo("==============================================")
        rospy.loginfo("Calibration saved successfully!")
        rospy.loginfo(f"File: {CALIB_PATH}")
        rospy.loginfo("Offsets (rad):")
        for k, v in offsets.items():
            rospy.loginfo(f"  {k}: {v:.6f}")
        rospy.loginfo("==============================================")

    # ------------------------------------------------
    def run(self):
        # Wait until we receive at least one joint state
        rospy.loginfo("Waiting for /joint_states...")
        while not rospy.is_shutdown():
            with self.lock:
                if self.latest_joint_state is not None:
                    break
            rospy.sleep(0.1)

        # Wait for user confirmation
        self.wait_for_user()

        # Save calibration
        self.save_calibration()

        rospy.loginfo("Calibration complete. You may now close this node.")
        rospy.signal_shutdown("Calibration finished")

# ==================================================
if __name__ == "__main__":
    HomeCalibrationNode().run()
