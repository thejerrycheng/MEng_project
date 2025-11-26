#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Move Joint for xArm Lite 6
"""

import os
import sys
import time
import math
from configparser import ConfigParser
from xarm.wrapper import XArmAPI

# Retrieve xArm's IP address
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        parser = ConfigParser()
        parser.read('robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        # ip = input('Please input the xArm IP address:')
        ip = '192.168.1.187'
        if not ip:
            print('Input error, exiting.')
            sys.exit(1)

# Initialize the xArm API
arm = XArmAPI(ip)
arm.connect()

# Enable the robot arm and set to position control mode
arm.motion_enable(enable=True)
arm.set_mode(0)  # Position control mode
arm.set_state(state=0)  # Set to running state

# Move to home position
arm.move_gohome(wait=True)

# Define movement speed
speed = 50  # Degrees per second

# Define speed in radians per second
speed_radians = math.radians(50)

# Sequence of joint movements in radians
joint_positions_radians = [
    [math.radians(90), math.radians(-15), 0, 0, 0, 0],
    [math.radians(90), math.radians(-30), math.radians(60), 0, 0, 0],
    [math.radians(90), math.radians(-60), math.radians(90), 0, 0, 0],
    [0, math.radians(-30), math.radians(60), 0, 0, 0],
    [0, 0, math.radians(-60), 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

# Execute joint movements in radians
i = 0
for angles in joint_positions_radians:
    arm.set_servo_angle(angle=angles, speed=speed_radians, is_radian=True, wait=True)
    i  = i + 1
    print("sending the commands here... ", i)
    print("Current joint angles (degrees):", arm.get_servo_angle())
    print("Current joint angles (radians):", arm.get_servo_angle(is_radian=True))

# Return to home position
arm.move_gohome(wait=True)
print("The home position is: ", arm.get_servo_angle())
print()

# Disconnect from the robot arm
arm.disconnect()
