% Define DH parameters for the 6-DOF robot arm
L1 = 450; % Base height (mm)
L2 = 400; % Upper arm length (mm)
L3 = 300; % Forearm length (mm)

% Define the links using DH parameters
L(1) = Link('d', L1, 'a', 0, 'alpha', pi/2, 'revolute', 'modified');
L(2) = Link('d', 0, 'a', L2, 'alpha', 0, 'revolute', 'modified');
L(3) = Link('d', 0, 'a', L3, 'alpha', 0, 'revolute', 'modified');
L(4) = Link('d', 0, 'a', 0, 'alpha', pi/2, 'revolute', 'modified');
L(5) = Link('d', 0, 'a', 0, 'alpha', -pi/2, 'revolute', 'modified');
L(6) = Link('d', 0, 'a', 0, 'alpha', 0, 'revolute', 'modified');

% Create the serial-link robot model
robot = SerialLink(L, 'name', '6-DOF Robot Arm');

% Set up the target end-effector position and orientation
target_position = [500, 200, 600]; % Target [x, y, z] in mm
target_orientation = [pi/4, pi/6, pi/3]; % Target [roll, pitch, yaw] in radians

% Inverse kinematics to calculate joint angles
x = target_position(1);
y = target_position(2);
z = target_position(3);
roll = target_orientation(1);
pitch = target_orientation(2);
yaw = target_orientation(3);

% Solve for joint angles using geometric approach
theta1 = atan2(y, x);
r = sqrt(x^2 + y^2);
s = z - L1;
D = (r^2 + s^2 - L2^2 - L3^2) / (2 * L2 * L3);
theta3 = atan2(sqrt(1 - D^2), D); % Positive elbow solution
theta2 = atan2(s, r) - atan2(L3 * sin(theta3), L2 + L3 * cos(theta3));

% Use orientation to calculate wrist angles
theta4 = roll;
theta5 = pitch;
theta6 = yaw;

% Compile the joint angles
joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6];

% Display the computed joint angles
disp('Joint Angles (radians):');
disp(joint_angles);

% Forward kinematics to verify
end_effector_pose = robot.fkine(joint_angles);
disp('End-Effector Pose:');
disp(end_effector_pose);

% Plot the robot in the target configuration
figure;
robot.plot(joint_angles, 'workspace', [-1000 1000 -1000 1000 0 1000]);
title('6-DOF Robot Arm Simulation');
grid on;
