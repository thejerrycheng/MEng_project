import time
import math
import sys
import matplotlib
matplotlib.use('TkAgg')  # Use 'TkAgg' backend for interactive plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import animation

# ================== 1) DEFINE NEW DH PARAMETERS ==================
# All lengths are in millimeters (mm)
dh_params = {
    'a': [238.1, 19.65, 408.923, 0, 458.5, 0],
    'd': [0, 56.25, 39.875, 66.032, 0, 0],
    'alpha': [0, 0, 0, math.pi/2, math.pi/2, math.pi/2]
}

# ================== 2) INVERSE KINEMATICS FUNCTION ==================
def compute_ik_3dof(x, y, z, dh):
    """
    Compute inverse kinematics for the first three joints based on new DH parameters.
    Returns joint angles in radians.

    Parameters:
    - x, y, z: Desired end-effector position in mm.
    - dh: Dictionary containing 'a', 'd', and 'alpha' lists.

    Returns:
    - theta1, theta2, theta3: Joint angles in radians.
    """
    a = dh['a']
    d = dh['d']
    
    # Extract necessary DH parameters for the first three joints
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    d1 = d[0]
    d2 = d[1]
    d3 = d[2]
    
    # Compute the planar distance from the base to the projection of the end-effector
    r = math.sqrt(x**2 + y**2)
    s = z - d1  # Vertical distance from the base to the end-effector
    
    # Compute theta1
    theta1 = math.atan2(y, x)
    
    # Compute theta3 using the cosine law
    D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    D = max(-1.0, min(1.0, D))  # Clamp to [-1, 1] to avoid numerical errors
    theta3 = math.acos(D)
    
    # Compute theta2
    theta2 = math.atan2(s, r) - math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3))
    
    return theta1, theta2, theta3

# ================== 3) FORWARD KINEMATICS FUNCTION ==================
def forward_kinematics(joint_angles_rad, dh):
    """
    Compute forward kinematics for the first three joints based on new DH parameters.
    Returns the end-effector position (x, y, z) in mm.

    Parameters:
    - joint_angles_rad: Tuple or list containing joint angles (theta1, theta2, theta3) in radians.
    - dh: Dictionary containing 'a', 'd', and 'alpha' lists.

    Returns:
    - x, y, z: End-effector position in mm.
    """
    theta1, theta2, theta3 = joint_angles_rad
    a = dh['a']
    d = dh['d']
    alpha = dh['alpha']
    
    # Define the transformation matrix using DH parameters
    def dh_transform(theta, a_i, d_i, alpha_i):
        return np.array([
            [math.cos(theta), -math.sin(theta)*math.cos(alpha_i),  math.sin(theta)*math.sin(alpha_i), a_i*math.cos(theta)],
            [math.sin(theta),  math.cos(theta)*math.cos(alpha_i), -math.cos(theta)*math.sin(alpha_i), a_i*math.sin(theta)],
            [0,               math.sin(alpha_i),                 math.cos(alpha_i),                d_i],
            [0,               0,                                   0,                                1]
        ])
    
    # Compute individual transformation matrices
    T1 = dh_transform(theta1, a[0], d[0], alpha[0])
    T2 = dh_transform(theta2, a[1], d[1], alpha[1])
    T3 = dh_transform(theta3, a[2], d[2], alpha[2])
    
    # Compute the overall transformation matrix
    T = T1 @ T2 @ T3
    
    # Extract end-effector position
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    return x, y, z

# ================== 4) PRE-CALCULATE PATH ==================
def precompute_path(initial_position, movements, dh, steps=50):
    """
    Precompute a path of joint angles based on desired end-effector movements.

    Parameters:
    - initial_position: Tuple containing initial (x, y, z) in mm.
    - movements: List of tuples containing (dx, dy, dz) movements in mm.
    - dh: Dictionary containing 'a', 'd', and 'alpha' lists.
    - steps: Number of steps per movement segment.

    Returns:
    - joint_angle_path: List of tuples containing (theta1, theta2, theta3) in radians.
    """
    x, y, z = initial_position
    joint_angle_path = []
    
    for dx, dy, dz in movements:
        step_x = dx / steps
        step_y = dy / steps
        step_z = dz / steps
        
        for _ in range(steps):
            x += step_x
            y += step_y
            z += step_z
            theta1, theta2, theta3 = compute_ik_3dof(x, y, z, dh)
            joint_angle_path.append((theta1, theta2, theta3))
    
    return joint_angle_path

# Define the motion path as a series of (dx, dy, dz) movements
motion_path = [
    (10, 0, 0),    # Move +10 mm in X
    (-20, 0, 0),   # Move -20 mm in X
    (10, 0, 0),    # Move +10 mm in X
    (0, 10, 0),    # Move +10 mm in Y
    (0, -20, 0),   # Move -20 mm in Y
    (0, 10, 0),    # Move +10 mm in Y
    (0, 0, 10),    # Move +10 mm in Z
    (0, 0, -20),   # Move -20 mm in Z
    (0, 0, 10)      # Move +10 mm in Z
]

initial_position = (238.1, 0, 56.25)  # Starting at the base's end
joint_angle_path = precompute_path(initial_position, motion_path, dh_params)

# ================== 5) LOGGING ==================
log_time_steps = []
log_actual = [[], [], []]
log_desired = [[], [], []]
log_error = [[], [], []]
time_step = 0

def simulate_joint_movements(joint_angle_path, dh, delay=0.05):
    """
    Simulate the execution of the precomputed joint angle path and log data.

    Parameters:
    - joint_angle_path: List of tuples containing (theta1, theta2, theta3) in radians.
    - dh: Dictionary containing 'a', 'd', and 'alpha' lists.
    - delay: Time delay between each motion step in seconds.
    """
    global time_step
    current_joint_angles = [0.0, 0.0, 0.0]  # Initialize current joint angles (radians)
    
    for desired_angles in joint_angle_path:
        # Update current joint angles towards desired angles (simulate perfect movement)
        current_joint_angles = list(desired_angles)
        
        # Compute forward kinematics for actual end-effector position
        actual_x, actual_y, actual_z = forward_kinematics(current_joint_angles, dh)
        
        # Compute forward kinematics for desired end-effector position
        desired_x, desired_y, desired_z = forward_kinematics(desired_angles, dh)
        
        # Log data
        log_time_steps.append(time_step)
        for i in range(3):
            actual_deg = math.degrees(current_joint_angles[i])
            desired_deg = math.degrees(desired_angles[i])
            error = desired_deg - actual_deg
            log_actual[i].append(actual_deg)
            log_desired[i].append(desired_deg)
            log_error[i].append(error)
        
        time_step += 1
        time.sleep(delay)

# ================== 6) VISUALIZATION ==================
def plot_robot_arm(joint_angles_list, dh, interval=100):
    """
    Visualize the robot arm in 3D with frames at each joint.

    Parameters:
    - joint_angles_list: List of tuples containing joint angles (theta1, theta2, theta3) in radians.
    - dh: Dictionary containing 'a', 'd', and 'alpha' lists.
    - interval: Delay between frames in milliseconds for animation.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set plot limits based on DH parameters and motion path
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(0, 600)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3-DOF Robot Arm Visualization with Joint Frames')
    
    # Initialize lines and frames
    line, = ax.plot([], [], [], 'o-', lw=4, color='black')  # Robot arm
    frames = [ax.plot([], [], [], color=colors[i], lw=2)[0] for i, colors in enumerate(['r', 'g', 'b'])]  # Frames for each joint

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        for frame in frames:
            frame.set_data([], [])
            frame.set_3d_properties([])
        return [line] + frames

    def animate(i):
        theta1, theta2, theta3 = joint_angles_list[i]
        
        # Compute transformation matrices
        def dh_transform(theta, a_i, d_i, alpha_i):
            return np.array([
                [math.cos(theta), -math.sin(theta)*math.cos(alpha_i),  math.sin(theta)*math.sin(alpha_i), a_i*math.cos(theta)],
                [math.sin(theta),  math.cos(theta)*math.cos(alpha_i), -math.cos(theta)*math.sin(alpha_i), a_i*math.sin(theta)],
                [0,               math.sin(alpha_i),                 math.cos(alpha_i),                d_i],
                [0,               0,                                   0,                                1]
            ])
        
        T1 = dh_transform(theta1, dh['a'][0], dh['d'][0], dh['alpha'][0])
        T2 = T1 @ dh_transform(theta2, dh['a'][1], dh['d'][1], dh['alpha'][1])
        T3 = T2 @ dh_transform(theta3, dh['a'][2], dh['d'][2], dh['alpha'][2])
        
        # Extract positions of each joint
        origin = np.array([0, 0, 0, 1])
        joint1 = T1 @ origin
        joint2 = T2 @ origin
        joint3 = T3 @ origin
        
        # Extract end-effector position
        ee = joint3[:3]
        
        # Update robot arm line
        xs = [origin[0], joint1[0], joint2[0], joint3[0]]
        ys = [origin[1], joint1[1], joint2[1], joint3[1]]
        zs = [origin[2], joint1[2], joint2[2], joint3[2]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        
        # Update frames at each joint
        joints = [origin, joint1, joint2, joint3]
        for j in range(3):
            # Define the local frame axes
            x_axis = T1[:3, 0] if j == 0 else (T2[:3, 0] if j == 1 else T3[:3, 0])
            y_axis = T1[:3, 1] if j == 0 else (T2[:3, 1] if j == 1 else T3[:3, 1])
            z_axis = T1[:3, 2] if j == 0 else (T2[:3, 2] if j == 1 else T3[:3, 2])
            
            # Scale for visualization
            scale = 50  # Adjust as needed for visibility
            
            # Origin of the frame
            ox, oy, oz = joints[j]
            
            # Define end points of the frame axes
            frame_x = [ox, ox + x_axis[0]*scale]
            frame_y = [oy, oy + y_axis[1]*scale]
            frame_z = [oz, oz + z_axis[2]*scale]
            
            # Update frame lines
            frames[j].set_data([ox, frame_x[1]], [oy, frame_y[1]])
            frames[j].set_3d_properties([oz, frame_z[1]])
        
        return [line] + frames

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(joint_angles_list), interval=interval, blit=True)

    plt.show()

# ================== 7) MAIN EXECUTION ==================
def main():
    global joint_angle_path
    try:
        print("Starting pre-computed motion simulation...")
        simulate_joint_movements(joint_angle_path, dh_params)
        print("Motion simulation complete.")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
    finally:
        # Visualization after simulation
        plot_robot_arm(joint_angle_path, dh_params, interval=100)  # 100 ms between frames

    # ================== 8) PLOTTING ==================
    # Plotting Joint Angles and Errors
    print("Plotting logs (Angles and Error)...")
    
    # Ignore the first 10 timesteps to allow system stabilization
    start_index = 10
    
    # Joint Angles vs Time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax1.plot(
            log_time_steps[start_index:], 
            log_actual[i][start_index:], 
            label=f'Joint {i+1} Actual', 
            color=colors[i]
        )
        ax1.plot(
            log_time_steps[start_index:], 
            log_desired[i][start_index:], 
            label=f'Joint {i+1} Desired', 
            linestyle='--', 
            color=colors[i]
        )
    ax1.set_title('Joint Angles vs Time (Excluding First 10 Timesteps)')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Angle (degrees)')
    ax1.legend()
    ax1.grid(True)
    
    # Error vs Time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(3):
        ax2.plot(
            log_time_steps[start_index:], 
            log_error[i][start_index:], 
            label=f'Joint {i+1} Error', 
            color=colors[i]
        )
    ax2.set_title('Joint Angle Errors vs Time (Excluding First 10 Timesteps)')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Error (degrees)')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()
