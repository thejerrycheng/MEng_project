import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', etc.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################################
# 1) DH PARAMETERS & 3-DOF IK / FK
############################################
DH_PARAMS = {
    'd1': 0.4,   # offset along Z0
    'a2': 0.3,   # link length from J1->J2
    'a3': 0.25,  # link length from J2->J3
}

def compute_ik_3dof(x, y, z, dh):
    """
    Simple inverse kinematics for a 3-DOF arm controlling (x,y,z).
    Returns (theta1, theta2, theta3) in radians.
    """
    d1 = dh['d1']
    a2 = dh['a2']
    a3 = dh['a3']

    r = math.sqrt(x*x + y*y)
    s = z - d1

    theta1 = math.atan2(y, x)
    cosT3 = (r*r + s*s - a2*a2 - a3*a3) / (2*a2*a3)
    cosT3 = max(-1.0, min(1.0, cosT3))  # clamp
    theta3 = math.acos(cosT3)

    sinT3 = math.sin(theta3)
    theta2 = math.atan2(s, r) - math.atan2(a3*sinT3, a2 + a3*cosT3)

    return (theta1, theta2, theta3)

def compute_fk_3dof(theta1, theta2, theta3, dh):
    """
    Returns a 4x3 array of points:
    [base, joint1, joint2, end_effector]
    """
    d1 = dh['d1']
    a2 = dh['a2']
    a3 = dh['a3']

    base = np.array([0, 0, 0], dtype=float)
    joint1 = np.array([0, 0, d1], dtype=float)

    x2_local = a2*math.cos(theta2)
    z2_local = a2*math.sin(theta2)
    r = x2_local
    x2_global = r*math.cos(theta1)
    y2_global = r*math.sin(theta1)
    joint2 = joint1 + np.array([x2_global, y2_global, z2_local])

    x3_local = a3*math.cos(theta2+theta3)
    z3_local = a3*math.sin(theta2+theta3)
    r2 = x3_local
    x3_global = r2*math.cos(theta1)
    y3_global = r2*math.sin(theta1)
    end_effector = joint2 + np.array([x3_global, y3_global, z3_local])

    return np.vstack([base, joint1, joint2, end_effector])

############################################
# 2) ROTATION MATRICES & FRAME-DRAWING
############################################
def rotation_z(theta):
    """Return 3x3 rotation matrix about Z axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def rotation_y(theta):
    """Return 3x3 rotation matrix about Y axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def compute_frames_3dof(th1, th2, th3):
    """
    Approx. orientation frames for each joint.
    [R0, R1, R2, R3], each is a 3x3 rotation matrix.
    """
    R0 = np.eye(3)
    R1 = rotation_z(th1)          # after joint1
    R2 = R1 @ rotation_y(th2)     # after joint2
    R3 = R2 @ rotation_y(th3)     # end-effector
    return [R0, R1, R2, R3]

def draw_frames_on_arm(ax, points, frames, length=0.05):
    """
    Draw a small RGB frame at each of the 4 points:
    base->R0, joint1->R1, joint2->R2, end_effector->R3.
    Returns a list of line artists so we can remove them next time.
    """
    artists = []
    for i in range(4):
        origin = points[i]
        R = frames[i]
        # X-axis in red
        x_axis = origin + R[:,0]*length
        line_x, = ax.plot([origin[0], x_axis[0]],
                          [origin[1], x_axis[1]],
                          [origin[2], x_axis[2]],
                          color='red', linewidth=2)
        # Y-axis in green
        y_axis = origin + R[:,1]*length
        line_y, = ax.plot([origin[0], y_axis[0]],
                          [origin[1], y_axis[1]],
                          [origin[2], y_axis[2]],
                          color='green', linewidth=2)
        # Z-axis in blue
        z_axis = origin + R[:,2]*length
        line_z, = ax.plot([origin[0], z_axis[0]],
                          [origin[1], z_axis[1]],
                          [origin[2], z_axis[2]],
                          color='blue', linewidth=2)
        artists.extend([line_x, line_y, line_z])
    return artists

############################################
# 3) MAIN
############################################
def main():
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(0, 0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3-DOF Robot Arm Teleop (No arrow, frames at current step only)')

    # Joint angles vs time
    log_t1, log_t2, log_t3 = [], [], []

    # Start pose
    x, y, z = 0.1, 0.1, 0.1
    th1, th2, th3 = compute_ik_3dof(x, y, z, DH_PARAMS)

    # Arm line
    points = compute_fk_3dof(th1, th2, th3, DH_PARAMS)
    arm_line, = ax.plot(points[:,0], points[:,1], points[:,2],
                        '-o', color='blue')

    # Red path for end-effector
    ee_path = [points[-1]]
    path_line, = ax.plot([points[-1][0]], [points[-1][1]], [points[-1][2]],
                         color='red', linewidth=2)

    # Keep references to remove old frames
    current_frames = []

    plt.ion()
    plt.show()

    def update_plot(th1, th2, th3):
        nonlocal current_frames
        # 1) Remove old frames
        for artist in current_frames:
            artist.remove()
        current_frames.clear()

        # 2) Update arm line
        new_pts = compute_fk_3dof(th1, th2, th3, DH_PARAMS)
        arm_line.set_xdata(new_pts[:,0])
        arm_line.set_ydata(new_pts[:,1])
        arm_line.set_3d_properties(new_pts[:,2])

        # 3) End-effector path
        ee_new = new_pts[-1]
        ee_path.append(ee_new)
        px = [p[0] for p in ee_path]
        py = [p[1] for p in ee_path]
        pz = [p[2] for p in ee_path]
        path_line.set_xdata(px)
        path_line.set_ydata(py)
        path_line.set_3d_properties(pz)

        # 4) Draw new frames
        frames_list = compute_frames_3dof(th1, th2, th3)
        new_artists = draw_frames_on_arm(ax, new_pts, frames_list, length=0.05)
        current_frames.extend(new_artists)

        fig.canvas.draw()
        fig.canvas.flush_events()

    def move_endeffector(dx, dy, dz, steps=10):
        nonlocal x, y, z, th1, th2, th3
        step_x = dx/steps
        step_y = dy/steps
        step_z = dz/steps
        for _ in range(steps):
            x += step_x
            y += step_y
            z += step_z
            th1, th2, th3 = compute_ik_3dof(x, y, z, DH_PARAMS)
            log_t1.append(th1)
            log_t2.append(th2)
            log_t3.append(th3)
            update_plot(th1, th2, th3)
            plt.pause(0.025)

    # Move +0.1 in X, then -0.1
    move_endeffector(0.1, 0, 0)
    move_endeffector(-0.2, 0, 0)
    move_endeffector(0.1, 0, 0)


    # Move +0.1 in Y, then -0.1
    move_endeffector(0, 0.1, 0)
    move_endeffector(0, -0.2, 0)
    move_endeffector(0, 0.1, 0)

    # Move +0.1 in Z, then -0.1
    move_endeffector(0, 0, 0.1)
    move_endeffector(0, 0, -0.2)
    move_endeffector(0, 0, 0.1)

    plt.ioff()
    plt.show()

    # Plot angles vs time
    fig2, ax2 = plt.subplots()
    ax2.plot(log_t1, label='theta1 (rad)', color='red')
    ax2.plot(log_t2, label='theta2 (rad)', color='green')
    ax2.plot(log_t3, label='theta3 (rad)', color='blue')
    ax2.set_title('Joint Angles vs. Time')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Angle (radians)')
    ax2.legend()
    plt.show()

if __name__ == '__main__':
    main()
