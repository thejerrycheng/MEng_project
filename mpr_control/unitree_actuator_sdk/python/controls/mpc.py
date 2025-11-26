import time
import numpy as np
import cvxpy as cp
import sys
sys.path.append('../lib')
from unitree_actuator_sdk import *

# Motor parameters
gear_ratio = 6.33
J = 0.01  # Motor inertia
b = 0.1   # Damping coefficient
kp = 0.02  # Position gain
kd = 0.01  # Velocity gain
tau_min = -5.0  # Minimum torque
tau_max = 5.0   # Maximum torque
dt = 0.01  # Time step

# MPC parameters
N = 10  # Prediction horizon
Q = np.diag([10.0, 1.0])  # Weight for state error
R = np.array([[0.1]])  # Weight for control input

# Desired position and velocity
q_desired = 3.14  # Target position in radians
dq_desired = 0.0  # Target velocity in rad/s

# State-space matrices
A = np.array([[0, 1], [0, -b / J]])
B = np.array([[0], [1 / J]])

# Initialize serial communication
serial = SerialPort('/dev/ttyUSB0')
cmd = MotorCmd()
data = MotorData()

# Initialize state
x = np.zeros((2, 1))  # [q, dq]

def solve_mpc(x0, q_desired, dq_desired):
    """Solve the MPC optimization problem."""
    # Decision variables
    x_var = cp.Variable((2, N + 1))  # States: [q, dq]
    u_var = cp.Variable((1, N))      # Control inputs: torque

    # Constraints and cost
    constraints = []
    cost = 0

    # Initial state constraint
    constraints.append(x_var[:, 0] == x0[:, 0])

    for k in range(N):
        # Dynamics constraint: x_{k+1} = A x_k + B u_k
        constraints.append(x_var[:, k + 1] == A @ x_var[:, k] + B @ u_var[:, k])

        # Torque limits
        constraints.append(u_var[:, k] >= tau_min)
        constraints.append(u_var[:, k] <= tau_max)

        # Cost function
        state_error = x_var[:, k] - np.array([[q_desired], [dq_desired]])
        cost += cp.quad_form(state_error, Q) + cp.quad_form(u_var[:, k], R)

    # Terminal cost
    terminal_error = x_var[:, N] - np.array([[q_desired], [dq_desired]])
    cost += cp.quad_form(terminal_error, Q)

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Return the first control input
    return u_var.value[0, 0]

try:
    print("MPC Control Initialized. Press Ctrl+C to stop.")
    while True:
        # Read current state from the motor
        data.motorType = MotorType.GO_M8010_6
        cmd.motorType = MotorType.GO_M8010_6
        cmd.mode = queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC)
        cmd.id = 0
        serial.sendRecv(cmd, data)

        # Update state (position and velocity)
        x[0, 0] = data.q / gear_ratio  # Convert from rotor space to output space
        x[1, 0] = data.dq / gear_ratio

        # Solve the MPC problem
        optimal_torque = solve_mpc(x, q_desired, dq_desired)

        # Send the control command to the motor
        cmd.q = q_desired * gear_ratio
        cmd.dq = dq_desired * gear_ratio
        cmd.tau = optimal_torque
        cmd.kp = kp
        cmd.kd = kd
        serial.sendRecv(cmd, data)

        # Print feedback
        print(f"Position: {data.q / gear_ratio:.4f} rad")
        print(f"Velocity: {data.dq / gear_ratio:.4f} rad/s")
        print(f"Torque: {data.tau:.4f} Nm")
        print(f"Temperature: {data.temp:.2f} °C")

        time.sleep(dt)

except KeyboardInterrupt:
    print("MPC Control Stopped.")
finally:
    # Safely stop the motor
    cmd.q = data.q
    cmd.dq = 0.0
    cmd.tau = 0.0
    cmd.kp = 0.0
    cmd.kd = 0.0
    serial.sendRecv(cmd, data)
    print("Motor stopped.")
