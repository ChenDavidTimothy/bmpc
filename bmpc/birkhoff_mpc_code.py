from time import time

import casadi as ca
import numpy as np
from birkhoff_simulation import simulate
from casadi import cos, pi, sin

from bmpc.birkhoff import _compute_birkhoff_basis_components


# EXACT PARAMETERS FROM ORIGINAL - DO NOT CHANGE
Q_x = 100
Q_y = 100
Q_theta = 2000
R1 = 1
R2 = 1
R3 = 1
R4 = 1

step_horizon = 0.1  # time between steps in seconds
N = 10  # number of look ahead steps
rob_diam = 0.3  # diameter of the robot
wheel_radius = 1  # wheel radius
Lx = 0.3  # L in J Matrix (half robot x-axis length)
Ly = 0.3  # l in J Matrix (half robot y-axis length)
sim_time = 200  # simulation time

# specs
x_init = 0
y_init = 0
theta_init = 0
x_target = 15
y_target = 10
theta_target = pi / 4

v_max = 1
v_min = -1


def apply_control_and_integrate(step_horizon, t0, state_init, u_first):
    """Apply first control and integrate actual system forward - UNCHANGED from original"""
    f_value = f(state_init, u_first)
    next_state = ca.DM.full(state_init + (step_horizon * f_value))
    t0 = t0 + step_horizon
    return t0, ca.DM(next_state)


def DM2Arr(dm):
    return np.array(dm.full())


def generate_cgl_grid(N):
    """Generate Chebyshev-Gauss-Lobatto grid points"""
    j = np.arange(N + 1)
    xi_j = j / N
    tau_j = -np.cos(xi_j * np.pi)
    return tau_j


# MATHEMATICAL SETUP - EXACT FROM BIRKHOFF THEORY
tau_grid = generate_cgl_grid(N)
birkhoff_components = _compute_birkhoff_basis_components(tuple(tau_grid), -1.0, 1.0)

# state symbolic variables
x = ca.SX.sym("x")
y = ca.SX.sym("y")
theta = ca.SX.sym("theta")
states = ca.vertcat(x, y, theta)
n_states = states.numel()

# control symbolic variables
V_a = ca.SX.sym("V_a")
V_b = ca.SX.sym("V_b")
V_c = ca.SX.sym("V_c")
V_d = ca.SX.sym("V_d")
controls = ca.vertcat(V_a, V_b, V_c, V_d)
n_controls = controls.numel()

# Optimization variables: States, Virtual variables (derivatives), Controls
X = ca.SX.sym("X", n_states, N + 1)  # States at grid points
V = ca.SX.sym("V", n_states, N + 1)  # Virtual variables (derivatives)
U = ca.SX.sym("U", n_controls, N)  # Controls

# Parameters: initial state and target state
P = ca.SX.sym("P", n_states + n_states)

# Weight matrices
Q = ca.diagcat(Q_x, Q_y, Q_theta)
R = ca.diagcat(R1, R2, R3, R4)

# EXACT MECANUM WHEEL DYNAMICS FROM ORIGINAL
rot_3d_z = ca.vertcat(
    ca.horzcat(cos(theta), -sin(theta), 0),
    ca.horzcat(sin(theta), cos(theta), 0),
    ca.horzcat(0, 0, 1),
)

J = (wheel_radius / 4) * ca.DM(
    [
        [1, 1, 1, 1],
        [-1, 1, 1, -1],
        [-1 / (Lx + Ly), 1 / (Lx + Ly), -1 / (Lx + Ly), 1 / (Lx + Ly)],
    ]
)

RHS = rot_3d_z @ J @ controls
f = ca.Function("f", [states, controls], [RHS])

# BIRKHOFF MATHEMATICAL COMPONENTS
B_a = ca.DM(birkhoff_components.birkhoff_matrix_a)
w_B = ca.DM(birkhoff_components.birkhoff_quadrature_weights)

# Time scaling: map [-1,1] to [0, step_horizon*N]
time_scaling = (step_horizon * N) / 2.0

# COST FUNCTION USING BIRKHOFF QUADRATURE
cost_fn = 0
for k in range(N + 1):
    st = X[:, k]
    stage_cost = (st - P[n_states:]).T @ Q @ (st - P[n_states:])

    # Add control cost only for k < N (controls defined for k=0..N-1)
    if k < N:
        con = U[:, k]
        stage_cost += con.T @ R @ con

    # Birkhoff quadrature integration
    cost_fn += w_B[k] * time_scaling * stage_cost

# CONSTRAINTS - IMPLEMENTING PROBLEM P^N_a FROM BIRKHOFF PAPERS
g = []

# 1. INITIAL CONDITION: X[:, 0] = x^a (initial state)
g.append(X[:, 0] - P[:n_states])

# 2. BIRKHOFF INTERPOLATION CONSTRAINT: X = x^a * ones + B^a * V
# This implements equation: X[:, k] = x^a + sum_j(B^a[k,j] * V[:, j])
ones_vec = ca.DM.ones(N + 1, 1)
for k in range(N + 1):
    interpolation_constraint = X[:, k] - P[:n_states]  # Start with X[:, k] - x^a
    for j in range(N + 1):
        interpolation_constraint -= B_a[k, j] * V[:, j]  # Subtract B^a * V
    g.append(interpolation_constraint)

# 3. DYNAMICS CONSTRAINT: V = time_scaling * f(X, U)
# This enforces that virtual variables equal scaled dynamics
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    dynamics_constraint = V[:, k] - time_scaling * f(st, con)
    g.append(dynamics_constraint)

# Handle final virtual variable (no control at k=N)
# Use final state dynamics with final control (duplicated)
st_final = X[:, N]
con_final = U[:, N - 1]  # Use last available control
dynamics_final = V[:, N] - time_scaling * f(st_final, con_final)
g.append(dynamics_final)

# 4. CRITICAL: BIRKHOFF EQUIVALENCE CONDITION
# From Paper Equation (16): x^b = x^a + w_B^T * V
# This is the key constraint that makes Birkhoff interpolation mathematically correct
equivalence_constraint = X[:, N] - P[:n_states]  # x^b - x^a
for j in range(N + 1):
    equivalence_constraint -= w_B[j] * V[:, j]  # Subtract w_B^T * V
g.append(equivalence_constraint)

# Concatenate all constraints
g = ca.vertcat(*g)

# OPTIMIZATION PROBLEM FORMULATION
OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),
    V.reshape((-1, 1)),
    U.reshape((-1, 1)),
)

nlp_prob = {"f": cost_fn, "x": OPT_variables, "g": g, "p": P}

opts = {
    "ipopt": {
        "max_iter": 2000,
        "print_level": 0,
        "acceptable_tol": 1e-8,
        "acceptable_obj_change_tol": 1e-6,
    },
    "print_time": 0,
}

solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)

# VARIABLE BOUNDS
n_vars = OPT_variables.size1()
lbx = ca.DM.zeros((n_vars, 1))
ubx = ca.DM.zeros((n_vars, 1))

# State bounds (unbounded)
n_X = n_states * (N + 1)
for i in range(n_X):
    lbx[i] = -ca.inf
    ubx[i] = ca.inf

# Virtual variable bounds (unbounded)
n_V = n_states * (N + 1)
for i in range(n_X, n_X + n_V):
    lbx[i] = -ca.inf
    ubx[i] = ca.inf

# Control bounds
for i in range(n_X + n_V, n_vars):
    lbx[i] = v_min
    ubx[i] = v_max

args = {
    "lbg": ca.DM.zeros((g.size1(), 1)),
    "ubg": ca.DM.zeros((g.size1(), 1)),
    "lbx": lbx,
    "ubx": ubx,
}

# INITIALIZATION
t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])
state_target = ca.DM([x_target, y_target, theta_target])

t = ca.DM(t0)
u0 = ca.DM.zeros((n_controls, N))
X0 = ca.repmat(state_init, 1, N + 1)
V0 = ca.DM.zeros((n_states, N + 1))

mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])

# MAIN MPC LOOP
if __name__ == "__main__":
    main_loop = time()
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()

        # Set parameters
        args["p"] = ca.vertcat(state_init, state_target)

        # Set initial guess
        args["x0"] = ca.vertcat(
            ca.reshape(X0, n_states * (N + 1), 1),
            ca.reshape(V0, n_states * (N + 1), 1),
            ca.reshape(u0, n_controls * N, 1),
        )

        # Solve optimization
        sol = solver(
            x0=args["x0"],
            lbx=args["lbx"],
            ubx=args["ubx"],
            lbg=args["lbg"],
            ubg=args["ubg"],
            p=args["p"],
        )

        # Extract solution
        n_X = n_states * (N + 1)
        n_V = n_states * (N + 1)

        X_sol = ca.reshape(sol["x"][:n_X], n_states, N + 1)
        V_sol = ca.reshape(sol["x"][n_X : n_X + n_V], n_states, N + 1)
        u = ca.reshape(sol["x"][n_X + n_V :], n_controls, N)

        # Store results
        cat_states = np.dstack((cat_states, DM2Arr(X_sol)))
        cat_controls = np.vstack((cat_controls, DM2Arr(u[:, 0])))
        t = np.vstack((t, t0))

        # Apply first control and integrate real system forward
        t0, state_init = apply_control_and_integrate(step_horizon, t0, state_init, u[:, 0])

        # Warm start for next iteration
        X0 = ca.horzcat(X_sol[:, 1:], ca.reshape(X_sol[:, -1], -1, 1))
        V0 = ca.horzcat(V_sol[:, 1:], ca.reshape(V_sol[:, -1], -1, 1))
        u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))

        t2 = time()
        print(f"MPC Iteration: {mpc_iter}, Time: {(t2 - t1) * 1000:.2f}ms")
        times = np.vstack((times, t2 - t1))

        mpc_iter += 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print(f"\nTotal time: {main_loop_time - main_loop:.2f}s")
    print(f"Average iteration time: {np.array(times).mean() * 1000:.2f}ms")
    print(f"Final error: {float(ss_error):.6f}")

    # Simulate results
    simulate(
        cat_states,
        cat_controls,
        times,
        step_horizon,
        N,
        np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]),
        save=False,
    )
