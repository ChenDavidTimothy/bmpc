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
    """Apply first control and integrate actual system forward"""
    # This is the KEY difference: we integrate the REAL system, not shift grid points
    f_value = f(state_init, u_first)  # Evaluate actual dynamics
    next_state = ca.DM.full(state_init + (step_horizon * f_value))  # Euler integration
    t0 = t0 + step_horizon
    return t0, ca.DM(next_state)


def DM2Arr(dm):
    return np.array(dm.full())


# Generate CGL grid points
def generate_cgl_grid(N):
    j = np.arange(N + 1)
    xi_j = j / N
    tau_j = -np.cos(xi_j * np.pi)
    return tau_j


# Get Birkhoff components
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

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym("X", n_states, N + 1)

# matrix containing all virtual variables (derivatives)
V = ca.SX.sym("V", n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym("U", n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym("P", n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)

# controls weights matrix
R = ca.diagcat(R1, R2, R3, R4)

# Mecanum wheel dynamics (EXACT from original)
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

# Convert Birkhoff components to CasADi
B_a = ca.DM(birkhoff_components.birkhoff_matrix_a)
w_B = ca.DM(birkhoff_components.birkhoff_quadrature_weights)

cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # initial condition constraint

# Time scaling for [-1,1] to [0, step_horizon*N]
time_scaling = (step_horizon * N) / 2.0

# Birkhoff interpolation and dynamics constraints replace RK4
for k in range(N):
    st = X[:, k]
    con = U[:, k]

    # Cost using Birkhoff quadrature weights
    cost_fn = cost_fn + w_B[k] * time_scaling * (
        (st - P[n_states:]).T @ Q @ (st - P[n_states:]) + con.T @ R @ con
    )

    # Dynamics constraint: V_k = f(X_k, U_k) * time_scaling
    dynamics_constraint = V[:, k] - time_scaling * f(st, con)
    g = ca.vertcat(g, dynamics_constraint)

# Terminal cost
st_final = X[:, N]
cost_fn = cost_fn + w_B[N] * time_scaling * (st_final - P[n_states:]).T @ Q @ (
    st_final - P[n_states:]
)

# Birkhoff interpolation constraints: X = x_init + B^a * V
ones_vec = ca.DM.ones(N + 1, 1)
for k in range(N + 1):
    interpolation_constraint = X[:, k] - P[:n_states]
    for j in range(N + 1):
        interpolation_constraint -= B_a[k, j] * V[:, j]
    g = ca.vertcat(g, interpolation_constraint)

# Final virtual variable constraint (zero derivative at end)
g = ca.vertcat(g, V[:, N])

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

# Variable bounds
n_vars = OPT_variables.size1()
lbx = ca.DM.zeros((n_vars, 1))
ubx = ca.DM.zeros((n_vars, 1))

# State bounds (X) - unbounded
n_X = n_states * (N + 1)
for i in range(n_X):
    lbx[i] = -ca.inf
    ubx[i] = ca.inf

# Virtual variable bounds (V) - unbounded
n_V = n_states * (N + 1)
for i in range(n_X, n_X + n_V):
    lbx[i] = -ca.inf
    ubx[i] = ca.inf

# Control bounds (U)
for i in range(n_X + n_V, n_vars):
    lbx[i] = v_min
    ubx[i] = v_max

args = {
    "lbg": ca.DM.zeros((g.size1(), 1)),
    "ubg": ca.DM.zeros((g.size1(), 1)),
    "lbx": lbx,
    "ubx": ubx,
}

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

if __name__ == "__main__":
    main_loop = time()
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        args["p"] = ca.vertcat(
            state_init,
            state_target,
        )

        args["x0"] = ca.vertcat(
            ca.reshape(X0, n_states * (N + 1), 1),
            ca.reshape(V0, n_states * (N + 1), 1),
            ca.reshape(u0, n_controls * N, 1),
        )

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

        cat_states = np.dstack((cat_states, DM2Arr(X_sol)))
        cat_controls = np.vstack((cat_controls, DM2Arr(u[:, 0])))
        t = np.vstack((t, t0))

        # Apply first control and integrate real system forward
        t0, state_init = apply_control_and_integrate(step_horizon, t0, state_init, u[:, 0])

        # Warm start: shift trajectory for next optimization (this is correct for Birkhoff)
        X0 = ca.horzcat(X_sol[:, 1:], ca.reshape(X_sol[:, -1], -1, 1))
        V0 = ca.horzcat(V_sol[:, 1:], ca.reshape(V_sol[:, -1], -1, 1))
        u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))

        t2 = time()
        print(mpc_iter)
        print(t2 - t1)
        times = np.vstack((times, t2 - t1))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("avg iteration time: ", np.array(times).mean() * 1000, "ms")
    print("final error: ", ss_error)

    # simulate
    simulate(
        cat_states,
        cat_controls,
        times,
        step_horizon,
        N,
        np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]),
        save=False,
    )
