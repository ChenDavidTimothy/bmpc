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


def verify_minimal_constraints(X_sol, V_sol, U_sol, birkhoff_components, time_scaling, P):
    """Verify the minimal constraint set for mathematical correctness"""
    B_a = birkhoff_components.birkhoff_matrix_a
    w_B = birkhoff_components.birkhoff_quadrature_weights

    n_states = X_sol.shape[0]
    N = X_sol.shape[1] - 1

    # Convert to numpy for verification
    X_np = DM2Arr(X_sol)
    V_np = DM2Arr(V_sol)
    U_np = DM2Arr(U_sol)
    P_np = DM2Arr(P)

    x_a = P_np[:n_states]

    print("=== Minimal Birkhoff Constraint Verification ===")

    # Test 1: Interpolation constraint for ALL k (includes initial condition automatically)
    max_interp_error = 0
    for k in range(N + 1):
        expected = x_a + np.sum([B_a[k, j] * V_np[:, j] for j in range(N + 1)], axis=0)
        actual = X_np[:, k]
        error = np.linalg.norm(actual - expected)
        max_interp_error = max(max_interp_error, error)

    print(f"Max interpolation constraint error: {max_interp_error:.2e}")

    # Test 2: Dynamics constraint V = time_scaling * f(X, U)
    max_dyn_error = 0
    for k in range(N + 1):
        x, y, theta = X_np[:, k]
        v_a, v_b, v_c, v_d = U_np[:, k]

        # Mecanum wheel dynamics
        rot_3d_z = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )

        J = (wheel_radius / 4) * np.array(
            [
                [1, 1, 1, 1],
                [-1, 1, 1, -1],
                [-1 / (Lx + Ly), 1 / (Lx + Ly), -1 / (Lx + Ly), 1 / (Lx + Ly)],
            ]
        )

        dynamics = rot_3d_z @ J @ U_np[:, k]
        expected_V = time_scaling * dynamics
        actual_V = V_np[:, k]
        error = np.linalg.norm(actual_V - expected_V)
        max_dyn_error = max(max_dyn_error, error)

    print(f"Max dynamics constraint error: {max_dyn_error:.2e}")

    # Verify automatic properties for CGL grids
    # Property 1: Initial condition automatically satisfied
    initial_error = np.linalg.norm(X_np[:, 0] - x_a)
    print(f"Initial condition error (automatic): {initial_error:.2e}")

    # Property 2: Equivalence condition automatically satisfied
    x_b_expected = x_a + np.sum([w_B[j] * V_np[:, j] for j in range(N + 1)], axis=0)
    x_b_actual = X_np[:, N]
    equiv_error = np.linalg.norm(x_b_actual - x_b_expected)
    print(f"Equivalence condition error (automatic): {equiv_error:.2e}")

    # Verify CGL grid mathematical property: B^a[N,j] = w_B[j]
    weight_consistency_error = np.linalg.norm(B_a[N, :] - w_B)
    print(f"CGL property B^a[N,:] = w_B error: {weight_consistency_error:.2e}")

    # Verify CGL grid mathematical property: B^a[0,j] = 0
    initial_basis_error = np.linalg.norm(B_a[0, :])
    print(f"CGL property B^a[0,:] = 0 error: {initial_basis_error:.2e}")

    return max_interp_error < 1e-6 and max_dyn_error < 1e-6


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

# Optimization variables
X = ca.SX.sym("X", n_states, N + 1)  # States at grid points
V = ca.SX.sym("V", n_states, N + 1)  # Virtual variables (derivatives)
U = ca.SX.sym("U", n_controls, N + 1)  # Controls at ALL grid points

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
    con = U[:, k]

    stage_cost = (st - P[n_states:]).T @ Q @ (st - P[n_states:])
    stage_cost += con.T @ R @ con

    # Birkhoff quadrature integration
    cost_fn += w_B[k] * time_scaling * stage_cost

# MINIMAL CONSTRAINT SET FOR CGL GRIDS - ZERO REDUNDANCY
g = []

# Constraint 1: BIRKHOFF INTERPOLATION X = x^a + B^a * V
# This automatically enforces:
# - Initial condition at k=0: X[:, 0] = x^a (since B^a[0,j] = 0)
# - Equivalence condition at k=N: X[:, N] = x^a + w_B^T * V (since B^a[N,j] = w_B[j])
for k in range(N + 1):
    interpolation_constraint = X[:, k] - P[:n_states]  # X[:, k] - x^a
    for j in range(N + 1):
        interpolation_constraint -= B_a[k, j] * V[:, j]  # Subtract B^a * V
    g.append(interpolation_constraint)

# Constraint 2: DYNAMICS V = time_scaling * f(X, U)
for k in range(N + 1):
    st = X[:, k]
    con = U[:, k]
    dynamics_constraint = V[:, k] - time_scaling * f(st, con)
    g.append(dynamics_constraint)

# NO OTHER CONSTRAINTS NEEDED FOR CGL GRIDS!
# Mathematical proof:
# - Initial condition: automatic from interpolation at k=0
# - Equivalence condition: automatic from interpolation at k=N
# - All boundaries properly handled by CGL grid properties

# Concatenate all constraints
g = ca.vertcat(*g)

print(f"Constraint count: {g.size1()} (minimal for CGL grids)")
print(f"Expected: {(n_states * (N + 1)) + (n_states * (N + 1))} = {2 * n_states * (N + 1)}")

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
n_X = n_states * (N + 1)
n_V = n_states * (N + 1)
n_U = n_controls * (N + 1)
n_vars = n_X + n_V + n_U

lbx = ca.DM.zeros((n_vars, 1))
ubx = ca.DM.zeros((n_vars, 1))

# State bounds (unbounded)
for i in range(n_X):
    lbx[i] = -ca.inf
    ubx[i] = ca.inf

# Virtual variable bounds (unbounded)
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
u0 = ca.DM.zeros((n_controls, N + 1))
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
            ca.reshape(X0, n_X, 1),
            ca.reshape(V0, n_V, 1),
            ca.reshape(u0, n_U, 1),
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
        X_sol = ca.reshape(sol["x"][:n_X], n_states, N + 1)
        V_sol = ca.reshape(sol["x"][n_X : n_X + n_V], n_states, N + 1)
        u = ca.reshape(sol["x"][n_X + n_V :], n_controls, N + 1)

        # VERIFICATION: Check mathematical correctness (first iteration only)
        if mpc_iter == 0:
            is_valid = verify_minimal_constraints(
                X_sol, V_sol, u, birkhoff_components, time_scaling, args["p"]
            )
            if not is_valid:
                print("❌ MATHEMATICAL CONSTRAINTS VIOLATED!")
                break
            else:
                print("✅ Minimal Birkhoff constraints satisfied")

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
