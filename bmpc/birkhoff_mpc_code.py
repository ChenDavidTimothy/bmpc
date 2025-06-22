from time import time

import casadi as ca
import numpy as np
from casadi import cos, pi, sin

from bmpc.birkhoff import _compute_birkhoff_basis_components
from bmpc.bmpc_types import FloatArray


class BirkhoffMPC:
    """Production Birkhoff MPC controller for mecanum wheel robots."""

    def __init__(
        self,
        N: int = 10,
        step_horizon: float = 0.1,
        sim_time: float = 200.0,
        Q_diag: tuple[float, float, float] = (100.0, 100.0, 2000.0),
        R_diag: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        v_bounds: tuple[float, float] = (-1.0, 1.0),
        robot_params: dict | None = None,
        solver_opts: dict | None = None,
        debug: bool = False,
    ):
        self.N = N
        self.step_horizon = step_horizon
        self.sim_time = sim_time
        self.v_min, self.v_max = v_bounds
        self.debug = debug

        # Robot parameters
        robot_params = robot_params or {}
        self.wheel_radius = robot_params.get("wheel_radius", 1.0)
        self.Lx = robot_params.get("Lx", 0.3)
        self.Ly = robot_params.get("Ly", 0.3)

        # Initialize Birkhoff components
        self._setup_birkhoff_grid()
        self._setup_dynamics()
        self._setup_optimization(Q_diag, R_diag, solver_opts)

        # State tracking
        self.mpc_iter = 0
        self.solve_times = []

    def _setup_birkhoff_grid(self) -> None:
        """Initialize CGL grid and Birkhoff components."""
        j = np.arange(self.N + 1)
        xi_j = j / self.N
        tau_grid = -np.cos(xi_j * np.pi)

        self.birkhoff = _compute_birkhoff_basis_components(tuple(tau_grid), -1.0, 1.0)
        self.time_scaling = (self.step_horizon * self.N) / 2.0

        self.B_a = ca.DM(self.birkhoff.birkhoff_matrix_a)
        self.w_B = ca.DM(self.birkhoff.birkhoff_quadrature_weights)

    def _setup_dynamics(self) -> None:
        """Setup mecanum wheel dynamics."""
        # State and control symbols
        x, y, theta = ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("theta")
        self.states = ca.vertcat(x, y, theta)
        self.n_states = 3

        Va, Vb, Vc, Vd = ca.SX.sym("Va"), ca.SX.sym("Vb"), ca.SX.sym("Vc"), ca.SX.sym("Vd")
        self.controls = ca.vertcat(Va, Vb, Vc, Vd)
        self.n_controls = 4

        # Mecanum wheel transfer matrix
        rot_3d_z = ca.vertcat(
            ca.horzcat(cos(theta), -sin(theta), 0),
            ca.horzcat(sin(theta), cos(theta), 0),
            ca.horzcat(0, 0, 1),
        )

        J = (self.wheel_radius / 4) * ca.DM(
            [
                [1, 1, 1, 1],
                [-1, 1, 1, -1],
                [
                    -1 / (self.Lx + self.Ly),
                    1 / (self.Lx + self.Ly),
                    -1 / (self.Lx + self.Ly),
                    1 / (self.Lx + self.Ly),
                ],
            ]
        )

        self.dynamics = ca.Function(
            "f", [self.states, self.controls], [rot_3d_z @ J @ self.controls]
        )

    def _setup_optimization(
        self,
        Q_diag: tuple[float, float, float],
        R_diag: tuple[float, float, float, float],
        solver_opts: dict | None,
    ) -> None:
        """Setup optimization problem."""
        # Optimization variables
        X = ca.SX.sym("X", self.n_states, self.N + 1)
        V = ca.SX.sym("V", self.n_states, self.N + 1)
        U = ca.SX.sym("U", self.n_controls, self.N + 1)
        P = ca.SX.sym("P", 2 * self.n_states)  # [current_state, target_state]

        # Weight matrices
        Q = ca.diagcat(*Q_diag)
        R = ca.diagcat(*R_diag)

        # Cost function using Birkhoff quadrature
        cost = 0
        for k in range(self.N + 1):
            stage_cost = (X[:, k] - P[self.n_states :]).T @ Q @ (X[:, k] - P[self.n_states :])
            stage_cost += U[:, k].T @ R @ U[:, k]
            cost += self.w_B[k] * self.time_scaling * stage_cost

        # Constraints: Interpolation + Dynamics
        g = []
        for k in range(self.N + 1):
            # Interpolation: X[:, k] = x^a + B^a[k, :] @ V
            interp_constraint = X[:, k] - P[: self.n_states]
            for j in range(self.N + 1):
                interp_constraint -= self.B_a[k, j] * V[:, j]
            g.append(interp_constraint)

            # Dynamics: V[:, k] = time_scaling * f(X[:, k], U[:, k])
            g.append(V[:, k] - self.time_scaling * self.dynamics(X[:, k], U[:, k]))

        # Setup NLP
        opt_vars = ca.vertcat(X.reshape((-1, 1)), V.reshape((-1, 1)), U.reshape((-1, 1)))
        nlp = {"f": cost, "x": opt_vars, "g": ca.vertcat(*g), "p": P}

        # Solver options
        if solver_opts is None:
            solver_opts = {
                "ipopt": {
                    "max_iter": 2000,
                    "print_level": 0,
                    "acceptable_tol": 1e-8,
                    "acceptable_obj_change_tol": 1e-6,
                },
                "print_time": 0,
            }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, solver_opts)

        # Variable bounds
        n_vars = (self.n_states + self.n_states + self.n_controls) * (self.N + 1)
        self.lbx = ca.DM(
            [-ca.inf] * (2 * self.n_states * (self.N + 1))
            + [self.v_min] * (self.n_controls * (self.N + 1))
        )
        self.ubx = ca.DM(
            [ca.inf] * (2 * self.n_states * (self.N + 1))
            + [self.v_max] * (self.n_controls * (self.N + 1))
        )

        # Constraint bounds (all equality)
        n_constraints = 2 * self.n_states * (self.N + 1)
        self.lbg = ca.DM.zeros(n_constraints, 1)
        self.ubg = ca.DM.zeros(n_constraints, 1)

        # Dimensions for solution extraction
        self.n_X = self.n_states * (self.N + 1)
        self.n_V = self.n_states * (self.N + 1)
        self.n_U = self.n_controls * (self.N + 1)

    def solve_mpc(
        self, current_state: FloatArray, target_state: FloatArray, warm_start: dict | None = None
    ) -> tuple[FloatArray, dict]:
        """Solve single MPC step."""
        t_start = time()

        # Set parameters
        p = ca.vertcat(ca.DM(current_state), ca.DM(target_state))

        # Initial guess
        if warm_start is None:
            X0 = ca.repmat(ca.DM(current_state), 1, self.N + 1)
            V0 = ca.DM.zeros(self.n_states, self.N + 1)
            U0 = ca.DM.zeros(self.n_controls, self.N + 1)
        else:
            X0, V0, U0 = warm_start["X"], warm_start["V"], warm_start["U"]

        x0 = ca.vertcat(X0.reshape((-1, 1)), V0.reshape((-1, 1)), U0.reshape((-1, 1)))

        # Solve
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

            if self.solver.stats()["success"]:
                # Extract solution
                X_sol = ca.reshape(sol["x"][: self.n_X], self.n_states, self.N + 1)
                V_sol = ca.reshape(
                    sol["x"][self.n_X : self.n_X + self.n_V], self.n_states, self.N + 1
                )
                U_sol = ca.reshape(sol["x"][self.n_X + self.n_V :], self.n_controls, self.N + 1)

                # Warm start for next iteration
                warm_start_next = {
                    "X": ca.horzcat(X_sol[:, 1:], X_sol[:, -1:]),
                    "V": ca.horzcat(V_sol[:, 1:], V_sol[:, -1:]),
                    "U": ca.horzcat(U_sol[:, 1:], U_sol[:, -1:]),
                }

                solve_time = time() - t_start
                self.solve_times.append(solve_time)

                result = {
                    "success": True,
                    "optimal_control": np.array(U_sol[:, 0].full()).flatten(),
                    "predicted_trajectory": np.array(X_sol.full()),
                    "solve_time": solve_time,
                    "warm_start": warm_start_next,
                    "cost": float(sol["f"]),
                }

                if self.debug and self.mpc_iter == 0:
                    self._verify_solution(X_sol, V_sol, U_sol, p)

                return result["optimal_control"], result

            else:
                return np.zeros(self.n_controls), {"success": False, "message": "Solver failed"}

        except Exception as e:
            return np.zeros(self.n_controls), {"success": False, "message": str(e)}

    def _verify_solution(self, X_sol, V_sol, U_sol, p):
        """Debug verification of mathematical correctness."""
        X_np = np.array(X_sol.full())
        V_np = np.array(V_sol.full())
        p_np = np.array(p.full()).flatten()
        x_a = p_np[: self.n_states]

        # Check interpolation constraints: X[:, k] = x^a + Σ_j B^a[k,j] * V[:, j]
        B_a_np = np.array(self.B_a.full())
        max_interp_error = 0
        for k in range(self.N + 1):
            expected = x_a + V_np @ B_a_np[k, :]  # V_np is (3, 11), B_a_np[k, :] is (11,)
            actual = X_np[:, k]
            error = np.linalg.norm(actual - expected)
            max_interp_error = max(max_interp_error, error)

        if max_interp_error > 1e-6:
            print(f"⚠️  Interpolation error: {max_interp_error:.2e}")
        else:
            print(f"✅ Solution verified (interp error: {max_interp_error:.2e})")

    def run_mpc_loop(
        self, initial_state: FloatArray, target_state: FloatArray, tolerance: float = 1e-1
    ) -> dict:
        """Run complete MPC control loop."""
        current_state = np.array(initial_state, dtype=np.float64)
        target = np.array(target_state, dtype=np.float64)

        # Storage - match original format for animation
        prediction_horizons = []  # Store full prediction from each MPC step
        state_history = [current_state.copy()]
        control_history = []
        warm_start = None
        t = 0.0

        print(f"Starting MPC loop (N={self.N}, horizon={self.step_horizon * self.N:.1f}s)")

        while np.linalg.norm(current_state - target) > tolerance and t < self.sim_time:
            # Solve MPC
            u_opt, result = self.solve_mpc(current_state, target, warm_start)

            if not result["success"]:
                print(f"❌ MPC failed at t={t:.2f}: {result.get('message', 'Unknown error')}")
                break

            # Store prediction horizon for animation (matches original cat_states format)
            prediction_horizons.append(result["predicted_trajectory"])

            # Apply control and integrate
            f_val = self.dynamics(current_state, u_opt)
            current_state = current_state + self.step_horizon * np.array(f_val.full()).flatten()

            # Store results
            state_history.append(current_state.copy())
            control_history.append(u_opt.copy())
            warm_start = result["warm_start"]
            t += self.step_horizon
            self.mpc_iter += 1

            if self.mpc_iter % 10 == 0:
                error = np.linalg.norm(current_state - target)
                avg_time = np.mean(self.solve_times[-10:]) * 1000
                print(f"Iter {self.mpc_iter:3d}: error={error:.3f}, solve_time={avg_time:.1f}ms")

        final_error = np.linalg.norm(current_state - target)
        total_time = sum(self.solve_times)
        avg_time = np.mean(self.solve_times) * 1000

        print("\nMPC completed:")
        print(f"  Final error: {final_error:.6f}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average solve time: {avg_time:.2f}ms")
        print(f"  Iterations: {self.mpc_iter}")

        return {
            "state_history": np.array(state_history),
            "control_history": np.array(control_history),
            "prediction_horizons": prediction_horizons,
            "final_error": final_error,
            "total_time": total_time,
            "avg_solve_time": avg_time,
            "iterations": self.mpc_iter,
        }


def main():
    """Example usage."""
    # Problem setup
    initial_state = [0.0, 0.0, 0.0]  # x, y, theta
    target_state = [15.0, 10.0, pi / 4]

    # Initialize controller
    mpc = BirkhoffMPC(
        N=10,
        step_horizon=0.1,
        Q_diag=(100.0, 100.0, 2000.0),
        R_diag=(1.0, 1.0, 1.0, 1.0),
        debug=True,
    )

    # Run MPC
    results = mpc.run_mpc_loop(initial_state, target_state)

    # Optional: Visualize results
    try:
        from birkhoff_simulation import simulate

        # Convert to original format: cat_states shape (n_states, N+1, iterations)
        if results["prediction_horizons"]:
            cat_states = np.stack(results["prediction_horizons"], axis=2)  # (3, 11, iterations)

            # Controls: match exact length with cat_states
            controls = np.array(results["control_history"])  # Shape: (iterations, 4)

            # Times: match exact length with cat_states iterations
            times = np.array([[mpc.step_horizon]] * len(controls))  # Shape: (iterations, 1)

            # Reference
            reference = np.array(initial_state + target_state)

            print(
                f"Animation data: cat_states{cat_states.shape}, controls{controls.shape}, times{times.shape}"
            )

            simulate(cat_states, controls, times, mpc.step_horizon, mpc.N, reference, save=False)
        else:
            print("No predictions to visualize")

    except ImportError:
        print("Simulation not available")


if __name__ == "__main__":
    main()
