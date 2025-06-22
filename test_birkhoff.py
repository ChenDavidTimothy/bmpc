"""
BIRKHOFF MATHEMATICAL CORRECTNESS VERIFICATION

This test suite verifies the core mathematical properties of Birkhoff interpolation
against analytically known results, similar to the Radau mathematical correctness tests.

Focus: Mathematical correctness validation using external verification methods.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from maptor.birkhoff import (
    _compute_birkhoff_basis_components,
    _evaluate_birkhoff_interpolation_a_form,
    _evaluate_birkhoff_interpolation_b_form,
)


class TestBirkhoffMathematicalCorrectness:
    """
    Mathematical correctness tests using analytically verifiable cases.
    Similar to test_radau_mathematical_correctness.py approach.
    """

    @pytest.mark.parametrize("N", [2, 3, 4, 5])
    def test_polynomial_exactness_against_analytical_formulas(self, N):
        """
        Test polynomial exactness using simple uniform grids and known polynomial formulas.

        MATHEMATICAL BASIS: Birkhoff interpolation must reproduce polynomials of degree ≤ N exactly.
        VERIFICATION: Use analytical polynomial formulas as external truth.
        """
        # Use simple uniform grid on [0, 1]
        grid_points = np.linspace(0.1, 0.9, N)
        tau_a, tau_b = 0.0, 1.0

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)

        # Test with polynomials of degree 0 to N-1 (guaranteed exactness)
        for degree in range(N):
            # Create simple monomial: p(t) = t^degree
            def polynomial(t):
                return t**degree

            def polynomial_derivative(t):
                if degree == 0:
                    return 0.0
                return degree * (t ** (degree - 1))

            # Set up interpolation data
            y_initial = polynomial(tau_a)
            y_derivatives = np.array([polynomial_derivative(t) for t in grid_points])

            # Test interpolation at multiple points
            test_points = np.array([0.15, 0.3, 0.5, 0.7, 0.85])

            # Compute interpolated values
            interpolated = _evaluate_birkhoff_interpolation_a_form(
                components, y_initial, y_derivatives, test_points
            )

            # Compute exact values
            exact = np.array([polynomial(t) for t in test_points])

            # Should be exactly equal for polynomials (within reasonable numerical error)
            assert_allclose(
                interpolated,
                exact,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"Polynomial t^{degree} not reproduced exactly for N={N}",
            )

    @pytest.mark.parametrize(
        "grid_case",
        ["uniform_on_unit_interval", "uniform_on_symmetric_interval", "simple_non_uniform"],
    )
    def test_equivalence_condition_with_known_weights(self, grid_case):
        """
        Test the fundamental equivalence condition B^a - B^b = w^B using known cases.

        MATHEMATICAL BASIS: This is a core theorem from the Birkhoff paper.
        VERIFICATION: Matrix difference must equal quadrature weights exactly.
        """
        if grid_case == "uniform_on_unit_interval":
            grid_points = np.array([0.0, 0.5, 1.0])
            tau_a, tau_b = 0.0, 1.0
        elif grid_case == "uniform_on_symmetric_interval":
            grid_points = np.array([-1.0, 0.0, 1.0])
            tau_a, tau_b = -1.0, 1.0
        else:  # simple_non_uniform
            grid_points = np.array([0.1, 0.4, 0.8])
            tau_a, tau_b = 0.0, 1.0

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)

        # Test equivalence condition: B^a - B^b = w^B ⊗ 1^T
        matrix_a = components.birkhoff_matrix_a
        matrix_b = components.birkhoff_matrix_b
        weights = components.birkhoff_quadrature_weights

        # Compute difference matrix
        difference = matrix_a - matrix_b

        # Expected: each row should equal the weights vector
        expected = np.tile(weights, (len(grid_points), 1))

        assert_allclose(
            difference,
            expected,
            rtol=1e-13,
            atol=1e-15,
            err_msg=f"Equivalence condition B^a - B^b = w^B violated for {grid_case}",
        )

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_boundary_conditions_analytical_verification(self, N):
        """
        Test boundary conditions using simple grids where we can verify analytically.

        MATHEMATICAL BASIS: B_j^a(τ^a) = 0 and B_j^b(τ^b) = 0 for all j.
        VERIFICATION: Direct evaluation at boundary points must be zero.
        """
        # Use grid that includes the boundary points for direct testing
        grid_points = np.linspace(0.0, 1.0, N + 1)
        tau_a, tau_b = 0.0, 1.0

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)

        # Test a-form boundary condition: B_j^a(τ^a) = 0
        # Since τ^a = grid_points[0], we check the first row of matrix_a
        first_row_a = components.birkhoff_matrix_a[0, :]
        assert_allclose(
            first_row_a,
            np.zeros_like(first_row_a),
            atol=1e-14,
            err_msg=f"a-form boundary condition violated for N={N}",
        )

        # Test b-form boundary condition: B_j^b(τ^b) = 0
        # Since τ^b = grid_points[-1], we check the last row of matrix_b
        last_row_b = components.birkhoff_matrix_b[-1, :]
        assert_allclose(
            last_row_b,
            np.zeros_like(last_row_b),
            atol=1e-14,
            err_msg=f"b-form boundary condition violated for N={N}",
        )

    @pytest.mark.parametrize("N", [2, 3, 4])
    def test_quadrature_exactness_against_analytical_integrals(self, N):
        """
        Test quadrature weight computation against analytical integration formulas.

        MATHEMATICAL BASIS: w^B_j = ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ for Lagrange basis functions.
        VERIFICATION: Quadrature must integrate monomials exactly up to degree N.
        """
        # Use simple uniform grid for analytical verification
        grid_points = np.linspace(0.0, 1.0, N + 1)
        tau_a, tau_b = 0.0, 1.0

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)
        weights = components.birkhoff_quadrature_weights

        # Test integration of monomials t^k for k = 0, 1, ..., N
        for k in range(N + 1):
            # Analytical integral: ∫_0^1 t^k dt = 1/(k+1)
            analytical_integral = 1.0 / (k + 1)

            # Quadrature approximation: Σ w_j * (τ_j)^k
            monomial_values = grid_points**k
            quadrature_integral = np.dot(weights, monomial_values)

            assert_allclose(
                quadrature_integral,
                analytical_integral,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"Quadrature failed for monomial t^{k} with N={N}",
            )

    def test_interpolation_reproduction_of_simple_functions(self):
        """
        Test interpolation against known simple functions with analytical derivatives.

        MATHEMATICAL BASIS: Interpolation should reproduce the derivative conditions exactly.
        VERIFICATION: Use functions where we know derivatives analytically.
        """
        # Use simple 3-point grid
        grid_points = np.array([0.0, 0.5, 1.0])
        tau_a, tau_b = 0.0, 1.0

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)

        # Test with quadratic function: f(t) = t^2 + 2t + 1
        def test_function(t):
            return t**2 + 2 * t + 1

        def test_derivative(t):
            return 2 * t + 2

        # Set up interpolation
        y_initial = test_function(tau_a)  # f(0) = 1
        y_derivatives = np.array([test_derivative(t) for t in grid_points])  # [2, 3, 4]

        # Test interpolation condition at initial point
        interpolated_at_initial = _evaluate_birkhoff_interpolation_a_form(
            components, y_initial, y_derivatives, np.array([tau_a])
        )

        exact_at_initial = test_function(tau_a)
        assert_allclose(
            interpolated_at_initial[0],
            exact_at_initial,
            rtol=1e-13,
            atol=1e-15,
            err_msg="Initial point interpolation condition violated",
        )

        # Test that the function is reproduced exactly (since it's degree 2, within our capability)
        test_points = np.array([0.25, 0.75])
        interpolated = _evaluate_birkhoff_interpolation_a_form(
            components, y_initial, y_derivatives, test_points
        )
        exact = np.array([test_function(t) for t in test_points])

        assert_allclose(
            interpolated,
            exact,
            rtol=1e-12,
            atol=1e-14,
            err_msg="Quadratic function not reproduced exactly",
        )

    def test_equivalence_proposition_with_analytical_verification(self):
        """
        Test the equivalence proposition using analytically computed examples.

        MATHEMATICAL BASIS: I^N_a = I^N_b iff y(τ^b) = y(τ^a) + Σ w^B_j ẏ(τ_j).
        VERIFICATION: Construct examples where we can verify this condition analytically.
        """
        # Simple 3-point case for analytical verification
        grid_points = np.array([0.2, 0.5, 0.8])
        tau_a, tau_b = 0.0, 1.0

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)
        weights = components.birkhoff_quadrature_weights

        # Test case 1: Condition satisfied
        y_initial = 2.0
        y_derivatives = np.array([1.0, 2.0, 3.0])

        # Compute y_final using the equivalence condition
        y_final_correct = y_initial + np.dot(weights, y_derivatives)

        # Both interpolants should give identical results
        test_points = np.array([0.3, 0.6])

        a_form = _evaluate_birkhoff_interpolation_a_form(
            components, y_initial, y_derivatives, test_points
        )
        b_form = _evaluate_birkhoff_interpolation_b_form(
            components, y_final_correct, y_derivatives, test_points
        )

        assert_allclose(
            a_form,
            b_form,
            rtol=1e-13,
            atol=1e-15,
            err_msg="a-form and b-form not equivalent when condition satisfied",
        )

        # Test case 2: Condition violated
        y_final_wrong = y_final_correct + 0.1  # Violate the condition

        b_form_wrong = _evaluate_birkhoff_interpolation_b_form(
            components, y_final_wrong, y_derivatives, test_points
        )

        # Should NOT be equal
        max_difference = np.max(np.abs(a_form - b_form_wrong))
        assert max_difference > 1e-6, "a-form and b-form equivalent when condition violated"

    def test_consistency_with_lagrange_interpolation_limit_case(self):
        """
        Test consistency with standard Lagrange interpolation in the limit case.

        MATHEMATICAL BASIS: When derivative data comes from a polynomial,
        Birkhoff should reproduce standard Lagrange interpolation.
        VERIFICATION: Compare with direct Lagrange interpolation of same polynomial.
        """
        # Simple case: 3 points, quadratic polynomial
        grid_points = np.array([0.0, 0.5, 1.0])
        tau_a, tau_b = 0.0, 1.0

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)

        # Use polynomial p(t) = 3t^2 - 2t + 1
        def polynomial(t):
            return 3 * t**2 - 2 * t + 1

        def polynomial_derivative(t):
            return 6 * t - 2

        # Birkhoff interpolation setup
        y_initial = polynomial(tau_a)
        y_derivatives = np.array([polynomial_derivative(t) for t in grid_points])

        # Test points
        test_points = np.array([0.25, 0.75])

        # Birkhoff result
        birkhoff_result = _evaluate_birkhoff_interpolation_a_form(
            components, y_initial, y_derivatives, test_points
        )

        # Direct polynomial evaluation (external truth)
        direct_result = np.array([polynomial(t) for t in test_points])

        assert_allclose(
            birkhoff_result,
            direct_result,
            rtol=1e-12,
            atol=1e-14,
            err_msg="Birkhoff interpolation inconsistent with direct polynomial evaluation",
        )

    def test_cache_consistency_and_deterministic_behavior(self):
        """
        Test that repeated computations give identical results (like Radau cache test).

        MATHEMATICAL BASIS: Mathematical operations should be deterministic.
        VERIFICATION: Multiple computations should yield identical results.
        """
        grid_points = (0.1, 0.3, 0.7, 0.9)
        tau_a, tau_b = 0.0, 1.0

        # Compute twice
        comp1 = _compute_birkhoff_basis_components(grid_points, tau_a, tau_b)
        comp2 = _compute_birkhoff_basis_components(grid_points, tau_a, tau_b)

        # Should be identical due to caching
        assert comp1 is comp2, "Cache not working - should return same object"

        # Verify numerical consistency
        assert_allclose(
            comp1.birkhoff_quadrature_weights,
            comp2.birkhoff_quadrature_weights,
            rtol=1e-15,
            atol=1e-16,
            err_msg="Quadrature weights not consistent",
        )
        assert_allclose(
            comp1.birkhoff_matrix_a,
            comp2.birkhoff_matrix_a,
            rtol=1e-15,
            atol=1e-16,
            err_msg="Matrix A not consistent",
        )
        assert_allclose(
            comp1.birkhoff_matrix_b,
            comp2.birkhoff_matrix_b,
            rtol=1e-15,
            atol=1e-16,
            err_msg="Matrix B not consistent",
        )

    @pytest.mark.parametrize(
        "interval_case",
        [
            (0.0, 1.0),  # Unit interval
            (-1.0, 1.0),  # Symmetric interval
            (2.0, 5.0),  # Shifted interval
        ],
    )
    def test_interval_scaling_correctness(self, interval_case):
        """
        Test that Birkhoff interpolation works correctly on different intervals.

        MATHEMATICAL BASIS: The theory should work on any interval [τ^a, τ^b].
        VERIFICATION: Same relative behavior on different intervals.
        """
        tau_a, tau_b = interval_case

        # Create proportionally equivalent grids
        relative_positions = np.array([0.2, 0.5, 0.8])
        grid_points = tau_a + (tau_b - tau_a) * relative_positions

        components = _compute_birkhoff_basis_components(tuple(grid_points), tau_a, tau_b)

        # Test with simple linear function scaled to interval
        def linear_function(t):
            return 2 * (t - tau_a) / (tau_b - tau_a) + 1  # Maps interval to [1, 3]

        def linear_derivative(t):
            return 2 / (tau_b - tau_a)

        y_initial = linear_function(tau_a)
        y_derivatives = np.array([linear_derivative(t) for t in grid_points])

        # Test interpolation
        test_point = tau_a + 0.3 * (tau_b - tau_a)
        interpolated = _evaluate_birkhoff_interpolation_a_form(
            components, y_initial, y_derivatives, np.array([test_point])
        )

        exact = linear_function(test_point)

        assert_allclose(
            interpolated[0],
            exact,
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"Linear function not reproduced on interval {interval_case}",
        )


if __name__ == "__main__":
    """
    Run mathematical correctness verification.

    This test focuses on mathematical correctness using external verification,
    similar to the Radau mathematical correctness approach.
    """
    import subprocess
    import sys

    print("=" * 80)
    print("BIRKHOFF MATHEMATICAL CORRECTNESS VERIFICATION")
    print("=" * 80)
    print("Testing core mathematical properties against analytical formulas...")
    print("Focus: Mathematical correctness, not numerical precision limits")
    print()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False,
    )

    print()
    print("=" * 80)

    if result.returncode == 0:
        print("🎉 BIRKHOFF MATHEMATICAL CORRECTNESS VERIFIED 🎉")
        print("All core mathematical properties validated against analytical formulas.")
        print("Implementation correctly implements Birkhoff interpolation theory.")
    else:
        print("❌ MATHEMATICAL CORRECTNESS VERIFICATION FAILED")
        print("Core mathematical properties not satisfied - implementation issues detected.")

    print("=" * 80)
    sys.exit(result.returncode)
