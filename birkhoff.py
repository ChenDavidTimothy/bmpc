from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.integrate import quad

from .input_validation import _validate_positive_integer
from .mtor_types import FloatArray
from .utils.constants import NUMERICAL_ZERO


@dataclass
class BirkhoffBasisComponents:
    """Components for Birkhoff interpolation method as defined in Section 2 of the paper.

    Contains both a-form and b-form Birkhoff basis functions, quadrature weights,
    and matrices for arbitrary grid pseudospectral methods following the exact
    mathematical theory from the Birkhoff paper.
    """

    grid_points: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    tau_a: float = 0.0
    tau_b: float = 1.0

    # Lagrange basis functions and antiderivatives (L_j(τ) from Lemma)
    lagrange_antiderivatives_at_tau_a: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    lagrange_antiderivatives_at_tau_b: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # Birkhoff quadrature weights w^B_j := ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ (Definition, Section 2)
    birkhoff_quadrature_weights: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # Birkhoff basis functions B_j^a(τ) and B_j^b(τ) (Lemma, Section 2)
    birkhoff_basis_a: FloatArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    birkhoff_basis_b: FloatArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))

    # Birkhoff matrices B^a and B^b (Definition, Section 2)
    birkhoff_matrix_a: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    birkhoff_matrix_b: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


def _compute_barycentric_weights_birkhoff(nodes: FloatArray) -> FloatArray:
    num_nodes = len(nodes)
    if num_nodes == 1:
        return np.array([1.0], dtype=np.float64)

    nodes_col = nodes[:, np.newaxis]
    nodes_row = nodes[np.newaxis, :]
    differences_matrix = nodes_col - nodes_row

    diagonal_mask = np.eye(num_nodes, dtype=bool)

    near_zero_mask = np.abs(differences_matrix) < NUMERICAL_ZERO
    perturbation = np.sign(differences_matrix) * NUMERICAL_ZERO
    perturbation[perturbation == 0] = NUMERICAL_ZERO

    off_diagonal_near_zero = near_zero_mask & ~diagonal_mask
    differences_matrix = np.where(off_diagonal_near_zero, perturbation, differences_matrix)

    differences_matrix[diagonal_mask] = 1.0

    products = np.prod(differences_matrix, axis=1, dtype=np.float64)

    small_product_mask = np.abs(products) < NUMERICAL_ZERO**2
    safe_products = np.where(
        small_product_mask,
        np.where(products == 0, 1.0 / (NUMERICAL_ZERO**2), np.sign(products) / (NUMERICAL_ZERO**2)),
        1.0 / products,
    )

    return safe_products.astype(np.float64)


def _evaluate_lagrange_basis_at_point(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point: float,
) -> FloatArray:
    num_nodes = len(nodes)

    differences = np.abs(evaluation_point - nodes)
    coincident_mask = differences < NUMERICAL_ZERO

    if np.any(coincident_mask):
        lagrange_values = np.zeros(num_nodes, dtype=np.float64)
        lagrange_values[coincident_mask] = 1.0
        return lagrange_values

    diffs = evaluation_point - nodes

    near_zero_mask = np.abs(diffs) < NUMERICAL_ZERO
    safe_diffs = np.where(
        near_zero_mask, np.where(diffs == 0, NUMERICAL_ZERO, np.sign(diffs) * NUMERICAL_ZERO), diffs
    )

    terms = barycentric_weights / safe_diffs
    sum_terms = np.sum(terms)

    if abs(sum_terms) < NUMERICAL_ZERO:
        return np.zeros(num_nodes, dtype=np.float64)

    normalized_terms = terms / sum_terms
    return cast(FloatArray, normalized_terms)


def _compute_lagrange_antiderivative_at_point(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point: float,
    reference_point: float,
) -> FloatArray:
    num_nodes = len(nodes)
    antiderivatives = np.zeros(num_nodes, dtype=np.float64)

    for j in range(num_nodes):

        def lagrange_j(tau):
            lagrange_vals = _evaluate_lagrange_basis_at_point(nodes, barycentric_weights, tau)
            return lagrange_vals[j]

        if abs(evaluation_point - reference_point) > NUMERICAL_ZERO:
            integral_result, _ = quad(
                lagrange_j, reference_point, evaluation_point, epsabs=1e-12, epsrel=1e-10, limit=200
            )
            antiderivatives[j] = integral_result

    return antiderivatives


def _compute_birkhoff_quadrature_weights(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    tau_a: float,
    tau_b: float,
) -> FloatArray:
    num_nodes = len(nodes)
    weights = np.zeros(num_nodes, dtype=np.float64)

    for j in range(num_nodes):

        def lagrange_j(tau):
            lagrange_vals = _evaluate_lagrange_basis_at_point(nodes, barycentric_weights, tau)
            return lagrange_vals[j]

        integral_result, _ = quad(lagrange_j, tau_a, tau_b, epsabs=1e-12, epsrel=1e-10, limit=200)
        weights[j] = integral_result

    return weights


def _compute_birkhoff_basis_functions(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    tau_a: float,
    tau_b: float,
    evaluation_points: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    num_nodes = len(nodes)
    num_eval_points = len(evaluation_points)

    basis_a = np.zeros((num_eval_points, num_nodes), dtype=np.float64)
    basis_b = np.zeros((num_eval_points, num_nodes), dtype=np.float64)

    reference_point = tau_a

    antiderivatives_at_tau_a = _compute_lagrange_antiderivative_at_point(
        nodes, barycentric_weights, tau_a, reference_point
    )
    antiderivatives_at_tau_b = _compute_lagrange_antiderivative_at_point(
        nodes, barycentric_weights, tau_b, reference_point
    )

    for i, tau in enumerate(evaluation_points):
        antiderivatives_at_tau = _compute_lagrange_antiderivative_at_point(
            nodes, barycentric_weights, tau, reference_point
        )

        basis_a[i, :] = antiderivatives_at_tau - antiderivatives_at_tau_a
        basis_b[i, :] = antiderivatives_at_tau - antiderivatives_at_tau_b

    return basis_a, basis_b


def _construct_birkhoff_matrices(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    tau_a: float,
    tau_b: float,
) -> tuple[FloatArray, FloatArray]:
    basis_a, basis_b = _compute_birkhoff_basis_functions(
        nodes, barycentric_weights, tau_a, tau_b, nodes
    )

    return basis_a, basis_b


@functools.lru_cache(maxsize=32)
def _compute_birkhoff_basis_components(
    grid_points_tuple: tuple[float, ...],
    tau_a: float,
    tau_b: float,
) -> BirkhoffBasisComponents:
    """Compute complete Birkhoff basis components for arbitrary grid.

    Implements the complete mathematical theory from Section 2 of the Birkhoff paper
    for pseudospectral methods using arbitrary grids.

    Mathematical foundations:
    1. Lagrange basis functions ℓ_j(τ) with ℓ_j(τ_i) = δ_{ij}
    2. Antiderivatives L_j(τ) = ∫ ℓ_j(s) ds
    3. Birkhoff basis functions: B_j^a(τ) = L_j(τ) - L_j(τ^a), B_j^b(τ) = L_j(τ) - L_j(τ^b)
    4. Quadrature weights: w^B_j = ∫_{τ^a}^{τ^b} ℓ_j(τ) dτ
    5. Equivalence condition: B^a_j(τ) - B^b_j(τ) = w^B_j

    Args:
        grid_points_tuple: Arbitrary grid points π^N = {τ_0, τ_1, ..., τ_N}
        tau_a: Left endpoint of interval [τ^a, τ^b]
        tau_b: Right endpoint of interval [τ^a, τ^b]

    Returns:
        BirkhoffBasisComponents with complete mathematical objects

    Raises:
        ValueError: If grid points violate mathematical requirements
    """
    _validate_positive_integer(len(grid_points_tuple), "number of grid points")

    grid_points = np.array(grid_points_tuple, dtype=np.float64)

    if tau_b <= tau_a:
        raise ValueError(f"Interval endpoints must satisfy τ^a < τ^b, got τ^a={tau_a}, τ^b={tau_b}")

    if not np.all(np.diff(grid_points) > 0):
        raise ValueError("Grid points must be strictly increasing: τ_0 < τ_1 < ... < τ_N")
    if grid_points[0] < tau_a or grid_points[-1] > tau_b:
        raise ValueError("Grid points must satisfy τ^a ≤ τ_0 < ... < τ_N ≤ τ^b")

    barycentric_weights = _compute_barycentric_weights_birkhoff(grid_points)

    birkhoff_quadrature_weights = _compute_birkhoff_quadrature_weights(
        grid_points, barycentric_weights, tau_a, tau_b
    )

    reference_point = tau_a

    antiderivatives_at_tau_a = _compute_lagrange_antiderivative_at_point(
        grid_points, barycentric_weights, tau_a, reference_point
    )
    antiderivatives_at_tau_b = _compute_lagrange_antiderivative_at_point(
        grid_points, barycentric_weights, tau_b, reference_point
    )

    birkhoff_matrix_a, birkhoff_matrix_b = _construct_birkhoff_matrices(
        grid_points, barycentric_weights, tau_a, tau_b
    )

    basis_a, basis_b = _compute_birkhoff_basis_functions(
        grid_points, barycentric_weights, tau_a, tau_b, grid_points
    )

    return BirkhoffBasisComponents(
        grid_points=grid_points,
        tau_a=tau_a,
        tau_b=tau_b,
        lagrange_antiderivatives_at_tau_a=antiderivatives_at_tau_a,
        lagrange_antiderivatives_at_tau_b=antiderivatives_at_tau_b,
        birkhoff_quadrature_weights=birkhoff_quadrature_weights,
        birkhoff_basis_a=basis_a,
        birkhoff_basis_b=basis_b,
        birkhoff_matrix_a=birkhoff_matrix_a,
        birkhoff_matrix_b=birkhoff_matrix_b,
    )


def _evaluate_birkhoff_interpolation_a_form(
    basis_components: BirkhoffBasisComponents,
    y_initial: float,
    y_derivatives: FloatArray,
    evaluation_points: FloatArray,
) -> FloatArray:
    if len(y_derivatives) != len(basis_components.grid_points):
        raise ValueError(
            f"y_derivatives length {len(y_derivatives)} must match grid points {len(basis_components.grid_points)}"
        )

    num_eval_points = len(evaluation_points)
    interpolated_values = np.zeros(num_eval_points, dtype=np.float64)

    nodes = basis_components.grid_points
    barycentric_weights = _compute_barycentric_weights_birkhoff(nodes)
    basis_a, _ = _compute_birkhoff_basis_functions(
        nodes,
        barycentric_weights,
        basis_components.tau_a,
        basis_components.tau_b,
        evaluation_points,
    )

    for i in range(num_eval_points):
        interpolated_values[i] = y_initial + np.dot(basis_a[i, :], y_derivatives)

    return interpolated_values


def _evaluate_birkhoff_interpolation_b_form(
    basis_components: BirkhoffBasisComponents,
    y_final: float,
    y_derivatives: FloatArray,
    evaluation_points: FloatArray,
) -> FloatArray:
    if len(y_derivatives) != len(basis_components.grid_points):
        raise ValueError(
            f"y_derivatives length {len(y_derivatives)} must match grid points {len(basis_components.grid_points)}"
        )

    num_eval_points = len(evaluation_points)
    interpolated_values = np.zeros(num_eval_points, dtype=np.float64)

    nodes = basis_components.grid_points
    barycentric_weights = _compute_barycentric_weights_birkhoff(nodes)
    _, basis_b = _compute_birkhoff_basis_functions(
        nodes,
        barycentric_weights,
        basis_components.tau_a,
        basis_components.tau_b,
        evaluation_points,
    )

    for i in range(num_eval_points):
        interpolated_values[i] = np.dot(basis_b[i, :], y_derivatives) + y_final

    return interpolated_values
