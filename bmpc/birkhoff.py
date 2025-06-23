from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad

from bmpc.constants import NUMERICAL_ZERO

from .bmpc_types import FloatArray


def empty_float_array() -> FloatArray:
    return np.array([], dtype=np.float64)


def empty_float_matrix() -> FloatArray:
    return np.empty((0, 0), dtype=np.float64)


@dataclass
class BirkhoffBasisComponents:
    """Components for Birkhoff interpolation method as defined in Section 2 of the paper."""

    grid_points: FloatArray = field(default_factory=empty_float_array)
    tau_a: float = 0.0
    tau_b: float = 1.0

    lagrange_antiderivatives_at_tau_a: FloatArray = field(default_factory=empty_float_array)
    lagrange_antiderivatives_at_tau_b: FloatArray = field(default_factory=empty_float_array)

    birkhoff_quadrature_weights: FloatArray = field(default_factory=empty_float_array)

    birkhoff_matrix_a: FloatArray = field(default_factory=empty_float_matrix)
    birkhoff_matrix_b: FloatArray = field(default_factory=empty_float_matrix)


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
        np.where(
            products == 0,
            1.0 / (NUMERICAL_ZERO**2),
            np.sign(products) / (NUMERICAL_ZERO**2),
        ),
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
    safe_diffs = np.where(near_zero_mask, np.sign(diffs) * NUMERICAL_ZERO, diffs)

    terms = barycentric_weights / safe_diffs
    sum_terms = np.sum(terms)

    if abs(sum_terms) < NUMERICAL_ZERO:
        return np.zeros(num_nodes, dtype=np.float64)

    return terms / sum_terms


def _compute_lagrange_antiderivative_at_point(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point: float,
    reference_point: float = 0.0,
) -> FloatArray:
    num_nodes = len(nodes)
    antiderivatives = np.zeros(num_nodes, dtype=np.float64)

    for j in range(num_nodes):

        def lagrange_j(tau):
            return _evaluate_lagrange_basis_at_point(nodes, barycentric_weights, tau)[j]

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
            return _evaluate_lagrange_basis_at_point(nodes, barycentric_weights, tau)[j]

        integral_result, _ = quad(lagrange_j, tau_a, tau_b, epsabs=1e-12, epsrel=1e-10, limit=200)
        weights[j] = integral_result

    return weights


def _compute_birkhoff_basis_functions(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    tau_a: float,
    tau_b: float,
    evaluation_points: FloatArray,
    antiderivatives_at_tau_a: FloatArray,
    antiderivatives_at_tau_b: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    num_nodes = len(nodes)
    num_eval_points = len(evaluation_points)

    basis_a = np.zeros((num_eval_points, num_nodes), dtype=np.float64)
    basis_b = np.zeros((num_eval_points, num_nodes), dtype=np.float64)

    for i, tau in enumerate(evaluation_points):
        antiderivatives_at_tau = _compute_lagrange_antiderivative_at_point(
            nodes, barycentric_weights, tau, reference_point=tau_a
        )
        basis_a[i, :] = antiderivatives_at_tau - antiderivatives_at_tau_a
        basis_b[i, :] = antiderivatives_at_tau - antiderivatives_at_tau_b

    return basis_a, basis_b


@functools.lru_cache(maxsize=32)
def _compute_birkhoff_basis_components(
    grid_points_tuple: tuple[float, ...],
    tau_a: float,
    tau_b: float,
) -> BirkhoffBasisComponents:
    """Compute complete Birkhoff basis components for arbitrary grid."""

    grid_points = np.array(grid_points_tuple, dtype=np.float64)

    if tau_b <= tau_a:
        raise ValueError(f"Interval endpoints must satisfy τ^a < τ^b, got τ^a={tau_a}, τ^b={tau_b}")

    if not np.all(np.diff(grid_points) > 0):
        raise ValueError("Grid points must be strictly increasing: τ_0 < τ_1 < ... < τ_N")
    if grid_points[0] < tau_a or grid_points[-1] > tau_b:
        raise ValueError("Grid points must satisfy τ^a ≤ τ_0 < ... < τ_N ≤ τ^b")

    barycentric_weights = _compute_barycentric_weights_birkhoff(grid_points)

    antiderivatives_at_tau_a = _compute_lagrange_antiderivative_at_point(
        grid_points, barycentric_weights, tau_a, reference_point=tau_a
    )
    antiderivatives_at_tau_b = _compute_lagrange_antiderivative_at_point(
        grid_points, barycentric_weights, tau_b, reference_point=tau_a
    )

    birkhoff_matrix_a, birkhoff_matrix_b = _compute_birkhoff_basis_functions(
        grid_points,
        barycentric_weights,
        tau_a,
        tau_b,
        grid_points,
        antiderivatives_at_tau_a,
        antiderivatives_at_tau_b,
    )

    birkhoff_quadrature_weights = _compute_birkhoff_quadrature_weights(
        grid_points, barycentric_weights, tau_a, tau_b
    )

    return BirkhoffBasisComponents(
        grid_points=grid_points,
        tau_a=tau_a,
        tau_b=tau_b,
        lagrange_antiderivatives_at_tau_a=antiderivatives_at_tau_a,
        lagrange_antiderivatives_at_tau_b=antiderivatives_at_tau_b,
        birkhoff_quadrature_weights=birkhoff_quadrature_weights,
        birkhoff_matrix_a=birkhoff_matrix_a,
        birkhoff_matrix_b=birkhoff_matrix_b,
    )
