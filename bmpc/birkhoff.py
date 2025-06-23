from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad

from bmpc.constants import NUMERICAL_ZERO

from .bmpc_types import FloatArray


@dataclass
class BirkhoffBasisComponents:
    grid_points: FloatArray = field(default_factory=lambda: np.array([], dtype=np.float64))
    tau_a: float = 0.0
    tau_b: float = 1.0
    lagrange_antiderivatives_at_tau_b: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    birkhoff_quadrature_weights: FloatArray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    birkhoff_matrix_a: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    birkhoff_matrix_b: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


def _sanitize_diffs(diffs: FloatArray) -> FloatArray:
    return np.where(np.abs(diffs) < NUMERICAL_ZERO, np.sign(diffs) * NUMERICAL_ZERO, diffs)


def _compute_barycentric_weights_birkhoff(nodes: FloatArray) -> FloatArray:
    num_nodes = len(nodes)
    if num_nodes == 1:
        return np.array([1.0], dtype=np.float64)

    differences_matrix = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(differences_matrix, 1.0)

    near_zero = (np.abs(differences_matrix) < NUMERICAL_ZERO) & ~np.eye(num_nodes, dtype=bool)
    differences_matrix = np.where(
        near_zero, np.sign(differences_matrix) * NUMERICAL_ZERO, differences_matrix
    )

    products = np.prod(differences_matrix, axis=1, dtype=np.float64)
    safe_products = np.where(
        np.abs(products) < NUMERICAL_ZERO**2,
        np.sign(products) / (NUMERICAL_ZERO**2),
        1.0 / products,
    )
    return safe_products.astype(np.float64)


def _evaluate_lagrange_basis_at_point(
    nodes: FloatArray,
    barycentric_weights: FloatArray,
    evaluation_point: float,
) -> FloatArray:
    differences = np.abs(evaluation_point - nodes)
    if np.any(differences < NUMERICAL_ZERO):
        lagrange_values = np.zeros_like(nodes)
        lagrange_values[differences < NUMERICAL_ZERO] = 1.0
        return lagrange_values

    diffs = _sanitize_diffs(evaluation_point - nodes)
    terms = barycentric_weights / diffs
    return terms / np.sum(terms)


def _make_lagrange_j(j: int, nodes: FloatArray, weights: FloatArray):
    return lambda tau: _evaluate_lagrange_basis_at_point(nodes, weights, tau)[j]


def _integrate_lagrange_basis_set(
    nodes: FloatArray,
    weights: FloatArray,
    lower: float,
    upper: float,
) -> FloatArray:
    results = np.zeros(len(nodes), dtype=np.float64)
    if abs(upper - lower) < NUMERICAL_ZERO:
        return results

    for j in range(len(nodes)):
        results[j], _ = quad(
            _make_lagrange_j(j, nodes, weights),
            lower,
            upper,
            epsabs=1e-12,
            epsrel=1e-10,
            limit=200,
        )
    return results


def _compute_birkhoff_basis_functions(
    nodes: FloatArray,
    weights: FloatArray,
    tau_a: float,
    antiderivatives_at_tau_b: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    basis_a = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    basis_b = np.zeros_like(basis_a)

    for i, tau in enumerate(nodes):
        antideriv = _integrate_lagrange_basis_set(nodes, weights, tau_a, tau)
        basis_a[i, :] = antideriv
        basis_b[i, :] = antideriv - antiderivatives_at_tau_b

    return basis_a, basis_b


@functools.lru_cache(maxsize=32)
def _compute_birkhoff_basis_components(
    grid_points_tuple: tuple[float, ...],
    tau_a: float,
    tau_b: float,
) -> BirkhoffBasisComponents:
    grid_points = np.array(grid_points_tuple, dtype=np.float64)

    if tau_b <= tau_a:
        raise ValueError(f"Interval endpoints must satisfy τ^a < τ^b, got τ^a={tau_a}, τ^b={tau_b}")
    if not np.all(np.diff(grid_points) > 0):
        raise ValueError("Grid points must be strictly increasing: τ_0 < τ_1 < ... < τ_N")
    if grid_points[0] < tau_a or grid_points[-1] > tau_b:
        raise ValueError("Grid points must satisfy τ^a ≤ τ_0 < ... < τ_N ≤ τ^b")

    weights = _compute_barycentric_weights_birkhoff(grid_points)
    antiderivatives_at_tau_b = _integrate_lagrange_basis_set(grid_points, weights, tau_a, tau_b)
    matrix_a, matrix_b = _compute_birkhoff_basis_functions(
        grid_points, weights, tau_a, antiderivatives_at_tau_b
    )
    quadrature_weights = antiderivatives_at_tau_b.copy()

    return BirkhoffBasisComponents(
        grid_points=grid_points,
        tau_a=tau_a,
        tau_b=tau_b,
        lagrange_antiderivatives_at_tau_b=antiderivatives_at_tau_b,
        birkhoff_quadrature_weights=quadrature_weights,
        birkhoff_matrix_a=matrix_a,
        birkhoff_matrix_b=matrix_b,
    )
