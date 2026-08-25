from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GlobalNlsResult:
    positions: np.ndarray
    iterations: int
    stress: float
    runtime_s: float


def shortest_path_matrix(node_count: int, ranges: dict[Edge, float], fill_scale: float = 1.25) -> np.ndarray:
    matrix = np.full((node_count, node_count), np.inf, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    for (i, j), distance in ranges.items():
        if distance > 0.0 and math.isfinite(distance):
            matrix[i, j] = matrix[j, i] = min(matrix[i, j], float(distance))
    for k in range(node_count):
        matrix = np.minimum(matrix, matrix[:, [k]] + matrix[[k], :])
    finite = matrix[np.isfinite(matrix)]
    fill = float(np.max(finite) * fill_scale) if finite.size else 1.0
    matrix[~np.isfinite(matrix)] = fill
    return matrix


def classical_mds(distance_matrix: np.ndarray, dimensions: int = 2) -> np.ndarray:
    node_count = distance_matrix.shape[0]
    centering = np.eye(node_count) - np.ones((node_count, node_count)) / node_count
    gram = -0.5 * centering @ (distance_matrix * distance_matrix) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    selected = np.argsort(eigenvalues)[::-1][:dimensions]
    return eigenvectors[:, selected] * np.sqrt(np.maximum(eigenvalues[selected], 0.0))


def align_to_anchors(positions: np.ndarray, anchors: dict[int, tuple[float, float]]) -> np.ndarray:
    anchor_ids = sorted(anchors)
    if not anchor_ids:
        raise ValueError("at least one anchor is required")
    target = np.array([anchors[node_id] for node_id in anchor_ids], dtype=float)
    source = positions[anchor_ids]
    if len(anchor_ids) == 1:
        return positions + target[0] - source[0]
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.norm(source_centered) < 1e-12:
        return positions + target_mean - source_mean
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    rotated = source_centered @ rotation
    scale = np.sum(rotated * target_centered) / max(np.sum(rotated * rotated), 1e-12)
    return (positions - source_mean) @ rotation * scale + target_mean


def estimate_positions(
    node_count: int,
    ranges: dict[Edge, float],
    anchors: dict[int, tuple[float, float]],
    *,
    range_std: dict[Edge, float] | None = None,
    weighted: bool = False,
    max_iterations: int = 24,
    damping: float = 1e-1,
) -> GlobalNlsResult:
    """Estimate positions from reports and known anchors without truth coordinates."""
    if not ranges:
        raise ValueError("at least one range report is required")
    adjacency = {node_id: set() for node_id in range(node_count)}
    for i, j in ranges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    reachable = set(anchors)
    frontier = list(anchors)
    while frontier:
        node_id = frontier.pop()
        for neighbor in adjacency[node_id] - reachable:
            reachable.add(neighbor)
            frontier.append(neighbor)
    missing = sorted(set(range(node_count)) - reachable)
    if missing:
        raise ValueError(f"nodes are not connected to any anchor: {missing}")
    start_time = time.perf_counter()
    initial = classical_mds(shortest_path_matrix(node_count, ranges))
    positions = align_to_anchors(initial, anchors)
    for node_id, coordinate in anchors.items():
        positions[node_id] = coordinate

    edges = sorted(ranges)
    unknowns = [node_id for node_id in range(node_count) if node_id not in anchors]
    variable_index = {node_id: index for index, node_id in enumerate(unknowns)}
    iterations = 0

    def objective(candidate: np.ndarray) -> float:
        values = []
        for edge in edges:
            i, j = edge
            residual = float(np.linalg.norm(candidate[i] - candidate[j])) - ranges[edge]
            sigma = max(float((range_std or {}).get(edge, 1.0)), 1e-9) if weighted else 1.0
            values.append((residual / sigma) ** 2)
        return float(np.mean(values)) if values else float("inf")

    current_damping = damping
    for _ in range(max_iterations):
        rows: list[np.ndarray] = []
        residuals: list[float] = []
        for edge in edges:
            i, j = edge
            difference = positions[i] - positions[j]
            distance = max(float(np.linalg.norm(difference)), 1e-9)
            direction = difference / distance
            sigma = max(float((range_std or {}).get(edge, 1.0)), 1e-9) if weighted else 1.0
            row = np.zeros(2 * len(unknowns), dtype=float)
            if i in variable_index:
                offset = 2 * variable_index[i]
                row[offset : offset + 2] = direction / sigma
            if j in variable_index:
                offset = 2 * variable_index[j]
                row[offset : offset + 2] = -direction / sigma
            rows.append(row)
            residuals.append((distance - ranges[edge]) / sigma)
        if not rows or not unknowns:
            break

        jacobian = np.vstack(rows)
        residual_vector = np.array(residuals, dtype=float)
        lhs = jacobian.T @ jacobian
        rhs = jacobian.T @ residual_vector
        base_objective = objective(positions)
        accepted = False
        trial_damping = current_damping
        delta = np.zeros(2 * len(unknowns), dtype=float)
        for _attempt in range(8):
            system = lhs + trial_damping * np.eye(lhs.shape[0])
            try:
                delta = -np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                delta = -np.linalg.pinv(system) @ rhs
            candidate = positions.copy()
            for node_id in unknowns:
                offset = 2 * variable_index[node_id]
                candidate[node_id] += delta[offset : offset + 2]
            for node_id, coordinate in anchors.items():
                candidate[node_id] = coordinate
            if objective(candidate) <= base_objective:
                positions = candidate
                current_damping = max(trial_damping * 0.35, 1e-8)
                accepted = True
                break
            trial_damping *= 10.0
        iterations += 1
        if not accepted:
            current_damping = min(trial_damping, 1e8)
            break
        if float(np.linalg.norm(delta)) < 1e-5:
            break

    return GlobalNlsResult(
        positions=positions,
        iterations=iterations,
        stress=objective(positions),
        runtime_s=time.perf_counter() - start_time,
    )


def conditional_fim(
    positions: np.ndarray,
    range_std: dict[Edge, float],
    anchor_ids: set[int],
) -> tuple[np.ndarray, int, float, float]:
    """Compute a plug-in range FIM at supplied true or estimated positions."""
    unknowns = [node_id for node_id in range(len(positions)) if node_id not in anchor_ids]
    variable_index = {node_id: index for index, node_id in enumerate(unknowns)}
    matrix = np.zeros((2 * len(unknowns), 2 * len(unknowns)), dtype=float)
    for (i, j), sigma_value in range_std.items():
        if i in anchor_ids and j in anchor_ids:
            continue
        difference = positions[i] - positions[j]
        distance = max(float(np.linalg.norm(difference)), 1e-9)
        direction = difference / distance
        block = np.outer(direction, direction) / max(float(sigma_value) ** 2, 1e-12)
        for left, right, sign in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
            if left not in variable_index or right not in variable_index:
                continue
            row = 2 * variable_index[left]
            column = 2 * variable_index[right]
            matrix[row : row + 2, column : column + 2] += sign * block
    eigenvalues = np.linalg.eigvalsh(matrix) if matrix.size else np.array([])
    tolerance = max(1e-9, float(np.max(eigenvalues)) * 1e-10) if eigenvalues.size else 1e-9
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(positive.size)
    lambda_min = float(np.min(positive)) if positive.size else 0.0
    if rank < matrix.shape[0]:
        bound = float("inf")
    else:
        bound = math.sqrt(float(np.trace(np.linalg.pinv(matrix, rcond=1e-10))) / max(len(unknowns), 1))
    return matrix, rank, lambda_min, bound


def unknown_node_rmse(estimated: np.ndarray, truth: np.ndarray, anchor_ids: set[int]) -> float:
    unknowns = [node_id for node_id in range(len(truth)) if node_id not in anchor_ids]
    if not unknowns:
        return 0.0
    error = estimated[unknowns] - truth[unknowns]
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
