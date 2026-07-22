"""
Performance metrics for multi-objective optimization.

Contains:
- Hypervolume (HV) calculation
- C-metric (Coverage metric) calculation
- Reference point determination
- Normalization utilities

Strictly follows the paper's Section V-A.
"""

import numpy as np
from typing import List, Tuple, Optional
from problem import Solution


def compute_hypervolume(solutions: List[Solution],
                        reference_point: Tuple[float, float],
                        normalize: bool = True) -> float:
    """
    Compute hypervolume (HV) of a solution set.

    HV measures the volume dominated by the solution set with respect
    to a reference point. Larger HV = better convergence and diversity.

    The paper normalizes objectives to eliminate scale differences
    and selects reference point from worst objective values.

    For minimization: f1 = cost, f2 = -satisfaction.
    Reference point should be larger than any point in both objectives.

    Args:
        solutions: List of non-dominated solutions
        reference_point: (ref_f1, ref_f2) - worst values in each dimension
        normalize: Whether objectives are already normalized

    Returns:
        Hypervolume value
    """
    if not solutions:
        return 0.0

    # Extract objective values as minimization
    points = []
    for sol in solutions:
        if sol._cost is not None and sol._satisfaction is not None:
            f1 = sol._cost
            f2 = -sol._satisfaction  # Convert to minimization
            points.append([f1, f2])

    if not points:
        return 0.0

    points = np.array(points)
    ref = np.array(reference_point)

    # Normalize if requested
    if normalize and len(points) > 0:
        # Find ideal and nadir points from the set
        ideal = np.min(points, axis=0)
        nadir = np.max(points, axis=0)

        # Avoid division by zero
        denom = nadir - ideal
        denom[denom == 0] = 1.0

        points = (points - ideal) / denom
        ref = (ref - ideal) / denom

    # Compute HV using the recursive method for 2D
    return _hypervolume_2d(points, ref)


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute 2D hypervolume efficiently.

    For 2 objectives: sort by f1, then sum rectangular areas.
    """
    if len(points) == 0:
        return 0.0

    # Filter points that are dominated by reference point
    mask = (points[:, 0] <= ref[0]) & (points[:, 1] <= ref[1])
    points = points[mask]

    if len(points) == 0:
        return 0.0

    # Sort by first objective ascending
    sorted_idx = np.argsort(points[:, 0])
    points = points[sorted_idx]

    hv = 0.0
    prev_f1 = 0.0  # For normalized: ideal point is (0, 0)

    for i in range(len(points)):
        f1, f2 = points[i]

        # Height from this point's f2 to ref f2
        height = ref[1] - f2
        if height <= 0:
            continue

        # Width from prev_f1 to current f1
        if i == 0:
            width = f1 - 0  # From ideal point
        else:
            width = f1 - points[i - 1, 0]

        if width > 0:
            hv += width * height

    # Add width from last point to ref
    last_width = ref[0] - points[-1, 0]
    if last_width > 0:
        hv += last_width * (ref[1] - np.min(points[:, 1]))

    return hv


def get_reference_point(all_solutions_across_runs: List,
                        scale_factor: float = 1.1) -> Tuple[float, float]:
    """
    Determine reference point for HV calculation (Section V-A).

    Uses the worst objective values from all solutions obtained by
    all algorithms in all runs.

    Args:
        all_solutions_across_runs: List of solution lists (one per run),
                                   or list of list-of-lists (one per algorithm).
        scale_factor: Multiply worst values by this factor (1.1 = 10% worse)

    Returns:
        (ref_f1, ref_f2) reference point (in minimization space: f1=cost, f2=-satisfaction)
    """
    max_cost = 0.0
    min_sat = float('inf')

    def extract(container):
        nonlocal max_cost, min_sat
        if isinstance(container, list):
            for item in container:
                if hasattr(item, '_cost'):
                    # It's a Solution
                    if item._cost is not None:
                        max_cost = max(max_cost, item._cost)
                    if item._satisfaction is not None:
                        min_sat = min(min_sat, item._satisfaction)
                elif isinstance(item, list):
                    extract(item)

    extract(all_solutions_across_runs)

    ref_f1 = max_cost * scale_factor if max_cost > 0 else scale_factor
    ref_f2 = -(min_sat / scale_factor) if min_sat > 0 else -min_sat * scale_factor
    if min_sat <= 0:
        ref_f2 = -min_sat * scale_factor
    if ref_f2 >= 0:
        ref_f2 = -0.1  # Ensure reference point is worse than any solution

    return (ref_f1, ref_f2)


def compute_c_metric(set_a: List[Solution], set_b: List[Solution]) -> float:
    """
    Compute C-metric C(A, B) (Section V-A).

    C(A, B) = percentage of solutions in B that are dominated by
    or equal to at least one solution in A.

    If C(A, B) = 1, every solution in B is dominated by some solution in A.
    If C(A, B) = 0, no solution in B is dominated by any solution in A.

    Note: C(A, B) is NOT necessarily equal to 1 - C(B, A).
    """
    if not set_b:
        return 0.0

    count_dominated = 0
    for sol_b in set_b:
        for sol_a in set_a:
            # Check if sol_a dominates or equals sol_b
            if (sol_a._cost <= sol_b._cost and
                sol_a._satisfaction >= sol_b._satisfaction and
                (sol_a._cost < sol_b._cost or sol_a._satisfaction > sol_b._satisfaction)):
                count_dominated += 1
                break  # sol_b is dominated by at least one solution in A

    return count_dominated / len(set_b)


def find_best_compromise_solution(pf: List[Solution],
                                  ideal_point: Optional[Tuple[float, float]] = None
                                  ) -> Solution:
    """
    Find the best-compromise solution on the Pareto front.

    Uses the minimum distance to the ideal point method (Section V-C.2).
    Each objective is normalized, then Euclidean distance to ideal point
    is computed. The solution with shortest distance is selected.

    Args:
        pf: Pareto front solutions
        ideal_point: (ideal_cost, ideal_satisfaction). If None, derived from pf.

    Returns:
        Best-compromise solution
    """
    if not pf:
        return None

    if ideal_point is None:
        ideal_cost = min(s._cost for s in pf if s._cost is not None)
        ideal_sat = max(s._satisfaction for s in pf if s._satisfaction is not None)
        ideal_point = (ideal_cost, ideal_sat)

    # Find nadir point for normalization
    nadir_cost = max(s._cost for s in pf if s._cost is not None)
    nadir_sat = min(s._satisfaction for s in pf if s._satisfaction is not None)

    best_sol = None
    best_dist = float('inf')

    for sol in pf:
        if sol._cost is None or sol._satisfaction is None:
            continue

        # Normalize
        norm_cost = (sol._cost - ideal_point[0]) / (nadir_cost - ideal_point[0]) if nadir_cost != ideal_point[0] else 0
        norm_sat = (ideal_point[1] - sol._satisfaction) / (ideal_point[1] - nadir_sat) if ideal_point[1] != nadir_sat else 0

        # Euclidean distance to ideal point (0, 0 after normalization)
        dist = np.sqrt(norm_cost**2 + norm_sat**2)

        if dist < best_dist:
            best_dist = dist
            best_sol = sol

    return best_sol


def find_extreme_solutions(pf: List[Solution]) -> Tuple[Solution, Solution]:
    """
    Find extreme solutions on the Pareto front.

    Returns:
        (best_cost_solution, best_satisfaction_solution)
    """
    if not pf:
        return None, None

    best_cost_sol = min(pf, key=lambda s: s._cost if s._cost is not None else float('inf'))
    best_sat_sol = max(pf, key=lambda s: s._satisfaction if s._satisfaction is not None else -float('inf'))

    return best_cost_sol, best_sat_sol
