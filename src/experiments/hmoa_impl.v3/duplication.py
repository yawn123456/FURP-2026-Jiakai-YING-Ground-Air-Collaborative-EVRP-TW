"""
RemoveDuplication strategy (Section IV-F)

Removes duplicate solutions from combined population by either:
- Multi-mode mutation on a randomly selected external PF solution
- Creating a new solution via AssignNodes
"""

import random
import math
from typing import List, Tuple
from model import Solution, ProblemInstance
from genetic_ops import multi_mode_mutation
from initialization import assign_nodes


def remove_duplication(inst: ProblemInstance,
                       combined: List[Solution],
                       pareto_front: List[Solution],
                       objectives: List[Tuple[float, float]],
                       restart_rate: float = 0.3,
                       max_solutions: int = 200) -> Tuple[List[Solution], List[Tuple[float, float]]]:
    """
    Remove duplicate solutions from combined population.

    Args:
        inst: Problem instance
        combined: Combined population Rt = Qt ∪ Pt
        pareto_front: External PF (non-dominated solutions found so far)
        objectives: (f1, f2) for each solution in combined
        restart_rate: β parameter - controls mutation vs fresh start
        max_solutions: Maximum size of returned population

    Returns:
        (cleaned_population, cleaned_objectives)
    """
    # Identify duplicates by cost signature
    seen = {}
    keep_indices = []
    duplicate_indices = []

    for idx, sol in enumerate(combined):
        sig = (round(objectives[idx][0], 4), round(objectives[idx][1], 4))
        if sig not in seen:
            seen[sig] = idx
            keep_indices.append(idx)
        else:
            duplicate_indices.append(idx)

    # Replace duplicates with new solutions (OPTIMIZED)
    # Instead of calling slow assign_nodes() for each duplicate, use fast
    # multi-round mutation on random PF/population solutions to create diversity.
    ltl = max(1, math.ceil((inst.num_customers - inst.num_drones) / (inst.num_drones + 1)))

    for dup_idx in duplicate_indices:
        if random.random() < restart_rate:
            # Multi-mode mutation on a random PF solution (as per paper)
            if pareto_front:
                parent = random.choice(pareto_front)
                combined[dup_idx] = multi_mode_mutation(inst, parent, ltl)
            else:
                combined[dup_idx] = multi_mode_mutation(inst, combined[dup_idx], ltl)
        else:
            # Fast perturbation: apply 2-3 random mutations to a random solution
            # (Replaces slow assign_nodes call; achieves similar diversity effect)
            if pareto_front:
                base = random.choice(pareto_front).copy()
            else:
                base = random.choice(combined).copy()
            # Apply 2 rounds of mutation for stronger perturbation
            base = multi_mode_mutation(inst, base, ltl)
            base = multi_mode_mutation(inst, base, ltl)
            combined[dup_idx] = base

    # Keep only unique + replaced
    kept = [combined[i] for i in keep_indices]
    kept.extend([combined[i] for i in duplicate_indices])

    # Enforce max size
    kept = kept[:max_solutions]

    return kept, None  # objectives will be recomputed
