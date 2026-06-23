"""
Pareto Local Search (Algorithm 3)

Uses N4 (2-Opt), N5 (Greedy-Deletion-Reinsertion), N6 (Random-Deletion-Reinsertion)
to explore the neighborhood of non-dominated solutions.
"""

from typing import List, Set, Tuple
from model import Solution, ProblemInstance
from neighborhood import n4_2opt, n5_greedy_deletion_reinsertion, n6_random_deletion_reinsertion
from evaluate import evaluate
from nsga2_utils import dominates


def pareto_local_search(inst: ProblemInstance,
                        pareto_front: List[Solution],
                        k_max: int = 5,
                        verbose: bool = False) -> List[Solution]:
    """
    Algorithm 3: Pareto Local Search (PLS)

    Args:
        inst: Problem instance
        pareto_front: Current Pareto front (list of non-dominated solutions)
        k_max: Maximum iterations (default 5)
        verbose: Print progress

    Returns:
        Improved Pareto front
    """
    H = [n4_2opt, n5_greedy_deletion_reinsertion, n6_random_deletion_reinsertion]

    # Initialize
    PF = list(pareto_front)  # current Pareto front
    # We track solutions by their unique representation (tuple of costs)
    PF_set = set()
    PF_objectives = []  # (f1, f2) for each solution in PF

    for sol in PF:
        f1, f2 = evaluate(inst, sol)
        sol.cost = f1
        sol.satisfaction = f2
        PF_objectives.append((f1, f2))

    k = 1
    PL = list(PF)  # solutions to explore

    visited_signatures = set()
    for f1, f2 in PF_objectives:
        visited_signatures.add((round(f1, 4), round(f2, 4)))

    while k < k_max and PL:
        if verbose:
            print(f"  PLS iteration {k}, exploring {len(PL)} solutions, PF size={len(PF)}")

        # PL' <- empty (new candidates for next iteration)
        PL_next = []
        PL_objectives = []
        for sol in PL:
            f1, f2 = evaluate(inst, sol)
            PL_objectives.append((f1, f2))

        for sol_idx, p in enumerate(PL):
            for Ni in H:
                # p' <- Ni(p)
                p_prime = Ni(inst, p)

                # Evaluate p_prime
                f1_p, f2_p = evaluate(inst, p_prime)
                sig = (round(f1_p, 4), round(f2_p, 4))

                # Check against all solutions in PF
                is_new_nondominated = True
                PF_updated = False

                # Compare with current PF
                new_PF = []
                new_PF_obj = []
                for pf_idx, pf_sol in enumerate(PF):
                    curr_f1, curr_f2 = PF_objectives[pf_idx]

                    if dominates((f1_p, f2_p), (curr_f1, curr_f2)):
                        # p_prime dominates existing PF solution => remove it
                        PF_updated = True
                        continue
                    elif dominates((curr_f1, curr_f2), (f1_p, f2_p)):
                        # Existing PF solution dominates p_prime
                        is_new_nondominated = False
                        new_PF.append(pf_sol)
                        new_PF_obj.append((curr_f1, curr_f2))
                    else:
                        # Non-dominated with each other
                        new_PF.append(pf_sol)
                        new_PF_obj.append((curr_f1, curr_f2))

                if is_new_nondominated and sig not in visited_signatures:
                    # p_prime is added to PF
                    visited_signatures.add(sig)
                    new_PF.append(p_prime)
                    new_PF_obj.append((f1_p, f2_p))
                    PF_updated = True

                    # Add to PL_next (new solutions to explore)
                    PL_next.append(p_prime)

                PF = new_PF
                PF_objectives = new_PF_obj

        # Update PL = PL_next for next iteration
        PL = PL_next
        k += 1

    # Return the final Pareto front as solution objects
    non_dominated = []
    for idx, sol in enumerate(PF):
        sol.cost = PF_objectives[idx][0]
        sol.satisfaction = PF_objectives[idx][1]
        non_dominated.append(sol)

    return non_dominated
