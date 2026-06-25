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

    Paper description: kmax determines the maximum iterations of PLS.
    H = {N4, N5, N6}. New solutions to the current Pareto front are explored by
    applying neighborhood operators. Only solutions that are new to PF (i.e., were
    not in PF before the current generation) are explored, as they have the highest
    chance to find new non-dominated solutions.

    Args:
        inst: Problem instance
        pareto_front: Current Pareto front
        k_max: Maximum iterations (paper: 5)
        verbose: Print progress

    Returns:
        Improved Pareto front
    """
    H = [n4_2opt, n5_greedy_deletion_reinsertion, n6_random_deletion_reinsertion]

    # Initialize PF and track objective values
    PF = list(pareto_front)
    PF_objectives = []
    for sol in PF:
        f1, f2 = evaluate(inst, sol)
        sol.cost = f1
        sol.satisfaction = f2
        PF_objectives.append((f1, f2))

    # Track visited signatures to avoid re-exploring
    visited_signatures = set()
    for f1, f2 in PF_objectives:
        visited_signatures.add((round(f1, 4), round(f2, 4)))

    # PF' ← Find solutions new to PF (initially all solutions are "new")
    # Per paper: "New solutions may be further improved, thereby only new
    # solutions to PF would be explored in the local search."
    PF_prime = list(PF)
    PF_prime_objectives = list(PF_objectives)

    k = 1

    # Paper: iteration repeats until PF' is empty or max iterations reached
    while k <= k_max and PF_prime:
        if verbose:
            print(f"  PLS iteration {k}, exploring {len(PF_prime)} new solutions, PF size={len(PF)}")

        # PL ← ∅ (candidates for next iteration)
        PL_next = []
        PL_next_objectives = []

        for p in PF_prime:
            for Ni in H:
                # p' ← Ni(p)
                p_prime = Ni(inst, p)

                # Evaluate p'
                f1_p, f2_p = evaluate(inst, p_prime)
                sig = (round(f1_p, 4), round(f2_p, 4))

                # Skip if already visited
                if sig in visited_signatures:
                    continue

                # Check if p' dominates p (p' is better than original solution p)
                # Per paper text: "if p' dominates p" — Update2 is triggered
                p1_obj = None
                for idx in range(len(PF_prime)):
                    if PF_prime[idx] is p:
                        p1_obj = PF_prime_objectives[idx]
                        break
                if p1_obj is None:
                    p1_obj = evaluate(inst, p)

                # Check if p' dominates the original solution p
                if not dominates((f1_p, f2_p), p1_obj):
                    # p' does not dominate p → skip per paper's condition
                    # (Paper line 8: only proceed if p' dominates p)
                    continue

                # Update2(p', PF): add p' to PF if not dominated, remove dominated
                added_to_PF = _update2(inst, p_prime, (f1_p, f2_p), PF, PF_objectives, visited_signatures)

                if added_to_PF:
                    visited_signatures.add(sig)
                    # Update2(p', PL): add p' to PL for next iteration
                    _update2(inst, p_prime, (f1_p, f2_p), PL_next, PL_next_objectives, visited_signatures)

        # PF' ← PL (prepare for next iteration)
        PF_prime = PL_next
        PF_prime_objectives = PL_next_objectives
        k += 1

    # Return final Pareto front
    non_dominated = []
    for idx, sol in enumerate(PF):
        sol.cost = PF_objectives[idx][0]
        sol.satisfaction = PF_objectives[idx][1]
        non_dominated.append(sol)

    return non_dominated


def _update2(inst: ProblemInstance, p_prime: Solution, p_obj: tuple,
             target_list: list, target_obj: list, visited: set) -> bool:
    """
    Update2 operator from paper Algorithm 3:
    - p' is added to target if no solution in target can dominate it
    - All solutions in target dominated by p' are removed
    - Returns True if p' was added to target
    """
    f1_p, f2_p = p_obj

    # Check if p' is dominated by any solution in target
    dominated_by_target = False
    new_target = []
    new_target_obj = []

    for idx, (f1_t, f2_t) in enumerate(target_obj):
        if dominates((f1_t, f2_t), (f1_p, f2_p)):
            # Existing target solution dominates p' → don't add p'
            dominated_by_target = True
            new_target.append(target_list[idx])
            new_target_obj.append((f1_t, f2_t))
        elif dominates((f1_p, f2_p), (f1_t, f2_t)):
            # p' dominates existing target solution → remove it
            continue
        else:
            # Non-dominated with each other → keep existing
            new_target.append(target_list[idx])
            new_target_obj.append((f1_t, f2_t))

    if not dominated_by_target:
        # p' is not dominated → add to target
        new_target.append(p_prime)
        new_target_obj.append((f1_p, f2_p))

        # Update in-place
        target_list.clear()
        target_list.extend(new_target)
        target_obj.clear()
        target_obj.extend(new_target_obj)
        return True

    # Update in-place (even if p' not added, dominated solutions were removed)
    target_list.clear()
    target_list.extend(new_target)
    target_obj.clear()
    target_obj.extend(new_target_obj)
    return False
