"""
Hybrid Multi-Objective Optimization Approach (HMOA)
Algorithm 1 — Main Framework

Paper: "Hybrid Multi-Objective Optimization Approach With Pareto Local Search
       for Collaborative Truck-Drone Routing Problems Considering Flexible Time Windows"
IEEE TITS, 2022
"""

import random
import math
from typing import List, Tuple, Optional
from model import Solution, ProblemInstance
from evaluate import evaluate
from initialization import assign_nodes
from genetic_ops import crossover, multi_mode_mutation
from duplication import remove_duplication
from nsga2_utils import non_dominated_sort, crowding_distance, select_next_population
from pls import pareto_local_search


def update_pf(external_pf: List[Solution], front_f1: List[Solution],
              inst: ProblemInstance) -> List[Solution]:
    """
    Update external Pareto front PF with new solutions from F1.

    Args:
        external_pf: Current external PF
        front_f1: Solutions in the first non-dominated front (F1)
        inst: Problem instance

    Returns:
        Updated PF
    """
    # Combine PF and F1
    all_solutions = list(external_pf) + list(front_f1)

    # Re-evaluate all and keep non-dominated
    objectives = []
    for sol in all_solutions:
        f1, f2 = evaluate(inst, sol)
        sol.cost = f1
        sol.satisfaction = f2
        objectives.append((f1, f2))

    # Find non-dominated set
    non_dominated = []
    n = len(all_solutions)

    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i
            f1_j, f2_j = objectives[j]
            f1_i, f2_i = objectives[i]
            if (f1_j <= f1_i and f2_j >= f2_i) and (f1_j < f1_i or f2_j > f2_i):
                dominated = True
                break
        if not dominated:
            non_dominated.append(all_solutions[i])

    return non_dominated


def hmoa(inst: ProblemInstance,
         pop_size: int = 200,
         max_iterations: int = 200,
         crossover_rate: float = 0.8,
         mutation_rate: float = 0.3,
         restart_rate: float = 0.3,
         pls_kmax: int = 5,
         verbose: bool = True) -> List[Solution]:
    """
    Algorithm 1: Framework of HMOA

    Args:
        inst: Problem instance
        pop_size: Population size n
        max_iterations: Maximum generations (iter)
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        restart_rate: β parameter for RemoveDuplication
        pls_kmax: Maximum PLS iterations
        verbose: Print progress

    Returns:
        Final Pareto front PF (list of non-dominated solutions)
    """
    # ---- Step 1: Initialization ----
    prob = 0.0  # Adaptive trigger probability for PLS
    PF = []     # External Pareto front
    LTL = max(1, math.ceil((inst.num_customers - inst.num_drones) / (inst.num_drones + 1)))

    if verbose:
        print(f"HMOA: {inst.num_customers} customers, {inst.num_drones} drones, "
              f"pop={pop_size}, iter={max_iterations}")
        print(f"  LTL = {LTL}")

    # Create initial population
    P = assign_nodes(inst, pop_size)

    # Evaluate initial population
    objectives = []
    for sol in P:
        f1, f2 = evaluate(inst, sol)
        sol.cost = f1
        sol.satisfaction = f2
        objectives.append((f1, f2))

    # ---- Main loop (lines 3-20) ----
    for t in range(1, max_iterations + 1):
        # Step 4: Generate offspring Qt via genetic operations
        Qt = []

        # Non-dominated sort and crowding distance (for tournament selection)
        fronts = non_dominated_sort(P, objectives)
        crowding_all = []
        for front in fronts:
            d = crowding_distance(P, objectives, front)
            crowding_all.extend(d)
        # Map solution index to rank
        rank_map = {}
        for rank_idx, front in enumerate(fronts):
            for sol_idx in front:
                rank_map[sol_idx] = rank_idx

        # Crossover
        for _ in range(pop_size // 2):
            # Binary tournament selection based on rank and crowding distance (per paper)
            pop_indices = list(range(len(P)))
            i = random.choice(pop_indices)
            j = random.choice(pop_indices)
            # Select better of the two
            if rank_map[i] < rank_map[j]:
                p1 = P[i]
            elif rank_map[j] < rank_map[i]:
                p1 = P[j]
            else:
                p1 = P[i] if crowding_all[i] >= crowding_all[j] else P[j]

            i = random.choice(pop_indices)
            j = random.choice(pop_indices)
            if rank_map[i] < rank_map[j]:
                p2 = P[i]
            elif rank_map[j] < rank_map[i]:
                p2 = P[j]
            else:
                p2 = P[i] if crowding_all[i] >= crowding_all[j] else P[j]

            if random.random() < crossover_rate:
                c1, c2 = crossover(inst, p1, p2)
                Qt.append(c1)
                Qt.append(c2)
            else:
                Qt.append(p1.copy())
                Qt.append(p2.copy())

        # Mutation (multi-mode)
        for i in range(len(Qt)):
            if random.random() < mutation_rate:
                Qt[i] = multi_mode_mutation(inst, Qt[i], LTL)

        # Step 5: Rt = Qt ∪ Pt
        Rt = Qt + P

        # Re-evaluate Rt
        rt_objectives = []
        for sol in Rt:
            f1, f2 = evaluate(inst, sol)
            sol.cost = f1
            sol.satisfaction = f2
            rt_objectives.append((f1, f2))

        # Step 6: RemoveDuplication
        # Paper: Replace duplicates with either multi-mode mutation on PF solution
        # (prob = restart_rate) or new AssignNodes solution (prob = 1-restart_rate)
        unique_Rt, _ = remove_duplication(
            inst, Rt, PF, rt_objectives,
            restart_rate=restart_rate,
            max_solutions=2 * pop_size
        )
        # Re-evaluate kept solutions after duplication removal
        unique_obj = []
        for sol in unique_Rt:
            f1, f2 = evaluate(inst, sol)
            sol.cost = f1
            sol.satisfaction = f2
            unique_obj.append((f1, f2))

        # Step 7: Non-dominated sort
        fronts = non_dominated_sort(unique_Rt, unique_obj)

        # Steps 8-19: Build next population Pt+1
        Pt1 = []
        Pt1_obj = []

        for rank_idx, front in enumerate(fronts):
            if len(Pt1) + len(front) <= pop_size:
                for fi in front:
                    Pt1.append(unique_Rt[fi])
                    Pt1_obj.append(unique_obj[fi])

                # Step 10-14: Update PF and run PLS on F1
                if rank_idx == 0 and front:
                    # Step 11: PF <- Update1(PF, F1)
                    f1_solutions = [unique_Rt[fi] for fi in front]
                    PF = update_pf(PF, f1_solutions, inst)

                    # Step 12-13: Trigger PLS with probability prob
                    if random.random() < prob:
                        if verbose and t % 20 == 0:
                            print(f"  Gen {t}: Running PLS on PF (|PF|={len(PF)}, prob={prob:.3f})")
                        PF = pareto_local_search(inst, PF, k_max=pls_kmax, verbose=False)

                        # Step 14: F1 <- PF (replace first front with PF)
                        # Select up to pop_size solutions from PF
                        Pt1 = list(PF[:pop_size])
                        Pt1_obj = [(s.cost, s.satisfaction) for s in Pt1]

            else:
                # Partial selection from this front based on crowding distance
                remaining = pop_size - len(Pt1)
                dists = crowding_distance(unique_Rt, unique_obj, front)

                # Sort by crowding distance descending
                sorted_front = sorted(
                    range(len(front)),
                    key=lambda i: dists[i],
                    reverse=True
                )
                for i in range(remaining):
                    fi = front[sorted_front[i]]
                    Pt1.append(unique_Rt[fi])
                    Pt1_obj.append(unique_obj[fi])
                break

        # Step 20: Update for next generation
        P = Pt1[:pop_size]
        objectives = Pt1_obj[:pop_size]
        prob = prob + 1.0 / max_iterations  # prob increases each generation

        if verbose and t % 20 == 0:
            best_f1 = min(o[0] for o in objectives)
            best_f2 = max(o[1] for o in objectives)
            pareto_count = len(fronts[0]) if fronts else 0
            print(f"  Gen {t:4d}: PF={len(PF):4d}, best_cost={best_f1:.2f}, "
                  f"best_sat={best_f2:.2f}, prob={prob:.3f}")

    # Final: update PF one more time
    all_solutions = list(P) + list(PF)
    all_obj = list(objectives) + [(s.cost, s.satisfaction) for s in PF]

    # Deduplicate by precise objective values
    seen = {}
    dedup_sol = []
    dedup_obj = []
    for idx, sol in enumerate(all_solutions):
        f1, f2 = all_obj[idx]
        sig = (round(f1, 6), round(f2, 6))
        if sig not in seen:
            seen[sig] = True
            dedup_sol.append(sol)
            dedup_obj.append((f1, f2))

    fronts = non_dominated_sort(dedup_sol, dedup_obj)
    final_PF = []
    for fi in fronts[0]:
        final_PF.append(dedup_sol[fi])
        dedup_sol[fi].cost = dedup_obj[fi][0]
        dedup_sol[fi].satisfaction = dedup_obj[fi][1]

    # Sort by cost
    final_PF.sort(key=lambda s: s.cost)

    if verbose:
        print(f"\nHMOA completed. Final PF size: {len(final_PF)} (unique, non-dominated)")

    return final_PF
