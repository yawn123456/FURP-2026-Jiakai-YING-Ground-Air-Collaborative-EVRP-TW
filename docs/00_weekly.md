# Weekly Progress Log

> Update this file **every week**. Add a new entry at the top for each week.
> This is the first thing we check during review. Keep it honest and specific — it also feeds your attendance record (Rule 1).

**How to use:** copy the *Week template* block below for each new week. Newest week goes at the top.

---

## Week template — copy me

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- _What did you actually do / finish?_

**Challenges & blockers**
- _What got in the way? What are you stuck on?_

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

### Week 1 — 2026-06-10

**Attended this week's meeting:** Yes

**Progress this week**
- Set up repository from the FURP template.
- Configured GitHub authentication (gh auth login) for push access.
- Initialized `/src` directory structure: `data/`, `experiments/`, `results/`.
- Added `.gitignore` fix to allow tracking `.gitkeep` in `data/` directories.
- Implemented and pushed OR-Tools VRP baseline solver (`src/experiments/vrp_ortools_simple.py`):
  - 4 vehicles, 20 customers, capacity 100
  - Total distance: 530 (feasible solution)
  - Runtime: ~5 seconds
  - Generated route visualization (`src/results/vrp_ortools_route.png`)
- Exported and committed environment config files:
  - `requirements.txt` (pip freeze)
  - `environment.yml` (conda env export)
- Verified the solver runs successfully in `vrp_env312` (Python 3.12.13, OR-Tools 9.15).

**Challenges & blockers**
- `.gitignore` initially blocked `data/` directories entirely — fixed with `data/**` + negation rule for `.gitkeep`.
- None blocking so far.

**Next steps**
- Research truck-drone collaborative delivery literature.
- Design EVRP-TW mathematical model formulation.
- Implement electric vehicle constraints (charging stations, battery capacity) on top of the baseline VRP solver.

**Hours spent (optional):** 4h

**Links (optional):**
- Repository: https://github.com/yawn123456/FURP-2026-Jiakai-YING-Ground-Air-Collaborative-EVRP-TW-Hybrid-Optimization-for-Truck-Drone-Delivery
- OR-Tools VRP solver: `src/experiments/vrp_ortools_simple.py`
- Route visualization: `src/results/vrp_ortools_route.png`

### Week 1 — 2026-06-23

**Attended this week's meeting:** No (email leave: No)

**Progress this week**
- Reproduced the HMOA (Hybrid Multi-Objective Optimization Approach) by Luo et al. (IEEE TITS, 2022) for collaborative truck-drone routing, integrating NSGA-II with Pareto Local Search (PLS).
- Validated the algorithm on 20-customer and 50-customer instances, achieving 30 Pareto-non-dominated solutions on the 20-customer case (best cost: 3,133.99).
- Fixed a bug in the initialization logic and optimized performance, yielding a **10.6× speedup** (13.89s for 20 customers, 37.68s for 50 customers).
- Completed a comparative analysis of POMO (learning-based), GA/EA (evolutionary), and OR (exact) methods across objective quality, efficiency, constraint flexibility, scalability, and implementation complexity.
- Analyzed main difficulties when adding energy and time window constraints: non-linear energy modeling, charging decision coupling, hard time window feasibility, and truck-drone temporal coordination.
- Drafted insights and framework adaptations for extending the approach to EVRP-TW, including a proposed 3-part chromosome (truck route | charging station indices | charging ratio), segment-by-segment energy evaluation, E-Insertion heuristic for initial solutions, and a constraint repair priority chain (Connectivity → Time windows → Energy → Capacity).

**Challenges & blockers**
- **Non-linear coupling between constraints** (energy × time windows × route topology) causes combinatorial explosion — enabling 3 drones simultaneously reduced feasible flight space by over 60% purely due to coordination constraints.
- **Hard time window constraints** yield very few feasible random initial solutions; the paper avoids this via flexible time windows with satisfaction decay, but extending to hard windows requires time-window-aware initialization.
- **State-dependent energy evaluation** — swapping two nodes requires re-evaluating the entire path's energy feasibility, making neighborhood operators expensive.
- **Charging decision coupling** — location, duration, and strategy are strongly coupled with routing, causing search space explosion.

**Next steps**
- Extend the HMOA framework to EVRP-TW: replace "drone flights" with "charging station visits" in the chromosome representation.
- Implement the proposed EVRP-TW chromosome (3-part encoding) and energy-aware Repair operator.
- Develop the E-Insertion heuristic for charging-station-aware initial solution construction.
- Benchmark against OR-Tools (CP-SAT) on small instances (≤20 customers) for optimality reference.
- Explore AM (Attention Model) for learning charging station selection within Repair for large-scale instances (≥200 customers).

**Hours spent (optional):** ---

**Links (optional):** _Luo et al., IEEE TITS 2022 — HMOA for collaborative truck-drone routing_

---

*Last updated: 2026-06-23*
