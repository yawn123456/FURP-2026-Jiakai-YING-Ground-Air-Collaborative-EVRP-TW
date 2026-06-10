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
