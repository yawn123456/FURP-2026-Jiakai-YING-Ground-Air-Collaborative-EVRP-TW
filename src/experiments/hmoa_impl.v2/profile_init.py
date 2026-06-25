"""Profile initialization for 50 customers with debug output"""
import sys, os, time, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import generate_random_instance
from model import Solution
from initialization import nearest_neighbor_tw

random.seed(42)
inst = generate_random_instance(
    num_customers=50, num_drones=3, area_size=100.0,
    tw_width=80.0, tw_horizon=480.0, wbli=0.2, wbui=0.2, seed=12345
)

all_cust_ids = [c.id for c in inst.customers]
n_cust = len(all_cust_ids)
m = inst.num_drones
ltl = max(1, math.ceil((n_cust - m) / (m + 1)))

print(f"n_cust={n_cust}, m={m}, LTL={ltl}")
sys.stdout.flush()

# Test ONE iteration
t0 = time.time()

shuffled = all_cust_ids.copy()
random.shuffle(shuffled)
ct_size = max(ltl, random.randint(ltl, min(ltl + 3, n_cust)))
ct_size = min(ct_size, n_cust - 1)
ct = shuffled[:ct_size]
cd = shuffled[ct_size:]
cd = [n for n in cd if inst.node_map[n].drone_eligible]

print(f"ct_size={ct_size}, cd={len(cd)}")
sys.stdout.flush()

t1 = time.time()
ct_route = nearest_neighbor_tw(inst, ct)
t2 = time.time()
print(f"nearest_neighbor_tw: {t2-t1:.3f}s, route len={len(ct_route)}")

# Simulate assignment loop
used_positions = {d: set() for d in range(m)}
unassigned = set(cd)
assignments = []
route_len = len(ct_route)

iteration = 0
t_assign_start = time.time()
while unassigned and iteration < 50:
    candidates = []
    for cust_j in unassigned:
        best = None
        best_cost = float('inf')
        for d in range(m):
            used = used_positions[d]
            for launch_i in range(1, route_len - 2):
                if launch_i in used:
                    continue
                for land_k in range(launch_i + 1, route_len - 1):
                    if land_k in used:
                        continue
                    ln = ct_route[launch_i]
                    lk = ct_route[land_k]
                    dl = inst.drone_distance(ln, cust_j)
                    dr = inst.drone_distance(cust_j, lk)
                    if dl + dr <= inst.drone_endurance:
                        cost = (dl + dr) * inst.drone_cost_per_km
                        if cost < best_cost:
                            best_cost = cost
                            best = (launch_i, cust_j, land_k, d)
        if best:
            candidates.append(best)

    if not candidates:
        print(f"  Iter {iteration}: No feasible candidates, {len(unassigned)} remaining")
        break

    candidates.sort(key=lambda x: inst.drone_distance(ct_route[x[0]], x[1]) +
                   inst.drone_distance(x[1], ct_route[x[2]]))
    best = candidates[0]
    assignments.append(best)
    unassigned.remove(best[1])
    used_positions[best[3]].add(best[0])
    used_positions[best[3]].add(best[2])
    iteration += 1

t_assign = time.time() - t_assign_start
print(f"Assignment loop: {iteration} iterations, {t_assign:.3f}s, {len(unassigned)} remaining")
print(f"Total: {time.time()-t0:.3f}s")
sys.stdout.flush()
