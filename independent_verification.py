"""
Independent Verification of CAD (Completion-Aware Decreasing) Heuristic
=======================================================================

This script independently verifies the CAD heuristic results using:
1. A completely separate implementation of all algorithms
2. Different random number generation
3. Different instance generation methodology
4. Cross-validation against known optimal solutions
"""
import numpy as np
import time
from scipy.stats import wilcoxon, mannwhitneyu
import json

###############################################################################
# INDEPENDENT IMPLEMENTATIONS (written from scratch)
###############################################################################

def independent_ffd(sizes, C):
    """Independent FFD implementation using different data structures."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
    bin_loads = []
    bin_contents = []
    for idx in order:
        s = sizes[idx]
        best = -1
        for b in range(len(bin_loads)):
            if bin_loads[b] + s <= C:
                best = b
                break
        if best >= 0:
            bin_loads[best] += s
            bin_contents[best].append(s)
        else:
            bin_loads.append(s)
            bin_contents.append([s])
    return len(bin_contents), bin_contents

def independent_cad(sizes, C):
    """Independent CAD implementation using different coding style."""
    n = len(sizes)
    order = sorted(range(n), key=lambda i: sizes[i], reverse=True)
    available = [True] * n
    bin_loads = []
    bin_contents = []
    
    for pos in range(n):
        item_idx = order[pos]
        if not available[item_idx]:
            continue
        
        item_size = sizes[item_idx]
        available[item_idx] = False
        
        # Best-fit: find bin with smallest remaining capacity that fits
        chosen_bin = -1
        min_gap = C + 1
        for b in range(len(bin_loads)):
            gap = C - bin_loads[b]
            if gap >= item_size and gap < min_gap:
                min_gap = gap
                chosen_bin = b
        
        if chosen_bin >= 0:
            bin_loads[chosen_bin] += item_size
            bin_contents[chosen_bin].append(item_size)
        else:
            bin_loads.append(item_size)
            bin_contents.append([item_size])
            chosen_bin = len(bin_loads) - 1
        
        space_left = C - bin_loads[chosen_bin]
        if space_left == 0:
            continue
        
        # COMPLETION: Find best single item to fill remaining
        best_fit_idx = -1
        best_fit_gap = space_left + 1
        for j in order:
            if not available[j]:
                continue
            if sizes[j] <= space_left:
                gap = space_left - sizes[j]
                if gap < best_fit_gap:
                    best_fit_idx = j
                    best_fit_gap = gap
                if gap == 0:
                    break
        
        # Accept if waste is ≤ 15% of remaining
        if best_fit_idx >= 0 and best_fit_gap <= space_left * 0.15:
            bin_loads[chosen_bin] += sizes[best_fit_idx]
            bin_contents[chosen_bin].append(sizes[best_fit_idx])
            available[best_fit_idx] = False
            space_left = C - bin_loads[chosen_bin]
            if space_left == 0:
                continue
        
        # COMPLETION: Find best pair of items
        if space_left > 0:
            best_pair = None
            best_pair_gap = space_left + 1
            for j1 in order:
                if not available[j1]:
                    continue
                s1 = sizes[j1]
                if s1 > space_left:
                    continue
                if s1 < space_left * 0.25:
                    break
                need = space_left - s1
                for j2 in order:
                    if not available[j2] or j2 == j1:
                        continue
                    s2 = sizes[j2]
                    if s2 <= need:
                        gap = need - s2
                        if gap < best_pair_gap:
                            best_pair = (j1, j2)
                            best_pair_gap = gap
                        break
                if best_pair and best_pair_gap == 0:
                    break
            
            if best_pair and best_pair_gap <= space_left * 0.1:
                j1, j2 = best_pair
                bin_loads[chosen_bin] += sizes[j1] + sizes[j2]
                bin_contents[chosen_bin].extend([sizes[j1], sizes[j2]])
                available[j1] = False
                available[j2] = False
    
    # Place any remaining items
    for i in range(n):
        if available[i]:
            s = sizes[i]
            placed = False
            for b in range(len(bin_loads)):
                if bin_loads[b] + s <= C:
                    bin_loads[b] += s
                    bin_contents[b].append(s)
                    placed = True
                    break
            if not placed:
                bin_loads.append(s)
                bin_contents.append([s])
    
    return len(bin_contents), bin_contents

###############################################################################
# VERIFICATION
###############################################################################

def verify_packing(sizes, bin_contents, C):
    """Verify a packing is valid."""
    all_items = sorted([x for b in bin_contents for x in b])
    original = sorted(sizes)
    if all_items != original:
        return False, "Items mismatch"
    for b in bin_contents:
        if sum(b) > C:
            return False, f"Overflow: {sum(b)} > {C}"
    return True, "OK"

###############################################################################
# TEST 1: Verify correctness on known-optimal instances
###############################################################################
print("=" * 70)
print("TEST 1: Correctness on instances with known optima")
print("=" * 70)

# Triplet instances: optimal is n/3
for n in [30, 60, 90, 120]:
    C = 1000
    perfect = 0
    total = 20
    for seed in range(total):
        rng = np.random.RandomState(seed + 7777)
        items = []
        for _ in range(n // 3):
            a = rng.randint(C//4+1, C//2)
            b = rng.randint(1, min(C-a, C//2))
            c = C - a - b
            items.extend([a, b, c])
        rng.shuffle(items)
        
        nb, bc = independent_cad(list(items), C)
        ok, msg = verify_packing(list(items), bc, C)
        assert ok, f"Invalid packing: {msg}"
        if nb == n // 3:
            perfect += 1
    
    nb_ffd, _ = independent_ffd(list(items), C)
    print(f"  Triplet n={n:>3}: CAD finds optimal {perfect}/{total} times (FFD on last: {nb_ffd} vs opt {n//3})")

###############################################################################
# TEST 2: Reproduce main result on hard_150
###############################################################################
print(f"\n{'=' * 70}")
print("TEST 2: Reproduce main result - hard_150 instances")
print("Using DIFFERENT seeds than original experiment")
print("=" * 70)

C = 150
ffd_bins_list = []
cad_bins_list = []

for n in [100, 200, 300, 500]:
    ffd_b = []
    cad_b = []
    for seed in range(50):
        # Use completely different seed strategy
        rng = np.random.RandomState(seed * 13 + n * 7 + 9999)
        items = list(rng.randint(20, 101, size=n))
        
        nb_ffd, bc_ffd = independent_ffd(items[:], C)
        nb_cad, bc_cad = independent_cad(items[:], C)
        
        ok1, _ = verify_packing(items, bc_ffd, C)
        ok2, _ = verify_packing(items, bc_cad, C)
        assert ok1 and ok2
        
        ffd_b.append(nb_ffd)
        cad_b.append(nb_cad)
        ffd_bins_list.append(nb_ffd)
        cad_bins_list.append(nb_cad)
    
    diffs = [f - c for f, c in zip(ffd_b, cad_b)]
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    losses = sum(1 for d in diffs if d < 0)
    
    non_zero = [d for d in diffs if d != 0]
    if len(non_zero) >= 5:
        stat, p = wilcoxon(non_zero)
    else:
        p = 1.0
    
    lb_list = [int(np.ceil(sum(rng.randint(20,101,size=n))/C)) for _ in range(1)]  # approx
    
    print(f"  n={n:>3}: FFD_avg={np.mean(ffd_b):.1f} CAD_avg={np.mean(cad_b):.1f} "
          f"wins={wins} ties={ties} losses={losses} p={p:.2e}")

# Overall
all_diffs = [f - c for f, c in zip(ffd_bins_list, cad_bins_list)]
non_zero = [d for d in all_diffs if d != 0]
stat, p = wilcoxon(non_zero)
total_wins = sum(1 for d in all_diffs if d > 0)
total_ties = sum(1 for d in all_diffs if d == 0)
total_losses = sum(1 for d in all_diffs if d < 0)
print(f"\n  OVERALL (200 instances with different seeds):")
print(f"  CAD wins={total_wins}, ties={total_ties}, losses={total_losses}")
print(f"  Total bins saved: {sum(all_diffs)}")
print(f"  Wilcoxon p-value: {p:.2e}")
print(f"  VERIFIED: {'YES' if p < 0.001 and total_losses == 0 else 'NEEDS REVIEW'}")

###############################################################################
# TEST 3: Verify CAD never produces MORE bins than FFD on hard instances
###############################################################################
print(f"\n{'=' * 70}")
print("TEST 3: Monotonicity check - Does CAD ever use MORE bins than FFD?")
print("Testing 1000 random hard_150 instances")
print("=" * 70)

C = 150
violations = 0
total_tests = 1000
for seed in range(total_tests):
    rng = np.random.RandomState(seed + 50000)
    n = rng.randint(50, 501)
    items = list(rng.randint(20, 101, size=n))
    
    nb_ffd, _ = independent_ffd(items[:], C)
    nb_cad, bc = independent_cad(items[:], C)
    ok, _ = verify_packing(items, bc, C)
    assert ok
    
    if nb_cad > nb_ffd:
        violations += 1
        if violations <= 5:
            print(f"  VIOLATION at seed={seed}, n={n}: FFD={nb_ffd}, CAD={nb_cad}")

print(f"\n  Violations: {violations}/{total_tests}")
print(f"  VERIFIED: CAD {'NEVER' if violations == 0 else 'SOMETIMES'} worse than FFD on hard_150")

###############################################################################
# TEST 4: Timing benchmark
###############################################################################
print(f"\n{'=' * 70}")
print("TEST 4: Runtime comparison")
print("=" * 70)

for n in [100, 500, 1000, 2000]:
    rng = np.random.RandomState(42)
    items = list(rng.randint(20, 101, size=n))
    C = 150
    
    # Time FFD
    t0 = time.perf_counter()
    for _ in range(10):
        independent_ffd(items[:], C)
    ffd_time = (time.perf_counter() - t0) / 10
    
    # Time CAD
    t0 = time.perf_counter()
    for _ in range(10):
        independent_cad(items[:], C)
    cad_time = (time.perf_counter() - t0) / 10
    
    print(f"  n={n:>5}: FFD={ffd_time*1000:>8.2f}ms  CAD={cad_time*1000:>8.2f}ms  "
          f"ratio={cad_time/ffd_time:.1f}x")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
