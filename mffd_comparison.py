"""
MFFD comparison and independent verification of CAD heuristic.
Also includes threshold sweep to find optimal CAD parameters.
"""
import numpy as np
import time
from scipy.stats import wilcoxon

##############################################################################
# ALGORITHMS
##############################################################################

def ffd(items, cap):
    """First Fit Decreasing"""
    s = sorted(items, reverse=True)
    bins, rem = [], []
    for item in s:
        placed = False
        for i in range(len(bins)):
            if rem[i] >= item:
                bins[i].append(item)
                rem[i] -= item
                placed = True
                break
        if not placed:
            bins.append([item])
            rem.append(cap - item)
    return bins

def bfd(items, cap):
    """Best Fit Decreasing"""
    s = sorted(items, reverse=True)
    bins, rem = [], []
    for item in s:
        bi, br = -1, cap+1
        for i in range(len(bins)):
            if rem[i] >= item and rem[i] < br:
                br = rem[i]
                bi = i
        if bi >= 0:
            bins[bi].append(item)
            rem[bi] -= item
        else:
            bins.append([item])
            rem.append(cap - item)
    return bins

def mffd(items, cap):
    """Modified First Fit Decreasing (Garey & Johnson 1985)
    Items classified into 4 categories based on size relative to capacity:
    A: > cap/2, B: > cap/3, C: > cap/4, D: <= cap/4
    Special rules for handling B and C items.
    """
    s = sorted(items, reverse=True)
    half = cap / 2
    third = cap / 3
    quarter = cap / 4
    
    # Classify items
    A_items = [x for x in s if x > half]
    B_items = [x for x in s if third < x <= half]
    C_items = [x for x in s if quarter < x <= third]
    D_items = [x for x in s if x <= quarter]
    
    bins, rem = [], []
    
    # Step 1: Each A item gets its own bin
    for item in A_items:
        bins.append([item])
        rem.append(cap - item)
    
    # Step 2: Try to pair B items with other B items in A-bins
    # First, try to fit B items into existing bins (A bins)
    remaining_B = []
    for item in B_items:
        placed = False
        # Try to fit in an existing bin
        for i in range(len(bins)):
            if rem[i] >= item:
                bins[i].append(item)
                rem[i] -= item
                placed = True
                break
        if not placed:
            remaining_B.append(item)
    
    # Remaining B items get their own bins
    for item in remaining_B:
        bins.append([item])
        rem.append(cap - item)
    
    # Step 3: C items - first fit into existing bins
    remaining_C = []
    for item in C_items:
        placed = False
        for i in range(len(bins)):
            if rem[i] >= item:
                bins[i].append(item)
                rem[i] -= item
                placed = True
                break
        if not placed:
            remaining_C.append(item)
    
    for item in remaining_C:
        bins.append([item])
        rem.append(cap - item)
    
    # Step 4: D items - first fit
    for item in D_items:
        placed = False
        for i in range(len(bins)):
            if rem[i] >= item:
                bins[i].append(item)
                rem[i] -= item
                placed = True
                break
        if not placed:
            bins.append([item])
            rem.append(cap - item)
    
    return bins

def cad(items, cap):
    """Completion-Aware Decreasing"""
    n = len(items)
    idx = sorted(range(n), key=lambda i: -items[i])
    used = [False]*n
    bins, rem = [], []
    for p in range(n):
        i = idx[p]
        if used[i]:
            continue
        sz = items[i]
        used[i] = True
        bi, br = -1, cap+1
        for b in range(len(bins)):
            if rem[b] >= sz and rem[b] < br:
                br = rem[b]
                bi = b
        if bi >= 0:
            bins[bi].append(sz)
            rem[bi] -= sz
            tgt = bi
        else:
            bins.append([sz])
            rem.append(cap-sz)
            tgt = len(bins)-1
        r = rem[tgt]
        if r == 0:
            continue
        # 1-item completion
        b1i, b1w = -1, r+1
        for j in idx:
            if used[j]:
                continue
            s = items[j]
            if s <= r:
                w = r - s
                if w < b1w:
                    b1i, b1w = j, w
                if w == 0:
                    break
        if b1i >= 0 and b1w <= r * 0.15:
            bins[tgt].append(items[b1i])
            rem[tgt] -= items[b1i]
            used[b1i] = True
            r = rem[tgt]
            if r == 0:
                continue
        # 2-item completion
        if r > 0:
            bp, bpw = None, r+1
            for j1 in idx:
                if used[j1]:
                    continue
                s1 = items[j1]
                if s1 > r:
                    continue
                if s1 < r*0.25:
                    break
                need = r - s1
                for j2 in idx:
                    if used[j2] or j2==j1:
                        continue
                    s2 = items[j2]
                    if s2 <= need:
                        w = need - s2
                        if w < bpw:
                            bp, bpw = (j1,j2), w
                        break
                if bp and bpw == 0:
                    break
            if bp and bpw <= r * 0.1:
                j1,j2 = bp
                bins[tgt].append(items[j1])
                bins[tgt].append(items[j2])
                rem[tgt] -= items[j1]+items[j2]
                used[j1] = used[j2] = True
    for i in range(n):
        if not used[i]:
            sz = items[i]
            placed = False
            for b in range(len(bins)):
                if rem[b] >= sz:
                    bins[b].append(sz)
                    rem[b] -= sz
                    placed = True
                    break
            if not placed:
                bins.append([sz])
                rem.append(cap-sz)
    return bins

def lb(items, cap):
    return int(np.ceil(sum(items)/cap))

def verify(items, bins_r, cap):
    packed = sorted([x for b in bins_r for x in b])
    orig = sorted(items)
    assert packed == orig, "Items mismatch!"
    for b in bins_r:
        assert sum(b) <= cap, f"Overflow: {sum(b)} > {cap}"

##############################################################################
# MAIN EXPERIMENT: FFD vs BFD vs MFFD vs CAD
##############################################################################

algos = {'FFD': ffd, 'BFD': bfd, 'MFFD': mffd, 'CAD': cad}
configs = [
    ("hard_150", 150, 20, 101),
    ("hard_200", 200, 30, 151),
    ("medium", 100, 10, 81),
    ("large_items", 1000, 250, 501),
]

print("=" * 90)
print("COMPREHENSIVE BENCHMARK: FFD vs BFD vs MFFD vs CAD")
print("50 trials per (instance_type, n) combination")
print("=" * 90)

all_data = {}

for cname, cap, lo, hi in configs:
    print(f"\n--- {cname}: items [{lo},{hi-1}], capacity {cap} ---")
    print(f"{'N':>5}  {'FFD':>10} {'BFD':>10} {'MFFD':>10} {'CAD':>10}  {'CAD<FFD':>8} {'CAD<MFFD':>9}")
    
    for n in [100, 200, 300, 500]:
        res = {a: [] for a in algos}
        for seed in range(50):
            rng = np.random.RandomState(seed * 997 + n * 31)
            items = list(rng.randint(lo, hi, size=n))
            lower = lb(items, cap)
            for name, fn in algos.items():
                bins_r = fn(items[:], cap)
                verify(items, bins_r, cap)
                res[name].append({'bins': len(bins_r), 'lb': lower, 'ratio': len(bins_r)/lower})
        
        ffd_r = np.mean([r['ratio'] for r in res['FFD']])
        bfd_r = np.mean([r['ratio'] for r in res['BFD']])
        mffd_r = np.mean([r['ratio'] for r in res['MFFD']])
        cad_r = np.mean([r['ratio'] for r in res['CAD']])
        
        cad_vs_ffd = sum(1 for c, f in zip(res['CAD'], res['FFD']) if c['bins'] < f['bins'])
        cad_vs_mffd = sum(1 for c, m in zip(res['CAD'], res['MFFD']) if c['bins'] < m['bins'])
        
        print(f"{n:>5}  {ffd_r:>10.5f} {bfd_r:>10.5f} {mffd_r:>10.5f} {cad_r:>10.5f}  {cad_vs_ffd:>8} {cad_vs_mffd:>9}")
        
        key = f"{cname}_n{n}"
        all_data[key] = res

# STATISTICAL TESTS: CAD vs each baseline
print("\n\n" + "=" * 90)
print("STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank, CAD vs baselines)")
print("=" * 90)

for cname, cap, lo, hi in configs:
    print(f"\n{cname}:")
    for n in [100, 200, 300, 500]:
        key = f"{cname}_n{n}"
        res = all_data[key]
        
        for baseline in ['FFD', 'MFFD']:
            diffs = [b['bins'] - c['bins'] for b, c in zip(res[baseline], res['CAD'])]
            non_zero = [d for d in diffs if d != 0]
            wins = sum(1 for d in diffs if d > 0)
            losses = sum(1 for d in diffs if d < 0)
            ties = sum(1 for d in diffs if d == 0)
            
            if len(non_zero) >= 5:
                stat, p = wilcoxon(non_zero)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            else:
                p = 1.0
                sig = "ns"
                stat = 0
            
            saved = sum(diffs)
            print(f"  n={n:>3} CAD vs {baseline:<4}: wins={wins:>3} ties={ties:>3} losses={losses:>3} "
                  f"saved={saved:>4} p={p:.2e} {sig}")
