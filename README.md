# CAD: Completion-Aware Decreasing for Bin Packing

A novel O(n²) heuristic for the one-dimensional bin packing problem that outperforms
First-Fit Decreasing (FFD), Best-Fit Decreasing (BFD), and Modified FFD (MFFD) on
hard instances.

## Key Results

- **160 wins, 40 ties, 0 losses** vs FFD across 200 hard instances (p < 10⁻³¹)
- **55% waste reduction** relative to the L₂ lower bound on Scholl-type hard instances
- **Beats MFFD** (Garey & Johnson 1985) on all hard and medium instance classes
- Runtime: O(n²), under 400ms for n = 1,000 in Python

## How It Works

After placing each item using Best-Fit Decreasing, CAD immediately searches the
remaining unplaced items for 1-2 items that tightly fill the bin's residual capacity.
This breaks the strict size-ordered processing of classical heuristics, enabling better
item pairings that reduce total waste.

```
COMPLETION-AWARE DECREASING (CAD):
1. Sort items by decreasing size
2. For each unplaced item:
   a. Place using Best-Fit (tightest bin)
   b. SINGLE-ITEM COMPLETION: Find best item to fill residual (waste ≤ 15%)
   c. TWO-ITEM COMPLETION: Find best pair to fill residual (waste ≤ 10%)
3. Place remaining items via First-Fit
```

## Quick Start

```bash
pip install numpy scipy
python cad_heuristic.py
```

This runs the full benchmark suite (~2 minutes) and prints results.

## Usage

```python
from cad_heuristic import completion_aware_decreasing, first_fit_decreasing

items = [42, 79, 65, 38, 91, 27, 53, 84, 36, 72, 48, 61, 33, 88, 55]
capacity = 150

ffd_bins = first_fit_decreasing(items, capacity)
cad_bins = completion_aware_decreasing(items, capacity)

print(f"FFD: {len(ffd_bins)} bins")
print(f"CAD: {len(cad_bins)} bins")
```

## Files

| File | Description |
|------|-------------|
| `cad_heuristic.py` | All algorithms (FFD, BFD, MFFD, CAD) + benchmark harness |
| `independent_verification.py` | Independent re-implementation for verification |
| `mffd_comparison.py` | Detailed MFFD comparison with threshold sweep |
| `results_summary.json` | Pre-computed results for all configurations |
| `results_data.csv` | Raw per-instance data (1,400+ instances) |
| `paper/` | LaTeX manuscript for arXiv submission |

## Results Summary

### Hard-150 Instances (items [20,100], capacity 150)

| n | FFD Ratio | CAD Ratio | CAD Wins | Ties | Losses | p-value |
|---|-----------|-----------|----------|------|--------|---------|
| 60  | 1.0208 | 1.0085 | 15 | 35 | 0 | 1.08×10⁻⁴ |
| 100 | 1.0198 | 1.0084 | 23 | 27 | 0 | 1.62×10⁻⁶ |
| 200 | 1.0156 | 1.0075 | 33 | 17 | 0 | 9.22×10⁻⁹ |
| 300 | 1.0138 | 1.0065 | 41 |  9 | 0 | 5.44×10⁻¹⁰ |
| 500 | 1.0128 | 1.0057 | 50 |  0 | 0 | 1.86×10⁻¹⁰ |

## When CAD Helps (and When It Doesn't)

**Best for:** Items filling 13-67% of bin capacity (2-4 items per bin). Shipping containers,
cloud VM packing, CNC cutting, warehouse storage.

**Neutral on:** Very small items (FFD already optimal), very large items (limited completion
opportunities), structured instances (both near-optimal).

## Citation

```bibtex
@article{sanghavi2026cad,
  title={Completion-Aware Decreasing: A Simple Polynomial-Time Heuristic for Bin Packing with Reduced Waste},
  author={Sanghavi, Kathan},
  journal={arXiv preprint},
  year={2026}
}
```

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
