# Compact Finite Difference Solver for Weakly Singular Volterra Integro-Differential Equations

Reproducible code for the paper:

**"Convergence Degradation of Compact Finite Difference Schemes for Weakly Singular Volterra Integro-Differential Equations: From O(h^(4-α)) to O(h^(1-α)) and Recovery"**

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib

## Quick Start

```bash
python run_full_convergence.py
```

## File Structure

| File | Purpose |
|------|---------|
| `numerical_solver.py` | All solvers, test problems, figure generation |
| `run_full_convergence.py` | Full convergence study (Section 5-6) |

## Solvers

- `solve_vide_compact()` — Compact Padé + quadratic product integration (uniform mesh)
- `solve_vide_compact_graded()` — Compact on graded mesh via coordinate transformation
- `solve_vide_graded_mesh()` — Backward Euler on Brunner-type graded mesh
- `solve_vide_trapezoidal()` — Product trapezoidal baseline

## Test Problems

| Test | `u(x)` | Type |
|------|--------|------|
| 1 | `x²` | Smooth |
| 2 | `x^(1-α) + x²` | Single singular term |
| 3 | `x^(1-α) + x^(2-α) + x³` | Two singular terms |
| 4 | `sin(πx)` | Oscillatory |
| 5 | `x^(1-α) e^(-x) + x²/2` | Viscoelasticity-motivated |

## License

MIT
