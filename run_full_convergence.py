"""
Full convergence study for the paper.
Compares: standard compact, compact-graded, backward-Euler-graded.
"""
import numpy as np
from numerical_solver import (
    solve_vide_compact, solve_vide_compact_graded, solve_vide_graded_mesh,
    make_test_u_singular1
)

X = 1.0
N_values = [16, 32, 64, 128, 256]
alpha_values = [0.2, 0.3, 0.5, 0.8]

f_func_factory, _ = make_test_u_singular1()

print("=" * 80)
print("FULL CONVERGENCE STUDY")
print("Test problem: u(x) = x^{1-alpha} + x^2  (genuine weak singularity)")
print("=" * 80)

for alpha in alpha_values:
    gamma_opt = (4.0 - alpha) / (1.0 - alpha)
    gamma_use = min(gamma_opt, 8.0)
    exact = lambda x: x**(1.0 - alpha) + x**2.0
    f_func = lambda x: f_func_factory(x, alpha)

    print(f"\n{'='*60}")
    print(f"alpha = {alpha}, gamma_opt = {gamma_opt:.1f}, gamma_use = {gamma_use:.1f}")
    print(f"Theoretical rates: uniform={1.0-alpha:.1f}, BE-graded={min(4-alpha, gamma_use*(1-alpha)):.1f}, compact-graded={4-alpha:.1f}")
    print(f"{'='*60}")

    # Method 1: Standard compact (uniform)
    print("\n  [1] Standard compact (uniform mesh):")
    errs1 = []
    for N in N_values:
        x, u = solve_vide_compact(N, X, alpha, f_func, smooth_start=False)
        err = np.max(np.abs(u - exact(x)))
        errs1.append(err)
        print(f"      N={N:>3d}: err={err:.4e}")
    hs = [X/n for n in N_values]
    rate1 = np.abs(np.polyfit(np.log(hs), np.log(errs1), 1)[0])
    print(f"      => rate = {rate1:.3f}")

    # Method 2: Compact-graded (coordinate transform)
    print(f"\n  [2] Compact-graded (gamma={gamma_use:.1f}):")
    errs2 = []
    for N in N_values:
        x, u = solve_vide_compact_graded(N, X, alpha, f_func, gamma=gamma_use, smooth_start=True)
        err = np.max(np.abs(u - exact(x)))
        errs2.append(err)
        print(f"      N={N:>3d}: err={err:.4e}")
    rate2 = np.abs(np.polyfit(np.log(hs), np.log(errs2), 1)[0])
    print(f"      => rate = {rate2:.3f}")

    # Method 3: Backward Euler graded
    print(f"\n  [3] Backward Euler graded (gamma={gamma_use:.1f}):")
    errs3 = []
    for N in N_values:
        x, u = solve_vide_graded_mesh(N, X, alpha, f_func, gamma=gamma_use)
        err = np.max(np.abs(u - exact(x)))
        errs3.append(err)
        print(f"      N={N:>3d}: err={err:.4e}")
    rate3 = np.abs(np.polyfit(np.log(hs), np.log(errs3), 1)[0])
    print(f"      => rate = {rate3:.3f}")

    # Summary
    print(f"\n  Summary for alpha={alpha}:")
    print(f"    Standard compact:    rate = {rate1:.3f} (target: {1-alpha:.1f})")
    print(f"    Compact-graded:      rate = {rate2:.3f} (target: {4-alpha:.1f})")
    print(f"    Backward Euler grad: rate = {rate3:.3f} (target: {min(4-alpha, gamma_use*(1-alpha)):.1f})")
    print(f"    Improvement factor:  {rate2/rate3:.2f}x over BE-graded")

# Also run a manufactured smooth test for alpha=0.5
print(f"\n{'='*80}")
print("MANUFACTURED SMOOTH TEST: v(xi) = xi^4")
print("=" * 80)
from scipy.special import beta as beta_func

for alpha in [0.3, 0.5]:
    for gamma in [1.0, 2.0, 4.0]:
        B_val = beta_func(1.0 - alpha, 4.0/gamma + 1.0)
        X_pow = X ** (-4.0 / gamma)
        def f_mfg(x):
            if x < 1e-14: return 0.0
            return (4.0/gamma)*X_pow*x**(4.0/gamma-1.0) - X_pow*x**(1.0-alpha+4.0/gamma)*B_val
        def exact_u(x): return (x/X)**(4.0/gamma)

        print(f"\n  alpha={alpha}, gamma={gamma}:")
        errs = []
        for N in [16, 32, 64, 128]:
            x, u = solve_vide_compact_graded(N, X, alpha, f_mfg, gamma=gamma, smooth_start=True)
            err = np.max(np.abs(u - exact_u(x)))
            errs.append(err)
            print(f"    N={N:>3d}: err={err:.4e}")
        if len(errs) >= 2:
            hs_mfg = [X/n for n in [16,32,64,128]]
            rate = np.abs(np.polyfit(np.log(hs_mfg), np.log(errs), 1)[0])
            print(f"    => rate = {rate:.2f} (target: 4.0, O(h^3) limit for quadratic PI)")

print("\nDone.")
