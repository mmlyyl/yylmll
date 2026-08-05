"""
Compact Finite Difference + Product Integration Method
for Weakly Singular Volterra Integro-Differential Equations.

Solves: u'(x) = f(x) + ∫_0^x (x-t)^(-alpha) * u(t) dt
with u(0) = u0, where 0 < alpha < 1.

Reference convergence rate: O(h^{3}) for smooth solutions,
verified numerically.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time


# ============================================================
# Product integration weights (precomputed, translation-invariant)
# ============================================================

def precompute_basis_integrals(max_dist, alpha, n_gauss=32):
    """
    Precompute integrals of basis functions against singular kernel.

    For non-singular subinterval at distance d >= 1:
    I_n(d) = ∫_0^1 (d+1-s)^{-alpha} * s^n ds, n = 0, 1, 2

    For singular subinterval (d = 0):
    I_n(0) = ∫_0^1 (1-s)^{-alpha} * s^n ds = Beta(1-alpha, n+1)

    Returns arrays of length max_dist+1 for n=0,1,2.
    """
    I0 = np.zeros(max_dist + 1)
    I1 = np.zeros(max_dist + 1)
    I2 = np.zeros(max_dist + 1)

    # Singular case (d=0): exact formula
    I0[0] = 1.0 / (1.0 - alpha)
    I1[0] = beta_func(1.0 - alpha, 2.0)
    I2[0] = beta_func(1.0 - alpha, 3.0)

    # Non-singular (d >= 1): use analytic antiderivative
    # ∫ (d+1-s)^{-alpha} s^n ds
    # For d >= 1, the integrand is smooth, use Gauss-Legendre
    from numpy.polynomial.legendre import leggauss
    gauss_pts, gauss_wts = leggauss(n_gauss)

    for d in range(1, max_dist + 1):
        # Transform from [-1,1] to [0,1]
        s_vals = 0.5 * (gauss_pts + 1.0)
        w_vals = 0.5 * gauss_wts

        for n in range(3):
            integrand = (d + 1.0 - s_vals) ** (-alpha) * s_vals ** n
            val = np.sum(w_vals * integrand)
            if n == 0:
                I0[d] = val
            elif n == 1:
                I1[d] = val
            else:
                I2[d] = val

    return I0, I1, I2


def compute_weight_matrix(N, alpha):
    """
    Compute product integration weight matrix W of size (N+1) × (N+1).

    K_i = ∫_0^{x_i} (x_i - t)^{-alpha} u(t) dt = ∑_{j=0}^{i} W[i, j] * u_j

    Uses piecewise quadratic Lagrange interpolation for u(t)
    on each subinterval [x_{j-1}, x_j].

    For the last (singular) subinterval j=i, uses product integration
    with exact evaluation of the singular weight integral.
    """
    h = 1.0  # Use unit spacing; actual h^{1-alpha} factor applied at assembly

    # Precompute basis integrals
    I0, I1, I2 = precompute_basis_integrals(N, alpha)

    W = np.zeros((N + 1, N + 1))

    # Quadratic Lagrange basis functions on [0, 1]:
    # Using points at s = -1 (left neighbor), 0 (left endpoint), 1 (right endpoint)
    # Wait, we need 3-point interpolation for each subinterval [j-1, j].
    # Use points j-1, j, j+1 when available; j-2, j-1, j at the right end.

    for i in range(1, N + 1):
        for j in range(1, i + 1):
            d = i - j  # distance from singular point

            if j < i:
                # Non-singular: use points j-1, j, j+1 (or adjusted near boundaries)
                # Lagrange basis on reference interval s ∈ [0,1]:
                # L_{j-1}(s) = (s-0)(s-1)/2 = (s^2 - s)/2  ... wait
                # Points: x_{j-1} at s=-1, x_j at s=0, x_{j+1} at s=1
                # But s ∈ [0,1] is on [x_{j-1}, x_j] with s=0 at x_{j-1}, s=1 at x_j
                # Let me redefine: t = x_{j-1} + s*h, s ∈ [0,1]
                # x_{j-1} = t at s=0, x_j = t at s=1, x_{j+1} = t at s=2

                # Lagrange basis:
                # L_{j-1}(s) = (s-1)(s-2)/2
                # L_j(s) = -s(s-2)
                # L_{j+1}(s) = s(s-1)/2

                # ∫_0^1 (d+1-s)^{-alpha} L_k(s) ds

                # L_{j-1}: (s^2 - 3s + 2)/2
                a_jm1 = 0.5
                b_jm1 = -1.5
                c_jm1 = 1.0

                # L_j: -s^2 + 2s
                a_j = -1.0
                b_j = 2.0
                c_j = 0.0

                # L_{j+1}: (s^2 - s)/2
                a_jp1 = 0.5
                b_jp1 = -0.5
                c_jp1 = 0.0

                if j - 1 >= 0:
                    w = a_jm1 * I2[d] + b_jm1 * I1[d] + c_jm1 * I0[d]
                    W[i, j - 1] += w

                w = a_j * I2[d] + b_j * I1[d] + c_j * I0[d]
                W[i, j] += w

                if j + 1 <= N:
                    w = a_jp1 * I2[d] + b_jp1 * I1[d] + c_jp1 * I0[d]
                    W[i, j + 1] += w

            else:
                # Singular subinterval j = i (d = 0)
                # Use points i-2 (s=-1), i-1 (s=0), i (s=1)
                # where t = x_{i-1} + s*h, s ∈ [0,1]
                # x_{i-2} at s=-1, x_{i-1} at s=0, x_i at s=1

                # L_{i-2}(s) = s(s-1)/2
                a_im2 = 0.5
                b_im2 = -0.5
                c_im2 = 0.0

                # L_{i-1}(s) = -(s+1)(s-1) = -s^2 + 1
                a_im1 = -1.0
                b_im1 = 0.0
                c_im1 = 1.0

                # L_i(s) = s(s+1)/2 = (s^2 + s)/2
                a_i = 0.5
                b_i = 0.5
                c_i = 0.0

                if i - 2 >= 0:
                    w = a_im2 * I2[0] + b_im2 * I1[0] + c_im2 * I0[0]
                    W[i, i - 2] += w

                w = a_im1 * I2[0] + b_im1 * I1[0] + c_im1 * I0[0]
                W[i, i - 1] += w

                w = a_i * I2[0] + b_i * I1[0] + c_i * I0[0]
                W[i, i] += w

    return W


# ============================================================
# Solver
# ============================================================

# ============================================================
# Comparison Methods
# ============================================================

def compute_weight_matrix_trapezoidal(N, alpha):
    """
    Product trapezoidal integration: piecewise linear interpolation
    on each subinterval, integrated exactly against singular kernel.
    Convergence: O(h^{2-alpha}) on uniform mesh.
    """
    W = np.zeros((N + 1, N + 1))

    for i in range(1, N + 1):
        for j in range(1, i + 1):
            d = i - j

            if j < i:
                # Non-singular subinterval: use points j-1, j
                # L_{j-1}(s) = 1 - s, L_j(s) = s, s ∈ [0,1]
                I0 = 1.0 / (1.0 - alpha)
                I1_val = 1.0 / ((1.0 - alpha) * (2.0 - alpha))

                # More general: I_n(d) = ∫_0^1 (d+1-s)^{-alpha} s^n ds
                from numpy.polynomial.legendre import leggauss
                gauss_pts, gauss_wts = leggauss(16)
                s_vals = 0.5 * (gauss_pts + 1.0)
                w_vals = 0.5 * gauss_wts

                I0_d = np.sum(w_vals * (d + 1.0 - s_vals) ** (-alpha))
                I1_d = np.sum(w_vals * (d + 1.0 - s_vals) ** (-alpha) * s_vals)

                W[i, j - 1] += I0_d - I1_d  # L_{j-1}: 1-s
                W[i, j] += I1_d               # L_j: s

            else:
                # Singular subinterval: use points i-1, i
                # L_{i-1}(s) = 1-s, L_i(s) = s, s ∈ [0,1]
                from scipy.special import beta as beta_func
                I0_0 = 1.0 / (1.0 - alpha)
                I1_0 = beta_func(1.0 - alpha, 2.0)

                W[i, i - 1] += I0_0 - I1_0
                W[i, i] += I1_0

    return W


def solve_vide_trapezoidal(N, X, alpha, f_func, u0=0.0):
    """
    Product trapezoidal method: piecewise linear interpolation
    with product integration for singular kernel.
    Convergence: O(h^{2-alpha}) on uniform mesh.
    """
    h = X / N
    x = np.linspace(0, X, N + 1)
    f = np.array([f_func(xi) for xi in x])

    W_unit = compute_weight_matrix_trapezoidal(N, alpha)
    scale = h ** (1.0 - alpha)

    A = np.zeros((N, N))
    b = np.zeros(N)

    # Forward Euler + product trapezoidal for integral
    for i in range(1, N + 1):
        row = i - 1

        # u_i = u_{i-1} + h f_{i-1} + h K_{i-1}  (Forward Euler for ODE part)
        if i >= 1:
            A[row, i - 1] += 1.0
        if i >= 2:
            A[row, i - 2] -= 1.0

        # h * K_{i-1} terms
        for j in range(N + 1):
            w_km1_j = scale * W_unit[i - 1, j]
            if j == 0:
                b[row] += h * (f[i - 1] + w_km1_j * u0)
            else:
                A[row, j - 1] -= h * w_km1_j

    u_interior = np.linalg.solve(A, b)
    u = np.zeros(N + 1)
    u[0] = u0
    u[1:] = u_interior

    return x, u


def solve_vide_standard_trapezoidal(N, X, alpha, f_func, u0=0.0):
    """
    Standard trapezoidal method WITHOUT product integration.
    Direct trapezoidal quadrature for the singular integral.
    Expected: O(h^{1-alpha}) convergence on uniform mesh.
    Note: x-t = 0 at upper limit causes blow-up;
          we approximate ∫_0^x (x-t)^{-alpha} u(t) dt using
          composite trapezoidal with singular endpoint excluded,
          plus first subinterval handled analytically.
    """
    h = X / N
    x = np.linspace(0, X, N + 1)
    f = np.array([f_func(xi) for xi in x])

    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N + 1):
        row = i - 1

        # Forward Euler for derivative part
        if i >= 1:
            A[row, i - 1] += 1.0
        if i >= 2:
            A[row, i - 2] -= 1.0

        # Approximate K_{i} using composite trapezoidal
        # For last subinterval, use product trapezoidal to handle singularity
        K_contrib = np.zeros(N + 1)

        # Integrate over [0, x_i]
        for jj in range(1, i + 1):
            # Subinterval [x_{j-1}, x_j]
            s_val = x[i]
            if jj < i:
                # Non-singular: regular trapezoidal
                weight_left = 0.5 * h
                weight_right = 0.5 * h
                K_contrib[jj - 1] += weight_left * (x[i] - x[jj - 1]) ** (-alpha)
                K_contrib[jj] += weight_right * (x[i] - x[jj]) ** (-alpha)
            else:
                # Singular subinterval: product trapezoidal with linear approx
                # ∫_{x_{i-1}}^{x_i} (x_i-t)^{-alpha} [u_{i-1}(x_i-t)/h + u_i(t-x_{i-1})/h] dt
                h_local = h
                I0 = h_local ** (1.0 - alpha) / (1.0 - alpha)
                I1 = h_local ** (2.0 - alpha) / ((1.0 - alpha) * (2.0 - alpha))
                K_contrib[i - 1] += (I0 - I1 / h_local)
                K_contrib[i] += I1 / h_local

        for jj in range(N + 1):
            if jj == 0:
                b[row] += h * K_contrib[0] * u0
            else:
                A[row, jj - 1] -= h * K_contrib[jj]

        b[row] += h * f[i - 1]

    u_interior = np.linalg.solve(A, b)
    u = np.zeros(N + 1)
    u[0] = u0
    u[1:] = u_interior

    return x, u


def solve_vide_compact(N, X, alpha, f_func, u0=0.0, smooth_start=True):
    """
    Solve VIDE using compact finite difference + product integration.

    Scheme:
      (1/6)u'_{i-1} + (2/3)u'_i + (1/6)u'_{i+1} = (u_{i+1} - u_{i-1})/(2h)
    with u'_k = f_k + K_k, K_k = integral with singular kernel.

    smooth_start=True: compact formula from i=1 (original, O(h^4) for smooth u).
    smooth_start=False: backward Euler at i=1 (avoids f(0) blowup for singular u).
    """
    h = X / N
    x = np.linspace(0, X, N + 1)
    f = np.array([f_func(xi) for xi in x])

    # Compute weight matrix (for unit h, scale by h^{1-alpha} at assembly)
    W_unit = compute_weight_matrix(N, alpha)
    scale = h ** (1.0 - alpha)

    # System: A * u[1:] = b
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N + 1):
        row = i - 1
        xi = x[i]

        if i == 1 and not smooth_start:
            # First interior node: use backward Euler to avoid f(0) singularity
            # u_1 - u_0 = h * (f_1 + K_1)  =>  u_1 - h*K_1 = u_0 + h*f_1
            A[row, 0] = 1.0  # u_1
            for j in range(N + 1):
                w_1j = scale * W_unit[1, j]
                if j == 0:
                    b[row] += h * w_1j * u0
                else:
                    A[row, j - 1] -= h * w_1j
            b[row] += u0 + h * f[1]

        elif i < N:
            # Compact scheme at interior node (i >= 2)
            # u_{i+1}/(2h) - u_{i-1}/(2h) =
            #   (1/6)(f_{i-1}+K_{i-1}) + (2/3)(f_i+K_i) + (1/6)(f_{i+1}+K_{i+1})

            A[row, i] += 1.0 / (2.0 * h)  # u_{i+1}

            if i - 2 >= 0:
                A[row, i - 2] -= 1.0 / (2.0 * h)  # u_{i-1}
            else:
                # i = 1 handled above, this shouldn't happen
                b[row] += u0 / (2.0 * h)

            # Integral contributions: move to LHS
            coeffs = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
            for offset, ck in zip([-1, 0, 1], coeffs):
                k = i + offset
                if k < 0 or k > N:
                    continue

                # K_k = scale * sum_j W_unit[k, j] * u_j
                for j in range(N + 1):
                    w_kj = scale * W_unit[k, j]
                    if j == 0:
                        b[row] += ck * w_kj * u0
                    else:
                        A[row, j - 1] -= ck * w_kj

            # RHS: f contributions
            for offset, ck in zip([-1, 0, 1], coeffs):
                k = i + offset
                if 0 <= k <= N:
                    b[row] += ck * f[k]

        else:
            # Right boundary: (3u_N - 4u_{N-1} + u_{N-2})/(2h) = f_N + K_N
            A[row, N - 1] += 3.0  # u_N
            if N >= 2:
                A[row, N - 2] -= 4.0  # u_{N-1}
            if N >= 3:
                A[row, N - 3] += 1.0  # u_{N-2}

            # Integral K_N terms
            for j in range(N + 1):
                w_Nj = scale * W_unit[N, j]
                if j == 0:
                    b[row] += 2.0 * h * w_Nj * u0
                else:
                    A[row, j - 1] -= 2.0 * h * w_Nj

            # RHS
            b[row] += 2.0 * h * f[N]

    u_interior = np.linalg.solve(A, b)

    u = np.zeros(N + 1)
    u[0] = u0
    u[1:] = u_interior

    return x, u


# ============================================================
# Test Problems
# ============================================================

def make_test_u_power(beta=2.0):
    """
    Exact solution u(x) = x^beta.
    u'(x) = beta * x^{beta-1}
    ∫_0^x (x-t)^{-alpha} t^beta dt = x^{1-alpha+beta} B(1-alpha, beta+1)
    """
    def f_func(x, alpha):
        B_val = beta_func(1.0 - alpha, beta + 1.0)
        if x < 1e-14:
            return 0.0 if beta > 1 else 1.0
        return beta * x ** (beta - 1.0) - x ** (1.0 - alpha + beta) * B_val

    def exact(x):
        return x ** beta

    return f_func, exact


def make_test_u_sine():
    """Exact solution u(x) = sin(pi*x). f(x) computed by quadrature."""
    def f_func(x, alpha):
        uprime = np.pi * np.cos(np.pi * x)
        if x < 1e-14:
            return uprime
        integral, _ = quad(lambda t: (x - t) ** (-alpha) * np.sin(np.pi * t),
                          0, x, limit=200, epsabs=1e-13, epsrel=1e-13)
        return uprime - integral

    def exact(x):
        return np.sin(np.pi * x)

    return f_func, exact


# ============================================================
# Genuinely Singular Test Problems
# ============================================================

def make_test_u_singular1():
    """
    u(x) = x^{1-alpha} + x^2
    Contains the characteristic weak singularity of VIDE solutions.
    u'(x) = (1-alpha)*x^{-alpha} + 2x  (derivative blows up at x=0 for alpha>0)
    """
    def f_func(x, alpha):
        if x < 1e-14:
            return 1e10 if alpha > 0 else 0.0
        uprime = (1.0 - alpha) * x ** (-alpha) + 2.0 * x
        B1 = beta_func(1.0 - alpha, 2.0 - alpha)
        B2 = beta_func(1.0 - alpha, 3.0)
        integral = x ** (2.0 - 2.0 * alpha) * B1 + x ** (3.0 - alpha) * B2
        return uprime - integral

    def exact(x):
        return x ** (1.0 - np.array(x, dtype=float) * 0 + alpha)  # will be overridden
    # exact needs alpha at call time — handled in run_convergence_singular
    return f_func, None  # exact depends on alpha


def make_test_u_singular2():
    """
    u(x) = x^{1-alpha} + x^{2-alpha} + x^3
    Two singular terms — tests quadrature with multiple non-smooth components.
    """
    def f_func(x, alpha):
        if x < 1e-14:
            return 1e10 if alpha > 0 else 0.0
        uprime = ((1.0 - alpha) * x ** (-alpha)
                  + (2.0 - alpha) * x ** (1.0 - alpha)
                  + 3.0 * x ** 2.0)
        B1 = beta_func(1.0 - alpha, 2.0 - alpha)
        B2 = beta_func(1.0 - alpha, 3.0 - alpha)
        B3 = beta_func(1.0 - alpha, 4.0)
        integral = (x ** (2.0 - 2.0 * alpha) * B1
                    + x ** (3.0 - 2.0 * alpha) * B2
                    + x ** (4.0 - alpha) * B3)
        return uprime - integral

    return f_func, None


def make_test_u_viscoelastic():
    """
    Application-motivated test: fractional viscoelasticity.
    u(x) = x^{1-alpha} * E_{1-alpha, 2-alpha}(-x^{1-alpha}) + 0.5 * sin(2x)
    where E_{a,b} is the two-parameter Mittag-Leffler function.

    The Mittag-Leffler part approximates the relaxation response
    of a fractional Kelvin-Voigt material under step loading.
    For simplicity we use the asymptotic form near x=0.

    Simplified: u(x) = x^{1-alpha} * exp(-x) + 0.5*x^2
    (product of singular term and exponential decay + smooth polynomial)
    """
    def f_func(x, alpha):
        if x < 1e-14:
            return 1e10 if alpha > 0 else 0.0
        # u(x) = x^{1-alpha} * exp(-x) + 0.5*x^2
        # u'(x) = (1-alpha)*x^{-alpha}*exp(-x) - x^{1-alpha}*exp(-x) + x
        uprime = ((1.0 - alpha) * x ** (-alpha) - x ** (1.0 - alpha)) * np.exp(-x) + x

        # Integral computed by high-order quadrature
        integral, _ = quad(
            lambda t: (x - t) ** (-alpha) * (t ** (1.0 - alpha) * np.exp(-t) + 0.5 * t ** 2),
            0, x, limit=200, epsabs=1e-13, epsrel=1e-13
        )
        return uprime - integral

    def exact(x, alpha):
        return x ** (1.0 - alpha) * np.exp(-x) + 0.5 * x ** 2

    return f_func, exact


# ============================================================
# Graded Mesh Solver (Brunner-style)
# ============================================================

def solve_vide_graded_mesh(N, X, alpha, f_func, u0=0.0, gamma=None):
    """
    Solve VIDE on a graded mesh x_j = X * (j/N)^gamma.
    Uses backward Euler + product integration with piecewise quadratic
    interpolation adapted to non-uniform spacing.

    If gamma is None, uses gamma = (2-alpha)/(1-alpha) (robust practical grading).
    """
    if gamma is None:
        gamma = (2.0 - alpha) / (1.0 - alpha) if alpha < 1.0 else 5.0
    gamma = min(gamma, 8.0)  # Prevent extreme grading that causes numerical issues

    x = X * (np.arange(N + 1) / N) ** gamma
    x[0] = 0.0

    f = np.array([f_func(xi) for xi in x])

    # Precompute product integration weights for each row i
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N + 1):
        row = i - 1
        h_i = x[i] - x[i - 1]
        xi = x[i]

        # Backward Euler: (u_i - u_{i-1}) / h_i = f_i + K_i
        A[row, i - 1] = 1.0
        if i >= 2:
            A[row, i - 2] -= 1.0
        else:
            b[row] += u0

        # K_i = ∫_0^{xi} (xi - t)^{-alpha} u(t) dt
        # Integrate subinterval by subinterval with quadratic interpolation
        for j in range(1, i + 1):
            a_j, b_j = x[j - 1], x[j]

            if j < i:
                # Non-singular: 32-pt Gauss-Legendre
                K_contrib = _nonsingular_subinterval(xi, a_j, b_j, alpha, j, i, N, x)
            else:
                # Singular subinterval: use Gauss-Jacobi adapted nodes
                K_contrib = _singular_subinterval(xi, a_j, b_j, alpha, j, x)

            # Distribute contributions to A and b
            w = K_contrib  # array of length N+1 with contributions to each node
            for k, val in enumerate(w):
                if abs(val) < 1e-16:
                    continue
                if k == 0:
                    b[row] += h_i * val * u0
                else:
                    A[row, k - 1] -= h_i * val

        b[row] += h_i * f[i]

    u_interior = np.linalg.solve(A, b)
    u = np.zeros(N + 1)
    u[0] = u0
    u[1:] = u_interior
    return x, u


def _nonsingular_subinterval(xi, a, b, alpha, j, i, N, x_all):
    """
    Integrate ∫_a^b (xi - t)^{-alpha} * u(t) dt on non-singular subinterval.
    Uses quadratic Lagrange interpolation through 3 nearest nodes.
    Returns contribution vector w of length N+1.
    """
    from numpy.polynomial.legendre import leggauss
    w = np.zeros(len(x_all))

    # Select 3 interpolation nodes
    nodes = [j - 1, j, min(j + 1, N)]
    if j == 1:
        nodes = [0, 1, min(2, N)]

    gauss_pts, gauss_wts = leggauss(16)
    h_j = b - a

    for gp, gw in zip(gauss_pts, gauss_wts):
        t = a + 0.5 * (gp + 1.0) * h_j
        kernel = (xi - t) ** (-alpha)
        weight = gw * 0.5 * h_j * kernel

        for k in nodes:
            Lk = 1.0
            for m in nodes:
                if m != k:
                    denom = x_all[k] - x_all[m]
                    if abs(denom) > 1e-15:
                        Lk *= (t - x_all[m]) / denom
            w[k] += weight * Lk

    return w


def _singular_subinterval(xi, a, b, alpha, j, x_all):
    """
    Integrate ∫_a^b (xi - t)^{-alpha} * u(t) dt on singular subinterval [x_{i-1}, x_i].

    The kernel is singular at t=xi=b. Use quadratic interpolation through
    i-2, i-1, i and integrate with high-order Gauss-Jacobi quadrature
    matched to the (xi-t)^{-alpha} weight.
    """
    N = len(x_all) - 1
    i = j  # j == i for singular subinterval
    w = np.zeros(len(x_all))

    nodes = [max(0, i - 2), i - 1, i]
    # Deduplicate
    nodes = list(dict.fromkeys(nodes))

    h_j = b - a
    # Use 48-pt Gauss-Legendre (the integrand is singular but integrable)
    from numpy.polynomial.legendre import leggauss
    gauss_pts, gauss_wts = leggauss(48)

    for gp, gw in zip(gauss_pts, gauss_wts):
        t = a + 0.5 * (gp + 1.0) * h_j
        kernel = (xi - t) ** (-alpha)
        weight = gw * 0.5 * h_j * kernel

        for k in nodes:
            Lk = 1.0
            for m in nodes:
                if m != k:
                    denom = x_all[k] - x_all[m]
                    if abs(denom) > 1e-15:
                        Lk *= (t - x_all[m]) / denom
            w[k] += weight * Lk

    return w


def _nonsingular_subinterval_fast(xi, a, b, alpha, j, i, N, x_all):
    """Vectorized version for speed (deprecated, keep for reference)."""
    return _nonsingular_subinterval(xi, a, b, alpha, j, i, N, x_all)


# ============================================================
# Compact scheme on graded mesh via coordinate transformation
# ============================================================

def _stable_gamma_diff(a, b, gamma):
    """Compute a^gamma - b^gamma without catastrophic cancellation.

    For integer gamma: exact factorization a^g - b^g = (a-b) * sum_{p=0}^{g-1} a^p * b^{g-1-p}.
    For non-integer gamma: a^gamma * (1 - (b/a)^gamma), stable when b≈a.
    """
    if a < b:
        return -_stable_gamma_diff(b, a, gamma)
    if b <= 0:
        return a ** gamma
    g_int = int(gamma)
    if abs(gamma - g_int) < 1e-14:
        # Integer exponent: exact factorization
        diff = a - b
        geo_sum = 0.0
        for p in range(g_int):
            geo_sum += a ** p * b ** (g_int - 1 - p)
        return diff * geo_sum
    else:
        # Non-integer: ratio method
        ratio = b / a
        return a ** gamma * (1.0 - ratio ** gamma)


def solve_vide_compact_graded(N, X, alpha, f_func, u0=0.0, gamma=None, smooth_start=True):
    """
    Compact finite difference on graded mesh via coordinate transformation.

    Transform x = X * xi^gamma, where xi in [0,1] is UNIFORM.
    Let v(xi) = u(x(xi)) = u(X * xi^gamma).

    The VIDE u'(x) = f(x) + int_0^x (x-t)^(-alpha) u(t) dt becomes
    an integro-differential equation for v(xi) on a UNIFORM xi-grid,
    to which the compact Padé stencil is directly applied.

    The transformed kernel retains the standard (xi-s)^(-alpha)
    singularity, so product integration with Beta-function weights
    carries over with a variable coefficient.
    """
    import warnings
    from numpy.polynomial.legendre import leggauss

    if gamma is None:
        gamma = (2.0 - alpha) / (1.0 - alpha) if alpha < 1.0 else 5.0
    gamma_clamp = min(gamma, 8.0)
    if gamma > 8.0:
        warnings.warn(f"gamma={gamma:.1f} capped to 8.0 for numerical stability")

    # Uniform xi-grid
    h_xi = 1.0 / N
    xi = np.linspace(0, 1, N + 1)

    # Mapped physical grid (for output)
    x_phys = X * xi ** gamma_clamp
    x_phys[0] = 0.0

    # Precompute g(xi) and h(xi) coefficients
    g = np.zeros(N + 1)
    h = np.zeros(N + 1)
    for i in range(N + 1):
        if i == 0:
            g[0] = 0.0
            h[0] = 0.0
        else:
            xi_i = xi[i]
            pow_xi = xi_i ** (gamma_clamp - 1.0)
            g[i] = X * gamma_clamp * pow_xi * f_func(X * xi_i ** gamma_clamp)
            h[i] = X ** (2.0 - alpha) * gamma_clamp ** 2.0 * pow_xi

    # Precompute analytic weight matrix for gamma=1 fast path
    if gamma_clamp == 1.0:
        W_unit = compute_weight_matrix(N, alpha)
        scale_xi = h_xi ** (1.0 - alpha)
    else:
        W_unit = None
        scale_xi = None
        gl_pts, gl_wts = leggauss(16)

    # System assembly
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N + 1):
        row = i - 1

        # Determine which v' values appear in this row
        if i == 1 and not smooth_start:
            ks = [1]  # backward Euler: only v'_1
        elif i < N:
            ks = [i - 1, i, i + 1]  # compact stencil
        else:
            ks = [N]  # i == N: one-sided, only v'_N

        c_coeff = {i - 1: 1.0 / 6.0, i: 2.0 / 3.0, i + 1: 1.0 / 6.0}

        # --- Phase 1: accumulate h_k * I_k contributions from each k ---
        for k in ks:
            xi_k = xi[k]
            if k == 0:
                continue
            # Integral coefficient: backward Euler uses h_xi, interior
            # (and smooth i=1) uses Padé weights, right boundary uses 2*h_xi.
            if i == 1 and not smooth_start:
                ck = h_xi
            elif i < N:
                ck = c_coeff[k]
            else:
                ck = 2.0 * h_xi

            if W_unit is not None:
                # gamma=1: use exact analytic weight matrix (same as standard solver)
                w_factor = h[k] * ck * scale_xi
                for j in range(N + 1):
                    w_kj = w_factor * W_unit[k, j]
                    if j == 0:
                        b[row] += w_kj * u0
                    else:
                        A[row, j - 1] -= w_kj
            else:
                for j in range(1, k + 1):
                    a_s, b_s = xi[j - 1], xi[j]

                    if j < k:
                        # Non-singular: 16-pt Gauss-Legendre
                        for gp, gw in zip(gl_pts, gl_wts):
                            s = a_s + 0.5 * (gp + 1.0) * (b_s - a_s)
                            denom = _stable_gamma_diff(xi_k, s, gamma_clamp)
                            if denom <= 0:
                                continue
                            kernel_val = denom ** (-alpha) * s ** (gamma_clamp - 1.0)
                            weight = gw * 0.5 * (b_s - a_s) * kernel_val

                            nodes = [j - 1, j, min(j + 1, N)]
                            if j == 1:
                                nodes = [0, 1, min(2, N)]
                            seen = set()
                            nodes = [nd for nd in nodes if not (nd in seen or seen.add(nd))]

                            for nd in nodes:
                                Lk_val = 1.0
                                for m in nodes:
                                    if m != nd:
                                        d = xi[nd] - xi[m]
                                        Lk_val *= (s - xi[m]) / d if abs(d) > 1e-15 else 0.0
                                contrib = h[k] * ck * weight * Lk_val
                                if nd == 0:
                                    b[row] += contrib * u0
                                else:
                                    A[row, nd - 1] -= contrib

                    else:  # j == k: singular subinterval
                        # Remove singularity via u = t^(1-alpha) transformation.
                        # t ∈ [0,1], u = t^(1-alpha), dt/du = (1/(1-alpha)) * u^(alpha/(1-alpha))
                        h_sub = b_s - a_s
                        pow_alpha = 1.0 / (1.0 - alpha)
                        pow_beta = alpha / (1.0 - alpha)

                        # 32-pt Gauss-Legendre on u ∈ [0,1]
                        spts, swts = leggauss(32)

                        for offset, nd in enumerate([k - 2, k - 1, k]):
                            if nd < 0:
                                continue
                            integral_nd = 0.0
                            for gp, gw in zip(spts, swts):
                                u = 0.5 * (gp + 1.0)
                                t = u ** pow_alpha
                                dt_du = pow_alpha * u ** pow_beta

                                s = xi_k - t * h_sub
                                denom = _stable_gamma_diff(xi_k, s, gamma_clamp)
                                if denom <= 0:
                                    continue
                                kernel = denom ** (-alpha) * s ** (gamma_clamp - 1.0)

                                # Jacobian: h_sub * kernel * dt_du * gw * 0.5
                                jac = h_sub * kernel * dt_du * 0.5 * gw

                                # Fixed quadratic Lagrange basis:
                                #   L_{k-2}(t) = t*(t-1)/2
                                #   L_{k-1}(t) = 2t - t^2
                                #   L_k(t)     = 1 - 1.5t + 0.5t^2
                                if offset == 0:
                                    L_val = t * (t - 1.0) * 0.5
                                elif offset == 1:
                                    L_val = 2.0 * t - t * t
                                else:
                                    L_val = 1.0 - 1.5 * t + 0.5 * t * t

                                integral_nd += jac * L_val

                            contrib = h[k] * ck * integral_nd
                            if nd == 0:
                                b[row] += contrib * u0
                            else:
                                A[row, nd - 1] -= contrib

        # --- Phase 2: assemble stencil row and g_k RHS contributions ---
        if i == 1 and not smooth_start:
            A[row, 0] += 1.0
            b[row] += u0 + h_xi * g[1]
        elif i < N:
            A[row, i] += 1.0 / (2.0 * h_xi)
            if i - 2 >= 0:
                A[row, i - 2] -= 1.0 / (2.0 * h_xi)
            else:
                b[row] += u0 / (2.0 * h_xi)
            for k in ks:
                b[row] += c_coeff[k] * g[k]
        else:
            A[row, N - 1] += 3.0
            if N >= 2:
                A[row, N - 2] -= 4.0
            if N >= 3:
                A[row, N - 3] += 1.0
            b[row] += 2.0 * h_xi * g[N]

    # Solve and return
    v_interior = np.linalg.solve(A, b)
    v = np.zeros(N + 1)
    v[0] = u0
    v[1:] = v_interior

    return x_phys, v


# ============================================================
# Convergence Study (extended)
# ============================================================

def run_convergence_singular(alpha_values=None, N_values=None, X=1.0):
    """Run convergence study on genuinely singular test problems."""
    if alpha_values is None:
        alpha_values = [0.2, 0.5, 0.8]
    if N_values is None:
        N_values = [16, 32, 64, 128, 256]

    # Test 2: u(x) = x^{1-alpha} + x^2
    print("\n--- Test 2: u(x) = x^{1-alpha} + x^2 (genuine weak singularity) ---")
    f_func_factory, _ = make_test_u_singular1()
    results_sing1 = {}
    for alpha in alpha_values:
        f_func = lambda x: f_func_factory(x, alpha)
        exact_func = lambda x: x ** (1.0 - alpha) + x ** 2.0
        errs = []; hs = []
        for N in N_values:
            x, u = solve_vide_compact(N, X, alpha, f_func, smooth_start=False)
            u_ex = exact_func(x)
            err = np.max(np.abs(u - u_ex))
            errs.append(err); hs.append(X / N)
            print(f"  compact N={N:>4d}, err={err:.4e}")
        log_h, log_e = np.log(hs), np.log(errs)
        rate = np.abs(np.polyfit(log_h, log_e, 1)[0])
        results_sing1[alpha] = {'N': N_values, 'h': hs, 'errors': errs, 'rate': rate}
        pred_rate = 4.0 - alpha
        print(f"  => rate={rate:.3f} (theoretical for smooth: {pred_rate:.3f}, degradation: {pred_rate - rate:.2f})")

    return results_sing1


def run_convergence_graded(alpha_values=None, N_values=None, X=1.0):
    """Run convergence study with graded mesh (backward Euler) on singular problems."""
    if alpha_values is None:
        alpha_values = [0.2, 0.5, 0.8]
    if N_values is None:
        N_values = [16, 32, 64, 128, 256]

    f_func_factory, _ = make_test_u_singular1()
    results_graded = {}
    for alpha in alpha_values:
        gamma = (4.0 - alpha) / (1.0 - alpha)
        f_func = lambda x: f_func_factory(x, alpha)
        exact_func = lambda x: x ** (1.0 - alpha) + x ** 2.0
        errs = []; hs = []
        for N in N_values:
            x, u = solve_vide_graded_mesh(N, X, alpha, f_func, gamma=gamma)
            u_ex = exact_func(x)
            err = np.max(np.abs(u - u_ex))
            errs.append(err); hs.append(X / N)
            print(f"  graded N={N:>4d}, gamma={gamma:.2f}, err={err:.4e}")
        log_h, log_e = np.log(hs), np.log(errs)
        rate = np.abs(np.polyfit(log_h, log_e, 1)[0])
        results_graded[alpha] = {'N': N_values, 'h': hs, 'errors': errs, 'rate': rate}
        print(f"  => graded rate={rate:.3f} (target: {4.0 - alpha:.3f})")

    return results_graded


def run_convergence_compact_graded(alpha_values=None, N_values=None, X=1.0):
    """Run convergence study with compact-on-graded-mesh (coordinate transform)."""
    if alpha_values is None:
        alpha_values = [0.2, 0.5, 0.8]
    if N_values is None:
        N_values = [16, 32, 64, 128, 256]

    f_func_factory, _ = make_test_u_singular1()
    results = {}
    for alpha in alpha_values:
        gamma = (4.0 - alpha) / (1.0 - alpha)
        f_func = lambda x: f_func_factory(x, alpha)
        exact_func = lambda x: x ** (1.0 - alpha) + x ** 2.0
        errs = []; hs = []
        for N in N_values:
            try:
                x, u = solve_vide_compact_graded(N, X, alpha, f_func, gamma=gamma)
                u_ex = exact_func(x)
                err = np.max(np.abs(u - u_ex))
            except Exception as e:
                err = np.nan
                print(f"  compact-graded N={N:>4d}, gamma={gamma:.2f}, FAILED: {e}")
            errs.append(err); hs.append(X / N)
            print(f"  compact-graded N={N:>4d}, gamma={gamma:.2f}, err={err:.4e}")
        # Filter out NaN for rate calculation
        valid = [j for j, e in enumerate(errs) if not np.isnan(e) and e > 0]
        if len(valid) >= 2:
            log_h = np.log([hs[j] for j in valid])
            log_e = np.log([errs[j] for j in valid])
            rate = np.abs(np.polyfit(log_h, log_e, 1)[0])
        else:
            rate = np.nan
        results[alpha] = {'N': N_values, 'h': hs, 'errors': errs, 'rate': rate, 'gamma': gamma}
        print(f"  => compact-graded rate={rate:.3f} (target: {4.0 - alpha:.3f}, "
              f"BE-graded cf. {1.0:.1f})")

    return results


def run_convergence_singular2(alpha_values=None, N_values=None, X=1.0):
    """Test 3: u(x) = x^{1-alpha} + x^{2-alpha} + x^3 (two singular terms)."""
    if alpha_values is None:
        alpha_values = [0.3, 0.5, 0.7]
    if N_values is None:
        N_values = [16, 32, 64, 128, 256]

    f_func_factory, _ = make_test_u_singular2()
    results = {}
    for alpha in alpha_values:
        f_func = lambda x: f_func_factory(x, alpha)
        exact_func = lambda x: x ** (1.0 - alpha) + x ** (2.0 - alpha) + x ** 3.0
        errs = []; hs = []; errs_graded = []
        for N in N_values:
            # Uniform mesh (compact, singular start)
            x, u = solve_vide_compact(N, X, alpha, f_func, smooth_start=False)
            err = np.max(np.abs(u - exact_func(x)))
            errs.append(err)
            # Graded mesh
            xg, ug = solve_vide_graded_mesh(N, X, alpha, f_func)
            err_g = np.max(np.abs(ug - exact_func(xg)))
            errs_graded.append(err_g)
            hs.append(X / N)
            print(f"  N={N:>4d}: uniform_err={err:.4e}, graded_err={err_g:.4e}")

        log_h = np.log(hs)
        rate_u = np.abs(np.polyfit(log_h, np.log(errs), 1)[0])
        rate_g = np.abs(np.polyfit(log_h, np.log(errs_graded), 1)[0])
        results[alpha] = {
            'N': N_values, 'h': hs,
            'errors_uniform': errs, 'rate_uniform': rate_u,
            'errors_graded': errs_graded, 'rate_graded': rate_g
        }
        print(f"  => uniform rate={rate_u:.3f}, graded rate={rate_g:.3f}")

    return results


def run_viscoelastic_test(N_values=None, X=2.0):
    """Test 5: viscoelasticity-motivated problem."""
    if N_values is None:
        N_values = [32, 64, 128, 256, 512]

    f_func_factory, exact_factory = make_test_u_viscoelastic()
    alpha_test = 0.5
    f_func = lambda x: f_func_factory(x, alpha_test)
    exact_func = lambda x: exact_factory(x, alpha_test)

    print(f"\n--- Test 5: Viscoelasticity problem, alpha={alpha_test} ---")
    errs = []; hs = []
    for N in N_values:
        x, u = solve_vide_compact(N, X, alpha_test, f_func, smooth_start=False)
        u_ex = exact_func(x)
        err = np.max(np.abs(u - u_ex))
        errs.append(err); hs.append(X / N)
        print(f"  compact N={N:>4d}, err={err:.4e}")

    log_h, log_e = np.log(hs), np.log(errs)
    rate = np.abs(np.polyfit(log_h, log_e, 1)[0])
    print(f"  => rate={rate:.3f} (theoretical: {4.0 - alpha_test:.3f})")
    return {'N': N_values, 'h': hs, 'errors': errs, 'rate': rate, 'alpha': alpha_test}


# ============================================================
# Legacy wrappers (for backward compatibility)
# ============================================================

def run_convergence_legacy(alpha_values=None, N_values=None, X=1.0, method='compact'):
    """Legacy convergence study for smooth u(x)=x^2."""
    if alpha_values is None:
        alpha_values = [0.2, 0.5, 0.8]
    if N_values is None:
        N_values = [16, 32, 64, 128, 256]
    solvers = {'compact': solve_vide_compact, 'trapezoidal': solve_vide_trapezoidal}
    solver = solvers[method]
    results = {}
    f_func, exact = make_test_u_power(beta=2.0)
    timings = {}
    for alpha in alpha_values:
        errs = []; hs = []; times = []
        for N in N_values:
            t0 = time.time()
            x, u = solver(N, X, alpha, lambda x: f_func(x, alpha))
            elapsed = time.time() - t0
            u_ex = exact(x)
            err = np.max(np.abs(u - u_ex))
            errs.append(err); hs.append(X / N); times.append(elapsed)
            print(f"  [{method}] alpha={alpha}, N={N:>4d}, err={err:.4e}, time={elapsed:.4f}s")
        log_h, log_e = np.log(hs), np.log(errs)
        rate = np.abs(np.polyfit(log_h, log_e, 1)[0])
        results[alpha] = {'N': N_values, 'h': hs, 'errors': errs, 'rate': rate}
        timings[alpha] = times
        print(f"  => [{method}] alpha={alpha}: rate = {rate:.3f}\n")
    return results, timings


def run_sine_test_legacy(alpha=0.3, N_values=None, X=1.0):
    """Legacy sine test."""
    if N_values is None:
        N_values = [32, 64, 128, 256]
    f_func_gen, exact = make_test_u_sine()
    f_func = lambda x: f_func_gen(x, alpha)
    errs = []; hs = []
    for N in N_values:
        x, u = solve_vide_compact(N, X, alpha, f_func)
        u_ex = exact(x)
        err = np.max(np.abs(u - u_ex))
        errs.append(err); hs.append(X / N)
        print(f"  sin test: N={N:>4d}, err={err:.4e}")
    log_h, log_e = np.log(hs), np.log(errs)
    rate = np.abs(np.polyfit(log_h, log_e, 1)[0])
    print(f"  => sin(pi*x), alpha={alpha}: rate = {rate:.3f}")
    return {'N': N_values, 'h': hs, 'errors': errs, 'rate': rate}


def print_latex_table_legacy(conv_results, title="Error Table"):
    """Legacy LaTeX table printer."""
    print(f"\n{'='*65}")
    print(f"LaTeX {title}")
    print("=" * 65)
    N_vals = conv_results[list(conv_results.keys())[0]]['N']
    alphas = sorted(conv_results.keys())
    header = "N & h & " + " & ".join([f"$\\alpha={a}$ & rate" for a in alphas]) + " \\\\"
    print(header)
    print("\\midrule")
    for idx, N in enumerate(N_vals):
        h = conv_results[alphas[0]]['h'][idx]
        parts = [f"{N}", f"{h:.6f}"]
        for a in alphas:
            err = conv_results[a]['errors'][idx]
            if idx > 0:
                prev_err = conv_results[a]['errors'][idx - 1]
                prev_h = conv_results[a]['h'][idx - 1]
                rate = np.log(err / prev_err) / np.log(h / prev_h)
                parts.append(f"{err:.3e} & {rate:.2f}")
            else:
                parts.append(f"{err:.3e} & —")
        print(" & ".join(parts) + " \\\\")
    parts = ["\\multicolumn{2}{c}{Avg. rate}"]
    for a in alphas:
        parts.append(f"\\multicolumn{{2}}{{c}}{{{conv_results[a]['rate']:.2f}}}")
    print(" & ".join(parts) + " \\\\")
    print()


# ============================================================
# Extended Figure Generation
# ============================================================

def _setup_paper_style():
    """Configure matplotlib for journal-quality figures matching LaTeX elsarticle."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.3,
        'lines.markersize': 4.5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.4,
    })

# Academic color palette: 3-color + 2 reference lines
C0, C1, C2 = '#1b6e3a', '#b84215', '#2c5aa0'   # green, rust, blue — distinct when printed B/W
C_REF = '#555555'
C_GRADED = '#b84215'  # rust — compact-graded (main proposal)
C_UNIFORM = '#2c5aa0' # blue — uniform baseline
C_BE = '#1b6e3a'      # green — backward Euler reference


def make_extended_figures(results, save_dir='figures'):
    """Generate journal-quality figures for revised paper."""
    _setup_paper_style()
    os.makedirs(save_dir, exist_ok=True)

    smooth = results['smooth']
    singular1 = results.get('singular1')
    graded = results.get('graded')
    compact_graded = results.get('compact_graded')
    singular2 = results.get('singular2')
    visco = results.get('viscoelastic')
    trap = results.get('trapezoidal')

    # --- Figure 1: Smooth test convergence ---
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    alphas_smooth = [0.2, 0.5, 0.8]
    markers = ['o', 's', 'D']
    for alpha, m in zip(alphas_smooth, markers):
        res = smooth[alpha]
        ax.loglog(res['h'], res['errors'], marker=m, mfc='none', lw=1.3, ms=5,
                  label=rf'$\alpha={alpha}$ ({res["rate"]:.2f})')
    h_ref = smooth[0.2]['h']
    ax.loglog(h_ref, np.array(h_ref)**3 * 3e-2, '--', color=C_REF, lw=0.9,
              label=r'$O(h^3)$')
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$\|u - u_h\|_\infty$')
    ax.legend(loc='lower right')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig1_smooth.pdf'))
    fig.savefig(os.path.join(save_dir, 'fig1_smooth.png'))
    plt.close(fig)

    # --- Figure 2: Singular — uniform vs compact-graded (3 stacked panels) ---
    if singular1 and compact_graded:
        fig, axes = plt.subplots(3, 1, figsize=(4.8, 5.6))
        for idx, alpha in enumerate([0.2, 0.5, 0.8]):
            ax = axes[idx]
            su = singular1[alpha]
            cg = compact_graded.get(alpha)
            gr = graded.get(alpha) if graded else None
            ax.loglog(su['h'], su['errors'], 'o-', color=C_UNIFORM, lw=1.3, ms=5,
                      mfc='none', label=f'Uniform ($1-\\alpha$={1-alpha:.1f}, obs. {su["rate"]:.2f})')
            if cg and not np.isnan(cg['rate']):
                ax.loglog(cg['h'], cg['errors'], 'D-', color=C_GRADED, lw=1.3, ms=5,
                          label=f'Compact-graded (obs. {cg["rate"]:.2f})')
            if gr:
                ax.loglog(gr['h'], gr['errors'], 's--', color=C_BE, lw=1.0, ms=4.5,
                          alpha=0.7, label=f'BE-graded ({gr["rate"]:.2f})')
            ax.set_xlabel(r'$h$ (equiv. uniform)')
            ax.set_ylabel(r'$\|u - u_h\|_\infty$')
            ax.set_title(rf'$\alpha={alpha}$', fontweight='bold', loc='left')
            ax.legend(loc='lower right')
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'fig2_singular_comparison.pdf'))
        fig.savefig(os.path.join(save_dir, 'fig2_singular_comparison.png'))
        plt.close(fig)

    # --- Figure 3: Degradation summary ---
    if singular1:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        alphas_plot = np.linspace(0.05, 0.85, 50)
        pred_rates = 4.0 - alphas_plot
        sing_rates = 1.0 - alphas_plot
        avail_alphas = sorted(singular1.keys())
        actual_rates = [singular1[a]['rate'] for a in avail_alphas]
        ax.plot(avail_alphas, actual_rates, 'o-', color=C_UNIFORM, lw=1.5, ms=6,
                mfc='none', label='Uniform (observed)')
        if compact_graded:
            cg_alphas = sorted([a for a in compact_graded.keys()
                                if not np.isnan(compact_graded[a]['rate'])])
            cg_rates = [compact_graded[a]['rate'] for a in cg_alphas]
            ax.plot(cg_alphas, cg_rates, 'D-', color=C_GRADED, lw=1.5, ms=5.5,
                    label='Compact-graded')
        if graded:
            gr_alphas = sorted(graded.keys())
            gr_rates = [graded[a]['rate'] for a in gr_alphas]
            ax.plot(gr_alphas, gr_rates, 's--', color=C_BE, lw=1.2, ms=5,
                    alpha=0.7, label='BE-graded')
        ax.plot(alphas_plot, pred_rates, '--', color=C_REF, lw=0.9, alpha=0.6,
                label=r'$4-\alpha$ (smooth)')
        ax.plot(alphas_plot, sing_rates, ':', color=C_REF, lw=0.9, alpha=0.6,
                label=r'$1-\alpha$ (singular)')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Observed convergence rate')
        ax.legend(loc='lower left', ncol=2)
        ax.grid(True)
        ax.set_ylim(0, 4.2)
        ax.set_xlim(0, 0.9)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'fig3_degradation.pdf'))
        fig.savefig(os.path.join(save_dir, 'fig3_degradation.png'))
        plt.close(fig)

    # --- Figure 4: Two-singular-term test (2 columns, not split) ---
    if singular2:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        ax1, ax2 = axes
        for alpha, res in singular2.items():
            ax1.loglog(res['h'], res['errors_uniform'], 'o-', lw=1.3, ms=4.5,
                       mfc='none', label=rf'$\alpha={alpha}$ ({res["rate_uniform"]:.2f})')
            ax2.loglog(res['h'], res['errors_graded'], 's-', lw=1.3, ms=4.5,
                       mfc='none', label=rf'$\alpha={alpha}$ ({res["rate_graded"]:.2f})')
        ax1.set_xlabel(r'$h$'); ax1.set_ylabel(r'$\|e\|_\infty$')
        ax1.set_title('Uniform mesh', fontweight='bold', loc='left')
        ax1.legend(); ax1.grid(True)
        ax2.set_xlabel(r'$h$'); ax2.set_ylabel(r'$\|e\|_\infty$')
        ax2.set_title('Graded mesh (Brunner)', fontweight='bold', loc='left')
        ax2.legend(); ax2.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'fig4_twosingular.pdf'))
        fig.savefig(os.path.join(save_dir, 'fig4_twosingular.png'))
        plt.close(fig)

    # --- Figure 5: Viscoelastic application ---
    if visco:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
        alpha_v = visco['alpha']
        f_func_factory, exact_factory = make_test_u_viscoelastic()
        f_func_v = lambda x: f_func_factory(x, alpha_v)
        exact_v = lambda x: exact_factory(x, alpha_v)
        x_dense = np.linspace(0, 2, 1000)
        x_sol, u_sol = solve_vide_compact(128, 2.0, alpha_v, f_func_v, smooth_start=False)
        ax1.plot(x_dense, exact_v(x_dense), 'k-', lw=1.2, label='Exact')
        ax1.plot(x_sol, u_sol, 'o', color=C_GRADED, ms=3.5, label=r'Compact ($N=128$)')
        ax1.set_xlabel(r'$x$'); ax1.set_ylabel(r'$u(x)$')
        ax1.set_title(rf'Solution profile ($\alpha={alpha_v}$)', fontweight='bold', loc='left')
        ax1.legend(); ax1.grid(True)
        ax2.loglog(visco['h'], visco['errors'], 'o-', color=C_UNIFORM, lw=1.3, ms=5,
                   mfc='none', label=f"Observed rate = {visco['rate']:.2f}")
        ax2.loglog(visco['h'], np.array(visco['h'])**(4-alpha_v) * 5e-3,
                   '--', color=C_REF, lw=0.9, label=rf'$O(h^{{{4-alpha_v:.1f}}})$')
        ax2.set_xlabel(r'$h$'); ax2.set_ylabel(r'$\|e\|_\infty$')
        ax2.set_title('Convergence', fontweight='bold', loc='left')
        ax2.legend(); ax2.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'fig5_viscoelastic.pdf'))
        fig.savefig(os.path.join(save_dir, 'fig5_viscoelastic.png'))
        plt.close(fig)

    # --- Figure 6: Method comparison ---
    if trap:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
        for idx, alpha in enumerate([0.2, 0.5, 0.8]):
            ax = axes[idx]
            cr = smooth[alpha]
            tr = trap[alpha]
            ax.loglog(cr['h'], cr['errors'], 'o-', color=C_UNIFORM, lw=1.3, ms=5,
                      mfc='none', label=f'Compact ({cr["rate"]:.2f})')
            ax.loglog(tr['h'], tr['errors'], 's--', color=C_BE, lw=1.3, ms=5,
                      mfc='none', label=f'Prod. trap. ({tr["rate"]:.2f})')
            ax.set_xlabel(r'$h$'); ax.set_ylabel(r'$\|e\|_\infty$')
            ax.set_title(rf'$\alpha={alpha}$', fontweight='bold', loc='left')
            ax.legend(fontsize=7); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'fig6_comparison.pdf'))
        fig.savefig(os.path.join(save_dir, 'fig6_comparison.png'))
        plt.close(fig)

    print("  All extended figures saved.")


def print_latex_table_singular(results_sing1, results_graded):
    """LaTeX table: uniform vs graded comparison for singular test."""
    print("\n" + "=" * 80)
    print("Table: Uniform vs Graded Mesh — u(x) = x^{1-alpha} + x^2")
    print("=" * 80)
    alphas = sorted(results_sing1.keys())
    N_vals = results_sing1[alphas[0]]['N']

    header = (r"N & h & " + " & ".join([
        rf"$\alpha={a}$ (unif) & $\alpha={a}$ (grad)" for a in alphas
    ]) + r" \\")
    print(header)
    print(r"\midrule")

    for idx, N in enumerate(N_vals):
        h = results_sing1[alphas[0]]['h'][idx]
        parts = [str(N), f"{h:.4f}"]
        for a in alphas:
            eu = results_sing1[a]['errors'][idx]
            eg = results_graded[a]['errors'][idx]
            parts.append(f"{eu:.3e} & {eg:.3e}")
        print(" & ".join(parts) + r" \\")

    # Rate row
    parts = [r"\multicolumn{2}{c}{Rate}"]
    for a in alphas:
        ru = results_sing1[a]['rate']
        rg = results_graded[a]['rate']
        parts.append(f"{ru:.2f} & {rg:.2f}")
    print(" & ".join(parts) + r" \\")
    print()


# ============================================================
# Main (extended)
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Extended Numerical Study for VIDE Paper Revision")
    print("=" * 70)

    t0 = time.time()

    # Test 1: Smooth (keep existing baseline)
    print("\n" + "=" * 70)
    print("Test 1: Smooth solution u(x) = x^2 (baseline)")
    print("=" * 70)
    smooth_results, _ = run_convergence_legacy(method='compact')

    print("\n" + "=" * 70)
    print("Comparison: Product Trapezoidal (baseline)")
    print("=" * 70)
    trap_results, _ = run_convergence_legacy(method='trapezoidal')

    # Test 2: Singular — uniform mesh (the key new result)
    print("\n" + "=" * 70)
    print("Test 2: Singular solution u(x) = x^{1-alpha} + x^2 — UNIFORM mesh")
    print("=" * 70)
    singular1_results = run_convergence_singular()

    # Test 2b: Singular — graded mesh (backward Euler, baseline)
    print("\n" + "=" * 70)
    print("Test 2b: Singular solution — GRADED mesh (backward Euler)")
    print("=" * 70)
    graded_results = run_convergence_graded()

    # Test 2c: Singular — compact on graded mesh (NEW — coordinate transform)
    print("\n" + "=" * 70)
    print("Test 2c: Singular solution — COMPACT on graded mesh (coordinate transform)")
    print("=" * 70)
    compact_graded_results = run_convergence_compact_graded()

    # Test 3: Two singular terms
    print("\n" + "=" * 70)
    print("Test 3: u(x) = x^{1-alpha} + x^{2-alpha} + x^3 — uniform vs graded")
    print("=" * 70)
    singular2_results = run_convergence_singular2()

    # Test 4: sin(pi*x) (keep)
    print("\n" + "=" * 70)
    print("Test 4: Oscillatory solution u(x) = sin(pi*x)")
    print("=" * 70)
    sine_result = run_sine_test_legacy()

    # Test 5: Viscoelastic application
    print("\n" + "=" * 70)
    print("Test 5: Viscoelasticity-motivated problem")
    print("=" * 70)
    visco_result = run_viscoelastic_test()

    # Package results
    all_results = {
        'smooth': smooth_results,
        'singular1': singular1_results,
        'graded': graded_results,
        'compact_graded': compact_graded_results,
        'singular2': singular2_results,
        'sine': sine_result,
        'viscoelastic': visco_result,
        'trapezoidal': trap_results,
    }

    # Generate extended figures
    print("\n" + "=" * 70)
    print("Generating Extended Figures")
    print("=" * 70)
    make_extended_figures(all_results)

    # Print LaTeX tables
    print("\n" + "=" * 70)
    print("LaTeX Tables")
    print("=" * 70)
    print_latex_table_singular(singular1_results, graded_results)
    print_latex_table_legacy(smooth_results, "Compact Method — Smooth Solution")

    t1 = time.time()
    print(f"\nTotal time: {t1 - t0:.1f} seconds\nDone.")
