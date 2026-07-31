"""
Steady-state samples of the quenched KPZ equation with periodic BC

    d_t h = d_x^2 h + lambda * (d_x h)^2 + eta(x)

Method (Cole-Hopf):
    Z(x,t) = exp(lambda * h(x,t))  satisfies the LINEAR equation
        d_t Z = Z'' + lambda * eta(x) Z  =  L Z,   L = d_x^2 + lambda*eta(x)

    As t -> infinity, Z is dominated by the top eigenmode of L:
        Z(x,t) ~ phi_0(x) * exp(E_0 t)
    With periodic BC, L (discretized) is an irreducible symmetric matrix with
    nonnegative off-diagonal entries, so by Perron-Frobenius its top
    eigenvector phi_0 can be chosen strictly positive everywhere. Hence

        h_steady(x) = (1/lambda) * ln(phi_0(x))   (+ arbitrary additive const)

    is a well-defined, single-valued, periodic steady-state height profile,
    growing in time as h(x,t) ~ h_steady(x) + (E_0/lambda) * t.

Quenched noise: eta(x) is white noise with <eta(x)eta(x')> = 2D delta(x-x'),
discretized on N points with spacing dx=L/N as iid Gaussians of variance 2D/dx.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def sample_kpz_steady_state(L, lam, D, N=2000, seed=None, return_noise=False):
    """
    Generate one steady-state sample h(x) of quenched KPZ on a periodic
    domain of length L.

    Parameters
    ----------
    L : float          domain length
    lam : float         nonlinearity strength (lambda), lam != 0
    D : float           noise strength, <eta(x)eta(x')> = 2 D delta(x-x')
    N : int              number of grid points (periodic, x_N == x_0)
    seed : int or None   RNG seed
    return_noise : bool  also return the eta(x) realization used

    Returns
    -------
    x : (N,) array        grid points in [0, L)
    h : (N,) array         steady-state height, periodic, mean subtracted
    v : float               asymptotic growth velocity dh/dt = E0/lam
    eta : (N,) array        (only if return_noise=True) the noise realization
    """
    if lam == 0:
        raise ValueError("lambda must be nonzero (equation reduces to EW otherwise).")

    rng = np.random.default_rng(seed)
    dx = L / N

    # discretized white noise: var(eta_i) = 2D/dx so that sum dx*eta*eta -> 2D*delta
    eta = rng.normal(loc=0.0, scale=np.sqrt(2.0 * D / dx), size=N)

    # periodic discrete Laplacian: (f_{i+1} - 2 f_i + f_{i-1})/dx^2
    main = -2.0 * np.ones(N) / dx**2 + lam * eta
    off = np.ones(N - 1) / dx**2
    Lap = diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    Lap[0, N - 1] = 1.0 / dx**2   # periodic wrap-around
    Lap[N - 1, 0] = 1.0 / dx**2
    Lop = Lap.tocsr()

    # Top eigenpair (largest algebraic eigenvalue) of the symmetric operator L.
    # Plain which="LA" Lanczos converges very slowly here because the operator's
    # spectrum spans a huge range (~ -4/dx^2 to E0 ~ O(1)), so instead use
    # shift-invert mode: by Gershgorin, E0 <= max(lam*eta), so shifting just
    # above that bound and asking for the eigenvalue of largest magnitude of
    # (L - sigma*I)^-1 converges in a handful of iterations (banded LU factorize).
    sigma = np.max(lam * eta) + max(1.0, abs(lam) * np.std(eta))
    E0, phi = eigsh(Lop, k=1, sigma=sigma, which="LM")
    E0 = E0[0]
    phi = phi[:, 0]

    # Perron-Frobenius guarantees the top eigenvector is sign-definite (all
    # entries same sign). Fix its overall sign to be positive. For strongly
    # localized/disordered cases phi can decay far below float64 precision
    # away from its peak, so a few entries may flip sign from roundoff alone;
    # taking the absolute value and flooring at the smallest representable
    # positive double keeps log(phi) finite without materially changing h
    # (those points are numerically indistinguishable from phi=0 anyway).
    phi *= np.sign(phi[np.argmax(np.abs(phi))])
    phi = np.abs(phi)
    phi = np.maximum(phi, np.finfo(float).tiny)

    h = np.log(phi) / lam
    h -= h.mean()  # fix the arbitrary additive constant (eigenvector normalization)

    x = np.arange(N) * dx
    v = E0 / lam

    if return_noise:
        return x, h, v, eta
    return x, h, v


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L, lam, D = 50.0, 1.0, 0.05
    N = 4000

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed in range(5):
        x, h, v = sample_kpz_steady_state(L, lam, D, N=N, seed=seed)
        ax.plot(x, h, lw=1, label=f"sample {seed} (v={v:.3f})")

    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_title(f"Quenched KPZ periodic steady states  (L={L}, lambda={lam}, D={D})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("kpz_steady_state_samples.png", dpi=150)
    print("Saved kpz_steady_state_samples.png")

    # quick sanity check: roughness ~ how h fluctuates about linear trend
    x, h, v = sample_kpz_steady_state(L, lam, D, N=N, seed=0)
    print(f"velocity v = E0/lambda = {v:.4f}")
    print(f"std(h)     = {h.std():.4f}")
