"""
Disorder-averaged structure factor for the Anharmonic Larkin Model (ALM)
==========================================================================

For each disorder sample we build the exact ground state u_i (via alm.py),
Fourier transform it, and accumulate the power spectrum

    S(q) = < |u_hat(q)|^2 > / L

averaged over many independent realizations of the quenched force f_i.
q_k = 2*pi*k/L,  k = 1, ..., L-1  (k=0 mode is fixed to 0 by construction).

We then fit the small-q tail to

    S(q) ~ 1 / q^{1 + 2*zeta_s}

by linear regression in log-log space, to extract the spectral roughness
exponent zeta_s.
"""

import numpy as np
from scipy.stats import linregress

from alm import solve_ground_state


def structure_factor_average(L, c, n, Delta, n_samples, rng):
    """
    Returns
    -------
    q : array of positive wavevectors, shape (L//2,)
    S : disorder-averaged structure factor at those wavevectors
    """
    S_accum = np.zeros(L // 2)

    for _ in range(n_samples):
        f = rng.normal(0.0, np.sqrt(Delta), size=L)
        u, s, C, F = solve_ground_state(f, c, n)

        u_hat = np.fft.rfft(u)          # k = 0, ..., L//2
        power = (np.abs(u_hat) ** 2) / L

        # drop k=0 (mean mode, exactly zero by construction);
        # drop Nyquist if L even to keep a clean one-sided spectrum
        S_accum += power[1:L // 2 + 1]

    S_avg = S_accum / n_samples
    k = np.arange(1, L // 2 + 1)
    q = 2 * np.pi * k / L
    return q, S_avg


def fit_zeta_s(q, S, q_frac_max=0.15):
    """
    Fit S(q) ~ q^{-(1+2*zeta_s)} over the small-q tail
    (the lowest `q_frac_max` fraction of available wavevectors),
    via linear regression of log S vs log q.

    Returns zeta_s, its standard error, and the (slope, intercept) fit.
    """
    n_fit = max(4, int(len(q) * q_frac_max))
    log_q = np.log(q[:n_fit])
    log_S = np.log(S[:n_fit])

    res = linregress(log_q, log_S)
    slope = res.slope           # slope = -(1 + 2*zeta_s)
    zeta_s = -(slope + 1) / 2
    zeta_s_err = res.stderr / 2
    return zeta_s, zeta_s_err, res


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    L = 4096
    Delta = 1.0
    c = 0.0
    n = 2.0
    n_samples = 200

    print(f"Averaging structure factor over {n_samples} samples "
          f"(L={L}, c={c}, n={n}, Delta={Delta}) ...")
    q, S = structure_factor_average(L, c, n, Delta, n_samples, rng)

    zeta_s, zeta_s_err, res = fit_zeta_s(q, S, q_frac_max=0.15)
    print(f"fitted small-q slope           = {res.slope:.4f} +/- {res.stderr:.4f}")
    print(f"=> spectral roughness zeta_s   = {zeta_s:.4f} +/- {zeta_s_err:.4f}")
    print(f"   (R^2 of log-log fit: {res.rvalue**2:.5f})")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.loglog(q, S, '.', ms=3, alpha=0.5, color="#1f77b4", label="data (disorder avg.)")

    n_fit = max(4, int(len(q) * 0.15))
    q_fit = q[:n_fit]
    S_fit_line = np.exp(res.intercept) * q_fit ** res.slope
    ax.loglog(q_fit, S_fit_line, 'r-', lw=2,
               label=fr"fit: $S(q)\sim q^{{-(1+2\zeta_s)}}$, $\zeta_s={zeta_s:.3f}\pm{zeta_s_err:.3f}$")

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$S(q)$")
    ax.set_title(f"ALM structure factor  (L={L}, c={c}, n={n}, "
                 f"$\\Delta$={Delta}, {n_samples} samples)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig("alm_structure_factor.png", dpi=150)
    print("Saved plot to alm_structure_factor.png")
