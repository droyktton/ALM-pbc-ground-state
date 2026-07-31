"""
Disorder-averaged structure factor of the quenched KPZ steady state.

For each disorder realization we generate h(x) on the periodic ring [0,L)
using sample_kpz_steady_state (Cole-Hopf + top eigenfunction, see
kpz_quenched_steady_state.py), then compute the (discrete) Fourier transform

    h(q) = dx * sum_x h(x) exp(-i q x)

and the structure factor for that sample

    S_sample(q) = |h(q)|^2 / L

Averaging S_sample(q) over many independent disorder realizations gives the
disorder-averaged structure factor S(q) = <|h(q)|^2> / L.
"""

import numpy as np
from kpz_quenched_steady_state import sample_kpz_steady_state


def structure_factor_samples(L, lam, D, N=4000, n_samples=200, seed0=0,
                              progress=False):
    """
    Generate n_samples independent steady-state samples and accumulate
    their structure factor.

    Parameters
    ----------
    L, lam, D, N : as in sample_kpz_steady_state
    n_samples : number of independent disorder realizations to average over
    seed0 : base seed; realization i uses seed = seed0 + i
    progress : print progress every 10% of samples

    Returns
    -------
    q : (N,) array of wavenumbers, q = 2*pi*n/L,  n = -N/2..N/2-1 (fftshift order)
    S_mean : (N,) array, disorder-averaged structure factor S(q)
    S_sem : (N,) array, standard error of the mean across samples
    """
    dx = L / N
    q = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    S_accum = np.zeros(N)
    S_sq_accum = np.zeros(N)  # for computing variance -> SEM

    for i in range(n_samples):
        x, h, v = sample_kpz_steady_state(L, lam, D, N=N, seed=seed0 + i)
        # h is already exactly periodic by construction (it's log of a
        # periodic eigenvector), so no detrending is needed before the FFT.

        hq = np.fft.fftshift(np.fft.fft(h)) * dx
        S = np.abs(hq) ** 2 / L

        S_accum += S
        S_sq_accum += S ** 2

        if progress and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  {i + 1}/{n_samples} samples done")

    S_mean = S_accum / n_samples
    S_var = S_sq_accum / n_samples - S_mean ** 2
    S_sem = np.sqrt(np.maximum(S_var, 0.0) / n_samples)

    return q, S_mean, S_sem


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L, lam, D = 50.0, 1.0, 0.05
    N = 2000
    n_samples = 300

    print(f"Generating {n_samples} samples (L={L}, lambda={lam}, D={D}, N={N})...")
    q, S_mean, S_sem = structure_factor_samples(
        L, lam, D, N=N, n_samples=n_samples, seed0=0, progress=True
    )

    # keep only positive q (S(q) is symmetric for real h(x)), drop q=0
    mask = q > 0
    q_pos, S_pos, S_sem_pos = q[mask], S_mean[mask], S_sem[mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(q_pos, S_pos, yerr=S_sem_pos, fmt='o', ms=3, lw=0.7,
                capsize=1.5, label="measured S(q)")

    # mid-range power-law guide to the eye (slope -2), fit away from the
    # smallest q (system-size mode, see note below) and largest q (grid cutoff)
    lo, hi = int(0.05 * len(q_pos)), int(0.4 * len(q_pos))
    ref = S_pos[lo] * (q_pos[lo] / q_pos[lo:hi]) ** 2
    ax.plot(q_pos[lo:hi], ref, 'k--', lw=1, label=r"slope $-2$ guide")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("S(q)")
    ax.set_title(f"Disorder-averaged structure factor "
                 f"(L={L}, $\\lambda$={lam}, D={D}, {n_samples} samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("kpz_structure_factor.png", dpi=150)
    print("Saved kpz_structure_factor.png")

    print(f"\nNote: S(q) at the smallest wavenumber (q = 2*pi/L) is typically "
          f"anomalously large relative to a naive power law. This is a real "
          f"effect, not noise: h(x) = ln(phi_0(x))/lambda is (the log of) a "
          f"bound state of a random Schrodinger operator, and its "
          f"localization length is often comparable to L, so a single broad "
          f"hump dominates the box-scale (n=1) Fourier mode. Increase L "
          f"(at fixed D, lambda) to push this into the true asymptotic regime.")
