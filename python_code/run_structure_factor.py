"""
Command-line tool: disorder-averaged structure factor for the ALM
====================================================================

Computes, from the same set of disorder samples:
  - S(q)     : disorder-averaged structure factor, and the fitted
               spectral roughness exponent zeta_s from S(q) ~ q^{-(1+2*zeta_s)}
  - W^2(L)   : interface width squared, W^2 = < (1/L) sum_i (u_i - mean(u))^2 >,
               computed two independent ways as a consistency check:
                 1. directly in real space from each sample's u_i
                 2. via Parseval's theorem, summing over the same S(q) used above

Usage
-----
    python3 run_structure_factor.py -L 4096 -n 2.0
    python3 run_structure_factor.py -L 8192 -n 3.0 -c 1.0 --delta 1.0 --samples 300
    python3 run_structure_factor.py -L 4096 -n 2.0 --qfrac 0.1 -o my_plot.png --no-show

Requires alm.py in the same folder (it provides solve_ground_state).
"""

import argparse
import sys
import time

import numpy as np
from scipy.stats import linregress

from alm import solve_ground_state


def structure_factor_average(L, c, n, Delta, n_samples, rng):
    """
    Returns
    -------
    q      : positive wavevectors, shape (L//2,)
    S      : disorder-averaged structure factor S(q) = < |u_hat(q)|^2 > / L
    W2_avg : disorder-averaged interface width squared,
             W2 = < (1/L) * sum_i (u_i - mean(u))^2 >
             computed directly in real space from the *same* samples used for S(q).
    """
    S_accum = np.zeros(L // 2)
    W2_accum = 0.0

    for _ in range(n_samples):
        f = rng.normal(0.0, np.sqrt(Delta), size=L)
        u, s, C, F = solve_ground_state(f, c, n)   # u already has zero mean (step 7)

        u_hat = np.fft.rfft(u)
        power = (np.abs(u_hat) ** 2) / L
        S_accum += power[1:L // 2 + 1]   # drop k=0 mode (mean, exactly zero)

        W2_accum += np.mean(u ** 2)      # real-space width^2 for this sample

    S_avg = S_accum / n_samples
    W2_avg = W2_accum / n_samples
    k = np.arange(1, L // 2 + 1)
    q = 2 * np.pi * k / L
    return q, S_avg, W2_avg


def w2_from_S(q, S, L):
    """
    Parseval-theorem cross-check: reconstruct W^2 from the (one-sided)
    structure factor alone, W^2 = (1/L) * sum_{k != 0} S(q_k),
    with the Nyquist mode (k = L/2, present only for even L) counted once
    and all other k counted twice (to account for the mirrored negative
    frequencies of a real signal).
    """
    is_even = (L % 2 == 0)
    if is_even:
        S_no_nyquist = S[:-1]
        S_nyquist = S[-1]
        total = 2.0 * np.sum(S_no_nyquist) + S_nyquist
    else:
        total = 2.0 * np.sum(S)
    return total / L


def fit_zeta_s(q, S, q_frac_max=0.15):
    """
    Fit S(q) ~ q^{-(1+2*zeta_s)} over the small-q tail via log-log
    linear regression. Returns zeta_s, its standard error, and the
    full regression result.
    """
    n_fit = max(4, int(len(q) * q_frac_max))
    log_q = np.log(q[:n_fit])
    log_S = np.log(S[:n_fit])
    res = linregress(log_q, log_S)
    zeta_s = -(res.slope + 1) / 2
    zeta_s_err = res.stderr / 2
    return zeta_s, zeta_s_err, res


def main():
    parser = argparse.ArgumentParser(
        description="Disorder-averaged structure factor S(q) and spectral "
                    "roughness exponent zeta_s for the Anharmonic Larkin Model."
    )
    parser.add_argument("-L", type=int, required=True,
                        help="System size (number of sites). Should be even.")
    parser.add_argument("-n", type=float, required=True,
                        help="Anharmonicity exponent n (>1). n=2 -> quartic term |s|^4/4.")
    parser.add_argument("-c", type=float, default=1.0,
                        help="Harmonic elastic constant c >= 0 (default: 1.0).")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Disorder variance Delta (default: 1.0).")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of disorder realizations to average over (default: 200).")
    parser.add_argument("--qfrac", type=float, default=0.15,
                        help="Fraction of low-q modes used for the power-law fit (default: 0.15).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output PNG path (default: sf_L{L}_n{n}.png in current directory).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting entirely (just print the fit result).")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot to file without opening a display window.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to also dump the raw (q, S(q)) data as CSV.")
    args = parser.parse_args()

    if args.L % 2 != 0:
        print("Warning: L is odd; even L is recommended for a clean rfft spectrum.",
              file=sys.stderr)

    out_path = args.output or f"sf_L{args.L}_n{args.n}.png"

    rng = np.random.default_rng(args.seed)

    print(f"Averaging structure factor over {args.samples} samples "
          f"(L={args.L}, c={args.c}, n={args.n}, Delta={args.delta}) ...")
    t0 = time.time()
    q, S, W2_direct = structure_factor_average(args.L, args.c, args.n, args.delta, args.samples, rng)
    print(f"done in {time.time() - t0:.2f} s")

    zeta_s, zeta_s_err, res = fit_zeta_s(q, S, q_frac_max=args.qfrac)
    print(f"fitted small-q slope   = {res.slope:.4f} +/- {res.stderr:.4f}")
    print(f"spectral roughness zeta_s = {zeta_s:.4f} +/- {zeta_s_err:.4f}")
    print(f"R^2 of log-log fit     = {res.rvalue**2:.5f}")

    W2_parseval = w2_from_S(q, S, args.L)
    rel_diff = abs(W2_direct - W2_parseval) / W2_direct
    print(f"W^2(L={args.L}) [real space, direct]  = {W2_direct:.6f}")
    print(f"W^2(L={args.L}) [from S(q), Parseval]  = {W2_parseval:.6f}")
    print(f"relative difference (consistency check) = {rel_diff:.2e}")

    # Single machine-readable line for easy parsing in shell scripts / pipelines.
    # Format: key=value pairs, space-separated, no spaces within a value.
    print(f"RESULT L={args.L} n={args.n} c={args.c} delta={args.delta} "
          f"samples={args.samples} zeta_s={zeta_s:.6f} zeta_s_err={zeta_s_err:.6f} "
          f"W2_direct={W2_direct:.6f} W2_parseval={W2_parseval:.6f}")

    if args.csv:
        np.savetxt(args.csv, np.column_stack([q, S]), delimiter=",",
                   header="q,S(q)", comments="")
        print(f"Saved raw data to {args.csv}")

    if not args.no_plot:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.loglog(q, S, '.', ms=3, alpha=0.5, color="#1f77b4", label="data (disorder avg.)")

        n_fit = max(4, int(len(q) * args.qfrac))
        q_fit = q[:n_fit]
        S_fit_line = np.exp(res.intercept) * q_fit ** res.slope
        ax.loglog(q_fit, S_fit_line, 'r-', lw=2,
                  label=fr"fit: $\zeta_s={zeta_s:.3f}\pm{zeta_s_err:.3f}$")

        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$S(q)$")
        ax.set_title(f"ALM structure factor  (L={args.L}, c={args.c}, n={args.n}, "
                    f"$\\Delta$={args.delta}, {args.samples} samples)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        plt.tight_layout()

        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
