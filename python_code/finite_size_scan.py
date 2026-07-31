"""
Finite-size scan for the ALM: check convergence of zeta_s across L
=======================================================================

Diagnostic for whether a given q-fit-window is contaminated by finite-size
effects: run the SAME absolute q-window [qmin, qmax] across several system
sizes L, and see whether the fitted zeta_s(L) stabilizes as L grows.

- If zeta_s(L) is flat within error bars across the scanned L's, the window
  is a safe choice for extracting the asymptotic exponent.
- If zeta_s(L) drifts systematically with L, the window still contains
  finite-size (or crossover) contamination -- typically fixable by pushing
  qmin further from 2*pi/L_min (i.e. away from the IR edge), or by using
  larger L altogether.

Usage
-----
    python3 finite_size_scan.py -n 2.0 --qmin 0.01 --qmax 0.1 \\
        --Ls 1024 2048 4096 8192 --samples 200

    python3 finite_size_scan.py -n 2.0 --qmin 0.01 --qmax 0.1 \\
        --Ls 1024 2048 4096 8192 --samples 200 --no-show

Requires alm.py and run_structure_factor.py in the same folder.
"""

import argparse
import time

import numpy as np

from run_structure_factor import structure_factor_average, fit_zeta_s


def main():
    parser = argparse.ArgumentParser(
        description="Scan zeta_s(L) at a FIXED absolute q-window across "
                    "several system sizes, to diagnose finite-size effects."
    )
    parser.add_argument("-n", type=float, required=True,
                        help="Anharmonicity exponent n (>1).")
    parser.add_argument("-c", type=float, default=1.0,
                        help="Harmonic elastic constant c >= 0 (default: 1.0).")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Disorder variance Delta (default: 1.0).")
    parser.add_argument("--Ls", type=int, nargs="+", required=True,
                        help="List of system sizes to scan, e.g. --Ls 1024 2048 4096 8192.")
    parser.add_argument("--qmin", type=float, required=True,
                        help="Absolute lower bound of the fit window (fixed across all L).")
    parser.add_argument("--qmax", type=float, required=True,
                        help="Absolute upper bound of the fit window (fixed across all L).")
    parser.add_argument("--samples", type=int, default=200,
                        help="Disorder realizations per L (default: 200).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("-o", "--output", type=str, default="finite_size_scan.png",
                        help="Output plot path.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to dump per-L results as CSV.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without opening a display window.")
    args = parser.parse_args()

    q_nyquist_needed = args.qmax
    for L in args.Ls:
        if 2 * np.pi / L > args.qmin:
            print(f"Warning: L={L} has q_min_available={2*np.pi/L:.4g} > "
                  f"requested qmin={args.qmin:.4g}; the fit window will "
                  f"effectively start at q_min_available for this L.")
        if np.pi > q_nyquist_needed:
            pass  # fine, qmax well within Nyquist range for any reasonable L

    results = []
    for L in args.Ls:
        rng = np.random.default_rng(args.seed)
        print(f"\n--- L = {L} ---")
        t0 = time.time()
        q, S, W2_direct = structure_factor_average(L, args.c, args.n, args.delta,
                                                     args.samples, rng)
        try:
            zeta_s, zeta_s_err, res, (idx_min, idx_max) = fit_zeta_s(
                q, S, q_min=args.qmin, q_max=args.qmax
            )
        except ValueError as e:
            print(f"  skipped: {e}")
            continue
        dt = time.time() - t0
        n_modes = idx_max - idx_min
        print(f"  {n_modes} modes in [{q[idx_min]:.4g}, {q[idx_max-1]:.4g}], "
              f"zeta_s = {zeta_s:.4f} +/- {zeta_s_err:.4f}  "
              f"(R^2={res.rvalue**2:.4f}, {dt:.2f}s)")
        results.append(dict(L=L, zeta_s=zeta_s, zeta_s_err=zeta_s_err,
                            n_modes=n_modes, R2=res.rvalue**2, W2=W2_direct))

    if not results:
        print("No valid results (fit window too narrow for all requested L). "
              "Try smaller qmin, larger qmax, or larger L.")
        return

    Ls = np.array([r["L"] for r in results])
    zs = np.array([r["zeta_s"] for r in results])
    zs_err = np.array([r["zeta_s_err"] for r in results])

    print("\n=== Summary ===")
    print(f"{'L':>8} {'zeta_s':>10} {'+/- err':>10} {'n_modes':>8} {'R^2':>8}")
    for r in results:
        print(f"{r['L']:>8} {r['zeta_s']:>10.4f} {r['zeta_s_err']:>10.4f} "
              f"{r['n_modes']:>8} {r['R2']:>8.4f}")

    if len(results) >= 2:
        drift = zs[-1] - zs[0]
        pooled_err = np.sqrt(zs_err[-1]**2 + zs_err[0]**2)
        print(f"\nDrift from L={Ls[0]} to L={Ls[-1]}: "
              f"delta(zeta_s) = {drift:+.4f}  ({abs(drift)/pooled_err:.2f} sigma)")
        if abs(drift) < 2 * pooled_err:
            print("-> zeta_s appears STABLE across this L range within ~2 sigma: "
                  "window looks safe from finite-size contamination.")
        else:
            print("-> zeta_s DRIFTS beyond ~2 sigma across this L range: "
                  "window likely still contaminated (try larger qmin, smaller "
                  "qmax, or bigger L).")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("L,zeta_s,zeta_s_err,n_modes,R2,W2\n")
            for r in results:
                fh.write(f"{r['L']},{r['zeta_s']:.6f},{r['zeta_s_err']:.6f},"
                        f"{r['n_modes']},{r['R2']:.6f},{r['W2']:.6f}\n")
        print(f"Saved results to {args.csv}")

    # ------------------------------------------------------------------
    # Plot zeta_s(L)
    # ------------------------------------------------------------------
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(Ls, zs, yerr=zs_err, fmt='o-', capsize=3, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\zeta_s$")
    ax.set_title(fr"Finite-size scan  (n={args.n}, c={args.c}, $\Delta$={args.delta}, "
                fr"q$\in$[{args.qmin:.3g}, {args.qmax:.3g}])")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
