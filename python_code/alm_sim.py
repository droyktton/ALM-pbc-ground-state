import numpy as np
from scipy.optimize import brentq
import argparse
import sys

def generate_sample(L, Delta, n, c, seed):
    """
    Simulates the ground state of the Anharmonic Linear Model (ALM).
    """
    np.random.seed(seed)
    
    # -----------------------------
    # Disorder
    # -----------------------------
    # Generate Gaussian random forces
    f = np.random.normal(0.0, np.sqrt(Delta), size=L)
    # Ensure force balance (sum f_i = 0) for PBC stability
    f -= f.mean()

    # Cumulative disorder force (integrated stress)
    # Because mean(f) is 0, F is naturally periodic (F[0] == F[L])
    F = np.concatenate([[0.0], np.cumsum(f[:-1])])

    # -----------------------------
    # Constitutive inversion
    # -----------------------------
    def slope_from_stress(sigma):
        """
        Solve the constitutive law: c*s + sign(s)*|s|^{2n-1} = sigma
        This inversion determines the strain (slope) from the stress.
        """
        if c == 0:
            # Closed-form solution for the pure power-law case
            return np.sign(sigma) * abs(sigma)**(1.0/(2*n-1))

        def eq(s):
            return c*s + np.sign(s)*abs(s)**(2*n-1) - sigma

        # Bracket safely for robust root-finding
        smax = max(1.0, abs(sigma)**(1.0/(2*n-1)) + 1.0)
        return brentq(eq, -smax, smax)

    # Vectorized wrapper for performance on NumPy arrays
    vec_slope = np.vectorize(slope_from_stress)

    # -----------------------------
    # Constraint: zero total slope
    # -----------------------------
    def total_slope(C):
        """
        Helper to find the constant shift C that ensures PBC (sum of slopes = 0).
        """
        sigma = F + C
        s = vec_slope(sigma)
        return np.sum(s)

    # find C using the robust Brent's method [cite: 197, 201]
    # The range [-100, 100] is usually sufficient; 
    # for extreme L, you might scale this with sqrt(L)
    C = brentq(total_slope, -1000.0, 1000.0)

    # -----------------------------
    # Build solution
    # -----------------------------
    sigma = F + C
    s = vec_slope(sigma)

    # Integrate slopes to get displacements u
    u = np.zeros(L)
    u[1:] = np.cumsum(s[:-1])

    # Remove arbitrary global offset (center of mass)
    u -= u.mean()

    # -----------------------------
    # Diagnostics
    # -----------------------------
    print(f"--- Results for L={L}, n={n}, seed={seed} ---")
    print("Check periodicity (sum s):", np.sum(s))
    print("Mean displacement:", u.mean())

    return u

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Run ALM simulation for ground state.")
    parser.add_argument("-L", type=int, default=1024, help="System size (number of sites)")
    parser.add_argument("-n", type=float, default=2.0, help="Anharmonicity exponent")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--delta", type=float, default=1.0, help="Disorder strength (Delta)")
    parser.add_argument("--c", type=float, default=0.0, help="Linear coefficient (harmonic part)")
    parser.add_argument("--output", type=str, default="u_py.txt", help="Output filename")

    args = parser.parse_args()

    # Generate the displacement field
    u = generate_sample(L=args.L, Delta=args.delta, n=args.n, c=args.c, seed=args.seed)

    # Save to file with high precision [cite: 159]
    np.savetxt(args.output, u, fmt='%.18e')
    print(f"Displacements saved to {args.output}")

#python alm_sim.py -L 10000 -n 2.5 --seed 42
