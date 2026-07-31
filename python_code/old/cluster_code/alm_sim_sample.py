import numpy as np
from scipy.optimize import brentq
import argparse, os

def generate_sample(L, Delta, n, c, seed):
    np.random.seed(seed)
    f = np.random.normal(0.0, np.sqrt(Delta), size=L)
    f -= f.mean()
    F = np.concatenate([[0.0], np.cumsum(f[:-1])])

    def slope_from_stress(sigma):
        if c == 0:
            return np.sign(sigma) * abs(sigma)**(1.0/(2*n-1))
        def eq(s):
            return c*s + np.sign(s)*abs(s)**(2*n-1) - sigma
        smax = max(1.0, abs(sigma)**(1.0/(2*n-1)) + 1.0)
        return brentq(eq, -smax, smax)

    vec_slope = np.vectorize(slope_from_stress)

    def total_slope(C):
        return np.sum(vec_slope(F + C))

    C = brentq(total_slope, -1000.0, 1000.0)
    s = vec_slope(F + C)
    u = np.zeros(L)
    u[1:] = np.cumsum(s[:-1])
    u -= u.mean()
    return u, np.sum(s)


parser = argparse.ArgumentParser()
parser.add_argument('--seed_start', type=int, required=True)
parser.add_argument('--seed_end',   type=int, required=True)
parser.add_argument('--L',          type=int, default=8192)
parser.add_argument('--Delta',      type=float, default=1.0)
parser.add_argument('--c',          type=float, default=0.0)
parser.add_argument('--n_min',      type=int, default=2)
parser.add_argument('--n_max',      type=int, default=50)
parser.add_argument('--outdir',     type=str, default='results')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

q    = np.fft.fftfreq(args.L, d=1)
qpos = q[q > 0]

n_values = list(range(args.n_min, args.n_max + 1))

# sq_sum[i]     : sum of S(q) over valid seeds, for n_values[i]
# sq_per[i]     : list of per-seed S(q) arrays (for per-sample zeta fitting)
# valid_count[i]: number of valid seeds
sq_sum      = {n: np.zeros(len(qpos)) for n in n_values}
sq_per      = {n: [] for n in n_values}
valid_count = {n: 0  for n in n_values}

for n in n_values:
    print(f"n={n}, seeds {args.seed_start}–{args.seed_end-1}")
    for seed in range(args.seed_start, args.seed_end):
        u, periodicity = generate_sample(args.L, args.Delta, n, args.c, seed)
        if abs(periodicity) < 1e-7:
            sq = np.abs(np.fft.fft(u))**2
            sq_per_seed = sq[q > 0]
            sq_sum[n]  += sq_per_seed
            sq_per[n].append(sq_per_seed)
            valid_count[n] += 1

# Save partial results
outfile = os.path.join(args.outdir,
    f'partial_{args.seed_start:06d}_{args.seed_end:06d}.npz')
np.savez(outfile,
         n_values    = np.array(n_values),
         qpos        = qpos,
         sq_sum      = np.array([sq_sum[n]            for n in n_values]),
         sq_per      = np.array([sq_per[n]             for n in n_values], dtype=object),
         valid_count = np.array([valid_count[n]        for n in n_values]))

print(f"Saved {outfile}")
