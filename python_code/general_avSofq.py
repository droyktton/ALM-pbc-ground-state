#usage:
#python general_avSofq.py --L 1024 --ANH 2

import numpy as np
import glob

def compute_S_avg(L=None, ANH=None, base_pattern="L{L}_ANH{ANH}_seed*/u.txt"):
    pattern = base_pattern.format(L=L, ANH=ANH)
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched pattern: {pattern}")

    S_accum = None
    count = 0

    for f in files:
        u = np.loadtxt(f)
        Lf = u.size  # infer L from file

        u_k = np.fft.rfft(u)
        S_k = (np.abs(u_k)**2) / Lf  # normalize by L

        if S_accum is None:
            S_accum = np.zeros_like(S_k, dtype=np.float64)

        # safety check: ensure consistent lengths
        if S_accum.shape != S_k.shape:
            raise ValueError(f"Inconsistent L across files: {f}")

        S_accum += S_k
        count += 1

    S_avg = S_accum / count
    k = 2*np.pi*np.fft.rfftfreq(Lf)

    return k, S_avg, count

# example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--ANH", type=int, required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    k, S_avg, n = compute_S_avg(L=args.L, ANH=args.ANH)

    out = args.out or f"S_avg_L={args.L}_ANH={args.ANH}.txt"
    np.savetxt(out, np.column_stack([k, S_avg]))
    print(f"Averaged over {n} files -> {out}")
