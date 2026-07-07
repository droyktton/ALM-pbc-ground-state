#!/usr/bin/env python3

import glob
import struct
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --------------------------------------------------
# GPU if available
# --------------------------------------------------

try:
    import cupy as cp
    xp = cp
    gpu = True
    print("Using CuPy (GPU)")
except ImportError:
    xp = np
    gpu = False
    print("Using NumPy (CPU)")

# --------------------------------------------------
# Fit window (Fourier mode numbers)
#
# Example:
# python structure_factor.py 2 20
# --------------------------------------------------

imin = int(sys.argv[1]) if len(sys.argv) > 1 else 2
imax = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# --------------------------------------------------
# Binary header
# --------------------------------------------------

header_fmt = "<IIQQddd"
header_size = struct.calcsize(header_fmt)

files = sorted(glob.glob("u_*.bin"))

if len(files) == 0:
    raise RuntimeError("No u_*.bin files found.")

Sq = None
L = None
count = 0

# --------------------------------------------------
# Read all configurations
# --------------------------------------------------

for fname in files:

    with open(fname, "rb") as f:

        magic, version, Lfile, seed, Delta, n, c = \
            struct.unpack(header_fmt, f.read(header_size))

        if magic != 0x414C4D31:
            raise RuntimeError(f"{fname}: invalid file format")

        if L is None:
            L = Lfile
            Sq = xp.zeros(L//2 + 1, dtype=xp.float64)

        if Lfile != L:
            raise RuntimeError(f"{fname}: inconsistent L")

        u = np.fromfile(f, dtype=np.float64, count=L)

    u = xp.asarray(u)

    # Fourier transform
    uq = xp.fft.rfft(u)

    # Structure factor
    Sq += xp.abs(uq)**2

    count += 1

# --------------------------------------------------
# Average
# --------------------------------------------------

Sq /= (count * L)

if gpu:
    Sq = cp.asnumpy(Sq)

# --------------------------------------------------
# Wavevectors
# --------------------------------------------------

q = 2*np.pi*np.arange(len(Sq))/L

# Remove q=0
q = q[1:]
Sq = Sq[1:]

# --------------------------------------------------
# Fit
# --------------------------------------------------

qmin = 2*np.pi*imin/L
qmax = 2*np.pi*imax/L

mask = (q >= qmin) & (q <= qmax)

if mask.sum() < 3:
    raise RuntimeError("Fit interval contains too few points.")

fit = linregress(np.log(q[mask]), np.log(Sq[mask]))

m = fit.slope
b = fit.intercept

zeta = (-m - 1)/2
zeta_err = fit.stderr/2

Sq_fit = np.exp(b)*q**m

# --------------------------------------------------
# Print results
# --------------------------------------------------

n=2
print()
print("==========================================")
print(f"Configurations : {count}")
print(f"L              : {L}")
print(f"Delta          : {Delta}")
print(f"n              : {n}")
print(f"c              : {c}")
print("------------------------------------------")
print(f"Fit modes      : {imin} ... {imax}")
print(f"Slope          : {m:12.6f} ± {fit.stderr:.3e}")
print(f"zeta           : {zeta:12.6f} ± {zeta_err:.3e}")
print(f"zeta-(4n-2)/(4n-1)           : {zeta-(4*n-2)/(4*n-1):12.6f} ± {zeta_err:.3e}")
print(f"R²             : {fit.rvalue**2:.6f}")
print("==========================================")

# --------------------------------------------------
# Save data
# --------------------------------------------------

np.savetxt(
    "Sq.dat",
    np.column_stack((q, Sq)),
    header=(
        "q    <|u(q)|^2>/L\n"
        f"L={L}  samples={count}\n"
        f"Delta={Delta}  n={n}  c={c}"
    )
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(6,5))

plt.loglog(q, Sq, "ko", ms=3, label="Simulation")

plt.loglog(
    q,
    Sq_fit,
    "r-",
    lw=2,
    label=rf"Fit: $\zeta={zeta:.3f}$"
)

plt.axvline(qmin, color="gray", ls="--", lw=1)
plt.axvline(qmax, color="gray", ls="--", lw=1)

plt.xlabel(r"$q$")
plt.ylabel(r"$S(q)$")

plt.title(
    rf"$L={L}$, samples={count}, $\zeta={zeta:.3f}\pm{zeta_err:.3f}$"
)

plt.grid(True, which="both", alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig("Sq.png", dpi=300)

plt.show()
