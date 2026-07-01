import os
os.environ["CUPY_ACCELERATORS"] = ""

import sys
import argparse
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ── Parse Command Line Arguments ────────────────────────────
parser = argparse.ArgumentParser(description="Run ALM simulation for a given system size L.")
parser.add_argument(
    "-L", "--length", 
    type=int, 
    default=131072, 
    help="System size L (must be a power of 2 for optimal FFT performance). Default: 131072"
)
args = parser.parse_args()

# ── parameters ──────────────────────────────────────────────
L        = args.length  # Captured from command line
Delta    = 1.0
c        = 0.0          # must be 0 for full GPU path
nsamples = 20          # GPU makes large nsamples cheap
n_min, n_max = 2, 20
FIT_QMAX = 0.01
FIT_QMIN = 2.0*np.pi/L
tolerance_periodicity = 1e-8
# ────────────────────────────────────────────────────────────


def slopes_gpu(F_batch, n):
    """
    Constant memory footprint bisection solver.
    """
    exp = 1.0 / (2*n - 1)

    sigma = cp.empty_like(F_batch)
    s     = cp.empty_like(F_batch)

    def total_slope_batch(C_vec):
        nonlocal sigma, s
        cp.add(F_batch, C_vec[:, None], out=sigma)
        cp.abs(sigma, out=s)
        cp.power(s, exp, out=s)
        cp.multiply(s, cp.sign(sigma), out=s) 
        return s.sum(axis=1)

    lo = cp.full(nsamples, -1e4)
    hi = cp.full(nsamples,  1e4)
    for _ in range(70):
        mid  = (lo + hi) / 2.0
        fmid = total_slope_batch(mid)
        neg  = fmid < 0
        lo   = cp.where(neg, mid, lo)
        hi   = cp.where(neg, hi,  mid)

    C  = (lo + hi) / 2.0
    
    cp.add(F_batch, C[:, None], out=sigma)
    cp.abs(sigma, out=s)
    cp.power(s, exp, out=s)
    cp.multiply(s, cp.sign(sigma), out=s)

    u = cp.zeros_like(s)
    u[:, 1:] = cp.cumsum(s[:, :-1], axis=1)
    u    -= u.mean(axis=1, keepdims=True)

    periodicity = s.sum(axis=1)
    return u, periodicity

# ── frequency arrays ────────────────────────────────────────
q_np     = 2.0 * np.pi * np.fft.rfftfreq(L, d=1)
qpos_np  = q_np[1:]
fit_mask = (qpos_np >= FIT_QMIN) & (qpos_np < FIT_QMAX)
log_q    = np.log(qpos_np[fit_mask])

n_values       = list(range(n_min, n_max + 1))
narray         = []
zeta_avg_sq    = []
zeta_mean_arr  = []
zeta_std_arr   = []
sq_avg_store   = {}
fit_store      = {}
n_plot_values  = [2, 4, 8, 16]

seeds = list(range(nsamples))

print("Generating disorder...")
f_all = np.stack([
    np.random.default_rng(s).standard_normal(L) * np.sqrt(Delta)
    for s in seeds
])
f_all -= f_all.mean(axis=1, keepdims=True)
F_cpu = np.concatenate([np.zeros((nsamples, 1)),
                        np.cumsum(f_all[:, :-1], axis=1)], axis=1)
F_batch = cp.asarray(F_cpu)

for n in n_values:
    print(f"n = {n} ...", end=' ', flush=True)

    # 1. Run bisection on GPU
    u_batch_gpu, periodicity_gpu = slopes_gpu(F_batch, n)

    # 2. IMMEDIATELY pull outputs to CPU and free GPU pointers
    u_batch_np     = cp.asnumpy(u_batch_gpu)
    periodicity_np = cp.asnumpy(periodicity_gpu)
    
    del u_batch_gpu, periodicity_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # 3. All masking, slicing, and FFT math happens safely on CPU RAM
    valid_mask = np.abs(periodicity_np) < tolerance_periodicity
    n_valid    = int(valid_mask.sum())
    u_valid    = u_batch_np[valid_mask]

    # Compute S(q) on CPU via numpy
    Uq        = np.fft.rfft(u_valid, axis=1)
    sq_all_np = (np.abs(Uq)**2)[:, 1:]

    # Method 1: fit averaged S(q)
    sq_avg_np = sq_all_np.mean(axis=0)
    B, A      = np.polyfit(log_q, np.log(sq_avg_np[fit_mask]), 1)
    zeta_avg  = -(B + 1) / 2.0

    # Method 2: per-sample zeta (Vectorized to avoid long loops!)
    log_sq_mask = np.log(sq_all_np[:, fit_mask])
    log_q_mean  = np.mean(log_q)
    log_q_dev   = log_q - log_q_mean
    var_q       = np.sum(log_q_dev**2)
    
    # Vectorized analytical least-squares slope computation
    b_samples    = np.sum(log_q_dev * (log_sq_mask - np.mean(log_sq_mask, axis=1, keepdims=True)), axis=1) / var_q
    zeta_arr     = -(b_samples + 1) / 2.0
    zeta_mean    = zeta_arr.mean()
    zeta_std     = zeta_arr.std(ddof=1) / np.sqrt(n_valid)

    print(f"valid={n_valid}, zeta_avg={zeta_avg:.4f}, theory={(4*n-1)/(4*n-2):.4f}, difference={zeta_avg - (4*n-1)/(4*n-2):.4e}, zeta_mean={zeta_mean:.4f}, zeta_std={zeta_std:.4e}, difference={zeta_mean - (4*n-1)/(4*n-2):.4e}")


    narray.append(n)
    zeta_avg_sq.append(zeta_avg)
    zeta_mean_arr.append(zeta_mean)
    zeta_std_arr.append(zeta_std)

    if n in n_plot_values:
        sq_avg_store[n] = sq_avg_np
        fit_store[n]    = (zeta_avg, zeta_mean, B, A)

# ── save & plot (remains identical) ───────────────────
narray_np    = np.array(narray)
zeta_avg_np  = np.array(zeta_avg_sq)
zeta_mean_np = np.array(zeta_mean_arr)
zeta_std_np  = np.array(zeta_std_arr)
zeta_th      = (4.*narray_np - 1) / (4.*narray_np - 2)

np.savetxt('zeta_results.txt',
           np.column_stack((narray_np, zeta_avg_np, zeta_mean_np, zeta_std_np)),
           header='n  zeta_avg_sq  zeta_mean_samples  zeta_std_samples',
           fmt=['%d','%.18e','%.18e','%.18e'], comments='')

# Plot zeta vs n
plt.figure(figsize=(10,6))
plt.plot(narray_np, zeta_avg_np, 'o-', color='steelblue', label=r'$\zeta$ from $\langle S(q)\rangle$')
plt.errorbar(narray_np, zeta_mean_np, yerr=zeta_std_np, fmt='s--', color='darkorange', capsize=3, label=r'$\langle\zeta\rangle$ from individual $S(q)$')
plt.plot(narray_np, zeta_th, '--r', label=r'Global $\zeta=(4n-1)/(4n-2)$')
plt.xlabel('n'); plt.ylabel(r'$\zeta$'); plt.legend(); plt.grid()
plt.savefig('zeta_vs_n.png', dpi=150); plt.close()

# Plot residuals and differences
plt.figure(figsize=(10,6))
plt.plot(narray_np, zeta_avg_np - zeta_th, 'o-', color='steelblue', label=r'$\zeta[\langle S(q)\rangle]$')
plt.errorbar(narray_np, zeta_mean_np - zeta_th, yerr=zeta_std_np, fmt='s--', color='darkorange', capsize=3, label=r'$\langle \zeta[S(q)]\rangle$')
plt.axhline(0, color='k', ls='--')
plt.xlabel('n'); plt.ylabel(r'$\zeta - \zeta_s$')
plt.title('Residuals'); plt.legend(); plt.grid()
plt.savefig('zeta_residuals.png', dpi=150); plt.close()

# Plot difference between estimators
plt.figure(figsize=(10,6))
plt.errorbar(narray_np, zeta_mean_np - zeta_avg_np, yerr=zeta_std_np, fmt='o-', color='purple', capsize=3)
plt.axhline(0, color='k', ls='--')
plt.xlabel('n'); plt.ylabel(r'$\langle\zeta\rangle_\mathrm{samples} - \zeta[\langle S(q)\rangle]$')
plt.title('Difference between estimators'); plt.grid()
plt.savefig('zeta_comparison.png', dpi=150); plt.close()

# Plot S(q) for selected n values
fig, axes = plt.subplots(2, 2, figsize=(12,9))
for ax, n_val in zip(axes.flatten(), n_plot_values):
    if n_val not in sq_avg_store:
        continue
    sq_avg                    = sq_avg_store[n_val]
    zeta_avg, zeta_mean, B, A = fit_store[n_val]
    zeta_t                    = (4.*n_val-1)/(4.*n_val-2)
    ax.loglog(qpos_np, sq_avg, color='steelblue', alpha=0.6, lw=0.8, label=r'$\langle S(q)\rangle$')
    q_ext = np.logspace(np.log10(qpos_np[fit_mask].min()), np.log10(qpos_np[fit_mask].max()*2), 200)
    ax.loglog(q_ext, np.exp(A)*q_ext**B, 'r--', lw=2, label=(rf'Fit: $\zeta={zeta_avg:.3f}$' + '\n' + rf'$\langle\zeta\rangle={zeta_mean:.3f}$' + '\n' + rf'Theory: {zeta_t:.3f}'))
    ax.axvspan(qpos_np[fit_mask].min(), qpos_np[fit_mask].max(), alpha=0.12, color='green', label='Fit region')
    ax.set(xlabel=r'$q$', ylabel=r'$\langle S(q)\rangle$', title=rf'$n={n_val}$')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, which='both', alpha=0.3)
fig.suptitle(r'$\langle S(q)\rangle$ vs $q$', fontsize=13)
plt.tight_layout()
plt.savefig('sq_panel.png', dpi=150); plt.close()

print("Done.")
