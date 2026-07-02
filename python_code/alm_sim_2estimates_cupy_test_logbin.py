import os
os.environ["CUPY_ACCELERATORS"] = ""

import sys
import argparse
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ── Parse Command Line Arguments ────────────────────────────
parser = argparse.ArgumentParser(description="Run ALM simulation with Logarithmic Binning.")
parser.add_argument(
    "-L", "--length", 
    type=int, 
    default=131072, 
    help="System size L (must be a power of 2 for optimal FFT performance). Default: 131072"
)
args = parser.parse_args()

# ── parameters ──────────────────────────────────────────────
L        = args.length  
Delta    = 1.0
c        = 0.0          
nsamples = 20          
n_min, n_max = 2, 16
FIT_QMAX = 0.01
FIT_QMIN = 2.0 * np.pi / L
NUM_BINS = 30  # Number of geometric bins for log-binning
# ────────────────────────────────────────────────────────────

def slopes_gpu(F_batch, n):
    """
    Constant memory footprint bisection solver forced at high precision float64.
    """
    exp = cp.float64(1.0 / (2*n - 1))
    sigma = cp.empty_like(F_batch, dtype=cp.float64)
    s     = cp.empty_like(F_batch, dtype=cp.float64)

    def total_slope_batch(C_vec):
        nonlocal sigma, s
        cp.add(F_batch, C_vec[:, None], out=sigma)
        cp.abs(sigma, out=s)
        cp.power(s, exp, out=s)
        cp.multiply(s, cp.sign(sigma), out=s) 
        # Force high-precision tree reduction accumulator
        return s.sum(axis=1, dtype=cp.float64)

    lo = cp.full(nsamples, -1e4, dtype=cp.float64)
    hi = cp.full(nsamples,  1e4, dtype=cp.float64)
    
    # 53 iterations is the maximum mathematical resolution limit for 64-bit floats
    for _ in range(53):
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

    u = cp.zeros_like(s, dtype=cp.float64)
    u[:, 1:] = cp.cumsum(s[:, :-1], axis=1)
    u    -= u.mean(axis=1, keepdims=True)

    periodicity = s.sum(axis=1, dtype=cp.float64)
    return u, periodicity


def log_bin_data(q_vals, sq_vals, num_bins=30):
    """
    Bins q and S(q) data into geometrically spaced log-bins.
    Handles both 1D (averaged) and 2D (sample-by-sample) inputs.
    """
    q_min, q_max = q_vals.min(), q_vals.max()
    bin_edges = np.logspace(np.log10(q_min), np.log10(q_max), num_bins + 1)
    
    binned_q = []
    binned_sq = []
    
    for i in range(num_bins):
        mask = (q_vals >= bin_edges[i]) & (q_vals < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_q.append(np.mean(q_vals[mask]))
            if sq_vals.ndim == 1:
                binned_sq.append(np.mean(sq_vals[mask]))
            else:
                binned_sq.append(np.mean(sq_vals[:, mask], axis=1))
                
    if sq_vals.ndim == 1:
        return np.array(binned_q), np.array(binned_sq)
    else:
        return np.array(binned_q), np.column_stack(binned_sq)


# ── frequency arrays ────────────────────────────────────────
q_np     = 2.0 * np.pi * np.fft.rfftfreq(L, d=1)
qpos_np  = q_np[1:]
fit_mask = (qpos_np >= FIT_QMIN) & (qpos_np < FIT_QMAX)

n_values       = list(range(n_min, n_max + 1))
narray         = []
zeta_avg_sq    = []
zeta_avg_std_arr = []  
zeta_mean_arr  = []
zeta_std_arr   = []
sq_avg_store   = {}
fit_store      = {}
n_plot_values  = [2, 4, 8, 16]

seeds = list(range(nsamples))

print("Generating disorder...")
f_all = np.stack([
    np.random.default_rng(s).standard_normal(L, dtype=np.float64) * np.sqrt(Delta)
    for s in seeds
])
# Clean force drift globally before cumulative sums
f_all -= f_all.mean(axis=1, keepdims=True)
F_cpu = np.concatenate([np.zeros((nsamples, 1), dtype=np.float64),
                        np.cumsum(f_all[:, :-1], axis=1)], axis=1)
F_batch = cp.asarray(F_cpu, dtype=cp.float64)

for n in n_values:
    print(f"n = {n} ... ", end='', flush=True)

    u_batch_gpu, periodicity_gpu = slopes_gpu(F_batch, n)

    u_batch_np     = cp.asnumpy(u_batch_gpu)
    periodicity_np = cp.asnumpy(periodicity_gpu)
    
    del u_batch_gpu, periodicity_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # ─── UNBIASED SYSTEMATICS CORRECTOR ───
    # Subtract precise linear trend to perfectly close the loop
    ramp_grid  = np.arange(L, dtype=np.float64)
    u_periodic = u_batch_np - (periodicity_np[:, None] / L) * ramp_grid[None, :]
    u_periodic -= u_periodic.mean(axis=1, keepdims=True)

    Uq        = np.fft.rfft(u_periodic, axis=1)
    sq_all_np = (np.abs(Uq)**2)[:, 1:]

    # Extract our fitting window points
    q_window  = qpos_np[fit_mask]
    sq_window = sq_all_np[:, fit_mask]

    # ─── LOGARITHMIC BINNING STEP ───────────────────────────────────
    q_bin, sq_all_bin = log_bin_data(q_window, sq_window, num_bins=NUM_BINS)
    log_q_bin        = np.log(q_bin)

    # ─── Method 1: Fit Binned Averaged S(q) + Propagation ───────────
    sq_avg_bin = sq_all_bin.mean(axis=0)
    sq_std_bin = sq_all_bin.std(axis=0, ddof=1)
    
    B, A      = np.polyfit(log_q_bin, np.log(sq_avg_bin), 1)
    zeta_avg  = -(B + 1) / 2.0

    sq_sem_bin = sq_std_bin / np.sqrt(nsamples)
    y_err      = sq_sem_bin / sq_avg_bin
    log_q_mean = np.mean(log_q_bin)
    log_q_dev  = log_q_bin - log_q_mean
    var_q      = np.sum(log_q_dev**2)
    c_i        = log_q_dev / var_q
    B_err        = np.sqrt(np.sum((c_i**2) * (y_err**2)))
    zeta_avg_std = B_err / 2.0

    # ─── Method 2: Binned Per-Sample Zeta (Vectorized) ──────────────
    log_sq_mask = np.log(sq_all_bin)
    b_samples    = np.sum(log_q_dev * (log_sq_mask - np.mean(log_sq_mask, axis=1, keepdims=True)), axis=1) / var_q
    zeta_arr     = -(b_samples + 1) / 2.0
    zeta_mean    = zeta_arr.mean()
    zeta_std     = zeta_arr.std(ddof=1) / np.sqrt(nsamples)

    print(f"valid={nsamples}, zeta_avg={zeta_avg:.4f}±{zeta_avg_std:.2e}, theory={(4*n-1)/(4*n-2):.4f}, zeta_mean={zeta_mean:.4f}±{zeta_std:.2e}")

    narray.append(n)
    zeta_avg_sq.append(zeta_avg)
    zeta_avg_std_arr.append(zeta_avg_std)
    zeta_mean_arr.append(zeta_mean)
    zeta_std_arr.append(zeta_std)

    if n in n_plot_values:
        sq_avg_store[n] = sq_all_np.mean(axis=0)  # Store unbinned for comprehensive plotting
        fit_store[n]    = (zeta_avg, zeta_mean, B, A)

# ── save & plot ─────────────────────────────────────────────
narray_np       = np.array(narray)
zeta_avg_np     = np.array(zeta_avg_sq)
zeta_avg_std_np = np.array(zeta_avg_std_arr)
zeta_mean_np    = np.array(zeta_mean_arr)
zeta_std_np     = np.array(zeta_std_arr)
zeta_th         = (4.*narray_np - 1) / (4.*narray_np - 2)

np.savetxt('zeta_results.txt',
           np.column_stack((narray_np, zeta_avg_np, zeta_avg_std_np, zeta_mean_np, zeta_std_np)),
           header='n  zeta_avg_sq  zeta_avg_std  zeta_mean_samples  zeta_std_samples',
           fmt=['%d','%.18e','%.18e','%.18e','%.18e'], comments='')

# Plot zeta vs n
plt.figure(figsize=(10,6))
plt.errorbar(narray_np, zeta_avg_np, yerr=zeta_avg_std_np, fmt='o-', color='steelblue', capsize=3, label=r'$\zeta$ from log-binned $\langle S(q)\rangle$')
plt.errorbar(narray_np, zeta_mean_np, yerr=zeta_std_np, fmt='s--', color='darkorange', capsize=3, label=r'$\langle\zeta\rangle$ from log-binned individual $S(q)$')
plt.plot(narray_np, zeta_th, '--r', label=r'Global $\zeta=(4n-1)/(4n-2)$')
plt.xlabel('n'); plt.ylabel(r'$\zeta$'); plt.legend(); plt.grid()
plt.savefig('zeta_vs_n.png', dpi=150); plt.close()

# Plot residuals
plt.figure(figsize=(10,6))
plt.errorbar(narray_np, zeta_avg_np - zeta_th, yerr=zeta_avg_std_np, fmt='o-', color='steelblue', capsize=3, label=r'$\zeta[\langle S(q)\rangle]$')
plt.errorbar(narray_np, zeta_mean_np - zeta_th, yerr=zeta_std_np, fmt='s--', color='darkorange', capsize=3, label=r'$\langle \zeta[S(q)]\rangle$')
plt.axhline(0, color='k', ls='--')
plt.xlabel('n'); plt.ylabel(r'$\zeta - \zeta_s$')
plt.title('Residuals (With Logarithmic Binning)'); plt.legend(); plt.grid()
plt.savefig('zeta_residuals.png', dpi=150); plt.close()

# Plot difference between estimators
plt.figure(figsize=(10,6))
combined_err = np.sqrt(zeta_mean_np**2 + zeta_avg_std_np**2)
plt.errorbar(narray_np, zeta_mean_np - zeta_avg_np, yerr=combined_err, fmt='o-', color='purple', capsize=3)
plt.axhline(0, color='k', ls='--')
plt.xlabel('n'); plt.ylabel(r'$\langle\zeta\rangle_\mathrm{samples} - \zeta[\langle S(q)\rangle]$')
plt.title('Difference between estimators'); plt.grid()
plt.savefig('zeta_comparison.png', dpi=150); plt.close()

# Plot S(q) panel showing raw vs binned data
fig, axes = plt.subplots(2, 2, figsize=(12,9))
for ax, n_val in zip(axes.flatten(), n_plot_values):
    if n_val not in sq_avg_store:
        continue
    sq_avg                    = sq_avg_store[n_val]
    zeta_avg, zeta_mean, B, A = fit_store[n_val]
    zeta_t                    = (4.*n_val-1)/(4.*n_val-2)
    
    # 1. Plot raw backdrop points faintly
    ax.loglog(qpos_np, sq_avg, color='steelblue', alpha=0.25, lw=0.6, label=r'Raw $\langle S(q)\rangle$')
    
    # 2. Extract and display the exact binned points that were supplied to the regression matrix
    q_w = qpos_np[fit_mask]
    q_b, sq_b = log_bin_data(q_w, sq_avg[fit_mask], num_bins=NUM_BINS)
    ax.loglog(q_b, sq_b, 'o', color='darkblue', markersize=4, label='Log-Binned Data')
    
    # 3. Plot fit lines mapped out
    q_ext = np.logspace(np.log10(q_w.min()), np.log10(q_w.max()*1.5), 200)
    ax.loglog(q_ext, np.exp(A)*q_ext**B, 'r--', lw=2, label=(rf'Fit: $\zeta={zeta_avg:.3f}$' + '\n' + rf'$\langle\zeta\rangle={zeta_mean:.3f}$' + '\n' + rf'Theory: {zeta_t:.3f}'))
    
    ax.axvspan(q_w.min(), q_w.max(), alpha=0.08, color='green', label='Fit region')
    ax.set(xlabel=r'$q$', ylabel=r'$\langle S(q)\rangle$', title=rf'$n={n_val}$')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, which='both', alpha=0.3)
fig.suptitle(r'Log-Binned $\langle S(q)\rangle$ vs $q$', fontsize=13)
plt.tight_layout()
plt.savefig('sq_panel.png', dpi=150); plt.close()

print("Done.")
