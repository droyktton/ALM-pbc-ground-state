import os
os.environ["CUPY_ACCELERATORS"] = ""

import sys
import argparse
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ── Parse Command Line Arguments ────────────────────────────
parser = argparse.ArgumentParser(description="Run ALM simulation with Weighted Linear Regression.")
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
        return s.sum(axis=1, dtype=cp.float64)

    lo = cp.full(nsamples, -1e4, dtype=cp.float64)
    hi = cp.full(nsamples,  1e4, dtype=cp.float64)
    
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

# ── frequency arrays ────────────────────────────────────────
q_np     = 2.0 * np.pi * np.fft.rfftfreq(L, d=1)
qpos_np  = q_np[1:]
fit_mask = (qpos_np >= FIT_QMIN) & (qpos_np < FIT_QMAX)
log_q    = log_q_raw = np.log(qpos_np[fit_mask])

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
    ramp_grid  = np.arange(L, dtype=np.float64)
    u_periodic = u_batch_np - (periodicity_np[:, None] / L) * ramp_grid[None, :]
    u_periodic -= u_periodic.mean(axis=1, keepdims=True)

    Uq        = np.fft.rfft(u_periodic, axis=1)
    sq_all_np = (np.abs(Uq)**2)[:, 1:]

    # Extract our regular fitting window points
    sq_window = sq_all_np[:, fit_mask]

    # Calculate raw sample mean and standard deviations
    sq_avg_raw = sq_window.mean(axis=0)
    sq_std_raw = sq_window.std(axis=0, ddof=1)
    sq_sem_raw = sq_std_raw / np.sqrt(nsamples)

    # ─── WEIGHT CALCULATION ─────────────────────────────────────────
    # Variance of y = ln(S(q)) is (sigma_S / S)^2
    y_var = (sq_sem_raw / sq_avg_raw) ** 2
    # Statistical weight is the inverse variance
    weights = 1.0 / y_var

    # ─── Method 1: Weighted Least Squares Fit on <S(q)> ─────────────
    # np.polyfit accepts a 'w' parameter which expects 1/sigma (i.e. sqrt(weights))
    B, A      = np.polyfit(log_q, np.log(sq_avg_raw), 1, w=np.sqrt(weights))
    zeta_avg  = -(B + 1) / 2.0

    # Analytical Error Propagation for Weighted Regression
    # Matrix math: Covariance = (X^T * W * X)^-1
    W = weights
    S_w   = np.sum(W)
    S_wx  = np.sum(W * log_q)
    S_wxx = np.sum(W * log_q**2)
    delta = S_w * S_wxx - (S_wx)**2
    
    # Variance of the slope B is S_w / delta
    B_var = S_w / delta
    zeta_avg_std = np.sqrt(B_var) / 2.0

    # ─── Method 2: Weighted Per-Sample Zeta (Vectorized) ────────────
    # Every sample is individually fitted using the exact same structural weighting profile
    log_sq_mask = np.log(sq_window)
    
    # WLS analytical slope projection vector: c_i = (S_w * x_i - S_wx) * w_i / delta
    c_i_weighted = (S_w * log_q - S_wx) * weights / delta
    
    b_samples = np.sum(c_i_weighted * log_sq_mask, axis=1)
    zeta_arr  = -(b_samples + 1) / 2.0
    zeta_mean = zeta_arr.mean()
    zeta_std  = zeta_arr.std(ddof=1) / np.sqrt(nsamples)

    print(f"valid={nsamples}, zeta_avg={zeta_avg:.4f}±{zeta_avg_std:.2e}, theory={(4*n-1)/(4*n-2):.4f}, zeta_mean={zeta_mean:.4f}±{zeta_std:.2e}")

    narray.append(n)
    zeta_avg_sq.append(zeta_avg)
    zeta_avg_std_arr.append(zeta_avg_std)
    zeta_mean_arr.append(zeta_mean)
    zeta_std_arr.append(zeta_std)

    if n in n_plot_values:
        sq_avg_store[n] = sq_all_np.mean(axis=0)
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
plt.errorbar(narray_np, zeta_avg_np, yerr=zeta_avg_std_np, fmt='o-', color='steelblue', capsize=3, label=r'$\zeta$ from weighted $\langle S(q)\rangle$')
plt.errorbar(narray_np, zeta_mean_np, yerr=zeta_std_np, fmt='s--', color='darkorange', capsize=3, label=r'$\langle\zeta\rangle$ from weighted individual $S(q)$')
plt.plot(narray_np, zeta_th, '--r', label=r'Global $\zeta=(4n-1)/(4n-2)$')
plt.xlabel('n'); plt.ylabel(r'$\zeta$'); plt.legend(); plt.grid()
plt.savefig('zeta_vs_n.png', dpi=150); plt.close()

# Plot residuals
plt.figure(figsize=(10,6))
plt.errorbar(narray_np, zeta_avg_np - zeta_th, yerr=zeta_avg_std_np, fmt='o-', color='steelblue', capsize=3, label=r'$\zeta[\langle S(q)\rangle]$')
plt.errorbar(narray_np, zeta_mean_np - zeta_th, yerr=zeta_std_np, fmt='s--', color='darkorange', capsize=3, label=r'$\langle \zeta[S(q)]\rangle$')
plt.axhline(0, color='k', ls='--')
plt.xlabel('n'); plt.ylabel(r'$\zeta - \zeta_s$')
plt.title('Residuals (With Weighted Regression)'); plt.legend(); plt.grid()
plt.savefig('zeta_residuals.png', dpi=150); plt.close()

# Plot difference between estimators
plt.figure(figsize=(10,6))
combined_err = np.sqrt(zeta_mean_np**2 + zeta_avg_std_np**2)
plt.errorbar(narray_np, zeta_mean_np - zeta_avg_np, yerr=combined_err, fmt='o-', color='purple', capsize=3)
plt.axhline(0, color='k', ls='--')
plt.xlabel('n'); plt.ylabel(r'$\langle\zeta\rangle_\mathrm{samples} - \zeta[\langle S(q)\rangle]$')
plt.title('Difference between estimators'); plt.grid()
plt.savefig('zeta_comparison.png', dpi=150); plt.close()

# Plot S(q) panel showing raw vs weighted fit lines
fig, axes = plt.subplots(2, 2, figsize=(12,9))
for ax, n_val in zip(axes.flatten(), n_plot_values):
    if n_val not in sq_avg_store:
        continue
    sq_avg                    = sq_avg_store[n_val]
    zeta_avg, zeta_mean, B, A = fit_store[n_val]
    zeta_t                    = (4.*n_val-1)/(4.*n_val-2)
    
    ax.loglog(qpos_np, sq_avg, color='steelblue', alpha=0.5, lw=0.8, label=r'Raw $\langle S(q)\rangle$')
    
    q_w = qpos_np[fit_mask]
    q_ext = np.logspace(np.log10(q_w.min()), np.log10(q_w.max()*1.5), 200)
    ax.loglog(q_ext, np.exp(A)*q_ext**B, 'r--', lw=2, label=(rf'Fit: $\zeta={zeta_avg:.3f}$' + '\n' + rf'$\langle\zeta\rangle={zeta_mean:.3f}$' + '\n' + rf'Theory: {zeta_t:.3f}'))
    
    ax.axvspan(q_w.min(), q_w.max(), alpha=0.08, color='green', label='Fit region')
    ax.set(xlabel=r'$q$', ylabel=r'$\langle S(q)\rangle$', title=rf'$n={n_val}$')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, which='both', alpha=0.3)
fig.suptitle(r'Weighted Fit on Linear $\langle S(q)\rangle$', fontsize=13)
plt.tight_layout()
plt.savefig('sq_panel.png', dpi=150); plt.close()

print("Done.")
