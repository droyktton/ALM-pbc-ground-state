import os
os.environ["CUPY_ACCELERATORS"] = ""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# ── parameters ──────────────────────────────────────────────
L        = 131072
Delta    = 1.0
c        = 0.0          # must be 0 for full GPU path
nsamples = 500          # GPU makes large nsamples cheap
n_min, n_max = 2, 50
FIT_QMAX = 0.01
# ────────────────────────────────────────────────────────────

def slopes_gpu(F_batch, n):
    """
    F_batch : (nsamples, L) CuPy array of cumulative forces
    Returns  : (nsamples, L) slopes, (nsamples,) total slopes after C correction
    """
    exp = 1.0 / (2*n - 1)

    def total_slope_batch(C_vec):
        # C_vec: (nsamples,) — one C per sample
        sigma = F_batch + C_vec[:, None]          # (nsamples, L)
        s     = cp.sign(sigma) * cp.abs(sigma)**exp
        return s.sum(axis=1)                      # (nsamples,)

    # Bisection over C, vectorized over all samples simultaneously
    lo = cp.full(nsamples, -1e4)
    hi = cp.full(nsamples,  1e4)
    for _ in range(60):                           # 60 iterations → ~1e-18 precision
        mid  = (lo + hi) / 2.0
        fmid = total_slope_batch(mid)
        # move bracket
        neg  = fmid < 0
        lo   = cp.where(neg, mid, lo)
        hi   = cp.where(neg, hi,  mid)

    C  = (lo + hi) / 2.0
    sigma = F_batch + C[:, None]
    s     = cp.sign(sigma) * cp.abs(sigma)**exp

    # Displacement: cumsum along sites axis, subtract mean
    u     = cp.zeros_like(s)
    u[:, 1:] = cp.cumsum(s[:, :-1], axis=1)
    u    -= u.mean(axis=1, keepdims=True)

    periodicity = s.sum(axis=1)                   # should be ~0 for all samples
    return u, periodicity


def generate_batch_gpu(seeds, L, Delta, n):
    """Generate a full batch of samples on the GPU."""
    rng = cp.random.default_rng(seed=0)

    # Draw all random numbers at once: (nsamples, L)
    # Reproducible per-seed: use numpy for seeded draws, transfer to GPU
    f_all = np.stack([
        np.random.default_rng(s).standard_normal(L) * np.sqrt(Delta)
        for s in seeds
    ])                                            # (nsamples, L)  on CPU
    f_all -= f_all.mean(axis=1, keepdims=True)    # zero mean per sample

    F_batch = cp.asarray(
        np.concatenate([np.zeros((len(seeds), 1)),
                        np.cumsum(f_all[:, :-1], axis=1)], axis=1)
    )                                             # (nsamples, L) on GPU

    return generate_batch_gpu_F(F_batch, n)


def generate_batch_gpu_F(F_batch, n):
    return slopes_gpu(F_batch, n)


# ── frequency arrays ────────────────────────────────────────
q_np     = np.fft.rfftfreq(L, d=1)              # use rfft: L/2+1 freqs, real input
qpos_np  = q_np[1:]                              # drop DC (q=0)
fit_mask = qpos_np < FIT_QMAX
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

# Pre-generate disorder on CPU once (shared across all n)
print("Generating disorder...")
f_all = np.stack([
    np.random.default_rng(s).standard_normal(L) * np.sqrt(Delta)
    for s in seeds
])
f_all -= f_all.mean(axis=1, keepdims=True)
F_cpu = np.concatenate([np.zeros((nsamples, 1)),
                        np.cumsum(f_all[:, :-1], axis=1)], axis=1)
F_batch = cp.asarray(F_cpu)                      # transfer once, reuse for all n

for n in n_plot_values: #n_values:
    print(f"n = {n} ...", end=' ', flush=True)

    u_batch, periodicity = slopes_gpu(F_batch, n)

    # Validity mask
    valid_mask = cp.abs(periodicity) < 1e-7      # (nsamples,)
    n_valid    = int(valid_mask.sum())
    u_valid    = u_batch[valid_mask]              # (n_valid, L)

    # S(q) for all valid samples via rfft
    Uq        = cp.fft.rfft(u_valid, axis=1)     # (n_valid, L//2+1)
    sq_all    = (cp.abs(Uq)**2)[:, 1:]           # drop DC → (n_valid, len(qpos))

    # Transfer to CPU for fitting
    sq_all_np = cp.asnumpy(sq_all)               # (n_valid, len(qpos))

    # Method 1: fit averaged S(q)
    sq_avg_np = sq_all_np.mean(axis=0)
    B, A      = np.polyfit(log_q, np.log(sq_avg_np[fit_mask]), 1)
    zeta_avg  = -(B + 1) / 2.0

    # Method 2: per-sample zeta
    zeta_samples = []
    for sq_s in sq_all_np:
        b, _ = np.polyfit(log_q, np.log(sq_s[fit_mask]), 1)
        zeta_samples.append(-(b + 1) / 2.0)
    zeta_arr  = np.array(zeta_samples)
    zeta_mean = zeta_arr.mean()
    zeta_std  = zeta_arr.std(ddof=1) / np.sqrt(n_valid)

    print(f"valid={n_valid}, zeta_avg={zeta_avg:.4f}, theory={(4*n-1)/(4*n-2):.4f}")

    narray.append(n)
    zeta_avg_sq.append(zeta_avg)
    zeta_mean_arr.append(zeta_mean)
    zeta_std_arr.append(zeta_std)

    if n in n_plot_values:
        sq_avg_store[n] = sq_avg_np
        fit_store[n]    = (zeta_avg, zeta_mean, B, A)

# ── save & plot (identical to alm_sim.py) ───────────────────
narray_np    = np.array(narray)
zeta_avg_np  = np.array(zeta_avg_sq)
zeta_mean_np = np.array(zeta_mean_arr)
zeta_std_np  = np.array(zeta_std_arr)
zeta_th      = (4.*narray_np - 1) / (4.*narray_np - 2)

np.savetxt('zeta_results.txt',
           np.column_stack((narray_np, zeta_avg_np, zeta_mean_np, zeta_std_np)),
           header='n  zeta_avg_sq  zeta_mean_samples  zeta_std_samples',
           fmt=['%d','%.18e','%.18e','%.18e'], comments='')

plt.figure(figsize=(10,6))
plt.plot(narray_np, zeta_avg_np, 'o-', color='steelblue',
         label=r'$\zeta$ from $\langle S(q)\rangle$')
plt.errorbar(narray_np, zeta_mean_np, yerr=zeta_std_np,
             fmt='s--', color='darkorange', capsize=3,
             label=r'$\langle\zeta\rangle$ from individual $S(q)$')
plt.plot(narray_np, zeta_th, '--r',
         label=r'Theory $\zeta_s=(4n-1)/(4n-2)$')
plt.xlabel('n'); plt.ylabel(r'$\zeta$'); plt.legend(); plt.grid()
plt.savefig('zeta_vs_n.png', dpi=150); plt.close()

plt.figure(figsize=(10,6))
plt.plot(narray_np, zeta_avg_np - zeta_th, 'o-', color='steelblue',
         label=r'$\zeta[\langle S(q)\rangle]$')
plt.errorbar(narray_np, zeta_mean_np - zeta_th, yerr=zeta_std_np,
             fmt='s--', color='darkorange', capsize=3)
plt.axhline(0, color='k', ls='--')
plt.xlabel('n'); plt.ylabel(r'$\zeta - \zeta_s$')
plt.title('Residuals'); plt.legend(); plt.grid()
plt.savefig('zeta_residuals.png', dpi=150); plt.close()

plt.figure(figsize=(10,6))
plt.errorbar(narray_np, zeta_mean_np - zeta_avg_np, yerr=zeta_std_np,
             fmt='o-', color='purple', capsize=3)
plt.axhline(0, color='k', ls='--')
plt.xlabel('n')
plt.ylabel(r'$\langle\zeta\rangle_\mathrm{samples} - \zeta[\langle S(q)\rangle]$')
plt.title('Difference between estimators'); plt.grid()
plt.savefig('zeta_comparison.png', dpi=150); plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12,9))
for ax, n_val in zip(axes.flatten(), n_plot_values):
    if n_val not in sq_avg_store:
        continue
    sq_avg                    = sq_avg_store[n_val]
    zeta_avg, zeta_mean, B, A = fit_store[n_val]
    zeta_t                    = (4.*n_val-1)/(4.*n_val-2)
    ax.loglog(qpos_np, sq_avg, color='steelblue', alpha=0.6, lw=0.8,
              label=r'$\langle S(q)\rangle$')
    q_ext = np.logspace(np.log10(qpos_np[fit_mask].min()),
                        np.log10(qpos_np[fit_mask].max()*2), 200)
    ax.loglog(q_ext, np.exp(A)*q_ext**B, 'r--', lw=2,
              label=(rf'Fit: $\zeta={zeta_avg:.3f}$' + '\n' +
                     rf'$\langle\zeta\rangle={zeta_mean:.3f}$' + '\n' +
                     rf'Theory: {zeta_t:.3f}'))
    ax.axvspan(qpos_np[fit_mask].min(), qpos_np[fit_mask].max(),
               alpha=0.12, color='green', label='Fit region')
    ax.set(xlabel=r'$q$', ylabel=r'$\langle S(q)\rangle$', title=rf'$n={n_val}$')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, which='both', alpha=0.3)
fig.suptitle(r'$\langle S(q)\rangle$ vs $q$', fontsize=13)
plt.tight_layout()
plt.savefig('sq_panel.png', dpi=150); plt.close()

print("Done.")
