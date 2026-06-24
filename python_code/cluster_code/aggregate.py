import numpy as np
import matplotlib.pyplot as plt
import glob, os

OUTDIR   = 'results'
L        = 8192
FIT_QMAX = 0.01

files = sorted(glob.glob(os.path.join(OUTDIR, 'partial_*.npz')))
print(f"Found {len(files)} partial files")

# Load first file to get shape
ref      = np.load(files[0], allow_pickle=True)
n_values = ref['n_values']
qpos     = ref['qpos']
fit_mask = qpos < FIT_QMAX
log_q    = np.log(qpos[fit_mask])

sq_sum_total = np.zeros((len(n_values), len(qpos)))
valid_total  = np.zeros(len(n_values), dtype=int)
sq_per_all   = [[] for _ in n_values]   # sq_per_all[i] = list of per-seed arrays for n_values[i]

for f in files:
    data = np.load(f, allow_pickle=True)
    sq_sum_total += data['sq_sum']
    valid_total  += data['valid_count']
    for i, sq_list in enumerate(data['sq_per']):
        sq_per_all[i].extend(list(sq_list))

n_plot_values = [2, 4, 8, 16]
narray, zeta_avg_sq, zeta_mean_arr, zeta_std_arr = [], [], [], []
sq_avg_store, fit_store = {}, {}

for i, n in enumerate(n_values):
    valid = valid_total[i]
    if valid == 0:
        continue

    # Method 1: fit averaged S(q)
    sq_avg   = sq_sum_total[i] / valid
    log_sq   = np.log(sq_avg[fit_mask])
    B, A     = np.polyfit(log_q, log_sq, 1)
    zeta_avg = -(B + 1) / 2.0

    # Method 2: per-sample zeta
    zeta_samples = []
    for sq_s in sq_per_all[i]:
        b, _ = np.polyfit(log_q, np.log(sq_s[fit_mask]), 1)
        zeta_samples.append(-(b + 1) / 2.0)
    zeta_arr  = np.array(zeta_samples)
    zeta_mean = zeta_arr.mean()
    zeta_std  = zeta_arr.std(ddof=1) / np.sqrt(valid)

    narray.append(n)
    zeta_avg_sq.append(zeta_avg)
    zeta_mean_arr.append(zeta_mean)
    zeta_std_arr.append(zeta_std)

    if n in n_plot_values:
        sq_avg_store[n] = sq_avg
        fit_store[n]    = (zeta_avg, zeta_mean, B, A)

narray_np    = np.array(narray)
zeta_avg_np  = np.array(zeta_avg_sq)
zeta_mean_np = np.array(zeta_mean_arr)
zeta_std_np  = np.array(zeta_std_arr)
zeta_th      = (4.*narray_np - 1) / (4.*narray_np - 2)

np.savetxt('zeta_results.txt',
           np.column_stack((narray_np, zeta_avg_np, zeta_mean_np, zeta_std_np)),
           header='n  zeta_avg_sq  zeta_mean_samples  zeta_std_samples',
           fmt=['%d','%.18e','%.18e','%.18e'], comments='')

# --- Plots (same as alm_sim.py) ---
plt.figure(figsize=(10,6))
plt.plot(narray_np, zeta_avg_np, 'o-', color='steelblue',
         label=r'$\zeta$ from $\langle S(q)\rangle$')
plt.errorbar(narray_np, zeta_mean_np, yerr=zeta_std_np,
             fmt='s--', color='darkorange', capsize=3,
             label=r'$\langle\zeta\rangle$ from individual $S(q)$')
plt.plot(narray_np, zeta_th, '--r',
         label=r'Theory $\zeta_s=(4n-1)/(4n-2)$')
plt.xlabel('n'); plt.ylabel(r'$\zeta$')
plt.title('Roughness Exponent vs n'); plt.legend(); plt.grid()
plt.savefig('zeta_vs_n.png', dpi=150); plt.close()

plt.figure(figsize=(10,6))
plt.plot(narray_np, zeta_avg_np - zeta_th, 'o-', color='steelblue',
         label=r'$\zeta[\langle S(q)\rangle]$')
plt.errorbar(narray_np, zeta_mean_np - zeta_th, yerr=zeta_std_np,
             fmt='s--', color='darkorange', capsize=3,
             label=r'$\langle\zeta\rangle_\mathrm{samples}$')
plt.axhline(0, 'k--'); plt.xlabel('n')
plt.ylabel(r'$\zeta - \zeta_s$'); plt.title('Residuals')
plt.legend(); plt.grid()
plt.savefig('zeta_residuals.png', dpi=150); plt.close()

plt.figure(figsize=(10,6))
plt.errorbar(narray_np, zeta_mean_np - zeta_avg_np, yerr=zeta_std_np,
             fmt='o-', color='purple', capsize=3)
plt.axhline(0, 'k--'); plt.xlabel('n')
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
    ax.loglog(qpos, sq_avg, color='steelblue', alpha=0.6, lw=0.8,
              label=r'$\langle S(q)\rangle$')
    q_ext   = np.logspace(np.log10(qpos[fit_mask].min()),
                          np.log10(qpos[fit_mask].max()*2), 200)
    ax.loglog(q_ext, np.exp(A)*q_ext**B, 'r--', lw=2,
              label=(rf'Fit: $\zeta={zeta_avg:.3f}$' + '\n' +
                     rf'$\langle\zeta\rangle={zeta_mean:.3f}$' + '\n' +
                     rf'Theory: {zeta_t:.3f}'))
    ax.axvspan(qpos[fit_mask].min(), qpos[fit_mask].max(),
               alpha=0.12, color='green', label='Fit region')
    ax.set(xlabel=r'$q$', ylabel=r'$\langle S(q)\rangle$', title=rf'$n={n_val}$')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, which='both', alpha=0.3)
fig.suptitle(r'$\langle S(q)\rangle$ vs $q$', fontsize=13)
plt.tight_layout()
plt.savefig('sq_panel.png', dpi=150); plt.close()

print("Aggregation complete.")

