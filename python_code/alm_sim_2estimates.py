import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def generate_sample(L, Delta, n, c, seed):
    np.random.seed(seed)

    # Disorder
    f = np.random.normal(0.0, np.sqrt(Delta), size=L)
    f -= f.mean()

    # Cumulative disorder force
    F = np.concatenate([[0.0], np.cumsum(f[:-1])])

    # Constitutive inversion
    def slope_from_stress(sigma):
        if c == 0:
            return np.sign(sigma) * abs(sigma)**(1.0/(2*n-1))
        def eq(s):
            return c*s + np.sign(s)*abs(s)**(2*n-1) - sigma
        smax = max(1.0, abs(sigma)**(1.0/(2*n-1)) + 1.0)
        return brentq(eq, -smax, smax)

    vec_slope = np.vectorize(slope_from_stress)

    # Constraint: zero total slope
    def total_slope(C):
        sigma = F + C
        s = vec_slope(sigma)
        return np.sum(s)

    C = brentq(total_slope, -1000.0, 1000.0)

    # Build solution
    sigma = F + C
    s = vec_slope(sigma)
    u = np.zeros(L)
    u[1:] = np.cumsum(s[:-1])
    u -= u.mean()

    periodicity = np.sum(s)
    return u, periodicity


# Parameters
L       = 8192
Delta   = 1.0
c       = 0
nsamples = 50

# Precompute frequency arrays (same for all n)
q         = np.fft.fftfreq(L, d=1)
qpos      = q[q > 0]
fit_mask  = qpos < 0.01
log_q_fit = np.log(qpos[fit_mask])

# Output arrays
narray           = []
zetaarray        = []   # zeta from averaged S(q)
zeta_mean_array  = []   # mean of per-sample zeta
zeta_std_array   = []   # std error of per-sample zeta mean

# Values of n for which we store S(q) for the panel plot
n_plot_values = [2, 4, 8, 16]
sq_avg_store  = {}   # n -> averaged S(q) (positive freqs only)
fit_store     = {}   # n -> (zeta_avg_sq, zeta_mean, fit slope B from avg S(q))

for n in range(2, 51):
    print(f"Processing n={n}...")

    sq      = np.zeros(L)
    zeta_samples = []
    invalid = 0

    for seed in range(nsamples):
        u, periodicity = generate_sample(L, Delta, n, c, seed)

        if abs(periodicity) < 1e-7:
            sq_sample = np.abs(np.fft.fft(u))**2
            sq += sq_sample

            # Fit zeta for this individual sample
            sqpos_sample = sq_sample[q > 0]
            log_sq = np.log(sqpos_sample[fit_mask])
            coeffs = np.polyfit(log_q_fit, log_sq, 1)
            zeta_samples.append(-(coeffs[0] + 1) / 2.0)
        else:
            invalid += 1

    valid = nsamples - invalid
    if valid > 0:
        # --- Method 1: fit from averaged S(q) ---
        sq_avg   = sq / valid
        sqpos_avg = sq_avg[q > 0]
        log_sq_avg = np.log(sqpos_avg[fit_mask])
        coeffs = np.polyfit(log_q_fit, log_sq_avg, 1)
        zeta_from_avg_sq = -(coeffs[0] + 1) / 2.0

        # --- Method 2: average of per-sample zeta ---
        zeta_arr  = np.array(zeta_samples)
        zeta_mean = zeta_arr.mean()
        zeta_std  = zeta_arr.std(ddof=1) / np.sqrt(valid)

        narray.append(n)
        zetaarray.append(zeta_from_avg_sq)
        zeta_mean_array.append(zeta_mean)
        zeta_std_array.append(zeta_std)

        # Store for S(q) panel plot
        if n in n_plot_values:
            sq_avg_store[n] = sqpos_avg
            # full polyfit: slope B and intercept A, so S(q) ~ exp(A) * q^B
            B, A = np.polyfit(log_q_fit, log_sq_avg, 1)
            fit_store[n] = (zeta_from_avg_sq, zeta_mean, B, A)

narray_np      = np.array(narray)
zetaarray_np   = np.array(zetaarray)
zeta_mean_np   = np.array(zeta_mean_array)
zeta_std_np    = np.array(zeta_std_array)
zeta_theoretical = (4. * narray_np - 1) / (4. * narray_np - 2)

# 1) Save data with high numerical precision
data_to_save = np.column_stack((narray_np, zetaarray_np, zeta_mean_np, zeta_std_np))
np.savetxt('zeta_results.txt', data_to_save,
           header='n  zeta_from_avg_sq  zeta_mean_samples  zeta_std_samples',
           fmt=['%d', '%.18e', '%.18e', '%.18e'], comments='')

# 2) Plot 1: Both zeta estimates + theoretical
plt.figure(figsize=(10, 6))
plt.plot(narray_np, zetaarray_np,
         marker='o', linestyle='-', color='steelblue',
         label=r'$\zeta$ from $\langle S(q) \rangle$')
plt.errorbar(narray_np, zeta_mean_np, yerr=zeta_std_np,
             marker='s', linestyle='--', color='darkorange', capsize=3,
             label=r'$\langle \zeta \rangle$ from individual $S(q)$')
plt.plot(narray_np, zeta_theoretical,
         '--', color='red',
         label=r'Theoretical $\zeta_s = \frac{4n-1}{4n-2}$')
plt.xlabel('n')
plt.ylabel(r'Roughness Exponent $\zeta$')
plt.title('Roughness Exponent vs Anharmonicity Exponent n')
plt.legend()
plt.grid(True)
plt.savefig('zeta_vs_n.png', dpi=150)
plt.close()

# 3) Plot 2: Residuals for both estimates
plt.figure(figsize=(10, 6))
residuals_avg = zetaarray_np - zeta_theoretical
residuals_mean = zeta_mean_np - zeta_theoretical
plt.plot(narray_np, residuals_avg,
         marker='o', linestyle='-', color='steelblue',
         label=r'$\zeta$ from $\langle S(q) \rangle$')
plt.errorbar(narray_np, residuals_mean, yerr=zeta_std_np,
             marker='s', linestyle='--', color='darkorange', capsize=3,
             label=r'$\langle \zeta \rangle$ from individual $S(q)$')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('n')
plt.ylabel(r'Residual ($\zeta - \zeta_s$)')
plt.title('Residuals: Numerical vs Theoretical Prediction')
plt.legend()
plt.grid(True)
plt.savefig('zeta_residuals.png', dpi=150)
plt.close()

# 4) Plot 3: Direct comparison of the two numerical estimates
plt.figure(figsize=(10, 6))
plt.errorbar(narray_np, zeta_mean_np - zetaarray_np, yerr=zeta_std_np,
             marker='o', linestyle='-', color='purple', capsize=3)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('n')
plt.ylabel(r'$\langle \zeta \rangle_{\rm samples} - \zeta[\langle S(q) \rangle]$')
plt.title('Difference between the two estimators')
plt.grid(True)
plt.savefig('zeta_comparison.png', dpi=150)
plt.close()


# 5) Plot 4: S(q) vs q panel for selected n values
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, n_val in zip(axes, n_plot_values):
    sqpos_avg                    = sq_avg_store[n_val]
    zeta_avg, zeta_mean, B, A    = fit_store[n_val]
    zeta_th                      = (4.*n_val - 1) / (4.*n_val - 2)

    # Full averaged S(q) curve
    ax.loglog(qpos, sqpos_avg, color='steelblue', alpha=0.6,
              linewidth=0.8, label=r'$\langle S(q) \rangle$')

    # Power-law fit line extended slightly beyond fit region
    q_fit_range = qpos[fit_mask]
    q_ext       = np.logspace(np.log10(q_fit_range.min()),
                              np.log10(q_fit_range.max() * 2.0), 200)
    sq_fit      = np.exp(A) * q_ext**B
    ax.loglog(q_ext, sq_fit, color='red', linewidth=2, linestyle='--',
              label=(rf'Fit $\langle S(q)\rangle$: $\zeta={zeta_avg:.3f}$' + '\n' +
                     rf'$\langle\zeta\rangle_\mathrm{{samples}}={zeta_mean:.3f}$' + '\n' +
                     rf'Theory: $\zeta_s={zeta_th:.3f}$'))

    # Shade the fit region
    ax.axvspan(q_fit_range.min(), q_fit_range.max(),
               alpha=0.12, color='green', label='Fit region')

    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$\langle S(q) \rangle$')
    ax.set_title(rf'$n = {n_val}$')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(r'Structure factor $\langle S(q) \rangle$ vs $q$ — power-law fits', fontsize=13)
plt.tight_layout()
plt.savefig('sq_panel.png', dpi=150)
plt.close()

print("Execution complete. Data saved to zeta_results.txt and plots saved as PNG files.")
