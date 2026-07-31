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
narray = []
zetaarray = []

L = 8192
Delta = 1.0
c = 0
nsamples = 50

for n in range(2, 51):
    print(f"Processing n={n}...")
    sq = np.zeros(L)
    invalid = 0
    for seed in range(nsamples):
        u, periodicity = generate_sample(L, Delta, n, c, seed)
        if abs(periodicity) < 1e-7:
            sq += np.abs(np.fft.fft(u))**2
        else:
            invalid += 1

    if (nsamples - invalid) > 0:
        sq /= (nsamples - invalid)
    
    q = np.fft.fftfreq(L, d=1)
    qpos = q[q > 0]
    sqpos = sq[q > 0]

    # Fit for small q
    fit_indices = qpos < 0.01
    q_fit = qpos[fit_indices]
    sq_fit = sqpos[fit_indices]
    
    log_q = np.log(q_fit)
    log_sq = np.log(sq_fit)
    coeffs = np.polyfit(log_q, log_sq, 1)
    B = coeffs[0]
    zeta = -(B + 1) / 2.0
    
    narray.append(n)
    zetaarray.append(zeta)

narray_np = np.array(narray)
zetaarray_np = np.array(zetaarray)

# 1) Save with high numerical precision (18 decimal places)
data_to_save = np.column_stack((narray_np, zetaarray_np))
np.savetxt('zeta_results.txt', data_to_save, header='n zeta', fmt=['%d', '%.18e'], comments='')

# 2) Save Plots as PNG files
# Plot 1: Exponent vs n
plt.figure(figsize=(10, 6))
plt.plot(narray_np, zetaarray_np, marker='o', linestyle='-', label='Numerical $\zeta$')
zeta_theoretical = (4. * narray_np - 1) / (4. * narray_np - 2)
plt.plot(narray_np, zeta_theoretical, '--', color='red', label=r'Theoretical $\zeta_s = \frac{4n-1}{4n-2}$')
plt.xlabel('n')
plt.ylabel('Roughness Exponent $\zeta$')
plt.title('Roughness Exponent vs Anharmonicity Exponent n')
plt.legend()
plt.grid(True)
plt.savefig('zeta_vs_n.png')
plt.close()

# Plot 2: Residuals
plt.figure(figsize=(10, 6))
residuals = zetaarray_np - zeta_theoretical
plt.plot(narray_np, residuals, marker='s', linestyle='-', color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('n')
plt.ylabel('Residual ($\zeta - \zeta_s$)')
plt.title('Residuals: Numerical vs Theoretical Prediction')
plt.grid(True)
plt.savefig('zeta_residuals.png')
plt.close()

print("Execution complete. Data saved to zeta_results.txt and plots saved as PNG files.")
