import matplotlib.pyplot as plt
from kpz_structure_factor import structure_factor_samples

q, S_mean, S_sem = structure_factor_samples(L=50.0, lam=1.0, D=0.05,
                                             N=2000, n_samples=300, seed0=0)

# keep only q > 0 (S(q) is symmetric, and q=0 is undefined on a log scale)
mask = q > 0
q_pos, S_pos = q[mask], S_mean[mask]

plt.plot(q_pos, S_pos, 'o', ms=3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('q')
plt.ylabel('S(q)')
plt.show()
