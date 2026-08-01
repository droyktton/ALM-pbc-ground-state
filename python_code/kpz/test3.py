import matplotlib.pyplot as plt
import numpy as np
from kpz_structure_factor import structure_factor_samples

# 1. Compute the structure factor
q, S_mean, S_sem = structure_factor_samples(
    L=50.0, lam=1.0, D=1.0, N=1024, n_samples=300, seed0=0
)

# 2. Keep only q > 0 (S(q) is symmetric, and q=0 is undefined on a log scale)
mask = q > 0
q_pos, S_mean_pos, S_sem_pos = q[mask], S_mean[mask], S_sem[mask]

# 3. Export data to a file
# Stack arrays as columns
data_to_save = np.column_stack((q_pos, S_mean_pos, S_sem_pos))

# Save to a tab-delimited text file
np.savetxt(
    "kpz_structure_factor_data.txt",
    data_to_save,
    delimiter="\t",
    header="q\tS_mean\tS_sem",
    comments="",
)
print("Data successfully saved to 'kpz_structure_factor_data.txt'")

# 4. Plot the results
plt.plot(q_pos, S_mean_pos, "o", ms=3)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("q")
plt.ylabel("S(q)")
plt.show()
