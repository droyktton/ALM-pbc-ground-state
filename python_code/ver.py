import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# --- 1. Load Data ---
# Replace this with: df = pd.read_csv("your_file.csv")
df = pd.read_csv("scan_results.csv")

# --- 2. Fit zeta(n) from W2_direct ~ L^(2 * zeta(n)) ---
# Model: log(W2) = 2 * zeta * log(L) + const  =>  slope = 2 * zeta  =>  zeta = slope / 2

fitted_zeta = []

for n, group in df.groupby("n"):
    log_L = np.log(group["L"])
    log_W2 = np.log(group["W2_direct"])

    # Linear fit on log-log scale
    slope, intercept, r_value, p_value, std_err = linregress(log_L, log_W2)

    zeta_fit = slope / 2.0
    zeta_err = (std_err / 2.0)  # Error propagation for slope/2

    fitted_zeta.append({"n": n, "zeta_fit": zeta_fit, "zeta_fit_err": zeta_err})

df_fit = pd.DataFrame(fitted_zeta)

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(9, 6))

# Plot zeta_s(n, L) vs n for each fixed L
for L, group in df.groupby("L"):
    group_sorted = group.sort_values("n")
    ax.errorbar(
        group_sorted["n"],
        group_sorted["zeta_s"],
        yerr=group_sorted["zeta_s_err"],
        marker="o",
        linestyle="--",
        alpha=0.7,
        capsize=3,
	label=rf"$\zeta_s(n, L={L})$",  # ✅ Fixed
    )

# Overlay fitted global zeta(n) vs n
ax.errorbar(
    df_fit["n"],
    df_fit["zeta_fit"],
    yerr=df_fit["zeta_fit_err"],
    marker="S",
    color="black",
    linewidth=2.5,
    markersize=8,
    capsize=5,
    label=r"Global Fit $\zeta(n)$ (from $W^2 \sim L^{2\zeta}$)",
)

# Formatting
ax.set_xlabel("Parameter $n$", fontsize=12)
ax.set_ylabel(r"Roughness Exponent $\zeta$", fontsize=12)
ax.set_title(
    r"Comparison of Global Fit $\zeta(n)$ and Finite-Size Exponents $\zeta_s(n, L)$ vs $n$",
    fontsize=13,
    fontweight="bold",
)
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
ax.grid(True, ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("zeta_vs_n.png", dpi=300)
plt.show()

# Print fitted results table
print("Fitted Global Zeta Values:")
print(df_fit.to_string(index=False))
