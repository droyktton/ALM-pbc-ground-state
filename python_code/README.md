# ALM Ground State with Periodic Boundary Conditions

Numerical computation of the ground state of the **anharmonic Larkin model (ALM)** on a 1D lattice with periodic boundary conditions (PBC), and extraction of the roughness exponent ζ as a function of the anharmonicity exponent *n*.

---

## Physical Background

The Larkin model describes a 1D elastic line (e.g. a domain wall or contact line) subject to quenched random disorder. In the anharmonic generalization, the local elastic energy density scales as |s|^(2n) rather than the harmonic s², where s is the local slope. The ground-state displacement field u(x) is obtained by minimizing the total energy subject to a PBC constraint (zero net slope).

The roughness exponent ζ characterizes how displacement fluctuations scale with system size L:

```
⟨u²⟩ ~ L^(2ζ)
```

Equivalently, the structure factor (power spectrum) scales as:

```
S(q) = |FFT(u)|² ~ q^(-(2ζ+1))
```

The theoretical prediction for the anharmonic Larkin model is:

```
ζ_s = (4n − 1) / (4n − 2)
```

This code verifies that prediction numerically across a range of *n* values and compares two independent estimators of ζ.

---

## Method

### Ground-State Construction

For each disorder realization, the ground state is found analytically (not by iterative minimization):

1. **Disorder**: draw Gaussian random forces f_i with variance Δ, zero mean enforced explicitly.
2. **Cumulative force**: F_i = Σ_{j<i} f_j, the integrated disorder.
3. **Constitutive inversion**: at each site, the local stress σ = F + C determines the slope s via the nonlinear elastic law c·s + sign(s)|s|^(2n−1) = σ. For c = 0 (pure anharmonic), the inversion is analytic: s = sign(σ)|σ|^(1/(2n−1)). For c > 0, Brent's method is used.
4. **PBC constraint**: the constant C (uniform stress shift / Lagrange multiplier) is found by requiring Σ s_i = 0, again via Brent's method.
5. **Displacement**: u_i = Σ_{j<i} s_j, with mean subtracted.

### Roughness Exponent Estimation

Two estimators of ζ are computed from the structure factor S(q):

| Estimator | Method |
|-----------|--------|
| `zeta_avg_sq` | Average S(q) over all samples, then fit log S vs log q at small q |
| `zeta_mean` ± `zeta_std` | Fit ζ independently for each sample, then average |

If the two estimators agree, the system is **self-averaging**. Systematic disagreement signals rare-sample effects dominating the averaged spectrum.

The power-law fit uses frequencies q < 0.01 (the infrared scaling regime). The slope B of log S vs log q gives ζ = −(B + 1) / 2.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 8192 | System size (number of sites) |
| `Delta` | 1.0 | Disorder variance |
| `c` | 0 | Harmonic elastic constant (0 = pure anharmonic) |
| `nsamples` | 50 | Number of disorder realizations per value of *n* |
| `n` range | 2 – 50 | Anharmonicity exponent values |
| `n_plot_values` | [2, 4, 8, 16] | Values of *n* for the S(q) panel plot |

---

## Requirements

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

```bash
python alm_sim.py
```

Runtime is dominated by the Brent's-method inversions inside the seed loop. For `c = 0`, the analytic branch is used and the code is significantly faster. Expect several minutes for the full sweep n = 2…50 with 50 samples at L = 8192.

---

## Output Files

### Data

| File | Contents |
|------|----------|
| `zeta_results.txt` | Four columns: `n`, `zeta_from_avg_sq`, `zeta_mean_samples`, `zeta_std_samples` |

### Plots

| File | Description |
|------|-------------|
| `zeta_vs_n.png` | Both ζ estimators vs *n*, overlaid with theoretical prediction |
| `zeta_residuals.png` | Residuals ζ − ζ_s for both estimators |
| `zeta_comparison.png` | Difference ⟨ζ⟩_samples − ζ[⟨S(q)⟩] directly, showing estimator agreement |
| `sq_panel.png` | 2×2 panel: averaged S(q) vs q for n = 2, 4, 8, 16, with power-law fit and shaded fit region |

---

## Notes on Fit Quality

The `sq_panel.png` plot is useful for assessing the reliability of the extracted exponents. In particular:

- At small *n* (e.g. n = 2), ζ is large and the power-law regime is broad and clean.
- At large *n* (e.g. n = 16), ζ → 1 and the spectrum is steep; the q < 0.01 cutoff may approach the finite-size limit, potentially biasing the fit. Inspect the shaded fit region against the full S(q) curve to check.
- Any curvature of S(q) within the fit region indicates corrections to scaling that the simple linear fit does not capture.

---

## Reference

The theoretical exponent ζ_s = (4n−1)/(4n−2) is derived in the context of the anharmonic Larkin model with periodic boundary conditions. See the repository for the associated paper or notes.
