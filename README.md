# Anharmonic Larkin Model (ALM)

We consider a one-dimensional elastic interface described by a displacement field $u_i$ on a periodic lattice of size $L$.

The local slope is

$$s_i = u_{i+1} - u_i.$$

## Hamiltonian

The Anharmonic Larkin Model is defined by

$$H[u] =
\sum_{i=0}^{L-1}
\left[
\frac{c}{2}s_i^2
+
\frac{1}{2n}|s_i|^{2n}
-
f_i u_i
\right],$$

where:
- $c \ge 0$ is the harmonic elastic constant
- $n > 1$ controls anharmonicity
- $f_i$ is a quenched random force

The disorder is Gaussian:

$$
f_i \sim \mathcal{N}(0, \Delta),
$$

with the constraint

$$
\sum_i f_i = 0.
$$

---

## Equilibrium condition

Define the elastic energy density

$$
W(s) = \frac{c}{2}s^2 + \frac{1}{2n}|s|^{2n}.
$$

The stress is

$$
\sigma_i = \frac{dW}{ds_i}
= c s_i + |s_i|^{2n-2}s_i.
$$

Since

$$
s_i = u_{i+1} - u_i,
$$

variation of the Hamiltonian yields the force balance equation

$$
\sigma_i - \sigma_{i-1} = f_i.
$$

Equivalently,

$$
\sigma_{i+1} - \sigma_i = f_i.
$$

---

## Integrated form

Summing the force balance gives

$$
\sigma_i = C + \sum_{j=0}^{i-1} f_j,
$$

where $C$ is an integration constant.

Define

$$
F_i = \sum_{j=0}^{i-1} f_j.
$$

Then

$$
\sigma_i = F_i + C.
$$

---

## Constitutive relation

The slope satisfies

$$
c s_i + |s_i|^{2n-2}s_i = F_i + C.
$$

For $c = 0$, this becomes explicit:

$$
s_i = \mathrm{sign}(F_i + C)\, |F_i + C|^{\frac{1}{2n-1}}.
$$

---

## Periodicity constraint

Periodic boundary conditions require

$$
u_L = u_0.
$$

Thus

$$
\sum_{i=0}^{L-1} s_i = 0.
$$

Define

$$
G(C) = \sum_{i=0}^{L-1} s(F_i + C).
$$

The equilibrium condition is

$$
G(C) = 0.
$$

---

## Monotonicity and uniqueness

Since the elasticity is convex,

$$
\frac{d\sigma}{ds} > 0
\quad \Rightarrow \quad
\frac{ds}{d\sigma} > 0,
$$

we have

$$
G'(C) = \sum_i \frac{ds}{d\sigma}(F_i + C) > 0.
$$

Thus $G(C)$ is strictly monotonic and has a unique root.

---

## Ground state construction

1. Generate disorder $f_i \sim \mathcal{N}(0,\Delta)$  
2. Enforce zero mean:
   $$
   f_i \leftarrow f_i - \frac{1}{L}\sum_j f_j
   $$
3. Compute cumulative force:
   $$
   F_i = \sum_{j<i} f_j
   $$
4. Solve:
   $$
   \sum_i s(F_i + C) = 0
   $$
5. Compute slopes:
   $$
   s_i = s(F_i + C)
   $$
6. Reconstruct interface:
   $$
   u_i = \sum_{j<i} s_j
   $$
7. Remove mean height:
   $$
   u_i \leftarrow u_i - \frac{1}{L}\sum_j u_j
   $$

---

## Summary

The $L$-dimensional minimization problem reduces exactly to a single scalar root-finding problem for $C$, due to convexity and the linear structure of the quenched random-force disorder.
