# Anharmonic Larkin Model (ALM)

We consider a one-dimensional elastic interface described by a displacement
field \(u_i\) defined on a periodic lattice of size \(L\). The local slope is

$$
s_i = u_{i+1}-u_i .
$$

The Anharmonic Larkin Model (ALM) is defined by the Hamiltonian

$$
H[u]
=
\sum_{i=0}^{L-1}
\left[
\frac{c}{2}s_i^2
+
\frac{1}{2n}|s_i|^{2n}
-
f_i u_i
\right],
$$

where

- \(c \ge 0\) is the harmonic elastic coefficient,
- \(n>1\) controls the anharmonic elasticity,
- \(f_i\) is a quenched random force,
- periodic boundary conditions are imposed.

The random forces are Gaussian,

$$
f_i \sim \mathcal N(0,\Delta),
$$

and satisfy

$$
\sum_i f_i = 0,
$$

which removes the zero mode required by periodic boundary conditions.

---

# Equilibrium condition

The elastic energy depends only on the local slope,

$$
W(s)
=
\frac{c}{2}s^2
+
\frac{1}{2n}|s|^{2n}.
$$

Its derivative defines the local stress,

$$
\sigma_i
=
\frac{dW}{ds_i}
=
c\,s_i
+
|s_i|^{2n-2}s_i.
$$

To obtain the equilibrium equations, vary the Hamiltonian with respect to
\(u_i\).

Since

$$
s_i=u_{i+1}-u_i,
$$

the variation of the elastic part gives

$$
\sigma_i-\sigma_{i-1},
$$

while the disorder contributes

$$
-f_i.
$$

The stationarity condition

$$
\frac{\partial H}{\partial u_i}=0
$$

therefore yields

$$
\sigma_i-\sigma_{i-1}=f_i.
$$

Equivalently,

$$
\sigma_{i+1}-\sigma_i=f_i.
$$

This is the discrete force-balance equation.

---

# Integrated form

The equilibrium equation can be integrated once:

$$
\sigma_i
=
C+\sum_{j=0}^{i-1}f_j,
$$

where \(C\) is an integration constant.

Define the cumulative random force

$$
F_i
=
\sum_{j=0}^{i-1}f_j.
$$

Then

$$
\sigma_i = F_i + C.
$$

The entire disorder dependence enters only through the random walk \(F_i\).

---

# Constitutive inversion

The local slope is obtained by inverting

$$
c\,s_i
+
|s_i|^{2n-2}s_i
=
F_i+C.
$$

For general \(c>0\), this equation is solved independently at each site.

For the pure anharmonic case \(c=0\), the inversion is explicit:

$$
s_i
=
\operatorname{sign}(F_i+C)
\,|F_i+C|^{1/(2n-1)}.
$$

---

# Periodicity constraint

Periodic boundary conditions imply

$$
u_L=u_0.
$$

Since

$$
u_L-u_0
=
\sum_{i=0}^{L-1}s_i,
$$

the slopes must satisfy

$$
\sum_{i=0}^{L-1}s_i=0.
$$

Substituting the constitutive relation gives

$$
G(C)
=
\sum_{i=0}^{L-1}
s(F_i+C)
=
0.
$$

Because the elastic energy is strictly convex,

$$
\frac{d\sigma}{ds}>0,
$$

and therefore

$$
\frac{ds}{d\sigma}>0.
$$

Consequently,

$$
G'(C)
=
\sum_i
\frac{ds}{d\sigma}(F_i+C)
>0.
$$

Hence \(G(C)\) is strictly monotonic and possesses a unique root.

The equilibrium configuration is therefore uniquely determined by a single
scalar parameter \(C\).

---

# Exact construction of the ground state

The ground state is obtained through the following steps:

1. Generate the random forces \(f_i\).
2. Enforce zero mean:
   $$
   f_i \leftarrow f_i-\frac1L\sum_j f_j.
   $$
3. Compute the cumulative force:
   $$
   F_i=\sum_{j<i}f_j.
   $$
4. Find the unique root \(C\) of
   $$
   \sum_i s(F_i+C)=0.
   $$
5. Compute the slopes:
   $$
   s_i=s(F_i+C).
   $$
6. Reconstruct the interface:
   $$
   u_i=\sum_{j<i}s_j.
   $$
7. Remove the arbitrary mean displacement:
   $$
   u_i \leftarrow u_i-\frac1L\sum_j u_j.
   $$

The original \(L\)-dimensional minimization problem is thus reduced to a
one-dimensional root-finding problem for the integration constant \(C\).
