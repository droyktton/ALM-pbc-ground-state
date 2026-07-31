from kpz_quenched_steady_state import sample_kpz_steady_state

# generate one sample
x, h, v = sample_kpz_steady_state(L=50.0, lam=1.0, D=0.05, N=4000, seed=0)

# x: grid points (N,) array over [0, L)
# h: steady-state height profile h(x), periodic, mean-subtracted
# v: growth velocity (dh/dt in the long-time limit)

import matplotlib.pyplot as plt
plt.plot(x, h)
plt.xlabel("x"); plt.ylabel("h(x)")
plt.show()


## many samples
#samples = [sample_kpz_steady_state(L=50, lam=1, D=0.05, N=4000, seed=s)[1] 
#           for s in range(100)]
