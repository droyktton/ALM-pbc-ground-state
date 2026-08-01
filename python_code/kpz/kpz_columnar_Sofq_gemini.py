import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def generate_steady_state(N, L, lam, D):
    """
    Generates a single steady-state height profile h_steady(x)
    for a realization of quenched noise.
    """
    dx = L / N

    # 1. Quenched Noise Potential
    eta = np.random.normal(loc=0.0, scale=np.sqrt(2.0 * D / dx), size=N)
    V = lam * eta

    # 2. Discrete Laplacian with Periodic BCs
    main_diag = -2.0 * np.ones(N) / (dx**2)
    off_diag = np.ones(N - 1) / (dx**2)

    L_matrix = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N), format='csr')
    L_matrix[0, N - 1] += 1.0 / (dx**2)
    L_matrix[N - 1, 0] += 1.0 / (dx**2)
    L_matrix += sp.diags(V, 0, format='csr')

    # 3. Ground state eigenvector via Shift-Invert / Lanczos
    eigenvalues, eigenvectors = spla.eigsh(L_matrix, k=1, which='LA')
    phi_0 = eigenvectors[:, 0]

    if np.mean(phi_0) < 0:
        phi_0 = -phi_0

    # Prevent log(0) underflow
    phi_0 = np.clip(phi_0, a_min=np.finfo(float).tiny, a_max=None)

    # Height profile mapped back
    h_steady = (1.0 / lam) * np.log(phi_0)
    h_steady -= np.mean(h_steady)  # Zero-mean subtraction

    return h_steady


def compute_average_structure_factor(N=2000, L=200.0, lam=1.0, D=0.5, num_realizations=50):
    """
    Computes ensemble-averaged structure factor S(q) = (1/N) <|h~(q)|^2>
    across multiple realizations of quenched disorder.
    """
    dx = L / N
    # Wavevectors for Real FFT: q = 2pi * k / L
    q = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)
    
    # Accumulate power spectrum sum
    S_q_sum = np.zeros(len(q), dtype=np.float64)

    for i in range(num_realizations):
        h = generate_steady_state(N, L, lam, D)
        
        # Real Discrete Fourier Transform: h_q = dx * sum_j h_j e^{-i q x_j}
        h_q = np.fft.rfft(h) * dx
        
        # Power Spectrum: S(q) = |h_q|^2 / L
        S_q_sum += (np.abs(h_q)**2) / L

    # Ensemble Average
    S_q_avg = S_q_sum / num_realizations

    return q, S_q_avg


# ==========================================
# Run & Plot
# ==========================================
if __name__ == "__main__":
    N = 2000
    L = 100.0
    lam = 1.0
    D = 0.5
    num_realizations = 100

    q, Sq = compute_average_structure_factor(
        N=N, L=L, lam=lam, D=D, num_realizations=num_realizations
    )

    # Exclude q = 0 (DC component)
    q_fit = q[1:]
    Sq_fit = Sq[1:]

    # Visualizing S(q)
    plt.figure(figsize=(8, 5))
    plt.loglog(q_fit, Sq_fit, 'o-', color='navy', ms=3, alpha=0.7, label=rf"Ensemble ($M={num_realizations}$)")
    
    # Theoretical Guide Line S(q) ~ q^(-2) for reference (Edwards-Wilkinson scaling)
    plt.loglog(q_fit, 0.5 * (q_fit**-2), '--', color='tab:red', label=r"$q^{-2}$ reference")

    plt.xlabel(r"Wavevector $q$", fontsize=12)
    plt.ylabel(r"Structure Factor $S(q)$", fontsize=12)
    plt.title(r"Ensemble Steady-State Structure Factor $S(q) = \frac{1}{L} \langle |\hat{h}(q)|^2 \rangle$", fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
