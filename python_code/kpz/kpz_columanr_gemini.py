import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def solve_quenched_kpz_steady_state(N=1000, L=100.0, lam=1.0, D=0.1, seed=None):
    """
    Solves the steady-state height profile of the 1D Quenched KPZ equation
    via Cole-Hopf linearization and Perron-Frobenius top eigenvector recovery.

    Parameters:
    -----------
    N : int
        Number of spatial discretization points.
    L : float
        System size (periodic length).
    lam : float
        Nonlinear coupling strength (lambda).
    D : float
        Noise strength <eta(x) eta(x')> = 2D delta(x-x').
    seed : int, optional
        Random seed for reproducible quenched disorder.

    Returns:
    --------
    x : ndarray (N,)
        Spatial coordinates.
    h_steady : ndarray (N,)
        Steady-state height profile (up to an arbitrary additive constant).
    v_growth : float
        Global steady growth velocity v = E_0 / lambda.
    phi_0 : ndarray (N,)
        Top eigenvector of the linear operator L.
    """
    if seed is not None:
        np.random.seed(seed)

    dx = L / N
    x = np.linspace(0, L - dx, N)

    # 1. Generate Quenched White Noise Potential
    # <eta_i eta_j> = (2D / dx) * delta_ij
    eta = np.random.normal(loc=0.0, scale=np.sqrt(2.0 * D / dx), size=N)
    V = lam * eta  # Potential term in L = d_x^2 + lambda * eta(x)

    # 2. Construct Discrete Laplacian with Periodic BCs
    main_diag = -2.0 * np.ones(N) / (dx**2)
    off_diag = np.ones(N - 1) / (dx**2)

    # Sparse tridiagonal matrix
    L_matrix = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N), format='csr')

    # Periodic boundary terms (top-right and bottom-left corners)
    L_matrix[0, N - 1] += 1.0 / (dx**2)
    L_matrix[N - 1, 0] += 1.0 / (dx**2)

    # Add diagonal potential V(x)
    L_matrix += sp.diags(V, 0, format='csr')

    # 3. Solve for Largest Algebraic Eigenvalue (E_0) and Eigenvector (phi_0)
    # Using 'LA' (Largest Algebraic)
    eigenvalues, eigenvectors = spla.eigsh(L_matrix, k=1, which='LA')
    
    E_0 = eigenvalues[0]
    phi_0 = eigenvectors[:, 0]

    # 4. Enforce Positive Sign (Perron-Frobenius guarantees all entries have same sign)
    if np.mean(phi_0) < 0:
        phi_0 = -phi_0

    # Small floor to prevent log(0) in case of extreme localization/underflow
    phi_0 = np.clip(phi_0, a_min=np.finfo(float).tiny, a_max=None)

    # 5. Map back to Height Profile and Velocity
    h_steady = (1.0 / lam) * np.log(phi_0)
    
    # Zero-center height profile for clean representation
    h_steady -= np.mean(h_steady)
    
    v_growth = E_0 / lam

    return x, h_steady, v_growth, phi_0

# ==========================================
# Example Execution & Visualization
# ==========================================
if __name__ == "__main__":
    N = 2000
    L = 10.0
    lam = 1.0
    D = 0.5

    x, h_steady, v_growth, phi_0 = solve_quenched_kpz_steady_state(
        N=N, L=L, lam=lam, D=D, seed=42
    )

    print(f"Computed Steady Growth Velocity (v = E_0/lambda): {v_growth:.6f}")

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Plot Ground State Eigenfunction
    ax[0].plot(x, phi_0, color='tab:red', lw=1.5)
    ax[0].set_ylabel(r"Top Mode $\phi_0(x)$")
    ax[0].set_title("Perron-Frobenius Ground State & Steady Height Profile")
    ax[0].grid(True, alpha=0.3)

    # Plot Height Profile
    ax[1].plot(x, h_steady, color='tab:blue', lw=1.5)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"$h_{\mathrm{steady}}(x)$")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
