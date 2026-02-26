import numpy as np
import matplotlib.pyplot as plt
from lbm_d2q9_mcmp import LBM_MCMP
from tqdm import tqdm
# --- Model ---
nx, ny = 150, 150
lbm = LBM_MCMP(
    nx=nx, ny=ny, n_comp=2,
    omega=[1.0, 1.0],
    cs2=1.0/3.0,
    G_mat=np.array([[0.0, 3.0],
                    [3.0, 0.0]], dtype=np.float64),  # moderate demixing
    rho0=[1.0, 1.0],
    G_ads=[0.0, 0.0],      # no walls
    psi_solid=[0.0, 0.0],
    body_force=[0.0, 0.0]
)

# --- Two droplets (all arrays in shape (nx, ny)) ---
R = 24
sep = 3 * R
x1, y1 = nx//2 - sep//2, ny//2
x2, y2 = nx//2 + sep//2, ny//2

# Circle masks via broadcasting (avoid meshgrid transposes)
x = np.arange(nx)[:, None]    # (nx, 1)
y = np.arange(ny)[None, :]    # (1, ny)
mask1 = ((x - x1)**2 + (y - y1)**2) <= R**2
mask2 = ((x - x2)**2 + (y - y2)**2) <= R**2
maskA = mask1 | mask2

# Densities
rho = np.zeros((2, nx, ny), dtype=np.float64)
rho[0][maskA] = 1.0
rho[1][maskA] = 1e-3
rho[0][~maskA] = 1e-3
rho[1][~maskA] = 1.0

# Initial velocities (simple: uniform inside droplets)
U0 = 0.01
u_init = np.zeros((2, nx, ny), dtype=np.float64)
u_init[0][mask1] = +U0
u_init[0][mask2] = -U0

lbm.initialize_density_field(rho_init=rho, u_init=u_init)

# --- Run and plot every 1000 steps ---
TOTAL_STEPS = 20000
SNAP_EVERY = 1000

for step in tqdm(range(1, TOTAL_STEPS + 1)):
    lbm.step()
    if step % SNAP_EVERY == 0:
        # Phase field (transpose only for plotting)
        # phiT = lbm.compute_phase_field().T
        rhoA_T = lbm.rho[0].T
        plt.figure(figsize=(9, 4))
        plt.imshow(rhoA_T, origin="lower", extent=[0, nx, 0, ny],
                   cmap="RdBu_r", interpolation="nearest")
        speed = np.hypot(lbm.u[0], lbm.u[1])
        umax = float(np.max(speed))
        plt.title(f"Two-droplet collision | t={lbm.timestep} | max|u|={umax:.2e}")
        plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar(label="φ")
        plt.tight_layout()
plt.show()