import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the directory containing the LBM module to the Python path
# This assumes the script is run from the root of the dpm_teach repository
sys.path.append('5_simulation/5-2_lbpm_d2q9_sc')

from lbm_d2q9_mcmp import LBM_MCMP

# Context: Python 3 / NumPy / Matplotlib / PoreSpy
import porespy as ps

# --- Parameters ---
nx, ny = 100, 100   # solver arrays use shape (nx, ny)
n_comp = 2
omega = [1.0, 1.0]  # tau = 1.0
cs2 = 1.0 / 3.0
G_mat = np.array([[0.0, 3.0],
                  [3.0, 0.0]], dtype=float)
rho0_ref = [1.0, 1.0]
G_ads = [-0.4, +0.4]      # component-0 wets, component-1 dewets
psi_solid = [0.5, 0.5]
body_force = np.array([1e-4, 0.0])  # gentle drive in +x

# --- Generate porous medium (blobs) using PoreSpy ---
porosity_target = 0.75
blobiness = 0.5
seed = 42

# PoreSpy returns image with shape (ny, nx) for 2D; True means pore
pores_im = ps.generators.blobs(shape=[ny, nx],
                               porosity=porosity_target,
                               blobiness=blobiness,
                               seed=seed)

# Align with solver's (nx, ny) and invert to solid mask
# pores_im: True = pore; solid_mask: True = solid
solid_mask = (~pores_im).T  # transpose to (nx, ny)
plt.imshow(solid_mask)
plt.show()

# Optional: clear solids near left/right to encourage percolation paths
clear = 4
solid_mask[:clear, :] = False
solid_mask[-clear:, :] = False

# --- Initial fluid configuration ---
rho_init = np.zeros((n_comp, nx, ny), dtype=np.float64)
pore = ~solid_mask

# Wetting fluid 0 fills pores
rho_init[0][pore] = 1.0
rho_init[1][pore] = 0.05

# Inlet (left) reservoir stripe
inlet_w = 6
inlet = np.zeros((nx, ny), dtype=bool)
inlet[:inlet_w, :] = True
inlet &= pore
rho1_in, rho0_in = 1.0, 0.05
rho_init[1][inlet] = rho1_in
rho_init[0][inlet] = rho0_in

# Outlet (right) reservoir stripe
outlet_w = 6
outlet = np.zeros((nx, ny), dtype=bool)
outlet[-outlet_w:, :] = True
outlet &= pore
rho1_out, rho0_out = 0.05, 1.0

# --- Instantiate solver ---
solver = LBM_MCMP(nx, ny, n_comp=n_comp, omega=omega, cs2=cs2,
                  G_mat=G_mat, rho0=rho0_ref, G_ads=G_ads,
                  psi_solid=psi_solid, body_force=body_force)

solver.initialize_density_field(rho_init)
solver.set_solids_from_mask(solid_mask)

# --- Helper: enforce reservoirs by resetting rho & f to equilibrium on stripes ---
def enforce_reservoir(mask, rho_pair):
    # rho_pair: (rho0_target, rho1_target)
    rho_target = np.zeros_like(solver.rho)
    rho_target[0][mask] = rho_pair[0]
    rho_target[1][mask] = rho_pair[1]

    rho_new = solver.rho.copy()
    rho_new[:, mask] = rho_target[:, mask]

    # zero velocity in reservoir (can set small ux if desired)
    u_zero = np.zeros_like(solver.u)
    f_eq_res = solver.compute_equilibrium(rho_new, u_zero)

    solver.rho[:, mask] = rho_new[:, mask]
    solver.f[:, :, mask] = f_eq_res[:, :, mask]

# --- Diagnostics ---
def saturation_nonwet():
    pore_local = ~solver.solid_mask
    rho_tot = solver.rho.sum(axis=0)
    mask = pore_local & (rho_tot > 1e-12)
    return solver.rho[1][mask].sum() / rho_tot[mask].sum()

def fractional_flow(x_star):
    rho0_x = solver.rho[0][x_star, :]
    rho1_x = solver.rho[1][x_star, :]
    ux_x = solver.u[0][x_star, :]
    Q0 = np.sum(rho0_x * ux_x)
    Q1 = np.sum(rho1_x * ux_x)
    denom = max(Q0 + Q1, 1e-16)
    return Q1 / denom

# --- Time stepping ---
n_steps = 10000
mid_x = nx // 2

for t in tqdm(range(n_steps)):
    # Enforce inlet/outlet each step (simple open boundaries)
    enforce_reservoir(inlet, (rho0_in, rho1_in))
    enforce_reservoir(outlet, (rho0_out, rho1_out))

    solver.step()

    # if t % 600 == 0:
    #     S_nw = saturation_nonwet()
    #     F_nw = fractional_flow(mid_x)
    #     umax = np.amax(np.abs(solver.u))
    #     print(f"t={t}, S_nw={S_nw:.4f}, F_nw={F_nw:.4f}, max|u|={umax:.3e}")

# --- Final fields ---
rho0 = solver.rho[0]
rho1 = solver.rho[1]
phi = solver.compute_phase_field()          # (nx, ny)
u_mag = np.sqrt(solver.u[0]**2 + solver.u[1]**2)

# --- Plot configuration (overlay solids) ---
def plot_field(ax, arr, title, cmap="viridis", vmin=None, vmax=None):
    # Transpose to display x horizontally, y vertically
    im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    # Overlay solids as hatch-like alpha mask
    ax.imshow(solid_mask.T, origin='lower', cmap='gray', alpha=0.25)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    return im

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
im0 = plot_field(axs[0, 0], rho0, r"rho0 (wetting)", cmap="Blues")
im1 = plot_field(axs[0, 1], rho1, r"rho1 (non-wetting)", cmap="Reds")
im2 = plot_field(axs[1, 0], phi, r"phase field phi", cmap="RdBu_r", vmin=-1, vmax=1)
im3 = plot_field(axs[1,1], u_mag, r"|u| (velocity magnitude)", cmap="magma")

fig.colorbar(im0, ax=axs[0,0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axs[0,1], fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=axs[1,0], fraction=0.046, pad=0.04)
fig.colorbar(im3, ax=axs[1,1], fraction=0.046, pad=0.04)
fig.suptitle("Porous-Medium Drainage with PoreSpy blobs", fontsize=14)
plt.tight_layout()
plt.show()