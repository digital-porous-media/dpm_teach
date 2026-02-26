import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lbm_d2q9_mcmp import LBM_MCMP

nx, ny = 128, 128
n_comp = 2
omega = [1.0, 1.0]
cs2 = 1.0 / 3.0
G_mat = np.array([[0.0, 3.0],
                  [3.0, 0.0]])
rho0_ref = [1.0, 1.0]

# Present-phase lattice densities (encode mass-density ratio)
rho0_present = 1.0   # heavy (e.g., water)
rho1_present = 0.5   # light (e.g., oil)
rho_absent = 0.0



# Initialize two layers: heavy on top (RT-unstable)
x = np.arange(nx)[:, None]
y = np.arange(ny)[None, :]
mid = ny//2
# upper = y < mid
# lower = y >= mid

rho_init = np.zeros((n_comp, nx, ny))
rho_init[0, :, :ny // 4] = rho0_present   # component 0 (heavy) upper half
# # rho_init[1][upper] = rho_absent
# # rho_init[0][lower] = rho_absent
rho_init[1, :, ny // 4:] = rho1_present   # component 1 (light) lower half
for i in range(nx):
    interface = ny//4 + int(4 * np.sin(2 * np.pi * i / nx))
    rho_init[0, i, :interface] = 1.0
    rho_init[1, i, interface:] = 0.5
# # Small sinusoidal perturbation at interface
# amp = 0.3
# x_coords = np.arange(nx)
# rho_init[0, ny // 4, :] += amp * np.sin(2 * np.pi * x_coords / nx)
# rho_init[1, ny // 4, :] -= amp * np.sin(2 * np.pi * x_coords / nx)

# Gravity downward
g_lat = np.array([0.0, -1e-4])

solver = LBM_MCMP(nx, ny, n_comp=n_comp, omega=omega, cs2=cs2,
                  G_mat=G_mat, rho0=rho0_ref, body_force=g_lat)
solver.initialize_density_field(rho_init)

# Periodic lateral boundaries; bounce-back at top/bottom optional
# solid_mask = np.zeros((nx, ny), dtype=bool)
# Walls are neutral (no wetting preference)
G_ads = [0.0, 0.0]
solid_mask = np.zeros((nx, ny), dtype=bool)
solid_mask[0] = True
solid_mask[-1] = True
solver.set_solids_from_mask(solid_mask)

for t in tqdm(range(2_000)):
    solver.step()
    if t % 500 == 0:
        phi = solver.compute_phase_field()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        im0 = ax[0].imshow(phi.T, origin='upper', cmap='RdBu_r', vmin=-1, vmax=1)
        ax[0].set_title("Phase Field")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(solver.rho[0].T, origin='upper', cmap='RdBu_r')
        plt.colorbar(im1, ax=ax[1], label='ρ')
        ax[1].set_title("Density Field: Component 0 (heavy)")

        im2 = ax[2].imshow(solver.rho[1].T, origin='upper', cmap='RdBu_r')
        ax[2].set_title("Density Field: Component 1 (light)")

        plt.colorbar(im2, ax=ax[2], label='ρ')
        plt.tight_layout()
        plt.show()
# Visualize results
phi = solver.compute_phase_field()

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
im0 = ax[0].imshow(phi.T, origin='upper', cmap='RdBu_r', vmin=-1, vmax=1)
ax[0].set_title("Phase Field")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

plt.colorbar(im1, ax=ax[1], label='ρ')
im1 = ax[1].imshow(solver.rho[0].T, origin='upper', cmap='RdBu_r')
ax[1].set_title("Density Field: Component 0 (heavy)")

im2 = ax[2].imshow(solver.rho[1].T, origin='upper', cmap='RdBu_r')
ax[2].set_title("Density Field: Component 1 (light)")

plt.colorbar(im2, ax=ax[2], label='ρ')
plt.tight_layout()
plt.show()
