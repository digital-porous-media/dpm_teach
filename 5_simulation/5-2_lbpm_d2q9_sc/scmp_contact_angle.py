"""
Shan-Chen Contact Angle Test Script

Author(s): AI Assistant
Last Update: Oct. 2023

This script demonstrates how the fluid-solid interaction parameter, G_ads,
controls the contact angle of a fluid droplet on a solid surface in the
Shan-Chen single-component multiphase (SCMP) model.

To run this script:
1. Navigate to the root directory of the project.
2. Execute the following command:
   python 5_simulation/5-2_lbpm_d2q9_sc/contact_angle_test.py

Key Concepts:
- G_ads > 0: The fluid is repelled by the solid surface (hydrophobic),
             leading to a contact angle > 90°.
- G_ads = 0: The surface is neutral, leading to a contact angle of ~90°.
- G_ads < 0: The fluid is attracted to the solid surface (hydrophilic),
             leading to a contact angle < 90°.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the directory containing the LBM module to the Python path
# This assumes the script is run from the root of the dpm_teach repository
sys.path.append('5_simulation/5-2_lbpm_d2q9_sc')

from lbm_d2q9_scmp import LBM_SCMP

def make_tanh_droplet(nx, ny, center_x, center_y, R, rho_l, rho_g, W):
    """
    Build a diffuse (tanh) circular droplet in a (nx, ny) lattice.
    Returns rho_init of shape (nx, ny).
    """
    # Coordinate grids matching your field layout (first axis = x, second = y)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    # Radial distance from droplet center
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

    # Tanh profile
    rho = rho_g + 0.5 * (rho_l - rho_g) * (1.0 - np.tanh((r - R) / W))

    # Optional: tiny floor to avoid division-by-zero (keep it very small)
    rho = np.clip(rho, 1e-6, None)
    return rho

def run_contact_angle_test(G_ads_value, max_steps=5000):
    """
    Initializes and runs a contact angle simulation.

    Parameters:
    ---
    G_ads_value : float
        The fluid-solid interaction strength.
    max_steps : int
        Number of simulation steps to run.
    """
    print(f"--- Running Contact Angle Test with G_ads = {G_ads_value} ---")

    # 1. Simulation Parameters
    nx, ny = 200, 100  # Grid dimensions
    omega = 1.0          # Relaxation time
    G = -5.2  # Fluid-fluid interaction (for phase separation)

    # 2. Define Initial and Boundary Conditions
    # Create the solid mask (a flat wall at the bottom)
    solid_mask = np.zeros((nx, ny), dtype=bool)
    solid_mask[:, 0] = True   # Wall at y=0 (bottom)
    # solid_mask[:, -1] = True  # Wall at y=ny-1 (top) to prevent wrapping
    # solid_mask[0, :] = True   # Wall at x=0 (left)
    # solid_mask[-1, :] = True  # Wall at x=nx-1 (right)

    # Initial density profile: a semi-circular droplet
    rho_init = np.full((nx, ny), 0.15)  # Background low-density phase
    center_x, center_y = nx // 2, 0
    radius = 25
    y, x = np.ogrid[:ny, :nx]
    mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
    rho_init[mask.T] = 2.15  # High-density droplet
    # rho_init[solid_mask] = 1.0

    # nx, ny = 200, 100
    # center_x, center_y = nx // 2, 2      # bottom wall contact
    # R = 25
    # rho_l, rho_g = 2.15, 0.15               # start with milder densities
    # W = 4                                 # interface half-width (3–5 recommended)

    # rho_init = make_tanh_droplet(nx, ny, center_x, center_y, R, rho_l, rho_g, W)

    # 3. Initialize and Run the LBM Simulation
    lbm = LBM_SCMP(nx=nx, ny=ny, omega=omega, G=G, G_ads=G_ads_value, rho_0=1, body_force=[1.0E-4, 0.0])
    lbm.initialize_density_field(rho_init)
    lbm.set_solids_from_mask(solid_mask)

    # Run the simulation
    for step in tqdm(range(max_steps)):
        lbm.step()
        # lbm.rho[solid_mask] = 1.0
        # if (step + 1) % 1 == 0:
        #     print(f'Step {step + 1}/{max_steps}')

    print('Simulation finished.')

    # 4. Visualize the Final Equilibrium State
    plt.figure(figsize=(10, 5))
    plt.imshow(lbm.rho.T, origin='lower', cmap='viridis', vmin=0, vmax=2)
    plt.title(f'Final Density Profile (Equilibrium) with G_ads = {G_ads_value}')
    plt.colorbar(label='Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    # --- EXPERIMENT HERE ---
    # Change the value of G_ads to see its effect on wettability.
    # - G_ads = -3.7  (Hydrophilic / Wetting, angle < 90°)
    # - G_ads =  -3.2  (Neutral, angle ≈ 90°)
    # - G_ads =  -2.0  (Hydrophobic / Non-wetting, angle > 90°)

    G_ads_to_test = -6

    run_contact_angle_test(G_ads_value=G_ads_to_test, max_steps=5000)
