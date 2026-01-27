import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use your existing module
sys.path.append('5_simulation/5-2_lbpm_d2q9_sc')
from lbm_d2q9_scmp import LBM_SCMP

def make_tanh_slab(nx, ny, x0, rho_l, rho_g, W):
    """
    Create a left liquid / right gas slab with a smooth tanh interface at x=x0.
    Returns rho_init of shape (nx, ny).
    """
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    # Left side (X < x0) ~ rho_l, right side ~ rho_g
    rho = rho_g + 0.5 * (rho_l - rho_g) * (1.0 + np.tanh((x0 - X) / W))
    return np.clip(rho, 1e-6, None)

def run_capillary_imbibition(G_ads_value=-3.0, max_steps=6000):
    """
    Capillary imbibition in a slit tube using your LBM_SCMP workflow.
    Shows the final density field.
    """
    print(f"--- Capillary Imbibition (slit tube) with G_ads = {G_ads_value} ---")

    # Domain and parameters
    nx, ny = 400, 64            # long tube, moderate height
    omega = 1.0                 # tau = 1.0
    G = -5.2                    # single-component SC attraction
    rho_l, rho_g = 2.2, 0.15    # liquid/gas densities
    W = 4                       # interface half-width
    x0 = 100                    # initial interface location

    # Slit tube walls: top and bottom solid
    solid_mask = np.zeros((nx, ny), dtype=bool)
    solid_mask[:, 0]  = True    # bottom wall
    solid_mask[:, -1] = True    # top wall
    # solid_mask[0, :]  = True    # left wall
    # solid_mask[-1, :] = True    # right wall
    # Initial slab: left reservoir (liquid), right channel (gas)
    rho_init = make_tanh_slab(nx, ny, x0, rho_l, rho_g, W)

    # Initialize solver
    lbm = LBM_SCMP(nx=nx, ny=ny, omega=omega, G=G, G_ads=G_ads_value, rho_0=1)
    # If your class supports an explicit wall pseudopotential knob:
    # lbm.psi_wall = 0.6
    lbm.initialize_density_field(rho_init)
    lbm.set_solids_from_mask(solid_mask)

    # Run
    for step in tqdm(range(max_steps)):
        lbm.step(apply_body_force=False)

    print('Simulation finished.')

    # Show final phase configuration (density field)
    plt.figure(figsize=(12, 4))
    plt.imshow(lbm.rho.T, origin='lower', cmap='viridis', vmin=rho_g, vmax=rho_l)
    plt.colorbar(label='Density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Capillary Imbibition in Slit Tube (Final Density), G_ads = {G_ads_value}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Try a wetting case; adjust G_ads for contact angle control
    run_capillary_imbibition(G_ads_value=-8.0, max_steps=1000)