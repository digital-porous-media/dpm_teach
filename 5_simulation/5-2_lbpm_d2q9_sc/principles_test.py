import numpy as np
import matplotlib.pyplot as plt
from lbm_d2q9_scmp import LBM_SCMP


def calculate_delta_pressure(rho_field, pressure_field):
    nx, ny = rho_field.shape
    rho_gas = np.sum(rho_field[:4, :4]) / 16.
    rho_liq = np.sum(rho_field[nx // 2 - 2: nx // 2 + 2, ny // 2 - 2: ny // 2 + 2]) / 16.
    press_gas = np.sum(pressure_field[:4, :4]) / 16.
    press_liq = np.sum(pressure_field[nx // 2 - 2: nx // 2 + 2, ny // 2 - 2: ny // 2 + 2]) / 16.

    delta_p = press_liq - press_gas
    rho_av = 0.5 * (rho_liq + rho_gas)
    
    print(f"Densities: {rho_gas = :.4f}, {rho_liq = :.4f}")
    print(f"Pressures: p_g = {press_gas :.6f}, p_l = {press_liq :.6f}")
    print(f"Pressure difference: {delta_p = :.6f}")
    

def compute_pressure(sim):
    """
    Shan-Chen EOS pressure: p = cs2 * rho + 0.5 * G * cs2 * psi^2
    Assumes sim has attributes: rho, psi, cs2, G
    """
    cs2 = getattr(sim, 'cs2', 1/3)
    G = getattr(sim, 'G', -4.0)
    # psi = sim.compute_pseudopotential(sim.rho)
    return cs2 * sim.rho + 0.5 * G * cs2 * (sim.psi ** 2)

def initialize_circular_bubble(nx, ny, R, rho_liq=1.6, rho_gas=0.1, center=None):
    """
    Create a density field with a circular bubble of radius R.
    Inside bubble: rho_gas; outside: rho_liq
    """
    if center is None:
        cx, cy = nx // 2, ny // 2
    else:
        cx, cy = center

    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    rho = np.where(r <= R, rho_liq, rho_gas)
    u = np.zeros((2, nx, ny))
    return rho, u

def initialize_elliptical_bubble(nx, ny, rx, ry, rho_liq=1.6, rho_gas=0.1, center=None):
    """
    Create a density field with an elliptical bubble of semi-axes rx and ry.
    Inside bubble: rho_gas; outside: rho_liq
    """
    if center is None:
        cx, cy = nx // 2, ny // 2
    else:
        cx, cy = center

    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    
    # Prevent division by zero for very small radii
    rx = max(rx, 1e-9)
    ry = max(ry, 1e-9)
    
    # Ellipse equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
    ellipse_eq = ((x - cx) / rx)**2 + ((y - cy) / ry)**2
    
    # Note: The original function had rho_liq inside the bubble.
    # This implementation places rho_gas inside the bubble and rho_liq outside.
    rho = np.where(ellipse_eq <= 1, rho_gas, rho_liq)
    u = np.zeros((2, nx, ny))
    return rho, u

def has_converged(prev_rho, rho, tol=1e-10):
    """Check convergence by L2 change in rho."""
    diff = np.linalg.norm((rho - prev_rho).ravel()) / (np.linalg.norm(rho.ravel()) + 1e-15)
    return diff < tol

def relax_to_equilibrium(sim, max_steps=5000, check_every=500, tol=1e-10, verbose=True):
    """
    Advance the simulation until the density field converges or max_steps reached.
    """
    prev_rho = sim.rho.copy()
    if max_steps < check_every:
        check_every = max_steps
    for step in range(0, max_steps + 1):
        sim.step(apply_body_force=False)
        if step % check_every == 0:
            if verbose:
                print(f"  Step {step}:")
                calculate_delta_pressure(sim.rho, compute_pressure(sim))
            if step > 0 and has_converged(prev_rho, sim.rho, tol=tol):
                if verbose:
                    print(f"  ✓ Converged at step {step}")
                return step
            prev_rho = sim.rho.copy()
    if verbose:
        print("  ⚠ Reached max_steps without strict convergence")
    return max_steps


def run_laplace_ift_validation(
    nx=128, ny=128,
    radius=15.,
    omega=1.0, cs2=1/3, G=-4.0,
    rho_liq=1.6, rho_gas=0.1,
    max_steps=100, check_every=500, tol=1e-10,
):
    """
    Laplace-law IFT validation:
      - Create bubbles of various radii R
      - Relax to equilibrium
      - Measure ΔP across interface and effective radius R_eff
      - Fit ΔP vs 1/R_eff to estimate σ (interfacial tension)

    Returns:
      results: dict with keys:
        'sigma_est', 'intercept', 'r2', 'invR', 'R_eff', 'deltaP', 'radii', 'snapshots'
    """

    print("\n" + "="*70)
    print("IFT VALIDATION VIA LAPLACE LAW (ΔP = σ * 1/R)")
    print("="*70)
    print(f"Domain: {nx} x {ny}, omega={omega}, cs2={cs2}, G={G}, rho_liq={rho_liq}, rho_gas={rho_gas}")
    print(f"Radii to test: {radius}")
    
    print("\n" + "-"*70)
    print(f"Setting up bubble with nominal radius R = {radius} (lattice units)")
    # Create simulator
    sim = LBM_SCMP(nx=nx, ny=ny, omega=omega, cs2=cs2, G=G, rho_0=1.0)

    # Initialize circular bubble
    # rho_init, u_init = initialize_circular_bubble(nx, ny, radius, rho_liq=rho_liq, rho_gas=rho_gas)
    rho_init, u_init = initialize_elliptical_bubble(nx, ny, radius * 0.75, radius * 1.5, rho_liq=rho_liq, rho_gas=rho_gas)
    sim.initialize_density_field(rho_init, u_init)

    # No solids
    sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))

    # Relax
    _ = relax_to_equilibrium(sim, max_steps=max_steps, check_every=check_every, tol=tol, verbose=True)

    # Optional: plot last snapshot fields
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(sim.rho.T, origin='lower', cmap='viridis')
    axs[0].set_title('Density ρ')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(compute_pressure(sim).T, origin='lower', cmap='plasma')
    axs[1].set_title('Pressure p')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    # run_laplace_ift_validation(
    #     nx=64, ny=64,
    #     radius=15.,
    #     omega=1, cs2=1/3, G=-4.7,
    #     rho_liq=2.1, rho_gas=0.15, max_steps=10000, check_every=100
    # )
    
    run_laplace_ift_validation(
        nx=64, ny=64,
        radius=10.,
        omega=1, cs2=1/3, G=-4.7,
        rho_liq=2.1, rho_gas=0.15, max_steps=10000, check_every=100
    )


    
    
    
    
    