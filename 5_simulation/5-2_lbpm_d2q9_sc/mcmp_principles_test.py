import numpy as np
import matplotlib.pyplot as plt
from lbm_d2q9_mcmp import LBM_MCMP


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


def kruger_validation(
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

    nx, ny = 64, 64
    n_comp = 2
    omega = [1.0, 1.0]
    G_mat = np.array([[0.0, 6.0],
                      [6.0, 0.0]])
    # G_ads = [1, -1]
    # psi_solid = [0.5, 0.5]

    # No solids
    solid_mask = np.zeros((nx, ny), dtype=bool)

    # Semicircular droplet touching bottom
    rho_init = np.zeros((2, nx, ny))
    rho_init[0, :, :ny // 2] = 0.1
    rho_init[1, :, :ny // 2] = 0.9

    rho_init[0, :, ny // 2:] = 0.9
    rho_init[1, :, ny // 2:] = 0.1


    print("\n" + "="*70)
    print("MCMP Validation Test with Timm Kruger's Textbook Example")
    print("="*70)
    print(f"Domain: {nx} x {ny}, omega={omega}, cs2={cs2}, G={G_mat[1, 0]}, rho_liq={rho_liq}, rho_gas={rho_gas}")

    print("\n" + "-"*70)
    solver = LBM_MCMP(nx, ny, n_comp=n_comp, omega=omega, G_mat=G_mat, rho0=[1.0, 1.0])
    solver.initialize_density_field(rho_init)
    solver.set_solids_from_mask(solid_mask)

    # Relax
    _ = relax_to_equilibrium(solver, max_steps=max_steps, check_every=check_every, tol=tol, verbose=True)

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
        omega=1, cs2=1 / 3, G=-4.7,
        rho_liq=2.5, rho_gas=0.15, max_steps=1000, check_every=100
    )
