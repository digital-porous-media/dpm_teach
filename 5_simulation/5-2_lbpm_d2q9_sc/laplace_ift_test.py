import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from lbm_d2q9_scmp import LBM_SCMP

def compute_pressure(sim):
    """
    Shan-Chen EOS pressure: p = cs2 * rho + 0.5 * G * cs2 * psi^2
    Assumes sim has attributes: rho, psi, cs2, G
    """
    cs2 = getattr(sim, 'cs2', 1/3)
    G = getattr(sim, 'G', -4.0)
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

def classify_inside_outside_by_threshold(rho, thresh=None):
    """
    Classify inside/outside bubble by density threshold.
    Use midpoint of min/max rho if no threshold provided.
    """
    if thresh is None:
        rho_min, rho_max = np.min(rho), np.max(rho)
        thresh = 0.5 * (rho_min + rho_max)
    inside = rho < thresh  # gas is lower density in SCMP
    outside = rho >= thresh
    return inside, outside

def erode_mask(mask, n_iter=2):
    """Erode a boolean mask to avoid sampling in the diffuse interface."""
    if np.any(mask):
        return ndimage.binary_erosion(mask, iterations=n_iter)
    return mask

def has_converged(prev_rho, rho, tol=1e-10):
    """Check convergence by L2 change in rho."""
    diff = np.linalg.norm((rho - prev_rho).ravel()) / (np.linalg.norm(rho.ravel()) + 1e-15)
    return diff < tol

def relax_to_equilibrium(sim, max_steps=5000, check_every=500, tol=1e-10, verbose=True):
    """
    Advance the simulation until the density field converges or max_steps reached.
    """
    prev_rho = sim.rho.copy()
    for step in range(1, max_steps + 1):
        sim.step(apply_body_force=False)
        if step % check_every == 0:
            if verbose:
                print(f"  Step {step}: checking convergence...")
            if has_converged(prev_rho, sim.rho, tol=tol):
                if verbose:
                    print(f"  ✓ Converged at step {step}")
                return step
            prev_rho = sim.rho.copy()
    if verbose:
        print("  ⚠ Reached max_steps without strict convergence")
    return max_steps

def measure_pressure_jump_and_effective_radius(sim, rho_liq, rho_gas, erosion_pixels=3):
    """
    Measure ΔP = P_inside - P_outside and effective radius R after equilibrium.
    Uses density threshold classification and erosion to avoid interface.
    """
    p = compute_pressure(sim)
    inside, outside = classify_inside_outside_by_threshold(sim.rho, thresh=(rho_liq + rho_gas) / 2)
    inside_core = erode_mask(inside, n_iter=erosion_pixels)
    outside_core = erode_mask(outside, n_iter=erosion_pixels)

    # If erosion removed everything (very small bubble), fallback to original masks
    if np.sum(inside_core) < 10:
        inside_core = inside
    if np.sum(outside_core) < 10:
        outside_core = outside

    P_in = np.mean(p[inside_core]) if np.any(inside_core) else np.mean(p[inside])
    P_out = np.mean(p[outside_core]) if np.any(outside_core) else np.mean(p[outside])
    deltaP = P_in - P_out

    # Effective radius via area
    A = np.sum(inside)  # lattice units; dx = 1
    R_eff = np.sqrt(A / np.pi)

    return deltaP, R_eff, p, inside, outside

def run_laplace_ift_validation(
    nx=128, ny=128,
    radii=(8, 12, 16, 20, 24),
    omega=1.0, cs2=1/3, G=-4.0,
    rho_liq=1.6, rho_gas=0.1,
    erosion_pixels=3,
    max_steps=100, check_every=500, tol=1e-10,
    seed=0, save_fig=True
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
    np.random.seed(seed)
    deltaPs = []
    invRs = []
    R_effs = []
    snapshots = []

    print("\n" + "="*70)
    print("IFT VALIDATION VIA LAPLACE LAW (ΔP = σ * 1/R)")
    print("="*70)
    print(f"Domain: {nx} x {ny}, omega={omega}, cs2={cs2}, G={G}, rho_liq={rho_liq}, rho_gas={rho_gas}")
    print(f"Radii to test: {radii}")

    for R in radii:
        print("\n" + "-"*70)
        print(f"Setting up bubble with nominal radius R = {R} (lattice units)")
        # Create simulator
        sim = LBM_SCMP(nx=nx, ny=ny, omega=omega, cs2=cs2, G=G, rho_0=1.0)

        # Initialize circular bubble
        rho_init, u_init = initialize_circular_bubble(nx, ny, R, rho_liq=rho_liq, rho_gas=rho_gas)
        sim.initialize_density_field(rho_init, u_init)

        # No solids
        sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))

        # Relax
        print("Relaxing to equilibrium...")
        steps = relax_to_equilibrium(sim, max_steps=max_steps, check_every=check_every, tol=tol, verbose=True)
        print(f"Equilibration finished in {steps} steps")

        # Measure ΔP and R_eff
        deltaP, R_eff, p, inside, outside = measure_pressure_jump_and_effective_radius(sim, rho_liq, rho_gas, erosion_pixels=erosion_pixels)
        invR = 1.0 / R_eff if R_eff > 0 else np.nan

        print(f"Measured ΔP = {deltaP:.6f}, R_eff = {R_eff:.3f}, 1/R_eff = {invR:.6f}")

        deltaPs.append(deltaP)
        invRs.append(invR)
        R_effs.append(R_eff)

        snapshots.append({
            'R_nominal': R,
            'R_eff': R_eff,
            'deltaP': deltaP,
            'invR': invR,
            'rho': sim.rho.copy(),
            'psi': sim.psi.copy(),
            'p': p.copy(),
            'inside_mask': inside.copy(),
            'outside_mask': outside.copy(),
            'steps': steps,
            'params': {'omega': omega, 'cs2': cs2, 'G': G, 'rho_liq': rho_liq, 'rho_gas': rho_gas}
        })

    # Convert to arrays
    invRs = np.array(invRs)
    deltaPs = np.array(deltaPs)
    R_effs = np.array(R_effs)

    # Filter out invalid entries (e.g., nan radii)
    valid = np.isfinite(invRs) & np.isfinite(deltaPs)
    if not np.all(valid):
        print("⚠ Filtering out invalid measurements where radius or ΔP is NaN.")
    invRs_valid = invRs[valid]
    deltaPs_valid = deltaPs[valid]

    # Fit ΔP = σ * (1/R) + b
    if len(invRs_valid) >= 2:
        coeffs = np.polyfit(invRs_valid, deltaPs_valid, 1)
        sigma_est, intercept = coeffs[0], coeffs[1]
        deltaP_pred = sigma_est * invRs_valid + intercept
        ss_res = np.sum((deltaPs_valid - deltaP_pred)**2)
        ss_tot = np.sum((deltaPs_valid - np.mean(deltaPs_valid))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print("\n" + "="*70)
        print("FIT RESULTS: ΔP vs 1/R_eff")
        print("="*70)
        print(f"Estimated σ (slope) = {sigma_est:.6f}")
        print(f"Intercept b = {intercept:.6f}")
        print(f"R² = {r2:.4f}")
    else:
        print("⚠ Not enough valid points to perform a linear fit.")
        sigma_est, intercept, r2 = np.nan, np.nan, np.nan
        deltaP_pred = np.array([])

    # Plot ΔP vs 1/R with fit
    if save_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(invRs_valid, deltaPs_valid, color='b', label='Measurements', zorder=3)
        if len(invRs_valid) >= 2:
            x_line = np.linspace(np.min(invRs_valid), np.max(invRs_valid), 200)
            y_line = sigma_est * x_line + intercept
            ax.plot(x_line, y_line, 'r--', label=f'Fit: ΔP = σ*(1/R) + b\nσ={sigma_est:.4f}, R²={r2:.3f}')
        ax.set_xlabel('1 / R_eff')
        ax.set_ylabel('ΔP (inside - outside)')
        ax.set_title('Laplace IFT Validation (ΔP vs 1/R_eff)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig('ift_laplace_validation.png', dpi=150)
        print("Saved: ift_laplace_validation.png")
        plt.close()

        # Optional: plot last snapshot fields
        if len(snapshots) > 0:
            snap = snapshots[-1]
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            im0 = axs[0].imshow(snap['rho'].T, origin='lower', cmap='viridis')
            axs[0].set_title('Density ρ')
            plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            im1 = axs[1].imshow(snap['p'].T, origin='lower', cmap='plasma')
            axs[1].set_title('Pressure p')
            plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            im2 = axs[2].imshow(snap['inside_mask'].T, origin='lower', cmap='gray')
            axs[2].set_title('Inside (bubble) mask')
            plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            plt.suptitle(f"Snapshot at R_nom={snap['R_nominal']}, R_eff={snap['R_eff']:.2f}, ΔP={snap['deltaP']:.4f}")
            plt.tight_layout()
            plt.savefig('ift_laplace_snapshot.png', dpi=150)
            print("Saved: ift_laplace_snapshot.png")
            plt.close()

    results = {
        'sigma_est': float(sigma_est),
        'intercept': float(intercept),
        'r2': float(r2),
        'invR': invRs.tolist(),
        'R_eff': R_effs.tolist(),
        'deltaP': deltaPs.tolist(),
        'radii': list(radii),
        'snapshots': snapshots
    }
    return results

if __name__ == "__main__":
    np.random.seed(0)
    results = run_laplace_ift_validation(
        nx=64, ny=64,
        radii=(4, 8, 16, 24),
        omega=1, cs2=1/3, G=-4.7,
        rho_liq=2.1, rho_gas=0.15, max_steps=1000,
        save_fig=True
    )
    print("\nEstimated sigma:", results['sigma_est'])