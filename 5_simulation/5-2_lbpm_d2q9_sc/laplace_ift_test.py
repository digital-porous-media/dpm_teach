import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from lbm_d2q9_scmp import LBM_SCMP

# -----------------------------
# Utilities & EOS pressure
# -----------------------------
def compute_pressure_eos(sim):
    """
    Shan–Chen EOS pressure:
      p_EOS = cs2 * rho + 0.5 * G * cs2 * psi^2
    """
    cs2 = getattr(sim, 'cs2', 1.0 / 3.0)
    G = getattr(sim, 'G', -4.0)
    rho, _ = sim.compute_macroscopic(sim.f)
    return cs2 * rho + 0.5 * G * cs2 * (sim.psi**2)

def initialize_circular_bubble(nx, ny, R, rho_liq=1.6, rho_gas=0.1, center=None):
    """
    Create a density field with a circular LIQUID droplet (high-density) of radius R.
    Inside: rho_liq; outside: rho_gas.
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
    Classify inside/outside by density threshold.
    Here: "inside" = high-density (liquid droplet).
    """
    if thresh is None:
        rho_min, rho_max = np.min(rho), np.max(rho)
        thresh = 0.5 * (rho_min + rho_max)
    inside = rho >= thresh
    outside = rho < thresh
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
    for step in tqdm(range(1, max_steps + 1), desc="Relaxing to equilibrium"):
        sim.step()
        if step % check_every == 0:
            if has_converged(prev_rho, sim.rho, tol=tol):
                if verbose:
                    print(f"  ✓ Converged at step {step}")
                return step
            prev_rho = sim.rho.copy()
    if verbose:
        print("  ⚠ Reached max_steps without strict convergence")
    return max_steps

# -----------------------------
# Mechanical pressure tensor & normal stress
# -----------------------------
# D2Q9 velocities and weights
E = np.array([
    [0, 0],
    [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
], dtype=int)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

def compute_pressure_tensor(sim):
    """
    SCMP Shan–Chen pressure tensor:
      P_ab = cs2 * rho * delta_ab
             + 0.5 * G * sum_i w_i * psi(x) * psi(x+e_i) * e_{i,a} * e_{i,b}
    Returns P_xx, P_xy, P_yy (arrays nx x ny).
    """
    cs2 = getattr(sim, 'cs2', 1.0 / 3.0)
    G = getattr(sim, 'G', -4.0)
    rho = sim.rho
    psi = sim.psi

    P_xx = cs2 * rho
    P_yy = cs2 * rho
    P_xy = np.zeros_like(rho)

    # exclude rest direction i=0
    for i in range(1, 9):
        ex, ey = E[i]
        w = W[i]
        psi_shift = np.roll(np.roll(psi, ex, axis=0), ey, axis=1)
        contrib = 0.5 * G * w * psi * psi_shift
        P_xx += contrib * (ex * ex)
        P_xy += contrib * (ex * ey)
        P_yy += contrib * (ey * ey)

    return P_xx, P_xy, P_yy

def compute_interface_normal(psi, eps=1e-12):
    """
    Normal from grad psi using central differences.
    Returns n_x, n_y, and |grad psi|.
    """
    gx = 0.5 * (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0))
    gy = 0.5 * (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1))
    mag = np.sqrt(gx * gx + gy * gy) + eps
    nx = gx / mag
    ny = gy / mag
    return nx, ny, mag

def compute_normal_stress(P_xx, P_xy, P_yy, nx, ny):
    """
    p_n = n^T P n = nx^2 * P_xx + 2 nx ny * P_xy + ny^2 * P_yy
    """
    return (nx * nx) * P_xx + 2.0 * (nx * ny) * P_xy + (ny * ny) * P_yy

# -----------------------------
# Measurement helpers
# -----------------------------
def measure_pressure_jump_eos_and_radius(sim, rho_liq, rho_gas, erosion_pixels=3):
    """
    Measures ΔP_EOS = p_EOS(in) - p_EOS(out) and effective radius.
    """
    p_eos = compute_pressure_eos(sim)
    inside, outside = classify_inside_outside_by_threshold(sim.rho, thresh=(rho_liq + rho_gas) / 2)
    inside_core = erode_mask(inside, n_iter=erosion_pixels)
    outside_core = erode_mask(outside, n_iter=erosion_pixels)
    if np.sum(inside_core) < 10:
        inside_core = inside
    if np.sum(outside_core) < 10:
        outside_core = outside

    P_in = np.mean(p_eos[inside_core])
    P_out = np.mean(p_eos[outside_core])
    deltaP = P_in - P_out

    A = np.sum(inside)
    R_eff = np.sqrt(A / np.pi)
    return deltaP, R_eff, p_eos, inside, outside

def measure_pressure_jump_mech(sim, rho_liq, rho_gas, erosion_pixels=3):
    """
    Measures ΔP_mech = p_n(in) - p_n(out) using normal stress from the pressure tensor.
    """
    P_xx, P_xy, P_yy = compute_pressure_tensor(sim)
    nx_vec, ny_vec, _ = compute_interface_normal(sim.psi)
    p_n = compute_normal_stress(P_xx, P_xy, P_yy, nx_vec, ny_vec)

    inside, outside = classify_inside_outside_by_threshold(sim.rho, thresh=(rho_liq + rho_gas) / 2)
    inside_core = erode_mask(inside, n_iter=erosion_pixels)
    outside_core = erode_mask(outside, n_iter=erosion_pixels)
    if np.sum(inside_core) < 10: inside_core = inside
    if np.sum(outside_core) < 10: outside_core = outside

    p_in = np.mean(p_n[inside_core])
    p_out = np.mean(p_n[outside_core])
    deltaP_mech = p_in - p_out

    A = np.sum(inside)
    R_eff = np.sqrt(A / np.pi)
    return deltaP_mech, R_eff, p_n

# -----------------------------
# Runner: Laplace IFT validation (mechanical)
# -----------------------------
def run_laplace_ift_validation(
    nx=128, ny=128,
    radii=(8, 12, 16, 20, 24),
    omega=1.0, cs2=1/3, G=-4.0, rho_0=1.0,
    rho_liq=1.6, rho_gas=0.1,
    erosion_pixels=3,
    max_steps=5000, check_every=500, tol=1e-10,
    seed=0, save_fig=True
):
    """
    Laplace-law IFT validation:
      - Create liquid droplets of various radii R
      - Relax to equilibrium
      - Measure ΔP_EOS and ΔP_mech across interface
      - Fit ΔP_mech vs 1/R_eff to estimate σ (interfacial tension)
    """
    np.random.seed(seed)
    deltaPs_eos, deltaPs_mech, invRs, R_effs = [], [], [], []
    snapshots = []

    print("\n" + "="*70)
    print("IFT VALIDATION VIA LAPLACE LAW (ΔP_mech = σ * 1/R)")
    print("="*70)
    print(f"Domain: {nx} x {ny}, omega={omega}, cs2={cs2}, G={G}, rho_0={rho_0}, rho_liq={rho_liq}, rho_gas={rho_gas}")
    print(f"Radii to test: {radii}")

    for R in radii:
        print("\n" + "-"*70)
        print(f"Setting up droplet with nominal radius R = {R} (lattice units)")
        sim = LBM_SCMP(nx=nx, ny=ny, omega=omega, cs2=cs2, G=G, rho_0=rho_0)

        rho_init, u_init = initialize_circular_bubble(nx, ny, R, rho_liq=rho_liq, rho_gas=rho_gas)
        sim.initialize_density_field(rho_init, u_init)
        sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))

        print("Relaxing to equilibrium...")
        steps = relax_to_equilibrium(sim, max_steps=max_steps, check_every=check_every, tol=tol, verbose=True)
        print(f"Equilibration finished in {steps} steps")

        # Measure EOS and mechanical pressure jumps
        deltaP_eos, R_eff_eos, p_eos, inside, outside = measure_pressure_jump_eos_and_radius(sim, rho_liq, rho_gas, erosion_pixels=erosion_pixels)
        deltaP_mech, R_eff_mech, p_n = measure_pressure_jump_mech(sim, rho_liq, rho_gas, erosion_pixels=erosion_pixels)
        assert abs(R_eff_eos - R_eff_mech) < 1e-6
        R_eff = R_eff_mech
        invR = 1.0 / R_eff if R_eff > 0 else np.nan

        print(f"Measured (EOS)  ΔP = {deltaP_eos:.6f}")
        print(f"Measured (MECH) ΔP = {deltaP_mech:.6f}, R_eff = {R_eff:.3f}, 1/R_eff = {invR:.6f}")

        deltaPs_eos.append(deltaP_eos)
        deltaPs_mech.append(deltaP_mech)
        invRs.append(invR)
        R_effs.append(R_eff)

        snapshots.append({
            'R_nominal': R,
            'R_eff': R_eff,
            'deltaP_eos': deltaP_eos,
            'deltaP_mech': deltaP_mech,
            'invR': invR,
            'rho': sim.rho.copy(),
            'psi': sim.psi.copy(),
            'p_eos': p_eos.copy(),
            'p_n': p_n.copy(),
            'inside_mask': inside.copy(),
            'outside_mask': outside.copy(),
            'steps': steps,
            'params': {'omega': omega, 'cs2': cs2, 'G': G, 'rho_liq': rho_liq, 'rho_gas': rho_gas}
        })

    invRs = np.array(invRs)
    deltaPs_mech = np.array(deltaPs_mech)
    deltaPs_eos = np.array(deltaPs_eos)
    R_effs = np.array(R_effs)

    valid = np.isfinite(invRs) & np.isfinite(deltaPs_mech)
    invRs_valid = invRs[valid]
    deltaPs_mech_valid = deltaPs_mech[valid]

    # Fit ΔP_mech = σ * (1/R) + b
    if len(invRs_valid) >= 2:
        coeffs = np.polyfit(invRs_valid, deltaPs_mech_valid, 1)
        sigma_est, intercept = coeffs[0], coeffs[1]
        deltaP_pred = sigma_est * invRs_valid + intercept
        ss_res = np.sum((deltaPs_mech_valid - deltaP_pred)**2)
        ss_tot = np.sum((deltaPs_mech_valid - np.mean(deltaPs_mech_valid))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print("\n" + "="*70)
        print("FIT RESULTS: ΔP_mech vs 1/R_eff")
        print("="*70)
        print(f"Estimated σ (slope) = {sigma_est:.6f}")
        print(f"Intercept b = {intercept:.6f}")
        print(f"R² = {r2:.4f}")
    else:
        print("⚠ Not enough valid points to perform a mechanical fit.")
        sigma_est, intercept, r2 = np.nan, np.nan, np.nan

    # Plot mechanical vs EOS ΔP
    if save_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(invRs, deltaPs_mech, color='b', label='ΔP_mech (normal stress)', zorder=3)
        ax.scatter(invRs, deltaPs_eos, color='g', marker='x', label='ΔP_EOS (scalar)', zorder=3)
        if len(invRs_valid) >= 2:
            x_line = np.linspace(np.min(invRs_valid), np.max(invRs_valid), 200)
            y_line = sigma_est * x_line + intercept
            ax.plot(x_line, y_line, 'r--', label=f'Fit: ΔP_mech = σ*(1/R) + b\nσ={sigma_est:.4f}, R²={r2:.3f}')
        ax.set_xlabel('1 / R_eff')
        ax.set_ylabel('ΔP (inside - outside)')
        ax.set_title('Laplace IFT Validation: mechanical vs EOS ΔP')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        # Optional: plot last snapshot fields
        if len(snapshots) > 0:
            snap = snapshots[-1]
            fig2, axs = plt.subplots(1, 3, figsize=(15, 4))
            im0 = axs[0].imshow(snap['rho'].T, origin='lower', cmap='viridis')
            axs[0].set_title('Density ρ')
            plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            im1 = axs[1].imshow(snap['p_eos'].T, origin='lower', cmap='plasma')
            axs[1].set_title('EOS pressure p_EOS')
            plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            im2 = axs[2].imshow(snap['p_n'].T, origin='lower', cmap='magma')
            axs[2].set_title('Mechanical normal stress p_n')
            plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            plt.suptitle(f"Snapshot R_nom={snap['R_nominal']}, R_eff={snap['R_eff']:.2f}\nΔP_EOS={snap['deltaP_eos']:.4f}, ΔP_mech={snap['deltaP_mech']:.4f}")
            plt.tight_layout()
        plt.show()

    results = {
        'sigma_est': float(sigma_est),
        'intercept': float(intercept),
        'r2': float(r2),
        'invR': invRs.tolist(),
        'R_eff': R_effs.tolist(),
        'deltaP_mech': deltaPs_mech.tolist(),
        'deltaP_eos': deltaPs_eos.tolist(),
        'radii': list(radii),
        'snapshots': snapshots
    }
    return results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    np.random.seed(0)
    results = run_laplace_ift_validation(
        nx=128, ny=128,
        radii=[48, 24, 16, 8],
        omega=1.0, cs2=1.0 / 3.0, G=-5.2, rho_0=1.0,
        rho_liq=2.1, rho_gas=0.15,
        max_steps=5000, check_every=500, tol=1e-10,
        save_fig=True
    )
    print("\nEstimated sigma (mechanical fit):", results['sigma_est'])
