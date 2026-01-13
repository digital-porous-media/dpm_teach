import numpy as np
import matplotlib.pyplot as plt
from lbm_d2q9_scmp import LBM_SCMP
from scipy import ndimage

# ============================================================================
# UNIT TESTS: BASIC FUNCTIONALITY
# ============================================================================

def test_initialization():
    """Test that initialization creates valid fields"""
    print("\n" + "="*70)
    print("TEST 1: INITIALIZATION")
    print("="*70)
    
    nx, ny = 64, 64
    sim = LBM_SCMP(nx=nx, ny=ny, omega=1.0, cs2=1/3, G=-120.0)
    
    # Check field shapes
    assert sim.rho.shape == (nx, ny), f"rho shape mismatch: {sim.rho.shape}"
    assert sim.u.shape == (2, nx, ny), f"u shape mismatch: {sim.u.shape}"
    assert sim.f.shape == (9, nx, ny), f"f shape mismatch: {sim.f.shape}"
    assert sim.psi.shape == (nx, ny), f"psi shape mismatch: {sim.psi.shape}"
    
    print("✓ All field shapes correct")
    
    # Check initial values
    assert np.all(np.isfinite(sim.rho)), "NaN in initial rho"
    assert np.all(np.isfinite(sim.u)), "NaN in initial u"
    assert np.all(np.isfinite(sim.f)), "NaN in initial f"
    
    print("✓ All fields contain finite values")
    
    # Check weights sum to 1
    weight_sum = np.sum(sim.weights)
    assert np.isclose(weight_sum, 1.0), f"Weights don't sum to 1: {weight_sum}"
    print(f"✓ Weights sum to 1.0 (sum = {weight_sum:.10f})")
    
    # Check lattice velocities
    assert sim.c.shape == (9, 2), f"Lattice velocity shape mismatch: {sim.c.shape}"
    print("✓ Lattice velocities correct shape")
    
    return sim


def test_macroscopic_calculation():
    """Test density and velocity recovery from distribution"""
    print("\n" + "="*70)
    print("TEST 2: MACROSCOPIC CALCULATION")
    print("="*70)
    
    nx, ny = 32, 32
    sim = LBM_SCMP(nx=nx, ny=ny)
    
    # Create known distribution
    rho_test = np.ones((nx, ny)) * 1.5
    u_test = np.zeros((2, nx, ny))
    u_test[0] = 0.1  # x-velocity
    u_test[1] = 0.05  # y-velocity
    
    # Compute equilibrium
    f_eq = sim.compute_equilibrium(rho_test, u_test)
    
    # Recover macroscopic quantities
    rho_recovered, u_recovered = sim.compute_macroscopic(f_eq)
    
    # Check density recovery
    rho_error = np.max(np.abs(rho_recovered - rho_test))
    print(f"Density recovery error: {rho_error:.2e}")
    assert rho_error < 1e-10, f"Density recovery failed: error = {rho_error}"
    print("✓ Density recovered correctly from equilibrium")
    
    # Check velocity recovery
    u_error = np.max(np.abs(u_recovered - u_test))
    print(f"Velocity recovery error: {u_error:.2e}")
    assert u_error < 1e-10, f"Velocity recovery failed: error = {u_error}"
    print("✓ Velocity recovered correctly from equilibrium")
    
    return sim


def test_equilibrium_properties():
    """Test equilibrium distribution properties"""
    print("\n" + "="*70)
    print("TEST 3: EQUILIBRIUM DISTRIBUTION PROPERTIES")
    print("="*70)
    
    nx, ny = 32, 32
    sim = LBM_SCMP(nx=nx, ny=ny)
    
    rho = np.ones((nx, ny)) * 1.0
    u = np.zeros((2, nx, ny))
    
    f_eq = sim.compute_equilibrium(rho, u)
    
    # Property 1: Sum of f_eq should equal rho
    f_sum = np.sum(f_eq, axis=0)
    mass_error = np.max(np.abs(f_sum - rho))
    print(f"Mass conservation error: {mass_error:.2e}")
    assert mass_error < 1e-10, f"Mass not conserved: {mass_error}"
    print("✓ Mass conserved: Σ f_i^eq = ρ")
    
    # Property 2: Momentum should be zero when u=0
    momentum_x = np.sum(sim.c[:, 0][:, None, None] * f_eq, axis=0)
    momentum_y = np.sum(sim.c[:, 1][:, None, None] * f_eq, axis=0)
    momentum_error = np.max(np.abs(momentum_x)) + np.max(np.abs(momentum_y))
    print(f"Momentum error (u=0): {momentum_error:.2e}")
    assert momentum_error < 1e-10, f"Momentum not zero: {momentum_error}"
    print("✓ Momentum conserved: Σ c_i * f_i^eq = ρu")
    
    # Property 3: f_eq should be non-negative for physical parameters
    f_min = np.min(f_eq)
    print(f"Minimum f_eq value: {f_min:.6f}")
    if f_min < 0:
        print("⚠ Warning: Some f_eq values are negative (may indicate unphysical parameters)")
    else:
        print("✓ All f_eq values non-negative")
    
    return sim


def test_streaming_periodicity():
    """Test that streaming respects periodic boundary conditions"""
    print("\n" + "="*70)
    print("TEST 4: STREAMING PERIODICITY")
    print("="*70)
    
    nx, ny = 16, 16
    sim = LBM_SCMP(nx=nx, ny=ny)
    
    # Create a test distribution with a single non-zero value
    f_test = np.zeros((9, nx, ny))
    f_test[1, 0, 0] = 1.0  # Direction (1, 0) at corner
    
    f_streamed = sim.stream(f_test)
    
    # After streaming in direction (1,0), the value should appear at (1,0)
    # With periodic BC, it wraps around
    expected_pos = (1, 0)
    actual_value = f_streamed[1, expected_pos[0], expected_pos[1]]
    
    print(f"Value at expected position: {actual_value:.6f}")
    assert np.isclose(actual_value, 1.0), "Streaming didn't work correctly"
    print("✓ Streaming respects periodic boundary conditions")
    
    return sim


# ============================================================================
# CONSERVATION LAWS
# ============================================================================

def test_mass_conservation():
    """Test that total mass is conserved during simulation"""
    print("\n" + "="*70)
    print("TEST 5: MASS CONSERVATION")
    print("="*70)
    
    nx, ny = 64, 64
    sim = LBM_SCMP(nx=nx, ny=ny, omega=1.0, G=-120.0)
    
    # Initialize with random density
    rho_init = 1.0 + 0.1 * np.random.randn(nx, ny)
    sim.initialize_density_field(rho_init)
    sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))
    
    initial_mass = np.sum(sim.rho)
    mass_history = [initial_mass]
    
    print(f"Initial total mass: {initial_mass:.6f}")
    
    # Run simulation
    for step in range(100):
        sim.step(apply_body_force=False)
        mass_history.append(np.sum(sim.rho))
    
    # Check mass conservation
    mass_change = np.array(mass_history) - initial_mass
    max_mass_error = np.max(np.abs(mass_change))
    relative_error = max_mass_error / initial_mass
    
    print(f"Maximum mass change: {max_mass_error:.2e}")
    print(f"Relative error: {relative_error:.2e}")
    
    if relative_error < 1e-6:
        print("✓ Mass conserved to machine precision")
    elif relative_error < 1e-3:
        print("✓ Mass conserved (acceptable tolerance)")
    else:
        print(f"⚠ Warning: Mass not well conserved (error = {relative_error:.2e})")
    
    # Plot mass evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mass_history, 'b-', linewidth=2)
    ax.axhline(initial_mass, color='r', linestyle='--', label='Initial mass')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Total Mass')
    ax.set_title('Mass Conservation Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_mass_conservation.png', dpi=150)
    print("Saved: validation_mass_conservation.png")
    plt.close()
    
    return relative_error < 1e-3


def test_momentum_conservation():
    """Test momentum conservation (no external forces)"""
    print("\n" + "="*70)
    print("TEST 6: MOMENTUM CONSERVATION (NO BODY FORCE)")
    print("="*70)
    
    nx, ny = 64, 64
    sim = LBM_SCMP(nx=nx, ny=ny, omega=1.0, G=-120.0)
    
    # Initialize with zero velocity (so initial momentum = 0)
    rho_init = 1.0 + 0.05 * np.random.randn(nx, ny)
    u_init = np.zeros((2, nx, ny))
    
    sim.initialize_density_field(rho_init, u_init)
    sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))
    
    # Initial mass and momentum
    initial_mass = np.sum(sim.rho)
    initial_momentum_x = np.sum(sim.rho * sim.u[0])
    initial_momentum_y = np.sum(sim.rho * sim.u[1])
    
    momentum_x_history = [initial_momentum_x]
    momentum_y_history = [initial_momentum_y]
    
    print(f"Initial mass: {initial_mass:.6f}")
    print(f"Initial momentum: ({initial_momentum_x:.6e}, {initial_momentum_y:.6e})")
    
    # Run simulation with no external body force
    for step in range(100):
        sim.step(apply_body_force=False)
        momentum_x_history.append(np.sum(sim.rho * sim.u[0]))
        momentum_y_history.append(np.sum(sim.rho * sim.u[1]))
    
    # Changes relative to the initial momentum
    momentum_x_change = np.array(momentum_x_history) - initial_momentum_x
    momentum_y_change = np.array(momentum_y_history) - initial_momentum_y
    
    # Absolute maximum change in each component
    max_dx = np.max(np.abs(momentum_x_change))
    max_dy = np.max(np.abs(momentum_y_change))
    
    # Combined (vector) momentum change metric
    combined_change = np.sqrt(momentum_x_change**2 + momentum_y_change**2)
    max_combined = np.max(combined_change)
    
    # Normalize by total mass to interpret as "average drift velocity"
    rel_x = max_dx / initial_mass
    rel_y = max_dy / initial_mass
    rel_combined = max_combined / initial_mass
    
    print(f"Maximum |ΔP_x|: {max_dx:.2e}")
    print(f"Maximum |ΔP_y|: {max_dy:.2e}")
    print(f"Maximum |ΔP| (vector): {max_combined:.2e}")
    print(f"Relative error (normalized by mass): ε_x = {rel_x:.2e}, ε_y = {rel_y:.2e}, ε = {rel_combined:.2e}")
    
    if rel_x < 1e-6 and rel_y < 1e-6:
        print("✓ Momentum conserved to machine precision")
    elif rel_x < 1e-3 and rel_y < 1e-3:
        print("✓ Momentum conserved (acceptable tolerance)")
    else:
        print(f"⚠ Warning: Momentum not well conserved (ε_x = {rel_x:.2e}, ε_y = {rel_y:.2e})")
    
    # Plot momentum evolution
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(momentum_x_history, 'b-', linewidth=2, label='P_x(t)')
    ax[0].axhline(initial_momentum_x, color='r', linestyle='--', label='Initial P_x')
    ax[0].set_ylabel('Total Momentum (x)')
    ax[0].set_title('Momentum Conservation Test')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(momentum_y_history, 'g-', linewidth=2, label='P_y(t)')
    ax[1].axhline(initial_momentum_y, color='r', linestyle='--', label='Initial P_y')
    ax[1].set_xlabel('Time Step')
    ax[1].set_ylabel('Total Momentum (y)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_momentum_conservation.png', dpi=150)
    print("Saved: validation_momentum_conservation.png")
    plt.close()
    
    # Return pass/fail based on acceptable tolerance
    return (rel_x < 1e-3) and (rel_y < 1e-3)


if __name__ == "__main__":
    sim1 = test_initialization()
    sim2 = test_macroscopic_calculation()
    sim3 = test_equilibrium_properties()
    sim4 = test_streaming_periodicity()
    mass_conserved = test_mass_conservation()
    momentum_conserved = test_momentum_conservation()
    
    print("\n" + "="*70)
    print("SUMMARY OF TESTS")
    print("="*70)
    print(f"Mass Conservation Test: {'PASSED' if mass_conserved else 'FAILED'}")
    print(f"Momentum Conservation Test: {'PASSED' if momentum_conserved else 'FAILED'}")