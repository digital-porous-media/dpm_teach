import numpy as np
import matplotlib.pyplot as plt
from lbm_d2q9_scmp import LBM_SCMP

# ============================================================================
# PHASE SEPARATION DEMONSTRATION
# ============================================================================

def create_phase_separation_demo():
    """
    Demonstrate spinodal decomposition
    
    Start with homogeneous mixture with small random perturbations
    Watch as it spontaneously separates into liquid and gas phases
    """
    
    # Grid parameters
    nx, ny = 256, 256
    
    # Initialize LBM
    sim = LBM_SCMP(nx=nx, ny=ny, omega=1.0, cs2=1/3, G=-120.0)
    
    # Create initial density field: homogeneous + small random noise
    rho_mean = 1.0
    noise_amplitude = 0.05 * rho_mean  # 5% perturbation
    
    rho_init = rho_mean + np.random.normal(0, noise_amplitude, (nx, ny))
    
    # Initialize simulation
    sim.initialize_density_field(rho_init)
    
    # No solids for phase separation
    sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))
    
    print("\n" + "="*70)
    print("PHASE SEPARATION (SPINODAL DECOMPOSITION) DEMONSTRATION")
    print("="*70)
    print(f"Grid: {nx} x {ny}")
    print(f"Initial mean density: {rho_mean}")
    print(f"Noise amplitude: {noise_amplitude:.6f}")
    print(f"Shan-Chen G: {sim.G}")
    print(f"Relaxation parameter omega: {sim.omega}\n")
    
    # Simulation parameters
    n_steps = 5000
    plot_interval = 500
    
    # Storage for time evolution analysis
    times = []
    kinetic_energies = []
    rho_max_list = []
    rho_min_list = []
    interface_lengths = []
    
    print(f"{'Step':>6} | {'KE':>10} | {'ρ_max':>8} | {'ρ_min':>8} | {'Interface':>10}")
    print("-" * 70)
    
    # Run simulation
    for step in range(n_steps):
        # Execute LBM step (no body force for phase separation)
        sim.step(apply_body_force=False)
        
        # Collect diagnostics at intervals
        if step % plot_interval == 0:
            ke = sim.get_kinetic_energy()
            rho_max = np.max(sim.rho)
            rho_min = np.min(sim.rho)
            interface_len = sim.get_interface_length()
            
            times.append(step)
            kinetic_energies.append(ke)
            rho_max_list.append(rho_max)
            rho_min_list.append(rho_min)
            interface_lengths.append(interface_len)
            
            print(f"{step:6d} | {ke:10.3e} | {rho_max:8.4f} | {rho_min:8.4f} | {interface_len:10.0f}")
    
    print("-" * 70)
    print(f"Final kinetic energy: {kinetic_energies[-1]:.3e}")
    print(f"Final density range: [{rho_min_list[-1]:.4f}, {rho_max_list[-1]:.4f}]")
    print(f"Density separation: {rho_max_list[-1] - rho_min_list[-1]:.4f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Final density field
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(sim.rho.T, cmap='RdBu_r', origin='lower')
    ax1.set_title('Final Density Field ρ', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = plt.colorbar(im1, ax=ax1, label='ρ')
    
    # 2. Final velocity magnitude
    ax2 = plt.subplot(2, 3, 2)
    u_mag = sim.get_velocity_magnitude()
    im2 = ax2.imshow(u_mag.T, cmap='viridis', origin='lower')
    ax2.set_title('Final Velocity Magnitude |u|', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cbar2 = plt.colorbar(im2, ax=ax2, label='|u|')
    
    # 3. Density gradient (interfaces)
    ax3 = plt.subplot(2, 3, 3)
    grad_rho = sim.get_density_gradient()
    im3 = ax3.imshow(grad_rho.T, cmap='hot', origin='lower')
    ax3.set_title('Density Gradient |∇ρ| (Interfaces)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = plt.colorbar(im3, ax=ax3, label='|∇ρ|')
    
    # 4. Kinetic energy decay
    ax4 = plt.subplot(2, 3, 4)
    ax4.semilogy(times, kinetic_energies, 'b-', linewidth=2.5, marker='o', markersize=6)
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Kinetic Energy', fontsize=12)
    ax4.set_title('KE Decay (System Equilibrates)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_yscale('log')
    
    # 5. Density extrema evolution
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(times, rho_max_list, 'r-', label='ρ_max (liquid)', linewidth=2.5, marker='s', markersize=5)
    ax5.plot(times, rho_min_list, 'b-', label='ρ_min (gas)', linewidth=2.5, marker='^', markersize=5)
    ax5.axhline(rho_mean, color='k', linestyle='--', label='ρ_mean (initial)', linewidth=2)
    ax5.set_xlabel('Time Step', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Phase Separation: Density Divergence', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # 6. Interface length evolution
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(times, interface_lengths, 'g-', linewidth=2.5, marker='d', markersize=5)
    ax6.set_xlabel('Time Step', fontsize=12)
    ax6.set_ylabel('Interface Length (pixels)', fontsize=12)
    ax6.set_title('Interface Coarsening', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_separation_demo.png', dpi=150, bbox_inches='tight')
    print("\nSaved figure: phase_separation_demo.png")
    plt.show()
    
    return sim, times, kinetic_energies, rho_max_list, rho_min_list, interface_lengths


# ============================================================================
# ADVANCED: SHOW SNAPSHOTS AT DIFFERENT TIMES
# ============================================================================

def create_phase_separation_snapshots():
    """
    Show evolution of phase separation at multiple time points
    """
    
    nx, ny = 200, 200
    sim = LBM_SCMP(nx=nx, ny=ny, omega=1.0, cs2=1/3, G=-60.0, psi_init=4.0)
    
    # Initialize
    rho_mean = 1.0
    # noise_amplitude = 0.05 * rho_mean
    # rho_init = rho_mean + np.random.normal(0, noise_amplitude, (nx, ny))
    rho_init = rho_mean + np.random.uniform(-0.1, 0.1, (nx, ny))  # np.random.rand(nx, ny)
    # walls = np.zeros((nx, ny), dtype=bool)
    # walls[:, :5] = True
    # walls[:, -5:] = True
    # walls[:5, :] = True
    # walls[-5:, :] = True
    # x = np.arange(nx)
    # y = np.arange(ny)
    # X, Y = np.meshgrid(x, y, indexing='ij')
    # walls = (X < 5) | (X > nx - 5) | (Y < 5) | (Y > ny - 5)

    sim.initialize_density_field(rho_init)
    # sim.set_solids_from_mask(walls)
    sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))
    
    # Snapshot times
    snapshot_steps = [0, 100, 200, 400, 800, 1600, 3200, 6400]
    snapshots = {}
    
    print("Running phase separation with snapshots...")
    
    for step in range(max(snapshot_steps) + 1):
        sim.step(apply_body_force=False)
        
        if step in snapshot_steps:
            snapshots[step] = {
                'rho': sim.rho.copy(),
                'u_mag': sim.get_velocity_magnitude().copy(),
                'grad_rho': sim.get_density_gradient().copy(),
            }
            print(f"  Step {step}: ρ ∈ [{sim.rho.min():.4f}, {sim.rho.max():.4f}]")
    
    # Create visualization
    fig, axes = plt.subplots(3, len(snapshot_steps), figsize=(18, 10))
    
    for idx, step in enumerate(snapshot_steps):
        snap = snapshots[step]
        
        # Density
        ax = axes[0, idx]
        # im = ax.imshow(snap['rho'].T, cmap='RdBu_r', origin='lower')
        levels = np.linspace(snap['rho'].min(), snap['rho'].max(), 20)
        im = ax.contourf(snap['rho'].T, levels=levels, cmap='RdBu_r', origin='lower')
        rho_mean = np.mean(snap['rho'])
        ax.contour(snap['rho'].T, levels=[rho_mean], colors='k', linewidths=1.5, origin='lower')
        ax.set_title(f't = {step}', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Density ρ', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Velocity magnitude
        ax = axes[1, idx]
        im = ax.imshow(snap['u_mag'].T, cmap='viridis', origin='lower')
        if idx == 0:
            ax.set_ylabel('Velocity |u|', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Density gradient
        ax = axes[2, idx]
        im = ax.imshow(snap['grad_rho'].T, cmap='hot', origin='lower')
        if idx == 0:
            ax.set_ylabel('Gradient |∇ρ|', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Phase Separation Evolution: Spinodal Decomposition', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'phase_separation_snapshots.png', dpi=150, bbox_inches='tight')
    print("\nSaved figure: phase_separation_snapshots.png")
    plt.show()


# ============================================================================
# PARAMETER STUDY: EFFECT OF INTERACTION STRENGTH G
# ============================================================================

# def parameter_study_interaction_strength():
#     """
#     Show how interaction strength G affects phase separation
#     """
    
#     G_values = [-60.0, -120.0, -180.0]
#     nx, ny = 200, 200
#     n_steps = 2000
    
#     results = {}
    
#     print("\nParameter study: Effect of G on phase separation")
#     print("="*70)
    
#     for G in G_values:
#         print(f"\nRunning with G = {G}...")
        
#         sim = LBM_SCMP(nx=nx, ny=ny, omega=1.0, G=G, psi_init=4.0)
        
#         rho_mean = 200.0
#         noise_amplitude = 0.05 * rho_mean
#         rho_init = rho_mean + np.random.normal(0, noise_amplitude, (nx, ny))
        
#         sim.initialize_density_field(rho_init)
#         sim.set_solids_from_mask(np.zeros((nx, ny), dtype=bool))
        
#         rho_max_evolution = []
#         rho_min_evolution = []
        
#         for step in range(n_steps):
#             sim.step(apply_body_force=False)
            
#             if step % 100 == 0:
#                 rho_max_evolution.append(np.max(sim.rho))
#                 rho_min_evolution.append(np.min(sim.rho))
        
#         results[G] = {
#             'rho_final': sim.rho.copy(),
#             'rho_max_evolution': rho_max_evolution,
#             'rho_min_evolution': rho_min_evolution,
#         }
    
#     # Visualization
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
#     # Density fields
#     ax = axes[0]
#     for idx, G in enumerate(G_values):
#         ax_sub = plt.subplot(1, 3, idx+1)
#         im = ax_sub.imshow(results[G]['rho_final'].T, cmap='RdBu_r', origin='lower')
#         ax_sub.set_title(f'G = {G}', fontweight='bold')
#         ax_sub.set_xticks([])
#         ax_sub.set_yticks([])
#         plt.colorbar(im, ax=ax_sub, label='ρ')
    
#     # Evolution comparison
#     ax = axes[1]
#     steps_array = np.arange(len(results[G_values[0]]['rho_max_evolution'])) * 100
    
#     for G in G_values:
#         rho_range = np.array(results[G]['rho_max_evolution']) - np.array(results[G]['rho_min_evolution'])
#         ax.plot(steps_array, rho_range, linewidth=2.5, marker='o', label=f'G = {G}', markersize=5)
    
#     ax.set_xlabel('Time Step', fontsize=12)
#     ax.set_ylabel('Density Range (ρ_max - ρ_min)', fontsize=12)
#     ax.set_title('Phase Separation Speed vs. Interaction Strength', fontweight='bold', fontsize=13)
#     ax.legend(fontsize=11)
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('phase_separation_parameter_study.png', dpi=150, bbox_inches='tight')
#     print("\nSaved figure: phase_separation_parameter_study.png")
#     plt.show()

def run_with_discrete_bubbles():
    """Create discrete gas bubbles in liquid continuum"""
    
    nx, ny = 256, 256
    G = -100.0
    psi_0 = 4.0
    rho_0 = 1.0
    
    sim = LBM_SCMP(nx=nx, ny=ny, omega=1.0, cs2=1/3, G=G,
                              psi_init=psi_0)
    
    # Start with mostly liquid
    rho_init = np.ones((256, 256)) * 1.2
    
    # Add 5 small gas bubbles
    x = np.arange(256)
    y = np.arange(256)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    bubble_positions = [(64, 64), (192, 64), (128, 128), (64, 192), (192, 192)]
    
    for x_c, y_c in bubble_positions:
        bubble = 0.35 * np.exp(-((X - x_c)**2 + (Y - y_c)**2) / 1500)
        rho_init -= bubble
    
    sim.initialize_density_field(rho_init)
    sim.set_solids_from_mask(np.zeros((256, 256), dtype=bool))
    
    print(f"Initial density range: [{rho_init.min():.4f}, {rho_init.max():.4f}]")
    
    # Run simulation
    snapshot_times = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    snapshots = {}
    
    for step in range(max(snapshot_times) + 1):
        sim.step(apply_body_force=False)
        
        if step in snapshot_times:
            snapshots[step] = sim.rho.copy()
            print(f"Step {step}: ρ ∈ [{np.min(sim.rho):.4f}, {np.max(sim.rho):.4f}]")
    
    # Visualize
    fig, axes = plt.subplots(1, len(snapshot_times), figsize=(18, 4))
    vmin, vmax = 0.85, 1.25
    for idx, time in enumerate(snapshot_times):
        ax = axes[idx]
        im = ax.imshow(snapshots[time].T, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f't = {time}', fontweight='bold', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='ρ')
    
    plt.suptitle('Discrete Gas Bubbles in Liquid (Should See Circles!)',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================================
# RUN DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    
    # Run main phase separation demo
    print("\n" + "="*70)
    print("RUNNING PHASE SEPARATION DEMONSTRATIONS")
    print("="*70)
    
    # sim, times, ke, rho_max, rho_min, interface = create_phase_separation_demo()
    
    # Show snapshots
    print("\n" + "="*70)
    # create_phase_separation_snapshots()
    # Run it
    run_with_discrete_bubbles()