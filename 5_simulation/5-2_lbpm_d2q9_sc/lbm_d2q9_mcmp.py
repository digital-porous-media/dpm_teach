import numpy as np
from typing import Optional, List, Tuple, Iterable
class LBM_MCMP:
    """
    Multi-Component Multi-Phase Lattice Boltzmann Method using Shan-Chen Model
    Components are indexed as alpha = 0, ..., (n_comp - 1)
    """
    def __init__(self, nx: int, 
                 ny: int, 
                 n_comp: int = 2, 
                 omega: Iterable[float] = None,
                 cs2: float = 1.0 / 3.0,
                 G_mat: np.ndarray = None,
                 rho0: Iterable[float] = None,
                 G_ads: Iterable[float] = None,
                 psi_solid: Iterable[float] = None,
                 body_force: Optional[Iterable[float]] = None) -> None:
        """
        Initialize MCMP
        Parameters:
        ---
        nx, ny : int
            Grid dimensions
        n_comp: int
            Number of components (fluids). Default = 2
        omega: Iterable[float]
            Per-component relaxation parameter (1/tau). Must be of length (n_comp,). Default = (1.0, 1.0, ..., 1.0) (tau = 1.0 for all components)
        cs2: float
            Speed of sound squared. Default = 1/3 for D2Q9
        G_mat: np.ndarray
            Interaction strength matrix of shape (n_comp, n_comp). G_mat[alpha, beta] is the interaction strength between component alpha and beta. 
            Must be symmetric with zeros on the diagonal. 
            Default = 3.0 for all off-diagonal entries (strong phase separation) and 0 on the diagonal (no self-interaction)
        rho_0: Iterable[float]
            Per-component reference density for pseudopotential function. Default = (1.0, 1.0, ..., 1.0)
        G_ads: Iterable[float]
            Per-component fluid-solid interaction strength. Default = (-0.4, -0.4, ..., -0.4)
        psi_solid: float
            Solid pseudopotential function. Default = 0.5
        body_force: Iterable[float, float]
            Force components in x and y directions. Default = (0.0, 0.0)
        """
        self.nx = nx
        self.ny = ny

        self.n_comp = n_comp

        self.omega = np.asarray(omega) if all(omega) else np.ones(n_comp, dtype=np.float64)
        assert len(self.omega) == n_comp, f"omega must be of length {n_comp}, got {len(self.omega)}"

        self.cs2 = cs2

        self.G_mat = np.asarray(G_mat, dtype=np.float64) if G_mat is not None and G_mat.all() else (3.0 * (np.ones((n_comp, n_comp), dtype=np.float64) - np.eye(n_comp)))
        assert self.G_mat.shape == (n_comp, n_comp), f"G_mat must be of shape ({n_comp}, {n_comp}), got {self.G_mat.shape}"

        self.G_ads = np.asarray(G_ads) if G_ads is not None and all(G_ads) else (np.zeros(n_comp, dtype=np.float64))
        assert len(self.G_ads) == n_comp, f"G_ads must be of length {n_comp}, got {len(self.G_ads)}"

        self.tau = 1.0 / self.omega
        self.c = np.array([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int16)
        self.weights = np.array(
            [4. / 9.,
             1. / 9., 1. / 9., 1. / 9., 1. / 9.,
             1. / 36., 1. / 36., 1. / 36., 1. / 36.], dtype=np.float64)
        
        self.opposite = np.array(
            [0,
             3, 4, 1, 2,
             7, 8, 5, 6], dtype=np.uint8)
        
        # Fields
        self.rho = np.ones((n_comp, nx, ny), dtype=np.float64)

        self.rho_0 = np.asarray(rho0) if rho0 is not None else np.ones(n_comp, dtype=np.float64)
        self.u = np.zeros((2, nx, ny), dtype=np.float64)
        self.f = np.zeros((n_comp, 9, nx, ny), dtype=np.float64)
        self.f_eq = np.zeros((n_comp, 9, nx, ny), dtype=np.float64)
        self.psi = np.empty((n_comp,nx, ny), dtype=np.float64)
        self.psi_s = psi_solid if psi_solid is not None else 0.5 * np.ones(n_comp, dtype=np.float64) # self.rho_0 * (1 - np.exp(-1.))

        self.timestep = 0
        self.solid_mask = np.zeros((nx, ny), dtype=bool)
        self.max_velocity = 0.1  # Limit velocity for stability
        self.rho_min_threshold = 1.e-6  # Minimum density threshold
        self.body_force = np.asarray(body_force) if body_force is not None else np.zeros(2, dtype=np.float64)  # Force components

    def initialize_density_field(self, rho_init: np.ndarray, u_init: Optional[np.ndarray] = None) -> None:
        """
        Initialize simulation from arbitrary density field
        Parameters:
        ---
        rho_init: ndarray (nx, ny)
            Initial density field
        u_init: ndarray (2, nx, ny), optional
            Initial velocity field. Defaults to zero everywhere
        Returns:
        ---
        None
        """
        if rho_init.shape != (self.n_comp, self.nx, self.ny):
            raise ValueError(f"rho_init has shape {rho_init.shape}, which does not match grid shape ({self.n_comp}, {self.nx}, {self.ny})")
        self.rho = rho_init.copy().astype(np.float64)

        self.u = u_init.copy().astype(np.float64) if u_init is not None else np.zeros((2, self.nx, self.ny), dtype=np.float64)
        assert self.u.shape == (2, self.nx, self.ny), f"u_init has shape {u_init.shape}, which does not match expected (2, {self.nx}, {self.ny})"

        # Initialize equilibrium distribution
        self.psi = self.compute_pseudopotential(self.rho)
        self.f_eq = self.compute_equilibrium(self.rho, self.u)
        self.f = self.f_eq.copy()

    def set_solids_from_mask(self, solid_mask: np.ndarray) -> None:
        """
        Set solid nodes from a boolean mask
        Parameters:
        ---
        solid_mask: ndarray (nx, ny)
            Boolean array where True indicates solid nodes
        Returns:
        ---
        None
        """
        if solid_mask.shape != (self.nx, self.ny):
            raise ValueError(f"solid_mask has shape {solid_mask.shape}, which does not match grid shape ({self.nx}, {self.ny})")
        self.solid_mask = solid_mask.copy()
        # self.f_eq[:, self.solid_mask] = 0
        self.f[:, :, self.solid_mask] = 0
        self.u[:, self.solid_mask] = 0

    def compute_equilibrium(self, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute equilibrium distribution function
        Parameters:
        ---
        rho: ndarray (nx, ny)
            Density field
        u: ndarray (2, nx, ny)
            Velocity field
        Returns:
        ---
        f_eq: ndarray (9, nx, ny)
            Equilibrium distribution
        """
        f_eq = np.zeros_like(self.f)
        u_sq = u[0]**2 + u[1]**2
        for i in range(9):
            u_dot_c = self.c[i, 0] * u[0] + self.c[i, 1] * u[1]
            base = self.weights[i] * (1.0 + u_dot_c / self.cs2
                                      + (u_dot_c**2) / (2.0 * self.cs2**2)
                                      - u_sq / (2.0 * self.cs2))
            # Broadcast base (nx,ny) over components
            f_eq[:, i] = rho * base
        f_eq[:, :, self.solid_mask] = 0.0
        return f_eq

    def compute_macroscopic(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute macroscopic density and velocity from distribution function
        Parameters:
        ---
        f: ndarray (9, nx, ny)
            Distribution function
        Returns:
        ---
        rho: ndarray (nx, ny)
            Density field
        u: ndarray (2, nx, ny)
            Velocity field
        """
        rho = np.sum(f, axis=1) # Sum over distribution directions
        rho = np.clip(rho, self.rho_min_threshold, None)
        rho_tot = np.sum(rho, axis=0)  # Total density across all components

        # Total momentum 
        momentum_x = np.zeros((self.nx, self.ny), dtype=np.float64)
        momentum_y = np.zeros((self.nx, self.ny), dtype=np.float64)
        for i in range(9):
            momentum_x += np.sum(f[:, i] * self.c[i, 0], axis=0)
            momentum_y += np.sum(f[:, i] * self.c[i, 1], axis=0)
    
        u = np.zeros((2, self.nx, self.ny), dtype=np.float64)
        u[0] = momentum_x / np.maximum(rho_tot, 1e-12)
        u[1] = momentum_y / np.maximum(rho_tot, 1e-12)

        u[:, self.solid_mask] = 0.0

        return rho, u

    def compute_pseudopotential(self, rho: np.ndarray) -> None:
        """
        Compute pseudopotential function psi(rho)
        Parameters:
        ---
        rho: ndarray (nx, ny)
            Density field
        Returns:
        ---
        psi: ndarray (nx, ny)
            Pseudopotential field
        """
        psi = np.zeros_like(rho)
        for a in range(self.n_comp):
            psi[a] = self.rho_0[a] * (1 - np.exp(-rho[a] / self.rho_0[a]))
        psi[:, self.solid_mask] = 0.0
        return psi

    def compute_shan_chen_force(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute Shan-Chen intermolecular force. This force drives phase separation when G < 0.
        Returns:
        ---
        F : ndarray (2, nx, ny)
            Force field [Fx, Fy]
        """
        F = np.zeros((self.n_comp, 2, self.nx, self.ny), dtype=np.float64)
        has_wall = (self.solid_mask is not None) and self.solid_mask.any()

        # if has_wall:
        #     psi = np.where(self.solid_mask, 0.0, psi)
        for i in range(1, 9):  # skip rest direction
            ex, ey = int(self.c[i, 0]), int(self.c[i, 1])

            # Neighbor pseudopotential by rolling psi (fluid neighbors)
            psi_nb = np.roll(np.roll(psi, -ex, axis=0), -ey, axis=1)

            w_i = self.weights[i]
            solid_nb = np.roll(np.roll(self.solid_mask, -ex, axis=0), -ey, axis=1)
            fluid_nb = ~solid_nb

            # For each component alpha
            for a in range(self.n_comp):
                psi_a = psi[a]
                # Fluid-fluid interactions per component
                for b in range(self.n_comp):
                    G_ab = self.G_mat
                    psi_b_nb = np.roll(np.roll(psi[b], -ex, axis=0), -ey, axis=1)
                    F[a, 0] += -G_ab[a, b] * w_i * ex * psi_a * (psi_b_nb * fluid_nb)
                    F[a, 1] += -G_ab[a, b] * w_i * ey * psi_a * (psi_b_nb * fluid_nb)

                # Fluid-solid interactions per component
                if has_wall and self.G_ads[a] != 0.0:
                    F[a, 0] += -self.G_ads[a] * w_i * ex * psi_a * (self.psi_s[a] * solid_nb)
                    F[a, 1] += -self.G_ads[a] * w_i * ey * psi_a * (self.psi_s[a] * solid_nb)
        
        # Zero inside solids
        F[:, :, self.solid_mask] = 0.0
        return F

    def stream_and_bounceback(self, f_post: np.ndarray) -> np.ndarray:
        # f_post: post-collision distributions (9, nx, ny)
        f_next = np.zeros_like(f_post)
        solid = self.solid_mask
        fluid = ~solid

        for a in range(self.n_comp):
            for i in range(9):
                src = np.roll(np.roll(f_post[a, i], self.c[i, 0], axis=0), self.c[i, 1], axis=1)

                # Indicator function for upstream solid
                upstream_solid = np.roll(np.roll(solid, self.c[i, 0], axis=0), self.c[i, 1], axis=1)

                # For fluid cells:
                # - If upstream is fluid, take src (normal streaming)
                # - If upstream is solid, apply bounce-back: reflect local opposite
                normal_mask = fluid & (~upstream_solid)
                bounce_mask = fluid & upstream_solid

                f_next[a, i][normal_mask] = src[normal_mask]
                f_next[a, i][bounce_mask] = f_post[a, self.opposite[i]][bounce_mask]

            # Zero out solids for posterity
            f_next[a, :, self.solid_mask] = 0  # Zero out solid cells
        return f_next

    def compute_body_force(self) -> np.ndarray:
        """
        Apply body force
        Parameters:
        ---
        Returns:
        ---
        F : ndarray (2, nx, ny)
        """
        F_body = np.zeros((self.n_comp, 2, self.nx, self.ny), dtype=np.float64)
        for a in range(self.n_comp):
            F_body[a] = self.rho[a] * self.body_force[:, None, None]

        F_body[:, :, self.solid_mask] = 0.0

        return F_body
    
    def compute_guo_forcing(self, F, u) -> np.ndarray:
        """
        Compute Guo forcing
        Parameters:
        ---
        F : ndarray (2, nx, ny)
            Shan-Chen force + Body force
        Returns:
        ---
        guo_force : ndarray (2, nx, ny)
            Guo forcing term
        """
        guo_force = np.zeros_like(self.f, dtype=np.float64)
        ci = self.c[:, :, None, None]
        ui = u[None, :, :, :]
        Fi_vec = F[None, :, :, :]
        ci_dot_u = np.sum(ci * ui, axis=1)
        term2 = (ci - ui) / self.cs2 + (ci_dot_u[:, None, :, :] * ci) / (self.cs2 ** 2)

        for a in range(self.n_comp):
            Fi = F[a][None, :, :, :]
            Si = np.sum(term2 * Fi, axis=1)
            guo_force[a] = self.weights[:, None, None] * (1 - 0.5 * self.omega[a]) * Si
        
        guo_force[:, :, self.solid_mask] = 0.0
        return guo_force
    
    def compute_phase_field(self) -> np.ndarray:
        """
        Compute phase field (e.g., for visualization)
        Parameters:
        ---
        Returns:
        ---
        phi : ndarray (nx, ny)
            Phase field (e.g., normalized density or pseudopotential)
        """
        # Example: normalized density difference between first two components
        phi = (self.rho[0] - self.rho[1]) / np.maximum(self.rho[0] + self.rho[1], 1e-12)

        return phi
        

    def step(self, body_force_override: Iterable[float] = None):
        """
        Perform one simulation step
        Parameters:
        ---
        body_force_override: Iterable[float], optional
            If provided, use this body force instead of the one provided in the constructor. Should be a 2-element iterable for (Fx, Fy).
        
        Returns:
        ---
        None
        """
        # Macroscopic variables
        self.rho, u_temp = self.compute_macroscopic(self.f)

        # Compute pseudopotential
        self.psi = self.compute_pseudopotential(self.rho)

        # Compute Inter-species force
        F = self.compute_shan_chen_force(self.psi)

        # Apply body force, if specified
        if body_force_override is not None:
            self.body_force = body_force_override

        F += self.compute_body_force()

        # Update velocity with half-force correction
        rho_tot = np.sum(self.rho, axis=0)
        sumF = np.sum(F, axis=0)
        self.u = u_temp + 0.5 * sumF / np.maximum(rho_tot, 1e-1)
        self.u[:, self.solid_mask] = 0.0

        # Guo Forcing
        guo_force = self.compute_guo_forcing(F, self.u)

        # Compute equilibrium distribution
        self.f_eq = self.compute_equilibrium(self.rho, self.u)
        # Collide
        f_post = self.f - self.omega[:, None, None, None] * (self.f - self.f_eq) + guo_force

        # Stream and Bounceback
        self.f = self.stream_and_bounceback(f_post)

        self.timestep += 1

def plot_density_fields_MCMP(lbm):
    n_comp, nx, ny = lbm.rho.shape
    fig, axes = plt.subplots(1, n_comp + 1, figsize=(4*(n_comp+1), 4))
    for a in range(n_comp):
        im = axes[a].imshow(lbm.rho[a].T, origin='lower', cmap='viridis')
        axes[a].set_title(f'rho^{a}')
        plt.colorbar(im, ax=axes[a])
    rho_tot = lbm.rho.sum(axis=0)
    im = axes[-1].imshow(rho_tot.T, origin='lower', cmap='magma')
    axes[-1].set_title('rho_total')
    plt.colorbar(im, ax=axes[-1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    nx, ny = 128, 128
    n_comp = 2
    omega = [1.0, 1.0]                  # tau = 1 for both
    G_mat = np.array([[0.0, 3.0],       # cross-repulsion
                    [3.0, 0.0]])
    rho0 = [1.0, 1.0]
    G_ads = [0.0, 0.0]                  # no wall coupling (example)
    psi_solid = [0.5, 0.5]

    lbm = LBM_MCMP(nx, ny, n_comp, omega, G_mat=G_mat, rho0=rho0, G_ads=G_ads, psi_solid=psi_solid)

    # Initialize: component A on left, B on right
    rho_init = np.zeros((n_comp, nx, ny))
    rho_init[0, :nx//2, :] = 1.0
    rho_init[1, nx//2:, :] = 1.0
    lbm.initialize_density_field(rho_init)

    # Time stepping
    for _ in tqdm(range(10_000)):
        lbm.step()

    
    plot_density_fields_MCMP(lbm)