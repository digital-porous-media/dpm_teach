import numpy as np
from typing import Optional, List, Tuple, Iterable

from matplotlib.colors import LogNorm
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
        self.u_comp = np.zeros((n_comp, 2, nx, ny), dtype=np.float64)
        self.u = np.zeros((2, nx, ny), dtype=np.float64)
        self.f = np.zeros((n_comp, 9, nx, ny), dtype=np.float64)
        self.f_eq = np.zeros((n_comp, 9, nx, ny), dtype=np.float64)
        self.psi = np.empty((n_comp, nx, ny), dtype=np.float64)
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
        rho = np.sum(f, axis=1)  # Sum over distribution directions
        rho = np.clip(rho, self.rho_min_threshold, None)

        # Compute per-component momentum
        u = np.zeros((self.n_comp, 2, self.nx, self.ny), dtype=np.float64)

        for i in range(9):
            u[:, 0] += f[:, i] * self.c[i, 0]  # (n_comp, nx, ny)
            u[:, 1] += f[:, i] * self.c[i, 1]  # (n_comp, nx, ny)

        # Normalize per-component momentum by per-component density
        u[:, 0] /= np.maximum(rho, 1e-12)
        u[:, 1] /= np.maximum(rho, 1e-12)

        # Zero velocity in solids
        u[:, :, self.solid_mask] = 0.0

        return rho, u

    def compute_barycentric_velocity(self, F) -> np.ndarray:
        """
        Compute barycentric velocity from force contributions
        Parameters:
        ---
        F : ndarray (2, nx, ny)
            Force [Fx, Fy]
        Returns:
        ---
        u_b : ndarray (2, nx, ny)
            Barycentric velocity [u_b_x, u_b_y]
        """
        if F.ndim == 4 and F.shape[0] == self.n_comp:
            F_tot = np.sum(F, axis=0)  # (2, nx, ny)
        elif F.ndim == 3 and F.shape[0] == 2:
            F_tot = F
        else:
            raise ValueError(f"F shape {F.shape} must be (n_comp,2,nx,ny) or (2,nx,ny).")

        rho_tot = np.sum(self.rho, axis=0)
        j_tot = np.zeros((2, self.nx, self.ny), dtype=np.float64)
        for i in range(9):
            j_tot[0] += np.sum(self.f[:, i] * self.c[i, 0], axis=0)  # sum over components
            j_tot[1] += np.sum(self.f[:, i] * self.c[i, 1], axis=0)
        # Half-force correction and division by total density
        u_b = (j_tot + 0.5 * F_tot) / np.maximum(rho_tot, self.rho_min_threshold)
        u_b[:, self.solid_mask] = 0.0
        return u_b

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
                    #F[a, 0] += -self.G_ads[a] * w_i * ex * self.rho[a] * solid_nb
                    #F[a, 1] += -self.G_ads[a] * w_i * ey * self.rho[a] * solid_nb
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
        rho_tot = np.sum(self.rho, axis=0)
        F_mix = rho_tot * self.body_force[:, None, None]
        # F_body[0] = self.rho[0] * self.body_force[:, None, None]
        # F_body[1] = -self.rho[1] * self.body_force[:, None, None]
        for a in range(self.n_comp):
            F_body[a] = (self.rho[a] / np.maximum(rho_tot, 1e-12)) * F_mix  #* self.body_force[:, None, None]

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
        self.rho, self.u_comp = self.compute_macroscopic(self.f)

        # Compute pseudopotential
        self.psi = self.compute_pseudopotential(self.rho)

        # Compute Inter-species force
        F = self.compute_shan_chen_force(self.psi)

        # Apply body force, if specified
        if body_force_override is not None:
            self.body_force = body_force_override

        F += self.compute_body_force()

        #  Compute Barycentric velocity
        self.u = self.compute_barycentric_velocity(F)

        # Guo Forcing
        guo_force = self.compute_guo_forcing(F, self.u)

        # Compute equilibrium distribution
        self.f_eq = self.compute_equilibrium(self.rho, self.u)
        # Collide
        f_post = self.f - self.omega[:, None, None, None] * (self.f - self.f_eq) + guo_force

        # Stream and Bounceback
        self.f = self.stream_and_bounceback(f_post)

        self.timestep += 1

def plot_phase_field(lbm):
    phi = lbm.compute_phase_field()  # shape (nx, ny), indexed [x, y]
    nx, ny = lbm.nx, lbm.ny

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(phi.T, origin='lower', extent=[0, nx, 0, ny],
                   cmap='RdBu_r', vmin=-1.0, vmax=1.0, interpolation='nearest', aspect='equal')
    # Interface: phi = 0 contour
    cs = ax.contour(phi.T, levels=[0.0], colors='k', linewidths=1.5, origin='lower', extent=[0, nx, 0, ny])
    # Wall overlay (y=0 line)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.0)  # first fluid row above wall
    # Solid mask overlay (light gray)
    ax.imshow(lbm.solid_mask.T, origin='lower', extent=[0, nx, 0, ny],
              cmap='gray', alpha=0.15, vmin=0, vmax=1.0)

    ax.set_title("Phase field φ with interface (φ=0) contour")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("φ")
    plt.tight_layout()
    return fig, ax


def plot_speed(lbm):
    nx, ny = lbm.nx, lbm.ny
    speed = np.hypot(lbm.u[0], lbm.u[1])
    speed_plot = np.clip(speed, 1e-10, None)  # avoid zeros for LogNorm

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(speed_plot.T, origin='lower', extent=[0, nx, 0, ny],
                   cmap='plasma', norm=LogNorm(vmin=1e-10, vmax=speed_plot.max()),
                   interpolation='nearest', aspect='equal')
    ax.set_title("Velocity magnitude (log scale)")
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.0)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("|u|")
    plt.tight_layout()
    return fig, ax

def measure_contact_angle(lbm):
    phi = lbm.compute_phase_field()
    nx, ny = lbm.nx, lbm.ny
    y_wall = 1  # first fluid row

    # Find droplet horizontal center from φ row maximum
    phi_row = phi[:, y_wall]
    xc = np.argmax(phi_row)

    # Find left/right where φ crosses zero along the wall row
    left = xc
    while left > 1 and phi_row[left] > 0: left -= 1
    right = xc
    while right < nx-2 and phi_row[right] > 0: right += 1
    a = (right - left) / 2.0

    # Apex height: highest y at xc with φ>0
    h = 0
    for y in range(y_wall, ny):
        if phi[xc, y] > 0:
            h = y - y_wall
        else:
            break

    theta_rad = 2.0 * np.arctan2(h, a)
    theta_deg = theta_rad * 180.0 / np.pi
    return theta_deg, a, h, xc

if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # # Parameters
    # nx, ny = 64, 64
    # n_comp = 2
    # omega = [1.0, 1.0]
    # G_mat = np.array([[0.0, 12.0],
    #                   [12.0, 0.0]])
    # rho0_ref = [1.0, 1.0]

    # # Random perturbation
    # eps = (np.random.rand(nx, ny) - 0.5) * 2e-2
    # rho_init = np.zeros((n_comp, nx, ny), dtype=np.float64)
    # rho_init[0] = 1.0 + eps
    # rho_init[1] = 1.0 - eps

    # # No solids
    # solid_mask = np.zeros((nx, ny), dtype=bool)

    # # Instantiate and run
    # solver = LBM_MCMP(nx, ny, n_comp=n_comp, omega=omega, G_mat=G_mat, rho0=rho0_ref)
    # solver.initialize_density_field(rho_init)
    # solver.set_solids_from_mask(solid_mask)

    # n_steps = 12000
    # for t in tqdm(range(n_steps)):
    #     solver.step()
    #     if t % 1000 == 0:
    #         phi = solver.compute_phase_field()
    #         M0 = solver.rho[0].sum()
    #         M1 = solver.rho[1].sum()
    #         fig, ax = plt.subplots(figsize=(6, 5))
    #         # Density field
    #         levels = np.linspace(phi.min(), phi.max(), 100)
    #         im = ax.contourf(phi.T, levels=levels, cmap='RdBu_r', origin='lower')
    #         rho_mean = np.mean(solver.rho)
    #         ax.contour(solver.rho[0].T + solver.rho[1].T, levels=[rho_mean], colors='k', linewidths=1.5, origin='lower')

    #         print(f"t={t}, mass0={M0:.6f}, mass1={M1:.6f}, max|u|={np.amax(np.abs(solver.u), axis=None):.4e}")

    # plt.show()

    nx, ny = 256, 128
    n_comp = 2
    omega = [1.0, 1.0]
    G_mat = np.array([[0.0, 7.],
                      [7., 0.0]])
    G_ads = [3, -3]
    psi_solid = [0.5, 0.5]

    # Bottom wall
    solid_mask = np.zeros((nx, ny), dtype=bool)
    solid_mask[:, 0:2] = True  # y=0 is solid

    # Droplet geometry
    R = 35                                # droplet radius in lattice units
    xc, yc = nx//2, R + 2                  # center above wall so droplet touches at y=1
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')  # careful: (ny,nx)
    # Fields are shaped (nx, ny), so transpose indices
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2)

    # Initialize densities
    rho_init = np.zeros((2, nx, ny), dtype=np.float64)
    inside = dist.T <= R                   # (nx, ny)
    rho_init[0][inside] = 1.0                   # component A inside droplet
    rho_init[1][inside] = 1e-3
    rho_init[0][~inside] = 1e-3                 # component B outside droplet
    rho_init[1][~inside] = 1.0
    rho_init += 1e-6 * np.random.randn(*rho_init.shape)

    solver = LBM_MCMP(nx, ny, n_comp=n_comp, omega=omega, G_mat=G_mat, rho0=[1.0, 1.0],
                      G_ads=G_ads, psi_solid=psi_solid)
    solver.initialize_density_field(rho_init)
    solver.set_solids_from_mask(solid_mask)

    for t in tqdm(range(5_000)):
        solver.step()

    plot_phase_field(solver)
    # plot_speed(solver)
    theta_deg, a, h, xc = measure_contact_angle(solver)
    # print(f"Contact angle: {theta_deg:.2f} degrees")
    plt.show()
    # phi = solver.compute_phase_field()
    # # Simple binary mask for droplet
    # # droplet = phi > 0.5
    # # plt.imshow(droplet.T, origin='lower', cmap='gray')

    # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    # ax[0].imshow(phi.T, origin='lower', cmap='RdBu_r')
    # ax[0].set_title('Phase field')

    # ax[1].imshow(solver.rho[0].T, origin='lower', cmap='viridis')
    # ax[1].set_title('rho[0]')
    # ax[2].imshow(solver.rho[1].T, origin='lower', cmap='viridis')
    # ax[2].set_title('rho[1]')
    # plt.tight_layout()
    # plt.show()
