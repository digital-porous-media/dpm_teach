import numpy as np
from typing import Optional, List, Tuple, Iterable
class LBM_SCMP:
    """
    Single-Component Multi-Phase Lattice Boltzmann Method using Shan-Chen Model
    """
    def __init__(self, nx: int, ny: int, omega: float = 1.0, cs2: float = 1.0 / 3.0, G: float = -4.0, rho_0: float = 1.0, G_ads: float = -4.0, psi_solid: float = 0.5, body_force: Iterable[float] = (0.0, 0.0)) -> None:
        """
        Initialize SCMP
        Parameters:
        ---
        nx, ny : int
            Grid dimensions
        omega: float
            Relaxation parameter (1/tau). Default = 1.0 (tau = 1.0)
        cs2: float
            Speed of sound squared. Default = 1/3 for D2Q9
        G: float
            Shan-Chen Interaction strength. Default = -4.0
            Note: Negative values are used for the phase separation.
        rho_0: float
            Reference density for pseudopotential function. Default = 1.0
        G_ads: float
            Fluid-solid interaction strength. Default = -0.4
        psi_solid: float
            Solid pseudopotential function. Default = 0.5
        body_force: Iterable[float, float]
            Force components in x and y directions. Default = (0.0, 0.0)
        """
        self.nx = nx
        self.ny = ny
        self.omega = omega
        self.cs2 = cs2
        self.G = G
        self.G_ads = G_ads
        self.tau = 1.0 / omega
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
        self.rho = np.ones((nx, ny), dtype=np.float64)
        self.rho_0 = rho_0
        self.u = np.zeros((2, nx, ny), dtype=np.float64)
        self.f = np.zeros((9, nx, ny), dtype=np.float64)
        self.f_eq = np.zeros((9, nx, ny), dtype=np.float64)
        self.psi = np.empty((nx, ny), dtype=np.float64)
        self.psi_s = psi_solid  # self.rho_0 * (1 - np.exp(-1.))

        self.timestep = 0
        self.solid_mask = np.zeros((nx, ny), dtype=bool)
        self.max_velocity = 0.1  # Limit velocity for stability
        self.rho_min_threshold = 1.e-6  # Minimum density threshold
        self.body_force = body_force  # Force components)

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
        if rho_init.shape != (self.nx, self.ny):
            raise ValueError(f"rho_init has shape {rho_init.shape}, which does not match grid shape ({self.nx}, {self.ny})")
        self.rho = rho_init.copy().astype(np.float64)
        if u_init is None:
            self.u = np.zeros((2, self.nx, self.ny), dtype=np.float64)
        else:
            if u_init.shape != (2, self.nx, self.ny):
                raise ValueError(f"u_init has shape {u_init.shape}, which does not match expected (2, {self.nx}, {self.ny})")
            self.u = u_init.copy().astype(np.float64)
        # Initialize equilibrium distribution
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
        self.f[:, self.solid_mask] = 0
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
        f_eq = np.zeros((9, self.nx, self.ny))
        u_sq = u[0]**2 + u[1]**2
        for i in range(9):
            u_dot_c = self.c[i, 0] * u[0] + self.c[i, 1] * u[1]
            f_eq[i] = self.weights[i] * rho * (1 + u_dot_c / self.cs2 + (u_dot_c**2) / (2 * self.cs2**2) - u_sq / (2 * self.cs2))
        f_eq[:, self.solid_mask] = 0.0
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
        rho = np.sum(f, axis=0)
        rho = np.clip(rho, self.rho_min_threshold, None)
        u = np.zeros((2, self.nx, self.ny), dtype=np.float64)
        for i in range(9):
            u[0] += f[i] * self.c[i, 0]
            u[1] += f[i] * self.c[i, 1]
        u /= np.maximum(rho, 1e-9)

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
        psi = self.rho_0 * (1 - np.exp(-rho / self.rho_0))
        psi[self.solid_mask] = 0.0
        return psi

    def compute_shan_chen_force(self) -> np.ndarray:
        """
        Compute Shan-Chen intermolecular force. This force drives phase separation when G < 0.
        Returns:
        ---
        F : ndarray (2, nx, ny)
            Force field [Fx, Fy]
        """
        F = np.zeros((2, self.nx, self.ny), dtype=np.float64)
        psi = self.compute_pseudopotential(self.rho)
        has_wall = (self.solid_mask is not None) and self.solid_mask.any()

        # if has_wall:
        #     psi = np.where(self.solid_mask, 0.0, psi)
        for i in range(1, 9):  # skip rest direction
            ex, ey = int(self.c[i, 0]), int(self.c[i, 1])

            # Neighbor pseudopotential by rolling psi (fluid neighbors)
            psi_nb = np.roll(np.roll(psi, -ex, axis=0), -ey, axis=1)

            w_i = self.weights[i]

            if not has_wall:
                # Pure fluid–fluid SC
                F[0] += -self.G * w_i * ex * psi * psi_nb
                F[1] += -self.G * w_i * ey * psi * psi_nb
            else:
                # Identify whether the neighbor is a solid (rolled solid mask)
                solid_nb = np.roll(np.roll(self.solid_mask, -ex, axis=0), -ey, axis=1)
                fluid_nb = ~solid_nb

                # Fluid–fluid contribution (only where neighbor is fluid)
                F[0] += -self.G * w_i * ex * psi * (psi_nb * fluid_nb)
                F[1] += -self.G * w_i * ey * psi * (psi_nb * fluid_nb)

                # Fluid–solid contribution (use constant psi_wall for solid neighbors)
                # Tip: set self.psi_wall (e.g., 0.2–0.8) and self.G_ads for wetting strength
                F[0] += -self.G_ads * w_i * ex * psi * (self.psi_s * solid_nb)
                F[1] += -self.G_ads * w_i * ey * psi * (self.psi_s * solid_nb)

        # No force inside solids
        if (self.solid_mask is not None):
            F[:, self.solid_mask] = 0.0

        return F

    def stream_and_bounceback(self, f_post: np.ndarray) -> np.ndarray:
        # f_post: post-collision distributions (9, nx, ny)
        f_next = np.zeros_like(f_post)
        solid = self.solid_mask
        fluid = ~solid

        for i in range(9):
            src = np.roll(np.roll(f_post[i], self.c[i, 0], axis=0), self.c[i, 1], axis=1)

            # Indicator function for upstream solid
            upstream_solid = np.roll(np.roll(solid, self.c[i, 0], axis=0), self.c[i, 1], axis=1)

            # For fluid cells:
            # - If upstream is fluid, take src (normal streaming)
            # - If upstream is solid, apply bounce-back: reflect local opposite
            normal_mask = fluid & (~upstream_solid)
            bounce_mask = fluid & upstream_solid

            f_next[i][normal_mask] = src[normal_mask]
            f_next[i][bounce_mask] = f_post[self.opposite[i]][bounce_mask]

        # Zero out solids for posterity
        f_next[:, self.solid_mask] = 0  # Zero out solid cells
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
        b = np.asarray(self.body_force, dtype=np.float64)
        assert b.ndim == 1, f"Body force should be a 1D array of size (2,), got {b.ndim}D"
        F = self.rho[None, :, :] * b[:, None, None]

        if (self.solid_mask is not None) and self.solid_mask.any():
            F[:, self.solid_mask] = 0.0

        return F
    
    def compute_guo_forcing(self, F) -> np.ndarray:
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
        ui = self.u[None, :, :, :]
        Fi_vec = F[None, :, :, :]
        ci_dot_u = np.sum(ci * ui, axis=1)
        term2 = (ci - ui) / self.cs2 + (ci_dot_u[:, None, :, :] * ci) / (self.cs2 ** 2)
        guo_force = self.weights[:, None, None] * (1 - 0.5 * self.omega) * np.sum(term2 * Fi_vec, axis=1)
        return guo_force
        

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
        # Compute Shan-Chen force
        F = self.compute_shan_chen_force()
        # Apply body force, if specified
        if body_force_override is not None:
            self.body_force = body_force_override

        if self.body_force != (0.0, 0.0):
            F += self.compute_body_force()

        # Update velocity with half-force correction
        self.u = u_temp + 0.5 * F / np.maximum(self.rho, 1e-1)
        if (self.solid_mask is not None) and self.solid_mask.any():
            self.u[:, self.solid_mask] = 0.0

        # Guo Forcing
        guo_force = self.compute_guo_forcing(F)

        # Compute equilibrium distribution
        self.f_eq = self.compute_equilibrium(self.rho, self.u)
        # Collide
        f_new = self.f - self.omega * (self.f - self.f_eq) + guo_force

        # Stream and Bounceback
        self.f = self.stream_and_bounceback(f_new)

        self.timestep += 1
