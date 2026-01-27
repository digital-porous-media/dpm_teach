import numpy as np
from typing import Optional, List, Tuple
class LBM_SCMP:
    """
    Single-Component Multi-Phase Lattice Boltzmann Method using Shan-Chen Model
    """
    def __init__(self, nx: int, ny: int, omega: float = 1.0, cs2: float = 1.0 / 3.0, G: float = -4.0, rho_0: float = 1.0, G_ads: float = -4.0, psi_solid: float = 0.5) -> None:
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
        # print(f"SCMP LBM initialized: {nx}×{ny}")
        # print(f"  omega={omega}, tau={self.tau}, cs²={cs2}, G={G}")

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
        # print(f"Initialized from density array")
        # print(f"  ρ range: [{self.rho.min():.4f}, {self.rho.max():.4f}]")
        # print(f"  ρ mean: {self.rho.mean():.4f}")

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
        # self.psi_init = 1
        # psi = self.psi_init * np.exp(-self.rho_0 / np.maximum(rho, 1e-9))
        # psi = rho.copy() #1.0 - np.exp(-rho)
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
        # F_fluid = np.zeros_like(F)
        # for i in range(1, 9):
        #     psi_shifted = self.compute_pseudopotential(np.roll(np.roll(self.rho, -self.c[i, 0], axis=0), -self.c[i, 1], axis=1))
        #     # if (self.solid_mask is not None) and self.solid_mask.any():
        #     #     solid_nb = np.roll(np.roll(self.solid_mask, -self.c[i, 0], axis=0), -self.c[i, 1], axis=1)
        #     #     # Exclude contributions where the neighbor is solid
        #     #     psi_shifted = np.where(solid_nb, self.psi_s, psi_shifted)
        #     # if self.solid_mask is not None:
        #     #     psi_shifted[self.solid_mask] = 0
        #     F_fluid[0] += self.weights[i] * self.c[i, 0] * psi_shifted
        #     F_fluid[1] += self.weights[i] * self.c[i, 1] * psi_shifted
        # F_fluid *= -self.G * psi

        # F += F_fluid

        # if (self.solid_mask is not None) and self.solid_mask.any():

        #     F_solid = np.zeros_like(F)

        #     for i in range(1, 9):
        #         # Use consistent roll direction with F_fluid (pull scheme: -direction)
        #         solid_nb = np.roll(np.roll(self.solid_mask, -self.c[i, 0], axis=0), -self.c[i, 1], axis=1)
        #         F_solid[0] += self.weights[i] * self.c[i, 0] * solid_nb
        #         F_solid[1] += self.weights[i] * self.c[i, 1] * solid_nb
        #         # cx, cy = int(cx), int(cy)

        #         # solid_nb = solid_padded[
        #         #     1 - cx:1 - cx + self.nx,
        #         #     1 - cy:1 - cy + self.ny
        #         # ]

        #         # F_solid[0] += self.weights[i] * cx * self.psi_s * solid_nb
        #         # F_solid[1] += self.weights[i] * cy * self.psi_s * solid_nb
        #         # solid_nb = np.roll(np.roll(self.solid_mask, self.c[i, 0], axis=0), self.c[i, 1], axis=1)

        #         # F_solid[0] += self.weights[i] * self.c[i, 0] * solid_nb
        #         # F_solid[1] += self.weights[i] * self.c[i, 1] * solid_nb

        #     F_solid *= -self.G_ads * psi
        #     F += F_solid

        #     F[:, self.solid_mask] = 0.0

        # # # Clamp force magnitude for stability
        # # F_mag = np.sqrt(F[0]**2 + F[1]**2)
        # # max_force = 1e-3  # Limit force magnitude
        # # mask = F_mag > max_force
        # # if np.any(mask):
        # #     scale = max_force / (F_mag[mask] + 1e-9)
        # #     F[0][mask] *= scale
        # #     F[1][mask] *= scale
        # return F
    # def stream_and_bounceback(self, f_post: np.ndarray) -> np.ndarray:
    #     """
    #     Link-wise bounce-back streaming with periodic rolls.
    #     Blocks any link touching solids and reflects at the fluid source.
    #     f_post: (9, nx, ny)
    #     returns f_next: (9, nx, ny)
    #     """
    #     f_next = np.zeros_like(f_post)
    #     solid = self.solid_mask          # (nx, ny)
    #     fluid = ~solid
    #     c = self.c                       # (9, 2) integer directions
    #     opp = self.opposite              # length 9, opposite indices

    #     for i in range(9):
    #         ex, ey = int(c[i, 0]), int(c[i, 1])
    #         oi = opp[i]

    #         # Neighbor-solid mask at the SOURCE positions: s(x + e_i)
    #         neighbor_solid = np.roll(np.roll(solid, -ex, axis=0), -ey, axis=1)

    #         # Source masks
    #         travel_mask = fluid & (~neighbor_solid)  # fluid->fluid links
    #         bounce_mask = fluid & ( neighbor_solid)  # fluid->solid links

    #         # STREAM across fluid->fluid links: roll the masked source populations
    #         to_stream = f_post[i] * travel_mask
    #         streamed = np.roll(np.roll(to_stream, ex, axis=0), ey, axis=1)
    #         f_next[i] += streamed

    #         # BOUNCE-BACK at the source for fluid->solid links
    #         # reflect into opposite direction at the same source cell
    #         to_bounce = f_post[i] * bounce_mask
    #         f_next[oi] += to_bounce

    #     # Keep solids inert
    #     f_next[:, solid] = 0
    #     return f_next
    def stream_and_bounceback(self, f_post: np.ndarray) -> np.ndarray:
        # f_post: post-collision distributions (9, nx, ny)
        f_next = np.zeros_like(f_post)
        solid = self.solid_mask
        fluid = ~solid

        for i in range(9):
            # Pull scheme: source (upstream) for direction i is x - c[i]
            src = np.roll(np.roll(f_post[i], self.c[i, 0], axis=0), self.c[i, 1], axis=1)

            # Is the upstream neighbor solid?
            upstream_solid = np.roll(np.roll(solid, self.c[i, 0], axis=0), self.c[i, 1], axis=1)

            # For fluid cells:
            # - If upstream is fluid, take src (normal streaming)
            # - If upstream is solid, apply bounce-back: reflect local opposite
            normal_mask = fluid & (~upstream_solid)
            bounce_mask = fluid & upstream_solid

            f_next[i][normal_mask] = src[normal_mask]
            f_next[i][bounce_mask] = f_post[self.opposite[i]][bounce_mask]

        f_next[:, self.solid_mask] = 0  # Zero out solid cells
        return f_next

    def stream(self, f: np.ndarray) -> np.ndarray:
        """
        Streaming step (with periodic BCs)
        Parameters:
        ---
        f : ndarray (9, nx, ny)
            Distribution function
        Returns:
        ---
        f_streamed : ndarray (9, nx, ny)
            Streamed distribution function
        """
        f_streamed = np.zeros_like(f)
        for i in range(9):
            # Roll in x and y directions
            f_streamed[i] = np.roll(f[i], self.c[i, 0], axis=0)
            f_streamed[i] = np.roll(f_streamed[i], self.c[i, 1], axis=1)
        return f_streamed

    def bounce_back(self, f: np.ndarray) -> np.ndarray:
        """
        Bounce-back boundary condition at solid nodes
        Parameters:
        ---
        f : ndarray (9, nx, ny)
            Distribution function
        Returns:
        ---
        f_bounced : ndarray (9, nx, ny)
            Distribution function after bounce-back
        """
        f_bounced = f.copy()
        for i in range(1, 9):
            # solid_ahead = np.roll(np.roll(self.solid_mask, self.c[i, 0], axis=0), self.c[i, 1], axis=1)
            f_bounced[i][self.solid_mask] = self.f[self.opposite[i]][self.solid_mask]
        return f_bounced

    def apply_wall_density_bc(self) -> None:
        """
        Set fictitious density on solid nodes for wall force calculation.
        The density of a solid node is set to the density of its adjacent fluid neighbor.
        This is a simplified approach; more complex methods exist.
        """
        if not np.any(self.solid_mask):
            return

        # Iterate over lattice directions
        for i in range(1, 9):
            # Identify fluid nodes adjacent to solids in direction 'i'
            solid_shifted = np.roll(np.roll(self.solid_mask, self.c[i, 0], axis=0), self.c[i, 1], axis=1)
            fluid_near_solid = ~self.solid_mask & solid_shifted

            # Get the coordinates of these fluid nodes
            fluid_x, fluid_y = np.where(fluid_near_solid)

            # Get the coordinates of the corresponding solid nodes
            solid_x = fluid_x - self.c[i, 0] - 1
            solid_y = fluid_y - self.c[i, 1] - 1

            # Assign the fluid density to the adjacent solid node
            # This handles boundary wrapping correctly due to how np.roll works with negative indices
            self.rho[solid_x, solid_y] = self.rho[fluid_x, fluid_y]

    def add_body_force(self, force_magnitude: float, direction: Optional[Tuple[float, float]] = (0.0, 0.0)) -> None:
        """
        Apply body force
        Parameters:
        ---
        force_magnitude : float
            Magnitude of the body force
        direction : tuple of float, optional
            Direction of the body force. Defaults to (1.0, 0.0)
        Returns:
        ---
        None
        """
        direction = np.array(direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-9)  # Normalize direction
        self.u[0] += force_magnitude * direction[0]
        self.u[1] += force_magnitude * direction[1]

    def limit_force_smooth(self, F, k=2.0, alpha=2.0):
        F0 = alpha * self.cs2 * self.rho
        mag = np.sqrt(F[0]**2 + F[1]**2)
        scale = 1.0 / (1.0 + (mag / (F0 + 1e-12))**k)
        F_lim = np.empty_like(F)
        F_lim[0] = F[0] * scale
        F_lim[1] = F[1] * scale
        return F_lim
    
    def test_bounce_bottom_wall(self):
        nx, ny = self.nx, self.ny
        f_post = np.zeros((9, nx, ny))
        # Put unit populations in downward direction i=4 at y=1 (first fluid layer)
        i_down, i_up = 4, 2
        f_post[i_down, :, 1] = 1.0

        f_next = self.stream_and_bounceback(f_post)

        # They must be reflected to i_up at y=1, not leak to y=0 or wrap to top
        assert np.allclose(f_next[i_up, :, 1], 1.0)
        assert np.all(f_next[i_down, :, 0] == 0.0)  # nothing at the wall
        assert np.all(f_next[:, :, -1] == 0.0)      # no wrap to top
        print("Bounce-back test passed.")

    def _stats(self, name, arr):
        pass
        # print(f"{name}: min={np.nanmin(arr):.3e}, max={np.nanmax(arr):.3e}, "
            #   f"finite={np.isfinite(arr).all()}")

    def step(self, apply_body_force=False, force_magnitude=0.0, force_direction=(0.0, 0.0)):
        """
        Perform one simulation step
        Parameters:
        ---
        apply_body_force : bool
            Whether to apply body force
        force_magnitude : float
            Magnitude of the body force
        force_direction : tuple of float, optional
            Direction of the body force. Defaults to (1.0, 0.0)
        Returns:
        ---
        None
        """
        # self._stats("f (pre)", self.f)
        # Macroscopic variables
        self.rho, u_temp = self.compute_macroscopic(self.f)
        # self._stats("rho", self.rho)
        # self._stats("u_temp_mag", np.sqrt(u_temp[0]**2 + u_temp[1]**2))
        # self.rho[self.solid_mask] = self.rho_0  # Reset solid nodes
        # u_temp[:, self.solid_mask] = 0.0  # Reset solid nodes
        # self.apply_wall_density_bc()
        # Compute pseudopotential
        self.psi = self.compute_pseudopotential(self.rho)
        # Compute Shan-Chen force
        F = self.compute_shan_chen_force()
        
        # F = self.limit_force_smooth(F)
        # self._stats("|F|", np.sqrt(F[0]**2 + F[1]**2))
        # TODO: Apply body force total Force, not to velocity directly
        if apply_body_force:
            F[0] += self.rho * force_magnitude * force_direction[0]
            F[1] += self.rho * force_magnitude * force_direction[1]
            # self.add_body_force(force_magnitude, force_direction)
        # Update velocity with Shan-Chen force
        self.u = u_temp + 0.5 * F / np.maximum(self.rho, 1e-1)
        # # Clamp velocity magnitude for stability
        # u_mag = np.sqrt(self.u[0]**2 + self.u[1]**2)
        # low_rho = self.rho < 0.1
        # too_fast = u_mag > 0.1 * np.sqrt(self.cs2)  # 0.1 cs

        # mask = low_rho & too_fast
        # if np.any(mask):
        #     scale = (0.1 * np.sqrt(self.cs2)) / (u_mag[mask] + 1e-12)
        #     self.u[0][mask] *= scale
        # umax = 0.1  # Mach ~ 0.1 for D2Q9 with cs^2=1/3
        # mask = u_mag > umax
        # if np.any(mask):
        #     s = umax / (u_mag[mask] + 1e-12)
        #     self.u[0][mask] *= s
        #     self.u[1][mask] *= s
        if (self.solid_mask is not None) and self.solid_mask.any():
            self.u[:, self.solid_mask] = 0.0
        self._stats("u_mag (after half-force)", np.sqrt(self.u[0]**2 + self.u[1]**2))
        # Guo Forcing
        guo_force = np.zeros_like(self.f)
        ci = self.c[:, :, None, None]
        ui = self.u[None, :, :, :]
        Fi_vec = F[None, :, :, :]
        ci_dot_u = np.sum(ci * ui, axis=1)
        term2 = (ci - ui) / self.cs2 + (ci_dot_u[:, None, :, :] * ci) / (self.cs2 ** 2)
        guo_force = self.weights[:, None, None] * (1 - 0.5 * self.omega) * np.sum(term2 * Fi_vec, axis=1)
        # guo_force[:, self.solid_mask] = 0.0
        # Just after F is computed:
        # y1 = 1  # first fluid layer above bottom wall at y=0
        # Fy_near_wall = F[1, :, y1]
        # print(f"Fy near wall (mean): {Fy_near_wall.mean():.3e}, min: {Fy_near_wall.min():.3e}, max: {Fy_near_wall.max():.3e}")
        # fluid = ~self.solid_mask

        # # Mass source (should be ~0)
        # mass_src = np.sum(guo_force[:, fluid])

        # # Momentum source (should match sum of F over fluid)
        # mom_src_x = np.sum(self.c[:, 0, None, None] * guo_force)  # sum over i, x,y
        # mom_src_y = np.sum(self.c[:, 1, None, None] * guo_force)

        # Fx_sum = np.sum(F[0][fluid])
        # Fy_sum = np.sum(F[1][fluid])

        # print(f"mass_src ≈ {mass_src:.3e}")
        # print(f"mom_src_x/Fx_sum ≈ {mom_src_x / (Fx_sum + 1e-12):.3f}")
        # print(f"mom_src_y/Fy_sum ≈ {mom_src_y / (Fy_sum + 1e-12):.3f}")

        # # Net vertical force (diagnose drift direction)
        # Fy_net = np.sum(F[1][fluid])
        # print(f"Fy_net over fluid ≈ {Fy_net:.3e}")

        # Compute equilibrium distribution
        self.f_eq = self.compute_equilibrium(self.rho, self.u)
        # if (self.solid_mask is not None) and self.solid_mask.any():
        #     self.f_eq[:, self.solid_mask] = 0
        # Collide
        f_new = self.f - self.omega * (self.f - self.f_eq) + guo_force

        # Stream and Bounceback
        self.f = self.stream_and_bounceback(f_new)

        self.timestep += 1
