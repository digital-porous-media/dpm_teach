    def get_velocity_magnitude(self):
        """
        Compute velocity magnitude field
        $$|\\mathbf{u}| = \\sqrt{u_x^2 + u_y^2}$$
        Returns:
        --------
        u_mag : ndarray (nx, ny)
            Velocity magnitude at each point
        """
        return np.sqrt(self.u[0]**2 + self.u[1]**2)

    def get_kinetic_energy(self):
        """
        Compute total kinetic energy
        $$KE = \frac{1}{2} \\sum_{x,y} \rho(x,y) |\\mathbf{u}(x,y)|^2$$
        For phase separation: KE should decay to near zero
        For driven flow: KE should reach steady state
        Returns:
        --------
        ke : float
            Total kinetic energy (normalized by grid size)
        """
        u_mag_sq = self.u[0]**2 + self.u[1]**2
        ke = np.sum(self.rho * u_mag_sq) / (self.nx * self.ny)
        return ke

    def get_density_gradient(self):
        """
        Compute density gradient magnitude
        $$|\nabla \rho| = \\sqrt{\\left(\frac{\\partial \rho}{\\partial x}\right)^2 +
        \\left(\frac{\\partial \rho}{\\partial y}\right)^2}$$
        Large gradients indicate interfaces
        Returns:
        --------
        grad_rho : ndarray (nx, ny)
            Magnitude of density gradient
        """
        drho_dx = np.gradient(self.rho, axis=0)
        drho_dy = np.gradient(self.rho, axis=1)
        grad_rho = np.sqrt(drho_dx**2 + drho_dy**2)
        return grad_rho

    def get_interface_length(self, threshold=None):
        """
        Estimate interface length using gradient threshold
        Interfaces are where |∇ρ| exceeds threshold
        Parameters:
        -----------
        threshold : float, optional
            Gradient threshold for interface detection
            Default: 10% of max gradient
        Returns:
        --------
        interface_length : float
            Number of interface pixels
        """
        grad_rho = self.get_density_gradient()
        if threshold is None:
            threshold = 0.1 * np.max(grad_rho)
        interface_pixels = np.sum(grad_rho > threshold)
        return interface_pixels

    def get_phase_volumes(self, threshold=None):
        """
        Estimate volume fraction of each phase
        Uses density threshold to separate liquid and gas
        Parameters:
        -----------
        threshold : float, optional
            Density threshold for phase separation
            Default: mean density
        Returns:
        --------
        phi_liquid : float
            Volume fraction of high-density phase (0 to 1)
        phi_gas : float
            Volume fraction of low-density phase (0 to 1)
        """
        if threshold is None:
            threshold = np.mean(self.rho)
        liquid_pixels = np.sum(self.rho > threshold)
        total_pixels = self.nx * self.ny
        phi_liquid = liquid_pixels / total_pixels
        phi_gas = 1.0 - phi_liquid
        return phi_liquid, phi_gas

    def get_density_statistics(self):
        """
        Compute density field statistics
        Returns:
        --------
        stats : dict
            Dictionary containing:
            - 'mean': mean density
            - 'std': standard deviation
            - 'min': minimum density
            - 'max': maximum density
            - 'range': max - min
        """
        stats = {
            'mean': np.mean(self.rho),
            'std': np.std(self.rho),
            'min': np.min(self.rho),
            'max': np.max(self.rho),
            'range': np.max(self.rho) - np.min(self.rho)
        }
        return stats

    def measure_interfacial_tension(self, method='laplace'):
        """
        Measure interfacial tension from simulation
        Uses Young-Laplace equation: ΔP = γ * κ
        where κ is interface curvature
        Parameters:
        -----------
        method : str
            'laplace': Use pressure difference across curved interface
            'gradient': Use density gradient method
        Returns:
        --------
        gamma : float
            Interfacial tension (in simulation units)
        """
        if method == 'laplace':
            # For flat interface, measure pressure jump
            # This is more complex - requires interface detection
            # Simplified: use density gradient as proxy
            grad_rho = self.get_density_gradient()
            # Interfacial tension ~ integral of pressure tensor
            # Simplified approximation:
            gamma = np.max(grad_rho) * self.cs2 * abs(self.G) / 2.0
        elif method == 'gradient':
            # Method based on density gradient
            grad_rho = self.get_density_gradient()
            rho_max = np.max(self.rho)
            rho_min = np.min(self.rho)
            # Approximate surface tension from gradient
            gamma = np.mean(grad_rho) * (rho_max - rho_min) * abs(self.G) / 4.0
        else:
            raise ValueError(f"Unknown method: {method}")
        return gamma

    def get_pressure_field(self):
        """
        Compute pressure field from equation of state
        For Shan-Chen model:
        $$P = \rho c_s^2 + \frac{G}{2} \\psi^2$$
        Returns:
        --------
        P : ndarray (nx, ny)
            Pressure field
        """
        psi = self.compute_pseudopotential(self.rho)
        P = self.rho * self.cs2 + 0.5 * self.G * psi**2
        return P

    def get_pressure_jump_across_interface(self):
        """
        Measure pressure difference across interface
        For Young-Laplace: ΔP = γ * κ
        Returns:
        --------
        delta_p : float
            Pressure difference between liquid and gas phases
        """
        P = self.get_pressure_field()
        rho_threshold = np.mean(self.rho)
        P_liquid = np.mean(P[self.rho > rho_threshold])
        P_gas = np.mean(P[self.rho <= rho_threshold])
        delta_p = P_liquid - P_gas
        return delta_p

    def get_flow_rate(self, direction='x'):
        """
        Compute flow rate in specified direction
        $$Q = \\sum_{x,y} \rho(x,y) u_d(x,y)$$
        Parameters:
        -----------
        direction : str
            'x' or 'y'
        Returns:
        --------
        flow_rate : float
            Total flow rate
        """
        if direction == 'x':
            flow_rate = np.sum(self.rho * self.u[0])
        elif direction == 'y':
            flow_rate = np.sum(self.rho * self.u[1])
        else:
            raise ValueError("direction must be 'x' or 'y'")
        return flow_rate

    def get_average_velocity(self):
        """
        Compute volume-averaged velocity
        $$\\langle \\mathbf{u} \rangle = \frac{1}{V} \\sum_{x,y} \\mathbf{u}(x,y)$$
        Returns:
        --------
        u_avg : ndarray (2,)
            Average velocity [u_x, u_y]
        """
        u_avg = np.array([
            np.mean(self.u[0]),
            np.mean(self.u[1])
        ])
        return u_avg

    def get_capillary_number(self, reference_velocity=None):
        """
        Compute capillary number
        $$Ca = \frac{\\mu v}{\\gamma}$$
        Ratio of viscous to capillary forces
        Parameters:
        -----------
        reference_velocity : float, optional
            Reference velocity for Ca calculation
            Default: average velocity magnitude
        Returns:
        --------
        ca : float
            Capillary number
        """
        if reference_velocity is None:
            u_avg = self.get_average_velocity()
            reference_velocity = np.linalg.norm(u_avg)
        # Viscosity in LBM: nu = cs^2 * (tau - 0.5)
        nu = self.cs2 * (self.tau - 0.5)
        # Interfacial tension
        gamma = self.measure_interfacial_tension()
        # Capillary number
        ca = nu * reference_velocity / (gamma + 1e-9)
        return ca

    def get_reynolds_number(self, reference_velocity=None, characteristic_length=None):
        """
        Compute Reynolds number
        $$Re = \frac{\rho v L}{\\mu}$$
        Parameters:
        -----------
        reference_velocity : float, optional
            Reference velocity. Default: average velocity magnitude
        characteristic_length : float, optional
            Characteristic length. Default: sqrt(nx*ny)
        Returns:
        --------
        re : float
            Reynolds number
        """
        if reference_velocity is None:
            u_avg = self.get_average_velocity()
            reference_velocity = np.linalg.norm(u_avg)
        if characteristic_length is None:
            characteristic_length = np.sqrt(self.nx * self.ny)
        # Kinematic viscosity
        nu = self.cs2 * (self.tau - 0.5)
        # Reynolds number
        re = reference_velocity * characteristic_length / (nu + 1e-9)
        return re