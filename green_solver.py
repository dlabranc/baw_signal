import numpy as np
from scipy.integrate import simpson
from scipy.integrate import solve_ivp
from scipy.signal import hilbert

# ---------------------------------------------------------------
# Simple Driven Harmonic Oscillator Class
# ---------------------------------------------------------------
class DrivenHarmonicOscillator:
    """
    Driven harmonic oscillator:
        x'' + 2 γ x' + ω0^2 x = f(t)/m

    Solved using the causal Green’s function:
        G(t) = Θ(t) e^{-γ t} sin(ω_d t) / (m ω_d)
    """

    def __init__(self, m=1.0, gamma=0.2, omega0=1.0):
        self.m = m
        self.gamma = gamma
        self.omega0 = omega0
        self.omega_d = np.sqrt(abs(omega0**2 - (gamma**2)) )

    def green(self, t):
        """
        Causal Green's function G(t).
        """
        G = np.zeros_like(t)
        mask = t >= 0
        if self.omega0 > self.gamma:
            G[mask] = (
                np.exp(-self.gamma * t[mask] )
                * np.sin(self.omega_d * t[mask])
                / (self.m * self.omega_d)
            )
        elif self.omega0 < self.gamma:
            G[mask] = (
                np.exp(-self.gamma * t[mask] )
                * np.sinh(self.omega_d * t[mask])
                / (self.m * self.omega_d)
            )
        elif self.omega0 == self.gamma:
            G[mask] = (
                np.exp(-self.gamma * t[mask] )
                *  t[mask]
                / (self.m )
            )
        return G

    def solve_via_green(self, t, f_of_t):
        """
        Computes x(t) = ∫ G(t - t') f(t') dt'
        where f_of_t is an array f(t).
        """
        dt = t[1] - t[0]
        x = np.zeros_like(t)
        G = self.green  # alias

        for i in range(len(t)):
            tau = t[i] - t[: i + 1]      # only integrate where t' <= t
            integrand = G(tau) * f_of_t[: i + 1]
            x[i] = simpson(integrand, t[: i + 1])
        return x

    def solve_direct_ode(self, t, f_of_t):
        """
        Solves the same oscillator via ODE integration (verification).
        """

        def f_interp(t_query):
            # simple linear interpolation of forcing array
            return np.interp(t_query, t, f_of_t)

        def ode(t, y):
            x, v = y
            dxdt = v
            dvdt = (
                -2*self.gamma * v
                - self.omega0**2 * x
                + f_interp(t) / self.m
            )
            return [dxdt, dvdt]

        sol = solve_ivp(ode, (t[0], t[-1]), y0=[0, 0], t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[0]


    def solve_envelope_ode(self, t, B_of_t, omega_d, omega_c=None, A0=0.0 + 0.0j):
        """
        Solve the rotating-frame envelope equation for A(t):

            2(γ - i ω_c) Ȧ(t)
            - (ω0^2 - ω_c^2 - 2 i γ ω_c) A(t)
            = (B(t)/m) * exp(-i ∫Δ(t) dt),

        with Δ(t) = ω_d(t) - ω_c.

        Parameters
        ----------
        t : array_like
            Time grid (1D array, increasing).
        B_of_t : array_like
            Real drive array B(t) sampled on the same grid t, envelope.
        omega_d : array_like
            Time-dependent drive frequency ω_d(t) sampled on the same grid t.
        omega_c : float, optional
            Carrier frequency ω_c. If None, defaults to self.omega0.
        A0 : complex, optional
            Initial value of the envelope A(0).

        Returns
        -------
        A_t : np.ndarray (complex)
            Envelope A(t) evaluated on the grid t.
        """

        t = np.asarray(t)
        B_of_t = np.asarray(B_of_t)
        omega_d = np.asarray(omega_d)

        if omega_c is None:
            omega_c = self.omega0

        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m

        # interpolation of B(t) and ω_d(t)
        def B_interp(tq):
            return np.interp(tq, t, B_of_t)

        def omega_d_interp(tq):
            return np.interp(tq, t, omega_d)

        # precompute the complex coefficients
        num_coeff = (omega0**2 - omega_c**2 - 2j * gamma * omega_c)
        den_coeff = 2.0 * (gamma - 1j * omega_c)

        # real-valued ODE for Re(A), Im(A)
        def ode(tq, y):
            A = y[0] + 1j * y[1]

            Delta_t = omega_d_interp(tq) - omega_c

            drive = (B_interp(tq) / m) * np.exp(-1j * Delta_t * tq)

            dA_dt = (-num_coeff * A + drive) / den_coeff

            return [dA_dt.real, dA_dt.imag]

        # initial condition
        y0 = [A0.real, A0.imag]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            rtol=1e-8,
            atol=1e-10,
        )

        A_t = sol.y[0] + 1j * sol.y[1]
        return A_t
        
    def solve_envelope_ode_varfreq(self, t, B_of_t, omega_d_of_t, omega_c=None, A0=0.0 + 0.0j):
        """
        Solve the rotating-frame envelope equation for A(t) with time-dependent drive:

            2(γ - i ω_c) Ȧ(t)
            - (ω0^2 - ω_c^2 - 2 i γ ω_c) A(t)
            = (B(t)/m) * exp(-i φ(t)),

        with φ(t) = ∫_{t0}^{t} Δ(τ) dτ and Δ(t) = ω_d(t) - ω_c.

        Parameters
        ----------
        t : array_like
            Time grid (1D array, increasing).
        B_of_t : array_like
            Real drive array B(t) sampled on the same grid t, envelope.
        omega_d_of_t : array_like
            Drive angular frequency array ω_d(t) sampled on t.
        omega_c : float, optional
            Carrier frequency ω_c. If None, defaults to self.omega0.
        A0 : complex, optional
            Initial value of the envelope A(0).

        Returns
        -------
        A_t : np.ndarray (complex)
            Envelope A(t) evaluated on the grid t.
        """

        t = np.asarray(t)
        B_of_t = np.asarray(B_of_t)
        omega_d_of_t = np.asarray(omega_d_of_t)

        if omega_c is None:
            omega_c = self.omega0  # or self.omega_lambda in your BAW code

        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m

        # detuning array Δ(t) = ω_d(t) - ω_c
        Delta_of_t = omega_d_of_t - omega_c

        # phase φ(t) = ∫ Δ(t) dt computed on the grid (trapezoidal rule)
        dt = np.diff(t)
        phi = np.zeros_like(t, dtype=float)
        phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * dt)

        # precompute the complex coefficients
        num_coeff = (omega0**2 - omega_c**2 - 2j * gamma * omega_c)
        den_coeff = 2.0 * (gamma - 1j * omega_c)

        # interpolation of B(t) so solver can query at arbitrary times
        def B_interp(tq):
            return np.interp(tq, t, B_of_t)

        # interpolation of φ(t)
        def phi_interp(tq):
            return np.interp(tq, t, phi)

        # real-valued ODE for Re(A), Im(A)
        def ode(tq, y):
            A = y[0] + 1j * y[1]

            # RHS drive term: (B(t)/m) * exp(-i φ(t))
            drive = (B_interp(tq) / m) * np.exp(-1j * phi_interp(tq))

            dA_dt = (-num_coeff * A + drive) / den_coeff

            return [dA_dt.real, dA_dt.imag]

        # initial condition in real form
        y0 = [A0.real, A0.imag]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            rtol=1e-8,
            atol=1e-10,
        )

        A_t = sol.y[0] + 1j * sol.y[1]
        return A_t


    def solve_envelope_ode_second_order(self, t, B_of_t, omega_d_of_t,
                                    omega_c=None,
                                    A0=0.0 + 0.0j,
                                    A1=0.0 + 0.0j,
                                    return_V=False,
                                    rtol=1e-8,
                                    atol=1e-10,):
        r"""
        Second-order envelope equation for A(t) with A¨ term:

            A¨(t) + alpha Ȧ(t) + beta A(t) = (B(t)/m) * exp[-i φ(t)],

        where φ(t) = ∫ Δ(t) dt and Δ(t) = ω_d(t) - ω_c.
        """

        t = np.asarray(t)
        B_of_t = np.asarray(B_of_t)
        omega_d_of_t = np.asarray(omega_d_of_t)

        if omega_c is None:
            omega_c = self.omega0

        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m

        Delta_of_t = omega_d_of_t - omega_c

        # phase φ(t) = ∫ Δ(t) dt on the grid
        dt = np.diff(t)
        phi = np.zeros_like(t, dtype=float)
        phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * dt)

        # def B_interp(tq):
        #     return np.interp(tq, t, B_of_t)
        def B_interp(tq):
            Br = np.interp(tq, t, B_of_t.real)
            Bi = np.interp(tq, t, B_of_t.imag)
            return Br + 1j*Bi

        def phi_interp(tq):
            return np.interp(tq, t, phi)

        def ode(tq, y):
            A = y[0] + 1j * y[1]
            V = y[2] + 1j * y[3]

            phase = np.exp(-1j * phi_interp(tq))
            drive = (B_interp(tq) / m) * phase

            dA_dt = V
            dV_dt = -2 * (gamma - 1j*omega_c) * V \
                    - (omega0**2 - omega_c**2 - 2j*omega_c*gamma) * A \
                    + drive

            return [dA_dt.real, dA_dt.imag, dV_dt.real, dV_dt.imag]

        y0 = [A0.real, A0.imag, A1.real, A1.imag]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            rtol=rtol,
            atol=atol,
        )

        A_t = sol.y[0] + 1j * sol.y[1]
        V_t = sol.y[2] + 1j * sol.y[3]

        return (A_t, V_t) if return_V else A_t

    def solve_envelope_green(
        self, t, B_of_t, omega_d_of_t,
        omega_c=None,
        A0=0.0 + 0.0j,
        A1=0.0 + 0.0j,
        return_V=False,
        rtol=1e-8,
        atol=1e-10,
    ):
        r"""
        Fast second-order envelope solver via causal Green's function (FFT convolution).
    
        Solves:
            A¨(t) + 2(γ - i ω_c) Ȧ(t)
                  + (ω0^2 - ω_c^2 - 2 i γ ω_c) A(t)
            = (B(t)/m) * exp[-i φ(t)],
    
        where φ(t) = ∫ Δ(t) dt,  Δ(t)=ω_d(t)-ω_c.
    
        Notes
        -----
        - Works with complex B_of_t (envelope with quadrature).
        - Works with time-dependent omega_d_of_t via φ(t).
        - Requires *uniform* t for the FFT path; otherwise falls back to solve_ivp.
        """
    
        import numpy as np
        from scipy.integrate import solve_ivp  # fallback only
    
        t = np.asarray(t, dtype=float)
        B_of_t = np.asarray(B_of_t)
        omega_d_of_t = np.asarray(omega_d_of_t, dtype=float)
    
        if omega_c is None:
            omega_c = self.omega0
    
        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m
    
        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a 1D array with at least 2 points")
        if B_of_t.shape != t.shape:
            raise ValueError("B_of_t must have the same shape as t")
        if omega_d_of_t.shape != t.shape:
            raise ValueError("omega_d_of_t must have the same shape as t")
    
        dt = t[1] - t[0]
        uniform = np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-15)
    
        # ------------------------------------------------------------------
        # If grid is not uniform, fall back to your original ODE integrator.
        # ------------------------------------------------------------------
        if not uniform:
            Delta_of_t = omega_d_of_t - omega_c
    
            # trapezoidal phase on the grid (same convention as your original code)
            phi = np.zeros_like(t, dtype=float)
            phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * np.diff(t))
    
            def B_interp(tq):
                Br = np.interp(tq, t, B_of_t.real)
                Bi = np.interp(tq, t, B_of_t.imag)
                return Br + 1j * Bi
    
            def phi_interp(tq):
                return np.interp(tq, t, phi)
    
            def ode(tq, y):
                A = y[0] + 1j * y[1]
                V = y[2] + 1j * y[3]
                drive = (B_interp(tq) / m) * np.exp(-1j * phi_interp(tq))
    
                dA_dt = V
                dV_dt = (
                    -2 * (gamma - 1j * omega_c) * V
                    - (omega0**2 - omega_c**2 - 2j * omega_c * gamma) * A
                    + drive
                )
                return [dA_dt.real, dA_dt.imag, dV_dt.real, dV_dt.imag]
    
            y0 = [A0.real, A0.imag, A1.real, A1.imag]
            sol = solve_ivp(ode, (t[0], t[-1]), y0=y0, t_eval=t, rtol=rtol, atol=atol)
            A_t = sol.y[0] + 1j * sol.y[1]
            V_t = sol.y[2] + 1j * sol.y[3]
            return (A_t, V_t) if return_V else A_t
    
        # ------------------------------------------------------------------
        # FFT-Green fast path (uniform grid)
        # ------------------------------------------------------------------
    
        # Δ(t) and φ(t)=∫Δ dt (keep trapezoid to match your previous behavior)
        Delta_of_t = omega_d_of_t - omega_c
        phi = np.zeros_like(t, dtype=float)
        phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * dt)
    
        # source term s(t) = (B(t)/m) e^{-i φ(t)}
        s = (B_of_t / m) * np.exp(-1j * phi)
    
        # Green's function for:
        #   A¨ + 2(γ - i ω_c) Ȧ + (ω0^2 - ω_c^2 - 2 i γ ω_c) A = δ(t)
        # has:
        #   g(t)=Θ(t) e^{-(γ - i ω_c)t} sin(Ω t)/Ω
        # with Ω = sqrt(ω0^2 - γ^2) (complex allowed)
        a = gamma - 1j * omega_c
        Omega = np.sqrt(omega0**2 - gamma**2 + 0j)
    
        tau = t - t[0]
        if np.abs(Omega) == 0:
            # critical damping: sin(Ωt)/Ω -> t
            g = np.exp(-a * tau) * tau
            gdot = np.exp(-a * tau) * (1.0 - a * tau)
        else:
            g = np.exp(-a * tau) * (np.sin(Omega * tau) / Omega)
            gdot = np.exp(-a * tau) * (np.cos(Omega * tau) - (a / Omega) * np.sin(Omega * tau))
    
        # causal convolution via FFT: conv[n] ≈ dt * Σ_{k<=n} g[n-k] s[k]
        N = t.size
        L = 1 << int(np.ceil(np.log2(2 * N - 1)))
    
        S = np.fft.fft(s, n=L)
        G = np.fft.fft(g, n=L)
        Gd = np.fft.fft(gdot, n=L)
    
        A_forced = dt * np.fft.ifft(S * G)[:N]
        V_forced = dt * np.fft.ifft(S * Gd)[:N]
    
        # homogeneous solution enforcing A(0)=A0, A'(0)=A1
        if np.abs(Omega) == 0:
            # A_hom = e^{-a t} (C + D t)
            C = A0
            D = A1 + a * A0
            A_hom = np.exp(-a * tau) * (C + D * tau)
            V_hom = np.exp(-a * tau) * (D - a * (C + D * tau))
        else:
            C = A0
            D = (A1 + a * A0) / Omega
            cos = np.cos(Omega * tau)
            sin = np.sin(Omega * tau)
            exp = np.exp(-a * tau)
    
            A_hom = exp * (C * cos + D * sin)
            V_hom = exp * (-a * (C * cos + D * sin) + (-C * Omega * sin + D * Omega * cos))
    
        A_t = A_hom + A_forced
        V_t = V_hom + V_forced
    
        return (A_t, V_t) if return_V else A_t


# ---------------------------------------------------------------
# BAW mode class
# ---------------------------------------------------------------
from math import erf, sqrt, pi


class BAWMode:
    """
    Represents a single bulk-acoustic-wave (BAW) mode coupled to a
    gravitational strain h_+(t). Provides:
        - geometry factor ξ_λ
        - wrapper to compute B(t), Bdot(t), I(t)
        - optional helpers (e.g., generating chirp signals)
    """

    def __init__(
        self,
        n=3,
        d=1e-3,
        eta_x=0.1,
        eta_y=0.1,
        omega_lambda=2 * np.pi * 5e6,
        Q=1e7,
        k_lambda=1e-2,
    ):
        self.n = n
        self.d = d
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.omega_lambda = omega_lambda
        self.Q = Q
        self.k_lambda = k_lambda

        self.gamma_lambda = omega_lambda / Q


    def baw_specs_text(self):
        """
        Return a formatted text block listing all public attributes
        of the BAW mode object.
        """
        lines = []
        for name, value in vars(self).items():
            if not name.startswith("_"):
                lines.append(f"{name}: {value}")
        return "\n".join(lines)
    # -----------------------------------------------------------
    # Geometry factor ξ_λ
    # -----------------------------------------------------------
    def xi_lambda(self):
        """
        ξ_λ = (4 d / (n π)) *
              [ Erf(√n η_x) Erf(√n η_y) /
                (Erf(√(2n) η_x) Erf(√(2n) η_y)) ]
        """
        n = self.n
        ex = self.eta_x
        ey = self.eta_y
        d  = self.d

        num = erf(sqrt(n) * ex) * erf(sqrt(n) * ey)
        den = erf(sqrt(2 * n) * ex) * erf(sqrt(2 * n) * ey)
        return (4.0 * d / (n * pi)) * (num / den)

    # -----------------------------------------------------------
    # Solve B, Bdot, I for a given h_+(t)
    # -----------------------------------------------------------
    def solve_current(self, t, h_plus, use_green=True):
        """
        Wrapper that mirrors solve_baw_current().
        Returns:
            B(t), Bdot(t), I(t), f_drive(t)
        """

        dt = t[1] - t[0]

        # geometry factor
        xi = self.xi_lambda()

        # derivatives of h
        h_dot  = np.gradient(h_plus, dt)
        h_ddot = np.gradient(h_dot, dt)

        # driving force
        # B¨ + γ B˙ + ω^2 B = (1/2) ξ h¨_+
        f_drive = 0.5 * xi * h_ddot

        # oscillator
        osc = DrivenHarmonicOscillator(
            m=1.0,
            gamma=self.gamma_lambda,
            omega0=self.omega_lambda,
        )

        # solve
        if use_green:
            B = osc.solve_via_green(t, f_drive)
        else:
            B = osc.solve_direct_ode(t, f_drive)

        Bdot = np.gradient(B, dt)
        I = self.k_lambda * Bdot

        return B, Bdot, I, f_drive

    def solve_current_envelope(
        self,
        t,
        h_plus,
        omega_d_of_t=None,
        omega_c=None,
        A0=0.0 + 0.0j,
        A1=0.0 + 0.0j,
        use_hilbert=True,
        rescale=True,
    ):
        t = np.asarray(t, dtype=float)
        h_plus = np.asarray(h_plus, dtype=float)
        dt = t[1] - t[0]
    
        if omega_c is None:
            omega_c = self.omega_lambda
    
        if omega_d_of_t is None:
            omega_d_of_t = omega_c * np.ones_like(t)
        else:
            omega_d_of_t = np.asarray(omega_d_of_t, dtype=float)
            if omega_d_of_t.ndim == 0:
                omega_d_of_t = float(omega_d_of_t) * np.ones_like(t)
            if omega_d_of_t.shape != t.shape:
                raise ValueError("omega_d_of_t must be None, scalar, or same shape as t")
    
        omega0 = self.omega_lambda
        gamma  = self.gamma_lambda
        xi     = self.xi_lambda()
    
        # --- build a(t) from the analytic strain (slow) ---
        if use_hilbert:
            a_t = np.abs(hilbert(h_plus))
        else:
            a_t = np.abs(h_plus)
    
        a_dot  = np.gradient(a_t, dt)
        a_ddot = np.gradient(a_dot, dt)
        omega_dot = np.gradient(omega_d_of_t, dt)
    
        # phase θ(t)=∫ω_d(t)dt
        theta = np.zeros_like(t)
        theta[1:] = np.cumsum(0.5 * (omega_d_of_t[1:] + omega_d_of_t[:-1]) * np.diff(t))
    
        c = np.cos(theta)
        s = np.sin(theta)
    
        # analytic ḧ using h=a cosθ, includes chirp terms
        h_ddot = (
            a_ddot * c
            - (2.0 * a_dot * omega_d_of_t + a_t * omega_dot) * s
            - a_t * (omega_d_of_t**2) * c
        )
    
        # physical forcing (real)
        f_drive = 0.5 * xi * h_ddot
    
        # --- exact complex slow force envelope in the drive frame ---
        # f_+(t) ≈ F_env(t) e^{-iθ(t)}  =>  F_env = f_+(t) e^{+iθ(t)}
        if use_hilbert:
            f_plus = hilbert(f_drive)
        else:
            f_plus = f_drive.astype(complex)
    
        F_env = f_plus * np.exp(+1j * theta)   # complex, slow (contains the missing quadrature)
    
        # optional conditioning (recommended for tiny strains)
        scale = 1.0
        if rescale:
            scale = np.max(np.abs(F_env))
            if scale == 0.0:
                scale = 1.0
            F_env = F_env / scale
            A0 = A0 / scale
            A1 = A1 / scale
    
        # solve envelope ODE 
        osc = DrivenHarmonicOscillator(m=1.0, gamma=gamma, omega0=omega0)
        A_t, V_t = osc.solve_envelope_green(
            t=t,
            B_of_t=F_env,                # NOTE: now complex
            omega_d_of_t=omega_d_of_t,
            omega_c=omega_c,
            A0=A0,
            A1=A1,
            return_V=True,
            rtol=1e-8,
            atol=1e-11
        )
    
        if rescale:
            A_t = scale * A_t
            V_t = scale * V_t
            F_env = scale * F_env
    
        I_env = self.k_lambda * (V_t - 1j * omega_c * A_t)
        return A_t, V_t, I_env, f_drive, F_env


    # -----------------------------------------------------------
    # Example GW chirp with user-defined envelope
    # -----------------------------------------------------------
    def linear_chirp_strain(
        self,
        t,
        A0=1e-21,
        f_start=4.5e6,
        f_end=5.5e6,
        phi0=0.0,
        envelope=None,
    ):
        r"""
        h_+(t) = A_0 E(t) \cos(\phi(t) + \phi_0),
    
        where the instantaneous frequency is linear in time:
        f(t) = f_start + (f_end - f_start) * (t - t_0) / T
    
        E(t) is a user-supplied envelope:
          - If envelope is None: E(t) = 1 (flat).
          - If envelope is a callable: E(t) = envelope(t).
          - If envelope is an array: E(t) = np.asarray(envelope), must match t.
        """
        t = np.asarray(t)
        dt = t[1] - t[0]
        t0 = t[0]
        T = t[-1] - t0
    
        # Instantaneous frequency (linear chirp)
        f_t = f_start + (f_end - f_start) * (t - t0) / T
        omega_t = 2 * np.pi * f_t
    
        # Phase from numerical integration of omega(t)
        phi = np.cumsum(omega_t) * dt
        phi -= phi[0]
    
        # Envelope handling
        if envelope is None:
            env = np.ones_like(t)
        elif callable(envelope):
            env = np.asarray(envelope(t))
        else:
            env = np.asarray(envelope)
            if env.shape != t.shape:
                raise ValueError("envelope array must have the same shape as t")
    
        return A0 * env * np.cos(phi + phi0)


def time_at_frequency(t, f_t, f_target):
    """
    Given arrays t and f_t (same length), return the time(s) at which
    f_t(t) = f_target, using linear interpolation.

    f_target can be a scalar or array.
    """
    t = np.asarray(t)
    f_t = np.asarray(f_t)
    f_target = np.asarray(f_target, dtype=float)

    # Interpolant t(f)
    # We need frequency as x and time as y
    idx = np.argsort(f_t)
    f_sorted = f_t[idx]
    t_sorted = t[idx]

    # Interpolate times at target frequencies
    t_at = np.interp(f_target, f_sorted, t_sorted, left=np.nan, right=np.nan)

    # Preserve scalar input
    if np.ndim(f_target) == 0:
        return float(t_at)
    return t_at


def chirp_with_window(t, f_start, f_end, t_start, t_end, envelope=None, phi0=0.0):
    """
    Linear chirp active only in [t_start, t_end].

    Returns signal and instantaneous frequency array.
    """
    if envelope is None:
        envelope = np.ones_like(t)

    dt = t[1] - t[0]

    # initialize frequency array
    f_t = np.zeros_like(t)

    # chirp mask
    mask = (t >= t_start) & (t <= t_end)
    dT = t_end - t_start

    # linear frequency sweep in the window
    f_t[mask] = f_start + (f_end - f_start) * (t[mask] - t_start) / dT

    f_t[(t < t_start)] = f_start
    f_t[(t > t_end)] = f_end

    # angular frequency
    omega_t = 2 * np.pi * f_t

    # phase (continuous)
    phase = phi0 + np.cumsum(omega_t) * dt

    signal = envelope * np.cos(phase)
    return signal, f_t

def chirp_force(t, A=1.0, omega_min=0.5, omega_max=5.0, t0=5.0, sigma=3.0):
    """
    Smooth linear chirp with Gaussian envelope.
    
    Instantaneous frequency sweeps from omega_min to omega_max over the full interval.
    """
    # compute chirp rate α from ω(t) = ω_min + 2α t
    alpha = (omega_max - omega_min) / (2 * t[-1])
    
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    phase = omega_min * t + alpha * t**2
    
    return A * envelope * np.cos(phase)