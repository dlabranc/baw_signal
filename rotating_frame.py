import numpy as np
from scipy.integrate import simpson
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------
# Simple Driven Harmonic Oscillator Class
# ---------------------------------------------------------------
class DrivenHarmonicOscillator:
    """
    Driven harmonic oscillator:
        m x'' + m γ x' + m ω0^2 x = f(t)

    Green-function solution:
        G(t) = Θ(t) e^{-γ t / 2} sin(ω_d t) / (m ω_d)

    Rotating-frame / envelope solution:
        x(t) ≈ Re[ A(t) e^{-i ω_c t} ],
        where A(t) obeys a first-order ODE.
    """

    def __init__(self, m=1.0, gamma=0.2, omega0=1.0):
        self.m = m
        self.gamma = gamma
        self.omega0 = omega0
        self.omega_d = np.sqrt(omega0**2 - (gamma**2) / 4)

    def green(self, t):
        """
        Causal Green's function G(t).
        """
        G = np.zeros_like(t)
        mask = t >= 0
        G[mask] = (
            np.exp(-self.gamma * t[mask] / 2.0)
            * np.sin(self.omega_d * t[mask])
            / (self.m * self.omega_d)
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

        def ode(tq, y):
            x, v = y
            dxdt = v
            dvdt = (
                -self.gamma * v
                - self.omega0**2 * x
                + f_interp(tq) / self.m
            )
            return [dxdt, dvdt]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            y0=[0, 0],
            t_eval=t,
            rtol=1e-8,
            atol=1e-10
        )
        return sol.y[0]

    # -----------------------------------------------------------
    # Rotating-frame / envelope solver
    # -----------------------------------------------------------
    def solve_envelope(self, t, f_of_t, omega_c=None, A0=0.0 + 0.0j):
        r"""
        Rotating-frame / envelope solution for x(t).

        Assumes the drive is narrowband around a carrier ω_c:
            m x¨ + m γ x˙ + m ω0^2 x = f(t)

        We solve for the complex envelope A(t) such that:
            x(t) ≈ Re[ A(t) e^{-i ω_c t} ].

        Envelope equation (RWA / near-resonance):
            A˙(t) = (-γ/2 - iΔ) A(t)
                    + i f(t) e^{+i ω_c t} / (2 m ω_c),

            Δ = ω_c - ω0.

        Returns:
            x(t)          : approximate displacement (real)
            v(t)          : approximate velocity (real)
            A_t           : displacement envelope A(t) (complex)
            v_env(t)      : velocity envelope v_env(t) (complex),
                            such that v(t) ≈ Re[ v_env(t) e^{-i ω_c t} ].
        """
        t = np.asarray(t)
        f_of_t = np.asarray(f_of_t)
        if omega_c is None:
            omega_c = self.omega0

        omega_c = float(omega_c)
        Delta = omega_c - self.omega0
        m = self.m
        gamma = self.gamma

        # interpolation of the real forcing
        def f_interp(tq):
            return np.interp(tq, t, f_of_t)

        # ODE for the complex envelope A(t)
        def ode_env(tq, A):
            # A is a 1D array of length 1: A[0] is the complex envelope
            A0_local = A[0]
            f_val = f_interp(tq)
            drive_term = 1j * f_val * np.exp(1j * omega_c * tq) / (2.0 * m * omega_c)
            dA_dt = (-0.5 * gamma - 1j * Delta) * A0_local + drive_term
            # return as length-1 array
            return np.array([dA_dt], dtype=complex)

        # NOTE: y0 must be 1D; make it a length-1 complex array
        y0 = np.array([A0], dtype=complex)

        sol = solve_ivp(
            ode_env,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            rtol=1e-8,
            atol=1e-10
        )

        # sol.y has shape (1, N); take the first row
        A_t = sol.y[0]  # complex envelope A(t)

        # A˙(t) from the envelope equation (vectorized)
        drive_term = 1j * f_of_t * np.exp(1j * omega_c * t) / (2.0 * m * omega_c)
        A_dot = (-0.5 * gamma - 1j * Delta) * A_t + drive_term

        # velocity envelope: v_env(t) such that v(t) = Re[v_env e^{-i ω_c t}]
        v_env = A_dot - 1j * omega_c * A_t

        # reconstruct physical x(t), v(t)
        carrier = np.exp(-1j * omega_c * t)
        x = np.real(A_t * carrier)
        v = np.real(v_env * carrier)

        return x, v, A_t, v_env


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
    # Solve B, Bdot, I for a given h_+(t) (full solution)
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
            Bdot = np.gradient(B, dt)
        else:
            B = osc.solve_direct_ode(t, f_drive)
            Bdot = np.gradient(B, dt)

        I = self.k_lambda * Bdot

        return B, Bdot, I, f_drive

    # -----------------------------------------------------------
    # Solve in rotating frame: envelopes for B and I
    # -----------------------------------------------------------
    def solve_current_envelope(self, t, h_plus, omega_c=None, A0=0.0 + 0.0j):
        r"""
        Rotating-frame / envelope solution for the BAW current.

        Equation:
            B¨ + γ_λ B˙ + ω_λ^2 B = (1/2) ξ_λ h¨_+(t).

        We assume the response is narrowband around a carrier ω_c
        (default: ω_c = ω_λ) and write
            B(t) ≈ Re[ A(t) e^{-i ω_c t} ].

        Returns:
            B(t)      : approximate physical displacement (real)
            Bdot(t)   : approximate velocity (real)
            I(t)      : approximate current (real)
            f_drive(t): drive term (array)
            A_t       : displacement envelope A(t) (complex)
            I_env(t)  : current envelope I_env(t) (complex),
                        such that I(t) ≈ Re[ I_env(t) e^{-i ω_c t} ].
        """
        t = np.asarray(t)
        dt = t[1] - t[0]

        if omega_c is None:
            omega_c = self.omega_lambda

        # geometry factor
        xi = self.xi_lambda()

        # derivatives of h
        h_dot  = np.gradient(h_plus, dt)
        h_ddot = np.gradient(h_dot, dt)

        # driving force on B
        f_drive = 0.5 * xi * h_ddot

        # oscillator for B
        osc = DrivenHarmonicOscillator(
            m=1.0,
            gamma=self.gamma_lambda,
            omega0=self.omega_lambda,
        )

        # rotating-frame / envelope solution for B
        B, Bdot, A_t, v_env = osc.solve_envelope(t, f_drive, omega_c=omega_c, A0=A0)

        # current: I = k_λ B˙
        I = self.k_lambda * Bdot

        # envelope of the current:
        # B˙(t) = Re[ v_env(t) e^{-i ω_c t} ]  => I_env = k_λ v_env
        I_env = self.k_lambda * v_env

        return B, Bdot, I, f_drive, A_t, I_env

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


# -----------------------------------------------------------
# Utility: time at which a chirp hits a target frequency
# -----------------------------------------------------------
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
    idx = np.argsort(f_t)
    f_sorted = f_t[idx]
    t_sorted = t[idx]

    # Interpolate times at target frequencies
    t_at = np.interp(f_target, f_sorted, t_sorted, left=np.nan, right=np.nan)

    if np.ndim(f_target) == 0:
        return float(t_at)
    return t_at


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
