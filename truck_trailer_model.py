"""
truck_trailer_model.py
-----------------------
Continuous-time kinematic model of a truck with a single trailer,
translated from TruckTrailerStateFcn.m / TruckTrailerStateJacobianFcn.m.

States   x = [x, y, theta, beta]
    x, y   : global position of the trailer's rear-axle center [m]
    theta  : trailer heading angle (0 = east) [rad]
    beta   : truck heading relative to trailer (0 = aligned) [rad]

Inputs   u = [alpha, v]
    alpha  : truck steering angle [rad]
    v      : truck longitudinal velocity [m/s]

Parameters p = [M, L1, L2]
    M  : hitch length [m]
    L1 : truck length [m]
    L2 : trailer length [m]

CasADi builds the Jacobian automatically via algorithmic differentiation,
so a hand-written Jacobian (as in TruckTrailerStateJacobianFcn.m) is not
needed here -- CasADi's AD is exact and free, which is one advantage of
using it in place of fmincon's finite-difference / user-supplied Jacobian.
"""

import casadi as ca


def truck_trailer_ode(x, u, p):
    """Continuous-time dynamics dx/dt = f(x, u, p). All args are CasADi
    symbols (or numeric), returns a CasADi column vector of length 4."""
    theta = x[2]
    beta = x[3]
    alpha = u[0]
    v = u[1]
    M, L1, L2 = p[0], p[1], p[2]

    common = v * ca.cos(beta) * (1 + M / L1 * ca.tan(beta) * ca.tan(alpha))

    dxdt = ca.vertcat(
        common * ca.cos(theta),
        common * ca.sin(theta),
        v * (ca.sin(beta) / L2 - M / (L1 * L2) * ca.cos(beta) * ca.tan(alpha)),
        v * (ca.tan(alpha) / L1 - ca.sin(beta) / L2
             + M / (L1 * L2) * ca.cos(beta) * ca.tan(alpha)),
    )
    return dxdt


def rk4_step(x, u, p, Ts, n_sub=1):
    """RK4 integration of truck_trailer_ode over one sample interval Ts,
    optionally split into n_sub sub-steps for extra accuracy (mirrors the
    default fixed-step RK4 integration used internally by nlmpcMultistage)."""
    h = Ts / n_sub
    xk = x
    for _ in range(n_sub):
        k1 = truck_trailer_ode(xk, u, p)
        k2 = truck_trailer_ode(xk + h / 2 * k1, u, p)
        k3 = truck_trailer_ode(xk + h / 2 * k2, u, p)
        k4 = truck_trailer_ode(xk + h * k3, u, p)
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xk


class Params:
    """Mirrors the `params` struct in the MATLAB example."""

    def __init__(self, M=1.0, L1=6.0, W1=2.5, L2=10.0, W2=2.5,
                 Lwheel=1.0, Wwheel=0.4):
        self.M = M
        self.L1 = L1
        self.W1 = W1
        self.L2 = L2
        self.W2 = W2
        self.Lwheel = Lwheel
        self.Wwheel = Wwheel
