"""
truck_trailer_initial_guess.py
--------------------------------
Translated from TruckTrailerInitialGuess.m.

fmincon/IPOPT only find a *local* optimum, so a reasonable initial guess
matters a lot for this nonconvex parking problem. This reproduces the
MATLAB heuristic: one waypoint if starting above the obstacles, two
waypoints (going around) if starting below/beside them.
"""

import numpy as np


def truck_trailer_initial_guess(initial_pose, target_pose, u0, p):
    """
    Returns:
        X0  : (4, p+1) ndarray, state guess at each stage (0..p)
        U0  : (2, p)   ndarray, input guess at each stage (0..p-1)
        XY0 : (p+1, 2) ndarray, just the x-y guess (for plotting)
    """
    initial_pose = np.asarray(initial_pose, dtype=float).flatten()
    target_pose = np.asarray(target_pose, dtype=float).flatten()
    u0 = np.asarray(u0, dtype=float).flatten()

    if initial_pose[1] >= -5:
        # truck starts above the obstacles -> one waypoint
        p1 = round(p / 2)
        p2 = p - p1 + 1
        x_mid, y_mid, th_mid = 0.0, 10.0, np.pi / 4

        x_guess = np.concatenate([
            np.linspace(initial_pose[0], x_mid, p1),
            np.linspace(x_mid, target_pose[0], p2),
        ])
        y_guess = np.concatenate([
            np.linspace(initial_pose[1], y_mid, p1),
            np.linspace(y_mid, target_pose[1], p2),
        ])
        th_guess = np.concatenate([
            np.linspace(initial_pose[2], th_mid, p1),
            np.linspace(th_mid, target_pose[2], p2),
        ])
    else:
        # truck starts below the obstacles -> two waypoints (go around)
        p1 = round(p / 3)
        p2 = round(p / 3)
        p3 = p - p1 - p2 + 1
        sign = np.sign(initial_pose[0]) if initial_pose[0] != 0 else 1.0
        x1, y1, th1 = initial_pose[0] + sign * 10, 10.0, np.pi / 6
        x2, y2, th2 = 0.0, 10.0, np.pi / 3

        x_guess = np.concatenate([
            np.linspace(initial_pose[0], x1, p1),
            np.linspace(x1, x2, p2),
            np.linspace(x2, target_pose[0], p3),
        ])
        y_guess = np.concatenate([
            np.linspace(initial_pose[1], y1, p1),
            np.linspace(y1, y2, p2),
            np.linspace(y2, target_pose[1], p3),
        ])
        th_guess = np.concatenate([
            np.linspace(initial_pose[2], th1, p1),
            np.linspace(th1, th2, p2),
            np.linspace(th2, target_pose[2], p3),
        ])

    beta_guess = np.zeros(p + 1)
    # Note: due to rounding of p1/p2/(p3), concatenated length may be
    # p+2 (two segments) or p+3 (three segments) minus overlaps; MATLAB's
    # `linspace` segments share endpoints, so trim/pad to exactly p+1.
    X0 = np.vstack([x_guess, y_guess, th_guess])[:, : p + 1]
    if X0.shape[1] < p + 1:
        pad = np.tile(X0[:, -1:], (1, p + 1 - X0.shape[1]))
        X0 = np.hstack([X0, pad])
    X0 = np.vstack([X0, beta_guess])

    U0 = np.tile(u0.reshape(2, 1), (1, p))
    XY0 = X0[0:2, :].T
    return X0, U0, XY0
