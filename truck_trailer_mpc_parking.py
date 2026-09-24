"""
truck_trailer_mpc_parking.py
------------------------------
Python translation of TruckAndTrailerParkingUsingMPCExample.m

MATLAB uses Model Predictive Control Toolbox's `nlmpcMultistage` object,
which internally builds a nonlinear program (NLP) and solves it with
`fmincon` (Optimization Toolbox). Python has no toolbox that reproduces
`nlmpcMultistage` directly, so this script builds the *same* NLP by hand
with CasADi (symbolic modeling + automatic differentiation) and solves it
with IPOPT, which plays the role of fmincon here. Conceptually the two are
solving the exact same trajectory-optimization problem:

    minimize   sum_{k=1}^{p} u_k' * u_k                       (parking speed)
    subject to x_{k+1} = f_RK4(x_k, u_k)   for k = 1..p        (truck-trailer dynamics)
               x_1 = current state (initial pose)
               x_{p+1} = target pose                           (terminal constraint)
               steering/speed bounds                           (input constraints)
               |beta| <= 90 deg                                (state constraint)
               obstacle clearance >= 1 m for k = 2..p           (inequality constraints)

Run:
    python truck_trailer_mpc_parking.py
"""

import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation, PillowWriter

from truck_trailer_model import Params, rk4_step
from truck_trailer_geometry import default_obstacles, collision_constraints
from truck_trailer_initial_guess import truck_trailer_initial_guess


def build_and_solve(initial_pose, target_pose, params, p=40, Ts=0.5,
                     safety_distance=1.0, verbose=True):
    obstacles = default_obstacles()
    p_vec = [params.M, params.L1, params.L2]

    opti = ca.Opti()
    X = opti.variable(4, p + 1)   # states across all stages 1..p+1
    U = opti.variable(2, p)       # inputs across stages 1..p

    # ---- initial state (stage 1) ----
    opti.subject_to(X[:, 0] == initial_pose)

    # ---- dynamics (multiple shooting, RK4) ----
    for k in range(p):
        x_next = rk4_step(X[:, k], U[:, k], p_vec, Ts, n_sub=1)
        opti.subject_to(X[:, k + 1] == x_next)

    # ---- input bounds: nlobj.MV(1/2).Min/Max ----
    opti.subject_to(opti.bounded(-np.pi / 4, U[0, :], np.pi / 4))   # steering angle
    opti.subject_to(opti.bounded(-10, U[1, :], 10))                 # velocity

    # ---- state bounds: nlobj.States(4).Min/Max ----
    opti.subject_to(opti.bounded(-np.pi / 2, X[3, :], np.pi / 2))   # beta

    # ---- terminal constraint: nlobj.Model.TerminalState ----
    opti.subject_to(X[:, p] == target_pose)

    # ---- stage cost: TruckTrailerCost.m (u' * u), stages 1..p ----
    cost = 0
    for k in range(p):
        cost += ca.dot(U[:, k], U[:, k])
    opti.minimize(cost)

    # ---- collision-avoidance inequality constraints, stages 2..p ----
    # (0-indexed columns 1..p-1, since column 0 = stage1 = fixed initial
    #  state and column p = stage p+1 = terminal state already constrained)
    for k in range(1, p):
        g_list = collision_constraints(X[:, k], params, obstacles, safety_distance)
        for g in g_list:
            opti.subject_to(g <= 0)

    # ---- initial guess ----
    u0 = np.zeros(2)
    X0, U0, XY0 = truck_trailer_initial_guess(initial_pose, target_pose, u0, p)
    opti.set_initial(X, X0)
    opti.set_initial(U, U0)

    opts = {
        "ipopt.print_level": 0 if not verbose else 5,
        "print_time": verbose,
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-6,
    }
    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()
    elapsed = time.time() - t0

    Xopt = sol.value(X).T          # (p+1, 4)
    Uopt = sol.value(U).T          # (p, 2)
    Topt = np.arange(p + 1) * Ts
    cost_val = float(sol.value(cost))

    info = {
        "Xopt": Xopt,
        "MVopt": Uopt,
        "Topt": Topt,
        "Cost": cost_val,
        "SolveTime": elapsed,
        "stats": sol.stats(),
    }
    return info, XY0, obstacles


# -------------------------------------------------------------------------
# Plotting / animation (Python equivalent of TruckTrailerPlot.m)
# -------------------------------------------------------------------------

def rectangle_patch(ax, center_x, center_y, theta, length, width, **kwargs):
    """Matplotlib Rectangle patch centered at (center_x, center_y), rotated
    by theta [rad], with total size length (x-extent) x width (y-extent).
    The rotation/translation must be composed with ax.transData so the
    rectangle is placed in *data* coordinates, not raw figure coordinates."""
    rect = patches.Rectangle((-length / 2, -width / 2), length, width, **kwargs)
    t = (transforms.Affine2D().rotate(theta).translate(center_x, center_y)
         + ax.transData)
    rect.set_transform(t)
    return rect


def expand_to_full_state(x_partial, params):
    """Given trailer state [x2,y2,theta2,beta], compute the truck's rear
    axle pose [x1,y1,theta1] (mirrors expandToFullState in TruckTrailerPlot.m)."""
    x2, y2, theta2, beta = x_partial
    M, L1, L2 = params.M, params.L1, params.L2
    theta1 = theta2 + beta
    x1 = x2 + M * np.cos(theta1) + L2 * np.cos(theta2)
    y1 = y2 + M * np.sin(theta1) + L2 * np.sin(theta2)
    return x1, y1, theta1


def plot_static(initial_pose, target_pose, params, obstacles,
                 info=None, XY0=None, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(initial_pose[0], initial_pose[1], 'rx', markersize=10, label='Initial pose')
    ax.plot(target_pose[0], target_pose[1], 'gx', markersize=10, label='Target pose')

    for obs in obstacles:
        rect = rectangle_patch(ax, obs.x, obs.y, obs.theta, obs.width, obs.height,
                                facecolor='0.7', edgecolor='k')
        ax.add_patch(rect)

    if info is not None and XY0 is not None:
        ax.plot(XY0[:, 0], XY0[:, 1], 'c.', label='Initial guess')
        ax.plot(info["Xopt"][:, 0], info["Xopt"][:, 1], 'bo', markersize=3, label='Optimal path')

        # draw truck & trailer footprints at a few snapshots along the path
        n = info["Xopt"].shape[0]
        snap_idx = np.linspace(0, n - 1, 6).astype(int)
        for i in snap_idx:
            xk = info["Xopt"][i]
            x1, y1, theta1 = expand_to_full_state(xk, params)
            trailer_rect = rectangle_patch(
                ax,
                xk[0] + params.L2 / 2 * np.cos(xk[2]),
                xk[1] + params.L2 / 2 * np.sin(xk[2]),
                xk[2], params.L2, params.W2,
                facecolor='none', edgecolor='m', alpha=0.6)
            truck_rect = rectangle_patch(
                ax,
                xk[0] + params.L2 * np.cos(xk[2]) + (params.M + params.L1 / 2) * np.cos(theta1),
                xk[1] + params.L2 * np.sin(xk[2]) + (params.M + params.L1 / 2) * np.sin(theta1),
                theta1, params.L1, params.W1,
                facecolor='none', edgecolor=(1, 0.5, 0), alpha=0.6)
            ax.add_patch(trailer_rect)
            ax.add_patch(truck_rect)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_states_and_inputs(info, save_path=None):
    T = info["Topt"]
    X = info["Xopt"]
    U = info["MVopt"]

    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    axs[0, 0].plot(T[:-1], U[:, 0]); axs[0, 0].set_title(r'$\alpha$ (rad)'); axs[0, 0].grid(True)
    axs[0, 1].plot(T[:-1], U[:, 1]); axs[0, 1].set_title('v (m/s)'); axs[0, 1].grid(True)
    axs[1, 0].plot(T, X[:, 0]); axs[1, 0].set_title('x (m)'); axs[1, 0].grid(True)
    axs[1, 1].plot(T, X[:, 1]); axs[1, 1].set_title('y (m)'); axs[1, 1].grid(True)
    axs[2, 0].plot(T, X[:, 2]); axs[2, 0].set_title(r'$\theta$ (rad)'); axs[2, 0].grid(True)
    axs[2, 1].plot(T, X[:, 3]); axs[2, 1].set_title(r'$\beta$ (rad)'); axs[2, 1].grid(True)
    fig.suptitle('Optimal Trajectory of MVs and States')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, axs


def animate_parking(initial_pose, target_pose, params, obstacles, info, XY0,
                     save_path=None, fps=10):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(initial_pose[0], initial_pose[1], 'rx', markersize=10)
    ax.plot(target_pose[0], target_pose[1], 'gx', markersize=10)
    for obs in obstacles:
        rect = rectangle_patch(ax, obs.x, obs.y, obs.theta, obs.width, obs.height,
                                facecolor='0.7', edgecolor='k')
        ax.add_patch(rect)
    ax.plot(XY0[:, 0], XY0[:, 1], 'c.', markersize=3)
    ax.plot(info["Xopt"][:, 0], info["Xopt"][:, 1], 'b-', linewidth=0.7, alpha=0.5)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Automated Parking Animation')

    truck_patch = patches.Rectangle((0, 0), params.L1, params.W1,
                                     facecolor=(1, 0.5, 0), edgecolor='k')
    trailer_patch = patches.Rectangle((0, 0), params.L2, params.W2,
                                       facecolor='m', edgecolor='k', alpha=0.7)
    ax.add_patch(truck_patch)
    ax.add_patch(trailer_patch)

    X = info["Xopt"]

    def update(i):
        xk = X[i]
        x1, y1, theta1 = expand_to_full_state(xk, params)

        t_truck = (transforms.Affine2D()
                   .translate(-params.L1 / 2, -params.W1 / 2)
                   .rotate(theta1).translate(x1, y1) + ax.transData)
        truck_patch.set_transform(t_truck)

        trailer_cx = xk[0] + params.L2 / 2 * np.cos(xk[2])
        trailer_cy = xk[1] + params.L2 / 2 * np.sin(xk[2])
        t_trailer = (transforms.Affine2D()
                     .translate(-params.L2 / 2, -params.W2 / 2)
                     .rotate(xk[2]).translate(trailer_cx, trailer_cy) + ax.transData)
        trailer_patch.set_transform(t_trailer)
        return truck_patch, trailer_patch

    anim = FuncAnimation(fig, update, frames=X.shape[0], interval=1000 / fps, blit=False)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=fps))
    return anim


if __name__ == "__main__":
    import os

    # Save outputs next to this script (portable across OS / machines),
    # instead of a hardcoded path.
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    params = Params(M=1, L1=6, W1=2.5, L2=10, W2=2.5, Lwheel=1, Wwheel=0.4)

    initial_pose = np.array([0, 0, 0, 0])
    target_pose = np.array([0.0, -25.0, np.pi / 2, 0.0])

    print("Automated Parking Planner is running...")
    info, XY0, obstacles = build_and_solve(initial_pose, target_pose, params,
                                            p=40, Ts=0.5, verbose=False)
    print(f"Calculation Time = {info['SolveTime']:.3f}s; "
          f"Objective cost = {info['Cost']:.4f}; "
          f"Return status = {info['stats']['return_status']}")

    fig1, _ = plot_static(initial_pose, target_pose, params, obstacles,
                           info=info, XY0=XY0,
                           title=f"Automated Parking (x0={initial_pose[0]}, y0={initial_pose[1]})",
                           save_path=os.path.join(out_dir, "parking_path.png"))
    fig2, _ = plot_states_and_inputs(info, save_path=os.path.join(out_dir, "parking_states_inputs.png"))

    animate_parking(initial_pose, target_pose, params, obstacles, info, XY0,
                     save_path=os.path.join(out_dir, "parking_animation.gif"), fps=10)

    print("Saved: parking_path.png, parking_states_inputs.png, parking_animation.gif")
