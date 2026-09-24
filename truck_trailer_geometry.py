"""
truck_trailer_geometry.py
--------------------------
Obstacle-avoidance geometry, translated from TruckTrailerIneqConFcn.m.

IMPORTANT FIDELITY NOTE
------------------------
The original MATLAB function calls a Robotics System Toolbox GJK routine
(`controllib.internal.gjk.Base2d.checkCollision`) to get the *exact*
minimum distance between two convex polygons (truck/trailer rectangles vs.
obstacle rectangles). That routine is proprietary MATLAB code and has no
symbolic-differentiable counterpart usable inside a CasADi/IPOPT NLP.

Here it is replaced with a standard, differentiable oriented-rectangle
separation distance computed by projecting the center-to-center vector
onto itself and subtracting each rectangle's "support" (half-width along
that direction). This is the same idea used by the Separating Axis
Theorem and is exact for the case here (large axis-aligned obstacles,
small rotated truck/trailer rectangles clearly to one side), and behaves
smoothly enough for gradient-based NLP solvers -- similar in spirit to
why the MATLAB example itself disables the analytical Jacobian for this
constraint and falls back to finite differences (the exact GJK distance
is not smooth everywhere either).

If you need an exact, general-purpose OBB-OBB distance, consider a full
4-axis SAT formulation or `shapely` for offline verification (not
differentiable, so not usable inside the NLP itself).
"""

import casadi as ca


def rect_support(half_extent, theta, direction):
    """Support (max projection of the rectangle's half-extent) of a
    rectangle centered at the origin, rotated by `theta`, along a unit
    `direction` vector. half_extent = (hx, hy)."""
    hx, hy = half_extent
    dx, dy = direction[0], direction[1]
    # Project direction into the rectangle's local (rotated) frame.
    local_dx = ca.cos(theta) * dx + ca.sin(theta) * dy
    local_dy = -ca.sin(theta) * dx + ca.cos(theta) * dy
    return hx * ca.fabs(local_dx) + hy * ca.fabs(local_dy)


def rect_rect_gap(c1, theta1, half_extent1, c2, theta2, half_extent2):
    """Separating Axis Theorem (SAT) gap between two oriented rectangles.

    Two convex polygons are disjoint iff a separating axis exists among the
    edge normals of either polygon (4 candidate axes here, since both
    shapes are rectangles). For each candidate axis a (unit vector), the
    projected gap is:

        gap_a = |dot(c2 - c1, a)| - support1(a) - support2(a)

    and the polygons are guaranteed separated by at least max_a(gap_a)
    along that axis. Taking the max over the 4 axes gives a tight,
    differentiable, and *correct-signed* (positive iff truly separated)
    surrogate for the true minimum distance -- unlike projecting only onto
    the center-to-center direction (which is not always a valid separating
    axis and can under/over-estimate separation for elongated rectangles).
    """
    d = c2 - c1

    def axis_gap(axis):
        proj = ca.fabs(d[0] * axis[0] + d[1] * axis[1])
        s1 = rect_support(half_extent1, theta1, axis)
        s2 = rect_support(half_extent2, theta2, axis)
        return proj - s1 - s2

    axes = [
        ca.vertcat(ca.cos(theta1), ca.sin(theta1)),    # rect1 local x-axis
        ca.vertcat(-ca.sin(theta1), ca.cos(theta1)),   # rect1 local y-axis
        ca.vertcat(ca.cos(theta2), ca.sin(theta2)),    # rect2 local x-axis
        ca.vertcat(-ca.sin(theta2), ca.cos(theta2)),   # rect2 local y-axis
    ]
    gaps = [axis_gap(a) for a in axes]
    gap = gaps[0]
    for g in gaps[1:]:
        gap = ca.fmax(gap, g)
    return gap


class Obstacle:
    def __init__(self, x, y, width, height, theta=0.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.theta = theta

    @property
    def half_extent(self):
        return (self.width / 2.0, self.height / 2.0)


def default_obstacles():
    """Two static obstacles flanking the parking spot (matches the MATLAB
    example: two 17.5 x 20 m rectangles centered at (+/-11.25, -20))."""
    obs1 = Obstacle(x=-11.25, y=-20, width=17.5, height=20, theta=0.0)
    obs2 = Obstacle(x=11.25, y=-20, width=17.5, height=20, theta=0.0)
    return [obs1, obs2]


def truck_trailer_footprints(x, params):
    """Given state x = [x, y, theta, beta] (trailer rear-axle state),
    compute (center, heading) for the trailer and the truck bodies,
    matching the geometry used in TruckTrailerIneqConFcn.m."""
    M, L1, L2 = params.M, params.L1, params.L2

    trailer_x = x[0] + L2 / 2 * ca.cos(x[2])
    trailer_y = x[1] + L2 / 2 * ca.sin(x[2])
    trailer_theta = x[2]

    theta1 = x[2] + x[3]
    truck_x = x[0] + L2 * ca.cos(x[2]) + (M + L1 / 2) * ca.cos(theta1)
    truck_y = x[1] + L2 * ca.sin(x[2]) + (M + L1 / 2) * ca.sin(theta1)
    truck_theta = theta1

    trailer = (ca.vertcat(trailer_x, trailer_y), trailer_theta)
    truck = (ca.vertcat(truck_x, truck_y), truck_theta)
    return truck, trailer


def collision_constraints(x, params, obstacles, safety_distance=1.0):
    """Returns a list of inequality-constraint expressions g(x) <= 0, one
    per (body, obstacle) pair, matching `cineq = -distance + safetyDistance`
    in TruckTrailerIneqConFcn.m."""
    truck, trailer = truck_trailer_footprints(x, params)
    truck_half = (params.L1 / 2.0, params.W1 / 2.0)
    trailer_half = (params.L2 / 2.0, params.W2 / 2.0)

    g = []
    for obs in obstacles:
        obs_center = ca.vertcat(obs.x, obs.y)
        gap_trailer = rect_rect_gap(trailer[0], trailer[1], trailer_half,
                                     obs_center, obs.theta, obs.half_extent)
        g.append(safety_distance - gap_trailer)  # <= 0

    for obs in obstacles:
        obs_center = ca.vertcat(obs.x, obs.y)
        gap_truck = rect_rect_gap(truck[0], truck[1], truck_half,
                                   obs_center, obs.theta, obs.half_extent)
        g.append(safety_distance - gap_truck)  # <= 0

    return g
