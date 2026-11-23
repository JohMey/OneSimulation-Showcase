"""OneSimulation example: the speed of a small shallow-water wave.
Requires the installed, activated OneSimulation runtime.

The measured front speed is checked against the shallow-water celerity sqrt(g*h).
"""

# # Shallow-water waves: how fast a small disturbance travels
#
# A shallow lake of still-water depth $h$ is disturbed by a small, smooth ridge
# on its surface. The ridge immediately splits into two pulses that travel
# outward in opposite directions. For a small disturbance the shallow-water
# (Saint-Venant) equations predict that each pulse moves at the celerity
#
# $$c = \sqrt{g\,h},$$
#
# independent of its shape or amplitude. We solve the full nonlinear
# shallow-water system and check the measured pulse speed against this formula.
#
# OneSimulation integrates the conservative 2-D shallow-water equations
#
# $$\partial_t h + \nabla\cdot(h\,\mathbf u)=0,\qquad
#   \partial_t(h\,\mathbf u) + \nabla\cdot(h\,\mathbf u\otimes\mathbf u)
#   + \nabla\!\big(\tfrac12 g h^2\big) = 0,$$
#
# with a first-order Rusanov finite-volume scheme on a periodic unit-square mesh.
# The initial ridge depends only on $x$, so the flow stays one-dimensional and
# produces one left-going and one right-going pulse.

import numpy as np
from onesim.meshing import TensorCollocationMesh
from onesim.numerics.time import ForwardEulerTimeIntegrator
from onesim.physics.hydrodynamics import (
    PeriodicShallowWaterRusanov,
    ShallowWater2DSolver,
)
from onesim.visualization import plots, theme

theme.set()


# ## Parameters
#
# The unit square carries a still-water depth $h$. A narrow Gaussian ridge of
# tiny amplitude (one part in a thousand of the depth) sits at the centre. The
# final time is chosen so the two pulses travel well into the domain without
# reaching the periodic boundary, so no wrap-around contaminates the measurement.

g = 9.81
h = 1.0
amplitude = 1.0e-3
sigma = 0.03
x0 = 0.5
n = 200
final_time = 0.10

c_formula = np.sqrt(g * h)
print(
    f"predicted celerity c = sqrt(g*h) = {c_formula:.4f} m/s,  c*t = {c_formula*final_time:.3f}"
)


def initial_state(x, y):
    depth = h + amplitude * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
    return depth, np.zeros_like(x), np.zeros_like(y)


axis = (np.arange(n) + 0.5) / n
mesh = TensorCollocationMesh(
    axes=(axis, axis), coordinate_system="cartesian", periodic=(True, True)
)
result = ShallowWater2DSolver(
    mesh=mesh,
    spatial=PeriodicShallowWaterRusanov(),
    time_integrator=ForwardEulerTimeIntegrator(),
    gravity=g,
    final_time=final_time,
    cfl=0.4,
).run(initial_state)
print(f"Integrated {result.step_count} explicit steps to t = {result.time:.4f} s")


# ## Measure the pulse speed
#
# The flow is uniform in $y$, so we average along $y$ and work with the surface
# deviation $\delta h(x)=h(x)-h$. Each outgoing pulse keeps its shape, so the
# centroid of $|\delta h|$ on each half of the domain marks the pulse position.
# Dividing the travelled distance by the elapsed time gives the measured speed,
# which we compare with $c=\sqrt{g h}$.

delta_h = result.depth.mean(axis=1) - h
weight = np.abs(delta_h)

right = axis > x0
left = axis < x0
centroid_right = np.sum(axis[right] * weight[right]) / np.sum(weight[right])
centroid_left = np.sum(axis[left] * weight[left]) / np.sum(weight[left])

speed_right = (centroid_right - x0) / result.time
speed_left = (x0 - centroid_left) / result.time
speed_measured = 0.5 * (speed_right + speed_left)
rel_error = abs(speed_measured - c_formula) / c_formula

print(f"right-going pulse speed = {speed_right:.4f} m/s")
print(f"left-going  pulse speed = {speed_left:.4f} m/s")
print(f"measured celerity       = {speed_measured:.4f} m/s")
print(f"relative error vs sqrt(g*h) = {100*rel_error:.2f} %")


# ## Figure

predicted = [x0 - c_formula * result.time, x0 + c_formula * result.time]
plots.line(
    axis,
    {"computed surface deviation": delta_h * 1000.0},
    points={
        "predicted x0 ± sqrt(g h) t": (
            predicted,
            np.interp(predicted, axis, delta_h * 1000.0),
        )
    },
    vlines={
        "measured left pulse": centroid_left,
        "measured right pulse": centroid_right,
    },
    xlabel="position x [m]",
    ylabel="surface deviation [mm]",
    title=f"Shallow-water pulse: measured c = {speed_measured:.3f} m/s, sqrt(g h) = {c_formula:.3f} m/s",
)
plots.show()


# ## Going further
#
# Lengthen `final_time` (keeping the pulses inside the domain) or refine the mesh
# with a larger `n` and watch the measured speed settle ever closer to
# $\sqrt{g h}$. Change the still-water depth `h` and confirm the celerity scales
# as its square root. Raise `amplitude` towards the depth and the leading edge
# begins to steepen into a bore, where the small-disturbance formula no longer
# applies.
