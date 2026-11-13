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

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.physics.hydrodynamics.free_surface.numerics.equations.shallow_water.shallow_water_2d import (
    ShallowWater2DSolver,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The unit square carries a still-water depth $h$. A narrow Gaussian ridge of
# tiny amplitude (one part in a thousand of the depth) sits at the centre. The
# final time is chosen so the two pulses travel well into the domain without
# reaching the periodic boundary, so no wrap-around contaminates the measurement.

g = 9.81              # m/s^2, gravitational acceleration
h = 1.0               # m, still-water depth
amplitude = 1.0e-3    # m, ridge height (small, so the wave stays linear)
sigma = 0.03          # width of the Gaussian ridge
x0 = 0.5              # centre of the domain
n = 200               # finite-volume cells per direction
final_time = 0.10     # s

c_formula = np.sqrt(g * h)
print(f"predicted celerity c = sqrt(g*h) = {c_formula:.4f} m/s,  c*t = {c_formula*final_time:.3f}")


def initial_state(x, y):
    depth = h + amplitude * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
    return depth, np.zeros_like(x), np.zeros_like(y)


result = ShallowWater2DSolver(
    mesh_size=n, gravity=g, final_time=final_time, cfl=0.4,
).run(initial_state)
print(f"Integrated {result.step_count} explicit steps to t = {result.time:.4f} s")


# ## Measure the pulse speed
#
# The flow is uniform in $y$, so we average along $y$ and work with the surface
# deviation $\delta h(x)=h(x)-h$. Each outgoing pulse keeps its shape, so the
# centroid of $|\delta h|$ on each half of the domain marks the pulse position.
# Dividing the travelled distance by the elapsed time gives the measured speed,
# which we compare with $c=\sqrt{g h}$.

axis = (np.arange(n) + 0.5) / n
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

# The two pulses must be mirror images (symmetric split) and the celerity must
# match the formula to within a couple of percent. The lake must stay wet and
# the disturbance must remain small and linear.
assert rel_error < 0.02, "Measured wave speed deviates more than 2 % from sqrt(g*h)."
assert abs(speed_right - speed_left) / c_formula < 0.02, "The split is not symmetric."
assert np.all(result.depth > 0.0), "The lake must stay wet."
assert np.max(np.abs(delta_h)) < 5.0 * amplitude, "The disturbance left the linear regime."
# A right-going linear wave carries positive momentum where the surface rises.
momentum_x = result.momentum_x.mean(axis=1)
assert momentum_x[np.argmax(delta_h * right)] > 0.0, "Right-going crest should carry flow outward."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
ax.plot(axis, delta_h * 1000.0, color="#1f5c73", lw=1.8, label="computed surface deviation")
for centroid, label in ((centroid_left, "left pulse"), (centroid_right, "right pulse")):
    ax.axvline(centroid, color="#b4642e", ls="--", lw=1.2)
for predicted in (x0 - c_formula * result.time, x0 + c_formula * result.time):
    ax.axvline(predicted, color="#4a7a4a", ls=":", lw=1.4)
ax.plot([], [], color="#b4642e", ls="--", lw=1.2, label="measured pulse centroid")
ax.plot([], [], color="#4a7a4a", ls=":", lw=1.4, label=r"predicted $x_0 \pm \sqrt{gh}\,t$")
ax.set(xlabel="Position x", ylabel="Surface deviation / mm",
       title=f"Shallow-water pulse: measured c = {speed_measured:.3f} m/s, "
             f"sqrt(g*h) = {c_formula:.3f} m/s")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Lengthen `final_time` (keeping the pulses inside the domain) or refine the mesh
# with a larger `n` and watch the measured speed settle ever closer to
# $\sqrt{g h}$. Change the still-water depth `h` and confirm the celerity scales
# as its square root. Raise `amplitude` towards the depth and the leading edge
# begins to steepen into a bore, where the small-disturbance formula no longer
# applies.
