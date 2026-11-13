"""OneSimulation example: dam break on a dry horizontal bed (Ritter's solution).
Requires the installed, activated OneSimulation runtime.

The computed depth profile is checked against Ritter's closed-form dam-break solution.
"""


# # Dam break on a dry bed: Ritter's solution
#
# A column of still water of depth $h_0$ is held back by a wall on a flat,
# frictionless, dry bed. At $t=0$ the wall is removed instantaneously. The water
# collapses into a rarefaction wave: a wetting front races over the dry bed while a
# depression wave travels back into the reservoir.
#
# OneSimulation integrates the conservative 2-D shallow-water (Saint-Venant)
# equations
#
# $$\partial_t h + \nabla\cdot(h\,\mathbf u)=0,\qquad
#   \partial_t(h\,\mathbf u) + \nabla\cdot(h\,\mathbf u\otimes\mathbf u)
#   + \nabla\!\big(\tfrac12 g h^2\big) = 0,$$
#
# with a first-order Rusanov finite-volume scheme that admits dry cells. The dam is
# aligned with $y$, so the flow stays one-dimensional.
#
# **Reference (Ritter, 1892).** With $c_0=\sqrt{g h_0}$ and the dam at $x=x_d$,
# the depth for $t>0$ is
#
# $$h(x,t)=\frac{1}{9g}\left(2c_0-\frac{x-x_d}{t}\right)^2
#   \quad\text{for}\quad -c_0 t< x-x_d< 2c_0 t,$$
#
# with $h=h_0$ behind the rarefaction tail (which moves at $-c_0$) and $h=0$ ahead
# of the front (which moves at $2c_0$). At the dam site the depth is constant,
# $h(x_d,t)=4h_0/9$, and the velocity is $u(x_d,t)=2c_0/3$.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.physics.hydrodynamics.free_surface.numerics.equations.shallow_water.shallow_water_2d import (
    ShallowWater2DSolver,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The solver works on the periodic unit square. The reservoir occupies the middle
# half of the domain, $0.25<x<0.75$, and the rest of the bed is dry, so two
# mirror-image dam breaks develop (at $x=0.25$ towards the left and at $x=0.75$
# towards the right). The final time is chosen so that neither the fronts nor the
# two rarefaction tails meet each other or the periodic boundary.

g = 9.81              # m/s^2, gravitational acceleration
h0 = 0.20             # m, reservoir depth
x_left, x_right = 0.25, 0.75   # dam positions
n = 400               # finite-volume cells per direction
final_time = 0.07     # s

c0 = np.sqrt(g * h0)
print(f"c0 = sqrt(g h0) = {c0:.3f} m/s;  front travel 2 c0 t = {2*c0*final_time:.3f}, "
      f"tail travel c0 t = {c0*final_time:.3f}")
assert x_right + 2 * c0 * final_time < 1.0 and 2 * c0 * final_time < x_right - x_left, \
    "The dam-break waves must stay separate inside the periodic domain."


def initial_state(x, y):
    depth = np.where((x > x_left) & (x < x_right), h0, 0.0)
    return depth, np.zeros_like(x), np.zeros_like(y)


result = ShallowWater2DSolver(
    mesh_size=n, gravity=g, final_time=final_time, cfl=0.4,
).run(initial_state)
print(f"Integrated {result.step_count} explicit steps to t = {result.time:.4f} s")


# ## Comparison with Ritter's solution
#
# The flow is uniform in $y$, so we average along $y$. The right-hand dam break is
# compared with Ritter's profile point by point (L1 norm), and the dam-site depth
# and velocity are checked against $4h_0/9$ and $2c_0/3$. The wave positions are
# read off as the places where the depth crosses 10 % and 90 % of $h_0$: both
# levels lie inside the rarefaction, away from the kink at the tail and away from
# the thin wetting tip, which a first-order scheme inevitably smears. The tip itself
# is only required to lag Ritter's front by a few cells, never to run ahead of it.

axis = (np.arange(n) + 0.5) / n
dx = 1.0 / n
depth = result.depth.mean(axis=1)
velocity = np.divide(result.momentum_x.mean(axis=1), depth, out=np.zeros_like(depth), where=depth > 0)
t = result.time


def ritter_depth(x, x_dam, sign):
    # sign = +1: water on the left of the dam, front moving right; sign = -1: mirror image.
    xi = sign * (x - x_dam) / t
    h = np.where(xi <= -c0, h0, 0.0)
    rarefaction = (xi > -c0) & (xi < 2 * c0)
    h = np.where(rarefaction, (2 * c0 - xi) ** 2 / (9 * g), h)
    return h


def ritter_level(fraction):
    # Position (right-hand dam break) where Ritter's depth equals fraction * h0.
    return x_right + (2.0 - 3.0 * np.sqrt(fraction)) * c0 * t


def computed_level(fraction):
    # Outermost cell of the right-hand dam break whose depth still exceeds fraction * h0.
    inside = np.nonzero((axis >= 0.5) & (depth > fraction * h0))[0]
    i = inside.max()
    # Linear interpolation between the last cell above and the first cell below the level.
    return float(np.interp(fraction * h0, (depth[i + 1], depth[i]), (axis[i + 1], axis[i])))


h_ref = np.where(axis < 0.5, ritter_depth(axis, x_left, -1.0), ritter_depth(axis, x_right, +1.0))

right = axis >= 0.5
l1_error = float(np.sum(np.abs(depth[right] - h_ref[right]))) / float(np.sum(h_ref[right]))
h_dam = float(np.interp(x_right, axis, depth))
u_dam = float(np.interp(x_right, axis, velocity))
print(f"L1 depth error over the right-hand dam break: {100*l1_error:.2f} %")
print(f"depth at the dam site: {h_dam:.4f} m  (Ritter 4 h0 / 9 = {4*h0/9:.4f} m)")
print(f"velocity at the dam site: {u_dam:.4f} m/s  (Ritter 2 c0 / 3 = {2*c0/3:.4f} m/s)")
levels = {}
for fraction in (0.90, 0.50, 0.10, 0.01):
    levels[fraction] = (computed_level(fraction), ritter_level(fraction))
    print(f"depth level {100*fraction:4.0f} % of h0 at x = {levels[fraction][0]:.4f}  "
          f"(Ritter {levels[fraction][1]:.4f}, difference {(levels[fraction][0]-levels[fraction][1])/dx:+.1f} cells)")
tip = float(axis[np.max(np.nonzero(depth > 1e-6 * h0))])
print(f"wetting tip (depth > 1e-6 h0) at x = {tip:.4f}  (Ritter front x_d + 2 c0 t = {x_right + 2*c0*t:.4f})")

assert l1_error < 0.05, "Depth profile deviates from Ritter's solution by more than 5 % (L1)."
assert abs(h_dam - 4 * h0 / 9) / h0 < 0.03, "Dam-site depth is not 4 h0 / 9 within 3 % of h0."
assert abs(u_dam - 2 * c0 / 3) / c0 < 0.05, "Dam-site velocity is not 2 c0 / 3 within 5 % of c0."
for fraction in (0.90, 0.50, 0.10):
    computed, reference = levels[fraction]
    assert abs(computed - reference) < 5 * dx, \
        f"The {100*fraction:.0f} % depth level is more than 5 cells away from Ritter's position."
# The smeared tip lags the ideal front but must never overtake it.
assert tip <= x_right + 2 * c0 * t + dx, "The wetting tip must not run ahead of Ritter's front."
assert levels[0.01][0] > ritter_level(0.01) - 15 * dx, "The 1 % depth level lags Ritter's front by more than 15 cells."
# The reservoir core between the two tails must still be undisturbed, the bed far
# from the fronts must still be dry, and the two dam breaks must be mirror images.
core = (axis > 0.5 - 0.5 * c0 * t) & (axis < 0.5 + 0.5 * c0 * t)
assert np.allclose(depth[core], h0, rtol=1e-6, atol=0), "The reservoir core must be undisturbed."
assert np.all(depth[np.abs(axis - 0.5) > 0.47] < 1e-6 * h0), "The far bed must still be dry."
assert np.allclose(depth, depth[::-1], rtol=1e-9, atol=1e-9 * h0), "The two dam breaks must be mirror images."
assert np.all(depth >= 0.0), "Depth must stay non-negative."


# ## Figure

cream = "#fbf5eb"
fig, (ax_h, ax_u) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax_h, ax_u):
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
ax_h.plot(axis, h_ref, color="#214D43", lw=1.8, label="Ritter's solution")
ax_h.plot(axis, depth, color="#A85539", lw=1.4, ls="--", label="OneSimulation (Rusanov FV)")
ax_h.axvline(x_left, color="#777", lw=1, ls=":")
ax_h.axvline(x_right, color="#777", lw=1, ls=":")
ax_h.set(ylabel="Depth h / m", title=f"Dam break on a dry bed at t = {t:.3f} s")
ax_h.legend()
xi = (axis - x_right) / t
u_ref = np.where((xi > -c0) & (xi < 2 * c0), 2.0 / 3.0 * (c0 + xi), 0.0)
xi_l = -(axis - x_left) / t
u_ref = np.where((xi_l > -c0) & (xi_l < 2 * c0), -2.0 / 3.0 * (c0 + xi_l), u_ref)
ax_u.plot(axis, u_ref, color="#214D43", lw=1.8, label="Ritter's solution")
ax_u.plot(axis, velocity, color="#A85539", lw=1.4, ls="--", label="OneSimulation (Rusanov FV)")
ax_u.set(xlabel="Position x", ylabel="Velocity u / (m/s)")
ax_u.legend()
plt.show()


# ## Going further
#
# Refine the mesh with a larger `n` and watch the smeared wetting tip sharpen and
# the L1 error drop roughly in proportion to the cell size. Put a shallow layer of
# water on the downstream bed instead of leaving it dry: the front then turns into a
# bore whose speed and intermediate depth follow Stoker's wet-bed solution. Add a
# bottom slope through `bottom_slope` to study the dam break on an incline.
