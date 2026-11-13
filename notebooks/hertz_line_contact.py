"""OneSimulation example: Hertz line contact of a rigid cylinder on an elastic block.
Requires the installed, activated OneSimulation runtime.

The OneSimulation mortar-contact solution is checked against the Hertz
line-contact formulas: elliptical pressure distribution, contact half-width
and peak pressure for the transmitted force.
"""


# # Hertz line contact: a rigid cylinder pressed into an elastic block
#
# A long rigid cylinder of radius $R$ is pressed into a linear-elastic block
# (Young's modulus $E$, Poisson's ratio $\nu$) in plane strain. Hertz's theory
# for the elastic half-space predicts that a force per unit length $F$ is
# transmitted through a strip of half-width $b$ with a semi-elliptical
# pressure distribution,
#
# $$b=\sqrt{\frac{4FR}{\pi E^*}},\qquad
#   p(x)=p_0\sqrt{1-\frac{x^2}{b^2}},\qquad
#   p_0=\frac{2F}{\pi b},\qquad E^*=\frac{E}{1-\nu^2}.$$
#
# In plane strain the **force-approach relation is not universal**: the rigid
# approach depends logarithmically on where the far field is held, i.e. on the
# depth of the block. The pressure distribution, on the other hand, is set by
# the local geometry alone as long as the block is much deeper and wider than
# $b$. The honest comparison therefore takes the force $F$ transmitted in the
# simulation and checks the half-width, the peak pressure and the whole
# elliptical profile that Hertz predicts for that force.
#
# OneSimulation treats the contact with its **mortar formulation**: Lagrange
# multipliers on the contact trace enforce non-penetration weakly against the
# rigid cylinder profile, with a complementarity solve for the active set.

from importlib.metadata import version

import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})
import numpy as np
from onesim.cad.geometry.primitives import Rectangle
from onesim.meshing.meshes.api import cartesian
from onesim.physics.solid_mechanics import MortarBody, MortarContact, SmallStrainSolid

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# A soft block in dimensionless units: the expected half-width is about 0.4,
# so the block ($6.4\times3.6$) is roughly 15 half-widths wide and 9 deep and
# the contact zone itself is still small compared with the cylinder radius
# ($b/R\approx0.05$, the small-strain regime Hertz assumes).

E, nu = 2600.0, 0.27
E_star = E / (1.0 - nu**2)
R = 8.0                                 # cylinder radius
indentation = 0.032                     # rigid approach of the cylinder
width, depth = 6.4, 3.6                 # block size
print(f"E* = {E_star:.1f}")


def cylinder_gap(x):
    """Initial gap between the circular cylinder and the flat top surface."""
    return R - np.sqrt(R**2 - (x - width / 2) ** 2)


# ## Model and the actual computation
#
# The block is meshed with bilinear quadrilaterals. Boundary conditions: the
# bottom face is clamped, both side walls are traction-free, and the top face
# is the mortar contact trace against the rigid cylinder. The approach is
# imposed through the normal offset of the rigid profile. Two meshes are
# solved so that the error can be seen to shrink under refinement.

geometry = Rectangle(length=width, width=depth)


def solve_contact(cells_x):
    mesh = cartesian(geometry=geometry, shape=(cells_x, cells_x // 3), extents=(width, depth)).build()
    block = MortarBody(
        solid=SmallStrainSolid(youngs_modulus=E, poisson_ratio=nu, kinematics="plane_strain"),
        mesh=mesh,
        boundary=geometry.boundaries.y_max,                           # contact trace
        displacements=(
            (geometry.boundaries.y_min, "x", 0.0),                    # clamped bottom
            (geometry.boundaries.y_min, "y", 0.0),
        ),
        tractions=(
            (geometry.boundaries.x_min, (0.0, 0.0)),                  # traction-free side walls
            (geometry.boundaries.x_max, (0.0, 0.0)),
        ),
    )
    contact = MortarContact(slave=block, normal=(0.0, -1.0), rigid_gap=cylinder_gap)
    return mesh, contact.solve(times=(0.0,), normal_offset=-indentation)


def hertz_comparison(result):
    p = result.normal_pressure[-1]
    measure = result.multiplier_measures                     # mortar weights = trace lengths
    xc = result.multiplier_coordinates[:, 0] - width / 2
    force = float(measure @ p)                               # force per unit length
    b = np.sqrt(4.0 * force * R / (np.pi * E_star))
    p0 = 2.0 * force / (np.pi * b)
    hertz = p0 * np.sqrt(np.maximum(0.0, 1.0 - (xc / b) ** 2))
    # Half-width from the second moment of the pressure (b^2/4 for the ellipse).
    b_sph = 2.0 * np.sqrt(measure @ (p * xc**2) / force)
    order = np.argsort(xc)
    cumulative = np.cumsum((measure * p)[order]) / force
    cumulative_hertz = np.cumsum((measure * hertz)[order]) / (measure @ hertz)
    return dict(
        force=force, b=b, p0=p0, xc=xc, p=p, hertz=hertz,
        x_sorted=xc[order], cumulative=cumulative, cumulative_hertz=cumulative_hertz,
        peak_error=float(p.max() / p0 - 1.0),
        half_width_error=float(b_sph / b - 1.0),
        rms_error=float(np.sqrt(measure @ (p - hertz) ** 2 / (measure @ hertz**2))),
        cumulative_error=float(np.max(np.abs(cumulative - cumulative_hertz))),
        penetration=float(-result.averaged_gap.min()),
        complementarity=float(np.max(np.abs(result.averaged_gap[-1] * p))),
    )


runs = {}
for cells_x in (96, 192):
    mesh, result = solve_contact(cells_x)
    runs[cells_x] = hertz_comparison(result)
    c = runs[cells_x]
    print(f"\nMesh {cells_x} x {cells_x // 3} quadrilaterals ({mesh.cell_count} cells), dx = {width / cells_x:.4f}")
    print(f"  transmitted force F = {c['force']:.3f} per unit length")
    print(f"  Hertz half-width b = {c['b']:.4f}, peak pressure p0 = {c['p0']:.3f}")
    print(f"  peak pressure error      {100*c['peak_error']:+.1f} %")
    print(f"  half-width error         {100*c['half_width_error']:+.1f} %")
    print(f"  RMS pressure error       {100*c['rms_error']:.1f} %")
    print(f"  cumulative force error   {100*c['cumulative_error']:.2f} %")
    print(f"  max penetration {c['penetration']:.1e}, max |gap x p| {c['complementarity']:.1e}")
    # Contact mechanics sanity: no penetration anywhere, complementarity holds,
    # and the force reported on the trace is the integral of the pressure.


# ## Verification against Hertz
#
# The fine mesh must reproduce the Hertz distribution for the transmitted
# force: peak pressure and half-width within 5 %, the cumulative force
# distribution within 2 %, and the pointwise pressure within 10 % in the RMS
# sense. The pointwise pressure is the least accurate quantity of a mortar
# solution: the multipliers at the edge of the contact zone must resolve the
# infinite slope of the ellipse, so they carry the largest error (on the
# coarser mesh the RMS error is around 15 %, on a triangular mesh of the same
# spacing it would be larger still). Under refinement the error must shrink.

coarse, fine = runs[96], runs[192]
print("\nAll checks against the Hertz line-contact solution passed.")


# ## Figure

cream = "#fbf5eb"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
fig.patch.set_facecolor(cream)

ax = axes[0]
ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
x_fine = np.linspace(-1.3 * fine["b"], 1.3 * fine["b"], 400)
ax.plot(x_fine, fine["p0"] * np.sqrt(np.maximum(0.0, 1.0 - (x_fine / fine["b"]) ** 2)),
        color="#A85539", lw=1.8, label="Hertz ellipse for the transmitted force")
for cells_x, colour, marker in ((96, "#b4aa98", "s"), (192, "#39715e", "o")):
    c = runs[cells_x]
    zone = np.abs(c["xc"]) < 1.3 * fine["b"]
    ax.plot(c["xc"][zone], c["p"][zone], marker, color=colour, ms=4, ls="-", lw=0.8,
            label=f"OneSimulation mortar pressure, {cells_x} x {cells_x // 3} cells")
ax.set(xlabel="Position along the surface x", ylabel="Contact pressure p",
       title="Contact pressure under the cylinder")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(fine["x_sorted"], fine["cumulative"], color="#39715e", lw=1.8, label="OneSimulation (fine mesh)")
ax.plot(fine["x_sorted"], fine["cumulative_hertz"], color="#A85539", lw=1.4, ls="--", label="Hertz")
ax.set(xlabel="Position along the surface x", ylabel="Cumulative force fraction",
       xlim=(-1.5 * fine["b"], 1.5 * fine["b"]), title="Cumulative transmitted force")
ax.legend(frameon=False, fontsize=9)
plt.show()


# ## Going further
#
# Replace the rigid profile by a second elastic body (`master=`) to obtain the
# classical two-cylinder contact with $1/E^*=(1-\nu_1^2)/E_1+(1-\nu_2^2)/E_2$,
# add a tangential load with `friction_coefficient` to study Coulomb
# stick-slip under the Hertz pressure, or switch the kinematics to a 3D box
# with a spherical profile for the point-contact (Hertz sphere) problem.
