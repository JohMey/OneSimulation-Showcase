"""OneSimulation example: thick elliptic plate under pressure (NAFEMS benchmark LE10).
Requires the installed, activated OneSimulation runtime.

The 3D OneSimulation solution is compared with the published NAFEMS target
value of the direct stress sigma_yy at point D.
"""


# # Thick elliptic plate under pressure (NAFEMS LE10)
#
# The NAFEMS linear-elastic benchmark LE10 is a thick plate shaped like an
# elliptic ring: the inner boundary is the ellipse
# $(x/2)^2+(y/1)^2=1$, the outer boundary the ellipse
# $(x/3.25)^2+(y/2.75)^2=1$ (lengths in metres), and the plate is
# $0.6$ m thick ($-0.3\le z\le 0.3$). A uniform pressure of $1$ MPa acts on the
# upper face. Because of symmetry only the quarter $x\ge0,\ y\ge0$ is modelled.
#
# Supports of the benchmark:
#
# * symmetry face $y=0$ (DCD'C'): $u_y=0$,
# * symmetry face $x=0$ (ABA'B'): $u_x=0$,
# * outer curved face (BCB'C'): $u_x=u_y=0$,
# * mid-plane line of the outer face ($z=0$): $u_z=0$.
#
# The plate is therefore simply supported along the mid-line of its outer edge
# and bends under the load. Material: $E=210$ GPa, $\nu=0.3$.
#
# **Reference.** NAFEMS, *The Standard NAFEMS Benchmarks*, Rev. 3, test LE10
# (NAFEMS TNSB, 1990): the direct stress at point D $=(2,0,0.3)$, on the
# loaded face at the end of the hole's major axis, is
#
# $$\sigma_{yy}(D)=-5.38\ \text{MPa}.$$
#
# This is a published benchmark value for exactly this setup, so it is the
# reference used below, together with a three-mesh refinement study and the
# global force balance.
#
# **Method.** OneSimulation solves the full 3D small-strain equilibrium
# $-\nabla\!\cdot\boldsymbol\sigma=0$ with an isotropic Hooke law by the
# continuous Galerkin finite-element method on isoparametric 27-node
# hexahedra: quadratic geometry and tensor-product quadratic ($Q_2$)
# Lagrange displacement, $5\times5\times5$ Gauss points per cell. The
# `SmallStrainSolid` provider accepts component-wise displacement supports
# (rollers) on faces. The problem is stationary (no time integrator); the
# linear system is solved by a Newton step with a sparse direct (LU) solver.
# For comparison the finest mesh is also solved with trilinear ($Q_1$)
# displacement on the same curved cells.
#
# **One modelling note.** OneSimulation prescribes displacements on faces, not
# on single lines. The mid-plane line support is therefore applied on a narrow
# band of the outer face, $|z|\le\delta$ with $\delta=0.015$ m ($2.5\,\%$ of
# the thickness). The band carries the same vertical reaction at the same
# height; by Saint-Venant's principle the way this reaction is spread over a
# band that is narrow compared with the thickness does not change the stress
# at D, which lies $1.25$ m (two plate thicknesses) away. A run with twice the
# band width below quantifies this.

from importlib.metadata import version
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from onesim.cad.d3 import QuadraticHexahedron
from onesim.meshing import QuadraticHexahedralMesher
from onesim.meshing.generation.backends.structured.volume.quadratic_hex import QuadraticHexahedralMesh
from onesim.meshing.generation.contracts.polynomial.hex import HEX27_REFERENCE, QuadraticHexahedralGeometry
from onesim.physics.solid_mechanics import SmallStrainSolid

print("OneSimulation:", version("onesimulation"))


# ## Parameters

E, nu = 210e9, 0.30                      # Pa, -
pressure = 1.0e6                         # Pa, on the upper face
a_in, b_in = 2.0, 1.0                    # m, semi-axes of the inner ellipse
a_out, b_out = 3.25, 2.75                # m, semi-axes of the outer ellipse
thickness = 0.6                          # m
sigma_yy_target = -5.38e6                # Pa, NAFEMS LE10 target at point D
point_D = np.array([a_in, 0.0, thickness / 2])

lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
top_area = np.pi * (a_out * b_out - a_in * b_in) / 4      # quarter elliptic ring
print(f"Total load on the quarter model: {pressure * top_area / 1e6:.4f} MN")


# ## Geometry and mesh
#
# The quarter plate is the image of the unit cube under the map
#
# $$(\xi,\eta,\zeta)\mapsto\bigl(a(\xi)\cos\theta,\ b(\xi)\sin\theta,\ t(\zeta-\tfrac12)\bigr),
#   \quad \theta=\tfrac{\pi}{2}\eta,\quad a=(1-\xi)a_i+\xi a_o,\quad b=(1-\xi)b_i+\xi b_o,$$
#
# which joins points of equal elliptic angle on the two ellipses. The
# geometry object is a single 27-node hexahedron that carries the six named
# faces of this block. The reference faces become: $\xi=0$ the hole
# surface, $\xi=1$ the outer surface, $\eta=0$ the symmetry face $y=0$,
# $\eta=1$ the symmetry face $x=0$, $\zeta=0/1$ the lower/upper face.
#
# `QuadraticHexahedralMesher` partitions this block into a tensor grid of
# 27-node hexahedra; a `mesh_factory` places every geometry node exactly on
# the map above, so curved faces are interpolated quadratically per cell. The
# grid is graded toward the hole (radially), toward the major axis $\theta=0$
# and toward the loaded face, i.e. toward point D, and it resolves the support
# band with its own cell layers.

def plate_map(reference):
    xi, eta, zeta = np.asarray(reference, dtype=float).T
    theta = 0.5 * np.pi * eta
    a = (1 - xi) * a_in + xi * a_out
    b = (1 - xi) * b_in + xi * b_out
    return np.column_stack((a * np.cos(theta), b * np.sin(theta), thickness * (zeta - 0.5)))


def make_mesh_factory(band_halfwidth):
    class EllipticPlateMesh(QuadraticHexahedralMesh):
        """Q2 partition of the reference block with nodes on the exact plate map."""

        def __init__(self, geometry, reference_axes):
            super().__init__(geometry, reference_axes)
            axes = [np.sort(np.r_[a, (np.asarray(a[:-1]) + np.asarray(a[1:])) / 2]) for a in reference_axes]
            data = QuadraticHexahedralGeometry(plate_map(np.asarray(tuple(product(*axes)))), self.cells)
            self.coordinates, self.cell_volumes = data.coordinates, data.volumes
            self._carrier = self._build_carrier()
            z = self.coordinates[:, 2]
            # Outer-face facets inside the mid-plane support band |z| <= delta.
            self._facets["support_band"] = [
                face for face in self._facets["x_max"]
                if np.all(np.abs(z[list(face)]) <= band_halfwidth * (1 + 1e-9))
            ]

        def facets(self, selection):
            if isinstance(selection, str):
                return tuple(sorted(self._facets[selection]))
            return super().facets(selection)

    return EllipticPlateMesh


geometry = QuadraticHexahedron(nodes=tuple(map(tuple, plate_map(HEX27_REFERENCE))))
faces = geometry.boundaries


def geometric(n, ratio):
    """n cells on [0, 1]; the last cell is `ratio` times the first."""
    growth = ratio ** (1.0 / max(n - 1, 1))
    sizes = growth ** np.arange(n)
    nodes = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    nodes[-1] = 1.0
    return nodes


def two_sided(n, ratio):
    """n cells on [0, 1], refined toward both ends (centre cells `ratio` times larger)."""
    sizes = 1.0 + (ratio - 1.0) * np.sin(np.pi * (np.arange(n) + 0.5) / n)
    nodes = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    nodes[-1] = 1.0
    return nodes


def reference_axes(n, band_halfwidth):
    radial = geometric(n, 4.0)                       # fine at the hole
    angular = geometric(n, 3.0)                      # fine at theta = 0
    zb = band_halfwidth / thickness                  # band half-width in reference units
    nb = max(1, n // 6)                              # cells across each half of the band
    lower = (0.5 - zb) * (1 - geometric(n // 2, 4.0)[::-1])          # fine toward the band
    band = np.linspace(0.5 - zb, 0.5 + zb, 2 * nb + 1)
    upper = 0.5 + zb + (0.5 - zb) * two_sided(n // 2 + 1, 3.0)     # fine at band and loaded face
    vertical = np.unique(np.r_[lower, band[1:-1], upper])
    vertical[0], vertical[-1] = 0.0, 1.0
    return tuple(radial), tuple(angular), tuple(vertical)


# ## Model
#
# Boundary conditions on every face of the quarter model:
#
# | face | condition |
# |---|---|
# | upper face $z=+0.3$ | traction $(0,0,-p)$ |
# | lower face $z=-0.3$ | traction-free |
# | hole surface | traction-free |
# | symmetry face $y=0$ | $u_y=0$ (roller) |
# | symmetry face $x=0$ | $u_x=0$ (roller) |
# | outer face | $u_x=u_y=0$, and $u_z=0$ on the band $|z|\le\delta$ |

solid = SmallStrainSolid(youngs_modulus=E, poisson_ratio=nu, kinematics="volume")


def solve_plate(n, band_halfwidth=0.015, order=2):
    mesher = QuadraticHexahedralMesher(
        reference_axes=reference_axes(n, band_halfwidth),
        mesh_factory=make_mesh_factory(band_halfwidth),
    )
    mesh = mesher.generate(geometry)
    result = solid.solve(
        mesh=mesh, times=(0.0,), order=order,
        displacements=(
            (faces.x_max, "x", 0.0), (faces.x_max, "y", 0.0),     # outer face
            ("support_band", "z", 0.0),                          # mid-plane support band
            (faces.y_min, "y", 0.0),                             # symmetry y = 0
            (faces.y_max, "x", 0.0),                             # symmetry x = 0
        ),
        tractions=(
            (faces.z_max, (0.0, 0.0, -pressure)),                # loaded face
            (faces.z_min, (0.0, 0.0, 0.0)),                      # free lower face
            (faces.x_min, (0.0, 0.0, 0.0)),                      # free hole surface
        ),
    )
    return mesh, result


def stress_from_gradient(gradient):
    strain = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
    trace = np.trace(strain, axis1=-2, axis2=-1)[..., None, None]
    return lam * trace * np.eye(3) + 2 * mu * strain


# ## Solve
#
# Three meshes of the same graded family with 8, 12 and 16 cells along the
# radial and angular directions. The stress at D is obtained from the
# gradient of the quadratic displacement field at D.

runs = {}
meshes = (8, 12, 16)
for n in meshes:
    mesh, result = solve_plate(n)
    displacement = result.displacement[-1]
    sigma = stress_from_gradient(result.space.gradient(displacement, point_D[None, :]))[0]
    reaction_z = float(result.reactions[-1][:, 2].sum())
    runs[n] = dict(mesh=mesh, result=result, sigma_yy=sigma[1, 1], reaction=reaction_z,
                   dofs=displacement.size, cells=mesh.cell_count)
    print(f"mesh n={n:2d}: {mesh.cell_count:5d} hexahedra, {displacement.size:6d} unknowns, "
          f"sigma_yy(D) = {sigma[1, 1] / 1e6:.4f} MPa, vertical reaction = {reaction_z / 1e6:.5f} MN")

# Band-width check on the middle mesh: twice the support band.
_, wide = solve_plate(12, band_halfwidth=0.030)
sigma_wide = stress_from_gradient(wide.space.gradient(wide.displacement[-1], point_D[None, :]))[0][1, 1]

# Element-order comparison: trilinear (Q1) displacement on the finest mesh.
_, linear = solve_plate(meshes[-1], order=1)
sigma_q1 = stress_from_gradient(linear.space.gradient(linear.displacement[-1], point_D[None, :]))[0][1, 1]
q1_dofs = linear.displacement[-1].size


# ## Comparison with the reference
#
# The benchmark value is quoted to three significant digits, i.e. to about
# $\pm0.1\,\%$. The printout lists the computed stress at D on the three
# meshes with its deviation from the target, the change between the two
# finest meshes, the result with trilinear instead of quadratic elements, the
# effect of doubling the support band and the balance between the applied
# load and the vertical support reaction. Trilinear hexahedra are much stiffer
# in bending, so on the same mesh they are visibly further from converged.

values = np.array([runs[n]["sigma_yy"] for n in meshes])
fine = runs[meshes[-1]]
print("Direct stress sigma_yy at point D (MPa)")
print(f"  NAFEMS LE10 target            {sigma_yy_target / 1e6:9.4f}")
for n in meshes:
    v = runs[n]["sigma_yy"]
    print(f"  OneSimulation, n = {n:2d}         {v / 1e6:9.4f}   "
          f"deviation from target {100 * (v / sigma_yy_target - 1):+6.2f} %")
print(f"  change between the two finest meshes {100 * abs(values[2] / values[1] - 1):.3f} %")
print(f"Q1 displacement on the finest mesh ({q1_dofs} unknowns): sigma_yy(D) = {sigma_q1 / 1e6:.4f} MPa, "
      f"deviation from target {100 * (sigma_q1 / sigma_yy_target - 1):+.2f} %")
print(f"Support band twice as wide (n = 12): sigma_yy(D) = {sigma_wide / 1e6:.4f} MPa, "
      f"change {100 * abs(sigma_wide / runs[12]['sigma_yy'] - 1):.3f} %")
print(f"Force balance on the finest mesh: vertical reaction {fine['reaction'] / 1e6:.6f} MN, "
      f"applied load {pressure * top_area / 1e6:.6f} MN, "
      f"relative difference {abs(fine['reaction'] / (pressure * top_area) - 1):.1e}")


# ## Figure
#
# Left: $\sigma_{yy}$ on the loaded face. Middle: $\sigma_{yy}$ on the
# symmetry plane $y=0$ between the hole (left) and the supported outer edge
# (right), showing the bending through the thickness. Right: mesh study of
# $\sigma_{yy}(D)$ against the NAFEMS target.

result = fine["result"]
space, displacement = result.space, result.displacement[-1]
nr, nt, nz = (len(axis) - 1 for axis in reference_axes(meshes[-1], 0.015))
s = np.linspace(0.0, 1.0, 6)
m = len(s)
sub = np.array([(i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1)
                for i in range(m - 1) for j in range(m - 1)])
local_triangles = np.r_[sub[:, [0, 1, 2]], sub[:, [0, 2, 3]]]


def sample_face(cells, fixed_axis, fixed_value):
    """sigma_yy on one reference face of the given cells, as a triangulated sample grid per cell."""
    free = [axis for axis in range(3) if axis != fixed_axis]
    local = np.zeros((m * m, 3))
    local[:, free] = np.array(tuple(product(s, s)))
    local[:, fixed_axis] = fixed_value
    points, stresses, triangles = [], [], []
    for index, cell in enumerate(cells):
        points.append(space.geometry.map(cell, local))
        gradient = np.einsum("qnd,nc->qcd", space.gradients(cell, local), displacement[space.cell_dofs[cell]])
        stresses.append(stress_from_gradient(gradient)[:, 1, 1])
        triangles.append(local_triangles + index * m * m)
    return np.concatenate(points), np.concatenate(stresses), np.concatenate(triangles)


top_cells = [(i * nt + j) * nz + nz - 1 for i in range(nr) for j in range(nt)]
cut_cells = [(i * nt + 0) * nz + k for i in range(nr) for k in range(nz)]
top_points, top_syy, top_tri = sample_face(top_cells, 2, 1.0)
cut_points, cut_syy, cut_tri = sample_face(cut_cells, 1, 0.0)

cream = "#fbf5eb"
fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.25, 1.0))

ax = fig.add_subplot(grid[0])
ax.set_facecolor("#fffaf2")
levels = np.linspace(-6, 16, 45)
im = ax.tricontourf(top_points[:, 0], top_points[:, 1], top_tri, top_syy / 1e6, levels=levels,
                    cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-6, vcenter=0, vmax=16), extend="both")
fig.colorbar(im, ax=ax, label="sigma_yy / MPa", ticks=np.arange(-6, 17, 2))
ax.plot(*point_D[:2], "k*", ms=11)
ax.annotate("D", point_D[:2], xytext=(6, 6), textcoords="offset points")
th = np.linspace(0, np.pi / 2, 100)
for a, b in ((a_in, b_in), (a_out, b_out)):
    ax.plot(a * np.cos(th), b * np.sin(th), color="#3b3b3b", lw=0.8)
ax.set(xlabel="x / m", ylabel="y / m", title="sigma_yy on the loaded face z = +0.3 m", aspect="equal")

ax = fig.add_subplot(grid[1])
ax.set_facecolor("#fffaf2")
levels = np.linspace(-8, 8, 33)
im = ax.tricontourf(cut_points[:, 0], cut_points[:, 2], cut_tri, cut_syy / 1e6,
                    levels=levels, cmap="RdBu_r", extend="both")
fig.colorbar(im, ax=ax, label="sigma_yy / MPa", ticks=np.arange(-8, 9, 2))
ax.plot(point_D[0], point_D[2], "k*", ms=11, clip_on=False)
ax.plot([a_out, a_out], [-0.015, 0.015], color="k", lw=5, solid_capstyle="butt", clip_on=False)
ax.annotate("support band", (a_out, 0.0), xytext=(-75, 25), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set(xlabel="x / m", ylabel="z / m", title="sigma_yy on the symmetry plane y = 0", aspect="equal")

ax = fig.add_subplot(grid[2])
ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
dofs = np.array([runs[n]["dofs"] for n in meshes])
ax.semilogx(dofs, values / 1e6, "o-", color="#39715e", label="OneSimulation, Q2 hexahedra")
ax.semilogx([q1_dofs], [sigma_q1 / 1e6], "s", color="#c77b4a", label="OneSimulation, Q1 on the finest mesh")
ax.axhline(sigma_yy_target / 1e6, color="#A85539", lw=1.6, label="NAFEMS LE10 target -5.38 MPa")
ax.set(xlabel="Number of displacement unknowns", ylabel="sigma_yy(D) / MPa",
       title="Mesh study at point D")
ax.legend(frameon=False, fontsize=9, loc="center right")
plt.show()


# ## Going further
#
# Compute the reaction distribution along the support band to see how the
# shear force flows into the support, reduce the thickness to follow the
# transition toward thin-plate bending, or clamp the outer
# face completely to see how strongly the stress at D depends on the support.
# The same mapped-block mesh factory works for any ring-shaped plate between
# two curves that can be parametrised by a common angle.
