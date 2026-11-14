"""OneSimulation example: Hertz point contact of an elastic sphere on an elastic block in 3D.
Requires the installed, activated OneSimulation runtime.

A steel sphere is pressed onto a steel block. Both bodies are elastic 3D solids on
non-matching meshes (trilinear hexahedra in the block, linear tetrahedra in the sphere),
coupled by mortar contact. The contact radius, the pressure distribution, the
force-approach law and the subsurface stresses on the axis are graded against Hertz's
point-contact solution, with a three-level mesh study near the contact.
"""


# # Hertz contact of an elastic sphere on an elastic block
#
# A sphere of radius $R$ is pressed onto a flat block. Both are made of the same
# linear-elastic steel (Young's modulus $E$, Poisson's ratio $\nu$). Hertz's theory for
# two elastic half-spaces with a paraboloidal gap $r^2/2R$ (H. Hertz, *J. reine angew.
# Math.* 92, 156-171, 1882; K. L. Johnson, *Contact Mechanics*, Cambridge University
# Press 1985, Sec. 4.2 and 3.4) predicts for a transmitted force $F$
#
# $$\frac{1}{E^*}=\frac{1-\nu_1^2}{E_1}+\frac{1-\nu_2^2}{E_2},\qquad
#   a=\left(\frac{3FR}{4E^*}\right)^{1/3},\qquad
#   p_0=\frac{3F}{2\pi a^2},\qquad
#   \delta=\frac{a^2}{R}=\left(\frac{9F^2}{16RE^{*2}}\right)^{1/3},$$
#
# a circular contact patch of radius $a$ with the elliptical pressure distribution
# $p(r)=p_0\sqrt{1-r^2/a^2}$, and an approach $\delta$ of distant points. Below the centre
# of the contact the stresses are known in closed form (Johnson, eq. 3.45):
#
# $$\frac{\sigma_z}{p_0}=-\frac{1}{1+z^2/a^2},\qquad
#   \frac{\sigma_r}{p_0}=\frac{\sigma_\theta}{p_0}=-(1+\nu)\Bigl(1-\frac{z}{a}\arctan\frac{a}{z}\Bigr)
#   +\frac{1}{2}\,\frac{1}{1+z^2/a^2}.$$
#
# The largest shear stress $\tau_1=|\sigma_r-\sigma_z|/2$ does not occur on the surface
# but **below** it: for $\nu=0.3$ it reaches $0.31\,p_0$ at a depth $z\approx0.48\,a$, which
# is where rolling-contact fatigue starts.
#
# **The model is genuinely three-dimensional.** The sphere is the lower half of a full
# sphere, held by a frictionless rigid platen on its equatorial plane; the block is a
# finite cuboid on a frictionless support. One quarter of the assembly is computed, with
# symmetry rollers on the planes $x=0$ and $y=0$. The stress field varies in all three
# coordinates; the contact patch, the pressure and the subsurface stresses come out of
# the 3D solution, nothing is prescribed.
#
# **Reference kind.** Hertz's formulas are the exact elasticity solution for two
# half-spaces. Local quantities (contact radius, pressure distribution, peak pressure,
# subsurface stress on the axis) depend only on the transmitted force and the local
# geometry as long as the bodies are much larger than $a$ (here the block is $10a$ and the
# sphere $20a$), so they are compared directly with Hertz for the force that the
# simulation transmits. The approach of the platen to the block support contains, in
# addition to the Hertz term $\propto F^{2/3}$, a far-field term that is linear in $F$,
# depends on the finite bodies and can have either sign; it is separated by a two-parameter fit
# $\delta=c_1F^{2/3}+c_2F$ over five load levels, and $c_1$ is compared with Hertz. A
# three-level mesh study near the contact gives the observed order and a Richardson
# estimate.
#
# **Methods used.**
# - Spatial: continuous Galerkin finite elements, trilinear hexahedra (Q1) on a graded
#   tensor-product mesh of the block, linear tetrahedra (P1) on a graded unstructured
#   CAD mesh of the sphere; the two meshes do not match at the contact surface.
# - Contact: segment-to-segment **mortar** coupling (exact polygon clipping of the
#   projected block and sphere facets) with Lagrange multipliers for the normal contact
#   pressure; frictionless.
# - Nonlinear solver: primal-dual active-set method (semi-smooth Newton) for the
#   Signorini complementarity conditions on the condensed contact compliance.
# - Linear solver: sparse direct LU (SuperLU) of the block-diagonal stiffness, reused for
#   all load levels.
# - Loading: quasi-static, five imposed approaches of the platen.

from importlib.metadata import version
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import time

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
import gc
import numpy as np
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from onesim.cad.geometry.backends.occt.provider import OcctCadProvider
from onesim.cad.geometry.ports.kernel_port import SphereRequest
from onesim.cad.geometry.primitives import Geometry
from onesim.cad.operations.execution.model_geometry import GeometryService
from onesim.meshing import UnstructuredSimplexMesh
from onesim.meshing.generation.backends.cad.in_memory_cad import build_file_free_mesher
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import CORNERS, FACES, affine_hexahedral_geometry
from onesim.meshing.generation.contracts.requests import (
    HighOrderMeshControl, MeshGenerationRequest, MeshRefinement, MeshRefinementSupport, SpatialMeshControl,
)
from onesim.meshing.generation.operations.geometry.model_mesh import CadMeshingService
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.physics.solid_mechanics import MortarBody, MortarContact, SmallStrainSolid

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# A bearing-steel ball of 20 mm diameter on a steel block. The largest imposed approach
# gives a contact radius of about 0.5 mm ($a/R\approx0.05$) and a peak pressure of about
# 3.7 GPa, typical of a heavily loaded rolling bearing. The model is linear elastic
# (no yielding). The block is a $10a\times10a\times10a$ quarter, the sphere has
# $R=20a$.

E, nu = 210e9, 0.30                     # steel, both bodies
R = 10e-3                               # sphere radius [m]
a_ref = 0.5e-3                          # contact radius at the largest load (design value)
E_star = 1.0 / (2.0 * (1.0 - nu**2) / E)
block_size = 10 * a_ref                 # quarter block: 5 mm x 5 mm x 5 mm
patch = 1.2 * a_ref                     # block surface region that may come into contact
grading = 1.45                          # block cell growth ratio away from the contact
SPHERE_GRADING = ((0.3 * a_ref, 2.0 * a_ref), (0.8 * a_ref, 5 * a_ref))  # (size, ball radius)
delta_max = a_ref**2 / R                # largest imposed approach of the platen
deltas = delta_max * np.linspace(0.4, 1.0, 5)
mesh_levels = ((7.1, 7.7), (9.2, 10.0), (12.0, 13.0))   # contact cell sizes h = a_ref / n (block, sphere)
G = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
print(f"E* = {E_star / 1e9:.1f} GPa, largest approach {delta_max * 1e6:.1f} um")


# ## Geometry and mesh
#
# **Block.** A graded tensor-product mesh of trilinear hexahedra: uniform cells of size
# $h$ over the contact region ($0\le x,y\le1.2a$, depth $0\ldots a$), then growing by a
# factor 1.45 per cell to the outer faces. Its top face is split into the potential
# contact patch and the free remainder.
#
# **Sphere.** OneSimulation's CAD kernel cuts one octant out of a sphere; the free
# tetrahedral mesher grades it from cells of size $h_s$ around the pole (ball of radius
# $1.15a$) via $0.3a$ and $0.8a$ to the global size $0.25R$. The octant is mirrored so that
# its pole touches the block at the origin and its flat equatorial face lies at $z=R$.
# The curved surface is split into the cap above the contact patch ($r<2a$) and the rest.
#
# Boundary conditions on every face:
#
# | body | face | condition |
# |---|---|---|
# | block | $x=0$, $y=0$ | symmetry roller ($u_x=0$, $u_y=0$) |
# | block | bottom $z=-10a$ | frictionless support ($u_z=0$) |
# | block | $x=10a$, $y=10a$ | traction-free |
# | block | top, outside the patch | traction-free |
# | block | top patch $x,y\le1.2a$ | mortar contact (slave side) |
# | sphere | $x=0$, $y=0$ | symmetry roller |
# | sphere | equatorial plane $z=R$ | frictionless rigid platen ($u_z=0$) |
# | sphere | cap $r<2a$ | mortar contact (master side) |
# | sphere | rest of the spherical surface | traction-free |
#
# The load is the approach $\delta$ of the platen towards the block support, imposed as
# a normal offset of the contact gap.

BLOCK_FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "top_patch", "top_free")
SPHERE_FACES = ("plane_x", "plane_y", "equator", "cap", "surface")


@dataclass(frozen=True, eq=False, kw_only=True)
class BlockQuarter(Geometry):
    """Quarter block with its top face split into the contact patch and the free part."""

    dimension = 3

    def boundary_groups(self):
        return {**{name: (name,) for name in BLOCK_FACES}, "all": BLOCK_FACES}


@dataclass(frozen=True, eq=False, kw_only=True)
class SphereQuarter(Geometry):
    """Quarter of the lower half sphere: two symmetry planes, equator, cap and surface."""

    dimension = 3

    def boundary_groups(self):
        return {**{name: (name,) for name in SPHERE_FACES}, "all": SPHERE_FACES}


class GradedHexMesh(StructuredHexahedralMesh):
    """Trilinear hexahedra on a tensor product of three graded node axes."""

    def __init__(self, geometry, axes):
        self.geometry, self.shape, self.axes = geometry, tuple(len(x) - 1 for x in axes), axes
        dims = tuple(len(x) for x in axes)
        X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        cells = np.array([[np.ravel_multi_index(tuple(np.add(i, c)), dims) for c in CORNERS]
                          for i in product(*(range(n) for n in self.shape))])
        data = affine_hexahedral_geometry(X, cells)
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.mesh_id = "hex:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        self._facets = {name: [] for name in BLOCK_FACES}
        sides = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        for index, cell in zip(product(*(range(n) for n in self.shape)), self.cells):
            for side, face in enumerate(FACES):
                axis = side // 2
                if index[axis] == (self.shape[axis] - 1 if side % 2 else 0):
                    name = sides[side]
                    if name == "z_max":
                        corners = self.coordinates[cell[list(face)], :2]
                        name = "top_patch" if corners.max() <= patch * (1 + 1e-9) else "top_free"
                    self._facets[name].append(tuple(cell[list(face)]))
        self._carrier = self._build_carrier()


def graded_axis(h, fine, length):
    """Uniform nodes of spacing h up to `fine`, then geometric growth up to `length`."""
    nodes = list(np.arange(0.0, fine + 0.5 * h, h))
    step = h
    while nodes[-1] < length * (1 - 1e-9):
        step *= grading
        nodes.append(min(nodes[-1] + step, length))
    if length - nodes[-2] < 0.5 * (nodes[-2] - nodes[-3]):
        nodes.pop(-2)
    return np.array(nodes)


def block_mesh(n):
    h = a_ref / n
    xy = graded_axis(h, patch, block_size)
    z = -graded_axis(h, a_ref, block_size)[::-1]
    return GradedHexMesh(BlockQuarter(), (xy, xy, z))


def sphere_mesh(ns):
    kernel = OcctCadProvider()
    cad = GeometryService("hertz_sphere", kernel)
    ball = cad.construct(kernel.create_sphere, SphereRequest(cad.reference("ball"), R))
    octant = cad.boolean("intersection", ball, cad.box(width=R, height=R, depth=R, name="box"), name="octant")
    body = next(e.entity_id for e in octant.data.entities if e.kind == "body")
    pole = (0.0, 0.0, R)
    controls = tuple(MeshRefinement(body, size, "volume", MeshRefinementSupport("ball", pole, radius))
                     for size, radius in ((a_ref / ns, 1.15 * a_ref), *SPHERE_GRADING))
    request = MeshGenerationRequest(
        octant.reference.artifact_id, octant.reference.version, global_size=0.25 * R, refinements=controls,
        max_candidate_cells=2_000_000, high_order=HighOrderMeshControl(polynomial_order=1),
        spatial=SpatialMeshControl("free_tetrahedral", body, engine="cad_native", boundary_conforming=False))
    native = CadMeshingService(provider_factory=build_file_free_mesher).generate(octant, request)
    X = np.array(native.data.coordinates, dtype=float)
    C = np.array(native.data.cells)[:, [0, 2, 1, 3]]       # keep positive volumes after the mirror
    X[:, 2] = R - X[:, 2]                                   # pole at the origin, equator at z = R
    groups = {name: [] for name in SPHERE_FACES}
    for facet in boundary_faces(C, X):
        P = X[list(facet.vertices)]
        if np.all(np.abs(P[:, 0]) < 1e-12 * R):
            name = "plane_x"
        elif np.all(np.abs(P[:, 1]) < 1e-12 * R):
            name = "plane_y"
        elif np.all(np.abs(P[:, 2] - R) < 1e-12 * R):
            name = "equator"
        else:
            name = "cap" if np.hypot(P[:, 0], P[:, 1]).max() < 2 * a_ref else "surface"
        groups[name].append(facet.vertices)
    mesh = UnstructuredSimplexMesh(geometry=SphereQuarter(), coordinates=X, cells=C, boundary_facets=groups)
    return mesh, groups


# ## Model
#
# Each body is a `MortarBody` (a small-strain linear-elastic 3D solid with its supports
# and tractions); `MortarContact` couples the block patch (slave) to the sphere cap
# (master) along the contact normal $-z$, from the sphere into the block.

steel = SmallStrainSolid(youngs_modulus=E, poisson_ratio=nu, kinematics="volume")


def contact_model(block_grid, sphere_grid):
    b, s = block_grid.geometry.boundaries, sphere_grid.geometry.boundaries
    block = MortarBody(
        solid=steel, mesh=block_grid, boundary=b.top_patch,
        displacements=((b.x_min, "x", 0.0), (b.y_min, "y", 0.0), (b.z_min, "z", 0.0)),
        tractions=((b.x_max, (0.0, 0.0, 0.0)), (b.y_max, (0.0, 0.0, 0.0)), (b.top_free, (0.0, 0.0, 0.0))),
    )
    sphere = MortarBody(
        solid=steel, mesh=sphere_grid, boundary=s.cap,
        displacements=((s.plane_x, "x", 0.0), (s.plane_y, "y", 0.0), (s.equator, "z", 0.0)),
        tractions=((s.surface, (0.0, 0.0, 0.0)),),
    )
    return MortarContact(slave=block, master=sphere, normal=(0.0, 0.0, -1.0))


# ## Solve
#
# For every mesh level the five load levels are solved with one factorization. The
# post-processing computes the stresses from the displacement fields: in the block the
# exact strain of each trilinear cell at its centre, in the sphere the constant strain of
# each tetrahedron. All Hertz values below use the force that the simulation transmits.

def hertz(force):
    a = (3 * force * R / (4 * E_star)) ** (1 / 3)
    return a, 3 * force / (2 * np.pi * a**2)


def axis_stresses(z, a, p0):
    s = z / a
    sz = -p0 / (1 + s**2)
    sr = p0 * (-(1 + nu) * (1 - s * np.arctan2(1.0, s)) + 0.5 / (1 + s**2))
    return sz, sr


s_scan = np.linspace(1e-4, 3, 300001)
tau_scan = 0.5 * np.subtract(*axis_stresses(s_scan, 1.0, 1.0)[::-1])
TAU_HERTZ, Z_HERTZ = tau_scan.max(), s_scan[np.argmax(tau_scan)]
print(f"Hertz: max shear on the axis {TAU_HERTZ:.4f} p0 at z = {Z_HERTZ:.4f} a (nu = {nu})")


def stress_tensor(strain):
    eye = np.eye(3)
    return lam * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * eye + 2 * G * strain


def block_stresses(grid, X, u):
    """Strain of every trilinear cell at its centre, from the nodal displacements."""
    xs, ys, zs = grid.axes
    index = (np.searchsorted(xs, X[:, 0] - 1e-12), np.searchsorted(ys, X[:, 1] - 1e-12),
             np.searchsorted(zs, X[:, 2] - 1e-12))
    U = np.zeros((len(xs), len(ys), len(zs), 3))
    U[index] = u
    grad = np.zeros((len(xs) - 1, len(ys) - 1, len(zs) - 1, 3, 3))
    dU = (np.diff(U, axis=0), np.diff(U, axis=1), np.diff(U, axis=2))
    widths = (np.diff(xs)[:, None, None, None], np.diff(ys)[None, :, None, None], np.diff(zs)[None, None, :, None])
    avg = {0: lambda A: 0.25 * (A[:, :-1, :-1] + A[:, 1:, :-1] + A[:, :-1, 1:] + A[:, 1:, 1:]),
           1: lambda A: 0.25 * (A[:-1, :, :-1] + A[1:, :, :-1] + A[:-1, :, 1:] + A[1:, :, 1:]),
           2: lambda A: 0.25 * (A[:-1, :-1] + A[1:, :-1] + A[:-1, 1:] + A[1:, 1:])}
    for k in range(3):
        grad[..., :, k] = avg[k](dU[k]) / widths[k]
    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    centres = np.stack(np.meshgrid(*(0.5 * (x[1:] + x[:-1]) for x in (xs, ys, zs)), indexing="ij"), -1)
    return centres, stress_tensor(strain)


def tet_stresses(X, C, u):
    """Constant strain of every linear tetrahedron."""
    P, V = X[C], u[C]
    J = np.swapaxes(P[:, 1:] - P[:, :1], 1, 2)          # columns = edge vectors
    dV = np.swapaxes(V[:, 1:] - V[:, :1], 1, 2)
    grad = dV @ np.linalg.inv(J)
    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    return stress_tensor(strain), np.abs(np.linalg.det(J)) / 6


def max_shear(stress):
    principal = np.linalg.eigvalsh(stress)
    return 0.5 * (principal[..., -1] - principal[..., 0])


def run_level(n_block, n_sphere):
    started = time.perf_counter()
    block_grid = block_mesh(n_block)
    sphere_grid, sphere_groups = sphere_mesh(n_sphere)
    contact = contact_model(block_grid, sphere_grid)
    result = contact.solve(times=tuple(float(k) for k in range(1, len(deltas) + 1)),
                           normal_offset=lambda t: -deltas[int(round(t)) - 1])
    w, xc = result.multiplier_measures, result.multiplier_coordinates
    r = np.hypot(xc[:, 0], xc[:, 1])
    forces = 4 * result.normal_pressure @ w                   # full sphere = 4 quarters
    p = result.normal_pressure[-1]
    force = forces[-1]
    a_h, p0_h = hertz(force)
    # contact radius from the second moment of the pressure (= 2a^2/5 for the ellipse)
    a_moment = np.sqrt(2.5 * (w @ (p * r**2)) / (w @ p))
    # least-squares ellipse p0 sqrt(1 - r^2/a^2) through the nodal pressures
    fit = lambda a: np.sqrt(np.clip(1 - (r / a) ** 2, 0, None))
    misfit = lambda a: (w * (p - (w * p * fit(a)).sum() / (w * fit(a) ** 2).sum() * fit(a)) ** 2).sum()
    trial = np.linspace(0.7 * a_h, 1.3 * a_h, 241)
    a_fit = trial[np.argmin([misfit(a) for a in trial])]
    p0_fit = (w * p * fit(a_fit)).sum() / (w * fit(a_fit) ** 2).sum()
    hertz_p = p0_h * fit(a_h)
    rms = np.sqrt((w * (p - hertz_p) ** 2).sum() / (w * hertz_p**2).sum())
    inner = r < 0.8 * a_h
    rms_inner = np.sqrt((w * (p - hertz_p) ** 2)[inner].sum() / (w * hertz_p**2)[inner].sum())
    # stresses on the axis: first column of block cells (cell centres at x = y = h/2)
    centres, sigma = block_stresses(block_grid, result.slave_coordinates, result.slave_displacement[-1])
    z_axis = -centres[0, 0, :, 2]
    tau_axis = max_shear(sigma[0, 0])
    order = np.argsort(z_axis)
    z_axis, tau_axis, sz_axis, sx_axis = (z_axis[order], tau_axis[order],
                                          sigma[0, 0, order, 2, 2], sigma[0, 0, order, 0, 0])
    k = int(np.argmax(tau_axis))
    zz = z_axis[k - 1:k + 2]                                  # parabola through the peak
    coeff = np.polyfit(zz, tau_axis[k - 1:k + 2], 2)
    z_peak = -coeff[1] / (2 * coeff[0])
    tau_peak = np.polyval(coeff, z_peak)
    # force-approach law: delta = c1 F^(2/3) + c2 F
    design = np.column_stack((forces ** (2 / 3), forces))
    c1, c2 = np.linalg.lstsq(design, deltas, rcond=None)[0]
    # sphere stresses (for the figure)
    tree = cKDTree(result.master_coordinates)
    Xs = np.asarray(sphere_grid.coordinates)
    Cs = np.asarray(sphere_grid.cells)
    us = result.master_displacement[-1][tree.query(Xs)[1]]
    sigma_s, vol_s = tet_stresses(Xs, Cs, us)
    tau_s = max_shear(sigma_s)
    level = dict(
        n=(n_block, n_sphere), h=a_ref / n_block, hex=block_grid.cell_count, tets=sphere_grid.cell_count,
        dofs=3 * (len(block_grid.coordinates) + len(Xs)), contact_nodes=len(w), forces=forces, force=force,
        a_h=a_h, p0_h=p0_h, a_moment=a_moment, a_fit=a_fit, p0_fit=p0_fit, p_max=p.max(), rms=rms, rms_inner=rms_inner,
        r=r, p=p, z_axis=z_axis, tau_axis=tau_axis, sz_axis=sz_axis, sx_axis=sx_axis, z_peak=z_peak,
        tau_peak=tau_peak, c1=c1, c2=c2, gap=result.averaged_gap[-1], pressure=p,
        block_axes=block_grid.axes, block_centres=centres, block_tau=max_shear(sigma),
        Xs=Xs, Cs=Cs, tau_s=tau_s, vol_s=vol_s, sphere_groups=sphere_groups,
        seconds=time.perf_counter() - started,
    )
    del contact, result, block_grid, sphere_grid
    gc.collect()
    return level


levels = []
for n_block, n_sphere in reversed(mesh_levels):           # finest first: it needs the most memory
    lv = run_level(n_block, n_sphere)
    levels.insert(0, lv)
    print(f"h = a/{n_block:<4} block {lv['hex']:6d} hexahedra, sphere {lv['tets']:6d} tetrahedra, "
          f"{lv['dofs']:6d} unknowns, {lv['contact_nodes']} contact nodes: "
          f"F = {lv['force']:.1f} N, a = {lv['a_moment'] * 1e3:.4f} mm  ({lv['seconds']:.0f} s)")
fine = levels[-1]


# ## Comparison with the reference
#
# All Hertz values use the force transmitted on the finest mesh. The pressure is
# compared node by node with the Hertz ellipse (RMS deviation weighted with the mortar
# areas); the contact radius is measured from the second moment of the pressure and from
# a least-squares ellipse fit; the axis stresses are those of the first column of block
# cells (centres at $x=y=h/2\approx0.04a$, where they differ from the axis values by far
# less than the discretization error). Complementarity: no penetration and zero
# pressure wherever the surfaces are open. The far-field coefficient $c_2$ comes out
# negative: the platen and the support sit at a finite distance from the contact, where
# a half-space would still move, so the finite assembly is slightly stiffer than two
# half-spaces referred to infinitely distant points.

def deviation(value, reference):
    return f"{value:12.4g}  {reference:12.4g}  {100 * (value / reference - 1):+7.2f} %"


a_h, p0_h = fine["a_h"], fine["p0_h"]
c1_hertz = (9 / (16 * R * E_star**2)) ** (1 / 3)
print(f"Finest mesh: transmitted force F = {fine['force']:.1f} N (full sphere)")
print("quantity                                   OneSimulation     Hertz      deviation")
print("contact radius a, pressure moment [mm]  ", deviation(fine["a_moment"] * 1e3, a_h * 1e3))
print("contact radius a, ellipse fit [mm]      ", deviation(fine["a_fit"] * 1e3, a_h * 1e3))
print("peak pressure p0, ellipse fit [GPa]     ", deviation(fine["p0_fit"] / 1e9, p0_h / 1e9))
print("largest nodal pressure [GPa]            ", deviation(fine["p_max"] / 1e9, p0_h / 1e9))
print("max subsurface shear tau1 / p0          ", deviation(fine["tau_peak"] / p0_h, TAU_HERTZ))
print("depth of the max shear z / a            ", deviation(fine["z_peak"] / a_h, Z_HERTZ))
print("Hertz coefficient c1 of delta(F)        ", deviation(fine["c1"], c1_hertz))
print(f"far-field compliance c2 = {fine['c2'] * 1e9:+.3f} nm/N "
      f"(delta_far / delta at the largest load: {fine['c2'] * fine['force'] / deltas[-1]:+.3f})")
print(f"RMS deviation of the nodal pressures from the Hertz ellipse: {100 * fine['rms']:.1f} % over the whole "
      f"contact, {100 * fine['rms_inner']:.1f} % for r < 0.8a")
z_ref = fine["z_axis"]
sz_ref, sr_ref = axis_stresses(z_ref, a_h, p0_h)
band = z_ref < 3 * a_h
print(f"axis stresses for z < 3a, max |FE - Hertz| / p0: sigma_z {np.abs(fine['sz_axis'] - sz_ref)[band].max() / p0_h:.3f}, "
      f"sigma_r {np.abs(fine['sx_axis'] - sr_ref)[band].max() / p0_h:.3f}")
print(f"largest penetration {max(0.0, -fine['gap'].min()) * 1e9:.2e} nm, "
      f"max |gap x pressure| {np.max(np.abs(fine['gap'] * fine['pressure'])):.1e} N/m")

# **Mesh study.** Three meshes with the contact cell size reduced by a factor of about
# 1.3 each time (block and sphere together). For every level the quantities are
# normalized with the Hertz values for that level's own transmitted force; the observed
# order $p$ solves $(f_1-f_2)/(f_2-f_3)=(h_1^p-h_2^p)/(h_2^p-h_3^p)$ and the Richardson
# estimate is $f_3+(f_3-f_2)/((h_2/h_3)^p-1)$.

def richardson(values, sizes):
    (f1, f2, f3), (h1, h2, h3) = values, sizes
    ratio = (f1 - f2) / (f2 - f3)
    g = lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio
    p = brentq(g, 0.2, 6.0) if g(0.2) * g(6.0) < 0 else float("nan")
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1)


sizes = [lv["h"] for lv in levels]
print("\nmesh study (ratios to Hertz for each level's own force)")
print("quantity          " + "".join(f"   h = a/{lv['n'][0]:<5}" for lv in levels)
      + "  change last two  order  Richardson  vs Hertz")
for label, key, reference in (("a / a_H", "a_moment", "a_h"), ("p0 fit / p0_H", "p0_fit", "p0_h"),
                              ("tau1 max / p0_H", "tau_peak", "p0_h")):
    vals = [lv[key] / lv[reference] for lv in levels]
    target = TAU_HERTZ if key == "tau_peak" else 1.0
    p_obs, extrapolated = richardson(vals, sizes)
    print(f"{label:18s}" + "".join(f"{v:13.5f}" for v in vals)
          + f"   {100 * (vals[2] / vals[1] - 1):+7.3f} %   {p_obs:5.2f}   {extrapolated:8.5f}"
          + f"   {100 * (extrapolated / target - 1):+6.2f} %")
forces = [lv["force"] for lv in levels]
print(f"transmitted force F [N]: " + ", ".join(f"{f:.1f}" for f in forces)
      + f"  (change last two {100 * (forces[2] / forces[1] - 1):+.2f} %)")


# ## Figure
#
# Top left: the quarter model in 3D, coloured with the maximum shear stress; the cut
# planes $x=0$ and $y=0$ show the interior. Top centre: the section $y=0$ around the
# contact (mirrored to $-x$), sphere above and block below; the white contour marks the
# Hertz value $0.31p_0$. Top right: contact pressure of every contact node against the
# Hertz ellipse. Bottom: stresses on the axis against Johnson's closed form, the
# force-approach law, and the mesh study.

tau_scale = 1 / p0_h
fig = plt.figure(figsize=(16, 10), constrained_layout=True)
grid_spec = fig.add_gridspec(2, 3)

# 3D view of the quarter model
ax = fig.add_subplot(grid_spec[0, 0], projection="3d")
norm = matplotlib.colors.Normalize(0.0, 0.32)
xs, ys, zs = fine["block_axes"]
tau_b = fine["block_tau"] * tau_scale
polys, values = [], []
for i, j in product(range(len(xs) - 1), range(len(zs) - 1)):         # y = 0 face
    polys.append([(xs[i], 0, zs[j]), (xs[i + 1], 0, zs[j]), (xs[i + 1], 0, zs[j + 1]), (xs[i], 0, zs[j + 1])])
    values.append(tau_b[i, 0, j])
for i, j in product(range(len(ys) - 1), range(len(zs) - 1)):         # x = 0 face
    polys.append([(0, ys[i], zs[j]), (0, ys[i + 1], zs[j]), (0, ys[i + 1], zs[j + 1]), (0, ys[i], zs[j + 1])])
    values.append(tau_b[0, i, j])
for i, j in product(range(len(xs) - 1), range(len(ys) - 1)):         # top face
    polys.append([(xs[i], ys[j], 0), (xs[i + 1], ys[j], 0), (xs[i + 1], ys[j + 1], 0), (xs[i], ys[j + 1], 0)])
    values.append(tau_b[i, j, -1])
Xs, Cs, tau_s = fine["Xs"], fine["Cs"], fine["tau_s"] * tau_scale
node_tau = np.zeros(len(Xs))
node_vol = np.zeros(len(Xs))
np.add.at(node_tau, Cs.ravel(), np.repeat(tau_s * fine["vol_s"], 4))
np.add.at(node_vol, Cs.ravel(), np.repeat(fine["vol_s"], 4))
node_tau /= node_vol
for name in ("plane_x", "plane_y", "cap", "surface"):
    for facet in fine["sphere_groups"][name]:
        polys.append(Xs[list(facet)])
        values.append(node_tau[list(facet)].mean())
collection = Poly3DCollection([np.asarray(q) * 1e3 for q in polys], facecolors=FIELD_CMAP(norm(np.array(values))),
                              edgecolors="none")
ax.add_collection3d(collection)
ax.set(xlim=(0, 10), ylim=(0, 10), zlim=(-5, 10), xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
ax.set_box_aspect((10, 10, 15))
ax.view_init(elev=18, azim=225)
ax.set_title("Quarter model: max shear stress / p0")
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax, shrink=0.6, label=r"$\tau_1/p_0$")

# section y = 0 around the contact
ax = fig.add_subplot(grid_spec[0, 1])
zoom = 2.5 * a_h
cx, cz = fine["block_centres"][:, 0, :, 0], fine["block_centres"][:, 0, :, 2]
levels_tau = np.linspace(0, 0.32, 17)
mx = np.concatenate((-cx[::-1], cx)) / a_h                   # mirror the quarter to -x
mz = np.concatenate((cz[::-1], cz)) / a_h
mt = np.concatenate((tau_b[::-1, 0, :], tau_b[:, 0, :]))
ax.contourf(mx, mz, mt, levels=levels_tau, cmap=FIELD_CMAP, extend="max")
ax.contour(mx, mz, mt, levels=[0.31], colors="white", linewidths=1.2)
plane = np.array(fine["sphere_groups"]["plane_y"])
for sign in (1, -1):
    tri = Triangulation(sign * Xs[:, 0] / a_h, Xs[:, 2] / a_h, plane)
    ax.tricontourf(tri, node_tau, levels=levels_tau, cmap=FIELD_CMAP, extend="max")
    ax.tricontour(tri, node_tau, levels=[0.31], colors="white", linewidths=1.2)
ax.axhline(0, color=INK, lw=0.8)
ax.plot([0], [-Z_HERTZ], "x", color=CREAM, ms=9, mew=2, label="Hertz maximum (0.31 p0 at z = 0.48a)")
ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal", xlabel="x / a", ylabel="z / a",
       title="Section y = 0: max shear stress / p0")
ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=CREAM)

# contact pressure
ax = fig.add_subplot(grid_spec[0, 2])
rr = np.linspace(0, 1.3, 300)
ax.plot(rr, np.sqrt(np.clip(1 - rr**2, 0, None)), color=COPPER, lw=2, label="Hertz ellipse")
for lv, marker, colour in zip(levels, ("s", "^", "o"), ("#B4AA98", "#8B9A8D", GREEN)):
    ax.plot(lv["r"] / lv["a_h"], lv["p"] / lv["p0_h"], marker, ms=3, color=colour, ls="none",
            label=f"OneSimulation, h = a/{lv['n'][0]}")
ax.set(xlim=(0, 1.3), xlabel="r / a", ylabel="p / p0", title="Contact pressure at every contact node")
ax.grid(alpha=0.4)
ax.legend(frameon=False, fontsize=9)

# axis stresses
ax = fig.add_subplot(grid_spec[1, 0])
zz = np.linspace(1e-3, 4, 400) * a_h
sz_h, sr_h = axis_stresses(zz, a_h, p0_h)
ax.plot(-sz_h / p0_h, zz / a_h, color=COPPER, lw=1.8, label=r"$-\sigma_z$ Hertz")
ax.plot(-sr_h / p0_h, zz / a_h, color=COPPER, lw=1.8, ls="--", label=r"$-\sigma_r$ Hertz")
ax.plot((sr_h - sz_h) / (2 * p0_h), zz / a_h, color=INK, lw=1.8, ls=":", label=r"$\tau_1$ Hertz")
ax.plot(-fine["sz_axis"] / p0_h, fine["z_axis"] / a_h, "o", ms=3.5, color=GREEN, label="OneSimulation")
ax.plot(-fine["sx_axis"] / p0_h, fine["z_axis"] / a_h, "s", ms=3.5, color=GREEN)
ax.plot(fine["tau_axis"] / p0_h, fine["z_axis"] / a_h, "^", ms=3.5, color=GREEN)
ax.set(ylim=(4, 0), xlabel="stress / p0", ylabel="depth below the contact centre z / a",
       title="Stresses on the axis")
ax.grid(alpha=0.4)
ax.legend(frameon=False, fontsize=9, loc="lower right")

# force-approach law
ax = fig.add_subplot(grid_spec[1, 1])
FF = np.linspace(0, fine["forces"].max() * 1.05, 200)
ax.plot(fine["forces"], deltas * 1e6, "o", color=GREEN, label="OneSimulation (imposed approach)")
ax.plot(FF, (fine["c1"] * FF ** (2 / 3) + fine["c2"] * FF) * 1e6, color=GREEN, lw=1,
        label=r"fit $c_1F^{2/3}+c_2F$")
ax.plot(FF, c1_hertz * FF ** (2 / 3) * 1e6, color=COPPER, lw=1.8, ls="--", label=r"Hertz term $c_1F^{2/3}$")
ax.plot(FF, fine["c1"] * FF ** (2 / 3) * 1e6, color=GREEN, lw=1.8, ls=":", label=r"fitted $c_1F^{2/3}$")
ax.set(xlabel="force on the full sphere F [N]", ylabel=r"approach $\delta$ [$\mu$m]",
       title="Force-approach law")
ax.grid(alpha=0.4)
ax.legend(frameon=False, fontsize=9)

# mesh study
ax = fig.add_subplot(grid_spec[1, 2])
hh = np.array([lv["n"][0] for lv in levels])
ax.plot(hh, [100 * (lv["a_moment"] / lv["a_h"] - 1) for lv in levels], "o-", label="contact radius a")
ax.plot(hh, [100 * (lv["p0_fit"] / lv["p0_h"] - 1) for lv in levels], "s-", label="p0 (ellipse fit)")
ax.plot(hh, [100 * (lv["tau_peak"] / lv["p0_h"] / TAU_HERTZ - 1) for lv in levels], "^-", label="max shear tau1")
ax.plot(hh, [100 * lv["rms"] for lv in levels], "d-", label="RMS pressure deviation")
ax.axhline(0, color=INK, lw=0.8)
ax.set(xlabel="contact cells per a_ref (block)", ylabel="deviation from Hertz [%]", title="Mesh study")
ax.grid(alpha=0.4)
ax.legend(frameon=False, fontsize=9)
plt.show()


# ## Going further
#
# - Give the sphere and the block different materials (for example a ceramic ball on a
#   steel race): only $E^*$ changes, and the mortar model needs no other modification.
# - Add a tangential load with `friction_coefficient` to study partial slip (Cattaneo-
#   Mindlin) under the Hertz pressure.
# - Replace the sphere by an ellipsoidal or toroidal cap to obtain elliptical Hertz
#   contacts, as in ball bearings with conforming raceways.
