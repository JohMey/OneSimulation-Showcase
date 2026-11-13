"""OneSimulation example: iron sphere in an initially uniform magnetic field.
Requires the installed, activated OneSimulation runtime.

A small, highly permeable iron sphere (relative permeability up to 4000) placed in a
uniform 1 mT field concentrates the flux: inside, the field is uniform and three times
stronger, outside a dipole field is added. The 3D field is computed with two genuinely
different formulations - second-order Nedelec vector potential with a Coulomb-gauge
multiplier, and the magnetic scalar potential with curved quadratic Lagrange elements -
and graded against the exact solution, with a mesh study, a permeability sweep and an
air-domain-size study.
"""


# # Iron sphere in a uniform magnetic field
#
# An iron sphere of diameter 0.25 mm sits in a magnetic field that is uniform far away,
# $\mathbf B_0=B_0\hat{\mathbf z}$ with $B_0=1$ mT. There are no currents, so
# $\nabla\times\mathbf H=0$ and $\nabla\cdot\mathbf B=0$ with $\mathbf B=\mu_0\mu_r\mathbf H$
# in the iron ($\mu_r=4000$) and $\mathbf B=\mu_0\mathbf H$ in the air. The permeable
# sphere pulls the field lines into itself: the field inside becomes almost three times
# the applied field, and just outside it is strongly enhanced at the poles and nearly
# cancelled at the equator. The field depends on the distance from the centre *and* on
# the angle to the field direction, a genuinely three-dimensional (axisymmetric) field.
#
# **Reference (exact solution).** With the magnetic scalar potential $\mathbf H=-\nabla\psi$
# the problem separates in spherical coordinates (J. C. Maxwell, *A Treatise on
# Electricity and Magnetism*, Vol. II, §437, 1873; J. D. Jackson, *Classical
# Electrodynamics*, 3rd ed., §5.11, Wiley 1999). Inside, the field is exactly uniform,
#
# $$\mathbf B_\mathrm{in}=\frac{3\mu_r}{\mu_r+2}\,\mathbf B_0=2.99850\ \mathrm{mT}\quad(\mu_r=4000),$$
#
# and outside the applied field is superposed with the field of a point dipole at the
# centre,
#
# $$\mathbf B=\mathbf B_0+B_0K\frac{a^3}{r^3}\big(3(\hat{\mathbf z}\cdot\hat{\mathbf r})\hat{\mathbf r}-\hat{\mathbf z}\big),
#   \qquad K=\frac{\mu_r-1}{\mu_r+2},$$
#
# so that on the axis $B_z=B_0(1+2Ka^3/r^3)$ and in the equatorial plane
# $B_z=B_0(1-Ka^3/r^3)$. The ratio $B_\mathrm{in}/B_0$ is the *field-concentration factor*;
# it tends to 3 for $\mu_r\to\infty$.
#
# **Two formulations, two element families.**
# 1. *Vector potential* $\mathbf B=\nabla\times\mathbf A$,
#    $\nabla\times(\mu^{-1}\nabla\times\mathbf A)=0$: curl-conforming Nedelec elements of the
#    first kind, order 2, on straight-sided tetrahedra, with the Coulomb gauge
#    $\nabla\cdot\mathbf A=0$ enforced exactly by a nodal Lagrange multiplier (mixed
#    saddle-point system), sparse direct LU.
# 2. *Magnetic scalar potential* $\psi$, $\nabla\cdot(\mu\nabla\psi)=0$: isoparametric
#    quadratic Lagrange elements (P2) on curved 10-node tetrahedra, sparse direct LU. The
#    equation is the same scalar elliptic problem as electrostatics with the permittivity
#    replaced by the permeability, so OneSimulation's `PoissonEquation` model solves it
#    (potential $\to\psi$, permittivity $\to\mu$, electric field $\to\mathbf H$).
#
# Both are stationary problems (no time integration). The scalar potential is the classic
# choice for current-free regions; with $\mu_r=4000$ the interior $\mathbf H$ is only
# 1/1334 of the applied field, which makes it the harder test for that formulation.
#
# **Domain and boundary conditions.** The field is mirror-symmetric about the planes
# $x=0$ and $y=0$ (they contain the field direction) and antisymmetric in $\psi$ about the
# plane $z=0$. One octant of a sphere of radius $R=4a$ around the iron is modelled:
#
# | boundary | vector potential | scalar potential |
# |---|---|---|
# | symmetry planes $x=0$, $y=0$ | magnetic insulation $\mathbf n\times\mathbf A=0$ ($B_n=0$) | zero normal flux $\mu\,\partial_n\psi=0$ |
# | plane $z=0$ | zero tangential field $\mathbf n\times\mathbf H=0$ | $\psi=0$ |
# | outer sphere $r=R$ | exact $\mathbf A$ (uniform field + dipole) | exact $\psi$ (uniform field + dipole) |
#
# Imposing the exact far field (uniform field plus the dipole correction) on $r=R$ makes
# the truncated domain exact. The *air-domain-size study* replaces it with the bare
# uniform field and shows the truncation error that a too small air domain produces;
# for that case the truncated problem itself has a closed-form solution, so the
# finite-element result can be checked even with the wrong far field.
#
# **Scale model.** Without currents, magnetostatics has no intrinsic length: multiplying
# all lengths by a factor leaves $\mathbf B$ unchanged at corresponding points ($\mathbf A$
# scales with the length). The notebook therefore computes a 1000:1 model ($a=125$ mm
# instead of 0.125 mm) and reports positions in units of $a$. This keeps the
# gauge-multiplier block of the vector-potential system well scaled in SI units; for the
# real 0.125 mm sphere it would be $10^6$ times smaller than the curl-curl block and the
# LU residual would suffer.

from dataclasses import dataclass, replace
from importlib.metadata import version
import time
from collections import defaultdict

import numpy as np
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
from scipy.spatial import cKDTree

from onesim.cad.geometry.backends.occt.provider import OcctCadProvider
from onesim.cad.geometry.ports.kernel_port import SphereRequest, TopologyOperationRequest
from onesim.cad.geometry.primitives import Geometry
from onesim.cad.operations.execution.model_geometry import GeometryService
from onesim.materials import VacuumPermeability
from onesim.meshing import UnstructuredSimplexMesh
from onesim.meshing.generation.backends.cad.in_memory_cad import build_file_free_mesher
from onesim.meshing.generation.contracts.polynomial.tetrahedron import QuadraticTetrahedralGeometry
from onesim.meshing.generation.contracts.requests import (
    HighOrderMeshControl, MeshGenerationRequest, MeshRefinement, MeshRefinementSupport, SpatialMeshControl,
)
from onesim.meshing.generation.operations.geometry.model_mesh import CadMeshingService
from onesim.meshing.meshes.mesh_declarations import TetrahedralMesh
from onesim.meshing.meshes.models.geometry.discretization_data import CoordinateSet
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, Lagrange, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement
from onesim.physics.magnetostatics import (
    CoulombGauge, MagneticFieldBoundary, MagneticInsulation, MagneticPotentialBoundary,
    MagneticVectorPotential, Magnetostatics,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.problem_definition.values.declarations.immutable_values import FrozenMap

print("OneSimulation:", version("onesimulation"))
start_time = time.perf_counter()


# ## Parameters

a = 0.125                  # m, sphere radius of the 1000:1 scale model (real sphere: 0.125 mm)
B0 = 1e-3                  # T, applied uniform flux density along z
mu_r = 4000.0              # relative permeability of the iron
mu0 = float(VacuumPermeability().evaluate())
H0 = B0 / mu0
R = 4 * a                  # radius of the modelled air domain

mesh_levels = (0.18, 0.126, 0.088)     # interface cell size h_s / a (refinement ratio about 1.42)
sweep_mu_r = (2.0, 10.0, 100.0, 1000.0)   # permeability sweep (plus 4000 from the mesh study)
study_radii = (2.0, 3.0, 4.0)          # air-domain radii R / a for the uniform-field boundary


def K_factor(mur):
    return (mur - 1.0) / (mur + 2.0)


def B_inside_exact(mur):
    return 3.0 * mur / (mur + 2.0) * B0


print(f"exact interior field: {1e3 * B_inside_exact(mu_r):.5f} mT, concentration factor "
      f"{B_inside_exact(mu_r) / B0:.5f}, dipole factor K = {K_factor(mu_r):.6f}")


# ## Exact solution

def psi_exact(x, y, z, mur):
    r = np.sqrt(x * x + y * y + z * z)
    return np.where(r < a, -3.0 / (mur + 2.0) * H0 * z,
                    -H0 * z + H0 * K_factor(mur) * a**3 * z / np.maximum(r, 1e-30) ** 3)


def A_exact(x, y, z, mur):
    """Exact vector potential A = g(r) (-y, x, 0); it satisfies the Coulomb gauge."""
    r = np.sqrt(x * x + y * y + z * z)
    g = np.where(r < a, 0.5 * B_inside_exact(mur), 0.5 * B0 + B0 * K_factor(mur) * a**3 / np.maximum(r, 1e-30) ** 3)
    return np.stack((-g * y, g * x, np.zeros_like(x)), axis=-1)


def B_exact(points, mur):
    p = np.atleast_2d(points)
    r = np.linalg.norm(p, axis=1)
    rhat, zhat = p / r[:, None], np.array([0.0, 0.0, 1.0])
    B = B0 * zhat + (B0 * K_factor(mur) * a**3 / r**3)[:, None] * (3 * rhat[:, 2:3] * rhat - zhat)
    B[r < a] = B_inside_exact(mur) * zhat
    return B


def B_inside_truncated(mur, radius, boundary):
    """Interior field when the bare uniform field is imposed on r = R (closed form).

    For a dipole term D cos(theta)/r^2 in a < r < R the conditions at the sphere give
    D = K C a^3 as before; the outer condition fixes C. Prescribing the tangential A
    fixes B_r = B0 cos(theta) on r = R; prescribing psi fixes the tangential H.
    """
    q = (a / radius) ** 3
    factor = 1 + 2 * K_factor(mur) * q if boundary == "A" else 1 - K_factor(mur) * q
    return B_inside_exact(mur) / factor


# ## Geometry and mesh
#
# OneSimulation's CAD kernel builds the octant of the air sphere and the octant of the
# iron ball and partitions them into two solids with a shared spherical interface. The
# conforming tetrahedral mesh is graded: the cell size $h_s$ on the iron surface and along
# its three edge arcs, $2.5h_s$ in a ball of radius $1.6a$, growing to $0.7a$ towards the
# outer sphere. Inside the iron the field is uniform, so the interior is meshed more
# coarsely than the surface. The mesher returns curved quadratic (10-node) tetrahedra
# whose outer CAD faces are exact; the mid-edge nodes on the internal iron-air interface
# are additionally projected onto the sphere here, so that the scalar-potential model sees
# the interface curved as well. The vector-potential model uses the straight-sided
# (4-node) version of the same mesh.

GROUPS = ("plane_x", "plane_y", "plane_z", "outer")
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


@dataclass(frozen=True, eq=False, kw_only=True)
class AirOctant(Geometry):
    """Octant of the air sphere (with the iron inside) and its four named boundaries."""

    dimension = 3

    @property
    def volume(self):
        return self.domain()

    def boundary_groups(self):
        return {**{name: (name,) for name in GROUPS}, "all": GROUPS}


def interface_faces(cells, iron):
    owners = defaultdict(list)
    for k, cell in enumerate(cells):
        for face in TET_FACES:
            owners[tuple(sorted(cell[list(face)]))].append(k)
    return [face for face, cs in owners.items() if len(cs) == 2 and iron[cs[0]] != iron[cs[1]]]


def build_mesh(radius, hs):
    started = time.perf_counter()
    kernel = OcctCadProvider()
    cad = GeometryService("iron_sphere", kernel)
    outer = cad.construct(kernel.create_sphere, SphereRequest(cad.reference("outer"), radius))
    air = cad.boolean("intersection", outer, cad.box(width=radius, height=radius, depth=radius, name="box_air"),
                      name="air")
    ball = cad.construct(kernel.create_sphere, SphereRequest(cad.reference("ball"), a))
    iron = cad.boolean("intersection", ball, cad.box(width=radius, height=radius, depth=radius, name="box_iron"),
                       name="iron")
    part = cad.construct(kernel.topology_operation, TopologyOperationRequest(
        cad.reference("partition"), air.reference, iron.reference, "partition"))
    # Name the CAD faces from their geometry: planes by their normal, the iron surface by
    # its exact area (one eighth of 4 pi a^2), the rest is the outer sphere.
    faces, discs, surface = {name: [] for name in GROUPS}, [], None
    for face in kernel.structured_meshing_topology(part.reference).faces:
        if face.surface_type == "plane":
            faces["plane_" + "xyz"[int(np.argmax(np.abs(face.normal)))]].append(face.entity_id)
            if abs(face.area - np.pi * a * a / 4) < 1e-6 * a * a:
                discs.append(face.entity_id)
        elif abs(face.area - np.pi * a * a / 2) < 1e-6 * a * a:
            surface = face.entity_id
        else:
            faces["outer"].append(face.entity_id)
    body = next(e.entity_id for e in part.data.entities if e.kind == "body")
    arcs = [MeshRefinement(body, hs, "volume", MeshRefinementSupport(
        "ball", tuple(a * np.roll((np.cos(t), np.sin(t), 0.0), k)), 0.15 * a))
        for k in range(3) for t in np.linspace(0.0, np.pi / 2, 9)]
    controls = (MeshRefinement(surface, hs, "surface"), *arcs,
                *(MeshRefinement(disc, 2 * hs, "surface") for disc in discs),
                MeshRefinement(body, 2.5 * hs, "volume", MeshRefinementSupport("ball", (0.0, 0.0, 0.0), 1.6 * a)))
    request = MeshGenerationRequest(
        part.reference.artifact_id, part.reference.version, global_size=min(0.7 * a, 0.175 * radius),
        refinements=controls, max_candidate_cells=2_000_000, high_order=HighOrderMeshControl(polynomial_order=2),
        spatial=SpatialMeshControl("free_tetrahedral", body, engine="cad_native"))
    native = CadMeshingService(provider_factory=build_file_free_mesher).generate(part, request)
    domain = AirOctant()

    X4, C4 = np.asarray(native.data.coordinates), np.asarray(native.data.cells)
    solids = np.asarray(native.data.cell_domain_ids)
    centroid_r = np.linalg.norm(X4[C4].mean(axis=1), axis=1)
    iron_cells = solids == solids[np.argmin(centroid_r)]

    # Curved 10-node mesh: project the interface mid-edge nodes onto r = a.
    carrier = native.carrier
    X10, C10 = np.array(carrier.coordinates.values), np.asarray(carrier.connectivity.entities)
    midpoint = {tuple(sorted((c[i], c[j]))): c[4 + m] for c in C10 for m, (i, j) in enumerate(TET_EDGES)}
    moved = {midpoint[tuple(sorted(e))] for f in interface_faces(C10[:, :4], iron_cells)
             for e in ((f[0], f[1]), (f[0], f[2]), (f[1], f[2]))}
    for node in moved:
        X10[node] *= a / np.linalg.norm(X10[node])
    curved = replace(carrier, coordinates=CoordinateSet(tuple(map(tuple, X10)), unit="m"),
                     measures=tuple(map(float, QuadraticTetrahedralGeometry(X10, C10).volumes)))
    mesh10 = TetrahedralMesh(domain, replace(native, carrier=curved),
                             FrozenMap({k: tuple(v) for k, v in faces.items()}), ())

    # Straight-sided mesh with the exterior facets named by their position.
    named = {name: [] for name in GROUPS}
    for facet in boundary_faces(C4, X4):
        on_plane = [i for i in range(3) if np.all(np.abs(X4[list(facet.vertices), i]) < 1e-9 * radius)]
        named["plane_" + "xyz"[on_plane[0]] if on_plane else "outer"].append(facet.vertices)
    mesh4 = UnstructuredSimplexMesh(geometry=domain, coordinates=X4, cells=C4, boundary_facets=named)
    print(f"R = {radius / a:.0f} a, h_s = {hs / a:.3f} a: {len(C4)} tetrahedra ({iron_cells.sum()} in the iron), "
          f"{len(moved)} interface nodes curved, {time.perf_counter() - started:.0f} s")
    return dict(domain=domain, mesh10=mesh10, mesh4=mesh4, X=X4, C=C4, iron=iron_cells, cells=len(C4), hs=hs,
                radius=radius)


# ## Model
#
# **Permeability.** The vector-potential model works on straight-sided cells, so its iron
# region is the polyhedron bounded by the flat interface triangles: a point is iron if
# the ray from the centre reaches it before crossing that faceted surface. The
# scalar-potential model works on the curved mesh, whose interface is the sphere itself,
# so $r<a$ decides. Either way the material jump lies exactly on cell faces.

class FacetedIronPermeability:
    """mu0 mu_r inside the faceted iron polyhedron, mu0 in the air (pointwise)."""

    def __init__(self, mesh, mur):
        triangles = mesh["X"][np.array(interface_faces(mesh["C"], mesh["iron"]))]
        self.mur = mur
        self.inverse = np.linalg.inv(np.transpose(triangles, (0, 2, 1)))
        centres = triangles.mean(axis=1)
        self.tree = cKDTree(centres / np.linalg.norm(centres, axis=1)[:, None])
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        self.inner_radius = np.min(np.abs(np.einsum("ij,ij->i", normals, triangles[:, 0]))
                                   / np.linalg.norm(normals, axis=1))

    def is_iron(self, p):
        r = np.linalg.norm(p)
        if r >= a or r < self.inner_radius:
            return r < a
        # Barycentric weights of the ray direction in the nearby interface triangles; the
        # triangle hit by the ray has all weights non-negative (largest minimum weight).
        candidates = self.tree.query(p / r, k=12)[1]
        weights = np.einsum("kij,j->ki", self.inverse[candidates], p)
        hit = weights[np.argmax(weights.min(axis=1))]
        return hit.sum() < 1.0

    def evaluate(self, **state):
        p = np.array([float(state["x"]), float(state["y"]), float(state["z"])])
        return mu0 * self.mur if self.is_iron(p) else mu0


class CurvedIronPermeability:
    def __init__(self, mur):
        self.mur = mur

    def evaluate(self, **state):
        x, y, z = (np.asarray(state[k], float) for k in "xyz")
        mu = np.where(x * x + y * y + z * z < a * a, mu0 * self.mur, mu0)
        return mu if mu.ndim else float(mu)


def solve_vector_potential(mesh, mur, far_field="exact"):
    started = time.perf_counter()
    domain = mesh["domain"]
    model = Magnetostatics(domain=domain.volume, permeability=FacetedIronPermeability(mesh, mur),
                           formulation=MagneticVectorPotential(gauge=CoulombGauge()))
    if far_field == "exact":
        outer_value = lambda x, y, z: A_exact(x, y, z, mur)
    else:
        outer_value = lambda x, y, z: np.stack((-0.5 * B0 * y, 0.5 * B0 * x, 0.0 * x), axis=-1)
    conditions = (
        MagneticPotentialBoundary(boundary=domain.boundaries.outer, physics=model, value=outer_value),
        MagneticInsulation(boundary=domain.boundaries.plane_x | domain.boundaries.plane_y, physics=model),
        MagneticFieldBoundary(boundary=domain.boundaries.plane_z, physics=model, value=(0.0, 0.0, 0.0)),
    )
    problem = Problem(name="Iron sphere, vector potential", geometry=domain, physics=(model,),
                      boundary_conditions=conditions)
    result = Simulation(problem=problem, mesh=mesh["mesh4"], numerics=Numerics(
        spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=SparseDirect())).solve_stationary()
    B = result.field(model.magnetic_flux_density)
    A = result.field(model.vector_potential)
    return dict(B=lambda p: np.asarray(B.evaluate(np.atleast_2d(p))),
                A=lambda p: np.asarray(A.evaluate(np.atleast_2d(p))), seconds=time.perf_counter() - started)


def solve_scalar_potential(mesh, mur, far_field="exact"):
    started = time.perf_counter()
    domain = mesh["domain"]
    model = PoissonEquation(domain=domain.volume, permittivity=CurvedIronPermeability(mur))
    if far_field == "exact":
        outer_value = lambda x, y, z: psi_exact(x, y, z, mur)
    else:
        outer_value = lambda x, y, z: -H0 * z
    conditions = (
        FixedPotential(boundary=domain.boundaries.outer, physics=model, value=outer_value),
        FixedPotential(boundary=domain.boundaries.plane_z, physics=model, value=0.0),
        ZeroNormalElectricDisplacement(boundary=domain.boundaries.plane_x | domain.boundaries.plane_y, physics=model),
    )
    problem = Problem(name="Iron sphere, scalar potential", geometry=domain, physics=(model,),
                      boundary_conditions=conditions)
    result = Simulation(problem=problem, mesh=mesh["mesh10"], numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())).solve_stationary()
    H = result.field(model.electric_field)          # H = -grad(psi)

    def B(points):
        p = np.atleast_2d(points)
        mu = np.where(np.linalg.norm(p, axis=1) < a, mu0 * mur, mu0)
        return mu[:, None] * np.asarray(H.evaluate(p))

    return dict(B=B, seconds=time.perf_counter() - started)


# ## Solve
#
# Observables of every run:
# - the **flux through the iron's mid-plane** $\Phi$ (the section $z=0$, $r<a$), exact
#   $B_\mathrm{in}\pi a^2$. With the vector potential it follows from Stokes' theorem as the
#   circulation of $\mathbf A$ around the equator of the sphere (the tangential $\mathbf A$ of
#   Nedelec elements is continuous there); with the scalar potential $B_z$ is integrated over
#   the section with a Gauss-Legendre polar rule;
# - the volume-averaged interior field $\langle B_z\rangle$ (4-point Gauss rule on the straight
#   iron cells, which lie inside the iron for both meshes);
# - the dipole factor $K$, least-squares fitted to $\mathbf B-\mathbf B_0$ at 720 points of the
#   shell $1.1a\le r\le 2a$ (all angles), where the dipole part of the field is strongest.

GAUSS = np.full((4, 4), 0.1381966011250105) + np.eye(4) * (0.5854101966249685 - 0.1381966011250105)
offset = 1e-6 * a                       # keeps sample points strictly inside the octant
phi_ring = (np.arange(400) + 0.5) * (np.pi / 2) / 400
equator = np.column_stack((a * (1 - 1e-6) * np.cos(phi_ring), a * (1 - 1e-6) * np.sin(phi_ring),
                           np.full_like(phi_ring, offset)))
g_r, g_w = np.polynomial.legendre.leggauss(24)
g_r, g_w = 0.5 * a * (g_r + 1), 0.5 * a * g_w
P_r, P_phi = np.meshgrid(g_r, phi_ring[::8], indexing="ij")
section = np.column_stack(((P_r * np.cos(P_phi)).ravel(), (P_r * np.sin(P_phi)).ravel(), np.full(P_r.size, offset)))
section_weights = (g_w[:, None] * g_r[:, None] * np.full(P_phi.shape, np.pi / 2 / P_phi.shape[1])).ravel()
s_r, s_theta, s_phi = np.meshgrid(np.linspace(1.1, 2.0, 10) * a, np.linspace(0.04, 1.53, 12),
                                  np.linspace(0.06, 1.51, 6), indexing="ij")
shell = np.column_stack(((s_r * np.sin(s_theta) * np.cos(s_phi)).ravel(),
                         (s_r * np.sin(s_theta) * np.sin(s_phi)).ravel(), (s_r * np.cos(s_theta)).ravel()))
shell_r = np.linalg.norm(shell, axis=1)
dipole_shape = (B0 * a**3 / shell_r**3)[:, None] * (3 * shell[:, 2:3] / shell_r[:, None] * shell / shell_r[:, None]
                                                     - np.array([0.0, 0.0, 1.0]))


def observables(mesh, run, dipole=True):
    corners = mesh["X"][mesh["C"][mesh["iron"]]]
    volumes = np.abs(np.einsum("ni,ni->n", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                               corners[:, 3] - corners[:, 0])) / 6
    Bz = run["B"](np.einsum("qv,nvd->nqd", GAUSS, corners).reshape(-1, 3))[:, 2].reshape(-1, 4)
    run["B_mean"] = np.sum(volumes * Bz.mean(axis=1)) / volumes.sum()
    if dipole:
        if "A" in run:       # Stokes: flux = circulation of A (x 4 for the full section)
            A = run["A"](equator)
            run["flux"] = 4 * np.mean(-A[:, 0] * np.sin(phi_ring) + A[:, 1] * np.cos(phi_ring)) * (np.pi / 2) * a
        else:
            run["flux"] = 4 * np.sum(section_weights * run["B"](section)[:, 2])
        dB = run["B"](shell) - np.array([0.0, 0.0, B0])
        run["K"] = np.sum(dB * dipole_shape) / np.sum(dipole_shape**2)
    return run


study = []
for level in mesh_levels:
    mesh = build_mesh(R, level * a)
    runs = {"A": observables(mesh, solve_vector_potential(mesh, mu_r)),
            "psi": observables(mesh, solve_scalar_potential(mesh, mu_r))}
    study.append((mesh, runs))
    print(f"   vector potential {runs['A']['seconds']:.0f} s, scalar potential {runs['psi']['seconds']:.0f} s")
final_mesh, final = study[-1]
coarse_mesh = study[0][0]

# Permeability sweep (vector potential, coarsest mesh) and air-domain-size study with the
# bare uniform field on r = R (coarsest mesh size, both formulations).
sweep = {4000.0: study[0][1]["A"]["B_mean"]}
for mur in sweep_mu_r:
    sweep[mur] = observables(coarse_mesh, solve_vector_potential(coarse_mesh, mur), dipole=False)["B_mean"]
truncation = []
for factor in study_radii:
    mesh = coarse_mesh if factor * a == R else build_mesh(factor * a, mesh_levels[0] * a)
    truncation.append((factor,
                       observables(mesh, solve_vector_potential(mesh, mu_r, far_field="uniform"), False)["B_mean"],
                       observables(mesh, solve_scalar_potential(mesh, mu_r, far_field="uniform"), False)["B_mean"]))


# ## Comparison with the reference
#
# **Mesh study** (exact far field on $r=4a$). For three meshes with interface size $h_s$
# the observed order $p$ follows from successive differences,
# $p=\ln\frac{f_1-f_2}{f_2-f_3}/\ln r$ for the (nearly constant) refinement ratio $r$, and
# the Richardson estimate is $f_3+(f_3-f_2)/(r^p-1)$.

B_ref, K_ref = B_inside_exact(mu_r), K_factor(mu_r)
flux_ref = B_ref * np.pi * a**2
labels = {"A": "vector potential", "psi": "scalar potential"}
quantities = (("flux", flux_ref, "mid-plane flux"), ("B_mean", B_ref, "<B_z> inside"), ("K", K_ref, "dipole factor K"))
print(f"\nExact: mid-plane flux = {flux_ref / (B0 * np.pi * a**2):.5f} B0 pi a^2, <B_z> = {1e3 * B_ref:.5f} mT, "
      f"K = {K_ref:.6f}. Relative deviations:")
print(" h_s / a   tetrahedra  | flux A      <Bz> A      K A        | flux psi    <Bz> psi    K psi")
for mesh, runs in study:
    print(f" {mesh['hs'] / a:6.3f}   {mesh['cells']:10d}  | "
          + "  ".join(f"{runs['A'][k] / r - 1:+.3e}" for k, r, _ in quantities) + " | "
          + "  ".join(f"{runs['psi'][k] / r - 1:+.3e}" for k, r, _ in quantities))
step = mesh_levels[1] / mesh_levels[2]
for name in ("A", "psi"):
    for key, ref, label in quantities:
        f1, f2, f3 = (runs[name][key] for _, runs in study)
        p = np.log(abs((f1 - f2) / (f2 - f3))) / np.log(step)
        rich = f3 + (f3 - f2) / (step**p - 1)
        print(f" {labels[name]}, {label:16s}: change between the last two meshes {f3 / f2 - 1:+.2e}, "
              f"observed order {p:5.2f}, Richardson estimate {rich / ref - 1:+.2e} from exact")

# How to read the mesh study: the vector-potential model sees a *faceted* sphere (straight
# tetrahedra), so its mid-plane flux carries the area deficit of the polygonal section,
# which falls like $h_s^2$; its volume-averaged interior field is already at the 1e-4 level
# on all meshes, where the level-to-level changes are dominated by the unstructured
# remeshing rather than by a smooth error expansion. The scalar-potential model has the
# curved interface and no such geometric error. The dipole factor is the most sensitive
# quantity because it is extracted from the near field outside the sphere.
#
# **Field along lines** on the finest mesh: inside the iron, along the field axis and in
# the equatorial plane, against the exact solution.
print("\nB_z on the finest mesh [mT]: exact | vector potential (dev) | scalar potential (dev)")
samples = [("centre", (0.0, 0.0, 0.0)), ("inside, axis 0.6a", (0, 0, 0.6)), ("inside, equator 0.6a", (0.6, 0, 0)),
           ("inside, oblique", (0.4, 0.3, 0.5)), ("axis 1.1a", (0, 0, 1.1)), ("axis 1.5a", (0, 0, 1.5)),
           ("axis 3a", (0, 0, 3.0)), ("equator 1.1a", (1.1, 0, 0)), ("equator 1.5a", (1.5, 0, 0)),
           ("equator 3a", (3.0, 0, 0)), ("oblique 1.4a", (0.7, 0.7, 0.99))]
for label, point in samples:
    p = np.maximum(np.array(point) * a, offset)[None, :]
    exact = B_exact(p, mu_r)[0, 2]
    values = [final[name]["B"](p)[0, 2] for name in ("A", "psi")]
    print(f" {label:21s} {1e3 * exact:9.5f} | {1e3 * values[0]:9.5f} ({values[0] / exact - 1:+.1e}) | "
          f"{1e3 * values[1]:9.5f} ({values[1] / exact - 1:+.1e})")
r_line = np.linspace(1.02, 3.9, 200) * a
for label, pts in (("axis", np.column_stack((np.full_like(r_line, offset), np.full_like(r_line, offset), r_line))),
                   ("equator", np.column_stack((r_line, np.full_like(r_line, offset), np.full_like(r_line, offset))))):
    exact = B_exact(pts, mu_r)
    errors = [np.sqrt(np.mean(np.sum((final[n]["B"](pts) - exact) ** 2, axis=1))) / B0 for n in ("A", "psi")]
    print(f" RMS |B - B_exact| / B0 on the {label} line 1.02a...3.9a: vector potential {errors[0]:.1e}, "
          f"scalar potential {errors[1]:.1e}")

# **Field-concentration factor** for the permeability range of the original study
# (vector potential, coarsest mesh).
print("\nConcentration factor B_in / B0 (vector potential, coarsest mesh)")
for mur in sorted(sweep):
    print(f" mu_r = {mur:6.0f}: computed {sweep[mur] / B0:.5f}, exact {B_inside_exact(mur) / B0:.5f} "
          f"({sweep[mur] / B_inside_exact(mur) - 1:+.1e})")

# **Air-domain size.** With only the uniform field imposed on $r=R$ the interior field is
# wrong by a truncation error of order $(a/R)^3$ - too low with the vector potential
# (prescribed normal $B$), too high with the scalar potential (prescribed tangential
# $H$). The finite-element values must follow the closed-form values of the truncated
# problem.
print("\nUniform-field boundary on r = R (coarsest mesh size): <Bz> / exact infinite-domain value - 1")
print("  R / a   vector potential   truncated exact    scalar potential   truncated exact")
for factor, B_A, B_psi in truncation:
    print(f"  {factor:4.1f}   {B_A / B_ref - 1:+.3e}         {B_inside_truncated(mu_r, factor * a, 'A') / B_ref - 1:+.3e}"
          f"         {B_psi / B_ref - 1:+.3e}         {B_inside_truncated(mu_r, factor * a, 'psi') / B_ref - 1:+.3e}")
print(f"exact dipole far field on r = 4a, same mesh: vector potential {study[0][1]['A']['B_mean'] / B_ref - 1:+.3e}, "
      f"scalar potential {study[0][1]['psi']['B_mean'] / B_ref - 1:+.3e}")
print(f"Total run time: {time.perf_counter() - start_time:.0f} s")


# ## Figure
#
# Left: magnitude of $\mathbf B$ in the meridian plane $y=0$ from the vector-potential
# solution on the finest mesh (the computed quadrant mirrored about the symmetry
# planes) with the field lines, which are the contours of the flux function
# $\rho A_\varphi$. Middle: $B_z$ along the axis and in the equatorial plane for both
# formulations against the exact solution, and the concentration factor over $\mu_r$.
# Right: mesh convergence of the mid-plane flux, the mean interior field and the dipole
# factor, and the air-domain-size study.

n_grid = 181
g = np.linspace(0.0, 2.5, n_grid) * a
GX, GZ = np.meshgrid(g, g)
pts = np.column_stack((np.maximum(GX.ravel(), offset), np.full(GX.size, offset), np.maximum(GZ.ravel(), offset)))
B_map = np.linalg.norm(final["A"]["B"](pts), axis=1).reshape(GX.shape)
flux = (pts[:, 0] * final["A"]["A"](pts)[:, 1]).reshape(GX.shape)       # rho * A_phi on y = 0
full = lambda F: np.block([[F[::-1, ::-1], F[::-1, 1:]], [F[1:, ::-1], F[1:, 1:]]])
X_full = np.concatenate((-g[::-1], g[1:])) / a
B_full, flux_full = full(B_map), full(flux)

fig = plt.figure(figsize=(17, 9.4), constrained_layout=True)
layout = fig.add_gridspec(2, 3, width_ratios=(1.35, 1, 1))
ax = fig.add_subplot(layout[:, 0])
image = ax.contourf(X_full, X_full, 1e3 * B_full, levels=np.linspace(0, 3.2, 65), cmap=FIELD_CMAP, extend="max")
ax.contour(X_full, X_full, flux_full, levels=B0 / 2 * (np.arange(0.25, 2.6, 0.25) * a) ** 2,
           colors=INK, linewidths=0.8)
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=CREAM, lw=1.4, ls="--"))
ax.set_aspect("equal")
ax.set(xlabel="x / a", ylabel="z / a  (applied field along +z)",
       title=f"|B| and field lines in the meridian plane (mu_r = {mu_r:.0f}, B0 = 1 mT)")
fig.colorbar(image, ax=ax, label="|B| / mT", shrink=0.8, ticks=np.arange(0, 3.3, 0.5))

ax = fig.add_subplot(layout[0, 1])
r_plot = np.linspace(0.0, 3.9, 400) * a
for label, direction, colour in (("axis", 2, COPPER), ("equator", 0, GREEN)):
    line = np.full((len(r_plot), 3), offset)
    line[:, direction] = np.maximum(r_plot, offset)
    ax.plot(r_plot / a, 1e3 * B_exact(line, mu_r)[:, 2], color=colour, lw=1.6, label=f"exact, {label}")
    marks = np.full((22, 3), offset)
    marks[:, direction] = np.linspace(0.05, 3.85, 22) * a
    ax.plot(marks[:, direction] / a, 1e3 * final["A"]["B"](marks)[:, 2], "o", ms=5, mfc="none", color=colour,
            label=f"vector potential, {label}")
    ax.plot(marks[:, direction] / a, 1e3 * final["psi"]["B"](marks)[:, 2], "x", ms=5, color=colour,
            label=f"scalar potential, {label}")
ax.axvline(1.0, color="#8B9A8D", lw=0.8, ls=":")
ax.set(xlabel="distance from the centre r / a", ylabel="B_z / mT", title="Field along the axis and the equator")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=2)

ax = fig.add_subplot(layout[1, 1])
mur_line = np.logspace(0, 4, 200)
ax.semilogx(mur_line, B_inside_exact(mur_line) / B0, color=INK, lw=1.4, label="exact 3 mu_r / (mu_r + 2)")
murs = np.array(sorted(sweep))
ax.semilogx(murs, [sweep[m] / B0 for m in murs], "o", color=COPPER, ms=7, mfc="none",
            label="vector potential (coarsest mesh)")
ax.semilogx([mu_r], [final["psi"]["B_mean"] / B0], "x", color=GREEN, ms=9, label="scalar potential (finest mesh)")
ax.axhline(3.0, color="#8B9A8D", lw=0.8, ls=":")
ax.set(xlabel="relative permeability mu_r", ylabel="B_in / B0", title="Field-concentration factor")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = fig.add_subplot(layout[0, 2])
cells = np.array([mesh["cells"] for mesh, _ in study])
for name, colour in (("A", COPPER), ("psi", GREEN)):
    for (key, ref, label), marker, style in zip(quantities, ("o", "s", "^"), ("-", ":", "--")):
        ax.loglog(cells, [abs(runs[name][key] / ref - 1) for _, runs in study], marker + style, color=colour,
                  mfc="none" if key == "K" else colour, label=f"{label}, {labels[name]}")
ax.set(xlabel="tetrahedra (octant)", ylabel="|relative deviation from exact|", title="Mesh convergence")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, ncol=1)

ax = fig.add_subplot(layout[1, 2])
inv_R = np.linspace(0.0, 0.52, 100)
radius_line = a / np.maximum(inv_R, 1e-9)
ax.plot(inv_R, 100 * (B_inside_truncated(mu_r, radius_line, "A") / B_ref - 1), color=COPPER, lw=1.2,
        label="closed form, A prescribed")
ax.plot(inv_R, 100 * (B_inside_truncated(mu_r, radius_line, "psi") / B_ref - 1), color=GREEN, lw=1.2,
        label="closed form, psi prescribed")
ax.plot([1 / f for f, _, _ in truncation], [100 * (b / B_ref - 1) for _, b, _ in truncation], "o", color=COPPER,
        mfc="none", ms=7, label="vector potential, uniform far field")
ax.plot([1 / f for f, _, _ in truncation], [100 * (b / B_ref - 1) for _, _, b in truncation], "s", color=GREEN,
        mfc="none", ms=7, label="scalar potential, uniform far field")
ax.plot([a / R] * 2, [100 * (study[0][1][n]["B_mean"] / B_ref - 1) for n in ("A", "psi")], "*", color=INK, ms=10,
        label="exact dipole far field (R = 4a)")
ax.axhline(0.0, color="#8B9A8D", lw=0.8)
ax.set(xlabel="a / R  (air-domain radius R)", ylabel="interior field deviation / %",
       title="Air-domain size with a uniform far field")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5)
plt.show()


# ## Going further
#
# Replace the sphere by a prolate spheroid: the interior field stays uniform and the
# concentration factor becomes $1/(N+1/(\mu_r-1))$ with the demagnetising factor $N$ of
# the spheroid, which approaches $1/N$ for iron - a needle concentrates the field far
# more than a sphere. A nonlinear $B$-$H$ curve turns the problem into a fixed-point
# iteration on the permeability; a time-harmonic applied field adds eddy currents and a
# skin layer at the surface.
