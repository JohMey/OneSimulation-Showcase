"""OneSimulation example: capacitance matrix of two conducting spheres.
Requires the installed, activated OneSimulation runtime.

The full 2x2 Maxwell capacitance matrix of two unequal spheres is computed from a
3D finite-element field (curved quadratic tetrahedra) and compared with the exact
bispherical-coordinate series solution, through two independent routes: field
energy and Gauss-law charge.
"""


# # Capacitance matrix of two spheres
#
# Two conducting spheres in free space, radii $a=0.1$ m and $b=0.3$ m, centres
# $c=0.5$ m apart (a gap of only $0.1$ m), carry charges that depend linearly on both
# potentials:
#
# $$\begin{pmatrix}Q_1\\Q_2\end{pmatrix}=
#   \begin{pmatrix}C_{11}&C_{12}\\C_{21}&C_{22}\end{pmatrix}
#   \begin{pmatrix}V_1\\V_2\end{pmatrix}.$$
#
# This **Maxwell capacitance matrix** is symmetric ($C_{12}=C_{21}$), its diagonal is
# positive and its off-diagonal term negative. The circuit-style **mutual capacitance
# matrix** follows from it: each sphere has a capacitance to ground (infinity)
# $C_{11}+C_{12}$ and $C_{22}+C_{21}$, and the two spheres are coupled by $-C_{12}$.
#
# Column $j$ of the matrix is obtained by holding sphere $j$ at $1$ V and the other at
# $0$ V, solving Laplace's equation $\nabla\cdot(\varepsilon_0\nabla V)=0$ in the air
# around them and measuring the charge on both spheres. The field is genuinely
# three-dimensional: it is strongly concentrated in the narrow gap and decays like
# $1/r$ far away.
#
# **Reference (exact solution).** In bispherical coordinates the problem separates and
# the capacitance coefficients are rapidly converging series (W. R. Smythe, *Static and
# Dynamic Electricity*, McGraw-Hill 1950, ch. 5; J. Lekner, "Capacitance coefficients
# of two spheres", *J. Electrostatics* 69 (2011) 11-14):
#
# $$C_{11}=4\pi\varepsilon_0\,ab\sinh u\sum_{n=0}^{\infty}\frac{1}{a\sinh nu+b\sinh(n+1)u},\qquad
#   C_{22}=4\pi\varepsilon_0\,ab\sinh u\sum_{n=0}^{\infty}\frac{1}{b\sinh nu+a\sinh(n+1)u},$$
#
# $$C_{12}=-4\pi\varepsilon_0\,\frac{ab}{c}\sinh u\sum_{n=1}^{\infty}\frac{1}{\sinh nu},\qquad
#   \cosh u=\frac{c^2-a^2-b^2}{2ab}.$$
#
# For a very distant partner $C_{11}\to4\pi\varepsilon_0 a$, the capacitance of an
# isolated sphere.

from dataclasses import dataclass
from importlib.metadata import version
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter

from onesim.cad.geometry.backends.occt.provider import OcctCadProvider
from onesim.cad.geometry.ports.kernel_port import SphereRequest
from onesim.cad.geometry.primitives import Geometry
from onesim.cad.operations.execution.model_geometry import GeometryService
from onesim.materials import VacuumPermittivity
from onesim.meshing.generation.backends.cad.in_memory_cad import build_file_free_mesher
from onesim.meshing.generation.contracts.requests import (
    HighOrderMeshControl, MeshGenerationRequest, MeshRefinement, MeshRefinementSupport,
    SpatialMeshControl,
)
from onesim.meshing.generation.operations.geometry.model_mesh import CadMeshingService
from onesim.meshing.meshes.mesh_declarations import TetrahedralMesh
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.problem_definition.values.declarations.immutable_values import FrozenMap

print("OneSimulation:", version("onesimulation"))
start_time = time.perf_counter()


# ## Parameters

a, b = 0.1, 0.3               # m, sphere radii
c = 0.5                       # m, centre-to-centre distance (gap c - a - b = 0.1 m)
x1, x2 = -c / 2, c / 2        # m, sphere centres on the x axis
outer_radius = 4.0            # m, grounded outer sphere that truncates the air domain
eps0 = float(VacuumPermittivity().evaluate())
F = 4 * np.pi * eps0          # F/m, so that F * radius is a capacitance

mesh_levels = (0.034, 0.024, 0.017)   # m, target size h on sphere 1 and across the gap
study_radii = (2.0, 3.0)              # m, extra outer radii for the truncation study
print(f"a = {a} m, b = {b} m, c = {c} m, gap = {c - a - b:.2f} m, outer radius {outer_radius} m")


# ## Reference: exact series solution

def series_capacitance(a, b, c, terms=200):
    """Maxwell capacitance matrix of two spheres from the bispherical series."""
    u = np.arccosh((c * c - a * a - b * b) / (2 * a * b))
    n = np.arange(terms)
    m = np.arange(1, terms)
    with np.errstate(over="ignore"):
        c11 = F * a * b * np.sinh(u) * np.sum(1 / (a * np.sinh(n * u) + b * np.sinh((n + 1) * u)))
        c22 = F * a * b * np.sinh(u) * np.sum(1 / (b * np.sinh(n * u) + a * np.sinh((n + 1) * u)))
        c12 = -F * a * b / c * np.sinh(u) * np.sum(1 / np.sinh(m * u))
    return np.array([[c11, c12], [c12, c22]])


C_ref = series_capacitance(a, b, c)
C_far = series_capacitance(a, b, 1000.0)
print("Exact Maxwell capacitance matrix / pF:")
print(f"  C11 = {C_ref[0, 0]*1e12:.4f}   C12 = {C_ref[0, 1]*1e12:.4f}   C22 = {C_ref[1, 1]*1e12:.4f}")
print(f"Isolated-sphere limit of the series (c = 1000 m): C11 / (4 pi eps0 a) = {C_far[0, 0]/(F*a):.6f}, "
      f"C22 / (4 pi eps0 b) = {C_far[1, 1]/(F*b):.6f}")


# ## Geometry and mesh
#
# The arrangement is rotationally symmetric about the line of centres (the $x$ axis),
# so the planes $y=0$ and $z=0$ are symmetry planes on which the normal electric
# displacement vanishes. The air domain is therefore one quarter of a large sphere of
# radius $R$ with the two conducting spheres removed; it is built with boolean
# operations in OneSimulation's CAD kernel and meshed with **curved quadratic
# tetrahedra** (edges projected onto the exact spherical surfaces). The mesh is
# refined on both spheres, in a ball that contains the gap and, more mildly, in the
# near field; it coarsens towards the outer sphere. All sizes scale with one level
# parameter $h$ (cell size on the small sphere and across the gap).


@dataclass(frozen=True, eq=False, kw_only=True)
class QuarterAirDomain(Geometry):
    """Quarter of a grounded outer sphere with conducting spheres removed."""

    sphere_count: int = 2
    dimension = 3

    @property
    def volume(self):
        return self.domain()

    def boundary_groups(self):
        names = tuple(f"sphere_{i + 1}" for i in range(self.sphere_count)) + (
            "outer", "plane_y", "plane_z")
        return {**{name: (name,) for name in names}, "all": names}


def quarter_mesh(spheres, radius, h):
    """Build the quarter air domain around `spheres` [(centre_x, r), ...] and mesh it."""
    kernel = OcctCadProvider()
    cad = GeometryService("air", kernel)
    outer = cad.construct(kernel.create_sphere, SphereRequest(cad.reference("outer"), radius))
    box = cad.place(cad.box(width=2 * radius, height=radius, depth=radius, name="box"),
                    translation=(-radius, 0.0, 0.0), name="box_placed")
    air = cad.boolean("intersection", outer, box, name="quarter")
    for i, (centre, r) in enumerate(spheres):
        ball = cad.construct(kernel.create_sphere, SphereRequest(cad.reference(f"s{i}"), r))
        ball = cad.place(ball, translation=(centre, 0.0, 0.0), name=f"s{i}_placed")
        air = cad.boolean("difference", air, ball, name=f"air_{i}")
    # Name every CAD face from its exact measure: a quarter sphere has area pi r^2,
    # the two symmetry planes are recognised by their normals.
    curved = {f"sphere_{i + 1}": np.pi * r * r for i, (_, r) in enumerate(spheres)}
    curved["outer"] = np.pi * radius * radius
    faces = {name: [] for name in (*curved, "plane_y", "plane_z")}
    for face in kernel.structured_meshing_topology(air.reference).faces:
        if face.surface_type == "plane":
            faces["plane_y" if abs(face.normal[1]) > 0.9 else "plane_z"].append(face.entity_id)
        else:
            name = min(curved, key=lambda k: abs(curved[k] - face.area) / curved[k])
            faces[name].append(face.entity_id)
    body = next(e.entity_id for e in air.data.entities if e.kind == "body")
    controls = [MeshRefinement(faces[f"sphere_{i + 1}"][0], h * min(r / a, 2.0), "surface")
                for i, (_, r) in enumerate(spheres)]
    if len(spheres) == 2:
        gap_centre = 0.5 * ((x1 + a) + (x2 - b))
        controls.append(MeshRefinement(body, h, "volume",
                                       MeshRefinementSupport("ball", (gap_centre, 0.0, 0.0), 0.12)))
    controls.append(MeshRefinement(body, 6 * h, "volume",
                                   MeshRefinementSupport("ball", (0.0, 0.0, 0.0), 0.6)))
    request = MeshGenerationRequest(
        air.reference.artifact_id, air.reference.version,
        global_size=0.15 * radius, refinements=tuple(controls),
        max_candidate_cells=1_000_000,
        high_order=HighOrderMeshControl(polynomial_order=2),
        spatial=SpatialMeshControl("free_tetrahedral", body, engine="cad_native"),
    )
    native = CadMeshingService(provider_factory=build_file_free_mesher).generate(air, request)
    domain = QuarterAirDomain(sphere_count=len(spheres))
    mesh = TetrahedralMesh(domain, native, FrozenMap({k: tuple(v) for k, v in faces.items()}), ())
    return domain, mesh


# Isoparametric quadrature on the curved 10-node tetrahedra: a collapsed (Duffy)
# 3x3x3 Gauss rule, mapped with the quadratic geometry so that the energy integral
# sees the exact curved cells, not their straight-sided chords.
_g, _w = np.polynomial.legendre.leggauss(3)
_g, _w = 0.5 * (_g + 1), 0.5 * _w
_U, _V, _W = (m.ravel() for m in np.meshgrid(_g, _g, _g, indexing="ij"))
_WU, _WV, _WW = (m.ravel() for m in np.meshgrid(_w, _w, _w, indexing="ij"))
_bary = np.column_stack((1 - _U, _U * (1 - _V), _U * _V * (1 - _W), _U * _V * _W))
_ref_weights = _WU * _WV * _WW * _U**2 * _V
_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_dbary = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
_shape = np.column_stack([_bary * (2 * _bary - 1)] + [4 * _bary[:, i] * _bary[:, j] for i, j in _edges])
_dshape = np.zeros((len(_ref_weights), 10, 3))
for k in range(4):
    _dshape[:, k] = (4 * _bary[:, k] - 1)[:, None] * _dbary[k]
for m, (i, j) in enumerate(_edges):
    _dshape[:, 4 + m] = 4 * (_bary[:, i][:, None] * _dbary[j] + _bary[:, j][:, None] * _dbary[i])


def quadrature(mesh):
    nodes = np.asarray(mesh.coordinates)[np.asarray(mesh.cells)]          # (cells, 10, 3)
    points = np.einsum("qn,cnd->cqd", _shape, nodes).reshape(-1, 3)
    jacobians = np.einsum("qnr,cnd->cqdr", _dshape, nodes)
    weights = (np.abs(np.linalg.det(jacobians)) * _ref_weights).ravel()
    return points, weights


# ## Model
#
# **Numerical method.**
# - Spatial method: continuous Galerkin finite elements, isoparametric quadratic Lagrange
#   elements (P2) on curved 10-node tetrahedra whose geometry is also quadratic, with a
#   collapsed Gauss rule of order 5 (125 points per cell) for the curved-cell integrals.
# - Time integration: none (stationary electrostatics).
# - Solver: one linear system per excitation, solved with a sparse direct factorisation.
#
# Boundary conditions on every face of the quarter domain: fixed potential on both
# spheres, zero potential on the outer sphere, zero normal electric displacement on the
# two symmetry planes.

def solve(domain, mesh, potentials):
    """Solve Laplace's equation with the given sphere potentials (outer sphere grounded)."""
    model = PoissonEquation(domain=domain.volume, permittivity=VacuumPermittivity())
    conditions = [FixedPotential(boundary=getattr(domain.boundaries, f"sphere_{i + 1}"),
                                 physics=model, value=v) for i, v in enumerate(potentials)]
    conditions += [
        FixedPotential(boundary=domain.boundaries.outer, physics=model, value=0.0),
        ZeroNormalElectricDisplacement(
            boundary=domain.boundaries.plane_y | domain.boundaries.plane_z, physics=model),
    ]
    problem = Problem(name="Two-sphere capacitance", geometry=domain, physics=(model,),
                      boundary_conditions=tuple(conditions))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2), quadrature_order=5),
                        linear_solver=SparseDirect())
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    return result.field(model.electric_field), result.field(model.potential)


def shell_charge(points, weights, field_values, centre, r_in, r_out):
    """Charge on a sphere from Gauss's law, averaged over a shell r_in < r < r_out.

    Q = -eps0 * integral(E . grad eta) with eta = 1 on the sphere and 0 beyond the
    shell (smooth cubic ramp). This is Gauss's law weighted over many closed surfaces;
    the factor 4 restores the full sphere from the quarter domain.
    """
    offset = points - np.array([centre, 0.0, 0.0])
    r = np.linalg.norm(offset, axis=1)
    s = np.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    d_eta = -6.0 * s * (1.0 - s) / (r_out - r_in)
    radial = np.einsum("ij,ij->i", field_values, offset) / r
    return -4.0 * eps0 * np.sum(weights * radial * d_eta)


def capacitance_matrices(domain, mesh):
    """Maxwell matrix from (i) mutual field energy and (ii) Gauss-law charges."""
    points, weights = quadrature(mesh)
    fields = [solve(domain, mesh, (1.0, 0.0)), solve(domain, mesh, (0.0, 1.0))]
    E = [np.asarray(f[0].evaluate(points)) for f in fields]
    # C_ij = eps0 * integral(grad V_i . grad V_j) (factor 4 for the quarter domain);
    # the diagonal is twice the stored energy at 1 V, the off-diagonal the mutual energy.
    C_energy = np.array([[4 * eps0 * np.sum(weights * np.einsum("ij,ij->i", E[i], E[j]))
                          for j in range(2)] for i in range(2)])
    shells = ((x1, a, c - b - 0.01), (x2, b, c - a - 0.01))   # stay clear of the other sphere
    C_charge = np.array([[shell_charge(points, weights, E[j], *shells[i])
                          for j in range(2)] for i in range(2)])
    return C_energy, C_charge, fields


def free_space(C, radius, centres=(x1, x2)):
    """Remove the grounded outer sphere: Kelvin-image correction of the potential coefficients.

    For charges near the centre of a grounded sphere of radius R, the image charge adds
    -1/(4 pi eps0 R) - (x.x')/(4 pi eps0 R^3) + ... to every potential coefficient.
    Adding these terms back to P = C^-1 gives the free-space matrix to O(R^-5).
    """
    d = np.asarray(centres, float)
    P = np.linalg.inv(C) + 1.0 / (F * radius) + np.outer(d, d) / (F * radius**3)
    return np.linalg.inv(P)


# ## Solve
#
# **Isolated-sphere check.** A single sphere of radius $a$ at the centre of the grounded
# outer sphere has the exact capacitance $4\pi\varepsilon_0\,aR/(R-a)$ (a spherical
# capacitor); after the image correction this is the isolated-sphere value
# $4\pi\varepsilon_0 a$.

domain_1, mesh_1 = quarter_mesh(((0.0, a),), outer_radius, mesh_levels[0])
points_1, weights_1 = quadrature(mesh_1)
E_single = np.asarray(solve(domain_1, mesh_1, (1.0,))[0].evaluate(points_1))
C_single_energy = 4 * eps0 * np.sum(weights_1 * np.einsum("ij,ij->i", E_single, E_single))
C_single_charge = shell_charge(points_1, weights_1, E_single, 0.0, a, 2 * a)
C_single_exact = F * a * outer_radius / (outer_radius - a)
print(f"Single sphere ({mesh_1.cell_count} curved tetrahedra): energy {C_single_energy*1e12:.4f} pF, "
      f"charge {C_single_charge*1e12:.4f} pF, exact spherical capacitor {C_single_exact*1e12:.4f} pF")

# **Mesh study** at $R=4$ m: three meshes, each solved twice (one sphere at 1 V).

study = []
for h in mesh_levels:
    domain, mesh = quarter_mesh(((x1, a), (x2, b)), outer_radius, h)
    C_energy, C_charge, fields = capacitance_matrices(domain, mesh)
    study.append(dict(h=h, cells=mesh.cell_count, nodes=len(mesh.coordinates),
                      energy=free_space(C_energy, outer_radius),
                      charge=free_space(C_charge, outer_radius), raw=C_energy))
    print(f"h = {h:.3f} m: {mesh.cell_count:6d} curved tetrahedra, {len(mesh.coordinates):6d} quadratic nodes")
final_fields = fields          # finest mesh, used in the figure

# **Outer-radius study** on the coarsest mesh level: the raw (truncated-domain) matrix
# changes like $1/R$, the image-corrected matrix must not.

radius_study = []
for radius in study_radii + (outer_radius,):
    if radius == outer_radius:
        C_raw = study[0]["raw"]
    else:
        domain, mesh = quarter_mesh(((x1, a), (x2, b)), radius, mesh_levels[0])
        C_raw = capacitance_matrices(domain, mesh)[0]
    radius_study.append((radius, C_raw, free_space(C_raw, radius)))


# ## Comparison with the reference
#
# Reference: the exact bispherical series (Smythe 1950; Lekner 2011) for the same radii
# and separation. The finite-element matrices are shown after the image correction of
# the grounded outer sphere. Two independent routes are compared: the field energy
# (by Dirichlet's principle the conforming finite-element energy approaches the exact
# diagonal terms from above) and the charges from Gauss's law, whose off-diagonal
# terms $C_{12}$ and $C_{21}$ come from two different solves and therefore also test
# reciprocity.


def rel(value, reference):
    return 100.0 * (value - reference) / abs(reference)


entries = (("C11", 0, 0), ("C22", 1, 1), ("C12", 0, 1), ("C21", 1, 0))
print("\nMesh study (energy route | charge route), values in pF, deviation from the series in %:")
for row in study:
    text = "  ".join(f"{name} {row['energy'][i, j]*1e12:8.4f} ({rel(row['energy'][i, j], C_ref[i, j]):+.3f}) | "
                     f"{row['charge'][i, j]*1e12:8.4f} ({rel(row['charge'][i, j], C_ref[i, j]):+.3f})"
                     for name, i, j in entries[:3])
    print(f"  h = {row['h']:.3f} m, {row['cells']:6d} cells:  {text}")
fine, medium = study[-1], study[-2]
print("Change between the last two meshes (energy route): "
      + ", ".join(f"{name} {rel(fine['energy'][i, j], medium['energy'][i, j]):+.3f} %" for name, i, j in entries[:3]))
print("Change between the last two meshes (charge route): "
      + ", ".join(f"{name} {rel(fine['charge'][i, j], medium['charge'][i, j]):+.3f} %" for name, i, j in entries[:3]))

C_fem = fine["energy"]
C_q = fine["charge"]
print("\nFinest mesh against the exact series:")
print(f"{'':>6} {'series':>10} {'energy':>10} {'dev %':>8} {'charge':>10} {'dev %':>8}")
for name, i, j in entries:
    print(f"{name:>6} {C_ref[i, j]*1e12:10.4f} {C_fem[i, j]*1e12:10.4f} {rel(C_fem[i, j], C_ref[i, j]):+8.3f} "
          f"{C_q[i, j]*1e12:10.4f} {rel(C_q[i, j], C_ref[i, j]):+8.3f}")
print(f"Reciprocity from the independent charge columns: C12 = {C_q[0, 1]*1e12:.4f} pF, "
      f"C21 = {C_q[1, 0]*1e12:.4f} pF ({rel(C_q[0, 1], C_q[1, 0]):+.3f} %)")

mutual_ref = np.array([C_ref[0, 0] + C_ref[0, 1], C_ref[1, 1] + C_ref[1, 0], -C_ref[0, 1]])
mutual_fem = np.array([C_fem[0, 0] + C_fem[0, 1], C_fem[1, 1] + C_fem[1, 0], -C_fem[0, 1]])
print("\nMutual capacitances (energy route): "
      + ", ".join(f"{name} {v*1e12:.4f} pF (series {r*1e12:.4f}, {rel(v, r):+.3f} %)"
                  for name, v, r in zip(("sphere 1 to ground", "sphere 2 to ground", "sphere 1 to sphere 2"),
                                        mutual_fem, mutual_ref)))
print(f"Stored energy with both spheres at 1 V: {0.5*np.sum(C_fem)*1e12:.4f} pJ "
      f"(series {0.5*np.sum(C_ref)*1e12:.4f} pJ)")

print(f"\nIsolated sphere: energy route {rel(C_single_energy, C_single_exact):+.3f} %, "
      f"charge route {rel(C_single_charge, C_single_exact):+.3f} % against 4 pi eps0 aR/(R-a); "
      f"image-corrected {free_space(np.array([[C_single_energy]]), outer_radius, (0.0,))[0, 0]*1e12:.4f} pF "
      f"vs 4 pi eps0 a = {F*a*1e12:.4f} pF")

print("\nOuter-radius study (coarsest mesh level, energy route), C11 / C22 / C12 in pF:")
for radius, C_raw, C_corr in radius_study:
    print(f"  R = {radius:.0f} m: raw {C_raw[0, 0]*1e12:8.4f} {C_raw[1, 1]*1e12:8.4f} {C_raw[0, 1]*1e12:8.4f}   "
          f"image-corrected {C_corr[0, 0]*1e12:8.4f} {C_corr[1, 1]*1e12:8.4f} {C_corr[0, 1]*1e12:8.4f}")
print(f"Total run time: {time.perf_counter() - start_time:.0f} s")


# ## Figure

E_field, V_field = final_fields[0]           # sphere 1 at 1 V, sphere 2 grounded
nx, ny = 361, 181
gx = np.linspace(-0.55, 0.65, nx)
gy = np.linspace(0.0, 0.6, ny)
GX, GY = np.meshgrid(gx, gy)
inside = ((np.hypot(GX - x1, GY) > a * 1.002) & (np.hypot(GX - x2, GY) > b * 1.002))
grid = np.column_stack((GX[inside], GY[inside], np.full(inside.sum(), 1e-9)))
V_map = np.full(GX.shape, np.nan)
E_map = np.full(GX.shape, np.nan)
V_map[inside] = np.asarray(V_field.evaluate(grid)).ravel()
E_map[inside] = np.linalg.norm(np.asarray(E_field.evaluate(grid)), axis=1)
# The plane z = 0 is a symmetry plane: mirror the upper half to show the full section.
X_full = np.vstack((GX[::-1], GX[1:]))
Y_full = np.vstack((-GY[::-1], GY[1:]))
V_full = np.vstack((V_map[::-1], V_map[1:]))
E_full = np.vstack((E_map[::-1], E_map[1:]))

cream = "#fbf5eb"
fig = plt.figure(figsize=(14, 7.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
grid_spec = fig.add_gridspec(2, 2, width_ratios=(1.3, 1))
ax_map = fig.add_subplot(grid_spec[:, 0])
ax_mesh = fig.add_subplot(grid_spec[0, 1])
ax_radius = fig.add_subplot(grid_spec[1, 1])
for ax in (ax_map, ax_mesh, ax_radius):
    ax.set_facecolor("#fffaf2")

# Logarithmic colour scale: the field spans two decades between the gap and the far side.
image = ax_map.pcolormesh(X_full, Y_full, E_full, cmap="magma", shading="gouraud",
                          norm=LogNorm(vmin=np.nanpercentile(E_full, 1), vmax=np.nanmax(E_full)))
ax_map.contour(X_full, Y_full, V_full, levels=np.linspace(0.05, 0.95, 19),
               colors="white", linewidths=0.7, alpha=0.9)
for centre, r, label in ((x1, a, "1 V"), (x2, b, "0 V")):
    ax_map.add_patch(plt.Circle((centre, 0), r, fc="#c9b89a", ec="#4a3b2a", lw=1.2))
    ax_map.text(centre, 0, label, ha="center", va="center", fontsize=10, color="#2b2118")
ax_map.set_aspect("equal")
ax_map.set(xlim=(gx[0], gx[-1]), ylim=(-gy[-1], gy[-1]), xlabel="x / m", ylabel="y / m",
           title="Section z = 0 of the 3D field (sphere 1 at 1 V):\n"
                 "|E| (colour) and equipotentials 0.05 ... 0.95 V")
fig.colorbar(image, ax=ax_map, label="|E| / (V/m)", shrink=0.8)

cells = np.array([row["cells"] for row in study])
colors = {"C11": "#A85539", "C22": "#214D43", "C12": "#3887c5"}
for name, i, j in entries[:3]:
    ax_mesh.loglog(cells, [abs(rel(row["energy"][i, j], C_ref[i, j])) for row in study], "o-",
                   color=colors[name], label=f"{name}, field energy")
    ax_mesh.loglog(cells, [abs(rel(row["charge"][i, j], C_ref[i, j])) for row in study], "s--",
                   color=colors[name], mfc="none", label=f"{name}, Gauss-law charge")
ax_mesh.set_xticks(cells, [f"{n}" for n in cells])
ax_mesh.xaxis.set_minor_formatter(NullFormatter())
ax_mesh.grid(alpha=0.25, which="both")
ax_mesh.set(xlabel="Curved quadratic tetrahedra (quarter domain)",
            ylabel="|deviation from exact series| / %", title="Mesh convergence at R = 4 m")
ax_mesh.legend(fontsize=7.5, ncol=2)

radii = np.array([item[0] for item in radius_study])
for name, i, j in entries[:3]:
    ax_radius.plot(1 / radii, [rel(item[1][i, j], C_ref[i, j]) for item in radius_study], "o:",
                   color=colors[name], label=f"{name}, truncated domain")
    ax_radius.plot(1 / radii, [rel(item[2][i, j], C_ref[i, j]) for item in radius_study], "o-",
                   color=colors[name], label=f"{name}, image-corrected")
ax_radius.set_yscale("symlog", linthresh=0.2)
ax_radius.axhline(0.0, color="#777", lw=1)
ax_radius.grid(alpha=0.25)
ax_radius.set(xlabel="1 / R  (outer radius R in m)", ylabel="Deviation from exact series / %",
              title="Outer-boundary study (coarsest mesh)")
ax_radius.legend(fontsize=7.5, ncol=2)
plt.show()


# ## Going further
#
# Move the spheres closer together: the gap field and the coupling $-C_{12}$ grow
# rapidly and the series needs more terms, while the mesh needs a finer ball in the
# gap. Give the spheres equal radii to recover the symmetric case, or place a
# dielectric between them by meshing an additional region. With more than two
# conductors the same recipe (one unit-potential solve per conductor) gives the full
# $N\times N$ matrix.
