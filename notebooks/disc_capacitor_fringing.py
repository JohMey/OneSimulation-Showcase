"""OneSimulation example: capacitance of a disc capacitor with fringing fields (3D electrostatics).
Requires the installed, activated OneSimulation runtime.

Two metal discs with leads, separated by a quartz-glass disc and surrounded by air, are solved in 3D.
The capacitance follows from the stored field energy and from the electrode charge. The disc pair
without dielectric and leads is graded against the exact solution of Love's integral equation
(the accurate value for the circular-plate capacitor with fringing) and against Kirchhoff's formula;
the device is checked by a three-mesh convergence study and an air-domain study.
"""


# # Capacitance of a disc capacitor with fringing fields
#
# A capacitor in its simplest form: two metal discs of radius $a=10$ cm and thickness $5$ mm,
# separated by a $d=3$ cm thick disc of quartz glass ($\varepsilon_r=4.2$) of the same radius. Each
# disc carries a lead (radius $7.5$ mm) that runs along the axis to the boundary of a cylindrical air
# region of radius and height $20$ cm. The upper electrode is held at $V_0=1$ V, the lower one is
# grounded, and the outer boundary of the air carries no charge ($\mathbf D\cdot\mathbf n=0$).
#
# The field is not confined to the dielectric: it bulges out at the rim of the discs into the
# surrounding air (fringing), and the field at the sharp metal edges is singular. Both raise the
# capacitance above the ideal parallel-plate value $\varepsilon_0\varepsilon_r A/d$. OneSimulation
# solves $-\nabla\cdot(\varepsilon\nabla V)=0$ with second-order Lagrange elements on a graded
# tetrahedral mesh, and the capacitance is extracted in two independent ways:
#
# $$C_W=\frac{2W}{V_0^2},\quad W=\frac12\int_\Omega\varepsilon|\mathbf E|^2\,\mathrm dV,\qquad\qquad
#   C_Q=\frac{Q}{V_0},\quad Q=-\int_{z=\mathrm{mid}}\varepsilon E_z\,\mathrm dA .$$
#
# **Numerical method.** Spatial: continuous Galerkin finite elements, Lagrange P2 on straight-sided
# tetrahedra, cell-wise constant permittivity, element integrals with a 27-point collapsed Gauss rule
# (exact for the P2 stiffness). Time integration: none, the problem is stationary. Solver: sparse
# direct LU factorisation. Mesh: graded unstructured tetrahedra from the OCC/Netgen kernels, passed to
# OneSimulation as an unstructured simplex mesh. Reference for the benchmark: Nystrom solution of
# Love's integral equation with Gauss-Legendre quadrature.
#
# **References.** There is no closed form for the complete device, so it is graded by a mesh
# convergence study with a Richardson estimate, by the agreement of the energy and charge routes, and
# by an air-domain study. The fringing physics itself is graded on the classical benchmark of the
# same disc pair (same radius and gap, $d/a=0.3$) in free space: two thin coaxial discs at
# potentials $\pm V_0/2$. Its capacitance is known accurately from Love's integral equation,
#
# $$f(x)-\frac1\pi\int_{-1}^{1}\frac{\kappa\,f(t)}{\kappa^2+(x-t)^2}\,\mathrm dt=1,\qquad
#   C=2\varepsilon_0 a\int_{-1}^{1}f(t)\,\mathrm dt,\qquad \kappa=d/a,$$
#
# (E. R. Love, Q. J. Mech. Appl. Math. 2, 428, 1949; solved to high accuracy by M. Norgren and
# B. L. G. Jonsson, PIER 97, 357, 2009), and asymptotically by Kirchhoff's formula (G. Kirchhoff,
# Monatsber. Akad. Wiss. Berlin 1877, 144)
#
# $$C_\mathrm{K}=\frac{\varepsilon_0\pi a^2}{d}+\varepsilon_0 a\left[\ln\frac{16\pi a}{d}-1\right],$$
#
# which keeps only the leading terms of an asymptotic expansion for small $\kappa$; its relative error
# is of order $\kappa^2\ln^2\kappa$, a few per cent at $\kappa=0.3$. Love's equation is solved below by Gauss-Legendre
# Nystrom discretisation, which converges to all printed digits.

from dataclasses import dataclass
from importlib.metadata import version
import time
import numpy as np
import matplotlib.pyplot as plt
import netgen.occ as occ
from scipy.spatial import cKDTree

from onesim.cad.d3 import Cylinder
from onesim.materials import VacuumPermittivity
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters

a = 0.10                     # m, disc radius
t_plate = 0.005              # m, metal disc thickness
d = 0.03                     # m, dielectric thickness = electrode gap
r_lead = 0.0075              # m, lead radius
R_air, H_air = 0.20, 0.20    # m, air cylinder of the device
eps_r = 4.2                  # quartz glass
V0 = 1.0                     # V, terminal potential
z1 = 0.08                    # m, bottom of the lower disc
z2, z3, z4 = z1 + t_plate, z1 + t_plate + d, z1 + 2 * t_plate + d
eps0 = float(VacuumPermittivity().evaluate())
TOL = 1e-6

# Mesh levels (global size, size in the dielectric / on the discs, size along the disc rims), each
# finer by a factor sqrt(2).
device_levels = ((0.040, 0.0080, 0.0040), (0.0283, 0.00566, 0.00283), (0.020, 0.0040, 0.0020))
benchmark_levels = ((0.0424, 0.0113, 0.00283), (0.030, 0.0080, 0.0020), (0.0212, 0.00566, 0.00141))
R_sphere = 0.30              # m, air sphere around the benchmark disc pair (3 radii)

C_ideal_device = eps0 * eps_r * np.pi * a**2 / d
C_ideal_air = eps0 * np.pi * a**2 / d
kappa = d / a


def love_capacitance(kappa, n=400):
    x, w = np.polynomial.legendre.leggauss(n)
    kernel = kappa / np.pi / (kappa**2 + (x[:, None] - x[None, :]) ** 2)
    f = np.linalg.solve(np.eye(n) - kernel * w[None, :], np.ones(n))
    return 2 * eps0 * a * np.sum(w * f)


C_love = love_capacitance(kappa)
C_kirchhoff = C_ideal_air + eps0 * a * (np.log(16 * np.pi / kappa) - 1)
print(f"Love's equation, kappa = {kappa}: C = {C_love*1e12:.5f} pF (n = 400) / {love_capacitance(kappa, 800)*1e12:.5f} pF (n = 800)")
print(f"Kirchhoff's formula:            C = {C_kirchhoff*1e12:.5f} pF ({100*(C_kirchhoff/C_love-1):+.2f} % from Love)")
print(f"ideal plates eps0*A/d:          C = {C_ideal_air*1e12:.5f} pF ({100*(C_ideal_air/C_love-1):+.2f} % from Love)")
print(f"device, ideal eps0*eps_r*A/d:   C = {C_ideal_device*1e12:.5f} pF")


# ## Geometry and mesh
#
# The conductors are not part of the computational domain: the air and the dielectric are what is
# meshed, and the metal surfaces are boundaries that carry the electrode potentials. The solids are
# built with the OCC kernel and meshed with Netgen - the same CAD and meshing kernels OneSimulation's
# own tetrahedral mesher runs on - because the electrodes need their own named boundary groups; the
# mesh is handed to OneSimulation as an `UnstructuredSimplexMesh` together with a small geometry
# declaration that names these groups. The mesh is graded: fine along the four disc rims (the
# singular edges), medium in the dielectric gap, coarse far out in the air.
#
# The arrangement is rotationally symmetric, so one quarter is meshed; the two cut planes carry the
# symmetry condition $\mathbf D\cdot\mathbf n=0$, which is exact here. All energies and charges are
# multiplied by four. The benchmark disc pair consists of two zero-thickness discs embedded in a
# quarter air sphere; each disc is opened into a two-sided slit by duplicating its interior nodes for
# the cells above it, so that both faces become electrode boundaries.

@dataclass(frozen=True, eq=False, kw_only=True)
class QuarterDomain(Cylinder):
    """Quarter-domain declaration with explicitly named boundary groups."""

    def boundary_groups(self):
        names = ("outer", "symmetry", "terminal", "ground")
        return {**{name: (name,) for name in names}, "all": names}


def netgen_arrays(mesh):
    X = np.array(mesh.Coordinates())
    T = np.array([[v.nr for v in element.vertices] for element in mesh.Elements3D()]) - 1
    return X, T


def device_mesh(R, size, size_gap, size_edge):
    quadrant = occ.Box(occ.Pnt(0, 0, -1), occ.Pnt(1, 1, 1))
    air = occ.Cylinder(occ.Pnt(0, 0, 0), occ.Z, r=R, h=H_air)
    ground = occ.Cylinder(occ.Pnt(0, 0, z1), occ.Z, r=a, h=t_plate) + occ.Cylinder(occ.Pnt(0, 0, 0), occ.Z, r=r_lead, h=z1)
    terminal = (occ.Cylinder(occ.Pnt(0, 0, z3), occ.Z, r=a, h=t_plate)
                + occ.Cylinder(occ.Pnt(0, 0, z4), occ.Z, r=r_lead, h=H_air - z4))
    dielectric = occ.Cylinder(occ.Pnt(0, 0, z2), occ.Z, r=a, h=d) * quadrant
    gas = (air - ground - terminal) * quadrant - dielectric
    gas.mat("air")
    dielectric.mat("dielectric")
    dielectric.maxh = size_gap
    shape = occ.Glue([gas, dielectric])
    for face in shape.faces:
        lo, hi = face.bounding_box
        if hi.x - lo.x < TOL or hi.y - lo.y < TOL:
            face.name = "symmetry"
        elif abs(hi.x - R) < TOL or (hi.z - lo.z < TOL and (hi.z < TOL or lo.z > H_air - TOL)):
            face.name = "outer"
        elif abs(hi.x - a) < TOL and abs(lo.z - z2) < TOL and abs(hi.z - z3) < TOL:
            face.name = "interface"          # dielectric/air, interior
        elif hi.z < H_air / 2:
            face.name = "ground"
        else:
            face.name = "terminal"
    for edge in shape.edges:
        lo, hi = edge.bounding_box
        if abs(hi.x - a) < TOL and abs(hi.y - a) < TOL and hi.z - lo.z < TOL:
            edge.maxh = size_edge              # the four quarter-circle disc rims
    mesh = occ.OCCGeometry(shape).GenerateMesh(maxh=size, grading=0.25)
    X, T = netgen_arrays(mesh)
    in_dielectric = np.array([mesh.GetMaterial(e.index) == "dielectric" for e in mesh.Elements3D()])
    facets = {name: [] for name in ("outer", "symmetry", "terminal", "ground")}
    for element in mesh.Elements2D():
        name = mesh.GetBCName(element.index - 1)
        if name in facets:
            facets[name].append([v.nr - 1 for v in element.vertices])
    return X, T, in_dielectric, facets


def exterior_faces(T):
    F = np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]])
    _, first, count = np.unique(np.sort(F, axis=1), axis=0, return_index=True, return_counts=True)
    return F[first[count == 1]]


def disc_pair_mesh(R, size, size_disc, size_edge):
    quadrant = occ.Box(occ.Pnt(0, 0, -1), occ.Pnt(1, 1, 1))
    air = occ.Sphere(occ.Pnt(0, 0, 0), R) * quadrant
    discs = [occ.WorkPlane(occ.Axes((0, 0, z), n=occ.Z, h=occ.X)).Circle(a).Face() * quadrant for z in (d / 2, -d / 2)]
    shape = occ.Glue([air] + discs)
    for face in shape.faces:
        lo, hi = face.bounding_box
        if hi.x < a + TOL and hi.z - lo.z < TOL:
            face.maxh = size_disc
    for edge in shape.edges:
        lo, hi = edge.bounding_box
        if abs(hi.x - a) < TOL and abs(hi.y - a) < TOL and hi.z - lo.z < TOL:
            edge.maxh = size_edge
    X, T = netgen_arrays(occ.OCCGeometry(shape).GenerateMesh(maxh=size, grading=0.25))
    centroid_z = X[T][:, :, 2].mean(axis=1)
    blocks, n = [X], len(X)
    for z_disc in (d / 2, -d / 2):          # open each disc into a two-sided slit
        on_disc = np.where((np.abs(X[:, 2] - z_disc) < TOL) & (np.hypot(X[:, 0], X[:, 1]) < a * (1 - 1e-9)))[0]
        copy = np.full(n, -1)
        copy[on_disc] = n + np.arange(len(on_disc))
        n += len(on_disc)
        blocks.append(X[on_disc])
        above = centroid_z > z_disc
        cells = T[above]
        T[above] = np.where(copy[cells] >= 0, copy[cells], cells)
    X = np.vstack(blocks)
    faces = exterior_faces(T)
    centres = X[faces].mean(axis=1)
    radius = np.hypot(centres[:, 0], centres[:, 1])
    facets = {name: [] for name in ("outer", "symmetry", "terminal", "ground")}
    for face, c, r in zip(faces, centres, radius):
        if abs(c[0]) < TOL or abs(c[1]) < TOL:
            facets["symmetry"].append(face)
        elif r < a and abs(c[2] - d / 2) < TOL:
            facets["terminal"].append(face)
        elif r < a and abs(c[2] + d / 2) < TOL:
            facets["ground"].append(face)
        else:
            facets["outer"].append(face)
    return X, T, facets


# ## Model
#
# The permittivity is piecewise constant per cell (air or quartz, taken from the CAD solid each
# tetrahedron belongs to). Boundary conditions on every face: terminal $V=V_0$, ground $V=0$, the
# outer air boundary $\mathbf D\cdot\mathbf n=0$ (device; benchmark: either $\mathbf D\cdot\mathbf n=0$ or
# $V=V_0/2$, see below), symmetry planes $\mathbf D\cdot\mathbf n=0$.

class CellwisePermittivity:
    """Permittivity of the cell that contains the evaluation point."""

    unit = "F/m"

    def __init__(self, X, T, values):
        corners = X[T]
        self.values, self.origin = values, corners[:, 0]
        self.tree = cKDTree(corners.mean(axis=1))
        self.inverse = np.linalg.inv(np.stack([corners[:, k] - corners[:, 0] for k in (1, 2, 3)], axis=2))

    def evaluate(self, **state):
        p = np.array([state["x"], state["y"], state["z"]])
        _, near = self.tree.query(p, k=16)
        lam = np.einsum("kij,kj->ki", self.inverse[near], p - self.origin[near])
        bary = np.column_stack((1 - lam.sum(axis=1), lam))
        return float(self.values[near[np.argmax(bary.min(axis=1))]])


GAUSS4 = np.full((4, 4), 0.1381966011250105) + np.eye(4) * (0.5854101966249685 - 0.1381966011250105)


def conical_rule(n):
    """Collapsed Gauss rule on the reference tetrahedron (barycentric points, weights summing to 1)."""
    g, w = np.polynomial.legendre.leggauss(n)
    g, w = (g + 1) / 2, w / 2
    u, v, s = (axis.ravel() for axis in np.meshgrid(g, g, g, indexing="ij"))
    weight = np.prod([axis.ravel() for axis in np.meshgrid(w, w, w, indexing="ij")], axis=0)
    local = np.column_stack((u, (1 - u) * v, (1 - u) * (1 - v) * s))
    return np.column_stack((1 - local.sum(axis=1), local)), 6 * weight * (1 - u) ** 2 * (1 - v)


BAND_BARY, BAND_WEIGHTS = conical_rule(4)


def solve(X, T, facets, eps_cell, R, length, origin, outer_potential=None):
    geometry = QuarterDomain(radius=R, length=length, origin=origin)
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=X, cells=T, boundary_facets=facets)
    model = PoissonEquation(domain=geometry.volume, permittivity=CellwisePermittivity(X, T, eps_cell),
                            charge_density=0.0)
    outer = (ZeroNormalElectricDisplacement(boundary=geometry.boundaries.outer, physics=model)
             if outer_potential is None else
             FixedPotential(boundary=geometry.boundaries.outer, physics=model, value=outer_potential))
    problem = Problem(
        name="Disc capacitor", geometry=geometry, physics=(model,),
        boundary_conditions=(
            FixedPotential(boundary=geometry.boundaries.terminal, physics=model, value=V0),
            FixedPotential(boundary=geometry.boundaries.ground, physics=model, value=0.0),
            ZeroNormalElectricDisplacement(boundary=geometry.boundaries.symmetry, physics=model),
            outer,
        ),
    )
    result = Simulation(
        problem=problem, mesh=mesh,
        numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2), quadrature_order=3),
                          linear_solver=SparseDirect()),
    ).solve_stationary()
    E = result.field(model.electric_field)
    corners = X[T]
    volume = np.abs(np.einsum("ni,ni->n", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                              corners[:, 3] - corners[:, 0])) / 6.0
    # E is linear on each tetrahedron: the 4-point rule integrates |E|^2 exactly.
    E_gauss = np.asarray(E.evaluate(np.einsum("qv,nvd->nqd", GAUSS4, corners).reshape(-1, 3))).reshape(-1, 4, 3)
    energy = 4 * 0.5 * np.sum(eps_cell * volume * (E_gauss**2).sum(axis=2).mean(axis=1))
    return E, result.field(model.potential), energy, volume


def band_charge(E, X, T, eps_cell, volume, z_mid, half_width):
    """Charge from the flux of D through the planes between the electrodes, averaged over a band.

    With chi(z) a smooth step across the band, Q = -int eps E_z chi'(z) dV; for the exact field this
    equals the flux through every single plane between the electrodes (div D = 0, D.n = 0 outside).
    """
    corners = X[T]
    band = (corners[:, :, 2].max(axis=1) > z_mid - half_width) & (corners[:, :, 2].min(axis=1) < z_mid + half_width)
    points = np.einsum("qv,nvd->nqd", BAND_BARY, corners[band])
    zeta = points[..., 2] - z_mid
    dchi = np.where(np.abs(zeta) < half_width, np.cos(np.pi * zeta / (2 * half_width)) ** 2 / half_width, 0.0)
    Ez = np.asarray(E.evaluate(points.reshape(-1, 3)))[:, 2].reshape(points.shape[:2])
    return -4 * np.sum((eps_cell[band] * volume[band])[:, None] * BAND_WEIGHTS[None, :] * Ez * dchi)


# ## Solve
#
# **Device**, documented geometry, on three meshes, plus the coarsest mesh in an air cylinder of
# twice the radius to see the influence of the insulated outer wall.

def run_device(R, level):
    start = time.perf_counter()
    X, T, in_dielectric, facets = device_mesh(R, *level)
    eps_cell = np.where(in_dielectric, eps_r * eps0, eps0)
    E, V, energy, volume = solve(X, T, facets, eps_cell, R, H_air, (0.0, 0.0, 0.0))
    Q = band_charge(E, X, T, eps_cell, volume, 0.5 * (z2 + z3), 0.012)
    print(f"device R = {100*R:.0f} cm, rim size {1e3*level[2]:.2f} mm: {len(T)} tetrahedra "
          f"({in_dielectric.sum()} in the dielectric), {time.perf_counter()-start:.0f} s")
    return dict(tets=len(T), C_W=2 * energy / V0**2, C_Q=Q / V0, E=E, V=V, R=R)


device = [run_device(R_air, level) for level in device_levels]
device_wide = run_device(2 * R_air, device_levels[0])

# **Benchmark disc pair** (zero thickness, air, no leads) in a quarter sphere of radius $3a$, once with
# a charge-free outer surface and once with the outer surface at the mid potential $V_0/2$. For the
# dipole far field of the pair the first underestimates and the second overestimates the free-space
# capacitance, and for a sphere the image-dipole argument gives the first error twice the size of the
# second, so $C_\infty\approx(C_N+2C_D)/3$ with the leading domain error removed.

def run_benchmark(level, outer_potential):
    start = time.perf_counter()
    X, T, facets = disc_pair_mesh(R_sphere, *level)
    eps_cell = np.full(len(T), eps0)
    E, V, energy, volume = solve(X, T, facets, eps_cell, R_sphere, 2 * R_sphere, (0.0, 0.0, -R_sphere), outer_potential)
    label = "D.n = 0" if outer_potential is None else "V = V0/2"
    print(f"disc pair, outer {label:8s}, rim size {1e3*level[2]:.2f} mm: {len(T)} tetrahedra, "
          f"{time.perf_counter()-start:.0f} s")
    return dict(tets=len(T), C=2 * energy / V0**2, E=E, V=V)


bench_N = [run_benchmark(level, None) for level in benchmark_levels]
bench_D = [run_benchmark(level, V0 / 2) for level in benchmark_levels]


# ## Comparison with the reference
#
# **Benchmark: fringing capacitance of the circular disc pair, $d/a=0.3$.**

# The field at the edge of a zero-thickness disc is singular like $r^{-1/2}$, so the energy error of the
# finite-element solution decreases only like $h$ (first order) and always from above (Dirichlet
# principle). The Richardson estimate uses this theoretical order; the observed order is printed too.

ratio = np.sqrt(2)


print(f"reference (Love): {C_love*1e12:.4f} pF;  Kirchhoff {C_kirchhoff*1e12:.4f} pF;  ideal eps0*A/d {C_ideal_air*1e12:.4f} pF")
print(" tetrahedra   C, D.n=0 [pF]   C, V=V0/2 [pF]   (C_N+2C_D)/3 [pF]   dev from Love")
bench_est = []
for n_run, d_run in zip(bench_N, bench_D):
    estimate = (n_run["C"] + 2 * d_run["C"]) / 3
    bench_est.append(estimate)
    print(f" {n_run['tets']:>10d}   {n_run['C']*1e12:13.4f}   {d_run['C']*1e12:14.4f}   {estimate*1e12:17.4f}   "
          f"{100*(estimate/C_love-1):+10.3f} %")
p_b = np.log(abs(bench_est[0] - bench_est[1]) / abs(bench_est[1] - bench_est[2])) / np.log(ratio)
C_b_rich = bench_est[2] + (bench_est[2] - bench_est[1]) / (ratio - 1)
print(f"change between the last two meshes: {100*abs(bench_est[2]/bench_est[1]-1):.3f} %;  observed order {p_b:.2f}")
print(f"first-order Richardson estimate {C_b_rich*1e12:.4f} pF ({100*(C_b_rich/C_love-1):+.3f} % from Love)")
C_b_obs = bench_est[2] + (bench_est[2] - bench_est[1]) / (ratio**p_b - 1)
print(f"(with the observed order instead: {C_b_obs*1e12:.4f} pF, {100*(C_b_obs/C_love-1):+.3f} % from Love)")
print(f"bracket on the finest mesh: {bench_N[-1]['C']*1e12:.4f} pF <= C <= {bench_D[-1]['C']*1e12:.4f} pF "
      f"(Love {C_love*1e12:.4f} pF, plus the discretisation error that raises both bounds)")
print(f"Kirchhoff's formula is {100*(1-C_kirchhoff/C_love):.2f} % below Love; the ideal plate formula misses "
      f"{100*(1-C_ideal_air/C_love):.1f} % of the capacitance")

# **Device: energy and charge routes, mesh convergence, air domain.** The plates are 5 mm thick, so
# their edges are right-angled corners with a much weaker singularity than the thin benchmark discs,
# and the capacitance is dominated by the nearly uniform field in the quartz: it converges fast.

print("\n tetrahedra   C_W energy [pF]   C_Q charge [pF]   C_W vs C_Q")
for run in device:
    print(f" {run['tets']:>10d}   {run['C_W']*1e12:15.4f}   {run['C_Q']*1e12:15.4f}   {100*(run['C_W']/run['C_Q']-1):+9.3f} %")
C_all = np.array([[run["C_W"], run["C_Q"]] for run in device])
print(f"energy route: change between the last two meshes {100*abs(device[2]['C_W']/device[1]['C_W']-1):.4f} %;  "
      f"charge route: {100*abs(device[2]['C_Q']/device[1]['C_Q']-1):.4f} %")
print(f"all six values (3 meshes x 2 routes) lie within {100*(C_all.max()-C_all.min())/C_all.mean():.3f} % of each other")
print(f"air cylinder of radius {100*device_wide['R']:.0f} cm instead of {100*R_air:.0f} cm (coarsest mesh): "
      f"C_W {device_wide['C_W']*1e12:.4f} pF vs {device[0]['C_W']*1e12:.4f} pF ({100*(device_wide['C_W']/device[0]['C_W']-1):+.3f} %)")
C_device = device[-1]["C_W"]
print(f"\ndevice capacitance {C_device*1e12:.3f} pF;  ideal eps0*eps_r*A/d = {C_ideal_device*1e12:.3f} pF "
      f"underestimates it by {100*(1-C_ideal_device/C_device):.1f} %")
print(f"fringing (incl. leads) adds {(C_device-C_ideal_device)*1e12:.3f} pF; the free-space fringing of the "
      f"bare disc pair (Love minus ideal) is {(C_love-C_ideal_air)*1e12:.3f} pF")


# ## Figure
#
# The potential is evaluated on the vertical section at 45 degrees inside the quarter model and shown
# mirrored about the axis (the field is rotationally symmetric).

def section(field, r_max, z_lo, z_hi, n_r, n_z, keep):
    r = np.linspace(1e-4, r_max, n_r)
    z = np.linspace(z_lo, z_hi, n_z)
    Rg, Zg = np.meshgrid(r, z)
    mask = keep(Rg, Zg)
    values = np.full(Rg.shape, np.nan)
    pts = np.column_stack((Rg[mask] / np.sqrt(2), Rg[mask] / np.sqrt(2), Zg[mask]))
    out = np.asarray(field.evaluate(pts))
    values[mask] = out if out.ndim == 1 else np.linalg.norm(out, axis=1)
    return Rg, Zg, values


def outside_metal(R, Z, margin=2e-4):
    plate = (R < a + margin) & (((Z > z1 - margin) & (Z < z2 + margin)) | ((Z > z3 - margin) & (Z < z4 + margin)))
    lead = (R < r_lead + margin) & ((Z < z1 + margin) | (Z > z4 - margin))
    inside_box = (R < 0.995 * R_air) & (Z > 1e-6) & (Z < H_air - 1e-6)
    return ~plate & ~lead & inside_box


def metal_outline(ax, mirror=True):
    for sign in ((-1, 1) if mirror else (1,)):
        for z_lo in (z1, z3):
            ax.add_patch(plt.Rectangle((min(0, sign * a) * 100, z_lo * 100), a * 100, t_plate * 100, color="#8a8a8a", zorder=3))
        ax.add_patch(plt.Rectangle((min(0, sign * r_lead) * 100, 0), r_lead * 100, z1 * 100, color="#8a8a8a", zorder=3))
        ax.add_patch(plt.Rectangle((min(0, sign * r_lead) * 100, z4 * 100), r_lead * 100, (H_air - z4) * 100, color="#8a8a8a", zorder=3))
    ax.add_patch(plt.Rectangle((-a * 100 if mirror else 0, z2 * 100), (2 if mirror else 1) * a * 100, d * 100,
                               fill=False, ec="#2f6f8f", lw=1.0, ls="--", zorder=3))


cream = "#fbf5eb"
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in axes.ravel():
    ax.set_facecolor("#fffaf2")

finest = device[-1]
Rg, Zg, Vs = section(finest["V"], R_air, 0.0, H_air, 201, 201, outside_metal)
ax = axes[0, 0]
Rm, Zm, Vm = np.hstack((-Rg[:, :0:-1], Rg)), np.hstack((Zg[:, :0:-1], Zg)), np.hstack((Vs[:, :0:-1], Vs))
im = ax.contourf(100 * Rm, 100 * Zm, Vm, levels=np.linspace(0, V0, 41), cmap="RdYlBu_r")
ax.contour(100 * Rm, 100 * Zm, Vm, levels=np.linspace(0.05, 0.95, 19), colors="k", linewidths=0.5)
metal_outline(ax)
ax.set_aspect("equal")
ax.set(xlabel="x / cm", ylabel="z / cm", title="Device: potential, section through the axis")
fig.colorbar(im, ax=ax, label="V / V", shrink=0.85)

ax = axes[0, 1]
Rg, Zg, Es = section(finest["E"], 0.13, 0.065, 0.135, 261, 141, outside_metal)
im = ax.contourf(100 * Rg, 100 * Zg, np.log10(Es), levels=50, cmap="viridis")
ax.contour(100 * Rg, 100 * Zg, section(finest["V"], 0.13, 0.065, 0.135, 261, 141, outside_metal)[2],
           levels=np.linspace(0.05, 0.95, 19), colors="w", linewidths=0.5)
metal_outline(ax, mirror=False)
ax.set_xlim(7, 13); ax.set_ylim(6.5, 13.5); ax.set_aspect("equal")
ax.set(xlabel="r / cm", ylabel="z / cm", title="Fringing field at the disc rim, log10 |E| [V/m]")
fig.colorbar(im, ax=ax, shrink=0.85)

ax = axes[1, 0]
ax.grid(alpha=0.25)
tets = [run["tets"] for run in bench_N]
ax.plot(tets, np.array([run["C"] for run in bench_N]) * 1e12, "v-", color="#3887c5", label="FEM, outer D.n = 0 (lower bound in R)")
ax.plot(tets, np.array([run["C"] for run in bench_D]) * 1e12, "^-", color="#A85539", label="FEM, outer V = V0/2 (upper bound in R)")
ax.plot(tets, np.array(bench_est) * 1e12, "o-", color="#214D43", label="(C_N + 2 C_D) / 3")
ax.axhline(C_love * 1e12, color="k", lw=1.4, label="Love's integral equation (exact)")
ax.axhline(C_kirchhoff * 1e12, color="#6b8e23", lw=1.2, ls="--", label="Kirchhoff's formula")
ax.set_xscale("log")
ax.set(xlabel="tetrahedra (quarter model)", ylabel="C / pF",
       title=f"Benchmark disc pair d/a = 0.3 (ideal plates: {C_ideal_air*1e12:.2f} pF)")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.grid(alpha=0.25)
tets = [run["tets"] for run in device]
ax.plot(tets, np.array([run["C_W"] for run in device]) * 1e12, "o-", color="#A85539", label="from field energy 2W/V0^2")
ax.plot(tets, np.array([run["C_Q"] for run in device]) * 1e12, "s--", color="#3887c5", label="from electrode charge Q/V0")
ax.plot([device_wide["tets"]], [device_wide["C_W"] * 1e12], "D", color="#6b8e23", label="air cylinder 40 cm (coarse mesh)")
ax.set_xscale("log")
ax.set(xlabel="tetrahedra (quarter model)", ylabel="C / pF",
       title=f"Device: mesh convergence (ideal eps0 eps_r A/d = {C_ideal_device*1e12:.2f} pF)")
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Shrink the gap: the fringing share falls roughly like $(d/a)\ln(a/d)$ and Kirchhoff's formula becomes
# accurate. Replace the quartz by a high-permittivity ceramic and the fringing in the air becomes a
# small correction, while thicker discs add the rim capacitance that Kirchhoff's thick-plate formula
# describes. The same energy and charge routes give the full capacitance matrix of several electrodes
# when each one is driven in turn.
