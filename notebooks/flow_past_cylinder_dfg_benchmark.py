"""OneSimulation example: steady laminar flow past a cylinder in a channel (DFG benchmark 2D-1).
Requires the installed, activated OneSimulation runtime.

A circular cylinder sits slightly off-centre in a plane channel with a parabolic inflow at
Re = 20. The two-dimensional flow field is computed with Taylor-Hood finite elements on curved,
graded quadratic cells; drag, lift, the pressure drop across the cylinder and the length of the
recirculation zone are graded against the Schaefer-Turek (1996) reference intervals and the
high-accuracy reference values, with a three-level mesh study.
"""


# # Flow past a cylinder: the DFG benchmark 2D-1
#
# The benchmark "laminar flow around a cylinder" of the German Research Foundation (DFG)
# priority programme is the standard test for incompressible flow solvers. A Newtonian fluid
# ($\rho=1$ kg/m³, $\nu=10^{-3}$ m²/s) flows through a channel of length 2.2 m and height
# $H=0.41$ m past a cylinder of diameter $D=0.1$ m centred at $(0.2, 0.2)$ m - slightly below
# the channel axis, so the flow is not symmetric and the cylinder carries a small lift. The
# inflow profile is parabolic with the peak speed $U_m=0.3$ m/s, i.e. the mean speed
# $\bar U=2U_m/3=0.2$ m/s and the Reynolds number $Re=\bar U D/\nu=20$. At this Reynolds number
# the flow is steady: a pair of standing vortices forms behind the cylinder.
#
# The benchmark asks for the drag and lift coefficients
#
# $$c_D=\frac{2F_D}{\rho\bar U^2D},\qquad c_L=\frac{2F_L}{\rho\bar U^2D},$$
#
# the pressure difference $\Delta p=p(0.15,0.2)-p(0.25,0.2)$ between the front and the back
# point of the cylinder, and the length $L_a$ of the recirculation zone behind it.
#
# **Numerical method.**
#
# * Spatial: continuous Galerkin finite elements, Taylor-Hood P2/P1 (quadratic velocity, linear
#   pressure, inf-sup stable) on **curved, isoparametric ten-node tetrahedra**: every cell edge
#   on the cylinder carries a mid-edge node on the exact circle, so the cylinder is represented
#   by quadratic arcs instead of a polygon. The full Navier-Stokes equations with the symmetric
#   viscous stress are solved in their native 3D form on a slab one cell thick whose two faces
#   normal to $z$ form a periodic pair; the computed flow therefore depends on $(x, y)$ only.
# * Mesh: unstructured triangulation of the channel section, graded towards the cylinder and
#   along the near wake, extruded into one layer of prisms that are each split into three
#   tetrahedra.
# * Solver: stationary problem, Newton iteration with a sparse direct LU solver; each finer mesh
#   starts from the solution on the previous one.
# * Forces: the drag and lift are not integrated from the surface traction but from the
#   **volume (weak) form of the momentum balance** tested with a discrete function that is one
#   on the cylinder and zero on all other boundaries. For the Galerkin solution this equals the
#   reaction force of the discrete equations and converges much faster than the surface
#   integral (M. Giles, M. Larson, M. Levenstam, E. Sueli, Oxford NA report 97/06, 1997;
#   V. John, Int. J. Numer. Meth. Fluids 40, 775-798, 2002). The surface integral is evaluated
#   too, for comparison.
#
# **Reference.** M. Schaefer and S. Turek, "Benchmark computations of laminar flow around a
# cylinder", in E. H. Hirschel (ed.), *Flow Simulation with High-Performance Computers II*,
# Notes on Numerical Fluid Mechanics 52, Vieweg 1996, pp. 547-566, test case 2D-1: the published
# intervals $c_D\in[5.57,5.59]$, $c_L\in[0.0104,0.0110]$, $L_a\in[0.0842,0.0852]$ m,
# $\Delta p\in[0.1172,0.1176]$ Pa; and the high-accuracy values $c_D=5.57953523384$,
# $c_L=0.010618948146$, $\Delta p=0.11752016697$ Pa of later studies (V. John and G. Matthies,
# Int. J. Numer. Meth. Fluids 37, 885-903, 2001; FEATFLOW benchmark collection, TU Dortmund).
# A three-level mesh study with observed order and Richardson extrapolation completes the
# checks.

from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import time

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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.optimize import brentq
import netgen.occ as occ
from netgen.meshing import MeshingParameters

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import NewtonianFluid
from onesim.meshing import PeriodicPair
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.polynomial.tetrahedron import EDGES, QuadraticTetrahedralGeometry
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FixedVelocity, FlowTraction, IncompressibleNavierStokes, NoSlipWall
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions.values import InitialValue
from onesim.simulation.problem_definition.boundaries import PeriodicBoundary

print("OneSimulation:", version("onesimulation"))


# ## Parameters

L_channel, H = 2.2, 0.41           # m, channel length and height
xc, yc, R = 0.2, 0.2, 0.05          # m, cylinder centre and radius
D = 2 * R                           # m, cylinder diameter
U_max = 0.3                         # m/s, peak inflow speed
rho, mu = 1.0, 1.0e-3               # kg/m^3, Pa s
U_mean = 2 * U_max / 3
reynolds = rho * U_mean * D / mu
levels = (2 / 3, 1.0, 1.5)          # mesh levels: every cell size scales with 1/level
print(f"mean inflow speed {U_mean:.3f} m/s, Re = {reynolds:.1f}")

reference = {   # Schaefer & Turek (1996) intervals and high-accuracy values
    "c_D": ((5.57, 5.59), 5.57953523384),
    "c_L": ((0.0104, 0.0110), 0.010618948146),
    "dp": ((0.1172, 0.1176), 0.11752016697),
    "L_a": ((0.0842, 0.0852), None),
}


# ## Geometry and mesh
#
# The channel section minus the disc is triangulated by the Netgen mesher (shipped with the
# OneSimulation runtime) with a size field: cell size $6\,\text{mm}/\ell$ on the cylinder,
# growing over one diameter around it, $12\,\text{mm}/\ell$ along the wake up to $x=1$ m and
# $40\,\text{mm}/\ell$ far away, where $\ell$ is the mesh level. The triangles are extruded into
# one layer of prisms of thickness $4\,\text{mm}/\ell$, each split into three tetrahedra with
# a vertex-ordering rule that keeps the mesh conforming. Ten-node tetrahedra are formed by adding
# the edge midpoints; the midpoints of all edges lying on the cylinder surface are moved
# radially onto the circle, which makes the cells next to the cylinder curved.
#
# Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | inlet $x=0$ | parabolic velocity $u=4U_m\,y(H-y)/H^2$, $v=w=0$ |
# | channel walls $y=0$, $y=H$ and the cylinder | no slip, $\mathbf u=0$ |
# | outlet $x=2.2$ m | traction-free, $\boldsymbol\sigma\cdot\mathbf n=0$ (this also fixes the pressure level) |
# | slab faces normal to $z$ | periodic pair |
#
# The benchmark prescribes the "do-nothing" outflow of the Laplace form of the viscous term;
# the traction-free outlet of the symmetric-stress form used here differs only in the
# tangential stress at the outlet, two metres behind the cylinder. At $Re=20$ such outlet
# disturbances decay over a fraction of the channel height, so the cylinder does not feel the
# difference.

@dataclass(frozen=True, eq=False, kw_only=True)
class ChannelSlab(Geometry):
    """Channel section with a cylinder, extruded into a slab one cell thick."""

    dimension = 3

    def boundary_groups(self):
        names = ("inlet", "outlet", "walls", "cylinder", "front", "back")
        return {**{name: (name,) for name in names}, "all": names}


class CurvedTetMesh(StructuredHexahedralMesh):
    """Ten-node (quadratic, isoparametric) tetrahedra with named boundary facets."""

    def __init__(self, geometry, coordinates, cells, facets):
        data = QuadraticTetrahedralGeometry(coordinates, cells)
        self.geometry = geometry
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (self.cell_count,)
        self.mesh_id = "channel-tet10:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        self._facets = facets
        self._carrier = self._build_carrier()

    def _build_carrier(self):
        carrier = super()._build_carrier()
        return replace(carrier, connectivity=replace(carrier.connectivity, entity_kind="tetrahedron10"),
                       mesh_kind="unstructured")

    def to_mapping(self):
        return {**super().to_mapping(), "generator": "channel-tet10", "geometry_mapping": "quadratic_tet10"}


def to_circle(points):
    d = points - (xc, yc)
    return (xc, yc) + R * d / np.linalg.norm(d, axis=1)[:, None]


def channel_section(level):
    shape = occ.WorkPlane().Rectangle(L_channel, H).Face() - occ.WorkPlane().Circle(xc, yc, R).Face()
    h_cyl, h_wake, h_far = 0.006 / level, 0.012 / level, 0.04 / level
    for edge in shape.edges:
        if abs(edge.center.x - xc) < R + 1e-6 and abs(edge.center.y - yc) < R + 1e-6:
            edge.maxh = h_cyl
    params = MeshingParameters(maxh=h_far, grading=0.15)
    for r in np.linspace(R, 2 * R, 6):                       # ring around the cylinder
        for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            params.RestrictH(x=xc + r * np.cos(a), y=yc + r * np.sin(a), z=0,
                             h=h_cyl * (1 + 1.5 * (r - R) / R))
    for x in np.arange(xc, 1.0, h_wake / 2):                 # near wake
        for y in np.linspace(yc - 1.2 * R, yc + 1.2 * R, 7):
            params.RestrictH(x=x, y=y, z=0, h=h_wake if x > xc + 2 * R else 2.5 * h_cyl)
    ng = occ.OCCGeometry(shape, dim=2).GenerateMesh(mp=params)
    xy = np.array([p.p for p in ng.Points()])[:, :2]
    triangles = np.array([[v.nr for v in e.vertices] for e in ng.Elements2D()]) - 1
    on_cylinder = np.abs(np.hypot(xy[:, 0] - xc, xy[:, 1] - yc) - R) < 1e-7
    xy[on_cylinder] = to_circle(xy[on_cylinder])
    return xy, triangles, on_cylinder


def build_mesh(level):
    xy, triangles, on_cylinder = channel_section(level)
    n2, dz = len(xy), 0.004 / level
    coords = np.vstack([np.column_stack((xy, np.full(n2, z))) for z in (0.0, dz)])
    i, j, k = np.sort(triangles, axis=1).T
    I, J, K = i + n2, j + n2, k + n2
    cells = np.vstack([np.column_stack((i, j, k, K)), np.column_stack((i, j, J, K)),
                       np.column_stack((i, I, J, K))])
    p = coords[cells]
    negative = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) < 0
    cells[negative] = cells[negative][:, [0, 2, 1, 3]]
    geometry = ChannelSlab()
    facets = {name: [] for name in geometry.boundary_groups()["all"]}
    for face in boundary_faces(cells, coords):
        v = np.asarray(face.vertices)
        q = coords[v]
        if np.all(q[:, 2] < 1e-12):
            name = "front"
        elif np.all(q[:, 2] > dz - 1e-12):
            name = "back"
        elif np.all(np.abs(q[:, 0]) < 1e-9):
            name = "inlet"
        elif np.all(np.abs(q[:, 0] - L_channel) < 1e-9):
            name = "outlet"
        elif np.all(np.abs(q[:, 1]) < 1e-9) or np.all(np.abs(q[:, 1] - H) < 1e-9):
            name = "walls"
        else:
            name = "cylinder"
        facets[name].append(tuple(int(n) for n in v))
    # ten-node cells: edge midpoints, those of edges on the cylinder moved onto the circle
    edges = np.sort(cells[:, list(EDGES)].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    middle = 0.5 * (coords[unique[:, 0]] + coords[unique[:, 1]])
    curved = on_cylinder[unique[:, 0] % n2] & on_cylinder[unique[:, 1] % n2]
    middle[curved, :2] = to_circle(middle[curved, :2])
    nodes = np.vstack([coords, middle])
    cells10 = np.hstack([cells, n2 * 2 + inverse.reshape(len(cells), 6)])
    mesh = CurvedTetMesh(geometry, nodes, cells10, facets)
    return dict(level=level, geometry=geometry, mesh=mesh, xy=xy, triangles=triangles, on_cylinder=on_cylinder,
                dz=dz, curved_edges=int(curved.sum()))


for level in levels:
    xy, triangles, on_cylinder = channel_section(level)
    print(f"level {level:.2f}: {len(triangles):6d} triangles in the section, {int(on_cylinder.sum()):3d} arcs "
          f"on the cylinder, {3 * len(triangles):6d} tetrahedra in the slab")


# ## Model
#
# One `IncompressibleNavierStokes` model per mesh with the boundary conditions of the table.
# The velocity is sampled for the Newton starting value from the previous mesh (or the
# parabolic inflow profile everywhere on the first mesh).

fluid = NewtonianFluid(density=rho, dynamic_viscosity=mu)


def inflow(x, y, z):
    u = 4 * U_max * y * (H - y) / H**2
    return np.column_stack((u, np.zeros_like(u), np.zeros_like(u)))


def no_traction(x, y, z):
    return np.zeros((len(np.asarray(x)), 3))


def previous_velocity(run):
    """Velocity of a coarser solution as a starting value (points on the cylinder nudged inside the fluid)."""
    def value(x, y, z):
        p = np.column_stack((x, y))
        r = np.hypot(p[:, 0] - xc, p[:, 1] - yc)
        near = r < R + 1e-4
        p[near] = (xc, yc) + (R + 1e-4) * (p[near] - (xc, yc)) / r[near, None]
        p[:, 0] = np.clip(p[:, 0], 0.0, L_channel)
        p[:, 1] = np.clip(p[:, 1], 0.0, H)
        return np.asarray(run["velocity"].evaluate(np.column_stack((p, np.full(len(p), run["dz"] / 2)))))
    return value


def solve(data, start=None):
    begin = time.perf_counter()
    geometry, mesh = data["geometry"], data["mesh"]
    b = geometry.boundaries
    model = IncompressibleNavierStokes(domain=geometry.domain(), material=fluid)
    problem = Problem(
        name="DFG benchmark 2D-1", geometry=geometry, physics=(model,),
        boundary_conditions=(
            FixedVelocity(boundary=b.inlet, physics=model, value=inflow),
            NoSlipWall(boundary=b.walls | b.cylinder, physics=model),
            FlowTraction(boundary=b.outlet, physics=model, value=no_traction),
            PeriodicBoundary(pair=PeriodicPair(first=b.front, second=b.back), physics=model),
        ),
        initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
    )
    numerics = Numerics(
        spatial=FiniteElement(element=TaylorHood(), quadrature_order=5),
        nonlinear_solver=Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-10,
                                relative_tolerance=1e-10, maximum_iterations=30),
        initial_guess={model.velocity: inflow if start is None else previous_velocity(start)},
    )
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    return dict(data, velocity=result.field(model.velocity), pressure=result.field(model.pressure),
                seconds=time.perf_counter() - begin)


# ## Post-processing on the section
#
# On the slab face $z=0$ every ten-node tetrahedron has one of the section triangles as a face,
# and the trace of the quadratic velocity on that face is the six-node (curved) quadratic
# triangle interpolant of its nodal values; the pressure trace is linear. So the velocity is
# sampled at the six nodes of every section triangle and the forces are integrated exactly over
# the curved section with a 7-point Gauss rule (degree 5).
#
# *Volume form.* With the discrete function $\phi$ equal to one at all velocity nodes on the
# cylinder and zero at all other nodes, the force on the cylinder per unit span is
#
# $$F_c=-\int_\Omega\Big[\rho\,(\mathbf u\cdot\nabla)\mathbf u\cdot\mathbf e_c\,\phi
#   +\mu\,(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T}):\nabla(\phi\,\mathbf e_c)
#   -p\,\nabla\cdot(\phi\,\mathbf e_c)\Big]\,\mathrm dA ,\qquad c=x,y,$$
#
# which is the weak momentum balance tested with $\phi\,\mathbf e_c$; only the triangles that
# touch the cylinder contribute. *Surface form.* $F_c=\oint(-p\,\mathbf n+\mu(\nabla\mathbf u+
# \nabla\mathbf u^{\mathsf T})\mathbf n)\cdot\mathbf e_c\,\mathrm ds$ with a 5-point Gauss rule on
# every curved boundary arc, $\mathbf n$ pointing from the cylinder into the fluid.

A1, B1 = 0.059715871789770, 0.470142064105115
A2, B2 = 0.797426985353087, 0.101286507323456
TRI_POINTS = np.array([[1 / 3, 1 / 3], [B1, B1], [A1, B1], [B1, A1], [B2, B2], [A2, B2], [B2, A2]])
TRI_WEIGHTS = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3) / 2
DL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


def p2_shape(s, t):
    l0, l1, l2 = 1 - s - t, s, t
    N = np.array([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0])
    dN = np.array([(4 * l0 - 1) * DL[0], (4 * l1 - 1) * DL[1], (4 * l2 - 1) * DL[2],
                   4 * (l0 * DL[1] + l1 * DL[0]), 4 * (l1 * DL[2] + l2 * DL[1]), 4 * (l2 * DL[0] + l0 * DL[2])])
    return N, dN


def section_p2(run):
    """Six-node section triangles (curved on the cylinder) with the sampled solution."""
    xy, triangles, on_cylinder = run["xy"], run["triangles"], run["on_cylinder"]
    edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    middle = 0.5 * (xy[unique[:, 0]] + xy[unique[:, 1]])
    arc = on_cylinder[unique[:, 0]] & on_cylinder[unique[:, 1]]
    middle[arc] = to_circle(middle[arc])
    nodes = np.vstack([xy, middle])
    t6 = np.column_stack((triangles, len(xy) + inverse.reshape(3, len(triangles)).T))
    sample = np.column_stack((nodes, np.zeros(len(nodes))))
    U = np.asarray(run["velocity"].evaluate(sample))
    P = np.asarray(run["pressure"].evaluate(sample[: len(xy)])).reshape(-1)
    phi = np.r_[on_cylinder, arc].astype(float)
    return dict(nodes=nodes, t6=t6, U=U, P=P, phi=phi)


def volume_forces(run, s2):
    X, Ut, Pt, phit = s2["nodes"][s2["t6"]], s2["U"][s2["t6"]][:, :, :2], s2["P"][run["triangles"]], s2["phi"][s2["t6"]]
    touch = phit.max(axis=1) > 0
    X, Ut, Pt, phit = X[touch], Ut[touch], Pt[touch], phit[touch]
    F = np.zeros(2)
    for (s, t), w in zip(TRI_POINTS, TRI_WEIGHTS):
        N, dN = p2_shape(s, t)
        J = np.einsum("tkd,ke->tde", X, dN)
        G = np.einsum("ke,tef->tkf", dN, np.linalg.inv(J))
        detJ = np.linalg.det(J)
        u, gu = np.einsum("k,tkc->tc", N, Ut), np.einsum("tkf,tkc->tcf", G, Ut)
        ph, gph = phit @ N, np.einsum("tkf,tk->tf", G, phit)
        p = Pt @ np.array([1 - s - t, s, t])
        convection = rho * np.einsum("tf,tcf->tc", u, gu)
        for c in range(2):
            viscous = mu * np.einsum("tf,tf->t", gu[:, c, :] + gu[:, :, c], gph)
            F[c] -= np.sum(w * detJ * (convection[:, c] * ph + viscous - p * gph[:, c]))
    return F


GAUSS5, WEIGHT5 = np.polynomial.legendre.leggauss(5)
GAUSS5, WEIGHT5 = (GAUSS5 + 1) / 2, WEIGHT5 / 2


def surface_forces(run, s2):
    t6, nodes, U, P, triangles, on = s2["t6"], s2["nodes"], s2["U"], s2["P"], run["triangles"], run["on_cylinder"]
    F = np.zeros(2)
    for a, b, start, direction in ((0, 1, (0, 0), (1, 0)), (1, 2, (1, 0), (-1, 1)), (2, 0, (0, 1), (0, -1))):
        sel = on[triangles[:, a]] & on[triangles[:, b]]
        X, Ut, Pt = nodes[t6[sel]], U[t6[sel]][:, :, :2], P[triangles[sel]]
        for tau, w in zip(GAUSS5, WEIGHT5):
            s, t = start[0] + tau * direction[0], start[1] + tau * direction[1]
            N, dN = p2_shape(s, t)
            J = np.einsum("tkd,ke->tde", X, dN)
            G = np.einsum("ke,tef->tkf", dN, np.linalg.inv(J))
            x = np.einsum("k,tkd->td", N, X)
            tangent = J @ np.array(direction, float)
            ds = np.linalg.norm(tangent, axis=1)
            normal = np.column_stack((tangent[:, 1], -tangent[:, 0])) / ds[:, None]
            normal *= np.sign(np.einsum("td,td->t", normal, x - (xc, yc)))[:, None]
            gu = np.einsum("tkf,tkc->tcf", G, Ut)
            p = Pt @ np.array([1 - s - t, s, t])
            traction = -p[:, None] * normal + mu * np.einsum("tcf,tf->tc", gu + gu.transpose(0, 2, 1), normal)
            F += np.sum(w * ds[:, None] * traction, axis=0)
    return F


def benchmark_values(run):
    s2 = section_p2(run)
    scale = 2.0 / (rho * U_mean**2 * D)
    F_vol, F_srf = volume_forces(run, s2), surface_forces(run, s2)
    p_front, p_back = np.asarray(run["pressure"].evaluate(np.array([[0.15, 0.2, 0.0], [0.25, 0.2, 0.0]]))).ravel()
    xs = np.linspace(0.2501, 0.45, 4001)
    ux = np.asarray(run["velocity"].evaluate(np.column_stack((xs, np.full_like(xs, yc), np.zeros_like(xs)))))[:, 0]
    k = np.flatnonzero(ux > 0)[0]
    x_end = xs[k - 1] - ux[k - 1] * (xs[k] - xs[k - 1]) / (ux[k] - ux[k - 1])
    mid = np.asarray(run["velocity"].evaluate(np.column_stack((s2["nodes"][::5], np.full(len(s2["nodes"][::5]), run["dz"] / 2)))))
    return dict(c_D=scale * F_vol[0], c_L=scale * F_vol[1], c_D_surface=scale * F_srf[0], c_L_surface=scale * F_srf[1],
                dp=p_front - p_back, L_a=x_end - (xc + R), section=s2,
                z_variation=float(np.max(np.abs(mid - s2["U"][::5]))) / U_max, w_max=float(np.max(np.abs(s2["U"][:, 2]))) / U_max)


# ## Solve

runs = []
for level in levels:
    data = build_mesh(level)
    run = solve(data, runs[-1] if runs else None)
    run.update(benchmark_values(run))
    runs.append(run)
    print(f"level {level:.2f}: {run['mesh'].cell_count:6d} curved tetrahedra ({run['curved_edges']} curved edges), "
          f"c_D = {run['c_D']:.6f}, c_L = {run['c_L']:.6f}, dp = {run['dp']:.6f} Pa, L_a = {run['L_a']:.5f} m  "
          f"({run['seconds']:.0f} s)")
final = runs[-1]


# ## Comparison with the reference
#
# **Benchmark quantities on the finest mesh** against the published intervals and the
# high-accuracy values. **Mesh study.** For three meshes with cell sizes $h\propto1/\ell$ the
# observed order $p$ solves $(f_1-f_2)/(f_2-f_3)=(h_1^p-h_2^p)/(h_2^p-h_3^p)$, and the
# Richardson estimate is $f_3+(f_3-f_2)/((h_2/h_3)^p-1)$. The three meshes are generated
# independently (not by nested refinement), so a small quantity such as the lift - the
# difference of two large pressure forces on the upper and lower half of the cylinder - need not
# converge monotonically; its spread over the meshes is then the honest error indicator.

def richardson(values):
    f1, f2, f3 = values
    h1, h2, h3 = 1.0 / np.asarray(levels)
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda q: (h1**q - h2**q) / (h2**q - h3**q) - ratio, 0.3, 12.0)
    except ValueError:
        return np.nan, np.nan
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1)


labels = {"c_D": "drag coefficient c_D", "c_L": "lift coefficient c_L", "dp": "pressure difference dp / Pa",
          "L_a": "recirculation length L_a / m"}
print(f"{'quantity':30s} {'finest mesh':>12s} {'interval (1996)':>18s} {'high-accuracy':>14s} {'deviation':>10s}")
for key, (interval, accurate) in reference.items():
    value = final[key]
    inside = "inside" if interval[0] <= value <= interval[1] else "OUTSIDE"
    dev = f"{value / accurate - 1:+.2e}" if accurate else "      -"
    acc = f"{accurate:.6f}" if accurate else "-"
    print(f"{labels[key]:30s} {value:12.6f}   [{interval[0]:.4f}, {interval[1]:.4f}] {inside:7s} {acc:>14s} {dev:>10s}")

print("\nMesh study (levels " + ", ".join(f"{lv:.2f}" for lv in levels) + ")")
study = {}
for key in ("c_D", "c_L", "dp", "L_a", "c_D_surface"):
    values = [r[key] for r in runs]
    p, extrapolated = richardson(values)
    study[key] = (values, p, extrapolated)
    accurate = reference.get(key.replace("_surface", ""), (None, None))[1]
    line = f"  {key:12s} " + "  ".join(f"{v:.6f}" for v in values) + f"   change last two meshes {abs(values[2] / values[1] - 1):.1e}"
    if np.isfinite(p):
        line += f", observed order {p:.2f}, Richardson {extrapolated:.6f}"
        if accurate:
            line += f" ({extrapolated / accurate - 1:+.1e} from the high-accuracy value)"
    else:
        line += (f", not monotone on these non-nested meshes (spread {np.ptp(values) / values[2]:.1e});"
                 " no Richardson estimate")
    print(line)

print("\nDrag from the volume form vs the surface traction integral (deviation from 5.57953523384):")
for r in runs:
    print(f"  level {r['level']:.2f}: volume {r['c_D']:.6f} ({r['c_D'] / 5.57953523384 - 1:+.1e}),  surface "
          f"{r['c_D_surface']:.6f} ({r['c_D_surface'] / 5.57953523384 - 1:+.1e});  lift volume {r['c_L']:.6f}, "
          f"surface {r['c_L_surface']:.6f}")
print(f"\nTwo-dimensionality on the finest mesh: max |w| / U_max = {final['w_max']:.1e}, "
      f"max |u(z = dz/2) - u(z = 0)| / U_max = {final['z_variation']:.1e}")


# ## Figure
#
# Top: speed and streamlines in the whole channel. Bottom, from left: vorticity and
# streamlines around the cylinder (the standing vortex pair behind it), the curved graded mesh
# next to the cylinder, and the mesh convergence of the benchmark quantities against the
# high-accuracy values.

s2 = final["section"]
nodes, t6, U = s2["nodes"], s2["t6"], s2["U"]
sub = np.vstack([t6[:, [0, 3, 5]], t6[:, [3, 1, 4]], t6[:, [5, 4, 2]], t6[:, [3, 4, 5]]])
tri = Triangulation(nodes[:, 0], nodes[:, 1], sub)
speed = np.hypot(U[:, 0], U[:, 1])

# vorticity at the nodes, averaged over the triangles sharing a node
vorticity, count = np.zeros(len(nodes)), np.zeros(len(nodes))
local = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]])
X, Ut = nodes[t6], U[t6][:, :, :2]
for n, (s, t) in enumerate(local):
    _, dN = p2_shape(s, t)
    G = np.einsum("ke,tef->tkf", dN, np.linalg.inv(np.einsum("tkd,ke->tde", X, dN)))
    gu = np.einsum("tkf,tkc->tcf", G, Ut)
    np.add.at(vorticity, t6[:, n], gu[:, 1, 0] - gu[:, 0, 1])
    np.add.at(count, t6[:, n], 1)
vorticity /= count


def stream_grid(x0, x1, y0, y1, nx, ny):
    gx, gy = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    GX, GY = np.meshgrid(gx, gy)
    ux = LinearTriInterpolator(tri, U[:, 0])(GX, GY)
    uy = LinearTriInterpolator(tri, U[:, 1])(GX, GY)
    inside = np.hypot(GX - xc, GY - yc) < R
    return gx, gy, np.ma.masked_where(inside, ux.filled(0.0)), np.ma.masked_where(inside, uy.filled(0.0))


fig = plt.figure(figsize=(16, 9.6), constrained_layout=True)
layout = fig.add_gridspec(2, 3, height_ratios=(0.62, 1.0), width_ratios=(1.15, 0.9, 1.0))

ax = fig.add_subplot(layout[0, :])
filled = ax.tricontourf(tri, speed, levels=np.linspace(0, speed.max(), 41), cmap=FIELD_CMAP)
gx, gy, ux, uy = stream_grid(0.0, L_channel, 0.0, H, 440, 82)
ax.streamplot(gx, gy, ux, uy, density=(4.0, 0.9), color=INK, linewidth=0.5, arrowsize=0.6)
ax.add_patch(plt.Circle((xc, yc), R, color="#D5D7C8", ec=INK, lw=0.8))
ax.set_aspect("equal")
ax.set(xlim=(0, L_channel), ylim=(0, H), xlabel="x / m", ylabel="y / m",
       title=f"Speed |u| and streamlines, Re = {reynolds:.0f} (finest mesh, {final['mesh'].cell_count} curved P2/P1 cells)")
fig.colorbar(filled, ax=ax, label="|u| / (m/s)", shrink=0.9, pad=0.01, format="%.2f")

ax = fig.add_subplot(layout[1, 0])
limit = np.percentile(np.abs(vorticity), 99.0)
filled = ax.tricontourf(tri, np.clip(vorticity, -limit, limit), levels=np.linspace(-limit, limit, 41),
                        cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(0.0, -limit, limit))
gx, gy, ux, uy = stream_grid(0.1, 0.45, 0.0, H, 300, 300)
ax.streamplot(gx, gy, ux, uy, density=2.2, color=INK, linewidth=0.5, arrowsize=0.6)
ax.add_patch(plt.Circle((xc, yc), R, color="#D5D7C8", ec=INK, lw=0.8))
ax.axvline(xc + R + final["L_a"], color=COPPER, lw=1.0, ls="--")
ax.set_aspect("equal")
ax.set(xlim=(0.1, 0.45), ylim=(0.05, 0.35), xlabel="x / m", ylabel="y / m",
       title=f"Vorticity and streamlines (L_a = {final['L_a'] * 1e3:.1f} mm)")
fig.colorbar(filled, ax=ax, label="vorticity / (1/s)", shrink=0.7, format="%.0f")

ax = fig.add_subplot(layout[1, 1])
segments = []
theta = np.linspace(0, 1, 9)
for a, b, m in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
    pa, pb, pm = nodes[t6[:, a]], nodes[t6[:, b]], nodes[t6[:, m]]
    curve = (2 * (1 - theta[:, None, None]) * (0.5 - theta[:, None, None]) * pa[None]
             + 4 * theta[:, None, None] * (1 - theta[:, None, None]) * pm[None]
             + theta[:, None, None] * (2 * theta[:, None, None] - 1) * pb[None])
    segments.append(curve)
window = (np.abs(nodes[t6[:, 0], 0] - xc) < 0.1) & (np.abs(nodes[t6[:, 0], 1] - yc) < 0.1)
for curve in segments:
    ax.plot(curve[:, window, 0], curve[:, window, 1], color=GREEN, lw=0.35)
ax.add_patch(plt.Circle((xc, yc), R, color="#E4D6A0", ec=COPPER, lw=1.2, zorder=0))
ax.set_aspect("equal")
ax.set(xlim=(xc - 0.08, xc + 0.08), ylim=(yc - 0.08, yc + 0.08), xlabel="x / m", ylabel="y / m",
       title="Graded mesh near the cylinder (curved edges on the circle)")

ax = fig.add_subplot(layout[1, 2])
h = 1.0 / np.asarray(levels)
for key, marker, colour, name in (("c_D", "o-", GREEN, "c_D, volume form"), ("c_D_surface", "s--", "#8B9A8D", "c_D, surface integral"),
                                  ("c_L", "^-", COPPER, "c_L, volume form"), ("dp", "D-", "#214D43", "pressure difference")):
    accurate = reference[key.replace("_surface", "")][1]
    ax.loglog(h, [abs(r[key] / accurate - 1) for r in runs], marker, color=colour, label=name)
ax.loglog(h, 0.5 * abs(runs[0]["c_D"] / 5.57953523384 - 1) * h**2 / h[0] ** 2, ":", color=INK, lw=0.8, label="slope 2")
ax.axhline(0.01 / 5.58, color=GREEN, lw=0.6, alpha=0.6)
ax.text(h[0] * 0.97, 0.01 / 5.58 * 1.15, "half-width of the 1996 c_D interval", fontsize=7.5, color=GREEN, ha="right")
ax.set_xticks(h, [f"{v:.2f}" for v in h])
ax.minorticks_off()
ax.set(xlabel="relative cell size 1 / level", ylabel="|value / high-accuracy value - 1|",
       title="Mesh convergence against the high-accuracy values")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
plt.show()


# ## Going further
#
# The companion test case 2D-2 raises the peak inflow speed to $U_m=1.5$ m/s ($Re=100$): the
# standing vortex pair breaks up into a periodic von Karman vortex street. The benchmark then
# asks for the Strouhal number $St=Df/\bar U\in[0.2950, 0.3050]$, the maximum drag coefficient
# $c_{D,\max}\in[3.22, 3.24]$ and the maximum lift coefficient $c_{L,\max}\in[0.99, 1.01]$. The
# same model runs it with `Simulation.run` and a BDF2 time integrator in place of the
# stationary solve; resolving the shedding needs about 50 time steps per period over several
# periods after the start-up, so it is a much longer computation than this steady case. Raising
# the mesh level further pushes the drag and the pressure difference closer still to the
# high-accuracy values; the lift, a small difference of large pressure forces, converges most
# slowly and profits most from refinement right at the cylinder.
