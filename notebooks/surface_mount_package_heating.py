"""OneSimulation example: heat transfer in a surface-mount package next to a hot voltage regulator.
Requires the installed, activated OneSimulation runtime.

A silicon die in a 16-lead plastic gull-wing package (SOIC-16) sits on an FR-4 circuit board. It
dissipates 20 mW and is connected through a thin copper interconnect, an aluminium lead and a
copper ground plane to a voltage regulator at 50 degC. The steady 3D temperature field is computed
with isoparametric tensor-product finite elements on a curved, multi-block hexahedral mesh
(quadratic and linear elements compared) and checked against the published peak die temperature,
the global energy balance, a thermal-resistance network and mesh studies.
"""


# # Heat transfer in a surface-mount package for a silicon chip
#
# Designers of printed-circuit boards have to keep heat-sensitive integrated circuits away from
# heat sources. Here a voltage regulator, whose footprint on the board is held at 50 °C, shares
# a copper ground plane with one lead of a small-outline package. Inside the package a silicon
# die dissipates its own power. The question is how hot the die gets.
#
# **Geometry** (all dimensions in mm; $x$ along the package, $z$ normal to the board):
#
# | part | geometry | conductivity / (W/(m K)) |
# |---|---|---|
# | circuit board (FR-4) | $20\times10\times1$, $-1.9\le z\le-0.9$ | 0.3 |
# | ground plane (copper) | $4\times4\times0.1$ in the board top, $-8\le x\le-4$, $-5\le y\le-1$ | 400 |
# | regulator footprint | $-10\le x\le-8$, $-5\le y\le-1$ on the board top, held at 50 °C | - |
# | package body (plastic) | $9.9\times3.9$, $\lvert z\rvert\le0.75$; side walls vertical for $\lvert z\rvert\le0.1$, then tapered to a half-width of 1.72 at the top and bottom faces | 0.2 |
# | silicon die | $1\times1\times0.1$, centred at the origin | 130 |
# | 16 gull-wing leads (aluminium) | $0.4\times0.2$ cross-section, pitch 1.27; stub 0.26 long, a 90° bend (inner radius 0.14), a vertical leg of 0.32, a second 90° bend, a 0.16 long foot on the board | 238 |
# | interconnect (copper) | 5 µm thick L-shaped strip in the plane $z=0$, 0.4 wide, from lead 1 ($x=-4.445$) along $y$ to $y=0$, then along $x$ to the die | 400 |
#
# These are the dimensions of a narrow-body SOIC-16 (JEDEC MS-012: body 9.9 x 3.9 mm, lead pitch
# 1.27 mm, lead width 0.31-0.51 mm, lead thickness 0.19-0.25 mm). The body stands 0.15 mm above
# the board, so the air gap under it is part of the cooled surface.
#
# **Physics.** The steady heat equation $\nabla\cdot(k\nabla T)+Q=0$ with the piecewise constant
# conductivities above and a uniform source $Q=2\times10^8$ W/m³ in the die, i.e.
# $P=20$ mW. Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | regulator footprint on the board top | $T=50$ °C |
# | every other exterior face (board, package, leads, air gap) | forced convection $-k\,\partial_nT=h\,(T-T_\infty)$, $h=50$ W/(m² K), $T_\infty=30$ °C |
#
# The thin copper layers (5 µm interconnect, 0.1 mm ground plane) are resolved as true 3D
# layers of cells, not lumped into a shell model. The field is genuinely three-dimensional:
# the heat flows from the regulator through the ground plane, up lead 1, along the
# interconnect into the package, and from the die through the plastic into the air and the
# leads.
#
# **Numerical method.** Continuous Galerkin finite elements with tensor-product Lagrange
# elements on hexahedra: quadratic (Q2, 27 nodes) and, for comparison, trilinear (Q1, 8 nodes).
# The mesh is a conforming multi-block hexahedral mesh built here in numpy: a graded
# tensor-product block for the board, a tapered block for the package (the taper is an exact
# bilinear map), and sixteen swept blocks for the leads. Every cell carries 27 geometry nodes,
# so the circular lead bends are represented by curved (isoparametric) quadratic cells. Both
# element orders use the same curved cells. The problem is linear and steady: one sparse direct
# LU solve per mesh, no time integration.
#
# **Reference.** (i) The published value for this set-up, a maximum silicon temperature of
# 47.6 °C (obtained with a shell approximation for the copper layers on a free tetrahedral mesh),
# (ii) the global energy balance: heat entering through the regulator footprint plus the die power
# equals the heat convected to the air, (iii) a thermal-resistance network as an independent
# plausibility estimate, and (iv) three-level mesh studies of the maximum die temperature
# for both element orders, with observed order and Richardson extrapolation.

from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
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
import numpy as np
from matplotlib import cm, colors
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from onesim.cad.geometry.primitives import Geometry
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES
from onesim.meshing.generation.contracts.polynomial.hex import HEX27_REFERENCE, QuadraticHexahedralGeometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, FixedTemperature, HeatConduction
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters

mm = 1e-3
K_FR4, K_CU, K_AL, K_SI, K_PLASTIC = 0.3, 400.0, 238.0, 130.0, 0.2   # W/(m K)
Q_DIE = 2e8                     # W/m^3
V_DIE = 1.0 * 1.0 * 0.1 * mm**3
P_DIE = Q_DIE * V_DIE           # W
h_air, T_air, T_reg = 50.0, 303.15, 323.15
T_published = 47.6              # degC, published maximum silicon temperature
PITCH, LEAD_W, LEAD_X0 = 1.27, 0.4, -4.645    # mm
T_IC = 0.0025                   # mm, half thickness of the interconnect
DIE_SIZE, DIE_GROWTH = 0.08, 0.5  # mm and mm/mm: in-plane cell size at the die edges and its growth
print(f"die power P = {P_DIE * 1e3:.1f} mW")


# ## Geometry and mesh
#
# **Blocks.** The board and the package are separate tensor-product blocks whose grid lines pass
# through every material boundary: in the package the lead edges, the die and the interconnect
# strip, in the board the lead edges (where the feet stand), the ground plane and the regulator
# footprint. In $z$ the package grid contains the die faces ($\pm0.05$ mm), the two faces of the
# 5 µm interconnect and the start of the taper ($\pm0.1$ mm). Above
# $\lvert z\rvert=0.1$ mm the package nodes are moved inward linearly in $y$ to follow the
# tapered side walls. Each lead is a swept block: its cross-section grid is the package grid on
# the side wall ($x$ lines of the lead, $z$ lines between $\pm0.1$ mm), and it is swept along
# the lead centre-plane path, stub, quarter circle, leg, quarter circle, foot. The foot meets
# the board top on board grid lines and the stub meets the package wall on package grid lines,
# so all blocks share their nodes. Nodes are merged by position.
#
# **Curved cells.** Every cell is a 27-node hexahedron: its edge midpoints, face centres and
# centre are sampled from the exact block maps. In the bends the cells therefore follow the
# circular arcs to third order in the cell size.
#
# **Grading and refinement level.** The temperature gradients are steepest in the plastic at the
# die edges, where the conductivity jumps from 130 to 0.2 W/(m K). The package grid is therefore
# graded: in the package plane the cell size is 0.08 mm at the die edges and grows by 0.5 mm per
# mm of distance up to 0.45 mm; across the tapered plastic it grows from 0.05 mm at the die level
# to about 0.28 mm at the package faces. A level $\ell$ divides all these sizes by $\ell$, as well
# as the 0.05 mm through the die, 0.3 mm through the board and 0.2 mm along the leads; the board
# grid grows away from the package. Material interfaces
# stay on cell faces at every level, so each quadrature point lies inside one material.

@dataclass(frozen=True, eq=False, kw_only=True)
class PackageOnBoard(Geometry):
    """Board, package, die, leads and copper layers as one conforming solid."""

    dimension = 3

    def boundary_groups(self):
        names = ("regulator", "air")
        return {**{name: (name,) for name in names}, "all": names}


def subdivide(breaks, size):
    """Nodes through all break points; each interval split into cells no larger than size(midpoint)."""
    out = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(np.ceil((b - a) / size(0.5 * (a + b)) - 1e-9)))
        out.extend(a + (b - a) * np.arange(1, n + 1) / n)
    return np.array(out)


def grid_lines(level):
    """Grid lines (mm) of the package block, the board block and the lead sweep at a refinement level."""
    h = 0.45 / level
    leads = sorted({LEAD_X0 + PITCH * k + d for k in range(8) for d in (0.0, LEAD_W)})
    # Package: lead edges, die, interconnect strip; uniform spacing h.
    def near_die(v):   # graded towards the die edges, where the conductivity jumps 650-fold
        return min(h, (DIE_SIZE + DIE_GROWTH * abs(abs(v) - 0.5)) / level)
    px = subdivide(sorted({-4.95, -0.5, 0.5, 4.95, *leads}), near_die)
    py = subdivide([-1.95, -0.5, -0.2, 0.2, 0.5, 1.95], near_die)
    pz = subdivide([-0.75, -0.1, -0.05, -T_IC, T_IC, 0.05, 0.1, 0.75],
                   lambda z: 0.05 / level if abs(z) < 0.1 else (0.05 + 0.35 * (abs(z) - 0.1)) / level)
    # Board: the lead lines of the package (where the feet stand), the ground plane and the
    # regulator footprint; spacing h near the package, growing towards the board edges.
    lead_lines = [x for x in px if any(LEAD_X0 + PITCH * k - 1e-9 <= x <= LEAD_X0 + PITCH * k + LEAD_W + 1e-9
                                         for k in range(8))]
    bx = subdivide(sorted({-10.0, -8.0, -6.5, -4.0, 7.0, 10.0, *lead_lines}),
                   lambda x: h * (1.0 + 0.8 * max(abs(x) - 4.95, 0.0)))
    n_foot = int(np.ceil(0.16 / h - 1e-9))
    foot = list(-2.69 - 0.16 * np.arange(n_foot + 1) / n_foot)
    by = subdivide(sorted({-5.0, *foot, -1.0, 0.0, *(-v for v in foot), 5.0}),
                   lambda y: h * (1.0 + 0.8 * max(abs(y) - 2.85, 0.0)) if abs(y) > 2.69 else 1.5 * h)
    bz = subdivide([-1.9, -1.0, -0.9], lambda z: 0.3 / level)
    n_path = [max(1, int(np.ceil(length / (0.2 / level) - 1e-9))) for length in (0.26, 0.377, 0.32, 0.377)]
    return (px, py, pz), (bx, by, bz), n_path + [n_foot]


def with_midpoints(a):
    out = np.empty(2 * len(a) - 1)
    out[::2] = a
    out[1::2] = 0.5 * (a[1:] + a[:-1])
    return out


def half_width(z):
    """Package half-width: 1.95 mm for |z| <= 0.1 mm, tapering to 1.72 mm at |z| = 0.75 mm."""
    return 1.95 - 0.23 * np.clip((np.abs(z) - 0.1) / 0.65, 0.0, 1.0)


def lead_path(u, t):
    """(y, z) of a lead on the y < 0 side; u in [0, 5] runs over stub, bend, leg, bend, foot,
    t in [0, 0.2] across the thickness (t = 0 is the lower face of the stub)."""
    u, t = np.broadcast_arrays(np.asarray(u, float), np.asarray(t, float))
    segment = np.minimum(np.floor(u).astype(int), 4)
    s = u - segment
    y, z = np.empty(u.shape), np.empty(u.shape)
    m = segment == 0
    y[m], z[m] = -1.95 - 0.26 * s[m], -0.1 + t[m]
    m = segment == 1
    r, phi = 0.14 + t[m], 0.5 * np.pi * s[m]
    y[m], z[m] = -2.21 - r * np.sin(phi), -0.24 + r * np.cos(phi)
    m = segment == 2
    y[m], z[m] = -2.35 - t[m], -0.24 - 0.32 * s[m]
    m = segment == 3
    d, psi = 0.34 - t[m], 0.5 * np.pi * s[m]
    y[m], z[m] = -2.69 + d * np.cos(psi), -0.56 - d * np.sin(psi)
    m = segment == 4
    y[m], z[m] = -2.69 - 0.16 * s[m], -0.9 + t[m]
    return y, z


class BlockAssembler:
    """Collects 27-node cells from block node arrays and merges nodes by position."""

    def __init__(self):
        self.index, self.points, self.cells = {}, [], []
        self.lattice = np.rint(2 * HEX27_REFERENCE).astype(int)

    def add(self, P):
        flat = P.reshape(-1, 3)
        ids = np.empty(len(flat), dtype=np.int64)
        for i, key in enumerate(map(tuple, np.rint(flat / 1e-7).astype(np.int64))):
            ids[i] = self.index.setdefault(key, len(self.points))
            if ids[i] == len(self.points):
                self.points.append(flat[i])
        ids = ids.reshape(P.shape[:3])
        L = self.lattice
        for i, j, k in product(*(range((n - 1) // 2) for n in P.shape[:3])):
            self.cells.append(ids[2 * i + L[:, 0], 2 * j + L[:, 1], 2 * k + L[:, 2]])


def build_package(level):
    (px, py, pz), (bx, by, bz), n_path = grid_lines(level)
    blocks = BlockAssembler()
    X, Y, Z = np.meshgrid(with_midpoints(bx), with_midpoints(by), with_midpoints(bz), indexing="ij")
    blocks.add(np.stack((X, Y, Z), axis=-1))
    X, Y, Z = np.meshgrid(with_midpoints(px), with_midpoints(py), with_midpoints(pz), indexing="ij")
    blocks.add(np.stack((X, Y * half_width(Z) / 1.95, Z), axis=-1))
    t_lines = pz[np.abs(pz) <= 0.1 + 1e-9] + 0.1
    u = np.concatenate([[0.0]] + [k + np.arange(1, n + 1) / n for k, n in enumerate(n_path)])
    for k in range(8):
        x0 = LEAD_X0 + PITCH * k
        lx = px[(px >= x0 - 1e-9) & (px <= x0 + LEAD_W + 1e-9)]
        U, XX, TT = np.meshgrid(with_midpoints(u), with_midpoints(lx), with_midpoints(t_lines), indexing="ij")
        yy, zz = lead_path(U, TT)
        for side in (-1, 1):
            blocks.add(np.stack((XX, side * yy, zz), axis=-1))
    coordinates = np.array(blocks.points) * mm
    cells = np.array(blocks.cells)
    # Positive orientation: mirror the reference x direction of left-handed cells.
    v = coordinates[cells[:, :8]]
    left = np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 3] - v[:, 0]), v[:, 4] - v[:, 0]) < 0
    ref = HEX27_REFERENCE
    mirror = [int(np.flatnonzero(np.all(np.isclose(ref, [1 - r[0], r[1], r[2]]), axis=1))[0]) for r in ref]
    cells[left] = cells[left][:, mirror]
    return coordinates, cells


def material_of(x, y, z):
    """Conductivity at physical points (interfaces lie on cell faces, quadrature points never on them)."""
    x, y, z = (np.asarray(v) / mm for v in (x, y, z))
    k = np.full(x.shape, K_PLASTIC)
    k = np.where(z < -0.9, K_FR4, k)
    k = np.where((z < -0.9) & (z > -1.0) & (x > -8) & (x < -4) & (y > -5) & (y < -1), K_CU, k)
    k = np.where((z > -0.9) & (np.abs(y) > 1.95), K_AL, k)
    strip = (np.abs(z) < T_IC) & (((x > -4.645) & (x < -4.245) & (y > -1.95) & (y < 0.2))
                                  | ((x > -4.245) & (x < -0.5) & (np.abs(y) < 0.2)))
    k = np.where(strip, K_CU, k)
    return np.where((np.abs(x) < 0.5) & (np.abs(y) < 0.5) & (np.abs(z) < 0.05), K_SI, k)


CENTRE = int(np.flatnonzero(np.all(HEX27_REFERENCE == 0.5, axis=1))[0])


class PackageMesh(StructuredHexahedralMesh):
    """Conforming 27-node hexahedral mesh with the two named boundary groups."""

    def __init__(self, geometry, level):
        coordinates, cells = build_package(level)
        data = QuadraticHexahedralGeometry(coordinates, cells)
        self.geometry, self.level = geometry, level
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (self.cell_count, 1, 1)
        self.mesh_id = "soic16:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        owners = self.face_owners = {}
        for c, cell in enumerate(self.cells):
            for side, face in enumerate(FACES):
                owners.setdefault(tuple(sorted(cell[list(face)])), []).append((c, side))
        self._facets = {"regulator": [], "air": []}
        self.exterior = {"regulator": [], "air": []}
        for matches in owners.values():
            if len(matches) == 1:
                c, side = matches[0]
                face = tuple(self.cells[c][list(FACES[side])])
                q = self.coordinates[list(face)] / mm
                on_regulator = np.allclose(q[:, 2], -0.9) and q[:, 0].max() < -8 + 1e-9 and q[:, 1].max() < -1 + 1e-9
                name = "regulator" if on_regulator else "air"
                self._facets[name].append(face)
                self.exterior[name].append((c, side))
        centres = self.coordinates[self.cells[:, CENTRE]]
        self.cell_k = material_of(*centres.T)
        self._carrier = self._build_carrier()

    def _build_carrier(self):
        carrier = super()._build_carrier()
        return replace(carrier, connectivity=replace(carrier.connectivity, entity_kind="hexahedron27"))




# ## Model
#
# One `HeatConduction` problem serves both element orders. The conductivity is a pointwise
# material law (the table above); the die source is a vectorised volumetric heating. Both are
# plain Python objects with an `evaluate` port.

class PackageMaterials:
    """Piecewise constant conductivity of the assembly (the steady problem needs no capacity)."""

    density, specific_heat_capacity, thermal_conductivity = 1000.0, 1000.0, "piecewise"

    def evaluate(self, name, *, x, y, z, **state):
        if name != "thermal_conductivity":
            return 1000.0
        return float(material_of(x, y, z))


class DieHeating:
    """Uniform source Q in the silicon die, zero elsewhere."""

    def evaluate(self, points, *, time=0.0):
        x, y, z = (np.abs(points[:, i]) / mm for i in range(3))
        return np.where((x < 0.5) & (y < 0.5) & (z < 0.05), Q_DIE, 0.0)


GAUSS3 = np.array([0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)])
W3 = np.array([5.0, 8.0, 5.0]) / 18.0


def face_lattice(side):
    """Indices of the 9 nodes of a cell face in a 3x3 (a, b) array, and its fixed axis."""
    axis, value = side // 2, 2 * (side % 2)
    L = np.rint(2 * HEX27_REFERENCE).astype(int)
    others = [a for a in range(3) if a != axis]
    grid = np.empty((3, 3), dtype=int)
    for n in np.flatnonzero(L[:, axis] == value):
        grid[L[n, others[0]], L[n, others[1]]] = n
    return grid


FACE_LATTICE = [face_lattice(side) for side in range(6)]


def lagrange3(s):
    return np.stack((2 * s * s - 3 * s + 1, 4 * s * (1 - s), 2 * s * s - s), axis=-1)


def lagrange3_d(s):
    return np.stack((4 * s - 3, 4 - 8 * s, 4 * s - 1), axis=-1)


def solve(level, order):
    start = time.perf_counter()
    geometry = PackageOnBoard()
    mesh = PackageMesh(geometry, level)
    model = HeatConduction(domain=geometry.domain(), material=PackageMaterials(), heating=DieHeating())
    b = geometry.boundaries
    problem = Problem(name="SOIC-16 package next to a voltage regulator", geometry=geometry, physics=(model,),
                      boundary_conditions=(
                          FixedTemperature(boundary=b.regulator, physics=model, value=T_reg),
                          ConvectiveHeatFlux(boundary=b.air, physics=model, heat_transfer_coefficient=h_air,
                                             ambient_temperature=T_air)))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=order)), linear_solver=SparseDirect())).solve_stationary()
    field = result.field(model.temperature)
    operator = result.operators[model.model_id]
    seconds = time.perf_counter() - start
    # Nodal temperatures on all 27 geometry nodes of every cell.
    values = np.asarray(field.values).ravel()
    dof_xyz = operator.space.dof_coordinates
    nodal = np.full(len(mesh.coordinates), np.nan)
    distance, node = cKDTree(mesh.coordinates).query(dof_xyz)
    nodal[node] = values
    if order == 1:   # the Q1 trace is bilinear: fill edge, face and cell midpoints from the corners
        L = np.rint(2 * HEX27_REFERENCE).astype(int) / 2.0
        weights = np.prod(np.where(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1],
                                             [1, 1, 1], [0, 1, 1]])[None, :, :] == 1,
                                   L[:, None, :], 1 - L[:, None, :]), axis=-1)
        nodal_cells = np.einsum("nc,kc->kn", weights, nodal[mesh.cells[:, :8]])
        nodal[mesh.cells.ravel()] = nodal_cells.ravel()
    # Die maximum: Q2/Q1 field sampled on a 21 x 21 x 5 grid through the die.
    g = np.linspace(-0.5, 0.5, 21) * mm
    die = np.array(np.meshgrid(g, g, np.linspace(-0.05, 0.05, 5) * mm, indexing="ij")).reshape(3, -1).T
    T_die = np.asarray(field.evaluate(die)).ravel()
    # Convected heat: 3 x 3 Gauss points on each curved (biquadratic) cooled face.
    q_out = 0.0
    for c, side in mesh.exterior["air"]:
        ids = mesh.cells[c][FACE_LATTICE[side]]
        P, T = mesh.coordinates[ids], nodal[ids]
        for i, a in enumerate(GAUSS3):
            for j, bb in enumerate(GAUSS3):
                la, lb, da, db = lagrange3(a), lagrange3(bb), lagrange3_d(a), lagrange3_d(bb)
                Ta = la @ T @ lb
                dA = np.linalg.norm(np.cross(np.einsum("a,b,abk->k", da, lb, P), np.einsum("a,b,abk->k", la, db, P)))
                q_out += W3[i] * W3[j] * dA * h_air * (Ta - T_air)
    return dict(level=level, order=order, mesh=mesh, field=field, operator=operator, nodal=nodal,
                dofs=operator.problem_size, T_die=float(T_die.max()) - 273.15, q_out=q_out, seconds=seconds)


def regulator_heat(run):
    """Heat entering through the regulator footprint: the residual of the unconstrained FE system at
    the footprint nodes (the consistent, conservative boundary flux)."""
    operator = run["operator"]
    base, system, rhs, constraints = operator.weak_assembly(0.0)
    u = np.zeros(operator.problem_size)
    u[np.asarray(operator.field_binding.indices).ravel()] = np.asarray(run["field"].values).ravel()
    residual = system.matrix.tocsr() @ u - np.asarray(rhs)
    return float(residual[[c.dof_index for c in constraints]].sum())


# ## Solve
#
# Quadratic and linear elements on the same three meshes ($\ell=0.5$, 0.7 and 1.0). The levels
# are not nested (the material interfaces fix many grid lines), so each mesh is characterised by
# its representative cell size $h=(V/N)^{1/3}$, as in the grid-convergence procedure of
# Celik et al. (*J. Fluids Eng.* 130, 078001, 2008).

LEVELS = {2: (0.5, 0.7, 1.0), 1: (0.5, 0.7, 1.0)}
runs = {2: [], 1: []}
for order, levels in LEVELS.items():
    for level in levels:
        run = solve(level, order)
        run["h"] = (run["mesh"].cell_volumes.sum() / run["mesh"].cell_count) ** (1 / 3)
        runs[order].append(run)
        print(f"Q{order} level {level}: {run['mesh'].cell_count:6d} cells, {run['dofs']:6d} unknowns, "
              f"h = {run['h'] / mm * 1e3:.0f} um, max die temperature {run['T_die']:.4f} degC, "
              f"convected {run['q_out'] * 1e3:.4f} mW  ({run['seconds']:.0f} s)")
for order in runs:
    runs[order][-1]["q_reg"] = regulator_heat(runs[order][-1])


# ## Comparison with the reference
#
# Finite elements are too stiff on coarse meshes: here that means too conductive, so both element
# orders approach the die temperature from below. The edges of the die and of the 5 µm copper strip,
# where the conductivity jumps by more than three orders of magnitude, are singular lines of the
# temperature gradient; they limit the observed order of convergence to well below the smooth-field
# value (3 for the quadratic elements). The heat convected to the air is dominated by the heat
# entering through the regulator footprint, whose flux is singular along the footprint edge.

def richardson(values, sizes):
    """Observed order and extrapolated value from three meshes with representative sizes h."""
    f1, f2, f3 = values
    h1, h2, h3 = sizes
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.2, 12.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


q2, q1 = runs[2][-1], runs[1][-1]
print("Maximum silicon temperature, finest mesh of each element order")
print(f"   published value                   {T_published:8.2f} degC")
for run in (q2, q1):
    print(f"   Q{run['order']} ({run['dofs']:6d} unknowns)          {run['T_die']:8.3f} degC   "
          f"deviation {run['T_die'] - T_published:+.3f} K ({100 * (run['T_die'] - T_published) / (T_published - 30.0):+.2f} % "
          f"of the temperature rise above the air)")

print("Mesh studies (representative cell sizes " + ", ".join(f"{r['h'] / mm * 1e3:.0f}" for r in runs[2]) + " um)")
def trend(values, sizes, unit, digits):
    """Observed order and Richardson value, or the reason why none is defined."""
    p, extrapolated = richardson(values, sizes)
    if np.isfinite(p):
        return f"observed order {p:.2f}, Richardson {extrapolated:.{digits}f} {unit}", extrapolated
    if np.diff(values).prod() < 0:
        return "not monotone, so no observed order", np.nan
    return "monotone but not yet in the asymptotic range (no observed order between 0.2 and 12)", np.nan


extrapolated_T = {}
for order in (2, 1):
    sizes = [r["h"] for r in runs[order]]
    vals = [r["T_die"] for r in runs[order]]
    text, extrapolated_T[order] = trend(vals, sizes, "degC", 3)
    print(f"   Q{order} maximum die temperature: " + "  ".join(f"{v:.4f}" for v in vals) + f" degC; {text}; "
          f"change between the last two meshes {abs(vals[-1] - vals[-2]):.3f} K")
    heat = [r["q_out"] * 1e3 for r in runs[order]]
    text, _ = trend(heat, sizes, "mW", 3)
    print(f"   Q{order} heat convected to the air: " + "  ".join(f"{v:.3f}" for v in heat) + f" mW; {text}; "
          f"change between the last two meshes {100 * abs(heat[-1] / heat[-2] - 1):.2f} %")
reference_T = extrapolated_T[2] if np.isfinite(extrapolated_T[2]) else q2["T_die"]
print(f"   published value {T_published} degC against the Q2 estimate {reference_T:.2f} degC: "
      f"{T_published - reference_T:+.2f} K ({100 * (T_published - reference_T) / (reference_T - 30.0):+.1f} % "
      f"of the temperature rise)")

print("Energy balance on the finest mesh (heat in = heat out)")
for run in (q2, q1):
    heat_in = run["q_reg"] + P_DIE
    print(f"   Q{run['order']}: regulator footprint {run['q_reg'] * 1e3:.4f} mW + die {P_DIE * 1e3:.1f} mW = "
          f"{heat_in * 1e3:.4f} mW; convected to the air {run['q_out'] * 1e3:.4f} mW; "
          f"relative difference {abs(heat_in - run['q_out']) / run['q_out']:.1e}")


# **Thermal-resistance network** (an independent estimate by hand). Heat leaves the die
# through the plastic above and below it: a column that spreads at 45° from the $1\times1$ mm
# die over the 0.7 mm to the package faces has $R=t/(k\,a\,(a+2t))$; each face then loses it by
# convection over its area. The interconnect is a $5$ µm copper strip of 0.4 mm width and
# 2.15 + 3.75 mm length, $R=\ell/(k\,d\,w)$, connecting the die to lead 1, which sits on the
# ground plane at nearly 50 °C. The die node then follows from
# $T_j=(T_\infty/R_a+T_\mathrm{reg}/R_i+P)/(1/R_a+1/R_i)$. The network ignores conduction
# into the other leads and the board, so it is expected to over-estimate the die temperature
# by some kelvin.

a_die, t_mold = 1.0 * mm, 0.70 * mm
R_spread = t_mold / (K_PLASTIC * a_die * (a_die + 2 * t_mold))
A_face = 9.9 * mm * 2 * 1.835 * mm                      # mean area of the top (or bottom) face
R_side = R_spread + 1.0 / (h_air * A_face)
R_air = R_side / 2                                     # top and bottom in parallel
R_ic = (2.15 * mm + 3.745 * mm) / (K_CU * 2 * T_IC * mm * LEAD_W * mm)
T_network = (T_air / R_air + T_reg / R_ic + P_DIE) / (1 / R_air + 1 / R_ic) - 273.15
print(f"Network: R_spread = {R_spread:.0f} K/W per face, R_die-air = {R_air:.0f} K/W, "
      f"R_interconnect = {R_ic:.0f} K/W")
print(f"   die temperature, network {T_network:.2f} degC; Q2 {q2['T_die']:.2f} degC; "
      f"difference {T_network - q2['T_die']:+.2f} K")


# ## Figure
#
# Top: the surface temperature of the whole assembly, and the same view with the plastic body
# removed (die, interconnect, leads and board). Bottom: the section through lead 1 and the
# interconnect riser ($x=-4.445$ mm), the section $y=0$ through the die and the interconnect
# strip, the temperature on the lower face of the die, and the mesh convergence.

fine = runs[2][-1]
mesh, nodal = fine["mesh"], fine["nodal"] - 273.15
X = mesh.coordinates / mm
LATTICE = np.rint(2 * HEX27_REFERENCE).astype(int)
cells_lattice = np.empty((mesh.cell_count, 3, 3, 3), dtype=int)
cells_lattice[:, LATTICE[:, 0], LATTICE[:, 1], LATTICE[:, 2]] = mesh.cells
norm = colors.Normalize(vmin=30.0, vmax=50.0)


def quads_of(grid):
    """Four sub-quadrilaterals of a 3x3 node patch."""
    return [grid[i:i + 2, j:j + 2].ravel()[[0, 1, 3, 2]] for i in range(2) for j in range(2)]


def face_polygons(faces):
    polys, vals = [], []
    for c, side in faces:
        for quad in quads_of(mesh.cells[c][FACE_LATTICE[side]]):
            polys.append(X[quad])
            vals.append(nodal[quad].mean())
    return np.array(polys), np.array(vals)


def section(axis, value):
    """Triangulated node patches of all cells that have a lattice plane on coordinate axis = value."""
    tris = []
    for d in range(3):
        for i in range(3):
            patch = np.take(cells_lattice, i, axis=d + 1)
            on = np.all(np.abs(X[patch][..., axis] - value) < 1e-6, axis=(1, 2))
            for grid in patch[on]:
                for q in quads_of(grid):
                    tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    tris = np.array(tris)
    used = np.unique(tris)
    local = np.full(len(X), -1)
    local[used] = np.arange(len(used))
    others = [a for a in range(3) if a != axis]
    return Triangulation(X[used, others[0]], X[used, others[1]], local[tris]), nodal[used]


def draw_3d(ax, polys, vals, title):
    shade = FIELD_CMAP(norm(vals))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=shade, edgecolors=shade, linewidths=0.2))
    ax.set(xlim=(-10, 10), ylim=(-5, 5), zlim=(-1.9, 0.9), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm",
           title=title)
    ax.set_box_aspect((20, 10, 2.8 * 1.5), zoom=1.25)
    ax.view_init(elev=30, azim=-60)


fig = plt.figure(figsize=(22, 12), constrained_layout=True)
grid = fig.add_gridspec(2, 4, height_ratios=(1.15, 1.0))
ax = fig.add_subplot(grid[0, :2], projection="3d")
draw_3d(ax, *face_polygons(mesh.exterior["air"] + mesh.exterior["regulator"]),
        f"Surface temperature (Q2, {fine['dofs']} unknowns)")
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax, shrink=0.55, pad=0.15, label="T / degC")

xray = []
for matches in mesh.face_owners.values():
    solid = [(c, side) for c, side in matches if mesh.cell_k[c] != K_PLASTIC]
    if len(solid) == 1 and (len(matches) == 1 or mesh.cell_k[matches[0][0]] != mesh.cell_k[matches[-1][0]]):
        xray.append(solid[0])
ax = fig.add_subplot(grid[0, 2:], projection="3d")
draw_3d(ax, *face_polygons(xray), "Plastic body removed: die, interconnect, leads, board")
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax, shrink=0.55, pad=0.15, label="T / degC")

ax = fig.add_subplot(grid[1, 2])
g = np.linspace(-0.5, 0.5, 61)
GX, GY = np.meshgrid(g, g, indexing="ij")
bottom = np.column_stack((GX.ravel(), GY.ravel(), np.full(GX.size, -0.05))) * mm
TB = np.asarray(fine["field"].evaluate(bottom)).reshape(GX.shape) - 273.15
cs = ax.contourf(GX, GY, TB, levels=40, cmap=FIELD_CMAP)
ax.contour(GX, GY, TB, levels=10, colors=CREAM, linewidths=0.5)
ax.set(aspect="equal", xlabel="x / mm", ylabel="y / mm",
       title=f"Lower face of the die (max {TB.max():.2f} degC)")
fig.colorbar(cs, ax=ax, shrink=0.8, label="T / degC")

ax = fig.add_subplot(grid[1, 0])
tri, vals = section(0, -4.445)
cs = ax.tricontourf(tri, vals, levels=np.linspace(30, 50, 41), cmap=FIELD_CMAP)
ax.tricontour(tri, vals, levels=np.arange(31, 50, 1.0), colors=CREAM, linewidths=0.4)
ax.set(xlim=(-5, 0.6), ylim=(-1.9, 0.8), aspect="equal", xlabel="y / mm", ylabel="z / mm",
       title="Section x = -4.445 mm through lead 1")
fig.colorbar(cs, ax=ax, shrink=0.7, label="T / degC")

ax = fig.add_subplot(grid[1, 1])
tri, vals = section(1, 0.0)
cs = ax.tricontourf(tri, vals, levels=np.linspace(30, 50, 41), cmap=FIELD_CMAP)
ax.tricontour(tri, vals, levels=np.arange(31, 50, 1.0), colors=CREAM, linewidths=0.4)
ax.add_patch(plt.Rectangle((-0.5, -0.05), 1.0, 0.1, fill=False, ec=INK, lw=0.8))
ax.set(xlim=(-6, 2), ylim=(-1.9, 0.8), aspect="equal", xlabel="x / mm", ylabel="z / mm",
       title="Section y = 0 through the die and the strip")
fig.colorbar(cs, ax=ax, shrink=0.7, label="T / degC")

ax = fig.add_subplot(grid[1, 3])
for order, marker, color in ((2, "s", GREEN), (1, "o", COPPER)):
    hs = [r["h"] / mm * 1e3 for r in runs[order]]
    ax.plot(hs, [r["T_die"] for r in runs[order]], marker + "-", color=color, label=f"Q{order} elements")
ax.axhline(T_published, color=INK, lw=0.8, ls="--", label=f"published value {T_published} degC")
if np.isfinite(extrapolated_T[2]):
    ax.axhline(extrapolated_T[2], color=GREEN, lw=0.8, ls="-.", label=f"Q2 Richardson estimate {extrapolated_T[2]:.2f} degC")
ax.set(xlabel="representative cell size (V/N)^(1/3) / um", ylabel="maximum die temperature / degC",
       title="Mesh convergence of the die temperature")
ax.invert_xaxis()
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Raise the die power or move the regulator closer and watch the share of the regulator heat
# in the die temperature; replace the aluminium leads by copper alloy ($k\approx260$ W/(m K))
# or the plastic by a filled moulding compound ($k\approx0.8$ W/(m K)); add the shell model
# of the copper layers as a second variant and compare it with the resolved 5 µm layer; or make
# the problem transient (`solve_transient`) to see how fast the die follows a power step.
