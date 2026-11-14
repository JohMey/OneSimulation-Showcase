"""OneSimulation example: thermal stress in a fuel-cell bipolar plate heated at start-up.
Requires the installed, activated OneSimulation runtime.

A titanium bipolar plate of a proton-exchange-membrane fuel-cell stack is warmed by an aluminium
heating element in its gas-feed channel, cooled by the gas in the feed slits and by the
surroundings, and clamped by the tie rods. The steady temperature and the thermal stress are
computed in 3D on one eighth of the plate with quadratic 27-node hexahedra, and checked against the
exact homogeneous expansion-and-clamping solution, the heat balance, the published
temperature, stress and displacement levels and a three-level mesh study.
"""


# # Thermal stress in a fuel-cell bipolar plate
#
# A PEM fuel cell runs just below 100 °C, so a cold stack has to be heated before it can start.
# In the stack considered here every bipolar plate carries a heating element in the middle of
# its gas-feed channel; together the elements of the 25 plates deliver 200 W. The plates are
# titanium, the heating elements aluminium. Heating makes the plate non-uniformly hot, and the
# thermal expansion that cannot develop freely produces stress. The question is how large the
# stress is, where it peaks, and how far the plate bulges out of plane, which matters for the
# sealing of the membrane.
#
# **Geometry.** The plate is $240\times180\times10$ mm with two gas-feed slits, tie-rod holes and
# a central active area. With its three symmetry planes one eighth is modelled,
# $0\le x\le120$ mm, $0\le y\le90$ mm, $0\le z\le5$ mm (half the thickness), with the symmetry
# planes $x=120$ mm, $y=90$ mm and the mid-plane $z=0$:
#
# | feature | position (mm) |
# |---|---|
# | active part (titanium, clamped with 1 MPa) | $30\le x\le120$, $35\le y\le90$ |
# | manifold (titanium, clamped with 8 MPa) | the rest of the plate |
# | gas slit 1 | $10\le x\le20$, $35\le y\le70$, corner fillets $R=2.5$ |
# | gas slit 2 | $35\le x\le100$, $10\le y\le20$, corner fillets $R=2.5$ |
# | tie-rod hole, $R_0=5$ | centre $(15,15)$; half holes at $(15,90)$ and $(120,15)$ |
# | heating element (aluminium), half disc $R_0=5$ | centre $(70,90)$ on the symmetry plane |
#
# All fields depend on all three coordinates; the plate is thin, so the dependence on $z$ is
# weak but present (the faces of the slits are cooled, the top face is not).
#
# **Heat transfer** (steady, $-\nabla\cdot(k\nabla T)=Q$). The heating element dissipates
# $200\,\text{W}/25$ per plate, i.e. $2$ W in the modelled eighth, uniformly over its volume.
#
# | face | thermal condition |
# |---|---|
# | outer edges $x=0$ and $y=0$ | convection to the surroundings, $h_1=5$ W/(m² K), $T_\infty=20$ °C |
# | walls of the two gas slits | convection to the feed gas, $h_2=50$ W/(m² K), $T_\mathrm{gas}=80$ °C |
# | tie-rod holes | insulated (rods not modelled) |
# | top face $z=5$ mm | insulated (contact with the neighbouring cell, which is heated alike) |
# | symmetry planes $x=120$, $y=90$, $z=0$ | zero flux |
#
# **Structure** (linear elasticity of the titanium only; the soft-mounted heating element carries
# no load). Thermal strain $\varepsilon_\mathrm{th}=\alpha\,(T-T_\mathrm{ref})\,\mathbf I$ with
# the stress-free reference temperature $T_\mathrm{ref}=0$ °C.
#
# | face | mechanical condition |
# |---|---|
# | symmetry plane $x=120$ mm | $u_x=0$ (roller) |
# | symmetry plane $y=90$ mm | $u_y=0$ (roller) |
# | mid-plane $z=0$ | $u_z=0$ (roller) |
# | top face of the active part | tie-rod clamping pressure $P_1=1$ MPa |
# | top face of the manifold | tie-rod clamping pressure $P_2=8$ MPa |
# | outer edges, slit walls, holes, seat of the heating element | traction-free |
#
# **Materials** (values of the published problem set-up): titanium $E=105$ GPa, $\nu=0.33$,
# $\alpha=7.06\times10^{-6}$ 1/K, $k=7.5$ W/(m K); aluminium $k=160$ W/(m K). Densities and heat
# capacities do not enter the steady problem.
#
# **Numerical method.**
#
# * Spatial: continuous Galerkin finite elements with **27-node quadratic (Q2) Lagrange
#   hexahedra** for both the temperature and the displacement, on an unstructured all-hexahedral
#   mesh: the plate section is triangulated with a size field (DistMesh algorithm, P.-O. Persson
#   and G. Strang, SIAM Review 46, 329, 2004), every triangle is split into three quadrilaterals
#   through its centroid and edge midpoints, boundary midpoints are placed on the exact circular
#   arcs, and the quadrilaterals are swept through the thickness. Gauss quadrature with
#   $3\times3\times3$ points.
# * Coupling: one-way staggered. The temperature is solved first; the solid takes it at its own
#   quadrature points through finite-element evaluation (`FieldCoupling`). The displacement does
#   not feed back into the heat problem, so one pass is the exact coupled solution.
# * Solvers: both problems are linear and stationary; the heat problem is solved with a sparse
#   direct LU factorisation, the elastic problem with conjugate gradients preconditioned by
#   algebraic multigrid (relative residual $10^{-10}$).
#
# **Reference.** The published model gives no numbers in its text, only colour plots; their
# ranges (temperature, von Mises stress, out-of-plane displacement) are compared with
# a tolerance that reflects reading a colour bar. The quantitative checks are:
#
# * the **exact homogeneous solution**: at a uniform temperature $T_0$ and a uniform clamping
#   pressure $P$ on the whole top face, the plate with these supports expands freely in its plane
#   and is compressed through its thickness,
#   $$\mathbf u=\alpha(T_0-T_\mathrm{ref})\begin{pmatrix}x-120\,\text{mm}\\y-90\,\text{mm}\\z\end{pmatrix}
#     +\frac{P}{E}\begin{pmatrix}\nu\,(x-120\,\text{mm})\\\nu\,(y-90\,\text{mm})\\-z\end{pmatrix},\qquad
#     \boldsymbol\sigma=-P\,\mathbf e_z\otimes\mathbf e_z ,$$
#   exactly, holes and slits included (their walls carry no load because the in-plane stress
#   vanishes). It is solved on the real 3D mesh with the real supports; the quadratic hexahedra
#   contain this linear field, so it must be reproduced to round-off;
# * the **heat balance**: the 2 W of the heating element must leave through the cooled faces;
# * the classical **fully restrained bound** $E\alpha\,(T_\max-T_\min)/(1-\nu)$ for the thermal
#   stress (B. A. Boley and J. H. Weiner, *Theory of Thermal Stresses*, Wiley 1960, ch. 9), which
#   the free plate must stay well below;
# * a three-level mesh study of the peak temperature, the peak von Mises stress and the peak
#   out-of-plane displacement, with observed order and Richardson estimate.

from dataclasses import dataclass
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
from matplotlib import cm, colors
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, cKDTree

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES, trilinear_hexahedral_geometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.field_transfer import FiniteElementEvaluation
from onesim.numerics.linear import AlgebraicMultigrid, ConjugateGradient, SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.physics.solid_mechanics import (
    FixedDisplacementComponents, IsotropicThermalExpansion, LinearElasticity, Traction, TractionFree,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling

print("OneSimulation:", version("onesimulation"))


# ## Parameters

mm = 1e-3
LX, LY, TH2 = 120.0, 90.0, 5.0              # mm: modelled eighth, half thickness
R0, FILLET = 5.0, 2.5                       # mm: holes and heating element, slit fillets
HOLES = ((15.0, 15.0), (15.0, 90.0), (120.0, 15.0))
SLITS = ((10.0, 20.0, 35.0, 70.0), (35.0, 100.0, 10.0, 20.0))      # x0, x1, y0, y1 in mm
HEATER = (70.0, 90.0)
P_heater = 200.0 / 25 / 4                   # W in the modelled eighth (one quarter of a plate)
h_surr, T_surr = 5.0, 293.15                # outer edges
h_gas, T_gas = 50.0, 353.15                 # gas slits
k_ti, k_al = 7.5, 160.0                     # W/(m K)
E_ti, nu_ti, alpha_ti = 105e9, 0.33, 7.06e-6
T_ref = 273.15                              # stress-free temperature
P_active, P_manifold = 1e6, 8e6             # Pa, tie-rod clamping
LEVELS = ((14.0, 3.5), (10.0, 2.5), (7.0, 1.75))   # (far, near) triangle sizes in mm of the three meshes
published = {"T_max": 436.0, "T_min": 367.0, "vm_max": 33.0, "w_max": 6.3}   # read from the colour bars


# ## Geometry and mesh
#
# The plate section is described by signed distance functions (rectangle minus discs and
# rounded slits). It is split into three regions that are meshed separately and share their
# boundary nodes: the manifold, the active part and the half disc of the heating element. Points
# on every boundary and interface are placed first, spaced by a size function that is
# $h_\mathrm{near}$ on the holes, slits and heating element and grows with 0.3 times the distance
# up to $h_\mathrm{far}$. Each region is then filled by the DistMesh force-equilibrium iteration
# with these points fixed. The triangles are split into three quadrilaterals each (corner, two
# edge midpoints, centroid); the midpoints of boundary edges on the circular arcs are moved onto
# the arcs. The quadrilaterals are swept through the half thickness into one layer of 27-node
# hexahedra (quadratic in all three directions). The temperature mesh contains the heating
# element; the elastic mesh is the same mesh without it.


def d_rect(p, x0, x1, y0, y1):
    return -np.minimum(np.minimum(p[:, 1] - y0, y1 - p[:, 1]), np.minimum(p[:, 0] - x0, x1 - p[:, 0]))


def d_circle(p, c, r):
    return np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - r


def d_slit(p, x0, x1, y0, y1, r=FILLET):
    q = np.abs(p - ((x0 + x1) / 2, (y0 + y1) / 2)) - ((x1 - x0) / 2 - r, (y1 - y0) / 2 - r)
    return np.hypot(np.maximum(q[:, 0], 0), np.maximum(q[:, 1], 0)) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0) - r


def d_plate(p):
    d = d_rect(p, 0, LX, 0, LY)
    for c in HOLES:
        d = np.maximum(d, -d_circle(p, c, R0))
    for s in SLITS:
        d = np.maximum(d, -d_slit(p, *s))
    return d


def d_heater(p):
    return np.maximum(d_circle(p, HEATER, R0), p[:, 1] - LY)


def d_active(p):
    return np.maximum(d_rect(p, 30, LX, 35, LY), -d_heater(p))


def d_manifold(p):
    return np.maximum(d_plate(p), -d_rect(p, 30, LX + 1, 35, LY + 1))


def size_field(p, h_far, h_near):
    d = np.full(len(p), np.inf)
    for c in HOLES + (HEATER,):
        d = np.minimum(d, np.abs(d_circle(p, c, R0)))
    for s in SLITS:
        d = np.minimum(d, np.abs(d_slit(p, *s)))
    return np.minimum(h_far, h_near + 0.3 * d)


def arc(c, r, a0, a1, n=20000):
    a = np.linspace(a0, a1, n)
    return np.column_stack((c[0] + r * np.cos(a), c[1] + r * np.sin(a)))


def boundary_points(h_far, h_near):
    def resample(P):
        s = np.concatenate(([0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))))
        fine = np.linspace(0, s[-1], 40000)
        Q = np.column_stack((np.interp(fine, s, P[:, 0]), np.interp(fine, s, P[:, 1])))
        h = size_field(Q, h_far, h_near)
        n = np.concatenate(([0], np.cumsum(np.diff(fine) / (0.5 * (h[1:] + h[:-1])))))
        t = np.interp(np.linspace(0, n[-1], max(1, int(np.ceil(n[-1]))) + 1), n, fine)
        return np.column_stack((np.interp(t, fine, Q[:, 0]), np.interp(t, fine, Q[:, 1])))

    lines = [[(0, 0), (LX, 0)], [(LX, 0), (LX, 10)], [(LX, 20), (LX, 35)], [(LX, 35), (LX, LY)],
             [(LX, LY), (75, LY)], [(75, LY), (65, LY)], [(65, LY), (30, LY)], [(30, LY), (20, LY)],
             [(10, LY), (0, LY)], [(0, LY), (0, 0)], [(30, LY), (30, 35)], [(30, 35), (LX, 35)]]
    pieces = [resample(np.array(P, float)) for P in lines]
    pieces += [resample(arc(HOLES[0], R0, 0, 2 * np.pi)), resample(arc(HOLES[1], R0, np.pi, 2 * np.pi)),
               resample(arc(HOLES[2], R0, np.pi / 2, 1.5 * np.pi)), resample(arc(HEATER, R0, np.pi, 2 * np.pi))]
    for x0, x1, y0, y1 in SLITS:
        r = FILLET
        pieces.append(resample(np.vstack([
            arc((x1 - r, y0 + r), r, -np.pi / 2, 0), arc((x1 - r, y1 - r), r, 0, np.pi / 2),
            arc((x0 + r, y1 - r), r, np.pi / 2, np.pi), arc((x0 + r, y0 + r), r, np.pi, 1.5 * np.pi),
            [[x1 - r, y0]]])))
    P = np.vstack(pieces)
    _, first = np.unique(np.round(P * 1e6).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(first)]


def distmesh(fd, fixed, box, h_far, h_near, iterations=250):
    """DistMesh (Persson and Strang 2004) with fixed boundary points."""
    (x0, y0), (x1, y1) = box
    X, Y = np.meshgrid(np.arange(x0, x1 + h_near, h_near), np.arange(y0, y1 + h_near, h_near * np.sqrt(3) / 2))
    X[1::2] += h_near / 2
    p = np.column_stack((X.ravel(), Y.ravel()))
    p = p[fd(p) < -0.3 * h_near]
    keep = np.random.default_rng(0).random(len(p)) < (h_near / size_field(p, h_far, h_near)) ** 2
    p = p[keep]
    p = p[cKDTree(fixed).query(p)[0] > 0.5 * size_field(p, h_far, h_near)]
    nf = len(fixed)
    p = np.vstack((fixed, p))
    for it in range(iterations):
        tri = Delaunay(p).simplices
        tri = tri[fd(p[tri].mean(axis=1)) < -1e-3 * h_near]
        bars = np.unique(np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1), axis=0)
        vec = p[bars[:, 0]] - p[bars[:, 1]]
        length = np.hypot(*vec.T)
        h = size_field(0.5 * (p[bars[:, 0]] + p[bars[:, 1]]), h_far, h_near)
        target = h * 1.2 * np.sqrt(np.sum(length**2) / np.sum(h**2))
        force = (np.maximum(target - length, 0) / length)[:, None] * vec
        total = np.zeros_like(p)
        np.add.at(total, bars[:, 0], force)
        np.add.at(total, bars[:, 1], -force)
        total[:nf] = 0
        p = p + 0.2 * total
        d = fd(p)
        out = d > 0
        if np.any(out):
            e = 1e-7 * h_near
            gx = (fd(p[out] + [e, 0]) - d[out]) / e
            gy = (fd(p[out] + [0, e]) - d[out]) / e
            p[out] -= (d[out] / (gx**2 + gy**2))[:, None] * np.column_stack((gx, gy))
        p[:nf] = fixed
        if it > 30 and np.max(np.hypot(*(0.2 * total[d < -1e-3 * h_near]).T), initial=0) < 1e-4 * h_near:
            break
    p = p[(np.arange(len(p)) < nf) | (fd(p) < -0.3 * size_field(p, h_far, h_near))]
    tri = Delaunay(p).simplices
    tri = tri[fd(p[tri].mean(axis=1)) < -1e-3 * h_near]
    a = p[tri]
    ccw = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0]) > 0
    tri[~ccw] = tri[~ccw][:, [0, 2, 1]]
    return p, tri


def plate_section(h_far, h_near):
    """Triangles of the three regions (0 manifold, 1 active part, 2 heating element) with shared nodes."""
    fixed = boundary_points(h_far, h_near)
    points, index, triangles, region = [], {}, [], []
    boxes = (((0, 0), (LX, LY)), ((30, 35), (LX, LY)), ((65, 85), (75, LY)))
    for k, (fd, box) in enumerate(zip((d_manifold, d_active, d_heater), boxes)):
        p, t = distmesh(fd, fixed[np.abs(fd(fixed)) < 1e-6], box, h_far, h_near)
        ids = []
        for q in p:
            key = (round(q[0] * 1e6), round(q[1] * 1e6))
            if key not in index:
                index[key] = len(points)
                points.append(q)
            ids.append(index[key])
        triangles.append(np.array(ids)[t])
        region += [k] * len(t)
    return np.array(points), np.vstack(triangles), np.array(region)


def split_into_quads(p, t, region):
    """Three quadrilaterals per triangle; boundary midpoints on arcs are moved onto the arcs."""
    circles = [(np.array(c), R0) for c in HOLES + (HEATER,)]
    for x0, x1, y0, y1 in SLITS:
        circles += [(np.array(c), FILLET) for c in ((x1 - FILLET, y0 + FILLET), (x1 - FILLET, y1 - FILLET),
                                                    (x0 + FILLET, y1 - FILLET), (x0 + FILLET, y0 + FILLET))]
    edges = np.sort(np.vstack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = set(map(tuple, unique[counts == 1]))
    points, middle = list(p), {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in middle:
            m = 0.5 * (p[a] + p[b])
            for c, r in circles:
                on_arc = abs(np.hypot(*(p[a] - c)) - r) < 1e-6 and abs(np.hypot(*(p[b] - c)) - r) < 1e-6
                if on_arc and np.hypot(*(p[a] - p[b])) < r and (key in boundary or r == R0 and tuple(c) == HEATER):
                    m = c + r * (m - c) / np.hypot(*(m - c))
                    break
            else:
                if key in boundary:            # chord across a tangent point: Newton steps onto the boundary
                    for _ in range(3):
                        d, e = d_plate(m[None])[0], 1e-7
                        g = np.array([d_plate((m + (e, 0))[None])[0] - d, d_plate((m + (0, e))[None])[0] - d]) / e
                        m = m - d * g / (g @ g)
            middle[key] = len(points)
            points.append(m)
        return middle[key]

    quads, quad_region = [], []
    for (a, b, c), r in zip(t, region):
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        g = len(points)
        points.append((p[a] + p[b] + p[c]) / 3)
        quads += [(a, ab, g, ca), (b, bc, g, ab), (c, ca, g, bc)]
        quad_region += [r] * 3
    return np.array(points), np.array(quads), np.array(quad_region)


class HexahedralMesh(StructuredHexahedralMesh):
    """Unstructured trilinear hexahedra with named boundary quadrilaterals."""

    def __init__(self, geometry, coordinates, cells, facets):
        data = trilinear_hexahedral_geometry(np.asarray(coordinates, float), np.asarray(cells))
        self.geometry = geometry
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (self.cell_count, 1, 1)
        self.mesh_id = "plate-hex:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        self._facets = {name: list(faces) for name, faces in facets.items()}
        self._carrier = self._build_carrier()

    def to_mapping(self):
        return {**super().to_mapping(), "generator": "plate-hex", "geometry_mapping": "trilinear_hex8"}


@dataclass(frozen=True, eq=False, kw_only=True)
class PlateForHeat(Geometry):
    dimension = 3

    def boundary_groups(self):
        names = ("outer_edges", "gas_slits", "tie_rod_holes", "top", "symmetry")
        return {**{name: (name,) for name in names}, "all": names}


@dataclass(frozen=True, eq=False, kw_only=True)
class PlateForStress(Geometry):
    dimension = 3

    def boundary_groups(self):
        names = ("symmetry_x", "symmetry_y", "mid_plane", "top_active", "top_manifold", "free")
        return {**{name: (name,) for name in names}, "all": names}


def face_kind(P):
    """Boundary kind of a quadrilateral face from its corner coordinates in mm."""
    x, y, z = P.T
    if np.all(np.abs(z) < 1e-9):
        return "mid_plane"
    if np.all(np.abs(z - TH2) < 1e-9):
        return "top"
    if np.all(np.abs(x - LX) < 1e-9):
        return "symmetry_x"
    if np.all(np.abs(y - LY) < 1e-9):
        return "symmetry_y"
    if np.all(np.abs(x) < 1e-9) or np.all(np.abs(y) < 1e-9):
        return "outer_edges"
    if any(np.all(np.abs(d_slit(P[:, :2], *s)) < 2e-3) for s in SLITS):
        return "gas_slits"
    if np.all(np.abs(d_circle(P[:, :2], HEATER, R0)) < 1e-6):
        return "heater_seat"
    if any(np.all(np.abs(d_circle(P[:, :2], c, R0)) < 1e-6) for c in HOLES):
        return "tie_rod_holes"
    raise ValueError("unclassified boundary face")


HEAT_NAMES = {"mid_plane": "symmetry", "symmetry_x": "symmetry", "symmetry_y": "symmetry"}
STRESS_NAMES = {"outer_edges": "free", "gas_slits": "free", "tie_rod_holes": "free", "heater_seat": "free"}


def build_plate(h_far, h_near, layers=1):
    p, t, region = plate_section(h_far, h_near)
    xy, quads, quad_region = split_into_quads(p, t, region)
    n2 = len(xy)
    zs = np.linspace(0.0, TH2, layers + 1)
    coords = np.column_stack((np.tile(xy[:, 0], layers + 1), np.tile(xy[:, 1], layers + 1), np.repeat(zs, n2)))
    cells = np.vstack([np.column_stack((quads + k * n2, quads + (k + 1) * n2)) for k in range(layers)])
    cell_region = np.tile(quad_region, layers)
    out = dict(xy=xy, quads=quads, quad_region=quad_region, triangles=len(t), layers=layers, h=h_near)
    for kind, keep, geometry in (("heat", np.ones(len(cells), bool), PlateForHeat()),
                                 ("stress", cell_region != 2, PlateForStress())):
        kept = cells[keep]
        used = np.unique(kept)
        renumber = np.full(len(coords), -1)
        renumber[used] = np.arange(len(used))
        X, kept, kept_region = coords[used], renumber[kept], cell_region[keep]
        groups = {name: [] for name in geometry.boundary_groups()["all"]}
        owners = {}
        for c, cell in enumerate(kept):
            for face in FACES:
                corners = tuple(cell[list(face)])
                owners.setdefault(tuple(sorted(corners)), []).append((corners, c))
        for faces in owners.values():
            if len(faces) == 1:
                corners, c = faces[0]
                name = face_kind(X[list(corners)])
                if kind == "heat":
                    name = HEAT_NAMES.get(name, name)
                else:
                    name = STRESS_NAMES.get(name, name)
                    if name == "top":
                        name = "top_active" if kept_region[c] == 1 else "top_manifold"
                groups[name].append(corners)
        out[kind] = dict(geometry=geometry, coords=X * mm, cells=kept, region=kept_region,
                         groups={name: np.array(faces) for name, faces in groups.items()},
                         mesh=HexahedralMesh(geometry, X * mm, kept, groups))
    out["heater_volume"] = float(out["heat"]["mesh"].cell_volumes[out["heat"]["region"] == 2].sum())
    section_tri = Triangulation(xy[:, 0], xy[:, 1], np.vstack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]])))
    out["section_tri"], out["finder"] = section_tri, section_tri.get_trifinder()
    out["tri_region"] = np.concatenate((quad_region, quad_region))
    return out


meshes = [build_plate(h_far, h_near) for h_far, h_near in LEVELS]
for (h_far, h_near), data in zip(LEVELS, meshes):
    print(f"h = {h_far:4.1f}/{h_near:4.2f} mm: {data['triangles']:5d} triangles -> {len(data['quads']):5d} quadrilaterals, "
          f"{data['heat']['mesh'].cell_count:5d} Q2 hexahedra ({len(data['heat']['coords'])} vertices)")


# ## Model
#
# The conductivity and the heat source are assigned per cell region: a point belongs to the
# heating element if it lies in one of its quadrilaterals. The source density is the 2 W of the
# eighth divided by the meshed volume of the half disc, so the meshed heater delivers exactly
# the prescribed power.

titanium = IsotropicSolid(density=4940.0, specific_heat_capacity=710.0, thermal_conductivity=k_ti,
                          youngs_modulus=E_ti, poisson_ratio=nu_ti, thermal_expansion_coefficient=alpha_ti)


def in_heater(data, x, y):
    tri = data["finder"](np.ravel(x) / mm, np.ravel(y) / mm)
    return (tri >= 0) & (data["tri_region"][np.maximum(tri, 0)] == 2)


class PlateConductivity:
    """Titanium plate with the aluminium heating element (steady problem: capacity irrelevant)."""

    density, specific_heat_capacity, thermal_conductivity = 4940.0, 710.0, "by region"

    def __init__(self, data):
        self.data = data

    def evaluate(self, name, *, x, y, z, **state):
        if name != "thermal_conductivity":
            return 1000.0
        return float(np.where(in_heater(self.data, x, y), k_al, k_ti)[0])


class HeaterSource:
    def __init__(self, data):
        self.data, self.q = data, P_heater / data["heater_volume"]

    def evaluate(self, points, *, time=0.0):
        return np.where(in_heater(self.data, points[:, 0], points[:, 1]), self.q, 0.0)


def solve_heat(data):
    mesh_data = data["heat"]
    b = mesh_data["geometry"].boundaries
    heat = HeatConduction(domain=mesh_data["geometry"].domain(), material=PlateConductivity(data),
                          heating=HeaterSource(data))
    problem = Problem(name="Bipolar plate heating", geometry=mesh_data["geometry"], physics=(heat,), boundary_conditions=(
        ConvectiveHeatFlux(boundary=b.outer_edges, physics=heat, heat_transfer_coefficient=h_surr, ambient_temperature=T_surr),
        ConvectiveHeatFlux(boundary=b.gas_slits, physics=heat, heat_transfer_coefficient=h_gas, ambient_temperature=T_gas),
        HeatFlux(boundary=b.tie_rod_holes | b.top | b.symmetry, physics=heat, value=0.0)))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2), quadrature_order=4), linear_solver=SparseDirect())
    return Simulation(problem=problem, mesh=mesh_data["mesh"], numerics=numerics).solve_stationary().field(heat.temperature)


@dataclass(frozen=True)
class UniformExpansion(IsotropicThermalExpansion):
    """Thermal strain of a uniform temperature T0, whatever the coupled field: the free-expansion check."""

    uniform_temperature: float = 373.15

    def evaluate(self, *, temperature, coefficient):
        return super().evaluate(temperature=self.uniform_temperature, coefficient=coefficient)

    def evaluate_array(self, *, temperature, coefficient):
        return super().evaluate_array(temperature=np.full(np.shape(temperature), self.uniform_temperature),
                                      coefficient=coefficient)


def solve_stress(data, temperature, pressures=(P_active, P_manifold), strain=None):
    """Elastic titanium plate under the coupled temperature field and the clamping pressures."""
    mesh_data = data["stress"]
    b = mesh_data["geometry"].boundaries
    strain = IsotropicThermalExpansion(reference_temperature=T_ref) if strain is None else strain
    solid = LinearElasticity(domain=mesh_data["geometry"].domain(), material=titanium, thermal_strain=strain)
    couplings = (FieldCoupling(source=temperature, target=solid.temperature, transfer=FiniteElementEvaluation()),)
    problem = Problem(name="Bipolar plate thermal stress", geometry=mesh_data["geometry"], physics=(solid,), boundary_conditions=(
        FixedDisplacementComponents(boundary=b.symmetry_x, physics=solid, components=("x",)),
        FixedDisplacementComponents(boundary=b.symmetry_y, physics=solid, components=("y",)),
        FixedDisplacementComponents(boundary=b.mid_plane, physics=solid, components=("z",)),
        Traction(boundary=b.top_active, physics=solid, value=(0.0, 0.0, -pressures[0])),
        Traction(boundary=b.top_manifold, physics=solid, value=(0.0, 0.0, -pressures[1])),
        TractionFree(boundary=b.free, physics=solid)), couplings=couplings)
    solver = ConjugateGradient(preconditioner=AlgebraicMultigrid(), relative_tolerance=1e-10, maximum_iterations=3000)
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2), quadrature_order=4), linear_solver=solver)
    result = Simulation(problem=problem, mesh=mesh_data["mesh"], numerics=numerics).solve_stationary()
    return result.field(solid.displacement), result.field(solid.von_mises_stress), result.field(solid.stress)


GAUSS = np.array([0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)])
GAUSS_W = np.array([5.0, 8.0, 5.0]) / 18.0


def face_quadrature(coords, faces):
    """3x3 Gauss points and weights on bilinear boundary quadrilaterals."""
    P = coords[faces]                                           # (n, 4, 3)
    s, t = np.meshgrid(GAUSS, GAUSS, indexing="ij")
    s, t, w = s.ravel(), t.ravel(), np.outer(GAUSS_W, GAUSS_W).ravel()
    N = np.stack(((1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t), axis=1)
    dNs = np.stack((-(1 - t), 1 - t, t, -t), axis=1)
    dNt = np.stack((-(1 - s), -s, s, 1 - s), axis=1)
    points = np.einsum("qk,fkd->fqd", N, P)
    jac = np.linalg.norm(np.cross(np.einsum("qk,fkd->fqd", dNs, P), np.einsum("qk,fkd->fqd", dNt, P)), axis=2)
    return points.reshape(-1, 3), (jac * w).ravel()


def evaluate(field, points):
    return np.asarray(field.evaluate(np.asarray(points, float)))


def heat_out(data, temperature):
    heat = data["heat"]
    rates = {}
    for name, h, T_inf in (("outer_edges", h_surr, T_surr), ("gas_slits", h_gas, T_gas)):
        points, weights = face_quadrature(heat["coords"], heat["groups"][name])
        rates[name] = float(np.sum(weights * h * (evaluate(temperature, points).ravel() - T_inf)))
    return rates


def edge_samples(n=400):
    """Points just inside the material along every hole, slit and the heater seat, at three depths."""
    eps = 1e-3
    curves = [arc(HOLES[0], R0 + eps, 0, 2 * np.pi, n), arc(HOLES[1], R0 + eps, np.pi, 2 * np.pi, n // 2),
              arc(HOLES[2], R0 + eps, np.pi / 2, 1.5 * np.pi, n // 2), arc(HEATER, R0 + eps, np.pi, 2 * np.pi, n // 2)]
    for x0, x1, y0, y1 in SLITS:
        r = FILLET + eps
        corners = ((x1 - FILLET, y0 + FILLET), (x1 - FILLET, y1 - FILLET), (x0 + FILLET, y1 - FILLET), (x0 + FILLET, y0 + FILLET))
        ring = np.vstack([arc(c, r, a, a + np.pi / 2, n // 8) for c, a in zip(corners, (-np.pi / 2, 0, np.pi / 2, np.pi))])
        ring = np.vstack((ring, ring[:1]))
        s_len = np.concatenate(([0], np.cumsum(np.hypot(*np.diff(ring, axis=0).T))))
        t = np.linspace(0, s_len[-1], 3 * n)
        curves.append(np.column_stack((np.interp(t, s_len, ring[:, 0]), np.interp(t, s_len, ring[:, 1]))))
    xy = np.vstack(curves)
    xy = xy[(xy[:, 0] < LX) & (xy[:, 1] < LY)]
    return np.vstack([np.column_stack((xy, np.full(len(xy), z))) for z in (1e-6, TH2 / 2, TH2 - 1e-6)]) * mm


EDGES_XYZ = edge_samples()


def measure(data, temperature, displacement, von_mises):
    heat, stress = data["heat"], data["stress"]
    T = evaluate(temperature, heat["coords"]).ravel()
    u = evaluate(displacement, stress["coords"])
    vm = evaluate(von_mises, stress["coords"]).ravel()
    tri = data["finder"](EDGES_XYZ[:, 0] / mm, EDGES_XYZ[:, 1] / mm)
    edges = EDGES_XYZ[(tri >= 0) & (data["tri_region"][np.maximum(tri, 0)] != 2)]      # inside the elastic mesh
    vm_edges = evaluate(von_mises, edges).ravel()
    samples = np.vstack((stress["coords"], edges))
    vm_all = np.concatenate((vm, vm_edges))
    top = np.abs(stress["coords"][:, 2] - TH2 * mm) < 1e-12
    rates = heat_out(data, temperature)
    return dict(T=T, u=u, vm=vm, T_max=T.max(), T_min=T.min(), vm_max=vm_all.max(),
                vm_where=samples[vm_all.argmax()] / mm, w_max=u[top, 2].max(),
                q_out=rates, h=data["h"])


# ## Solve
#
# Three meshes; on each the temperature is solved first and then the stress with the coupled
# temperature field. On the coarsest mesh one more stress problem serves as the exact check:
# a uniform temperature of 100 °C with the manifold pressure of 8 MPa on the whole top face.
# Only the fields of the finest mesh are kept for the figure.

T_uniform = 373.15
runs = []
for data in meshes:
    start = time.perf_counter()
    temperature = solve_heat(data)
    t_heat = time.perf_counter() - start
    displacement, von_mises, stress = solve_stress(data, temperature)
    run = measure(data, temperature, displacement, von_mises)
    print(f"h_near = {data['h']:4.2f} mm ({data['heat']['mesh'].cell_count:5d} hexahedra): T = {run['T_min']:.2f} ... "
          f"{run['T_max']:.2f} K, peak von Mises {run['vm_max'] / 1e6:6.3f} MPa, w_max = {run['w_max'] * 1e6:.4f} um  "
          f"[heat {t_heat:.0f} s, total {time.perf_counter() - start:.0f} s]")
    if not runs:
        start = time.perf_counter()
        u_hom, vm_hom, s_hom = solve_stress(data, temperature, pressures=(P_manifold, P_manifold),
                                            strain=UniformExpansion(reference_temperature=T_ref, uniform_temperature=T_uniform))
        X = data["stress"]["coords"]
        hom = dict(X=X, u=evaluate(u_hom, X), s=evaluate(s_hom, X), vm=evaluate(vm_hom, X).ravel())
        del u_hom, vm_hom, s_hom
        print(f"homogeneous check problem on the coarsest mesh: {time.perf_counter() - start:.0f} s")
    if data is meshes[-1]:
        run.update(data=data, temperature=temperature, stress=stress)
    runs.append(run)
    del temperature, displacement, von_mises, stress
fine = runs[-1]


# ## Comparison with the reference
#
# **Exact homogeneous solution** (coarsest mesh, uniform 100 °C, uniform 8 MPa clamping).

X = hom["X"]
strain0, P = alpha_ti * (T_uniform - T_ref), P_manifold
r = np.column_stack((X[:, 0] - LX * mm, X[:, 1] - LY * mm, X[:, 2]))
u_exact = strain0 * r + P / E_ti * r * np.array([nu_ti, nu_ti, -1.0])
u_num, s_num, vm = hom["u"], hom["s"], hom["vm"]
err_u = np.max(np.linalg.norm(u_num - u_exact, axis=1)) / np.max(np.linalg.norm(u_exact, axis=1))
err_s = np.max(np.abs(s_num - np.array([0, 0, -P, 0, 0, 0])), axis=0) / P
print(f"homogeneous solution: max |u - u_exact| / max |u_exact| = {err_u:.1e}")
print(f"  max |sigma - sigma_exact| / P per component (xx, yy, zz, xy, yz, xz): " + ", ".join(f"{e:.1e}" for e in err_s))
print(f"  von Mises: {vm.min() / 1e6:.6f} ... {vm.max() / 1e6:.6f} MPa (exact {P / 1e6:.1f} MPa)")

# **Heat balance.** The power leaving through the gas slits and the outer edges must equal the
# 2 W of the heating element in the modelled eighth. The Galerkin solution satisfies the global
# balance exactly (the cooled faces are integrated exactly with 3x3 Gauss points), so the
# deviation is at round-off level on every mesh; the split between slits and edges converges.

print("\nheat balance           " + "".join(f"  h = {r['h']:.2f} mm" for r in runs))
for name in ("gas_slits", "outer_edges"):
    print(f"  {name:20s} " + "".join(f"  {r['q_out'][name]:9.5f} W" for r in runs))
print(f"  {'total (ref 2 W)':20s} " + "".join(f"  {sum(r['q_out'].values()):9.5f} W" for r in runs))
print(f"  {'relative deviation':20s} " + "".join(f"  {sum(r['q_out'].values()) / P_heater - 1:+10.1e}" for r in runs))

# **Published levels and the restrained bound** (finest mesh). The published model shows the
# temperature, the von Mises stress and the out-of-plane displacement as colour plots; the
# values below are read from their colour bars (about +-1 K, +-1 MPa, +-0.2 um).

bound = E_ti * alpha_ti * (fine["T_max"] - fine["T_min"]) / (1 - nu_ti)
print(f"\nquantity                     computed    published   deviation")
print(f"  T_max [K]                  {fine['T_max']:9.2f}   {published['T_max']:9.1f}   {fine['T_max'] / published['T_max'] - 1:+.1%}")
print(f"  T_min [K]                  {fine['T_min']:9.2f}   {published['T_min']:9.1f}   {fine['T_min'] / published['T_min'] - 1:+.1%}")
print(f"  peak von Mises [MPa]       {fine['vm_max'] / 1e6:9.2f}   {published['vm_max']:9.1f}   {fine['vm_max'] / 1e6 / published['vm_max'] - 1:+.1%}"
      f"   at (x, y, z) = ({fine['vm_where'][0]:.1f}, {fine['vm_where'][1]:.1f}, {fine['vm_where'][2]:.1f}) mm")
print(f"  w_max [um]                 {fine['w_max'] * 1e6:9.3f}   {published['w_max']:9.1f}   {fine['w_max'] * 1e6 / published['w_max'] - 1:+.1%}")
print(f"  restrained bound E alpha (T_max - T_min)/(1 - nu) = {bound / 1e6:.1f} MPa; peak / bound = {fine['vm_max'] / bound:.2f}")

# **Mesh study.** The cell size at the holes, slits and heater seat shrinks by a factor 1.4 per
# level (and the far-field size with it). Observed order and Richardson estimate from the three
# meshes; the peak von Mises stress is the maximum over all mesh vertices and over dense samples
# along every hole, slit and the heater seat at three depths.


def richardson(values, hs):
    f1, f2, f3 = values
    ratio = (f1 - f2) / (f2 - f3)
    lo, hi = 0.2, 12.0
    for _ in range(200):
        p = 0.5 * (lo + hi)
        if (hs[0] ** p - hs[1] ** p) / (hs[1] ** p - hs[2] ** p) > ratio:
            hi = p
        else:
            lo = p
    return p, f3 + (f3 - f2) * hs[2] ** p / (hs[1] ** p - hs[2] ** p)


hs = np.array([r["h"] for r in runs])
print("\nmesh study                 " + "".join(f"  h = {h:.2f} mm" for h in hs) + "   change last two   order   Richardson")
for key, label, scale in (("T_max", "T_max [K]", 1.0), ("vm_max", "peak von Mises [MPa]", 1e-6), ("w_max", "w_max [um]", 1e6)):
    values = np.array([r[key] for r in runs])
    monotone = (values[1] - values[0]) * (values[2] - values[1]) > 0
    p, rich = richardson(values, hs) if monotone else (np.nan, np.nan)
    if 0.5 < p < 6.0:
        tail = f"   {p:5.2f}   {rich * scale:10.4f}"
    else:
        tail = "   (changes at round-off/mesh-noise level: no asymptotic order)"
    print(f"  {label:24s} " + "".join(f"  {v * scale:10.4f}" for v in values) + f"   {abs(values[2] / values[1] - 1):12.1e}" + tail)


# ## Figure
#
# Top: temperature and von Mises stress on the surface of the whole plate (the computed eighth
# mirrored about its two in-plane symmetry planes; the upper half of the thickness is shown).
# Bottom left: out-of-plane displacement of the top face of the modelled eighth. Bottom middle:
# temperature and in-plane stresses along the line $x=70$ mm from the seat of the heating
# element down to gas slit 2, mid-thickness. Bottom right: mesh study.

heat_mesh, stress_mesh = fine["data"]["heat"], fine["data"]["stress"]


def surface(mesh_data, values, names):
    faces = np.vstack([mesh_data["groups"][n] for n in names if len(mesh_data["groups"][n])])
    quads = mesh_data["coords"][faces] / mm
    face_values = values[faces].mean(axis=1)
    copies, copy_values = [], []
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        q = quads.copy()
        if sx < 0:
            q[:, :, 0] = 2 * LX - q[:, :, 0]
        if sy < 0:
            q[:, :, 1] = 2 * LY - q[:, :, 1]
        copies.append(q)
        copy_values.append(face_values)
    return np.concatenate(copies), np.concatenate(copy_values)


fig = plt.figure(figsize=(17, 12.5), constrained_layout=True)
grid = fig.add_gridspec(2, 6, height_ratios=(1.2, 1.0))
panels = ((heat_mesh, fine["T"], ("top", "outer_edges", "gas_slits", "tie_rod_holes"), "temperature T [K]", 1.0),
          (stress_mesh, fine["vm"], ("top_active", "top_manifold", "free"), "von Mises stress [MPa]", 1e-6))
for col, (mesh_data, values, names, label, scale) in enumerate(panels):
    ax = fig.add_subplot(grid[0, 3 * col:3 * col + 3], projection="3d")
    polys, face_values = surface(mesh_data, values * scale, names)
    norm = colors.Normalize(vmin=face_values.min(), vmax=face_values.max())
    shades = FIELD_CMAP(norm(face_values))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=shades, edgecolors=shades, linewidths=0.3))
    ax.set(xlim=(0, 2 * LX), ylim=(0, 2 * LY), zlim=(0, TH2), xlabel="x [mm]", ylabel="y [mm]")
    ax.set_zticks([0, 5])
    ax.set_box_aspect((2 * LX, 2 * LY, 12))
    ax.view_init(elev=38, azim=-62)
    ax.set_title(("Temperature of the heated plate" if col == 0 else "von Mises stress"), pad=4)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax, shrink=0.5, pad=0.06, label=label)

ax = fig.add_subplot(grid[1, 0:2])
X = stress_mesh["coords"] / mm
top = np.abs(X[:, 2] - TH2) < 1e-9
tri_top = Triangulation(X[top, 0], X[top, 1])
w_top = fine["u"][top, 2] * 1e6
mask = in_heater(fine["data"], X[top, 0][tri_top.triangles].mean(axis=1) * mm, X[top, 1][tri_top.triangles].mean(axis=1) * mm)
mask |= d_plate(np.column_stack((X[top, 0][tri_top.triangles].mean(axis=1), X[top, 1][tri_top.triangles].mean(axis=1)))) > 0
tri_top.set_mask(mask)
shade = ax.tricontourf(tri_top, w_top, levels=30, cmap=FIELD_CMAP)
ax.tricontour(tri_top, w_top, levels=10, colors=INK, linewidths=0.4)
ax.set_aspect("equal")
ax.set(xlabel="x [mm]", ylabel="y [mm]", title="Out-of-plane displacement of the top face")
fig.colorbar(shade, ax=ax, shrink=0.8, label="w [um]")

ax = fig.add_subplot(grid[1, 2:4])
y_line = np.linspace(85.0 - 1e-6, 20.0 + 1e-6, 200)
line = np.column_stack((np.full(200, 70.0), y_line, np.full(200, 2.5))) * mm
T_line = evaluate(fine["temperature"], line).ravel()
s_line = evaluate(fine["stress"], line)
ax.plot(90 - y_line, s_line[:, 0] / 1e6, color=COPPER, lw=1.6, label="sigma_xx (hoop at the seat)")
ax.plot(90 - y_line, s_line[:, 1] / 1e6, color="#5F9175", lw=1.6, label="sigma_yy")
ax.plot(90 - y_line, s_line[:, 2] / 1e6, color=INK, lw=1.0, ls="--", label="sigma_zz (clamping 1 / 8 MPa)")
ax.axhline(0, color="#8B9A8D", lw=0.7)
ax.set(xlabel="distance from the heater centre, 90 - y [mm]", ylabel="stress [MPa]",
       title="Line x = 70 mm, mid-thickness")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(90 - y_line, T_line, color=GREEN, lw=1.8)
ax2.set_ylabel("T [K]", color=GREEN)

ax = fig.add_subplot(grid[1, 4:6])
for key, label, colour in (("T_max", "T_max - 273.15 K", GREEN), ("vm_max", "peak von Mises", COPPER), ("w_max", "w_max", INK)):
    values = np.array([r[key] for r in runs])
    if key == "T_max":
        values = values - 273.15
    ax.plot(hs, values / values[-1], "o-", color=colour, label=label)
ax.axhline(1.0, color="#8B9A8D", lw=0.8, ls=":")
ax.invert_xaxis()
ax.set(xlabel="cell size at the features h [mm]", ylabel="value / value on the finest mesh", title="Mesh study")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.show()


# ## Going further
#
# Make the heating transient: with the densities and heat capacities of the table the plate
# needs several minutes to approach the steady state, and the stress peak during warm-up can
# exceed the steady value because the heater seat is then much hotter than the rest of the plate.
# A second variant replaces the free sliding of the tie-rod holes by contact with the rods, which
# adds a constraint at the holes and raises the stress there.
