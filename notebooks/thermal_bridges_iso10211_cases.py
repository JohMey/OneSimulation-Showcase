"""OneSimulation example: 3D thermal bridge of a balcony slab between two floors (ISO 10211 case 3).
Requires the installed, activated OneSimulation runtime.

The steady 3D temperature field of the third test reference case of EN ISO 10211 - an
external building corner whose floor slab runs through the insulated wall and continues
outside as a balcony - is computed with cubic hexahedral finite elements (64-node Q3
Lagrange elements) on graded block meshes. All tabulated reference values of the standard
(heat flows, coupling coefficients, minimum surface temperatures and temperature weighting
factors) are graded against its tolerances, supported by an h-study with the cubic
elements, an element-order comparison (Q1, Q2, Q3 on the same mesh) and the energy balance.
"""


# # Thermal bridge in 3D: a balcony slab between two heated floors
#
# EN ISO 10211 (*Thermal bridges in building construction - Heat flows and surface
# temperatures - Detailed calculations*) validates numerical methods with four test
# reference cases; a program that reproduces all of them is classed as a
# *three-dimensional steady-state high-precision method*. Cases 1 and 2 are
# two-dimensional, cases 3 and 4 three-dimensional. This notebook computes **case 3**, a
# three-dimensional external corner of a building between two floors:
#
# - two external walls meet at a corner; each wall consists of an exterior leaf (brick),
#   an insulation layer and an interior leaf (aerated concrete);
# - the concrete floor slab between the lower and the upper room cuts through the
#   interior leaf and the insulation of one wall, crosses the brick leaf and continues
#   outside as a **balcony**, 0.6 m deep - the classical slab-edge thermal bridge;
# - a thin floor finish lies on the slab in the upper room.
#
# Coordinates (m): the outer corner of the building is the origin, the external faces
# are $x=0$ and $y=0$, the balcony sticks out to $y<0$, $z$ is vertical. The model ends at
# the cut planes $x=1.3$, $y=1.3$, $z=0$ and $z=2.15$.
#
# | no. | part | extent / m | $\lambda$ / (W/(m K)) |
# |---|---|---|---|
# | 3 | exterior leaf (brick) | wall region with $x<0.10$ or $y<0.10$ | 1.0 |
# | 2 | insulation | wall region with $0.10\le\min(x,y)<0.15$ | 0.04 |
# | 1 | interior leaf (aerated concrete) | wall region with $0.15\le\min(x,y)<0.30$ | 0.7 |
# | 4 | floor slab and balcony (concrete) | $0.15\le x\le1.3,\ -0.6\le y\le1.3,\ 1.0\le z\le1.15$ | 2.5 |
# | 5 | floor finish | $0.3\le x\le1.3,\ 0.3\le y\le1.3,\ 1.15\le z\le1.2$ | 1.0 |
#
# The wall region is $\{0\le x\le 1.3,\ 0\le y\le 1.3,\ \min(x,y)<0.3,\ 0\le z\le 2.15\}$;
# the slab replaces the wall materials where they overlap. The temperature solves
# $\nabla\cdot(\lambda\nabla T)=0$ with surface resistances $R_s$ (Robin condition
# $-\lambda\,\partial_nT=(T-\theta)/R_s$) towards three environments. Boundary conditions
# on every face:
#
# | face | environment | condition |
# |---|---|---|
# | $\alpha$: lower room - inner wall faces $x=0.3$ / $y=0.3$ below the slab and the slab soffit $z=1.0$ | lower room, $\theta_\alpha=20$ °C | $R_{s}=0.20$ m² K/W ($h=5$ W/(m² K)) |
# | $\beta$: upper room - inner wall faces above the floor and the floor surface $z=1.2$ | upper room, $\theta_\beta=15$ °C | $R_{s}=0.20$ m² K/W ($h=5$ W/(m² K)) |
# | $\gamma$: outside - external wall faces $x=0$, $y=0$ and all faces of the balcony | outside, $\theta_\gamma=0$ °C | $R_{s}=0.05$ m² K/W ($h=20$ W/(m² K)) |
# | $\delta$: the cut planes $x=1.3$, $y=1.3$, $z=0$, $z=2.15$ | - | adiabatic (zero flux) |
#
# The field is genuinely three-dimensional: it depends on the depth into each wall, on
# the distance from the corner and on the height relative to the slab.
#
# **Numerical method.**
# - Spatial: continuous Galerkin finite elements with **cubic tensor-product Lagrange
#   elements (Q3, 64 nodes per hexahedron)** on a block mesh that is aligned with every
#   material interface and geometrically graded towards the slab, the inner corner and
#   the balcony root. For comparison, the same problem is solved with Q1 (8-node) and Q2
#   (27-node) elements on the finest mesh.
# - Steady linear problem, sparse direct LU solver.
# - Because the problem is linear, a second solve with $\theta_\alpha=1$, $\theta_\beta=\theta_\gamma=0$
#   gives the thermal coupling coefficients $L_{ij}$ and the temperature weighting factors
#   $g$ of the standard by superposition.
#
# **Reference.** The test reference values of EN ISO 10211 (editions 2007 and 2017,
# normative annex on the validation of calculation methods), case 3: heat flows $\Phi_\alpha=46.09$ W, $\Phi_\beta=13.89$ W,
# $\Phi_\gamma=59.98$ W, coupling coefficients $L_{\alpha\gamma}=1.781$,
# $L_{\beta\gamma}=1.624$, $L_{\alpha\beta}=2.094$ W/K, minimum surface temperatures
# $\theta_{\min,\alpha}=11.32$ °C and $\theta_{\min,\beta}=11.11$ °C with their weighting
# factors. The standard requires heat flows within 1 % and temperatures within 0.1 K.
# Supporting checks: the energy balance and a three-level h-study of the cubic elements.

from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import gc
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES, affine_hexahedral_geometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters

LAMBDA = {"brick": 1.0, "insulation": 0.04, "aerated": 0.7, "slab": 2.5, "finish": 1.0}   # W/(m K)
THETA = {"alpha": 20.0, "beta": 15.0, "gamma": 0.0}       # degC
H_S = {"alpha": 1 / 0.20, "beta": 1 / 0.20, "gamma": 1 / 0.05}   # W/(m^2 K) from R_s
X_END, Y_BAL, Z_TOP = 1.3, -0.6, 2.15                      # cut planes, balcony front, wall top
Z_SLAB, Z_FLOOR, Z_FINISH = 1.0, 1.15, 1.2                 # slab soffit, slab top, floor surface
REF = {"Phi_alpha": 46.09, "Phi_beta": 13.89, "Phi_gamma": 59.98,          # W
       "L_ag": 1.781, "L_bg": 1.624, "L_ab": 2.094,                       # W/K
       "T_alpha": 11.32, "T_beta": 11.11,                                 # degC
       "g_alpha": (0.378, 0.399, 0.223), "g_beta": (0.331, 0.214, 0.455)}  # (gamma, alpha, beta) weights
TOL_PHI, TOL_T = 0.01, 0.1                                 # 1 % and 0.1 K (EN ISO 10211)

# One-dimensional U-values of the undisturbed wall (between the surface resistances) as
# orientation: lower room to outside and upper room to outside.
R_wall = 0.10 / LAMBDA["brick"] + 0.05 / LAMBDA["insulation"] + 0.15 / LAMBDA["aerated"]
U_wall = 1 / (0.20 + R_wall + 0.05)
print(f"undisturbed wall: R = {R_wall:.4f} m^2 K/W, U = {U_wall:.4f} W/(m^2 K)")


# ## Geometry and mesh
#
# A tensor-product block mesh is laid over the bounding box; its planes contain every
# material interface ($x,y\in\{0,0.10,0.15,0.30\}$, $y=-0.6$, $z\in\{1.0,1.15,1.2\}$), and
# the hexahedra outside the construction (the two rooms and the air around the balcony)
# are dropped. Each layer gets several cells (brick 2, insulation 2, interior leaf 3,
# slab 3, floor finish 2 on the finest mesh), and the cells in the rooms and in the
# balcony are geometrically graded from 20 mm next to the wall, the slab and the balcony
# root. A mesh level $\ell$ scales the cell counts by $\ell$ and the first cell width by
# $1/\ell$. The hexahedra are handed to OneSimulation as a native hexahedral mesh with the
# four named boundary groups $\alpha,\beta,\gamma,\delta$.

@dataclass(frozen=True, eq=False, kw_only=True)
class CornerConstruction(Geometry):
    """External corner with balcony slab of ISO 10211 case 3."""

    dimension = 3

    def boundary_groups(self):
        names = ("alpha", "beta", "gamma", "delta")
        return {**{name: (name,) for name in names}, "all": names}


def graded(a, b, n, first, towards="a"):
    """n cells on [a, b], the cell at the `towards` end of width `first`, geometric growth."""
    length = b - a
    if n == 1 or n * first >= length:
        s = np.linspace(0.0, length, n + 1)
    else:
        r = brentq(lambda r: first * (r**n - 1) / (r - 1) - length, 1 + 1e-9, 50.0)
        s = np.concatenate(([0.0], np.cumsum(first * r ** np.arange(n))))
    if towards == "b":
        s = s[-1] - s[::-1]
    out = a + s * length / s[-1]
    out[0], out[-1] = a, b
    return out


def axes_for(level):
    c = lambda n: max(1, int(np.floor(n * level + 0.5)))
    h0 = 0.02 / level
    walls = lambda: [graded(0.0, 0.10, c(2), h0), graded(0.10, 0.15, c(2), h0),
                     graded(0.15, 0.30, c(3), h0, "b"), graded(0.30, X_END, c(5), h0)]
    xs = np.unique(np.concatenate(walls()))
    ys = np.unique(np.concatenate([graded(Y_BAL, 0.0, c(4), h0, "b")] + walls()))
    zs = np.unique(np.concatenate([graded(0.0, Z_SLAB, c(5), h0, "b"), graded(Z_SLAB, Z_FLOOR, c(3), h0),
                                   graded(Z_FLOOR, Z_FINISH, c(2), h0), graded(Z_FINISH, Z_TOP, c(5), h0)]))
    return xs, ys, zs


def in_slab(x, y, z):
    return (x > 0.15) & (y > Y_BAL) & (z > Z_SLAB) & (z < Z_FLOOR)


def in_finish(x, y, z):
    return (x > 0.30) & (y > 0.30) & (z > Z_FLOOR) & (z < Z_FINISH)


def in_wall(x, y, z):
    return (np.minimum(x, y) < 0.30) & (x > 0.0) & (y > 0.0)


def in_solid(x, y, z):
    return in_slab(x, y, z) | in_finish(x, y, z) | in_wall(x, y, z)


def face_group(q):
    """Environment of a boundary face from its four corner points."""
    x, y, z = q.mean(axis=0)
    if (np.allclose(q[:, 2], 0.0) or np.allclose(q[:, 2], Z_TOP)
            or np.allclose(q[:, 0], X_END) or np.allclose(q[:, 1], X_END)):
        return "delta"
    if np.allclose(q[:, 0], 0.0) or np.allclose(q[:, 1], 0.0) or y < 0.0:
        return "gamma"
    return "alpha" if z <= Z_SLAB + 1e-12 else "beta"


class CornerMesh(StructuredHexahedralMesh):
    """Hex8 block mesh of the construction with the boundary groups alpha, beta, gamma, delta."""

    def __init__(self, geometry, xs, ys, zs):
        self.geometry = geometry
        centres = np.meshgrid(*(0.5 * (v[1:] + v[:-1]) for v in (xs, ys, zs)), indexing="ij")
        i, j, k = np.nonzero(in_solid(*centres))
        corners = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
        grid = np.stack([np.stack((i + a, j + b, k + c), axis=1) for a, b, c in corners], axis=1)
        shape = (len(xs), len(ys), len(zs))
        used, cells = np.unique(np.ravel_multi_index(grid.reshape(-1, 3).T, shape), return_inverse=True)
        gi = np.unravel_index(used, shape)
        data = affine_hexahedral_geometry(np.column_stack((xs[gi[0]], ys[gi[1]], zs[gi[2]])), cells.reshape(-1, 8))
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (len(xs) - 1, len(ys) - 1, len(zs) - 1)
        self.mesh_id = "iso10211-case3:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        owners = {}
        for cell in self.cells:
            for face in FACES:
                owners.setdefault(tuple(sorted(cell[list(face)])), []).append(tuple(cell[list(face)]))
        self._facets = {name: [] for name in geometry.boundary_groups()["all"]}
        for found in owners.values():
            if len(found) == 1:
                self._facets[face_group(self.coordinates[list(found[0])])].append(found[0])
        self._carrier = self._build_carrier()


LEVELS = (0.5, 0.75, 1.0)
geometry = CornerConstruction()
meshes = {level: CornerMesh(geometry, *axes_for(level)) for level in LEVELS}
for level, mesh in meshes.items():
    spans = {name: np.ptp(mesh.coordinates[np.array(mesh._facets[name])], axis=1) for name in ("alpha", "beta", "gamma")}
    area = {name: float(np.sum(np.prod(np.where(d > 1e-12, d, 1.0), axis=1))) for name, d in spans.items()}
    print(f"level {level}: {mesh.cell_count} hexahedra, volume {mesh.cell_volumes.sum():.5f} m^3, surface areas "
          + ", ".join(f"{name} {a:.4f} m^2" for name, a in area.items()))


# ## Model
#
# The conductivity is a position-dependent material law built from the table above.
# Every quadrature point of the elements lies strictly inside one hexahedron and every
# hexahedron lies inside one material, so each element sees exactly one conductivity.

@dataclass(frozen=True)
class CaseThreeConductivity:
    unit: str = "W/m/K"

    def evaluate(self, *, x, y, z, **state):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        depth = np.minimum(x, y)
        k = np.where(depth < 0.10, LAMBDA["brick"], np.where(depth < 0.15, LAMBDA["insulation"], LAMBDA["aerated"]))
        k = np.where(in_finish(x, y, z), LAMBDA["finish"], k)
        return np.where(in_slab(x, y, z), LAMBDA["slab"], k)


material = replace(IsotropicSolid(density=1500.0, specific_heat_capacity=1000.0, thermal_conductivity=1.0,
                                  youngs_modulus=1e9, poisson_ratio=0.3, thermal_expansion_coefficient=0.0),
                   thermal_conductivity=CaseThreeConductivity())

GAUSS, GAUSS_W = np.polynomial.legendre.leggauss(6)
GAUSS, GAUSS_W = 0.5 * (GAUSS + 1.0), 0.5 * GAUSS_W


def face_points(mesh, name, s, t):
    """Points at local coordinates (s, t) on every face of a group, and the face sizes."""
    pts, sizes = [], []
    S, T = np.meshgrid(s, t, indexing="ij")
    for face in mesh._facets[name]:
        q = mesh.coordinates[list(face)]
        lo, hi = q.min(axis=0), q.max(axis=0)
        u, v = [a for a in range(3) if hi[a] - lo[a] > 1e-12]
        p = np.repeat(lo[None], S.size, axis=0)
        p[:, u] += (hi[u] - lo[u]) * S.ravel()
        p[:, v] += (hi[v] - lo[v]) * T.ravel()
        pts.append(p)
        sizes.append((hi[u] - lo[u]) * (hi[v] - lo[v]))
    return np.array(pts), np.array(sizes)


def solve(level, order, theta, keep_field=False):
    """One steady solve; returns heat flows and surface temperatures (and the field if asked)."""
    start = time.time()
    mesh = meshes[level]
    model = HeatConduction(domain=geometry.domain(), material=material, heating=0.0)
    b = geometry.boundaries
    problem = Problem(
        name="ISO 10211 case 3", geometry=geometry, physics=(model,),
        boundary_conditions=tuple(
            ConvectiveHeatFlux(boundary=getattr(b, name), physics=model, heat_transfer_coefficient=H_S[name],
                               ambient_temperature=273.15 + theta[name]) for name in ("alpha", "beta", "gamma"))
        + (HeatFlux(boundary=b.delta, physics=model, value=0.0),))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=order)), linear_solver=SparseDirect())).solve_stationary()
    field = result.field(model.temperature)
    out = dict(level=level, order=order, cells=mesh.cell_count,
               unknowns=result.operators[model.model_id].problem_size)
    if keep_field:
        out["field"] = field
    W = np.outer(GAUSS_W, GAUSS_W).ravel()
    nodes = np.linspace(0.0, 1.0, 2 * order + 1)
    for name in ("alpha", "beta", "gamma"):
        # heat flow into the construction: h (theta - T) integrated with 6 x 6 Gauss points per face
        pts, sizes = face_points(mesh, name, GAUSS, GAUSS)
        T = np.asarray(field.evaluate(pts.reshape(-1, 3))).reshape(len(sizes), -1) - 273.15
        out["Phi_" + name] = float(np.sum(sizes[:, None] * W * H_S[name] * (theta[name] - T)))
        # lowest surface temperature: element nodes and edge midpoints of every face
        pts, _ = face_points(mesh, name, nodes, nodes)
        T = np.asarray(field.evaluate(pts.reshape(-1, 3))).ravel() - 273.15
        out["T_" + name] = float(T.min())
        out["where_" + name] = pts.reshape(-1, 3)[np.argmin(T)]
        out["surface_" + name] = T
    out["seconds"] = time.time() - start
    del result, field
    gc.collect()                      # release the assembled system before the next solve
    return out


# ## Solve
#
# The finest Q3 mesh is first solved with the unit load $\theta_\alpha=1$ °C,
# $\theta_\beta=\theta_\gamma=0$ for the coupling coefficients and weighting factors. Then the
# design load runs: the h-study with the cubic elements on the three mesh levels and the
# order comparison Q1/Q2 on the finest mesh. Only the finest cubic design field is kept in
# memory (for the figure).

unit = solve(LEVELS[-1], 3, {"alpha": 1.0, "beta": 0.0, "gamma": 0.0})
runs = []
for level, order in [(level, 3) for level in LEVELS] + [(LEVELS[-1], 1), (LEVELS[-1], 2)]:
    runs.append(solve(level, order, THETA, keep_field=(level, order) == (LEVELS[-1], 3)))
for r in runs:
    print(f"Q{r['order']} level {r['level']}: {r['cells']} cells, {r['unknowns']} unknowns, "
          f"Phi_alpha = {r['Phi_alpha']:.4f} W, Phi_beta = {r['Phi_beta']:.4f} W, Phi_gamma = {r['Phi_gamma']:.4f} W, "
          f"T_min,alpha = {r['T_alpha']:.4f} degC, T_min,beta = {r['T_beta']:.4f} degC  ({r['seconds']:.0f} s)")
fine = runs[2]


# ## Comparison with the reference
#
# Coupling coefficients from the unit solve (superscript $u$): $L_{\alpha\beta}=-\Phi^u_\beta$,
# $L_{\alpha\gamma}=-\Phi^u_\gamma$, and from the design solve
# $\Phi_\beta=L_{\alpha\beta}(\theta_\beta-\theta_\alpha)+L_{\beta\gamma}(\theta_\beta-\theta_\gamma)$.
# The weighting factor of an environment at a surface point is the temperature that point
# takes when that environment is at 1 and the others at 0; the points are the locations of
# the lowest surface temperatures found in the design solve.

L_ab = -unit["Phi_beta"]
L_ag = -unit["Phi_gamma"]
L_bg = (fine["Phi_beta"] - L_ab * (THETA["beta"] - THETA["alpha"])) / (THETA["beta"] - THETA["gamma"])
F = fine["field"]


def weights(name):
    k = np.argmin(fine["surface_" + name])          # surface point of the lowest design temperature
    g_a = unit["surface_" + name][k]
    g_b = (fine["T_" + name] - THETA["alpha"] * g_a) / THETA["beta"]
    return 1.0 - g_a - g_b, g_a, g_b


g_alpha, g_beta = weights("alpha"), weights("beta")

print("EN ISO 10211 case 3, Q3 elements on the finest mesh "
      f"({fine['cells']} hexahedra, {fine['unknowns']} unknowns)")
print("   quantity                       reference    computed    deviation   tolerance")
for key, label, value in (("Phi_alpha", "heat flow alpha / W", fine["Phi_alpha"]),
                          ("Phi_beta", "heat flow beta / W", fine["Phi_beta"]),
                          ("Phi_gamma", "heat flow gamma / W", -fine["Phi_gamma"]),
                          ("L_ag", "L alpha-gamma / (W/K)", L_ag),
                          ("L_bg", "L beta-gamma / (W/K)", L_bg),
                          ("L_ab", "L alpha-beta / (W/K)", L_ab)):
    print(f"   {label:28s} {REF[key]:9.3f}   {value:9.4f}    {100 * (value / REF[key] - 1):+7.3f} %    1 %")
for key, label in (("T_alpha", "min. temperature alpha / degC"), ("T_beta", "min. temperature beta / degC")):
    print(f"   {label:28s} {REF[key]:9.2f}   {fine[key]:9.4f}    {fine[key] - REF[key]:+7.4f} K    0.1 K")
for key, g in (("g_alpha", g_alpha), ("g_beta", g_beta)):
    print(f"   weighting factors at {key[2:]:5s} min. (gamma, alpha, beta): reference "
          + ", ".join(f"{v:.3f}" for v in REF[key]) + "; computed " + ", ".join(f"{v:.4f}" for v in g)
          + f"; largest difference {max(abs(a - b) for a, b in zip(g, REF[key])):.4f} "
          "(0.1 K on the 20 K span = 0.005)")
print(f"   heat flows between environments: alpha-gamma {L_ag * 20:.3f} W (ref. 35.62), "
      f"beta-gamma {L_bg * 15:.3f} W (ref. 24.36), alpha-beta {L_ab * 5:.3f} W (ref. 10.47)")
print("   location of the lowest temperatures: alpha at", np.round(fine["where_alpha"], 4),
      " beta at", np.round(fine["where_beta"], 4), "(the inner room corner at the slab soffit / floor surface)")

print("Energy balance (heat entering from the rooms = heat leaving to the outside)")
print(f"   Phi_alpha + Phi_beta = {fine['Phi_alpha'] + fine['Phi_beta']:.5f} W, -Phi_gamma = {-fine['Phi_gamma']:.5f} W, "
      f"relative difference {abs(fine['Phi_alpha'] + fine['Phi_beta'] + fine['Phi_gamma']) / -fine['Phi_gamma']:.1e}")


def richardson(values, levels):
    f1, f2, f3 = values
    h1, h2, h3 = 1.0 / np.asarray(levels, float)
    if (f1 - f2) * (f2 - f3) <= 0:
        return np.nan, np.nan
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.1, 15.0)
    except ValueError:
        return np.nan, np.nan
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


print("h-study of the cubic elements (h ~ 1/level; levels " + ", ".join(str(l) for l in LEVELS) + ")")
for key, label in (("Phi_alpha", "Phi_alpha / W"), ("Phi_beta", "Phi_beta / W"), ("Phi_gamma", "Phi_gamma / W"),
                   ("T_alpha", "T_min,alpha / degC"), ("T_beta", "T_min,beta / degC")):
    vals = [r[key] for r in runs[:3]]
    line = f"   {label:19s} " + "  ".join(f"{v:9.4f}" for v in vals)
    if key.startswith("Phi"):
        p, ext = richardson(vals, LEVELS)
        line += f"   change last two meshes {abs(vals[-1] / vals[-2] - 1):.1e}, observed order {p:.2f}, Richardson {ext:.4f}"
    else:   # point value at the re-entrant room corner: no asymptotic order, report the change only
        line += f"   change last two meshes {abs(vals[-1] - vals[-2]):.4f} K"
    print(line)
print("Element order on the finest mesh")
for r in (runs[3], runs[4], runs[2]):
    print(f"   Q{r['order']}: {r['unknowns']:6d} unknowns, Phi_gamma {-r['Phi_gamma']:.4f} W "
          f"({100 * (-r['Phi_gamma'] / REF['Phi_gamma'] - 1):+.3f} %), T_min,alpha {r['T_alpha']:.4f} degC, "
          f"T_min,beta {r['T_beta']:.4f} degC")


# ## Figure
#
# Top: surface temperature seen from the rooms (left) and from outside with the balcony
# (middle), and the temperature along the inner room corner $x=y=0.3$ (right). Bottom:
# vertical section $x=0.8$ m through the wall, the slab and the balcony; horizontal section
# through the slab $z=1.075$ m; and the convergence of the heat flow and of the minimum
# temperatures against the tolerances of the standard.

mesh = meshes[LEVELS[-1]]
norm = colors.Normalize(vmin=0.0, vmax=20.0)
sub = np.linspace(0.0, 1.0, 5)


def surface_quads(names):
    polys, vals = [], []
    for name in names:
        pts, _ = face_points(mesh, name, sub, sub)
        T = np.asarray(F.evaluate(pts.reshape(-1, 3))).reshape(len(pts), 5, 5) - 273.15
        P = pts.reshape(len(pts), 5, 5, 3)
        for a in range(4):
            for b in range(4):
                polys.append(np.stack((P[:, a, b], P[:, a + 1, b], P[:, a + 1, b + 1], P[:, a, b + 1]), axis=1))
                vals.append(0.25 * (T[:, a, b] + T[:, a + 1, b] + T[:, a + 1, b + 1] + T[:, a, b + 1]))
    return np.concatenate(polys), np.concatenate(vals)


fig = plt.figure(figsize=(18, 11), constrained_layout=True)
grid = fig.add_gridspec(2, 3)
views = ((("alpha", "beta"), 20, 45, norm, "Inner wall faces of both rooms (alpha 20 degC, beta 15 degC)"),
         (("gamma",), 20, 225, colors.Normalize(vmin=0.0, vmax=2.5), "Seen from outside: external faces and balcony"))
for col, (names, elev, azim, scale, title) in enumerate(views):
    ax3 = fig.add_subplot(grid[0, col], projection="3d")
    polys, vals = surface_quads(names)
    if col == 0:   # walls only: the floor and the slab soffit would hide the room corner
        vertical = np.ptp(polys[:, :, 2], axis=1) > 1e-12
        polys, vals = polys[vertical], vals[vertical]
    shade = FIELD_CMAP(scale(vals))
    ax3.add_collection3d(Poly3DCollection(polys, facecolors=shade, edgecolors=shade, linewidths=0.2,
                                          antialiased=False))
    ax3.set(xlim=(0, X_END), ylim=(Y_BAL, X_END), zlim=(0, Z_TOP), xlabel="x / m", ylabel="y / m", zlabel="z / m",
            title=title)
    ax3.set_box_aspect((X_END, X_END - Y_BAL, Z_TOP))
    ax3.view_init(elev=elev, azim=azim)
    fig.colorbar(cm.ScalarMappable(norm=scale, cmap=FIELD_CMAP), ax=ax3, shrink=0.6, label="surface temperature / degC")

ax = fig.add_subplot(grid[0, 2])
zline = np.linspace(0.0, Z_TOP, 431)
for z_lo, z_hi, colour, label in ((0.0, Z_SLAB, COPPER, "alpha: lower room"), (Z_FINISH, Z_TOP, GREEN, "beta: upper room")):
    zz = zline[(zline >= z_lo) & (zline <= z_hi)]
    T = np.asarray(F.evaluate(np.column_stack((np.full_like(zz, 0.3), np.full_like(zz, 0.3), zz)))).ravel() - 273.15
    ax.plot(T, zz, color=colour, lw=2, label=label)
ax.axhspan(Z_SLAB, Z_FINISH, color="#D5D7C8", alpha=0.6, lw=0, label="slab and floor finish")
ax.plot([REF["T_alpha"], REF["T_beta"]], [Z_SLAB, Z_FINISH], "o", mfc="none", mec=INK, ms=9,
        label="ISO 10211 minimum temperatures")
ax.set(xlabel="surface temperature in the room corner x = y = 0.3 m / degC", ylabel="z / m", ylim=(0, Z_TOP),
       title="Inner room corner: the cold spots sit at the slab")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.5)


def section(ax, axis_value, plane, extent, n=(260, 300)):
    a = np.linspace(extent[0][0] + 1e-6, extent[0][1] - 1e-6, n[0])
    b = np.linspace(extent[1][0] + 1e-6, extent[1][1] - 1e-6, n[1])
    A, B = np.meshgrid(a, b, indexing="ij")
    if plane == "x":
        P = np.column_stack((np.full(A.size, axis_value), A.ravel(), B.ravel()))
    else:
        P = np.column_stack((A.ravel(), B.ravel(), np.full(A.size, axis_value)))
    inside = in_solid(P[:, 0], P[:, 1], P[:, 2])
    T = np.full(len(P), np.nan)
    T[inside] = np.asarray(F.evaluate(P[inside])).ravel() - 273.15
    T = T.reshape(A.shape)
    cs = ax.contourf(A, B, T, levels=np.linspace(0, 20, 41), cmap=FIELD_CMAP)
    ax.contour(A, B, T, levels=np.arange(1, 20, 1), colors=CREAM, linewidths=0.5)
    return cs


ax = fig.add_subplot(grid[1, 0])
cs = section(ax, 0.8, "x", ((Y_BAL, X_END), (0.0, Z_TOP)))
for y0 in (0.10, 0.15, 0.30):
    ax.plot([y0, y0], [0, Z_SLAB], color=INK, lw=0.5, ls=":")
    ax.plot([y0, y0], [Z_FLOOR, Z_TOP], color=INK, lw=0.5, ls=":")
ax.set(aspect="equal", xlabel="y / m (outside at y < 0)", ylabel="z / m", xlim=(Y_BAL, X_END), ylim=(0, Z_TOP),
       title="Section x = 0.8 m: wall, slab and balcony")
fig.colorbar(cs, ax=ax, shrink=0.8, label="T / degC")

ax = fig.add_subplot(grid[1, 1])
cs = section(ax, 0.5 * (Z_SLAB + Z_FLOOR), "z", ((0.0, X_END), (Y_BAL, X_END)))
ax.plot([0.10, 0.10, 0.15], [X_END, 0.10, 0.10], color=INK, lw=0.5, ls=":")
ax.plot([0.15, 0.15], [Y_BAL, X_END], color=INK, lw=0.5, ls=":")
ax.set(aspect="equal", xlabel="x / m", ylabel="y / m", xlim=(0, X_END), ylim=(Y_BAL, X_END),
       title="Section z = 1.075 m through the slab and balcony")
fig.colorbar(cs, ax=ax, shrink=0.8, label="T / degC")

ax = fig.add_subplot(grid[1, 2])
cells = [r["unknowns"] for r in runs[:3]]
ax.semilogx(cells, [100 * (-r["Phi_gamma"] / REF["Phi_gamma"] - 1) for r in runs[:3]], "s-", color=GREEN,
            label="Q3 h-study: heat flow gamma")
ax.semilogx([r["unknowns"] for r in runs[3:5]], [100 * (-r["Phi_gamma"] / REF["Phi_gamma"] - 1) for r in runs[3:5]],
            "o", color=COPPER, ms=8, label="Q1, Q2 on the finest mesh: heat flow gamma")
for sign in (1, -1):
    ax.axhline(sign * 100 * TOL_PHI, color=GREEN, lw=0.8, ls=":")
ax.axhline(0.0, color="#9FA593", lw=0.6)
ax.set(xlabel="unknowns", ylabel="heat flow: deviation from 59.98 W / %", ylim=(-2.5, 2.5),
       title="Convergence against the ISO values\n(dotted: tolerances 1 % and 0.1 K)")
ax2 = ax.twinx()
for key, marker in (("T_alpha", "^"), ("T_beta", "v")):
    ax2.semilogx(cells, [r[key] - REF[key] for r in runs[:3]], marker + "--", color=COPPER, mfc="none",
                 label=f"Q3: min. temperature {key[2:]}")
for sign in (1, -1):
    ax2.axhline(sign * TOL_T, color=COPPER, lw=0.8, ls=":")
ax2.set(ylabel="min. surface temperature: deviation / K", ylim=(-0.25, 0.25))
ax.set_xticks([3e3, 1e4, 3e4, 6e4], ["3 000", "10 000", "30 000", "60 000"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
plt.show()


# ## Going further
#
# Insulate the balcony slab with a thermal break (a 60 mm insulating element of
# $\lambda=0.15$ W/(m K) in the slab inside the brick leaf) and compare the coupling
# coefficient $L_{\alpha\gamma}$ and the minimum surface temperature with the standard's
# mould criterion $f_{R_{si}}\ge0.7$; compute the linear thermal transmittance $\Psi$ of the
# slab edge by subtracting the one-dimensional wall losses; or run the two-dimensional
# test cases 1 and 2 of the standard.
