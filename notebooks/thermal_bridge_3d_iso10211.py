"""OneSimulation example: 3D thermal bridge of an iron bar through an insulation layer (ISO 10211 case 4).
Requires the installed, activated OneSimulation runtime.

The steady 3D temperature field of the fourth test reference case of EN ISO 10211 is computed
twice on the same graded block meshes: with cell-centred finite volumes (two-point fluxes with
least-squares gradients) and with quadratic hexahedral finite elements. Both are graded against
the tabulated heat flow and maximum external surface temperature of the standard and their
tolerances, the far-field one-dimensional solution, the energy balance and mesh studies.
"""


# # Thermal bridge in 3D: iron bar through an insulation layer
#
# EN ISO 10211 (*Thermal bridges in building construction - Heat flows and surface
# temperatures - Detailed calculations*) validates numerical methods with four test
# reference cases. A program that reproduces all of them is classed as a
# *three-dimensional steady-state high-precision method*. Case 4 is the three-dimensional
# one with the steepest gradients: a square insulation layer, 1 m x 1 m and 0.2 m thick,
# is pierced by an iron bar of 0.1 m x 0.05 m cross section. The bar starts flush with
# the cold (external) surface, crosses the insulation and sticks out 0.4 m into the warm
# (internal) room:
#
# | part | extent / m | conductivity / (W/(m K)) |
# |---|---|---|
# | insulation layer | $0\le x\le1,\ 0\le y\le0.2,\ 0\le z\le1$ | 0.1 |
# | iron bar | $0.45\le x\le0.55,\ 0\le y\le0.6,\ 0.475\le z\le0.525$ | 50 |
#
# The temperature solves the steady heat equation $\nabla\cdot(k\nabla T)=0$. Both sides
# exchange heat with the air through a surface resistance $R_s=0.10$ m² K/W
# ($h=1/R_s=10$ W/(m² K)); the internal air is at 1 °C, the external air at 0 °C. Boundary
# conditions on every face:
#
# | face | condition |
# |---|---|
# | external face $y=0$ (insulation and bar end) | $-k\,\partial_nT=h\,(T-0\,^\circ\mathrm{C})$ |
# | internal face $y=0.2$ of the insulation and all faces of the protruding bar | $-k\,\partial_nT=h\,(T-1\,^\circ\mathrm{C})$ |
# | the four cut edges of the layer ($x=0,1$; $z=0,1$) | adiabatic (zero flux) |
#
# The field is genuinely three-dimensional: it depends on the depth $y$ and on both
# distances from the bar. The problem is symmetric about the planes $x=0.5$ and $z=0.5$,
# so one quarter ($x\ge0.5$, $z\ge0.5$) is computed with zero-flux symmetry planes and
# mirrored for the figures; heat flows are multiplied by four.
#
# **Numerical methods (both on the same meshes, same OneSimulation problem).**
# - **Cell-centred finite volumes**: one temperature per hexahedron, two-point diffusive
#   face fluxes (`CentralDifference`) with least-squares cell gradients
#   (`LeastSquaresGradient`) for the non-orthogonal correction, Robin faces closed by the
#   exact face balance. On faces between bar and insulation the face conductivity is the
#   harmonic mean, the classical series-resistance rule of finite volumes (S. V. Patankar,
#   *Numerical Heat Transfer and Fluid Flow*, Hemisphere 1980, Sec. 4.2-3); the mesh has
#   equal cell widths on both sides of every material interface, so the harmonic mean is
#   the exact series resistance of the two half cells.
# - **Quadratic finite elements**: continuous Galerkin with 27-node tensor-product
#   Lagrange elements (Q2) on the same hexahedra.
# - Linear solver: sparse direct LU in both cases (the problem is linear and steady).
#
# **Reference.** The test reference values of EN ISO 10211:2007, Annex A (2017 edition:
# Annex C), case 4: heat flow $\Phi=0.540$ W between the internal and the external
# environment and the highest external surface temperature $0.805\,^\circ$C, to be met
# within 1 % (heat flow) and 0.005 K (temperature). Supporting checks: the one-dimensional
# solution far from the bar, the energy balance (heat in = heat out) and three-level mesh
# studies with observed order and Richardson estimate for both methods.

from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
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
from scipy import sparse
from scipy.optimize import brentq

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES, affine_hexahedral_geometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.fvm import CentralDifference, FiniteVolume, LeastSquaresGradient
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters

k_ins, k_iron = 0.1, 50.0        # W/(m K)
h_s = 1.0 / 0.10                 # W/(m^2 K), surface coefficient from R_s = 0.10 m^2 K/W
T_int, T_ext = 274.15, 273.15    # K (1 degC and 0 degC)
x_bar, z_bar = 0.55, 0.525       # outer faces of the bar in the quarter model
t_ins, y_tip = 0.2, 0.6          # insulation thickness, end of the bar
Phi_ref, Phi_tol = 0.540, 0.01   # W and relative tolerance (EN ISO 10211, case 4)
T_ref, T_tol = 0.805, 0.005      # degC and absolute tolerance
U_1d = 1.0 / (0.10 + t_ins / k_ins + 0.10)
print(f"1D far field: U = {U_1d:.5f} W/(m^2 K), heat flux {U_1d * 1.0:.5f} W/m^2, "
      f"external surface temperature {U_1d * 0.10:.5f} degC")


# ## Geometry and mesh
#
# The quarter model is covered by a tensor-product block mesh whose cells inside the
# insulation and inside the bar are kept and whose cells in the air beside the protruding
# bar are dropped. Spacing: geometric grading away from the bar surfaces $x=0.55$ and
# $z=0.525$ (with the same width on both sides of each surface), towards both faces of
# the insulation in $y$, and along the protruding bar from $y=0.2$. A mesh level $\ell$
# scales the cell counts by $\ell$ and the smallest cell width ($8$ mm$/\ell$) by $1/\ell$.
# The hexahedra are handed to OneSimulation as a native hexahedral mesh with five named
# boundary groups.

@dataclass(frozen=True, eq=False, kw_only=True)
class QuarterBridge(Geometry):
    """Quarter of the ISO 10211 case 4 construction."""

    dimension = 3

    def boundary_groups(self):
        names = ("external", "internal", "symmetry", "layer_edges")
        return {**{name: (name,) for name in names}, "all": names}


def from_surface(length, first, n):
    """n cells over length, the first one of width `first`, geometric growth."""
    if n == 1 or abs(n * first - length) < 1e-12:
        return np.linspace(0.0, length, n + 1)
    lo, hi = (1.0 + 1e-9, 20.0) if n * first < length else (1e-3, 1.0 - 1e-9)
    r = brentq(lambda r: first * (r**n - 1) / (r - 1) - length, lo, hi)
    out = np.concatenate(([0.0], np.cumsum(first * r ** np.arange(n))))
    out[-1] = length
    return out


def axes_for(level):
    c = lambda base: int(round(base * level))
    h0 = 0.008 / level
    xs = np.concatenate((x_bar - from_surface(0.05, h0, c(2))[::-1][:-1], x_bar + from_surface(0.45, h0, c(5))))
    zs = np.concatenate((z_bar - from_surface(0.025, h0, c(1))[::-1][:-1], z_bar + from_surface(0.475, h0, c(5))))
    y_cold = from_surface(t_ins / 2, h0, c(3))
    ys = np.concatenate((y_cold[:-1], t_ins - y_cold[::-1][:-1], t_ins + from_surface(y_tip - t_ins, h0, c(3))))
    return xs, ys, zs


def in_solid(x, y, z):
    return (y < t_ins) | ((x < x_bar) & (z < z_bar))


class BridgeMesh(StructuredHexahedralMesh):
    """Hex8 block mesh of the quarter construction (insulation + bar)."""

    def __init__(self, geometry, xs, ys, zs):
        self.geometry = geometry
        centres = np.meshgrid(*(0.5 * (v[1:] + v[:-1]) for v in (xs, ys, zs)), indexing="ij")
        keep = in_solid(*centres)
        index = -np.ones((len(xs), len(ys), len(zs)), dtype=np.int64)
        coordinates, cells = [], []
        corners = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
        for i, j, k in zip(*np.nonzero(keep)):
            ids = []
            for p, q, r in corners:
                node = (i + p, j + q, k + r)
                if index[node] < 0:
                    index[node] = len(coordinates)
                    coordinates.append((xs[node[0]], ys[node[1]], zs[node[2]]))
                ids.append(index[node])
            cells.append(ids)
        data = affine_hexahedral_geometry(np.array(coordinates), np.array(cells))
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (len(xs) - 1, len(ys) - 1, len(zs) - 1)
        self.mesh_id = "iso10211-case4:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        owners = {}
        for cell in self.cells:
            for face in FACES:
                owners.setdefault(tuple(sorted(cell[list(face)])), []).append(tuple(cell[list(face)]))
        self._facets = {name: [] for name in geometry.boundary_groups()["all"]}
        for matches in owners.values():
            if len(matches) == 1:
                q = self.coordinates[list(matches[0])]
                if np.allclose(q[:, 1], 0.0):
                    name = "external"
                elif np.allclose(q[:, 0], 0.5) or np.allclose(q[:, 2], 0.5):
                    name = "symmetry"
                elif np.allclose(q[:, 0], 1.0) or np.allclose(q[:, 2], 1.0):
                    name = "layer_edges"
                else:
                    name = "internal"
                self._facets[name].append(matches[0])
        self._carrier = self._build_carrier()


# ## Model
#
# One `HeatConduction` problem serves both methods. The conductivity is a
# position-dependent material law: 50 W/(m K) in the bar, 0.1 W/(m K) in the insulation
# and, on the bar-insulation interface (where only the finite-volume faces sample it),
# the harmonic mean of the two. Quadrature points of the finite elements always lie
# inside a cell, so the elements see the two plain values.

@dataclass(frozen=True)
class BridgeConductivity:
    unit: str = "W/m/K"

    def evaluate(self, *, x, y, z, **state):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        tol = 1e-9
        inner = (x < x_bar - tol) & (z < z_bar - tol)
        closed = (x < x_bar + tol) & (z < z_bar + tol)
        interface = closed & ~inner & (y < t_ins - tol)
        return np.where(interface, 2 * k_iron * k_ins / (k_iron + k_ins), np.where(closed, k_iron, k_ins))


material = replace(IsotropicSolid(density=500.0, specific_heat_capacity=1700.0, thermal_conductivity=k_ins,
                                  youngs_modulus=1e9, poisson_ratio=0.3, thermal_expansion_coefficient=0.0),
                   thermal_conductivity=BridgeConductivity())

METHODS = {
    "FV": FiniteVolume(gradient=LeastSquaresGradient(), diffusion=CentralDifference()),
    "Q2": FiniteElement(element=Lagrange(order=2)),
}


def solve(level, method):
    start = time.time()
    geometry = QuarterBridge()
    mesh = BridgeMesh(geometry, *axes_for(level))
    model = HeatConduction(domain=geometry.domain(), material=material, heating=0.0)
    b = geometry.boundaries
    problem = Problem(name="ISO 10211 case 4 thermal bridge", geometry=geometry, physics=(model,),
                      boundary_conditions=(
                          ConvectiveHeatFlux(boundary=b.internal, physics=model, heat_transfer_coefficient=h_s,
                                             ambient_temperature=T_int),
                          ConvectiveHeatFlux(boundary=b.external, physics=model, heat_transfer_coefficient=h_s,
                                             ambient_temperature=T_ext),
                          HeatFlux(boundary=b.symmetry | b.layer_edges, physics=model, value=0.0)))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=METHODS[method], linear_solver=SparseDirect())).solve_stationary()
    field = result.field(model.temperature)
    operator = result.operators[model.model_id]
    out = dict(level=level, method=method, mesh=mesh, field=field, unknowns=operator.problem_size)
    out.update(fv_surface(mesh, operator, field) if method == "FV" else fe_surface(mesh, field))
    out["seconds"] = time.time() - start
    return out


# **Surface quantities.** Finite volumes: the heat flow through a boundary is the sum of
# the conservative face fluxes that the cell balance itself uses
# (`conservative_face_fluxes`), and the surface temperature of an external face follows
# from its flux, $T_s=T_e+q/h$. Finite elements: the Robin flux $h\,(T-T_e)$ is integrated
# with 3 x 3 Gauss points per face (exact for the biquadratic trace), and the surface
# temperature is the Q2 trace itself.

def fv_surface(mesh, operator, field):
    g = operator.geometry
    flux = operator.conservative_face_fluxes(field.values)
    to_faces = sparse.csr_matrix(g.quadrature_to_faces)
    area = np.asarray(to_faces.sum(axis=1)).ravel()
    centre = (to_faces @ g.points) / area[:, None]
    boundary = np.unique(g.face_indices[g.boundary])
    cold = boundary[np.abs(centre[boundary, 1]) < 1e-12]
    warm = boundary[(centre[boundary, 1] > 1e-12)
                    & (np.abs(centre[boundary, 0] - 0.5) > 1e-12) & (np.abs(centre[boundary, 2] - 0.5) > 1e-12)
                    & (np.abs(centre[boundary, 0] - 1.0) > 1e-12) & (np.abs(centre[boundary, 2] - 1.0) > 1e-12)]
    T_face = T_ext + flux[cold] / area[cold] / h_s
    far = np.argmin(np.sum((centre[cold] - np.array([1.0, 0.0, 1.0])) ** 2, axis=1))
    return dict(Phi_out=4 * float(flux[cold].sum()), Phi_in=-4 * float(flux[warm].sum()),
                T_max=float(T_face.max()) - 273.15, T_far=float(T_face[far]) - 273.15,
                cold_centres=centre[cold], cold_T=T_face - 273.15)


GAUSS = np.array([0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)])
GW = np.array([5.0, 8.0, 5.0]) / 18.0


def face_rule(mesh, names):
    pts, wts = [], []
    for name in names:
        for face in mesh._facets[name]:
            q = mesh.coordinates[list(face)]
            lo, hi = q.min(axis=0), q.max(axis=0)
            flat = int(np.argmin(hi - lo))
            u, v = [ax for ax in range(3) if ax != flat]
            for i, gu in enumerate(GAUSS):
                for j, gv in enumerate(GAUSS):
                    p = lo.copy()
                    p[u] += (hi[u] - lo[u]) * gu
                    p[v] += (hi[v] - lo[v]) * gv
                    pts.append(p)
                    wts.append(GW[i] * GW[j] * (hi[u] - lo[u]) * (hi[v] - lo[v]))
    return np.array(pts), np.array(wts)


def fe_surface(mesh, field):
    pc, wc = face_rule(mesh, ("external",))
    pw, ww = face_rule(mesh, ("internal",))
    Tc = np.asarray(field.evaluate(pc)).ravel()
    Tw = np.asarray(field.evaluate(pw)).ravel()
    corner = float(np.asarray(field.evaluate(np.array([[0.5, 0.0, 0.5]]))).ravel()[0])
    far = float(np.asarray(field.evaluate(np.array([[1.0, 0.0, 1.0]]))).ravel()[0])
    return dict(Phi_out=4 * float(np.sum(wc * h_s * (Tc - T_ext))), Phi_in=4 * float(np.sum(ww * h_s * (T_int - Tw))),
                T_max=max(corner, float(Tc.max())) - 273.15, T_far=far - 273.15)


# ## Solve
#
# Three levels per method. The finite-volume levels $\ell=2,3,4$ have about 2 100, 7 000
# and 16 500 cells; the finite-element levels $\ell=1,2,3$ use the same mesh family
# (levels 2 and 3 are the two coarser finite-volume meshes) with about 2 600, 18 700 and
# 60 500 Q2 nodes.

runs = {"FV": [], "Q2": []}
for method, levels in (("FV", (2, 3, 4)), ("Q2", (1, 2, 3))):
    for level in levels:
        out = solve(level, method)
        runs[method].append(out)
        print(f"{method} level {level}: {out['mesh'].cell_count:6d} cells, {out['unknowns']:6d} unknowns, "
              f"Phi = {out['Phi_out']:.5f} W, T_max,ext = {out['T_max']:.5f} degC  ({out['seconds']:.0f} s)")


# ## Comparison with the reference

def richardson(values, refinement):
    f1, f2, f3 = values
    h1, h2, h3 = 1.0 / np.asarray(refinement, float)
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.2, 12.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


print("EN ISO 10211 case 4, finest mesh of each method")
print("   quantity                       reference      FV (cells)     Q2 (elements)")
fv, fe = runs["FV"][-1], runs["Q2"][-1]
print(f"   heat flow Phi / W              {Phi_ref:9.3f}    {fv['Phi_out']:9.4f}      {fe['Phi_out']:9.4f}")
print(f"     deviation (tolerance 1 %)                 {100 * (fv['Phi_out'] / Phi_ref - 1):+8.2f} %     "
      f"{100 * (fe['Phi_out'] / Phi_ref - 1):+8.2f} %")
print(f"   max external T / degC          {T_ref:9.3f}    {fv['T_max']:9.4f}      {fe['T_max']:9.4f}")
print(f"     deviation (tolerance 0.005 K)              {fv['T_max'] - T_ref:+8.4f} K     {fe['T_max'] - T_ref:+8.4f} K")

print("Mesh studies (h ~ 1/level)")
for method, levels in (("FV", (2, 3, 4)), ("Q2", (1, 2, 3))):
    for key, label in (("Phi_out", "Phi / W"), ("T_max", "T_max / degC")):
        vals = [r[key] for r in runs[method]]
        p, ext = richardson(vals, levels)
        print(f"   {method} {label:13s} " + "  ".join(f"{v:.5f}" for v in vals)
              + f"   order {p:.2f}, Richardson {ext:.5f}, change last two meshes {abs(vals[-1] - vals[-2]):.1e}")

print("Energy balance (heat entering on the warm side = heat leaving on the cold side)")
for method in runs:
    r = runs[method][-1]
    print(f"   {method}: in {r['Phi_in']:.6f} W, out {r['Phi_out']:.6f} W, relative difference "
          f"{abs(r['Phi_in'] - r['Phi_out']) / r['Phi_out']:.1e}")
print(f"1D far field at the layer corner (x = z = 1): external surface temperature, closed form "
      f"{U_1d * 0.10:.5f} degC")
for method in runs:
    print(f"   {method}: {runs[method][-1]['T_far']:.5f} degC")
print(f"Thermal bridge effect: Phi - U A = {fe['Phi_out'] - U_1d * 1.0:.4f} W "
      f"(point thermal transmittance chi = {(fe['Phi_out'] - U_1d) / 1.0:.4f} W/K)")


# ## Figure
#
# Left: surface temperature of the whole construction (the quarter mirrored), seen from
# the warm side with the protruding bar, and from the cold side with the warm spot where
# the bar ends. Middle: the section $z=0.5$ through the bar axis. Right: the external
# surface temperature around the bar end, and the mesh convergence of both methods.

fine = runs["Q2"][-1]
mesh, field = fine["mesh"], fine["field"]
nodal = np.asarray(field.evaluate(mesh.coordinates)).ravel() - 273.15
norm = colors.Normalize(vmin=0.0, vmax=1.0)
mirrors = [np.array([sx, 1.0, sz]) for sx in (1, -1) for sz in (1, -1)]
centre_shift = np.array([0.5, 0.0, 0.5])


def surface_polygons(names):
    polys, vals = [], []
    for name in names:
        for face in mesh._facets[name]:
            q = mesh.coordinates[list(face)]
            v = nodal[list(face)].mean()
            for m in mirrors:
                polys.append((q - centre_shift) * m + centre_shift)
                vals.append(v)
    return np.array(polys), np.array(vals)


fig = plt.figure(figsize=(17, 10), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.2, 1.0))
polys, vals = surface_polygons(("external", "internal", "layer_edges"))
for row, (elev, azim, title) in enumerate(((18, 40, "Warm side: surface temperature"),
                                           (22, 230, "Cold side: surface temperature"))):
    ax3 = fig.add_subplot(grid[row, 0], projection="3d")
    shade = FIELD_CMAP(norm(vals))
    ax3.add_collection3d(Poly3DCollection(polys[:, :, [0, 1, 2]], facecolors=shade, edgecolors=shade,
                                          linewidths=0.3, antialiased=False))
    ax3.set(xlim=(0, 1), ylim=(0, 0.6), zlim=(0, 1), xlabel="x / m", ylabel="y / m", zlabel="z / m", title=title)
    ax3.set_box_aspect((1, 0.6, 1))
    ax3.view_init(elev=elev, azim=azim)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax3, shrink=0.6, label="T / degC")

ax = fig.add_subplot(grid[:, 1])
section = [f for f in mesh._facets["symmetry"] if np.allclose(mesh.coordinates[list(f)][:, 2], 0.5)]
tris = np.array([t for f in section for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))])
used = np.unique(tris)
local = -np.ones(len(mesh.coordinates), dtype=int)
local[used] = np.arange(len(used))
X, Y = mesh.coordinates[used, 0], mesh.coordinates[used, 1]
for sign in (1, -1):
    tri = Triangulation(0.5 + sign * (X - 0.5), Y, local[tris])
    cs = ax.tricontourf(tri, nodal[used], levels=np.linspace(0, 1, 41), cmap=FIELD_CMAP)
    ax.tricontour(tri, nodal[used], levels=np.linspace(0.05, 0.95, 19), colors=CREAM, linewidths=0.5)
ax.add_patch(plt.Rectangle((0.45, 0.0), 0.1, y_tip, fill=False, ec=INK, lw=0.8))
ax.axhline(t_ins, color=INK, lw=0.6, ls=":")
ax.set(xlim=(0.15, 0.85), ylim=(0, 0.62), aspect="equal", xlabel="x / m", ylabel="y / m (cold face at y = 0)",
       title="Section z = 0.5 through the bar axis (Q2)")
fig.colorbar(cs, ax=ax, shrink=0.7, label="T / degC")

ax = fig.add_subplot(grid[0, 2])
cold = [f for f in mesh._facets["external"]]
tris = np.array([t for f in cold for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))])
used = np.unique(tris)
local[:] = -1
local[used] = np.arange(len(used))
Xc, Zc = mesh.coordinates[used, 0], mesh.coordinates[used, 2]
for sx in (1, -1):
    for sz in (1, -1):
        tri = Triangulation(0.5 + sx * (Xc - 0.5), 0.5 + sz * (Zc - 0.5), local[tris])
        cs = ax.tricontourf(tri, nodal[used], levels=np.linspace(0, 0.82, 42), cmap=FIELD_CMAP)
        ax.tricontour(tri, nodal[used], levels=[0.06, 0.1, 0.2, 0.4, 0.6, 0.8], colors=CREAM, linewidths=0.5)
ax.add_patch(plt.Rectangle((0.45, 0.475), 0.1, 0.05, fill=False, ec=INK, lw=0.8))
ax.set(xlim=(0.25, 0.75), ylim=(0.35, 0.65), aspect="equal", xlabel="x / m", ylabel="z / m",
       title=f"External surface around the bar end (max {fine['T_max']:.3f} degC)")
fig.colorbar(cs, ax=ax, shrink=0.8, label="T / degC")

ax = fig.add_subplot(grid[1, 2])
for method, marker, color in (("FV", "o", COPPER), ("Q2", "s", GREEN)):
    cells = [r["mesh"].cell_count for r in runs[method]]
    ax.semilogx(cells, [100 * (r["Phi_out"] / Phi_ref - 1) for r in runs[method]], marker + "-", color=color,
                label=f"{method}: heat flow")
for sign in (1, -1):
    ax.axhline(sign * 100 * Phi_tol, color=COPPER, lw=0.8, ls=":")
ax.axhline(0.0, color="#9FA593", lw=0.6)
ax.set(xlabel="hexahedral cells (quarter model)", ylabel="heat flow: deviation from 0.540 W / %",
       title="Mesh convergence against the ISO values\n(dotted: tolerances 1 % and 0.005 K)", ylim=(-1.2, 1.2))
ax2 = ax.twinx()
for method, marker, color in (("FV", "o", COPPER), ("Q2", "s", GREEN)):
    cells = [r["mesh"].cell_count for r in runs[method]]
    ax2.semilogx(cells, [1e3 * (r["T_max"] - T_ref) for r in runs[method]], marker + "--", mfc="none", color=color,
                 label=f"{method}: max external T")
for sign in (1, -1):
    ax2.axhline(sign * 1e3 * T_tol, color=GREEN, lw=0.8, ls=":")
ax2.set(ylabel="max external T: deviation from 0.805 degC / mK", ylim=(-6, 6))
lines = ax.get_legend_handles_labels()
lines2 = ax2.get_legend_handles_labels()
ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7, loc="lower right")
plt.show()


# ## Going further
#
# Run the other three ISO 10211 test cases (two 2D sections and the 3D corner of case 3);
# replace the bar by a stainless-steel anchor ($k=15$ W/(m K)) and compare the point
# thermal transmittance $\chi$; or add a thermal-break pad of low conductivity inside the
# insulation and watch the warm spot on the external face disappear.
