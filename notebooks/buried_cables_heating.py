"""OneSimulation example: Joule heating of three buried power cables.
Requires the installed, activated OneSimulation runtime.

Three single-core power cables lie side by side in the ground. Their conductors are heated by
the load current, and the heat flows through insulation, sheath and jacket into the soil and up
to the ground surface. The two-dimensional temperature field of the cross-section is computed
with quadratic finite elements on a curved multi-block hexahedral mesh and compared with the
IEC 60287 thermal circuit (Neher-McGrath / Kennelly image method with mutual heating), an exact
image solution, the energy balance and mesh and domain-size studies.
"""


# # Joule heating of buried power cables
#
# Three single-core cables of a 132 kV circuit are laid in flat formation at a depth of
# $L=1$ m (to the cable axes) with an axial spacing of $s=0.3$ m. Each cable consists of a
# copper conductor (1000 mm²), an XLPE insulation, a lead sheath and a polyethylene jacket. The
# conductors carry the load current $I=1200$ A (50 Hz), and the Joule heat
# $W=I^2R'_{ac}(\theta_c)$ per unit length depends on the conductor temperature $\theta_c$
# through the resistance of copper. The ground surface temperature is known (20 °C), the
# deep ground is at the same temperature. The cables are long, so the temperature varies only
# over the cross-section $(x,y)$:
#
# $$-\nabla\cdot\big(k(x,y)\,\nabla T\big)=q(x,y),\qquad q=\frac{W}{A_c}\ \text{in each conductor},$$
#
# with a piecewise constant conductivity $k$ that jumps by three orders of magnitude between
# copper and XLPE. The cables heat each other: the middle one runs hottest.
#
# **Numerical method.**
#
# * Spatial: continuous Galerkin finite elements, tensor-product Lagrange **Q2** elements on
#   **curved 27-node hexahedra**. The cross-section is covered by a conforming multi-block
#   quadrilateral mesh - an O-grid around every cable whose rings follow the layer interfaces
#   exactly (the mid-edge nodes lie on the circles), a square core in each conductor, and a graded
#   tensor-product grid for the soil - and extruded into a slab one cell thick whose two faces are
#   insulated. The field therefore depends on $(x, y)$ only.
# * Solver: the problem is linear and stationary for given losses: one sparse direct LU solve. The
#   temperature-dependent losses are handled by superposition: two solves with unit losses (outer
#   pair, middle cable) give the response coefficients, and the coupled loss balance
#   $W_p=I^2R'_{ac}(\theta_{c,p})$ is then solved by a fixed-point iteration.
#
# **Reference.** (i) The IEC 60287 thermal circuit (IEC 60287-1-1 losses, IEC 60287-2-1 thermal
# resistances) for buried cables, based on J. H. Neher and M. H. McGrath, "The calculation of the
# temperature rise and load capability of cable systems", AIEE Trans. 76 (1957) 752-772: the
# ground surface is an isotherm (Kennelly's assumption), each cable is replaced by a line source
# and its mirror image, and the mutual heating of the cables is superposed. (ii) An exact image
# solution of the genuinely two-dimensional problem for a homogeneous ground, which checks the
# discretisation itself. (iii) The energy balance, a three-level mesh study with observed order
# and Richardson estimate, and a domain-size study.

from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import math
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
from matplotlib.tri import Triangulation
from scipy.optimize import brentq

from onesim.cad.geometry.primitives import Geometry
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES
from onesim.meshing.generation.contracts.polynomial.hex import HEX27_REFERENCE, QuadraticHexahedralGeometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Cable construction (a typical 132 kV single-core XLPE cable, 1000 mm² copper):
#
# | layer | outer radius | thermal conductivity |
# |---|---|---|
# | copper conductor | 19.0 mm | 400 W/(m K) |
# | XLPE insulation (with screens) | 35.0 mm | 1/3.5 W/(m K) (IEC 60287-2-1: 3.5 K m/W) |
# | lead sheath | 37.5 mm | 35 W/(m K) |
# | PE jacket | 41.5 mm | 1/3.5 W/(m K) |
# | soil | - | 1.0 W/(m K) (1 K m/W) |
#
# Losses (IEC 60287-1-1): the AC resistance is $R'_{ac}=R'(\theta)(1+y_s+y_p)$ with the DC
# resistance $R'(\theta)=R'_{20}[1+\alpha_{20}(\theta-20)]$, $R'_{20}=0.0176$ mΩ/m (IEC 60228),
# $\alpha_{20}=3.93\cdot10^{-3}$ 1/K, and the skin and proximity factors
# $y_s=x_s^4/(192+0.8x_s^4)$, $y_p=F\,(d_c/s)^2\,[0.312(d_c/s)^2+1.18/(F+0.27)]$,
# $F=x_p^4/(192+0.8x_p^4)$, $x_{s,p}^2=8\pi f\,k_{s,p}\cdot10^{-7}/R'$ with $k_s=1$, $k_p=0.8$.
# The sheaths are single-point bonded (no circulating currents); sheath eddy losses and the
# dielectric loss are left out, so all heat is generated in the conductors.

mm = 1e-3
r_c, r_i, r_s, r_e = 19.0 * mm, 35.0 * mm, 37.5 * mm, 41.5 * mm
radii = (r_c, r_i, r_s, r_e)
k_layers = (400.0, 1 / 3.5, 35.0, 1 / 3.5)      # conductor, insulation, sheath, jacket
k_soil = 1.0
depth, spacing = 1.0, 0.30                       # m, laying depth (axes) and axial spacing
centres = ((-spacing, -depth), (0.0, -depth), (spacing, -depth))
T_surface = 20.0                                 # deg C, ground surface and deep ground
current, frequency = 1200.0, 50.0                # A, Hz
R20, alpha20 = 0.0176e-3, 3.93e-3                # ohm/m, 1/K
levels = (1.0, 1.5, 2.0)                         # mesh levels (cell size ~ 1/level)
domains = (5.0, 10.0, 20.0, 40.0)                # m, half-width = depth of the soil block
main_domain = domains[-1]


def ac_resistance(theta):
    R = R20 * (1 + alpha20 * (theta - 20.0))
    xs4 = (8 * np.pi * frequency * 1.0e-7 / R) ** 2
    xp4 = (8 * np.pi * frequency * 0.8e-7 / R) ** 2
    ys = xs4 / (192 + 0.8 * xs4)
    F = xp4 / (192 + 0.8 * xp4)
    ratio = 2 * r_c / spacing
    yp = F * ratio**2 * (0.312 * ratio**2 + 1.18 / (F + 0.27))
    return R * (1 + ys + yp)


def joule_loss(theta):
    return current**2 * ac_resistance(theta)


print(f"R'_ac(20 C) = {1e6 * ac_resistance(20.0):.3f} uOhm/m, R'_ac(90 C) = {1e6 * ac_resistance(90.0):.3f} uOhm/m;"
      f"  W(90 C) = {joule_loss(90.0):.2f} W/m per cable")


# ## IEC 60287 thermal circuit
#
# Thermal resistances per unit length (K m/W): insulation $T_1=\frac{\rho_1}{2\pi}\ln\frac{r_i}{r_c}$,
# jacket $T_3=\frac{\rho_3}{2\pi}\ln\frac{r_e}{r_s}$ (conductor and lead are negligible), and the
# external resistance of an isolated cable under an isothermal surface (exact for an isothermal
# cylinder in a half-space), $T_4=\frac{\rho_4}{2\pi}\ln\!\big(u+\sqrt{u^2-1}\big)$ with $u=2L/D_e$.
# Cable $k$ raises the temperature at cable $p$ by $W_k\,\frac{\rho_4}{2\pi}\ln\frac{d'_{pk}}{d_{pk}}$,
# where $d_{pk}$ is the axis distance and $d'_{pk}$ the distance to the image of cable $k$ above
# the surface. With all heat produced in the conductors,
#
# $$\theta_{c,p}=\theta_s+W_p\,(T_1+T_3+T_4)+\sum_{k\neq p}W_k\,\frac{\rho_4}{2\pi}\ln\frac{d'_{pk}}{d_{pk}},
#   \qquad W_p=I^2R'_{ac}(\theta_{c,p}).$$

T1 = 3.5 / (2 * np.pi) * np.log(r_i / r_c)
T3 = 3.5 / (2 * np.pi) * np.log(r_e / r_s)
u = depth / r_e


def mutual(p, k, shift=0.0):
    (xp, _), (xk, _) = centres[p], centres[k]
    d = abs(xp - xk)
    return 1 / k_soil / (2 * np.pi) * np.log(np.hypot(d, 2 * (depth + shift)) / d)


def iec_temperatures(shift=0.0):
    """Conductor and jacket-surface temperatures; `shift` lowers the effective isotherm by k/h."""
    uu = (depth + shift) / r_e
    T4 = 1 / k_soil / (2 * np.pi) * np.log(uu + np.sqrt(uu**2 - 1))
    theta = np.full(3, 70.0)
    for _ in range(100):
        W = joule_loss(theta)
        surface = T_surface + W * T4 + np.array([sum(W[k] * mutual(p, k, shift) for k in range(3) if k != p)
                                                  for p in range(3)])
        theta = surface + W * (T1 + T3)
    return dict(theta=theta, surface=surface, W=W, T4=T4)


iec = iec_temperatures()
print(f"T1 = {T1:.4f}, T3 = {T3:.4f}, T4 = {iec['T4']:.4f} K m/W; mutual (neighbour) "
      f"{mutual(0, 1):.4f}, (outer pair) {mutual(0, 2):.4f} K m/W")
print("IEC 60287: conductor temperatures " + ", ".join(f"{t:.2f}" for t in iec["theta"]) + " C, losses "
      + ", ".join(f"{w:.2f}" for w in iec["W"]) + " W/m")


# ## Geometry and mesh
#
# The soil block $|x|\le a$, $-a\le y\le 0$ (half-width = depth $=a$) is covered by quadrilateral
# blocks, each sampled with its mid-edge nodes so that every cell is a nine-node (biquadratic)
# quadrilateral:
#
# * around each cable a square of side $s$ holds an O-grid: $16\ell$ sectors around the
#   circumference; rings through the conductor, the insulation ($3\ell$ cells), the lead sheath
#   and the jacket (at least two cells each), and a graded transition from the jacket to the
#   square; a square core of $4\ell\times4\ell$ cells in the conductor;
# * the soil outside the three squares is a tensor-product grid whose lines continue the square
#   edges and grow geometrically (factor at most $2^{1/\ell}$) towards the far boundaries.
#
# The ring nodes, including the mid-edge nodes, lie exactly on the layer circles, so the curved
# cells reproduce the interfaces to within $5\cdot10^{-5}$ of the radius on the coarsest and
# $3\cdot10^{-6}$ on the finest mesh. The quadrilaterals are extruded into one layer of 27-node
# hexahedra 20 mm thick. The main soil block has $a=40$ m, far enough for the half-space of the
# IEC circuit; the domain-size study uses $a=5$, 10, 20 and 40 m.
#
# Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | ground surface $y=0$ | fixed temperature 20 °C (Kennelly; a convective variant follows below) |
# | bottom $y=-a$ | fixed temperature 20 °C (undisturbed deep ground) |
# | sides $x=\pm a$ | insulated (zero flux) |
# | slab faces normal to $z$ | insulated (zero flux), so the field is two-dimensional |

def doubled(t):
    """A 1D partition with its midpoints (positions of the quadratic nodes)."""
    out = np.empty(2 * len(t) - 1)
    out[::2], out[1::2] = t, 0.5 * (t[:-1] + t[1:])
    return out


def growing(length, n, first):
    """n cells over `length`, starting with width `first`, geometric growth."""
    ratio = brentq(lambda q: first * (q**n - 1) / (q - 1) - length, 1.0 + 1e-9, 10.0)
    widths = first * ratio ** np.arange(n)
    return np.r_[0.0, np.cumsum(widths)] * length / widths.sum()


def rotate(points, k):
    c, s = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
    return np.stack((c * points[..., 0] - s * points[..., 1], s * points[..., 0] + c * points[..., 1]), axis=-1)


def on_square(k, half, s):
    return rotate(np.stack((np.full_like(s, half), -half + 2 * half * s), axis=-1), k)


def on_circle(k, r, s):
    angle = -math.pi / 4 + s * math.pi / 2 + k * math.pi / 2
    return np.stack((r * np.cos(angle), r * np.sin(angle)), axis=-1)


def cable_blocks(cx, cy, level):
    """Nine-node patches (as arrays of node positions) of one cable's O-grid, with their layer index."""
    ns = int(round(4 * level))
    rings = ((r_c, r_i, int(round(3 * level)), 2), (r_i, r_s, max(2, int(round(level))), 3),
             (r_s, r_e, max(2, int(round(level))), 4))
    core, half = 0.55 * r_c, spacing / 2
    s = doubled(np.linspace(0, 1, ns + 1))
    patches = []
    for k in range(4):
        S, E = np.meshgrid(s, doubled(np.linspace(0, 1, int(round(1.5 * level)) + 1)), indexing="ij")
        patches.append(((1 - E)[..., None] * on_square(k, core, S) + E[..., None] * on_circle(k, r_c, S), 1))
        for r0, r1, n, layer in rings:
            S, E = np.meshgrid(s, doubled(np.linspace(0, 1, n + 1)), indexing="ij")
            patches.append((on_circle(k, r0 + E * (r1 - r0), S), layer))
        n_out = int(round(3 * level))
        S, E = np.meshgrid(s, doubled(growing(1.0, n_out, 0.35 / n_out)), indexing="ij")
        patches.append(((1 - E)[..., None] * on_circle(k, r_e, S) + E[..., None] * on_square(k, half, S), 0))
    S, E = np.meshgrid(s, s, indexing="ij")
    patches.append((np.stack((-core + 2 * core * S, -core + 2 * core * E), axis=-1), 1))
    return [(P + (cx, cy), layer) for P, layer in patches], ns


def soil_lines(a, level, ns):
    """Grid lines of the soil: square-edge spacing next to the cables, growth factor up to 2^(1/level)."""
    half = spacing / 2
    h = 2 * half / ns
    growth = 2.0 ** (1 / level)

    def cells_for(length):
        return max(3, math.ceil(math.log(1 + length * (growth - 1) / h) / math.log(growth)))

    far_x = growing(a - 3 * half, cells_for(a - 3 * half), h)
    X = np.r_[-3 * half - far_x[::-1], np.linspace(-3 * half, 3 * half, 3 * ns + 1)[1:-1], 3 * half + far_x]
    Y = np.r_[-depth - half - growing(a - depth - half, cells_for(a - depth - half), h)[::-1],
              np.linspace(-depth - half, -depth + half, ns + 1)[1:-1],
              -depth + half + growing(depth - half, cells_for(depth - half), h)]
    return X, Y


def cross_section(a, level):
    """Nodes, nine-node cells (3x3 node indices) and layer index of every cell."""
    keys, nodes, cells, layers = {}, [], [], []

    def add(P, layer, skip=None):
        ids = np.empty(P.shape[:2], dtype=int)
        for (i, j), p in np.ndenumerate(P[..., 0]):
            key = (round(P[i, j, 0] * 1e9), round(P[i, j, 1] * 1e9))
            ids[i, j] = keys.setdefault(key, len(nodes))
            if ids[i, j] == len(nodes):
                nodes.append(P[i, j])
        for i in range((P.shape[0] - 1) // 2):
            for j in range((P.shape[1] - 1) // 2):
                if skip is None or not skip[i, j]:
                    cell = ids[2 * i:2 * i + 3, 2 * j:2 * j + 3]
                    corner = P[[2 * i, 2 * i + 2, 2 * i], [2 * j, 2 * j, 2 * j + 2]]
                    (ux, uy), (vx, vy) = corner[1] - corner[0], corner[2] - corner[0]
                    turn = ux * vy - uy * vx
                    cells.append(cell if turn > 0 else cell.T)
                    layers.append(layer)

    for cx, cy in centres:
        patches, ns = cable_blocks(cx, cy, level)
        for P, layer in patches:
            add(P, layer)
    X, Y = soil_lines(a, level, ns)
    GX, GY = np.meshgrid(doubled(X), doubled(Y), indexing="ij")
    half = spacing / 2
    strip = np.outer((X[:-1] > -3 * half - 1e-9) & (X[1:] < 3 * half + 1e-9),
                     (Y[:-1] > -depth - half - 1e-9) & (Y[1:] < -depth + half + 1e-9))
    add(np.stack((GX, GY), axis=-1), 0, skip=strip)
    return np.array(nodes), np.array(cells), np.array(layers), ns


@dataclass(frozen=True, eq=False, kw_only=True)
class SoilSlab(Geometry):
    """Soil block with three cables, extruded into a slab one cell thick."""

    dimension = 3

    def boundary_groups(self):
        names = ("surface", "bottom", "sides", "slab_faces")
        return {**{name: (name,) for name in names}, "all": names}


REF = np.rint(2 * HEX27_REFERENCE).astype(int)      # (i, j, k) in {0, 1, 2} of the 27 nodes


class CableMesh(StructuredHexahedralMesh):
    """Curved 27-node hexahedra: the nine-node section cells extruded over one layer."""

    def __init__(self, geometry, a, level, thickness=0.02):
        xy, quads, layers, ns = cross_section(a, level)
        n2 = len(xy)
        coordinates = np.vstack([np.column_stack((xy, np.full(n2, z))) for z in (0.0, thickness / 2, thickness)])
        cells = quads[:, REF[:, 0], REF[:, 1]] + n2 * REF[:, 2]
        data = QuadraticHexahedralGeometry(coordinates, cells)
        self.geometry, self.a, self.level, self.thickness = geometry, a, level, thickness
        self.xy, self.quads, self.layers, self.sectors = xy, quads, layers, 4 * ns
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (self.cell_count, 1, 1)
        self.mesh_id = "buried-cables:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        owners = {}
        for cell in self.cells:
            for face in FACES:
                owners.setdefault(tuple(sorted(cell[list(face)])), []).append(tuple(cell[list(face)]))
        self._facets = {name: [] for name in geometry.boundary_groups()["all"]}
        for matches in owners.values():
            if len(matches) == 1:
                q = self.coordinates[list(matches[0])]
                if np.ptp(q[:, 2]) < 1e-12:
                    name = "slab_faces"
                elif np.allclose(q[:, 1], 0.0):
                    name = "surface"
                elif np.allclose(q[:, 1], -a):
                    name = "bottom"
                else:
                    name = "sides"
                self._facets[name].append(matches[0])
        self.conductor_area = [float(self.cell_volumes[(layers == 1) & (np.hypot(*(xy[quads[:, 1, 1]] - c).T) < r_c)].sum())
                               / thickness for c in centres]
        self._carrier = self._build_carrier()

    def _build_carrier(self):
        carrier = super()._build_carrier()
        return replace(carrier, connectivity=replace(carrier.connectivity, entity_kind="hexahedron27"))


for level in levels:
    m = CableMesh(SoilSlab(), main_domain, level)
    print(f"level {level:.1f}: {m.cell_count:5d} curved hexahedra, {m.sectors} sectors around each cable, "
          f"conductor area {1e6 * m.conductor_area[1]:.3f} mm^2 (circle {1e6 * math.pi * r_c**2:.3f} mm^2)")


# ## Model
#
# The conductivity is a pointwise material law (the layer table); the quadrature points always lie
# inside one layer because every interface is a cell boundary. The heat source is $W_n/A_n$ in
# conductor $n$, with $A_n$ the area of its meshed (curved) cross-section.

class LayeredGround:
    """Piecewise constant conductivity of the cable layers and the soil (steady: no capacity needed)."""

    density, specific_heat_capacity, thermal_conductivity = 1000.0, 1000.0, "piecewise"

    def __init__(self, uniform=False):
        self.k = [k_soil] * 4 if uniform else list(k_layers)

    def evaluate(self, name, *, x, y, z, **state):
        if name != "thermal_conductivity":
            return 1000.0
        for cx, cy in centres:
            r = math.hypot(x - cx, y - cy)
            if r < r_e:
                return self.k[(r > r_c) + (r > r_i) + (r > r_s)]
        return k_soil


class ConductorHeating:
    """Uniform source W_n / A_n in conductor n."""

    def __init__(self, mesh, losses):
        self.mesh, self.losses = mesh, losses

    def evaluate(self, points, *, time=0.0):
        q = np.zeros(len(points))
        for W, A, (cx, cy) in zip(self.losses, self.mesh.conductor_area, centres):
            q[np.hypot(points[:, 0] - cx, points[:, 1] - cy) < r_c] = W / A
        return q


def exact_uniform(x, y, losses):
    """Exact field for k = k_soil everywhere: disc sources under an isothermal surface (images at +L)."""
    T = np.full(np.shape(x), T_surface, dtype=float)
    for W, (cx, cy) in zip(losses, centres):
        r, image = np.hypot(x - cx, y - cy), np.hypot(x - cx, y + cy)
        T = T + W / (2 * np.pi * k_soil) * (np.log(image / np.maximum(r, r_c))
                                             + 0.5 * (1 - np.minimum(r, r_c) ** 2 / r_c**2))
    return T


# ### Post-processing on the section
#
# Conductor means and surface heat flows are integrated over the curved nine-node section cells
# with 4 x 4 Gauss points (the temperature and the heat flux are sampled at mid-thickness).

def lagrange3(s):
    return np.stack((2 * s * s - 3 * s + 1, 4 * s * (1 - s), 2 * s * s - s), axis=-1)


def lagrange3_d(s):
    return np.stack((4 * s - 3, 4 - 8 * s, 4 * s - 1), axis=-1)


G4, W4 = np.polynomial.legendre.leggauss(4)
G4, W4 = (G4 + 1) / 2, W4 / 2


def cell_quadrature(mesh, select):
    X = mesh.xy[mesh.quads[select]]                               # (n, 3, 3, 2)
    pts, wts = [], []
    for a_, wa in zip(G4, W4):
        for b_, wb in zip(G4, W4):
            La, Lb, dA, dB = lagrange3(a_), lagrange3(b_), lagrange3_d(a_), lagrange3_d(b_)
            x = np.einsum("i,j,nijd->nd", La, Lb, X)
            Jx, Jy = np.einsum("i,j,nijd->nd", dA, Lb, X), np.einsum("i,j,nijd->nd", La, dB, X)
            pts.append(x)
            wts.append(wa * wb * np.abs(Jx[:, 0] * Jy[:, 1] - Jx[:, 1] * Jy[:, 0]))
    return np.stack(pts, axis=1), np.stack(wts, axis=1)


def boundary_quadrature(mesh, y0):
    """Gauss points on the section edges lying on the line y = y0 (straight edges)."""
    pts, wts = [], []
    for quad in mesh.quads:
        for edge in (quad[:, 0], quad[:, 2], quad[0, :], quad[2, :]):
            q = mesh.xy[edge]
            if np.allclose(q[:, 1], y0):
                x0, x1 = q[0, 0], q[2, 0]
                pts += [(x0 + (x1 - x0) * g, y0) for g in G4]
                wts += list(abs(x1 - x0) * W4)
    return np.array(pts), np.array(wts)


def evaluate(field, xy, z):
    flat = xy.reshape(-1, 2)
    values = np.asarray(field.evaluate(np.column_stack((flat, np.full(len(flat), z)))))
    return values.reshape(xy.shape[:-1] + values.shape[1:])


def solve(mesh, losses, *, uniform=False, top="isothermal", h_air=None):
    begin = time.perf_counter()
    geometry = mesh.geometry
    model = HeatConduction(domain=geometry.domain(), material=LayeredGround(uniform),
                           heating=ConductorHeating(mesh, losses))
    b = geometry.boundaries
    if uniform:   # exactness check: the exact solution prescribed on the whole outer boundary
        conditions = (FixedTemperature(boundary=b.surface | b.bottom | b.sides, physics=model,
                                       value=lambda x, y, z: exact_uniform(np.asarray(x), np.asarray(y), losses)),
                      HeatFlux(boundary=b.slab_faces, physics=model, value=0.0))
    else:
        surface = (FixedTemperature(boundary=b.surface, physics=model, value=T_surface) if top == "isothermal" else
                   ConvectiveHeatFlux(boundary=b.surface, physics=model, heat_transfer_coefficient=h_air,
                                      ambient_temperature=T_surface))
        conditions = (surface, FixedTemperature(boundary=b.bottom, physics=model, value=T_surface),
                      HeatFlux(boundary=b.sides | b.slab_faces, physics=model, value=0.0))
    problem = Problem(name="Buried cables", geometry=geometry, physics=(model,), boundary_conditions=conditions)
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())).solve_stationary()
    T, q = result.field(model.temperature), result.field(model.heat_flux)
    z = mesh.thickness / 2
    conductor = []
    for c in centres:
        sel = (mesh.layers == 1) & (np.hypot(*(mesh.xy[mesh.quads[:, 1, 1]] - c).T) < r_c)
        pts, wts = cell_quadrature(mesh, sel)
        conductor.append(float(np.sum(evaluate(T, pts, z) * wts) / wts.sum()))
    phi = np.linspace(0, 2 * np.pi, 241)[:-1]
    jacket = [float(np.mean(evaluate(T, np.column_stack((cx + r_e * np.cos(phi), cy + r_e * np.sin(phi))), z)))
              for cx, cy in centres]
    top_pts, top_w = boundary_quadrature(mesh, 0.0)
    bot_pts, bot_w = boundary_quadrature(mesh, -mesh.a)
    q_top = evaluate(q, top_pts, z)[:, 1]
    q_bottom = -evaluate(q, bot_pts, z)[:, 1]
    return dict(T=T, conductor=np.array(conductor), jacket=np.array(jacket), Q_top=float(top_w @ q_top),
                Q_bottom=float(bot_w @ q_bottom), top_x=top_pts[:, 0], top_q=q_top, seconds=time.perf_counter() - begin)


# ## Solve
#
# For every mesh: two solves with unit losses, $\Phi_o$ (1 W/m in each outer cable) and
# $\Phi_m$ (1 W/m in the middle cable). Their conductor means give the response coefficients
# $a_{pq}$; the loss balance $\theta_{c,p}=\theta_s+\sum_q a_{pq}W_q$ with
# $W_q=I^2R'_{ac}(\theta_{c,q})$ is iterated to convergence, and every result is the
# superposition $\theta_s+W_o(\Phi_o-\theta_s)+W_m(\Phi_m-\theta_s)$.

def coupled_case(mesh, **options):
    outer = solve(mesh, (1.0, 0.0, 1.0), **options)
    middle = solve(mesh, (0.0, 1.0, 0.0), **options)
    theta = np.full(3, 70.0)
    for _ in range(100):
        W = joule_loss(theta)
        theta = T_surface + 0.5 * (W[0] + W[2]) * (outer["conductor"] - T_surface) + W[1] * (middle["conductor"] - T_surface)
    W = joule_loss(theta)
    Wo, Wm = 0.5 * (W[0] + W[2]), W[1]
    combine = lambda key: Wo * outer[key] + Wm * middle[key]
    return dict(theta=theta, W=W, Wo=Wo, Wm=Wm, outer=outer, middle=middle,
                jacket=T_surface + Wo * (outer["jacket"] - T_surface) + Wm * (middle["jacket"] - T_surface),
                Q_top=combine("Q_top"), Q_bottom=combine("Q_bottom"), top_x=outer["top_x"], top_q=combine("top_q"),
                seconds=outer["seconds"] + middle["seconds"], cells=mesh.cell_count, level=mesh.level, a=mesh.a, mesh=mesh)


mesh_study = []
for level in levels:
    case = coupled_case(CableMesh(SoilSlab(), main_domain, level))
    mesh_study.append(case)
    print(f"level {level:.1f}: {case['cells']:5d} cells, conductors " + ", ".join(f"{t:.4f}" for t in case["theta"])
          + f" C, losses " + ", ".join(f"{w:.3f}" for w in case["W"]) + f" W/m  ({case['seconds']:.0f} s)")
final = mesh_study[-1]

domain_study = {a: coupled_case(CableMesh(SoilSlab(), a, levels[1])) for a in domains[:-1]}
domain_study[main_domain] = mesh_study[1]

check_mesh = mesh_study[1]["mesh"]
exact_check = solve(check_mesh, tuple(final["W"]), uniform=True)

h_air = 10.0
convective = coupled_case(mesh_study[1]["mesh"], top="convective", h_air=h_air)
iec_convective = iec_temperatures(shift=k_soil / h_air)


# ## Comparison with the reference
#
# **IEC 60287 thermal circuit** against the finite-element solution on the finest mesh:
# conductor temperatures, mean temperature of the jacket surface and Joule losses. The circuit is
# not exact for this geometry: it treats each neighbour as a line source in homogeneous soil and
# each jacket surface as an isotherm, whereas the computed cables are poorly conducting XLPE and
# PE cylinders that slightly deflect the heat of their neighbours. Differences of a few tenths of
# a percent of the temperature rise are therefore expected; the soil block also ends at $a=40$ m,
# which the domain-size study below accounts for.

print("cable    conductor FEM  IEC 60287  diff [K]  rise dev  |  jacket surface FEM  IEC    diff [K]  |  W FEM   W IEC [W/m]")
for p, name in enumerate(("left", "middle", "right")):
    rise_dev = (final["theta"][p] - T_surface) / (iec["theta"][p] - T_surface) - 1
    print(f"{name:7s}  {final['theta'][p]:12.3f}  {iec['theta'][p]:9.3f}  {final['theta'][p] - iec['theta'][p]:+7.3f}  "
          f"{rise_dev:+8.2%}  |  {final['jacket'][p]:18.3f}  {iec['surface'][p]:6.3f}  {final['jacket'][p] - iec['surface'][p]:+7.3f}"
          f"   |  {final['W'][p]:6.3f}  {iec['W'][p]:6.3f}")

# **Exact image solution (check of the discretisation itself).** With every layer given the
# soil conductivity, the same losses, and the exact solution prescribed on the outer boundary,
# the computed field must reproduce the image solution. In conductor $p$ the exact mean is
# $\theta_s+\frac{W_p}{2\pi k}\big[\ln\frac{2L}{r_c}+\frac14\big]+\sum_{k\neq p}\frac{W_k}{2\pi k}\ln\frac{d'_{pk}}{d_{pk}}$.

W_v = final["W"]
exact_means = np.array([T_surface + W_v[p] / (2 * np.pi * k_soil) * (np.log(2 * depth / r_c) + 0.25)
                        + sum(W_v[k] / (2 * np.pi * k_soil)
                              * np.log(np.hypot(centres[p][0] - centres[k][0], 2 * depth) / abs(centres[p][0] - centres[k][0]))
                              for k in range(3) if k != p) for p in range(3)])
print("\nHomogeneous ground with exact boundary data (mesh level %.1f): conductor means" % levels[1])
for p in range(3):
    print(f"  cable {p}: FEM {exact_check['conductor'][p]:.5f} C, exact {exact_means[p]:.5f} C, deviation "
          f"{exact_check['conductor'][p] - exact_means[p]:+.1e} K ({(exact_check['conductor'][p] - exact_means[p]) / (exact_means[p] - T_surface):+.1e} of the rise)")
probe = np.column_stack((np.linspace(-0.6, 0.6, 241), np.full(241, -depth)))
probe = np.vstack((probe, np.column_stack((np.zeros(201), np.linspace(-0.001, -2.5, 201)))))
deviation = evaluate(exact_check["T"], probe, check_mesh.thickness / 2) - exact_uniform(probe[:, 0], probe[:, 1], W_v)
print(f"  largest deviation along the cable axis line and the vertical through the middle cable: "
      f"{np.max(np.abs(deviation)):.1e} K (peak rise {exact_means.max() - T_surface:.2f} K)")

# **Energy balance**: the heat leaving through the ground surface and the deep boundary must equal
# the Joule heat of the three conductors (fluxes from the computed temperature gradient).

for label, case in (("isothermal surface", final), ("convective surface", convective)):
    Q_in = case["W"].sum()
    Q_out = case["Q_top"] + case["Q_bottom"]
    print(f"\nEnergy balance ({label}): generated {Q_in:.4f} W/m; leaving through the surface {case['Q_top']:.4f} W/m "
          f"and the bottom {case['Q_bottom']:.4f} W/m; imbalance {Q_out / Q_in - 1:+.1e}")


def richardson(values, hs):
    f1, f2, f3 = values
    h1, h2, h3 = hs
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda q: (h1**q - h2**q) / (h2**q - h3**q) - ratio, 0.3, 12.0)
    except ValueError:
        return np.nan, np.nan
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1)


print(f"\nMesh study (soil block half-width = depth = {main_domain:.0f} m), conductor temperatures [C]:")
hs = 1.0 / np.asarray(levels)
for p, name in enumerate(("left", "middle", "right")):
    values = [c["theta"][p] for c in mesh_study]
    order, extrapolated = richardson(values, hs)
    text = (f", observed order {order:.2f}, Richardson {extrapolated:.4f}" if np.isfinite(order)
            else ", converged to round-off (no Richardson estimate)")
    print(f"  {name:6s} " + "  ".join(f"{v:.5f}" for v in values) + f"   change last two meshes {values[2] - values[1]:+.1e} K"
          + text + f"  (IEC {iec['theta'][p]:.3f})")

# **Domain size.** The IEC circuit describes a half-space; the computed soil block ends at the
# depth and half-width $a$, and its isothermal bottom draws off part of the heat. The conductor
# temperature therefore rises with $a$ towards the half-space value. A Richardson extrapolation
# in $1/a$ over the three largest blocks estimates that limit.

print("\nDomain-size study (mesh level %.1f), conductor temperatures [C] and heat share through the bottom:" % levels[1])
for a in domains:
    case = domain_study[a]
    print(f"  a = {a:4.0f} m: " + ", ".join(f"{t:.4f}" for t in case["theta"])
          + f"   bottom {case['Q_bottom'] / case['W'].sum():6.2%} of the heat")
half_space = []
for p, name in enumerate(("left", "middle", "right")):
    values = [domain_study[a]["theta"][p] for a in domains[-3:]]
    order, extrapolated = richardson(values, 1.0 / np.asarray(domains[-3:]))
    half_space.append(extrapolated)
    print(f"  {name:6s}: observed order in 1/a {order:.2f}, half-space estimate {extrapolated:.3f} C vs IEC 60287 "
          f"{iec['theta'][p]:.3f} C ({extrapolated - iec['theta'][p]:+.3f} K, "
          f"{(extrapolated - T_surface) / (iec['theta'][p] - T_surface) - 1:+.2%} of the rise)")
half_space = np.array(half_space)

print(f"\nConvective ground surface, h = {h_air:.0f} W/(m^2 K): FEM conductors "
      + ", ".join(f"{t:.3f}" for t in convective["theta"]) + " C;  IEC circuit with the isotherm moved down by k/h = "
      f"{k_soil / h_air:.2f} m (an engineering approximation): " + ", ".join(f"{t:.3f}" for t in iec_convective["theta"]) + " C")


# ## Figure
#
# Left: the temperature field of the cross-section with isotherms. Middle: close-up of the middle
# cable (copper, XLPE, lead, jacket), and the heat flux through the ground surface against the
# line-source-and-image formula $q(x)=\sum_n W_n L/[\pi((x-x_n)^2+L^2)]$. Right: deviation of the
# conductor temperatures from the IEC circuit in the mesh study and in the domain-size study.

mesh = final["mesh"]
xy, quads = mesh.xy, mesh.quads
used = np.zeros(len(xy), dtype=bool)
used[quads.ravel()] = True                # grid nodes inside the cable squares belong to no cell
node_T = np.full(len(xy), T_surface)
node_T[quads.reshape(len(quads), -1)] = (evaluate(final["outer"]["T"], xy[quads], mesh.thickness / 2) * final["Wo"]
                                         + evaluate(final["middle"]["T"], xy[quads], mesh.thickness / 2) * final["Wm"]
                                         - (final["Wo"] + final["Wm"] - 1) * T_surface).reshape(len(quads), -1)
sub = np.vstack([np.column_stack((quads[:, i, j], quads[:, i + 1, j], quads[:, i + 1, j + 1])) for i in (0, 1) for j in (0, 1)]
                + [np.column_stack((quads[:, i, j], quads[:, i + 1, j + 1], quads[:, i, j + 1])) for i in (0, 1) for j in (0, 1)])
tri = Triangulation(xy[:, 0], xy[:, 1], sub)

fig = plt.figure(figsize=(16, 9), constrained_layout=True)
layout = fig.add_gridspec(2, 3, width_ratios=(1.35, 0.8, 0.9))

ax = fig.add_subplot(layout[:, 0])
filled = ax.tricontourf(tri, node_T, levels=np.linspace(T_surface, node_T.max(), 61), cmap=FIELD_CMAP)
ax.tricontour(tri, node_T, levels=np.arange(22.0, node_T.max(), 4.0), colors=INK, linewidths=0.45)
ax.set_aspect("equal")
ax.set(xlim=(-1.2, 1.2), ylim=(-2.0, 0.0), xlabel="x / m", ylabel="y / m (ground surface at y = 0)",
       title=f"Cross-section temperature, I = {current:.0f} A (isotherms every 4 K)")
fig.colorbar(filled, ax=ax, label="T / C", shrink=0.6, pad=0.01, format="%.0f")

ax = fig.add_subplot(layout[0, 1])
near = (np.hypot(xy[:, 0], xy[:, 1] + depth) < 0.075) & used
filled = ax.tricontourf(tri, node_T, levels=np.linspace(node_T[near].min(), node_T.max(), 41), cmap=FIELD_CMAP, extend="min")
ax.tricontour(tri, node_T, levels=np.arange(np.ceil(node_T[near].min()), node_T.max(), 1.0), colors=INK, linewidths=0.4)
window = np.all(np.abs(xy[quads[:, 1, 1]] - centres[1]) < 0.08, axis=1)
for i in (0, 2):
    ax.plot(xy[quads[window, i, :], 0].T, xy[quads[window, i, :], 1].T, color=CREAM, lw=0.3, alpha=0.7)
    ax.plot(xy[quads[window, :, i], 0].T, xy[quads[window, :, i], 1].T, color=CREAM, lw=0.3, alpha=0.7)
ax.set_aspect("equal")
ax.set(xlim=(-0.07, 0.07), ylim=(-depth - 0.07, -depth + 0.07), xlabel="x / m", ylabel="y / m",
       title="Middle cable and its mesh (isotherms every 1 K)")
fig.colorbar(filled, ax=ax, label="T / C", shrink=0.85, format="%.0f")

ax = fig.add_subplot(layout[1, 1])
order = np.argsort(final["top_x"])
ax.plot(final["top_x"][order], final["top_q"][order], color=COPPER, lw=1.6, label="finite elements")
x_line = np.linspace(-3, 3, 601)
ax.plot(x_line, sum(W * depth / (np.pi * ((x_line - cx) ** 2 + depth**2)) for W, (cx, _) in zip(final["W"], centres)),
        ":", color=INK, lw=1.3, label="line sources + images")
ax.set(xlim=(-3, 3), xlabel="x / m", ylabel="heat flux out of the ground / (W/m^2)", title="Heat flux through the ground surface")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(layout[0, 2])
cells = [c["cells"] for c in mesh_study]
for p, name, colour, marker in ((0, "outer", GREEN, "o"), (1, "middle", COPPER, "s")):
    ax.semilogx(cells, [c["theta"][p] - iec["theta"][p] for c in mesh_study], marker + "-", color=colour,
                label=f"{name} conductor, a = {main_domain:.0f} m")
    ax.axhline(half_space[p] - iec["theta"][p], color=colour, ls=":", lw=1.0)
ax.axhline(0.0, color=INK, lw=0.8)
ax.set_xticks(cells, [str(c) for c in cells])
ax.minorticks_off()
ax.set(xlabel="number of cells", ylabel="FEM - IEC 60287 / K",
       title="Mesh study (dotted: half-space estimate)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(layout[1, 2])
sizes = np.array(domains)
for p, name, colour, marker in ((0, "outer", GREEN, "o"), (1, "middle", COPPER, "s")):
    ax.semilogx(sizes, [domain_study[a]["theta"][p] - iec["theta"][p] for a in domains], marker + "-", color=colour,
                label=f"{name} conductor")
    ax.plot([sizes[-1] * 1.6], [half_space[p] - iec["theta"][p]], marker, mfc="none", color=colour, ms=8)
ax.axhline(0.0, color=INK, lw=0.8, label="IEC 60287 (half-space)")
ax.set_xticks(list(sizes) + [sizes[-1] * 1.6], [f"{a:g}" for a in sizes] + ["extrap."])
ax.minorticks_off()
ax.set(xlabel="soil block half-width = depth a / m", ylabel="FEM - IEC 60287 / K",
       title=f"Domain-size study (mesh level {levels[1]:.1f})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.show()


# ## Going further
#
# Raise the current until the middle conductor reaches the 90 °C limit of XLPE: this is the
# rated current of the circuit, the quantity IEC 60287 is written for. Put the cables into a
# trench with a better-conducting backfill (one more layer index for the cells of the trench), or
# add the dielectric loss of the insulation and the eddy-current loss of the sheath as distributed
# sources. Drying of the soil around hot cables lowers its conductivity; a temperature-dependent
# soil conductivity turns the problem into a nonlinear one.
