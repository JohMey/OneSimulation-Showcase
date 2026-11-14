"""OneSimulation example: 72 hours in an insulated shipping box cooled by melting ice packs (3D).
Requires the installed, activated OneSimulation runtime.

A foam box carries temperature-sensitive goods (allowed range 2-8 degC). Four eutectic ice
plates behind foam layers keep the contents cold while the outside air follows a hot
summer-day cycle. The 3D transient temperature field with melting, latent heat storage,
isothermal air pockets and convective exchange with the ambient is computed with trilinear
hexahedral finite elements in enthalpy form (row-sum lumped capacity, Kirchhoff transform
for the phase-dependent conductivity) and OneSimulation's L-stable two-stage SDIRK2
integrator with Newton iterations. It is graded against the published probe temperatures
of this benchmark, the global energy balance and mesh and time-step studies.
"""


# # Isothermal shipping box: melting ice packs keep the goods at 2-8 °C
#
# Medicines and vaccines are shipped in foam boxes that must keep their contents between
# 2 °C and 8 °C for at least 24 hours. The cold source is a set of frozen eutectic plates
# (water with a thickener that suppresses convection once it has melted); a foam layer
# between plates and goods prevents the goods from dropping below 2 °C. While the ice
# melts, the plates stay close to 0 °C and absorb the heat leaking in through the walls.
#
# **Setup** (all lengths in cm; the published benchmark "isothermal box" of an industrial
# heat-transfer code, whose probe curves serve as reference below):
#
# | part | extent | k [W/(m K)] | ρ [kg/m³] | c_p [J/(kg K)] | initial T |
# |---|---|---|---|---|---|
# | foam box | 40 × 40 × 24, open top | 0.03 | 25 | 2 | 20 °C |
# | foam lid | 40 × 40 × 4 on top | 0.03 | 25 | 2 | 20 °C |
# | contents (generic material) | 24 × 24 × 20, centred, flush with the top of the box | 1 | 2000 | 800 | 5 °C |
# | 4 eutectic plates | 2 thick, 24 wide, 20 high, 2 cm foam between plate and contents | ice 2.3 / water 0.6 | 1000 | ice 2050 / water 4200 | −5 °C |
# | 4 air pockets | 4 × 4 × 20 at the corners of the contents | isothermal air, film coefficient 5 W/(m² K) on all its faces | 1.2 | 1006 | 20 °C |
#
# The plates melt at $T_{pc}=0$ °C with latent heat $L=333$ kJ/kg over a transition
# interval $\Delta T=3.5$ K (apparent-heat-capacity model with a smoothed step). All outer
# faces of box and lid exchange heat with the ambient air by convection,
# $q=h\,(T_\infty(t)-T)$ with $h=5$ W/(m² K), where $T_\infty(t)$ is the typical hourly
# temperature of a summer day (1 June, from 06:00) at the Seville weather station of the
# ASHRAE climatic data (ASHRAE Handbook - Fundamentals 2021, ch. 14, station 083910),
# swinging between 18 °C and 30 °C; it is tabulated below every 3 h for 72 h and
# interpolated linearly.
#
# The temperature obeys the heat equation in enthalpy form,
#
# $$\frac{\partial H(T)}{\partial t}=\nabla\cdot\big(k(T)\nabla T\big),\qquad
#   H(T)=\int_0^T\rho c_p\,\mathrm{d}T+\rho L\,\big(\theta(T)-\theta(0)\big),$$
#
# with $\rho c_p=\rho\,[(1-\theta)c_{p,\rm ice}+\theta\,c_{p,\rm water}]$ and
# $k=(1-\theta)k_{\rm ice}+\theta k_{\rm water}$ in the plates, where the liquid fraction
# $\theta(T)$ rises smoothly from 0 to 1 across $T_{pc}\pm\Delta T/2$ (quintic smoothed
# step, twice continuously differentiable; the apparent-heat-capacity method of
# C. Bonacina, G. Comini, A. Fasano and M. Primicerio, *Int. J. Heat Mass Transfer* 16,
# 1825, 1973). Each air pocket is one well-mixed temperature $T_a$ with
# $C_a\,\dot T_a=\int h\,(T-T_a)\,\mathrm{d}A$ over its walls.
#
# The field is three-dimensional: heat enters through the six faces of the box, flows
# around the plates through the foam, is absorbed by the plates and reaches the contents
# through the inner foam layers, the lid and the floor; the corners and the air pockets
# make it vary in all three directions. The box is symmetric about its two vertical
# mid-planes, so one quarter ($0\le x,y\le 20$ cm) is computed; heat flows and energies
# are multiplied by four, and the figures are mirrored. Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | outer faces of box and lid ($x=0$, $y=0$, bottom $z=0$, top $z=28$ cm) | convection $h\,(T_\infty(t)-T)$, $h=5$ W/(m² K) |
# | walls of the air pocket | convection $h\,(T_a-T)$ to the isothermal pocket air, $h=5$ W/(m² K) |
# | symmetry planes $x=20$ cm and $y=20$ cm | symmetry (zero flux) |
#
# The thin plastic film around the plates (0.3 mm, k = 0.3 W/(m K)) and the thin air gaps
# around contents and lid (50 µm, k = 0.025 W/(m K)) of the original model add contact
# resistances of 0.001 and 0.002 m² K/W. They are neglected here: the foam layers they sit
# on have 0.67 to 1.33 m² K/W, so they change the heat flows by less than 0.3 %.
#
# **Numerical method.**
#
# * Space: continuous Galerkin finite elements with **trilinear 8-node hexahedra (Q1)** on a
#   block mesh that follows every material interface; OneSimulation assembles the
#   conductivity, capacity and boundary (Robin) matrices of each material separately.
# * Phase change: **enthalpy formulation with row-sum lumped capacity** - the nodal
#   unknown of the time integrator is the stored heat $E_i=c_iT_i+V_i^{\rm ice}H(T_i)$, so
#   the latent heat is conserved exactly and the sharp capacity peak cannot cause
#   oscillations; the phase-dependent plate conductivity enters through the Kirchhoff
#   transform $\phi(T)=\int k\,\mathrm{d}T$ interpolated with the Q1 basis (group finite
#   elements, C. A. J. Fletcher, *Comput. Methods Appl. Mech. Eng.* 37, 225, 1983).
# * Time: OneSimulation's **two-stage, L-stable, stiffly accurate SDIRK2** integrator
#   ($\gamma=1-1/\sqrt2$; the implicit half of the ARS(2,2,2) scheme of U. M. Ascher,
#   S. J. Ruuth and R. J. Spiteri, *Appl. Numer. Math.* 25, 151, 1997), run with a fixed
#   step of 30 min.
# * Nonlinear and linear solvers: Newton's method with the exact Jacobian in every stage;
#   in the plate nodes the Newton unknown is the stored heat. The Jacobian systems are
#   solved by GMRES preconditioned with a sparse LU factorization of an earlier Jacobian,
#   refreshed when GMRES slows down.
#
# **Reference.** The published probe temperatures of this benchmark (minimum, average and
# maximum temperature of the contents and average temperature of the plates over 72 h),
# read from its published curves to about ±0.2 K, together with its statements that the
# contents stay inside 2-8 °C for the first 24 h, that the plates stay close to 0 °C for
# about a day and that the contents exceed 8 °C only after about 48 h. Further checks:
# the global energy balance (heat entering through the walls equals the change of
# sensible and latent heat), the latent-heat budget of the plates, and a three-level mesh
# study and a time-step study (60, 30, 15 min) with observed orders and Richardson estimate.

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
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES, affine_hexahedral_geometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time.methods.backends.imex_sdirk2 import AdaptiveIMEXSDIRK2Provider, IMEXSDIRK2Options
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters

k_foam, rho_foam, cp_foam = 0.03, 25.0, 2.0              # foam box and lid
k_cont, rho_cont, cp_cont = 1.0, 2000.0, 800.0           # contents
k_ice, cp_ice, k_wat, cp_wat, rho_w = 2.3, 2050.0, 0.6, 4200.0, 1000.0   # eutectic plates
T_pc, dT_pc, L_f = 0.0, 3.5, 333e3                       # degC, K, J/kg
rho_air, cp_air = 1.204, 1006.0                          # pocket air
h_out, h_pocket = 5.0, 5.0                               # W/(m^2 K)
T0_box, T0_contents, T0_plates = 20.0, 5.0, -5.0         # degC
V_pocket = 0.04 * 0.04 * 0.20                            # one air pocket, m^3
t_end_h = 72.0

# Hourly design-day temperature of Seville (1 June, from 06:00), every 3 h, degC
t_amb_h = np.arange(0.0, 73.0, 3.0)
T_amb_tab = np.array([23.1, 27.6, 29.8, 28.5, 23.4, 20.1, 18.1, 17.9, 23.2, 27.7, 29.95, 28.7, 23.6,
                      20.25, 18.25, 18.05, 23.3, 27.85, 30.1, 28.8, 23.75, 20.4, 18.35, 18.15, 23.45])


def T_ambient(t):
    """Ambient air temperature [degC] at time t [s]."""
    return np.interp(t / 3600.0, t_amb_h, T_amb_tab)


# Published probe temperatures of the benchmark, read from its curves (degC)
ref_t = np.array([1, 10, 22, 30, 36, 45, 48, 51, 55, 60, 65, 72], float)
REF = {
    "contents minimum": np.array([4.6, 4.85, 4.2, 5.0, 5.4, 5.4, 5.8, 6.3, 7.5, 9.3, 10.8, 12.6]),
    "contents average": np.array([5.5, 5.65, 4.85, 5.6, 5.95, 5.85, 6.3, 7.0, 8.0, 9.8, 11.0, 12.9]),
    "contents maximum": np.array([6.1, 6.4, 5.45, 6.35, 6.55, 6.3, 7.0, 8.3, 9.9, 11.3, 12.2, 14.0]),
    "plates average": np.array([-1.4, -0.6, -0.15, 0.3, 0.9, 2.0, 3.0, 5.0, 7.5, 11.2, 12.8, 14.5]),
}
REF_EVENTS = {"plates average reaches 2 degC": 45.0, "contents maximum reaches 8 degC": 50.5,
              "contents average reaches 8 degC": 55.0}

latent_quarter = rho_w * L_f * 2 * 0.02 * 0.24 / 2 * 0.20      # J, latent heat of the plates in one quarter
print(f"latent heat of the four plates: {4 * latent_quarter / 1e6:.3f} MJ "
      f"({4 * rho_w * 2 * 0.02 * 0.24 / 2 * 0.20:.2f} kg of ice)")
print(f"thermal resistance of a 4 cm foam wall incl. film: {1 / h_out + 0.04 / k_foam:.3f} m^2 K/W")


# ## Geometry and mesh
#
# The quarter box is covered by a tensor-product block mesh whose grid lines pass through
# every material interface: in $x$ and $y$ the breaks are 0, 4, 6, 8 and 20 cm (outer
# foam, plate, inner foam, contents), in $z$ 0, 4, 24 and 28 cm (floor, contents and
# plates, lid). Every hexahedron belongs to exactly one material; the air pocket is left
# out of the mesh, its walls are a boundary of the solid. A mesh level $m$ has
# $2m, 2m, m, 4m$ cells across the four $x$ (and $y$) segments and $2m, 8m, 2m$ cells over
# the three $z$ segments, so on the finest level ($m=3$) a plate is 6 cells thick, the
# 4 cm foam walls 6 cells and the contents 12 x 12 x 24 cells (cell sizes 3.3-10 mm).
#
# OneSimulation assembles each material on its own part of the mesh: the foam (box and
# lid together; the thin air gap between them is negligible, see above), the contents
# and the plates, the latter with unit conductivity and capacity so that the
# phase-dependent values can be applied afterwards. The boundary faces of every part are
# named by what lies beyond them: the outside, the air pocket, a symmetry plane, or
# another material (an internal interface, where the parts are joined node by node and
# no condition is imposed).

X_BREAKS = (0.0, 0.04, 0.06, 0.08, 0.20)
Z_BREAKS = (0.0, 0.04, 0.24, 0.28)
FOAM, CONTENTS, PLATES, POCKET, OUTSIDE = 0, 1, 2, 3, 4


def material_at(x, y, z):
    """Material index at points strictly inside the quarter box (or OUTSIDE)."""
    x, y, z = map(np.asarray, (x, y, z))
    middle = (z > 0.04) & (z < 0.24)
    contents = middle & (x > 0.08) & (y > 0.08)
    plates = middle & (((x > 0.04) & (x < 0.06) & (y > 0.08)) | ((y > 0.04) & (y < 0.06) & (x > 0.08)))
    pocket = middle & (x > 0.04) & (x < 0.08) & (y > 0.04) & (y < 0.08)
    outside = (x < 0) | (y < 0) | (z < 0) | (z > 0.28) | (x > 0.2) | (y > 0.2)
    return np.where(outside, OUTSIDE, np.where(contents, CONTENTS, np.where(plates, PLATES,
                    np.where(pocket, POCKET, FOAM))))


def axis(breaks, counts):
    return np.concatenate([[breaks[0]]] + [np.linspace(a, b, n + 1)[1:]
                                            for a, b, n in zip(breaks[:-1], breaks[1:], counts)])


@dataclass(frozen=True, eq=False, kw_only=True)
class BoxPart(Geometry):
    """One material of the quarter box with its named boundary faces."""

    dimension = 3
    names: tuple = ("outside", "pocket", "symmetry", "interface")

    def boundary_groups(self):
        return {**{name: (name,) for name in self.names}, "all": self.names}


class PartMesh(StructuredHexahedralMesh):
    """Hex8 block mesh of the cells of one material of the quarter box."""

    def __init__(self, xs, zs, part):
        cx, cz = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (zs[1:] + zs[:-1])
        I, J, K = np.nonzero(material_at(*np.meshgrid(cx, cx, cz, indexing="ij")) == part)
        corners = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
        used = np.zeros((len(xs), len(xs), len(zs)), bool)
        for p, q, r in corners:
            used[I + p, J + q, K + r] = True
        index = -np.ones(used.shape, dtype=np.int64)
        index[used] = np.arange(used.sum())
        ii, jj, kk = np.nonzero(used)
        data = affine_hexahedral_geometry(np.column_stack((xs[ii], xs[jj], zs[kk])),
                                          np.column_stack([index[I + p, J + q, K + r] for p, q, r in corners]))
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (len(xs) - 1, len(xs) - 1, len(zs) - 1)
        self.mesh_id = f"quarter-box-{part}:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        faces = np.concatenate([self.cells[:, list(face)] for face in FACES])
        owner = np.tile(np.arange(self.cell_count), len(FACES))
        _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
        single = counts[inverse.ravel()] == 1
        groups = {name: [] for name in BoxPart().names}
        centres = self.coordinates[self.cells].mean(axis=1)
        for face, cell in zip(faces[single], owner[single]):
            centre = self.coordinates[face].mean(axis=0)
            if np.isclose(centre[0], 0.2) or np.isclose(centre[1], 0.2):
                name = "symmetry"
            else:
                beyond = material_at(*(centre + 1e-6 * (centre - centres[cell]) / np.linalg.norm(centre - centres[cell])))
                name = {OUTSIDE: "outside", POCKET: "pocket"}.get(int(beyond), "interface")
            groups[name].append(tuple(face))
        names = tuple(name for name in groups if groups[name])
        self.geometry = BoxPart(names=names)
        self._facets = {name: groups[name] for name in names}
        self._carrier = self._build_carrier()


# ## Model
#
# **Assembly.** For each part a `HeatConduction` problem with Q1 Lagrange elements is
# compiled and its conductivity matrix, capacity matrix and the Robin matrices of the
# `ConvectiveHeatFlux` conditions are taken from OneSimulation's weak assembly; the
# boundary matrices are separated by assembling once with and once without the film
# coefficient. Every boundary group of every part carries an explicit condition: a
# `ConvectiveHeatFlux` on the outside and pocket walls (where the exchange is wanted) or
# a zero `HeatFlux` (symmetry planes, internal interfaces, and the reference assembly).

def assemble(mesh, k, rho_cp, outside=0.0, pocket=0.0):
    material = IsotropicSolid(density=1.0, specific_heat_capacity=rho_cp, thermal_conductivity=k,
                              youngs_modulus=1e9, poisson_ratio=0.3, thermal_expansion_coefficient=0.0)
    geometry = mesh.geometry
    model = HeatConduction(domain=geometry.domain(), material=material, heating=0.0)
    conditions = []
    for name in geometry.names:
        h = {"outside": outside, "pocket": pocket}.get(name, 0.0)
        selection = getattr(geometry.boundaries, name)
        conditions.append(ConvectiveHeatFlux(boundary=selection, physics=model, heat_transfer_coefficient=h,
                                             ambient_temperature=0.0) if h > 0
                          else HeatFlux(boundary=selection, physics=model, value=0.0))
    problem = Problem(name="Isothermal box: operator assembly", geometry=geometry, physics=(model,),
                      boundary_conditions=tuple(conditions))
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=1)), linear_solver=SparseDirect()))
    operator = simulation.compile()._operator(model, {})
    weak, system, _, _ = operator.weak_assembly(0.0)
    return system.matrix.tocsr(), weak.mass.matrix.tocsr(), np.asarray(operator.space.dof_coordinates)


def build_level(m):
    start = time.time()
    xs, zs = axis(X_BREAKS, (2 * m, 2 * m, m, 4 * m)), axis(Z_BREAKS, (2 * m, 8 * m, 2 * m))
    shape = (len(xs), len(xs), len(zs))
    key = lambda X: np.ravel_multi_index((np.searchsorted(xs, X[:, 0] - 1e-9), np.searchsorted(xs, X[:, 1] - 1e-9),
                                          np.searchsorted(zs, X[:, 2] - 1e-9)), shape)
    parts = {part: PartMesh(xs, zs, part) for part in (FOAM, CONTENTS, PLATES)}
    keys = np.unique(np.concatenate([key(mesh.coordinates) for mesh in parts.values()]))
    n = len(keys)
    nodes = np.column_stack([v[i] for v, i in zip((xs, xs, zs), np.unravel_index(keys, shape))])

    def to_global(A, X):
        index = np.searchsorted(keys, key(X))
        A = A.tocoo()
        return sparse.csr_matrix((A.data, (index[A.row], index[A.col])), shape=(n, n))

    K0, M0, X = assemble(parts[FOAM], k_foam, rho_foam * cp_foam)
    K1, _, _ = assemble(parts[FOAM], k_foam, rho_foam * cp_foam, outside=h_out)
    K2, _, _ = assemble(parts[FOAM], k_foam, rho_foam * cp_foam, pocket=h_pocket)
    ops = dict(K_foam=to_global(K0, X), M_foam=to_global(M0, X), B_out=to_global(K1 - K0, X),
               B_pocket=to_global(K2 - K0, X))
    K0, M0, X = assemble(parts[CONTENTS], k_cont, rho_cont * cp_cont)
    ops.update(K_contents=to_global(K0, X), M_contents=to_global(M0, X))
    K0, M0, X = assemble(parts[PLATES], 1.0, 1.0)
    K2, _, _ = assemble(parts[PLATES], 1.0, 1.0, pocket=h_pocket)
    ops.update(K_plates=to_global(K0, X), M_plates=to_global(M0, X))
    ops["B_pocket"] = ops["B_pocket"] + to_global(K2 - K0, X)
    for name, A in ops.items():
        A = ((A + A.T) * 0.5).tocsr()
        A.eliminate_zeros()
        ops[name] = A
    volume = {}
    for part, name in ((FOAM, "foam"), (CONTENTS, "contents"), (PLATES, "plates")):
        mesh, w = parts[part], np.zeros(n)
        np.add.at(w, np.searchsorted(keys, key(mesh.coordinates))[mesh.cells].ravel(), np.repeat(mesh.cell_volumes / 8, 8))
        volume[name] = w
    return dict(m=m, xs=xs, zs=zs, nodes=nodes, n=n, ops=ops, volume=volume,
                cells=sum(mesh.cell_count for mesh in parts.values()), seconds=time.time() - start)


# **Phase change.** Liquid fraction $\theta(T)$ (quintic smoothed step over
# $T_{pc}\pm\Delta T/2$), its integral $G(T)=\int_{-\infty}^T\theta\,\mathrm{d}T$, the
# volumetric enthalpy $H(T)$ of the plates and the Kirchhoff potential $\phi(T)$, all
# relative to 0 °C and in closed form.

def liquid_fraction(T):
    x = np.clip((T - T_pc) / dT_pc + 0.5, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x * x), 30 * x**2 * (1 - x) ** 2 / dT_pc


def fraction_integral(T):
    x = (T - T_pc) / dT_pc + 0.5
    xc = np.clip(x, 0.0, 1.0)
    return dT_pc * (xc**6 - 3 * xc**5 + 2.5 * xc**4) + dT_pc * np.maximum(x - 1.0, 0.0)


G_0, THETA_0 = float(fraction_integral(np.array(0.0))), float(liquid_fraction(np.array(0.0))[0])


def plate_enthalpy(T):
    theta, dtheta = liquid_fraction(T)
    H = rho_w * (cp_ice * T + (cp_wat - cp_ice) * (fraction_integral(T) - G_0)) + rho_w * L_f * (theta - THETA_0)
    return H, rho_w * (cp_ice + (cp_wat - cp_ice) * theta) + rho_w * L_f * dtheta


def plate_kirchhoff(T):
    theta, _ = liquid_fraction(T)
    return k_ice * T + (k_wat - k_ice) * (fraction_integral(T) - G_0), k_ice + (k_wat - k_ice) * theta


# **Semi-discrete system.** With the nodal temperatures $T$ and the pocket temperature
# $T_a$ the state of the integrator is the stored heat
# $E=\big(c\,T+V^{\rm pl}H(T),\ C_aT_a\big)$, with the lumped (row-sum) capacities $c$ of
# foam and contents and the lumped plate volumes $V^{\rm pl}$, and
#
# $$\dot E=-\big(K_f+K_c+B_\infty+B_a\big)T-K_{pl}\,\phi(T)+b_\infty T_\infty(t)+b_aT_a,
#   \qquad C_a\dot T_a=b_a^{\mathsf T}T-s_aT_a ,$$
#
# where $K$ are the conductivity matrices, $B$ the Robin matrices of the outside and
# pocket walls, $b=B\mathbf 1$ and $s_a=\mathbf 1^{\mathsf T}B_a\mathbf 1$. The rows of
# all conductivity matrices sum to zero, so $\sum\dot E$ is exactly the heat flow through
# the outer walls: the discrete energy balance holds by construction.

class QuarterBoxModel:
    def __init__(self, level):
        o = level["ops"]
        self.K = (o["K_foam"] + o["K_contents"] + o["B_out"] + o["B_pocket"]).tocsr()
        self.K_plates = o["K_plates"]
        self.b_out = np.asarray(o["B_out"].sum(axis=1)).ravel()
        self.b_pocket = np.asarray(o["B_pocket"].sum(axis=1)).ravel()
        self.s_pocket = self.b_pocket.sum()
        self.c = np.asarray(o["M_foam"].sum(axis=1)).ravel() + np.asarray(o["M_contents"].sum(axis=1)).ravel()
        self.v_plates = np.asarray(o["M_plates"].sum(axis=1)).ravel()
        self.plate_nodes = np.nonzero(self.v_plates > 0)[0]
        self.C_pocket = rho_air * cp_air * V_pocket
        self.scale = np.r_[self.c + self.v_plates * rho_w * cp_wat, self.C_pocket]   # J/K, for error norms

    def energy(self, T):
        return np.r_[self.c * T[:-1] + self.v_plates * plate_enthalpy(T[:-1])[0], self.C_pocket * T[-1]]

    def temperature(self, E, guess):
        """Invert E(T) node by node (linear except in the plates, Newton there)."""
        T = np.r_[E[:-1] / (self.c + self.v_plates * rho_w * cp_ice), E[-1] / self.C_pocket]
        p = self.plate_nodes
        Tp = guess[p].copy()
        for _ in range(100):
            H, dH = plate_enthalpy(Tp)
            step = (self.c[p] * Tp + self.v_plates[p] * H - E[p]) / (self.c[p] + self.v_plates[p] * dH)
            Tp -= np.clip(step, -2.0, 2.0)
            if np.max(np.abs(step)) < 1e-11:
                break
        T[p] = Tp
        return T

    def rate(self, T, t):
        phi, _ = plate_kirchhoff(T[:-1])
        f = -(self.K @ T[:-1]) - self.K_plates @ phi + self.b_out * T_ambient(t) + self.b_pocket * T[-1]
        return np.r_[f, self.b_pocket @ T[:-1] - self.s_pocket * T[-1]]

    def heat_in(self, T, t):
        """Heat flow through the outer walls of the quarter, W."""
        return float(self.b_out.sum() * T_ambient(t) - self.b_out @ T[:-1])

    def jacobian(self, T, a):
        """d/dT of E(T) - a * rate(T)."""
        _, dH = plate_enthalpy(T[:-1])
        _, k = plate_kirchhoff(T[:-1])
        J = sparse.diags(self.c + self.v_plates * dH) + a * (self.K + self.K_plates @ sparse.diags(k))
        return sparse.bmat([[J, -a * self.b_pocket[:, None]],
                            [-a * self.b_pocket[None, :], np.array([[self.C_pocket + a * self.s_pocket]])]], format="csc")


class StoredHeatAlgebra:
    """State algebra of the integrator: linear combinations, and a max-norm of the stored
    heat per node expressed in kelvin (|dE_i| / heat capacity of the node)."""

    def __init__(self, scale):
        self.scale = scale

    def weighted_sum(self, states, weights):
        return sum(w * s for s, w in zip(states, weights))

    def distance(self, a, b):
        return float(np.max(np.abs(a - b) / self.scale))


# ## Solve
#
# The integrator advances the stored heat $E$ with a fixed step $\Delta t$ (30 min on all
# meshes; the step-size control of the integrator is switched off by a very loose
# tolerance, and the time-step study below checks the step instead). In each of the two
# implicit stages $E=E^\ast+a\,\dot E(T)$, $a=\gamma\Delta t$, is solved by Newton's
# method until the temperature update is below $10^{-5}$ K, starting from the linear
# extrapolation of the last two steps. In the plate nodes the Newton unknown is the
# stored heat rather than the temperature (the temperature Jacobian is scaled by
# $\mathrm dT/\mathrm dE$), because $T(E)$ is smooth where the apparent capacity has its
# sharp peak; this keeps Newton from oscillating across the melting range (B. Nedjar,
# *Comput. Struct.* 80, 9, 2002). Because $a$ is constant, one sparse LU
# factorization of the Jacobian serves as preconditioner of GMRES for many steps; it is
# refreshed only when the latent capacity of the melting plates has changed enough that
# GMRES needs more than 20 iterations. All steps end on the hourly outputs, which
# include every kink of the ambient table.

GAMMA = 1.0 - 1.0 / np.sqrt(2.0)


def integrate(level, dt_min=30.0, keep_h=(24.0,)):
    start = time.time()
    model = QuarterBoxModel(level)
    vol = level["volume"]
    in_contents, in_plates = vol["contents"] > 0, vol["plates"] > 0
    T = np.full(level["n"] + 1, T0_box)
    T[:-1][in_contents] = T0_contents       # closed regions: the contents and plates set their own nodes
    T[:-1][in_plates] = T0_plates
    E = model.energy(T)
    dt = 60.0 * dt_min
    loose = 1e9                              # step-size control off: every step is accepted
    integrator = AdaptiveIMEXSDIRK2Provider(
        IMEXSDIRK2Options(relative_tolerance=1e-12, absolute_tolerance=loose, minimum_step=dt, maximum_step=dt),
        algebra=StoredHeatAlgebra(model.scale))
    memory = dict(stage_q=[], lu=None, history=[(-dt, T.copy()), (0.0, T.copy())])
    counts = dict(newton=0, krylov=0, factorizations=0, max_iterations=0)

    def implicit_rate(state, t):
        return model.rate(model.temperature(state, memory["history"][1][1]), t)

    def no_explicit_rate(state, t):
        return np.zeros_like(state)

    def stage_solver(base, a, rate, t, guess):
        (t0, T_0), (t1, T_1) = memory["history"]
        Tk = T_1 + (t - t1) / (t1 - t0) * (T_1 - T_0) if counts["newton"] else T_1.copy()
        plates = model.plate_nodes
        for iteration in range(1, 40):
            Ek = model.energy(Tk)
            residual = Ek - base - a * model.rate(Tk, t)
            # Newton in the stored heat of the plate nodes (smooth in the melting range) and in
            # the temperature elsewhere: right-scale the temperature Jacobian by dT/dE.
            _, dH = plate_enthalpy(Tk[plates])
            scale = np.ones(len(Tk))
            scale[plates] = 1.0 / (model.c[plates] + model.v_plates[plates] * dH)
            J = (model.jacobian(Tk, a) @ sparse.diags(scale)).tocsc()
            if memory["lu"] is None:
                memory["lu"] = splu(J)
                counts["factorizations"] += 1
            calls = [0]
            du, info = gmres(J, -residual, M=LinearOperator(J.shape, memory["lu"].solve), rtol=1e-10, atol=0.0,
                             restart=40, maxiter=2, callback=lambda r: calls.__setitem__(0, calls[0] + 1),
                             callback_type="pr_norm")
            if info != 0 or calls[0] > 20:      # Jacobian has drifted: refresh the factorization
                memory["lu"] = splu(J)
                counts["factorizations"] += 1
                if info != 0:
                    du = memory["lu"].solve(-residual)
            counts["newton"] += 1
            counts["krylov"] += calls[0]
            T_new = Tk + du
            E_new = Ek.copy()
            E_new[plates] += du[plates]
            T_new[plates] = model.temperature(E_new, Tk)[plates]
            change = np.max(np.abs(T_new - Tk))
            Tk = T_new
            if change < 1e-5:                   # quadratic convergence: the remaining error is ~1e-10 K
                break
        counts["max_iterations"] = max(counts["max_iterations"], iteration)
        memory["stage_q"].append(model.heat_in(Tk, t))
        return model.energy(Tk), iteration

    weights_c, weights_p = vol["contents"] / vol["contents"].sum(), vol["plates"] / vol["plates"].sum()
    probes = {name: [] for name in ("t", "cmin", "cavg", "cmax", "pavg", "melt", "q_in", "T_pocket", "E")}

    def record(t, E, T):
        probes["t"].append(t / 3600)
        probes["cmin"].append(T[:-1][in_contents].min())
        probes["cavg"].append(weights_c @ T[:-1])
        probes["cmax"].append(T[:-1][in_contents].max())
        probes["pavg"].append(weights_p @ T[:-1])
        probes["melt"].append(weights_p @ liquid_fraction(T[:-1])[0])
        probes["q_in"].append(model.heat_in(T, t))
        probes["T_pocket"].append(T[-1])
        probes["E"].append(E.sum())

    record(0.0, E, T)
    kept, heat_in = {}, 0.0
    steps = int(round(t_end_h * 3600 / dt))
    for n in range(steps):
        t = n * dt
        memory["stage_q"] = []
        outcome = integrator.advance(E, no_explicit_rate, implicit_rate, t, dt, implicit_stage_solver=stage_solver)
        q1, q2 = memory["stage_q"]
        heat_in += dt * ((1 - GAMMA) * q1 + GAMMA * q2)      # heat in, with the weights of the scheme
        E, t = outcome.state, outcome.end_time
        T = model.temperature(E, memory["history"][1][1])
        memory["history"] = [memory["history"][1], (t, T.copy())]
        hours = t / 3600
        if abs(hours - round(hours)) < 1e-9:
            record(t, E, T)
            if any(abs(hours - h) < 1e-9 for h in keep_h):
                kept[round(hours)] = T.copy()
    probes = {name: np.array(values) for name, values in probes.items()}
    return dict(level=level, model=model, probes=probes, kept=kept, heat_in=heat_in, steps=steps, counts=counts,
                seconds=time.time() - start)


# Three mesh levels with $\Delta t=30$ min, and the middle level once more with 60 and
# 15 min (time-step study).

levels = {m: build_level(m) for m in (1, 2, 3)}
runs = {}
for m, level in levels.items():
    runs[m] = integrate(level)
    r = runs[m]
    print(f"level m = {m}: {level['n']:6d} nodes, {level['cells']:6d} hexahedra, assembly {level['seconds']:4.0f} s; "
          f"{r['steps']} steps, {r['counts']['newton']} Newton iterations (at most {r['counts']['max_iterations']} per stage), "
          f"{r['counts']['krylov']} GMRES iterations, {r['counts']['factorizations']} LU factorizations, {r['seconds']:4.0f} s")
time_study = {dt_min: integrate(levels[2], dt_min=dt_min) for dt_min in (60.0, 15.0)}
time_study[30.0] = runs[2]
print("time-step study on level 2: " + ", ".join(f"dt = {k:4.1f} min: {v['seconds']:.0f} s" for k, v in time_study.items()))
fine = runs[3]
P = fine["probes"]


# ## Comparison with the reference
#
# **Probe temperatures** on the finest mesh against the published curves (read to about
# ±0.2 K; on the steep parts after 48 h a reading error of ±0.5 h in time corresponds to
# up to ±0.4 K). Beyond the reading error the comparison is limited by the ambient
# profile, which is itself read from the published 3-hourly curve, by the thin films that
# are neglected here, and by the start of the published run: its contents maximum begins
# at about 17 °C, so its surface nodes of the contents start warmer than 5 °C, which adds
# some heat in the first hours.

def at(probes, key, hours):
    return np.interp(hours, probes["t"], probes[key])


def crossing(probes, key, level, after=20.0):
    """First time after `after` hours at which the probe rises through `level`."""
    t, y = probes["t"], probes[key]
    i = np.nonzero((t[1:] > after) & (y[:-1] < level) & (y[1:] >= level))[0][0]
    return t[i] + (level - y[i]) * (t[i + 1] - t[i]) / (y[i + 1] - y[i])


KEYS = {"contents minimum": "cmin", "contents average": "cavg", "contents maximum": "cmax", "plates average": "pavg"}
print("  t [h] " + "".join(f"| {name:^27s}" for name in KEYS))
print("        " + "| computed  published  diff  " * len(KEYS))
for i, hours in enumerate(ref_t):
    row = f"  {hours:4.0f}  "
    for name, key in KEYS.items():
        value = at(P, key, hours)
        row += f"|  {value:6.2f}    {REF[name][i]:6.2f}  {value - REF[name][i]:+5.2f} "
    print(row)
for name, key in KEYS.items():
    d = at(P, key, ref_t) - REF[name]
    print(f"{name:18s}: mean deviation {d.mean():+.2f} K, largest |deviation| {np.abs(d).max():.2f} K")

# **Events and the 2-8 °C window.**

events = {"plates average reaches 2 degC": crossing(P, "pavg", 2.0),
          "contents maximum reaches 8 degC": crossing(P, "cmax", 8.0),
          "contents average reaches 8 degC": crossing(P, "cavg", 8.0)}
for name, value in events.items():
    print(f"{name:32s}: computed {value:5.1f} h, published about {REF_EVENTS[name]:4.1f} h, "
          f"difference {value - REF_EVENTS[name]:+.1f} h")
first_day = (P["t"] > 0) & (P["t"] <= 24)
print(f"contents during the first 24 h: {P['cmin'][first_day].min():.2f} to {P['cmax'][first_day].max():.2f} degC "
      f"(allowed 2 to 8 degC; published: inside the window)")
print(f"plates average after 1 h / 12 h / 24 h: {at(P, 'pavg', 1):.2f} / {at(P, 'pavg', 12):.2f} / "
      f"{at(P, 'pavg', 24):.2f} degC (published: about -1.4 / -0.5 / -0.1 degC)")

# **Energy balance** of the whole box (four quarters). The change of stored sensible and
# latent heat over 72 h is compared with the heat that entered through the walls,
# integrated (i) with the stage weights of the integrator - this must agree to round-off
# because every conductivity row sums to zero - and (ii) independently with the
# trapezoidal rule on the hourly outputs. The latent part is the melted fraction times
# the latent heat of the plates.

dE = 4 * (P["E"][-1] - P["E"][0])
Q_scheme = 4 * fine["heat_in"]
Q_trapz = 4 * np.trapezoid(P["q_in"], P["t"] * 3600)
print(f"stored heat gained in 72 h: {dE / 1e6:.5f} MJ")
print(f"heat in through the walls:  {Q_scheme / 1e6:.5f} MJ with the scheme weights (relative difference "
      f"{Q_scheme / dE - 1:+.1e}), {Q_trapz / 1e6:.5f} MJ with the trapezoidal rule ({Q_trapz / dE - 1:+.1e})")
print(f"latent heat absorbed: melted fraction {P['melt'][-1]:.4f} x {4 * latent_quarter / 1e6:.4f} MJ = "
      f"{P['melt'][-1] * 4 * latent_quarter / 1e6:.4f} MJ ({P['melt'][-1] * 4 * latent_quarter / dE:.0%} of the heat gained)")
print(f"mean heat leak while the plates melt (0-45 h): {4 * np.mean(P['q_in'][P['t'] <= 45]):.2f} W, "
      f"which melts the {4 * latent_quarter / 1e6:.2f} MJ in {4 * latent_quarter / (4 * np.mean(P['q_in'][P['t'] <= 45])) / 3600:.0f} h")

# **Mesh and time-step study.** Mesh levels $m=1,2,3$ (cell size $h\propto1/m$). The
# observed order follows from $(f_1-f_2)/(f_2-f_3)=(h_1^p-h_2^p)/(h_2^p-h_3^p)$ and the
# Richardson estimate is $f_3+(f_3-f_2)\,h_3^p/(h_2^p-h_3^p)$. The time-step study
# repeats level 2 with $\Delta t=60$, 30 and 15 min; for a second-order integrator the
# differences should drop by a factor of four per halving.

def richardson(f, h):
    ratio = (f[0] - f[1]) / (f[1] - f[2])
    if ratio <= 0:
        return np.nan, np.nan
    lo, hi = 0.05, 8.0
    for _ in range(200):
        p = 0.5 * (lo + hi)
        lo, hi = (lo, p) if (h[0]**p - h[1]**p) / (h[1]**p - h[2]**p) > ratio else (p, hi)
    return p, f[2] + (f[2] - f[1]) * h[2]**p / (h[1]**p - h[2]**p)


QUANTITIES = {
    "contents average at 24 h [degC]": lambda r: at(r["probes"], "cavg", 24.0),
    "contents maximum at 24 h [degC]": lambda r: at(r["probes"], "cmax", 24.0),
    "plates average at 24 h [degC]": lambda r: at(r["probes"], "pavg", 24.0),
    "plates reach 2 degC [h]": lambda r: crossing(r["probes"], "pavg", 2.0),
    "contents maximum reaches 8 degC [h]": lambda r: crossing(r["probes"], "cmax", 8.0),
    "contents average at 72 h [degC]": lambda r: at(r["probes"], "cavg", 72.0),
}
h_rel = np.array([1.0, 1 / 2, 1 / 3])
dt_rel = np.array([60.0, 30.0, 15.0])
print("\nquantity                              m = 1     m = 2     m = 3   change m=2->3  order  Richardson"
      "   dt = 60 / 30 / 15 min (m = 2)        order in time")
study = {}
for name, f in QUANTITIES.items():
    values = np.array([f(runs[m]) for m in (1, 2, 3)])
    p, extrapolated = richardson(values, h_rel)
    asymptotic = 0.5 < p < 6.0                  # otherwise the three meshes are not in the asymptotic range
    study[name] = values
    in_time = np.array([f(time_study[dt_min]) for dt_min in dt_rel])
    p_t, _ = richardson(in_time, dt_rel)
    print(f"{name:36s} " + "".join(f"{v:9.3f} " for v in values) + f"   {values[2] - values[1]:+8.3f}  "
          + (f"   {p:4.2f}  {extrapolated:9.3f}" if asymptotic else "   (not asymptotic)")
          + f"   " + " / ".join(f"{v:.4f}" for v in in_time) + f"   {p_t:5.2f}")


# ## Figure
#
# Top: the temperature after 24 h on the finest mesh - a cut-away view of the whole box
# (one quarter removed along the two symmetry planes), the horizontal section at
# mid-height of the contents ($z=14$ cm) and the vertical mid-section ($y=20$ cm); the
# air pockets are shown with their uniform air temperature. Bottom: the probe
# temperatures against the published curves with the allowed window, the melting of the
# plates and the heat leak, and the mesh study.

level = fine["level"]
xs, zs = level["xs"], level["zs"]
T24 = fine["kept"][24]
grid_q = np.full((len(xs), len(xs), len(zs)), np.nan)
ii, jj, kk = (np.searchsorted(v, level["nodes"][:, c] - 1e-9) for c, v in ((0, xs), (1, xs), (2, zs)))
grid_q[ii, jj, kk] = T24[:-1]
grid_q[np.isnan(grid_q)] = T24[-1]                   # inside the air pocket: the pocket air temperature
grid = np.concatenate((grid_q, grid_q[-2::-1]), axis=0)
grid = np.concatenate((grid, grid[:, -2::-1]), axis=1)
xf = np.concatenate((xs, 0.4 - xs[-2::-1])) * 100     # cm
zf = zs * 100
norm = colors.Normalize(vmin=-1.0, vmax=29.0)
cmap = matplotlib.colormaps["onesimulation"]

fig = plt.figure(figsize=(17, 11.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=(1.05, 1.0))

ax = fig.add_subplot(gs[0, 0], projection="3d")
polys, values = [], []
n2 = len(xs) - 1                                       # index of the mid-plane x = y = 20 cm


def add_face(fixed_axis, index, ranges):
    """Quads of one grid plane over the index ranges of the two free axes."""
    (a0, a1), (b0, b1) = ranges
    for a in range(a0, a1):
        for b in range(b0, b1):
            idx = []
            for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = [0, 0, 0]
                free = [ax_ for ax_ in range(3) if ax_ != fixed_axis]
                p[fixed_axis], p[free[0]], p[free[1]] = index, a + da, b + db
                idx.append(tuple(p))
            polys.append([(xf[p[0]], xf[p[1]], zf[p[2]]) for p in idx])
            values.append(np.mean([grid[p] for p in idx]))


nx, nz = len(xf) - 1, len(zf) - 1
add_face(2, nz, ((0, nx), (0, n2)))                    # top, y < 20 cm
add_face(2, nz, ((0, n2), (n2, nx)))                   # top, x < 20 cm, y > 20 cm
add_face(0, nx, ((0, n2), (0, nz)))                    # side x = 40 cm, y < 20 cm
add_face(1, nx, ((0, n2), (0, nz)))                    # side y = 40 cm, x < 20 cm
add_face(0, n2, ((n2, nx), (0, nz)))                   # cut plane x = 20 cm
add_face(1, n2, ((n2, nx), (0, nz)))                   # cut plane y = 20 cm
ax.add_collection3d(Poly3DCollection(polys, facecolors=cmap(norm(np.array(values))), edgecolors="none"))
ax.set(xlim=(0, 40), ylim=(0, 40), zlim=(0, 28), xlabel="x [cm]", ylabel="y [cm]", zlabel="z [cm]")
ax.set_box_aspect((40, 40, 28))
ax.view_init(elev=26, azim=40)
ax.set_title("Temperature after 24 h (one quarter cut away)")
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.08, label="T [°C]")

levels_T = np.linspace(-1, 29, 31)
ax = fig.add_subplot(gs[0, 1])
k14 = int(np.argmin(abs(zs - 0.14)))
X, Y = np.meshgrid(xf, xf, indexing="ij")
shade = ax.contourf(X, Y, grid[:, :, k14], levels=levels_T, cmap=FIELD_CMAP, extend="both")
ax.contour(X, Y, grid[:, :, k14], levels=[0, 2, 4, 6, 8, 12, 16, 20, 24], colors=INK, linewidths=0.6)
for a, b in ((4, 6), (34, 36)):
    for rect in (((a, 8), b - a, 24), ((8, a), 24, b - a)):
        ax.add_patch(plt.Rectangle(*rect, fill=False, ec=CREAM, lw=0.8, ls="--"))
ax.add_patch(plt.Rectangle((8, 8), 24, 24, fill=False, ec=CREAM, lw=0.8))
ax.set_aspect("equal")
ax.set(xlabel="x [cm]", ylabel="y [cm]", title=f"Section z = {zs[k14]*100:.0f} cm after 24 h (plates dashed)")
fig.colorbar(shade, ax=ax, shrink=0.8, label="T [°C]", ticks=np.arange(0, 30, 4))

ax = fig.add_subplot(gs[0, 2])
Xz, Zz = np.meshgrid(xf, zf, indexing="ij")
section = grid[:, n2, :]
shade = ax.contourf(Xz, Zz, section, levels=levels_T, cmap=FIELD_CMAP, extend="both")
ax.contour(Xz, Zz, section, levels=[0, 2, 4, 6, 8, 12, 16, 20, 24], colors=INK, linewidths=0.6)
for a in (4, 34):
    ax.add_patch(plt.Rectangle((a, 4), 2, 20, fill=False, ec=CREAM, lw=0.8, ls="--"))
ax.add_patch(plt.Rectangle((8, 4), 24, 20, fill=False, ec=CREAM, lw=0.8))
ax.axhline(24, color=CREAM, lw=0.6, ls=":")
ax.set_aspect("equal")
ax.set(xlabel="x [cm]", ylabel="z [cm]", title="Vertical mid-section y = 20 cm after 24 h")
fig.colorbar(shade, ax=ax, shrink=0.8, label="T [°C]", ticks=np.arange(0, 30, 4))

ax = fig.add_subplot(gs[1, 0])
styles = {"contents minimum": (GREEN, "--"), "contents average": (GREEN, "-"), "contents maximum": (GREEN, ":"),
          "plates average": (COPPER, "-")}
for name, key in KEYS.items():
    colour, line = styles[name]
    ax.plot(P["t"], P[key], line, color=colour, lw=1.6, label=f"{name} (3D FEM)")
    ax.plot(ref_t, REF[name], "o" if "plates" in name else "s", color=colour, ms=4.5, mfc="none")
ax.plot([], [], "s", color=INK, mfc="none", label="published probes")
ax.axhspan(2, 8, color=GREEN, alpha=0.08, lw=0)
ax.axhline(2, color="#8B9A8D", lw=0.8, ls="--")
ax.axhline(8, color="#8B9A8D", lw=0.8, ls="--")
ax.set(xlabel="time [h]", ylabel="T [°C]", title="Probe temperatures vs published benchmark", xlim=(0, 72), ylim=(-6, 16))
ax.set_xticks(np.arange(0, 73, 12))
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left")
twin = ax.twinx()
twin.plot(t_amb_h, T_amb_tab, color="#A85539", lw=0.8, alpha=0.6)
twin.set_ylabel("ambient air [°C] (thin line)", color="#A85539")
twin.set_ylim(-20, 32)

ax = fig.add_subplot(gs[1, 1])
ax.plot(P["t"], P["melt"], color=COPPER, lw=1.8, label="melted fraction of the plates")
for m, style in ((1, ":"), (2, "--")):
    ax.plot(runs[m]["probes"]["t"], runs[m]["probes"]["melt"], style, color=COPPER, lw=1.0, label=f"mesh m = {m}")
ax.set(xlabel="time [h]", ylabel="melted fraction", title="Melting of the plates and heat leak", xlim=(0, 72), ylim=(0, 1.05))
ax.set_xticks(np.arange(0, 73, 12))
ax.grid(alpha=0.3)
twin = ax.twinx()
twin.plot(P["t"], 4 * P["q_in"], color=GREEN, lw=1.4, label="heat flow into the box")
twin.set_ylabel("heat flow into the box [W]", color=GREEN)
twin.set_ylim(0, 14)
lines = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
ax.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="lower right")

ax = fig.add_subplot(gs[1, 2])
names = list(QUANTITIES)
ms_ = np.array([1, 2, 3])
ax.plot(ms_, study[names[0]], "o-", color=GREEN, label="contents average at 24 h [°C]")
ax.plot(ms_, study[names[1]], "s-", color="#5F9175", label="contents maximum at 24 h [°C]")
ax.set(xlabel="mesh level m (cell size ~ 1/m)", ylabel="T [°C]", title="Mesh study")
ax.set_xticks(ms_)
ax.grid(alpha=0.3)
twin = ax.twinx()
twin.plot(ms_, study[names[3]], "^-", color=COPPER, label="plates reach 2 °C [h]")
twin.plot(ms_, study[names[4]], "v-", color="#A85539", label="contents maximum reaches 8 °C [h]")
twin.set_ylabel("event time [h]", color=COPPER)
lines = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
ax.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="center right")
plt.show()


# ## Going further
#
# * Add the thin plastic film of the plates and the air gaps around contents and lid as
#   contact resistances on the internal interfaces; they change the heat flows by less
#   than 0.3 %.
# * Compare climates: replace the Seville design day by the hourly data of another
#   station, or add solar gains on the lid; the 24 h criterion is reached much earlier
#   on a hot afternoon start.
# * Design: thicker plates or a phase-change material melting at 5 °C instead of water
#   ice extend the hold time; with the latent-heat budget above, the required plate mass
#   for 48 h follows directly from the mean heat leak.
