"""OneSimulation example: thermal expansion of a Joule-heated copper-beryllium MEMS meander.
Requires the installed, activated OneSimulation runtime.

A 3D serpentine microstructure of the copper-beryllium alloy UNS C17500 is heated volumetrically,
cooled by convection and clamped to a circuit board. The coupled thermal-structural problem is
solved with quadratic finite elements and temperature-dependent material laws, and checked against
the published temperature, the exact heat balance, the exact free-expansion solution and a
three-level mesh study of the maximum displacement.
"""


# # Thermal expansion in a MEMS device
#
# Microgyroscopes and similar MEMS devices must keep their geometry stable while they are
# heated by the drive current. The device studied here is a meander of four hairpins etched
# from a 10 um thick layer of the copper-beryllium alloy UNS C17500: legs 10 um wide and
# 80 um tall, half-ring arches with radii 10 and 20 um, and short bars that join the hairpins
# at the bottom. The bottom face ($y=0$) is bonded to a circuit board. The device spans
# $190\times100\times10$ um, and all fields depend on all three coordinates.
#
# Loads, as in the original problem set-up:
#
# | face | thermal condition | mechanical condition |
# |---|---|---|
# | bonded bottom face $y=0$ | thermally insulated | clamped to the board (three variants below) |
# | front and back faces $z=0$, $z=10$ um | convection, $h=10$ W/(m² K), $T_\infty=298$ K | traction-free |
# | all side walls (legs, arches, bars) | convection, $h=10$ W/(m² K), $T_\infty=298$ K | traction-free |
#
# The Joule heating is a uniform volumetric source $Q=10^8$ W/m³. The thermal strain is
# $\varepsilon_\mathrm{th}=\alpha(T)\,(T-T_\mathrm{ref})\,\mathbf I$ with the secant expansion
# coefficient $\alpha(T)$ and the strain-free temperature $T_\mathrm{ref}=298$ K.
#
# **Board variants.** (1) *Fixed board*: the bonded face cannot move. (2) *Expanding FR-4 board*:
# the bonded face follows the free thermal expansion of a glass-epoxy board at the local device
# temperature, $\mathbf u=\alpha_\mathrm{FR4}(T-T_\mathrm{ref})\,(x,0,z)$. (3) *Matched board*:
# the board expands exactly like the alloy itself.
#
# **Material data.** UNS C17500 (Cu-2.5Co-0.6Be): $k=208$ W/(m K) and $E=138$ GPa at 20 °C,
# mean expansion coefficient $17.6\times10^{-6}$ 1/K over 20-200 °C, $\rho=8830$ kg/m³,
# $c_p=420$ J/(kg K) (Copper Development Association, *Standard Designations for Wrought and
# Cast Copper Alloys*, alloy C17500; ASTM B441), and $\nu=0.30$ (Materion, *Guide to Copper
# Beryllium*). These values are made temperature-dependent with linear laws about 293.15 K.
# The slopes are typical of copper alloys. $E$ drops by $3.5\times10^{-4}$ per K (Ledbetter
# and Naimon, *J. Phys. Chem. Ref. Data* 3, 897, 1974, for copper), and $k$ of the alloy rises
# by $2\times10^{-4}$ per K. The tangent expansion coefficient of copper rises by about
# $5\times10^{-4}$ per K, so the 20-200 °C mean of $17.6\times10^{-6}$ corresponds to
# $16.8\times10^{-6}$ 1/K at 20 °C. The secant coefficient used in the thermal-strain law
# rises with half that slope. The tabulated library functions behind the original model are
# not public, so these laws stand in for them. Over the 26 K temperature rise they change $E$
# by 0.9 % and $\alpha$ by 0.7 %. FR-4 laminate:
# in-plane $\alpha_\mathrm{FR4}=18\times10^{-6}$ 1/K, the upper end of the IPC-4101 range.
#
# **Numerical method.** Spatial: continuous Galerkin finite elements, quadratic (P2) Lagrange
# tetrahedra for both the temperature and the displacement, on a block-structured mesh built
# in numpy. Coupling: one-way staggered. The heat problem is solved first, and its P2 temperature
# field is passed to the elastic solid by finite-element evaluation at the solid's quadrature
# points (`FieldCoupling` + `IsotropicThermalExpansion`). The displacement does not feed back
# into the heat problem, so one pass is the exact coupled solution. The temperature-dependent
# conductivity makes the heat problem nonlinear, and it is solved with Newton's method
# (sparse direct LU inner solves). Each elastic solve is linear (sparse direct LU). No time
# integration is needed: the problem is stationary.
#
# **Reference.** (i) The published result of the original set-up: the device reaches about
# 323.67 K and is isothermal to better than 0.01 K. (ii) The exact global heat balance
# $\int h(T-T_\infty)\,\mathrm dA=QV$, and the lumped-body limit $T=T_\infty+QV/(hA)$, which is
# exact as the Biot number $hL/k\approx5\times10^{-7}$ goes to zero. (iii) The exact 3D solution for the
# matched board, $\mathbf u=\alpha(T)(T-T_\mathrm{ref})\,(x,y,z)$, with zero stress (uniform
# free expansion). (iv) A three-level mesh study (observed order and Richardson estimate) of
# the maximum displacement and of the arch-top displacements for the fixed board. (v) The published qualitative result that an
# expanding board lowers the peak stress by about an order of magnitude.

from dataclasses import dataclass
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
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.field_transfer import FiniteElementEvaluation
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.physics.solid_mechanics import (
    FixedDisplacement, IsotropicThermalExpansion, LinearElasticity, TractionFree,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling

print("OneSimulation:", version("onesimulation"))


# ## Parameters

um = 1e-6
leg_width = 10 * um          # m, width of every leg, arch and bar
leg_height = 80 * um         # m, straight part of a leg
arch_inner, arch_outer = 10 * um, 20 * um
depth = 10 * um              # m, layer thickness (z)
pitch = 50 * um              # m, hairpin spacing
n_hairpins, x_first = 4, -100 * um

Q = 1e8                      # W/m^3, Joule heating
h_conv, T_amb = 10.0, 298.0  # W/(m^2 K), K
T_ref = 298.0                # K, strain-free temperature
alpha_fr4 = 18e-6            # 1/K, in-plane expansion of the FR-4 board
T_published = 323.67         # K, published device temperature of the original set-up


@dataclass(frozen=True)
class LinearInTemperature:
    """Property law p(T) = p0 (1 + s (T - 293.15 K)) with its exact temperature derivative."""

    value_293: float
    slope: float
    unit: str
    dependencies = ("temperature",)

    def evaluate(self, temperature):
        return self.value_293 * (1.0 + self.slope * (np.asarray(temperature) - 293.15))

    def derivative(self, variable, temperature):
        return self.value_293 * self.slope + 0.0 * np.asarray(temperature)


conductivity = LinearInTemperature(208.0, 2.0e-4, "W/m/K")
youngs_modulus = LinearInTemperature(138e9, -3.5e-4, "Pa")
expansion = LinearInTemperature(16.8e-6, 2.5e-4, "1/K")    # secant, from T_ref
poisson = 0.30


def alloy(at_temperature=None):
    """C17500 with temperature-dependent laws, or with the laws evaluated at one temperature."""
    if at_temperature is None:
        k, E, a = conductivity, youngs_modulus, expansion
    else:
        k, E, a = (float(law.evaluate(at_temperature)) for law in (conductivity, youngs_modulus, expansion))
    return IsotropicSolid(density=8830.0, specific_heat_capacity=420.0, thermal_conductivity=k,
                          youngs_modulus=E, poisson_ratio=poisson, thermal_expansion_coefficient=a)


# Exact geometry of the profile: legs, bottom bars and half-ring arches.
area_profile = (2 * n_hairpins * leg_width * leg_height + (n_hairpins - 1) * leg_width**2
                + n_hairpins * 0.5 * np.pi * (arch_outer**2 - arch_inner**2))
perimeter = (2 * n_hairpins * 2 * (leg_width + leg_height) + (n_hairpins - 1) * 4 * leg_width
             + n_hairpins * (np.pi * (arch_outer + arch_inner) + 2 * leg_width)
             - 2 * (2 * n_hairpins + 2 * (n_hairpins - 1)) * leg_width)
bonded_length = (2 * n_hairpins + n_hairpins - 1) * leg_width
volume_exact = area_profile * depth
cooled_exact = 2 * area_profile + (perimeter - bonded_length) * depth
T_lumped = T_amb + Q * volume_exact / (h_conv * cooled_exact)
print(f"profile area {area_profile / um**2:.2f} um^2, volume {volume_exact / um**3:.1f} um^3, "
      f"cooled surface {cooled_exact / um**2:.1f} um^2")
print(f"lumped-body temperature T_inf + QV/(hA) = {T_lumped:.4f} K   (published: {T_published} K)")
print(f"Biot number h*leg_width/k = {h_conv * leg_width / 208.0:.1e}")


# ## Geometry and mesh
#
# OneSimulation's CAD primitives do not include this extruded meander, so the mesh is built
# directly. The profile is covered by structured quadrilateral blocks: each leg is split into
# a bottom block and an upper block, so the bottom bars join it conformally. Each arch is a
# polar block. A refinement level $n$ puts $n$ cells across every 10 um width and $n$ cells
# through the thickness. With quadratic elements this gives several nodes across every leg,
# bar and arch. The quadrilaterals are split into triangles and extruded in $z$, and every
# prism is split into three tetrahedra by a conforming vertex-ordering rule (Dompierre et
# al., Proc. 8th Int. Meshing Roundtable, 1999). The bottom face, the front and back faces
# and the side walls are named boundary groups. The arches are faceted with $6n$ straight
# segments along each half ring (an even number, so a vertex sits at the top of every arch).
# The interior arc vertices are moved outward by a factor $f$ (1.006 for $n=2$), so that every
# faceted ring has exactly the area of the round one. The meshed volume is then exact on
# every level, and the temperature rise no longer depends on the faceting.

@dataclass(frozen=True, eq=False, kw_only=True)
class MeanderDevice(Geometry):
    """Extruded meander with named faces: bonded base, front, back and side walls."""

    dimension = 3

    def boundary_groups(self):
        names = ("base", "front", "back", "side_walls")
        return {**{name: (name,) for name in names}, "all": names, "cooled": names[1:]}


def profile_blocks(n):
    unit = np.linspace(0.0, 1.0, n + 1)
    upper = np.linspace(0.0, 1.0, 7 * n + 1)
    blocks = []
    for i in range(n_hairpins):
        left = x_first + i * pitch
        for x0 in (left, left + 3 * leg_width):
            for y0, y1, v in ((0.0, leg_width, unit), (leg_width, leg_height, upper)):
                X, Y = np.meshgrid(x0 + leg_width * unit, y0 + (y1 - y0) * v, indexing="ij")
                blocks.append(np.stack((X, Y), axis=-1))
        segments = 6 * n
        theta = np.linspace(np.pi, 0.0, segments + 1)
        radius = arch_outer - (arch_outer - arch_inner) * unit
        TH, RR = np.meshgrid(theta, radius, indexing="ij")
        # Interior arc vertices sit slightly outside the circle (factor f), so that the faceted
        # half ring has exactly the area of the round one; the end vertices join the legs.
        s_d = np.sin(np.pi / segments)
        f = (-2 * s_d + np.sqrt(4 * s_d**2 + 4 * (segments - 2) * s_d * np.pi)) / (2 * (segments - 2) * s_d)
        RR[1:-1] *= f
        blocks.append(np.stack((left + 2 * leg_width + RR * np.cos(TH), leg_height + RR * np.sin(TH)), axis=-1))
    for j in range(n_hairpins - 1):
        X, Y = np.meshgrid(x_first + 4 * leg_width + j * pitch + leg_width * unit, leg_width * unit, indexing="ij")
        blocks.append(np.stack((X, Y), axis=-1))
    return blocks


def profile_mesh(n):
    points, index, quads = [], {}, []
    for block in profile_blocks(n):
        ids = np.empty(block.shape[:2], dtype=int)
        for a in range(block.shape[0]):
            for b in range(block.shape[1]):
                key = (round(block[a, b, 0] / 1e-11), round(block[a, b, 1] / 1e-11))
                if key not in index:
                    index[key] = len(points)
                    points.append(block[a, b])
                ids[a, b] = index[key]
        for a in range(block.shape[0] - 1):
            for b in range(block.shape[1] - 1):
                quads.append((ids[a, b], ids[a + 1, b], ids[a + 1, b + 1], ids[a, b + 1]))
    points, triangles = np.array(points), []
    for a, b, c, d in quads:
        if np.linalg.norm(points[a] - points[c]) <= np.linalg.norm(points[b] - points[d]):
            triangles += [(a, b, c), (a, c, d)]
        else:
            triangles += [(a, b, d), (b, c, d)]
    return points, np.array(triangles)


PRISM_ROTATIONS = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                            [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def split_prism(vertices):
    v = np.asarray(vertices)[PRISM_ROTATIONS[int(np.argmin(vertices))]]
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [(v[0], v[1], v[2], v[5]), (v[0], v[1], v[5], v[4]), (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]), (v[0], v[4], v[2], v[5]), (v[0], v[4], v[5], v[3])]


def build_mesh(n):
    section, triangles = profile_mesh(n)
    n_section, layers = len(section), n
    z_levels = np.linspace(0.0, depth, layers + 1)
    coordinates = np.column_stack((np.tile(section[:, 0], layers + 1), np.tile(section[:, 1], layers + 1),
                                   np.repeat(z_levels, n_section)))
    cells = []
    for k in range(layers):
        lower, upper = k * n_section, (k + 1) * n_section
        for a, b, c in triangles:
            cells += split_prism((a + lower, b + lower, c + lower, a + upper, b + upper, c + upper))
    cells = np.array(cells)
    p = coordinates[cells]
    signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    cells[signed < 0] = cells[signed < 0][:, [0, 2, 1, 3]]
    geometry = MeanderDevice()
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    tol = 1e-12
    for face in boundary_faces(cells, coordinates):
        v = np.asarray(face.vertices)
        q = coordinates[v]
        if np.all(np.abs(q[:, 2]) < tol):
            name = "back"
        elif np.all(np.abs(q[:, 2] - depth) < tol):
            name = "front"
        elif np.all(np.abs(q[:, 1]) < tol):
            name = "base"
        else:
            name = "side_walls"
        groups[name].append(tuple(v))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells, boundary_facets=groups)
    volumes = np.abs(signed) / 6.0
    return dict(n=n, geometry=geometry, mesh=mesh, coordinates=coordinates, groups=groups, section=section,
                triangles=triangles, n_section=n_section, layers=layers, volume=float(volumes.sum()))


# ## Model
#
# The heat model carries the uniform Joule source, convection on every cooled face and an
# explicit zero flux on the bonded face. The solid is coupled to the solved temperature
# field. Boundary conditions are listed per board variant in the introduction.

BARY = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def face_quadrature(data, names):
    faces = np.array([f for name in names for f in data["groups"][name]])
    p = data["coordinates"][faces]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    # 3-point rule on each triangle; exact for the quadratic temperature on a flat facet
    return np.einsum("qk,fkd->fqd", BARY, p).reshape(-1, 3), np.repeat(area / 3.0, 3), float(area.sum())


def solve_heat(data, material, nonlinear):
    geometry = data["geometry"]
    b = geometry.boundaries
    heat = HeatConduction(domain=geometry.domain(), material=material, heating=Q)
    problem = Problem(name="Joule-heated meander", geometry=geometry, physics=(heat,), boundary_conditions=(
        ConvectiveHeatFlux(boundary=b.cooled, physics=heat, heat_transfer_coefficient=h_conv, ambient_temperature=T_amb),
        HeatFlux(boundary=b.base, physics=heat, value=0.0)))
    if nonlinear:
        numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)),
                            nonlinear_solver=Newton(linear_solver=SparseDirect(), relative_tolerance=0.0,
                                                    absolute_tolerance=1e-9, maximum_iterations=20),
                            initial_guess={heat.temperature: T_lumped})
    else:
        numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    field = Simulation(problem=problem, mesh=data["mesh"], numerics=numerics).solve_stationary().field(heat.temperature)
    nodal = np.asarray(field.evaluate(data["coordinates"])).reshape(-1)
    points, weights, cooled_area = face_quadrature(data, ("front", "back", "side_walls"))
    heat_out = float(np.sum(weights * h_conv * (np.asarray(field.evaluate(points)).reshape(-1) - T_amb)))
    return dict(field=field, nodal=nodal, T_max=float(nodal.max()), T_min=float(nodal.min()),
                heat_out=heat_out, heat_in=Q * data["volume"], cooled_area=cooled_area)


def board_displacement(temperature_field, coefficient):
    """Bonded-face displacement of a board that expands freely at the local device temperature."""
    def value(x, y, z):
        points = np.column_stack((np.ravel(x), np.ravel(y), np.ravel(z)))
        T = np.asarray(temperature_field.evaluate(points)).reshape(-1)
        strain = np.asarray(coefficient(T)) * (T - T_ref)
        return np.stack((strain * points[:, 0], 0.0 * strain, strain * points[:, 2]), axis=-1).reshape(np.shape(x) + (3,))
    return value


# Outer top point of each arch, mid-thickness (a mesh vertex on every level).
arch_tops = np.array([[x_first + 2 * leg_width + i * pitch, leg_height + arch_outer, depth / 2]
                      for i in range(n_hairpins)])


def solve_solid(data, material, temperature_field, board="fixed"):
    geometry = data["geometry"]
    b = geometry.boundaries
    solid = LinearElasticity(domain=geometry.domain(), material=material,
                             thermal_strain=IsotropicThermalExpansion(reference_temperature=T_ref))
    if board == "fixed":
        support = (0.0, 0.0, 0.0)
    elif board == "FR-4":
        support = board_displacement(temperature_field, lambda T: alpha_fr4 + 0.0 * T)
    else:                                   # matched: the board expands like the alloy itself
        support = board_displacement(temperature_field, expansion.evaluate)
    problem = Problem(name=f"Meander on a {board} board", geometry=geometry, physics=(solid,), boundary_conditions=(
        FixedDisplacement(boundary=b.base, physics=solid, value=support),
        TractionFree(boundary=b.cooled, physics=solid)),
        couplings=(FieldCoupling(source=temperature_field, target=solid.temperature,
                                 transfer=FiniteElementEvaluation()),))
    result = Simulation(problem=problem, mesh=data["mesh"], numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())).solve_stationary()
    u = result.field(solid.displacement)
    nodal = np.asarray(u.evaluate(data["coordinates"]))
    magnitude = np.linalg.norm(nodal, axis=1)
    top = np.asarray(u.evaluate(arch_tops)).reshape(-1, 3)
    return dict(u=u, stress=result.field(solid.von_mises_stress), nodal=nodal, u_max=float(magnitude.max()),
                where_max=data["coordinates"][int(magnitude.argmax())], top=top)


# ## Solve
#
# **Temperature-dependent laws evaluated pointwise.** On the coarsest mesh ($n=1$) the
# complete model is solved with the laws evaluated at every quadrature point. Newton's method
# handles $k(T)$, and $E(T)$ and $\alpha(T)$ are evaluated at the coupled temperature. The
# same mesh is then solved again with the laws evaluated once, at the computed mean device
# temperature. The device is isothermal to within millikelvin, so the two must agree. The
# pointwise path is about ten times more expensive per assembly, so the mesh study below uses
# the laws evaluated at each mesh's own computed mean temperature.

start = time.perf_counter()
coarse = build_mesh(1)
heat_pointwise = solve_heat(coarse, alloy(), nonlinear=True)
solid_pointwise = solve_solid(coarse, alloy(), heat_pointwise["field"])
T_mean_coarse = float(heat_pointwise["nodal"].mean())
heat_frozen = solve_heat(coarse, alloy(T_mean_coarse), nonlinear=False)
solid_frozen = solve_solid(coarse, alloy(T_mean_coarse), heat_frozen["field"])
print(f"n = 1 ({coarse['mesh'].cell_count} tetrahedra): pointwise laws vs laws at T_mean = {T_mean_coarse:.4f} K:")
print(f"  T_max {heat_pointwise['T_max']:.6f} K vs {heat_frozen['T_max']:.6f} K;  "
      f"u_max {solid_pointwise['u_max'] / um * 1e3:.5f} nm vs {solid_frozen['u_max'] / um * 1e3:.5f} nm "
      f"(relative difference {solid_frozen['u_max'] / solid_pointwise['u_max'] - 1:+.1e})  [{time.perf_counter() - start:.0f} s]")

# **Mesh study, fixed board.** Levels $n=2,3,4$ (cells across every 10 um width and through
# the thickness).

levels = (2, 3, 4)
runs = []
for n in levels:
    start = time.perf_counter()
    data = build_mesh(n)
    heat = solve_heat(data, alloy(T_mean_coarse), nonlinear=False)
    T_mean = float(heat["nodal"].mean())
    material = alloy(T_mean)
    fixed = solve_solid(data, material, heat["field"], "fixed")
    runs.append(dict(data=data, heat=heat, T_mean=T_mean, material=material, fixed=fixed))
    print(f"n = {n}: {data['mesh'].cell_count:6d} tetrahedra, {len(data['coordinates']):6d} vertices, "
          f"T in [{heat['T_min']:.5f}, {heat['T_max']:.5f}] K, u_max = {fixed['u_max'] / um * 1e3:.4f} nm  "
          f"[{time.perf_counter() - start:.0f} s]")
final = runs[-1]

# **Board variants** on the medium mesh: the expanding FR-4 board and the matched board.

start = time.perf_counter()
medium = runs[1]
fr4 = solve_solid(medium["data"], medium["material"], medium["heat"]["field"], "FR-4")
matched = solve_solid(medium["data"], medium["material"], medium["heat"]["field"], "matched")
print(f"board variants on n = {medium['data']['n']}: [{time.perf_counter() - start:.0f} s]")


# ## Comparison with the reference
#
# **Thermal part.** The published device temperature, the lumped-body limit for the exact
# geometry, and the global heat balance of the computed field. The faceted arches make the
# meshed volume and surface slightly smaller than the exact ones, so the lumped limit is also
# printed for the meshed geometry, which is the value the FEM must reproduce on each mesh.

print("\nTemperature and heat balance (fixed-board meshes)")
print(" level   T_min [K]     T_max [K]     spread [mK]   lumped, meshed geometry [K]   heat in [uW]   imbalance")
for run in runs:
    heat, data = run["heat"], run["data"]
    lumped_mesh = T_amb + heat["heat_in"] / (h_conv * heat["cooled_area"])
    print(f"   {data['n']}    {heat['T_min']:.5f}   {heat['T_max']:.5f}   {1e3 * (heat['T_max'] - heat['T_min']):9.3f}"
          f"     {lumped_mesh:.5f}                    {1e6 * heat['heat_in']:.5f}      "
          f"{abs(heat['heat_out'] - heat['heat_in']) / heat['heat_in']:.1e}")
T_final = final["heat"]["T_max"]
print(f"T_max, finest mesh:         {T_final:.4f} K")
print(f"published device temperature: {T_published:.2f} K   deviation {T_final - T_published:+.4f} K "
      f"({100 * (T_final - T_amb) / (T_published - T_amb) - 100:+.3f} % of the temperature rise)")
print(f"lumped limit, exact geometry: {T_lumped:.4f} K   deviation {T_final - T_lumped:+.4f} K "
      f"({100 * (T_final - T_amb) / (T_lumped - T_amb) - 100:+.3f} % of the temperature rise)")

# **Matched board: exact free expansion.** With the board expanding like the alloy, the
# exact solution is $\mathbf u=\alpha(T)(T-T_\mathrm{ref})\,(x,y,z)$ with zero stress. The
# residual temperature spread of a few millikelvin is the only source of a deviation.

coords = medium["data"]["coordinates"]
T_nodes = medium["heat"]["nodal"]
strain_free = expansion.evaluate(T_nodes) * (T_nodes - T_ref)
u_exact = strain_free[:, None] * coords
scale_stress = float(youngs_modulus.evaluate(medium["T_mean"]) * expansion.evaluate(medium["T_mean"])
                     * (medium["T_mean"] - T_ref))
vm_matched = np.asarray(matched["stress"].evaluate(coords)).reshape(-1)
print("\nMatched board (exact uniform free expansion), medium mesh")
print(f"  max |u - u_exact| / max |u_exact|  = {np.abs(matched['nodal'] - u_exact).max() / np.abs(u_exact).max():.2e}")
print(f"  max von Mises / (E alpha dT)       = {vm_matched.max() / scale_stress:.2e}   (exact: 0)")
print(f"  top of the arches: u_y = {matched['nodal'][:, 1].max() / um * 1e3:.4f} nm, exact "
      f"{(strain_free * coords[:, 1]).max() / um * 1e3:.4f} nm")

# **Mesh study, fixed board.** Mesh size $\propto 1/n$. The edges where the clamped face
# meets the free side walls are singular: the elastic field of a clamped edge varies like
# $\rho^\lambda$ with $\lambda<1$ in the distance $\rho$ from the edge (Williams, *J. Appl.
# Mech.* 19, 526, 1952). This singularity limits the global convergence order of the
# displacement to about one, whatever the element order. The vertical expansion at the arch
# tops converges monotonically. The maximum displacement also contains the small sideways sway
# of the outer hairpins, which comes from the rotation at the clamped feet. Its successive
# changes are a few hundredths of a percent and not yet monotone, so no order is fitted for it.

def richardson(values, levels):
    """Observed order p and extrapolated value from three meshes with h ~ 1/level."""
    f1, f2, f3 = values
    h1, h2, h3 = 1.0 / np.asarray(levels, float)
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.3, 12.0)
    except ValueError:          # non-monotone sequence: no order can be fitted
        return np.nan, np.nan
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


study = {
    "max |u| over the device": [run["fixed"]["u_max"] for run in runs],
    "u_y at the top of arch 1": [run["fixed"]["top"][0, 1] for run in runs],
    "u_y at the top of arch 2": [run["fixed"]["top"][1, 1] for run in runs],
}
print("\nMesh study, fixed board (nm); h ~ 1/n")
print(f"  {'quantity':26s}" + "".join(f"{'n = ' + str(n):>12s}" for n in levels)
      + "   change last two   order   Richardson   finest vs Richardson")
for name, values in study.items():
    p, extrapolated = richardson(values, levels)
    study[name] = (values, p, extrapolated)
    tail = (f"   {p:5.2f}   {extrapolated / um * 1e3:10.5f}   {100 * (values[2] / extrapolated - 1):+.3f} %"
            if np.isfinite(p) else "     -   (successive changes not monotone: no order can be fitted)")
    print(f"  {name:26s}" + "".join(f"{v / um * 1e3:12.5f}" for v in values)
          + f"   {100 * abs(values[2] / values[1] - 1):12.3f} %" + tail)
print(f"  maximum displacement on the finest mesh at (x, y, z) = "
      f"({', '.join(f'{c / um:.1f}' for c in final['fixed']['where_max'])}) um")
uniform_rise = float(expansion.evaluate(final["T_mean"]) * (final["T_mean"] - T_ref)) * (leg_height + arch_outer)
print(f"  for scale: free vertical expansion of the 100 um tall device alpha dT H = {uniform_rise / um * 1e3:.4f} nm")

# **Expanding board.** The published result: including the board expansion lowers the
# maximum von Mises stress by about an order of magnitude. The peak sits at the re-entrant
# edges of the bonded face, where the elastic field of a clamped corner is singular, so the
# peak value depends on the mesh. The ratio is therefore also given at a regular point,
# half-way up the first leg on its centre line.

vm_fixed = np.asarray(medium["fixed"]["stress"].evaluate(coords)).reshape(-1)
vm_fr4 = np.asarray(fr4["stress"].evaluate(coords)).reshape(-1)
probe = np.array([[x_first + leg_width / 2, 20 * um, depth / 2]])
vm_probe = [float(np.asarray(s["stress"].evaluate(probe)).reshape(-1)[0]) for s in (medium["fixed"], fr4)]
print("\nFixed vs expanding FR-4 board, medium mesh")
print(f"  max von Mises: fixed {vm_fixed.max() / 1e6:.2f} MPa, FR-4 {vm_fr4.max() / 1e6:.3f} MPa, "
      f"ratio {vm_fixed.max() / vm_fr4.max():.1f}  (published: about one order of magnitude)")
print(f"  von Mises at y = 20 um on the first leg: fixed {vm_probe[0] / 1e6:.3f} MPa, FR-4 {vm_probe[1] / 1e6:.4f} MPa, "
      f"ratio {vm_probe[0] / vm_probe[1]:.1f}")
print(f"  u_max: fixed {medium['fixed']['u_max'] / um * 1e3:.4f} nm, FR-4 {fr4['u_max'] / um * 1e3:.4f} nm, "
      f"matched {matched['u_max'] / um * 1e3:.4f} nm")


# ## Figure
#
# Top left: surface temperature of the device (deviation from the mean, in mK). Top right:
# deformed device on the fixed board, displacement magnified 200 times, coloured by the
# displacement magnitude. Bottom left and middle: von Mises stress on the front face for the
# fixed and the expanding FR-4 board (same logarithmic colour scale). Right: mesh study of the
# maximum displacement and of the vertical displacement at the arch tops, plotted against
# the mesh size, with the Richardson estimates at zero mesh size.

data, heat, fixed = final["data"], final["heat"], final["fixed"]
coords_f, groups = data["coordinates"], data["groups"]
surface = np.array(groups["front"] + groups["back"] + groups["side_walls"])
fig = plt.figure(figsize=(17, 10.5), constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=(1.15, 1.0))


def surface_plot(ax, points, values, norm, title, label):
    ax.add_collection3d(Poly3DCollection(points[surface][:, :, [0, 2, 1]] / um,
                                         facecolors=FIELD_CMAP(norm(values[surface].mean(axis=1))),
                                         edgecolors="none", linewidths=0))
    ax.set(xlim=(-105, 95), ylim=(-25, 35), zlim=(0, 105), xlabel="x / um", ylabel="z / um", zlabel="y / um")
    ax.set_box_aspect((200, 60, 105))
    ax.view_init(elev=20, azim=-58)
    ax.set_title(title)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax, shrink=0.55, label=label, pad=0.16)


ax = fig.add_subplot(grid[0, 0], projection="3d")
dT_mK = 1e3 * (heat["nodal"] - heat["nodal"].mean())
surface_plot(ax, coords_f, dT_mK, colors.Normalize(dT_mK.min(), dT_mK.max()),
             f"Temperature, mean {heat['nodal'].mean():.3f} K", "T - T_mean / mK")

ax = fig.add_subplot(grid[0, 1], projection="3d")
u_mag = np.linalg.norm(fixed["nodal"], axis=1) / um * 1e3
deformed = coords_f + 200.0 * fixed["nodal"]
surface_plot(ax, deformed, u_mag, colors.Normalize(0.0, u_mag.max()),
             "Displacement on the fixed board (magnified 200x)", "|u| / nm")
outline = coords_f[np.array(groups["front"])][:, :, [0, 2, 1]] / um
ax.add_collection3d(Poly3DCollection(outline[::7], facecolors="none", edgecolors="#9FA593", linewidths=0.15))

section, triangles, ns = medium["data"]["section"], medium["data"]["triangles"], medium["data"]["n_section"]
front_ids = medium["data"]["layers"] * ns + np.arange(ns)
tri = Triangulation(section[:, 0] / um, section[:, 1] / um, triangles)
vm_levels = np.logspace(np.log10(max(vm_fr4[front_ids].min(), 1e3)), np.log10(vm_fixed[front_ids].max()), 40)
for k, (values, title) in enumerate(((vm_fixed, "von Mises stress, fixed board"),
                                     (vm_fr4, "von Mises stress, expanding FR-4 board"))):
    ax = fig.add_subplot(grid[1, k])
    filled = ax.tricontourf(tri, np.clip(values[front_ids], vm_levels[0], vm_levels[-1]), levels=vm_levels,
                            cmap=FIELD_CMAP, norm=colors.LogNorm(vm_levels[0], vm_levels[-1]))
    ax.set_aspect("equal")
    ax.set(xlabel="x / um", ylabel="y / um", title=title + " (front face)")
    fig.colorbar(filled, ax=ax, shrink=0.8, label="von Mises / Pa", ticks=[1e4, 1e5, 1e6, 1e7, 1e8])

ax = fig.add_subplot(grid[:, 2])
inverse = 1.0 / np.array(levels)
for (name, (values, p, extrapolated)), colour, marker in zip(study.items(), (COPPER, GREEN, "#5F9175"), "osD"):
    values = np.array(values) / um * 1e3
    ax.plot(inverse, values, marker + "-", color=colour, label=name)
    if np.isfinite(extrapolated):
        ax.plot([0.0, inverse[-1]], [extrapolated / um * 1e3, values[-1]], ":", color=colour)
        ax.plot(0.0, extrapolated / um * 1e3, marker, color=colour, mfc="none",
                label=f"Richardson estimate (order {p:.2f})")
ax.set_xlim(-0.02, 0.55)
ax.set(xlabel="mesh size 1/n (n = cells across each 10 um width)", ylabel="displacement / nm",
       title="Mesh study, fixed board")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Replace the uniform heat source by the Joule heat of an actual drive current (an electric
# conduction solve on the same mesh), add the board as a second, meshed body, or sweep the
# board's expansion coefficient to find the value that makes the bonded joint stress free.
