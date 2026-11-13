"""OneSimulation example: aquifer-system compaction above a bedrock step.
Requires the installed, activated OneSimulation runtime.

Pumping lowers the hydraulic head in the lower aquifer of a sedimentary basin whose
bedrock is faulted into a step. The 2D vertical section is computed twice, with the
Terzaghi method (transient groundwater flow plus vertical compaction) and with fully
coupled Biot poroelasticity, and graded against the published results of Leake and
Hsieh (1997), Terzaghi's one-dimensional consolidation of the confining unit at the
basin centre, a fluid-volume balance and a mesh and time-step study.
"""


# # Aquifer-system compaction above a bedrock step
#
# Fluid in the pores of an aquifer carries part of the weight of the overlying
# sediments. When pumping lowers the pore pressure, the solid skeleton has to take over
# that load and the sediments compact; at the land surface this appears as subsidence.
# Near the margin of a basin, where the sediment thickness changes abruptly over a
# buried bedrock step, the compaction is uneven and the ground is stretched
# horizontally, which is the classical mechanism for earth fissures.
#
# The set-up is the idealized basin section of S.A. Leake and P.A. Hsieh, *Simulation
# of deformation of sediments from decline of ground-water levels in an aquifer
# underlain by a bedrock step*, in: U.S. Geological Survey Open-File Report 97-47,
# pp. 10-13, 1997. The section is 5200 m wide and consists of three layers:
#
# | layer | depth | hydraulic conductivity $K$ | skeletal specific storage $S_{sk}$ | Young's modulus $E$ | Poisson's ratio |
# |---|---|---|---|---|---|
# | upper aquifer | 0-20 m | 25 m/d | $10^{-5}$ 1/m | $8\cdot10^{8}$ Pa | 0.25 |
# | confining unit | 20-40 m | 0.01 m/d | $10^{-4}$ 1/m | $8\cdot10^{7}$ Pa | 0.25 |
# | lower aquifer | 40-440 m | 25 m/d | $10^{-5}$ 1/m | $8\cdot10^{8}$ Pa | 0.25 |
#
# The bedrock under the right part of the basin ($x>4000$ m) rises to 120 m depth, so
# the lower aquifer thins from 400 m to 80 m. At the basin centre ($x=0$) the head in
# the lower aquifer is lowered by 6 m per year for 10 years; the upper aquifer is held
# at constant head. All quantities are changes from the initial equilibrium state.
#
# Boundary segments (Leake and Hsieh, Table 1):
#
# | segment | where | flow | mechanics (Biot) |
# |---|---|---|---|
# | A | base of the sediments including the step | no flow | fixed |
# | B | lateral edges of the confining unit and of the lower aquifer at $x=5200$ m | no flow | roller ($u_x=0$) |
# | C | lateral edges of the upper aquifer | head held constant | roller |
# | D | land surface | head held constant | traction-free |
# | E | lower aquifer at $x=0$ (pumping) | head $-(6\ \mathrm{m/yr})\,t$ | roller |
#
# **Two models.** The *Terzaghi method* solves only transient groundwater flow,
#
# $$S_{sk}\,\partial_t h-\nabla\cdot(K\nabla h)=0 ,$$
#
# and obtains the vertical displacement by stacking the one-dimensional compaction
# $\Delta b=-S_{sk}\,b\,\Delta h$ of every layer element from the base upward (Terzaghi,
# *Theoretical Soil Mechanics*, Wiley, 1943). The *Biot method* (M.A. Biot, J. Appl.
# Phys. 12, 155, 1941) couples Darcy flow and plane-strain elasticity:
#
# $$\nabla\cdot\big(\mathbf C:\boldsymbol\varepsilon(\mathbf u)-\alpha\,p\,\mathbf I\big)=0,\qquad
#   \partial_t\big(S\,p+\alpha\,\nabla\cdot\mathbf u\big)-\nabla\cdot\Big(\frac{\kappa}{\mu}\nabla p\Big)=0 ,$$
#
# with Biot coefficient $\alpha=1$. Its skeleton compressibility follows from $E,\nu$: in
# uniaxial strain $S_{sk}=\rho g/M$ with the oedometric modulus
# $M=E(1-\nu)/((1+\nu)(1-2\nu))$, which gives $1.02\cdot10^{-5}$ and $1.02\cdot10^{-4}$ 1/m,
# the values of the table. Leake and Hsieh neglect the compressibility of water and
# grains; here the storage of the pore constituents is kept at one tenth of the
# skeletal storage, $S\rho g=10^{-6}$ 1/m in the aquifers (porosity 0.25 times the water
# compressibility $4\cdot10^{-10}$ 1/Pa gives the same value) and $10^{-5}$ 1/m in the
# confining unit. It changes only the first days of the response, and it keeps the
# equal-order displacement-pressure pair well conditioned.
#
# **Method.** Both models use continuous Galerkin finite elements with quadratic
# Lagrange triangles (P2) on the same graded block mesh. The Biot model is a mixed
# displacement-pressure formulation with the equal-order pair P2 displacement / P2
# pressure, carrying the fluid content as the differential unknown. Time integration
# is the variable-step second-order backward differentiation formula (BDF2) with
# local error control: the first step is 0.0007 days to follow the fast initial
# pressure diffusion and grows (at most by a factor 2 per step) to one year once the
# response becomes quasi-steady; the first step is chosen so that the steps then fall
# on whole years. Both models use this integrator. Every stage is solved by Newton's
# method (one iteration for this linear problem) with a sparse direct LU solver.
#
# **Reference.** (i) The published results of Leake and Hsieh (1997) after 10 years,
# for the upper cell (80 m wide) of the columns on either side of the step, which moves
# with the land surface: Biot displacement 0.10 m on the thick side and below 0.08 m on
# the thin side, Terzaghi displacement above 0.18 m on the thick side and below 0.08 m
# on the thin side, and a horizontal tensile strain of more than $10^{-4}$ at the land
# surface above the step. They are quoted in the text of the report as rounded values
# and bounds, computed on a coarse grid of 80 m wide and 4-20 m high cells.
# (ii) At the basin centre, where the section is most nearly laterally uniform, the
# confining unit should follow Terzaghi's one-dimensional consolidation with a ramped
# boundary head, and the surface settlement should approach the uniaxial compaction of
# the column. (iii) The
# fluid-volume balance of the whole section. (iv) A three-level mesh study with
# observed order and Richardson estimate, and a time-step study.

from importlib.metadata import version
from dataclasses import dataclass
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
from matplotlib.tri import Triangulation
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from onesim.cad.geometry.primitives import Geometry
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.triangular import triangular_geometry
from onesim.numerics.fem import FiniteElement, Lagrange, Nitsche
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, AdaptiveTimeStep
from onesim.physics.geotechnics import (BiotConsolidation, BiotFixedComponent, BiotNormalTraction,
                                        GroundwaterFlow, PoreFluidFlux, PrescribedPorePressure)
from onesim.physics.solid_mechanics import FixedDisplacement
from onesim.results.storage.archives.models.output import Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation.problem_definition.boundaries.conservative import (OutwardFluxBoundary,
                                                                          PrescribedValueBoundary)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# **Units.** Time is measured in days and pore pressure as hydraulic head in metres of
# water ($\hat p=p/(\rho g)$); stiffnesses are divided by $\rho g$ accordingly. This
# is a consistent rescaling of the same equations (every coefficient is converted, so
# the physics is unchanged) and keeps the coupled displacement-pressure matrix well
# scaled: in SI units its entries span more than 16 orders of magnitude.

rho_g = 1000.0 * 9.82          # Pa per metre of head (Leake and Hsieh use g = 9.82 m/s^2)
day = 1.0                      # time unit
year = 365.25 * day
width, depth = 5200.0, 440.0   # m, section size
x_step, y_step = 4000.0, -120.0  # m, edge and top of the bedrock step
y_conf_top, y_conf_bottom = -20.0, -40.0
decline_rate = 6.0 / year      # m of head per day at segment E
end_time = 10 * year

nu = 0.25
E_aquifer, E_confining = 8e8 / rho_g, 8e7 / rho_g          # m of head
K_aquifer, K_confining = 25.0, 0.01                         # m/d
Ssk_aquifer, Ssk_confining = 1e-5, 1e-4                     # 1/m (Terzaghi method)
S_aquifer, S_confining = 1e-6, 1e-5                         # 1/m, storage of the pore constituents

oedometric = lambda E: E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
Ssk_biot = {name: 1.0 / oedometric(E) for name, E in (("aquifer", E_aquifer), ("confining", E_confining))}
print("skeletal specific storage implied by E and nu: aquifers %.3e 1/m, confining unit %.3e 1/m"
      % (Ssk_biot["aquifer"], Ssk_biot["confining"]))


def confining(y):
    y = np.asarray(y, dtype=float)
    return (y < y_conf_top) & (y > y_conf_bottom)


def layered(aquifer, confining_value):
    return lambda y: np.where(confining(y), confining_value, aquifer)


def plane_strain_stiffness(y):
    """Isotropic 6x6 elasticity tensor (Voigt order) of the layer at height y."""
    E = layered(E_aquifer, E_confining)(np.atleast_1d(y))
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    c = np.zeros((len(E), 6, 6))
    c[:, :3, :3] = lam[:, None, None]
    c[:, [0, 1, 2], [0, 1, 2]] += 2 * mu[:, None]
    c[:, [3, 4, 5], [3, 4, 5]] = mu[:, None]
    return c


def head_at_outlet(time, coordinate):
    return [-decline_rate * time]


# ## Geometry and mesh
#
# The section is covered by a tensor grid of node lines that contains every layer
# interface, the step top $y=-120$ m and the step edge $x=4000$ m; cells inside the
# bedrock step are removed and each quadrilateral is split into two triangles. The grid
# is graded towards the re-entrant corner of the step, where the elastic field is
# singular, and towards the layer interfaces; the 20 m confining unit has $4\ell$ cells
# across. Level $\ell$ scales the number of cells in every direction.


def graded(a, b, n, ratio):
    """n cells from a to b, the last cell `ratio` times the first."""
    sizes = ratio ** (np.arange(n) / max(n - 1, 1))
    t = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    t[-1] = 1.0
    return a + (b - a) * t


@dataclass(frozen=True, eq=False, kw_only=True)
class BasinSection(Geometry):
    """Vertical basin section with the boundary segments of Leake and Hsieh."""

    dimension = 2

    def boundary_groups(self):
        names = ("base", "outlet", "left_confining", "left_upper", "right_lower", "right_confining",
                 "right_upper", "surface")
        return {**{name: (name,) for name in names}, "all": names}


def build_mesh(level):
    xs = np.unique(np.r_[graded(0.0, x_step, 12 * level, 1 / 20), graded(x_step, width, 6 * level, 12)])
    ys = np.unique(np.r_[graded(-depth, y_step, 5 * level, 1 / 6), graded(y_step, y_conf_bottom, 3 * level, 1 / 3),
                         np.linspace(y_conf_bottom, y_conf_top, 2 * level + 1),
                         np.linspace(y_conf_top, 0.0, level + 1)])
    nx, ny = len(xs), len(ys)
    ids = np.arange(nx * ny).reshape(nx, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.column_stack((X.ravel(), Y.ravel()))
    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if xs[i] >= x_step and ys[j + 1] <= y_step:
                continue                                   # inside the bedrock step
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            triangles += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    triangles = np.array(triangles)
    used = np.unique(triangles)
    renumber = np.full(len(coords), -1)
    renumber[used] = np.arange(len(used))
    coords, triangles = coords[used], renumber[triangles]

    geometry = BasinSection()
    data = triangular_geometry(coords, triangles)
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    for edge in data.face_vertices[data.face_neighbour == -1]:
        p = coords[list(edge)]
        y_mid = p[:, 1].mean()
        if np.allclose(p[:, 0], 0.0):
            name = "outlet" if y_mid < y_conf_bottom else "left_confining" if y_mid < y_conf_top else "left_upper"
        elif np.allclose(p[:, 0], width):
            name = ("right_lower" if y_mid < y_conf_bottom else "right_confining" if y_mid < y_conf_top
                    else "right_upper")
        elif np.allclose(p[:, 1], 0.0):
            name = "surface"
        else:
            name = "base"
        groups[name].append(tuple(edge))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coords, cells=triangles, boundary_facets=groups)
    return dict(level=level, geometry=geometry, mesh=mesh, coords=coords, triangles=triangles, xs=xs, ys=ys)


# ## Model
#
# Both models get an explicit condition on every boundary segment (table above). The
# Biot pressure conditions on C, D and E are imposed weakly by Nitsche's method; the
# displacement supports are essential.

numerics_template = dict(
    spatial=FiniteElement(element=Lagrange(2), boundary_enforcement=Nitsche(penalty=16)),
    linear_solver=SparseDirect(),
    time_integrator=BDF(order=2),
)


def time_control(maximum):
    """First step chosen so that 19 doublings reach the maximum step exactly at t = maximum."""
    return AdaptiveTimeStep(initial=maximum / (2**19 - 1), minimum=1e-6 * day, maximum=maximum,
                            relative_tolerance=1e-3, absolute_tolerance=1e-3)


def output(fields):
    return Output(fields=fields, interval=1.0 * year)


def biot_run(data, maximum=1.0 * year):
    g = data["geometry"]
    b = g.boundaries
    model = BiotConsolidation(
        domain=g.domain(), kinematics="plane_strain", elastic_stiffness=plane_strain_stiffness,
        biot_coefficient=1.0, fluid_viscosity=1.0,
        specific_storage=layered(S_aquifer, S_confining),
        permeability=layered(K_aquifer, K_confining))   # mobility kappa/mu in m^2/(day m of head) = K
    rollers = b.outlet | b.left_confining | b.left_upper | b.right_lower | b.right_confining | b.right_upper
    problem = Problem(
        name="Biot method, bedrock step", geometry=g, physics=(model,),
        boundary_conditions=(
            FixedDisplacement(physics=model, boundary=b.base, value=(0.0, 0.0)),        # A
            BiotFixedComponent(physics=model, boundary=rollers, component="x"),         # B, C, E
            BiotNormalTraction(physics=model, boundary=b.surface, value=0.0),           # D
            PrescribedPorePressure(physics=model, boundary=b.outlet, value=head_at_outlet),             # E
            PrescribedPorePressure(physics=model, boundary=b.left_upper | b.right_upper | b.surface,
                                   value=0.0),                                                          # C, D
            PoreFluidFlux(physics=model, boundary=b.base | b.left_confining | b.right_lower | b.right_confining,
                          flux=0.0),                                                                    # A, B
        ),
        initial_conditions=(InitialValue(model.fluid_content, 0.0),))
    numerics = Numerics(**numerics_template, nonlinear_solver=Newton(
        linear_solver=SparseDirect(), jacobian="analytic", relative_tolerance=1e-3, absolute_tolerance=1e-12))
    start = time.time()
    result = Simulation(problem=problem, mesh=data["mesh"], numerics=numerics).run(
        start_time=0.0, end_time=end_time, time_step=time_control(maximum),
        output=output((model.pore_pressure, model.displacement, model.fluid_content, model.darcy_flux)))
    return dict(model=model, result=result, seconds=time.time() - start)


def terzaghi_run(data, maximum=1.0 * year):
    g = data["geometry"]
    b = g.boundaries
    model = GroundwaterFlow(domain=g.domain(), conductivity=layered(K_aquifer, K_confining),
                            specific_storage=layered(Ssk_aquifer, Ssk_confining))
    problem = Problem(
        name="Terzaghi method, bedrock step", geometry=g, physics=(model,),
        boundary_conditions=(
            PrescribedValueBoundary(physics=model, boundary=b.outlet, value=head_at_outlet),               # E
            PrescribedValueBoundary(physics=model, boundary=b.left_upper | b.right_upper | b.surface,
                                    value=lambda time, coordinate: [0.0]),                                 # C, D
            OutwardFluxBoundary(physics=model, boundary=b.base | b.left_confining | b.right_lower | b.right_confining,
                                flux=lambda interior, time, normal, coordinate: [0.0], jacobian=((0,),)),  # A, B
        ),
        initial_conditions=(InitialValue(model.head, 0.0),))
    numerics = Numerics(**numerics_template, nonlinear_solver=Newton(
        linear_solver=SparseDirect(), jacobian="analytic", relative_tolerance=1e-6, absolute_tolerance=1e-12))
    result = Simulation(problem=problem, mesh=data["mesh"], numerics=numerics).run(
        start_time=0.0, end_time=end_time, time_step=time_control(maximum), output=output((model.head,)))
    return dict(model=model, result=result)


# ## Solve
#
# The Biot model is solved on three meshes (levels 2, 3, 4); the Terzaghi method on
# the finest one. A second Biot run on the coarsest mesh with every step halved
# (maximum step half a year) measures the time-discretization error.

levels = (2, 3, 4)
runs = {}
for level in levels:
    data = build_mesh(level)
    run = biot_run(data)
    runs[level] = (data, run)
    steps = len(run["result"].times) - 1
    print(f"level {level}: {len(data['triangles']):5d} triangles, {3 * len(run['result'].field(run['model'].pore_pressure).values):6d} "
          f"unknowns (u_x, u_y, p), {steps} accepted BDF2 steps, {run['seconds']:.0f} s")
fine_steps = biot_run(runs[levels[0]][0], maximum=year / 2)
print(f"level {levels[0]} with all steps halved (maximum step half a year): "
      f"{len(fine_steps['result'].times) - 1} accepted steps, {fine_steps['seconds']:.0f} s")
data, biot = runs[levels[-1]]
start = time.time()
terzaghi = terzaghi_run(data)
print(f"Terzaghi method on level {levels[-1]}: {len(terzaghi['result'].times) - 1} steps, {time.time() - start:.0f} s")


# ## Comparison with the reference
#
# Field values are read with `field.evaluate` at physical points. Vertical
# displacements are negative downward.

def at(run, ref, points, t):
    return np.asarray(run["result"].field(ref).evaluate(np.atleast_2d(points), time=t))


def nearest_time(run, target):
    times = np.asarray(run["result"].times, dtype=float)
    return float(times[np.argmin(np.abs(times - target))])


def surface_strain(run, t, x):
    """Horizontal strain du_x/dx just below the land surface (from the P2 displacement)."""
    ux = at(run, run["model"].displacement, np.column_stack((x, np.full_like(x, -0.01))), t)[:, 0]
    return np.gradient(ux, x)


def terzaghi_columns(x_values, t, samples=441):
    """Terzaghi method: u_y(x, y) = integral of S_sk h from the base up to y, on vertical lines."""
    columns = []
    for x in np.atleast_1d(x_values):
        ys = np.linspace(y_step if x > x_step else -depth, 0.0, samples)
        h = at(terzaghi, terzaghi["model"].head, np.column_stack((np.full_like(ys, x), ys)), t)
        columns.append((ys, cumulative_trapezoid(layered(Ssk_aquifer, Ssk_confining)(ys) * h, ys, initial=0.0)))
    return columns


def terzaghi_displacement(points, t):
    return np.array([np.interp(y, *terzaghi_columns(x, t, samples=1321)[0]) for x, y in np.atleast_2d(points)])


T = end_time
model = biot["model"]
thick, thin = np.array([[x_step - 40.0, -10.0]]), np.array([[x_step + 40.0, -10.0]])
uy_thick = at(biot, model.displacement, thick, T)[0, 1]
uy_thin = at(biot, model.displacement, thin, T)[0, 1]
uyT_thick, uyT_thin = terzaghi_displacement(np.vstack((thick, thin)), T)
x_up = np.linspace(x_step, width, 601)
strain_up = surface_strain(biot, T, x_up)
print("(i) Leake and Hsieh (1997), 10 years, upper cell of the columns next to the step (y = -10 m)")
print(f"    Biot,     thick side x = 3960 m: settlement {-uy_thick:.4f} m   (published 0.10 m, "
      f"deviation {100 * (-uy_thick / 0.10 - 1):+.1f} %)")
print(f"    Biot,     thin side  x = 4040 m: settlement {-uy_thin:.4f} m   (published: below 0.08 m)")
print(f"    Terzaghi, thick side x = 3960 m: settlement {-uyT_thick:.4f} m   (published: above 0.18 m)")
print(f"    Terzaghi, thin side  x = 4040 m: settlement {-uyT_thin:.4f} m   (published: below 0.08 m)")
print(f"    Biot, maximum horizontal surface strain above the step: {strain_up.max():.3e} at x = "
      f"{x_up[np.argmax(strain_up)]:.0f} m   (published: tension above 1e-4)")

# (ii-a) Terzaghi consolidation of the confining unit at the basin centre. In uniaxial
# strain the confining unit obeys $\partial_t h=c_v\,\partial_\zeta^2 h$ with
# $c_v=K/(S\rho g+\rho g/M)$, $h=0$ at its top and $h=-\beta t$ at its base. With $\zeta$
# measured down from the top and thickness $b$ (Carslaw and Jaeger, *Conduction of Heat
# in Solids*, 2nd ed., Sec. 3.3),
# $$\frac{h}{-\beta}=\frac{t\zeta}{b}-\frac{\zeta(b^2-\zeta^2)}{6c_vb}
#   -\frac{2b^2}{\pi^3c_v}\sum_{n\ge1}\frac{(-1)^n}{n^3}e^{-c_vn^2\pi^2t/b^2}\sin\frac{n\pi\zeta}{b}.$$
# During the first days the drawdown of the lower aquifer is still confined to a zone
# of width $\sqrt{Dt}\approx1.5$ km around the outlet, so the column at $x=0$ is not yet
# fully laterally uniform: the horizontal contraction of the sediments adds a few
# per cent to the storage of the confining unit. Agreement to about 2 % of the boundary
# head decline $\beta t$ is expected, improving as the drawdown spreads.
S_total_conf = S_confining + Ssk_biot["confining"]
c_v = K_confining / S_total_conf                   # m^2/day
b_conf = y_conf_top - y_conf_bottom


def ramp_consolidation(zeta, t, terms=200):
    """Head / (-decline rate) in a layer with h=0 on top and h=-beta t at the base (zeta from the top)."""
    n = np.arange(1, terms + 1)
    series = ((-1.0) ** n / n**3) * np.exp(-c_v * (n * np.pi / b_conf) ** 2 * t)
    return (t * zeta / b_conf - zeta * (b_conf**2 - zeta**2) / (6 * c_v * b_conf)
            - 2 * b_conf**2 / (np.pi**3 * c_v) * np.sin(np.outer(zeta, n) * np.pi / b_conf) @ series)


print(f"(ii) basin centre x = 0: consolidation coefficient of the confining unit c_v = {c_v:.2f} m^2/d, "
      f"b^2/c_v = {b_conf**2 / c_v:.2f} d")
zeta = np.linspace(0.0, b_conf, 41)
column = np.column_stack((np.full_like(zeta, 1e-3), y_conf_top - zeta))
early = {}
for target in (0.5, 1.0, 2.0, 4.0):
    t = nearest_time(biot, target)
    h_num = at(biot, model.pore_pressure, column, t)
    h_ref = -decline_rate * ramp_consolidation(zeta, t)
    early[t] = (h_num, h_ref)
    scale = decline_rate * t
    print(f"    t = {t:6.3f} d: max |h - h_Terzaghi| / (beta t) = {np.max(np.abs(h_num - h_ref)) / scale:.4f}, "
          f"mid-layer head {h_num[20]:.5f} m vs {h_ref[20]:.5f} m ({100 * (h_num[20] / h_ref[20] - 1):+.2f} %)")

# (ii-b) Uniaxial compaction of the column at the basin centre after 10 years:
# $-\int h\,\rho g/M\,dy$ over the column, with the Biot heads. The regional subsidence
# bowl (leakage length about 4.5 km against 440 m of sediment) also stretches the
# section horizontally, so a deviation of order 1-2 % is physical, not numerical.
ys = np.linspace(-depth, 0.0, 2201)
h_col = at(biot, model.pore_pressure, np.column_stack((np.full_like(ys, 1e-3), ys)), T)
uniaxial = np.trapezoid(h_col / oedometric(layered(E_aquifer, E_confining)(ys)), ys)
uy_centre = at(biot, model.displacement, [[1e-3, 0.0]], T)[0, 1]
uyT_centre = terzaghi_displacement([[1e-3, 0.0]], T)[0]
print(f"    surface settlement at x = 0: Biot {-uy_centre:.4f} m, uniaxial compaction of the Biot column "
      f"{-uniaxial:.4f} m ({100 * (uy_centre / uniaxial - 1):+.2f} %), Terzaghi method {-uyT_centre:.4f} m "
      f"({100 * (uy_centre / uyT_centre - 1):+.2f} %, S_sk ratio {Ssk_biot['aquifer'] / Ssk_aquifer:.4f})")

# (iii) Fluid-volume balance: the fluid content gained by the section equals the time
# integral of the Darcy flux through the boundary. Segments A and B are impermeable,
# so only C, D (recharge from the upper aquifer) and E (pumping) carry flux.
gauss = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10])
gauss_w = np.array([5, 8, 5]) / 18


def boundary_inflow(run, data, t):
    total = {}
    coords = data["coords"]
    for names, normal in ((("outlet", "left_upper"), (-1.0, 0.0)), (("right_upper",), (1.0, 0.0)),
                          (("surface",), (0.0, 1.0))):
        edges = np.array([e for n in names for e in data["mesh"].facets(getattr(data["geometry"].boundaries, n))])
        p0, p1 = coords[edges[:, 0]], coords[edges[:, 1]]
        length = np.linalg.norm(p1 - p0, axis=1)
        pts = (p0[:, None] + gauss[None, :, None] * (p1 - p0)[:, None]).reshape(-1, 2)
        pts = pts - 1e-6 * np.asarray(normal)                       # just inside the section
        q = at(run, run["model"].darcy_flux, pts, t).reshape(len(edges), 3, 2)
        flux_n = q @ np.asarray(normal)
        total[names[0]] = -float(np.sum(length * (flux_n @ gauss_w)))
    return total


times = np.asarray(biot["result"].times, dtype=float)
inflow = np.array([sum(boundary_inflow(biot, data, t).values()) for t in times])
integrated_inflow = np.trapezoid(inflow, times)
stored = biot["result"].field(model.fluid_content).integrate(time=T).value
outlet_T = boundary_inflow(biot, data, T)["outlet"]
pumped = -np.trapezoid([boundary_inflow(biot, data, t)["outlet"] for t in times], times)
print(f"(iii) fluid-volume balance over 10 years (per metre of section): pumped volume {pumped:.0f} m^2, "
      f"recharge from the upper aquifer {pumped + integrated_inflow:.0f} m^2")
print(f"    fluid content change {stored:.2f} m^2 vs time integral of boundary inflow {integrated_inflow:.2f} m^2: "
      f"imbalance {abs(stored - integrated_inflow):.1f} m^2 = {100 * abs(stored - integrated_inflow) / pumped:.3f} % "
      f"of the pumped volume ({100 * (stored / integrated_inflow - 1):+.1f} % of the small net change); "
      f"pumping rate at 10 years {-outlet_T:.2f} m^2/d")

# (iv) Mesh and time-step study of the reported quantities at 10 years. For three
# meshes with $h\propto1/\ell$ the observed order $p$ solves
# $(f_1-f_2)/(f_2-f_3)=(h_1^p-h_2^p)/(h_2^p-h_3^p)$ and the Richardson estimate is
# $f_3+(f_3-f_2)/((h_2/h_3)^p-1)$.


def richardson(values, levels):
    f1, f2, f3 = values
    h1, h2, h3 = 1.0 / np.asarray(levels, float)
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.2, 10.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


def key_quantities(run):
    m = run["model"]
    return {
        "settlement at x = 0, surface / m": -at(run, m.displacement, [[1e-3, 0.0]], T)[0, 1],
        "settlement at x = 3960 m, y = -10 m / m": -at(run, m.displacement, thick, T)[0, 1],
        "settlement at x = 4040 m, y = -10 m / m": -at(run, m.displacement, thin, T)[0, 1],
        "max surface strain above the step / 1e-4": 1e4 * surface_strain(run, T, x_up).max(),
        "head at x = 3000 m, y = -240 m / m": at(run, m.pore_pressure, [[3000.0, -240.0]], T)[0],
    }


study = {level: key_quantities(run) for level, (_, run) in runs.items()}
time_study = key_quantities(fine_steps)
print("(iv) mesh study (Biot, 10 years), levels " + ", ".join(map(str, levels)))
for name in study[levels[0]]:
    values = [study[level][name] for level in levels]
    p, extrapolated = richardson(values, levels)
    order = f"order {p:4.2f}, Richardson {extrapolated:9.5f}" if np.isfinite(p) else "non-monotone (no order)"
    print(f"    {name:44s} " + "  ".join(f"{v:9.5f}" for v in values)
          + f"   {order}, change last two meshes {100 * abs(values[-1] / values[-2] - 1):.3f} %")
print(f"    time steps (level {levels[0]}): change with all steps halved")
for name, value in time_study.items():
    base = study[levels[0]][name]
    print(f"    {name:44s} {base:9.5f} -> {value:9.5f}  ({100 * (value / base - 1):+.4f} %)")
for target in (0.7, 1.4, 2.9):
    deviations = []
    for run in (runs[levels[0]][1], fine_steps):
        t = nearest_time(run, target)
        h = at(run, run["model"].pore_pressure, column, t)
        deviations.append((t, np.max(np.abs(h + decline_rate * ramp_consolidation(zeta, t))) / (decline_rate * t)))
    print(f"    confining unit at x = 0: max |h - h_Terzaghi| / (beta t) = {deviations[0][1]:.4f} at t = "
          f"{deviations[0][0]:.3f} d, with halved steps {deviations[1][1]:.4f} at t = {deviations[1][0]:.3f} d")


# ## Figure
#
# Top row: change of hydraulic head (Biot) after 2, 5 and 10 years, with the 10-year
# contours of the Terzaghi method (dashed) for comparison, as in Leake and Hsieh's
# Fig. 3A. Middle row: settlement (Biot); the 10-year panel adds the Terzaghi-method
# contours, which jump across the step while the Biot field is continuous (their
# Fig. 3B). Vertical scale exaggerated five times. Bottom row: settlement along the
# land surface (Biot and Terzaghi) with the published values next to the step, the
# horizontal strain at the land surface, the consolidation of the confining unit at
# the basin centre against Terzaghi's series, and the mesh study.

coords, triangles = data["coords"], data["triangles"]
edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
unique_edges, edge_index = np.unique(edges, axis=0, return_inverse=True)
edge_index = edge_index.reshape(3, -1).T + len(coords)
plot_points = np.vstack((coords, 0.5 * (coords[unique_edges[:, 0]] + coords[unique_edges[:, 1]])))
a, b, c = triangles.T
ab, bc, ca = edge_index.T
fine_triangles = np.vstack((np.column_stack((a, ab, ca)), np.column_stack((ab, b, bc)),
                            np.column_stack((ca, bc, c)), np.column_stack((ab, bc, ca))))
section = Triangulation(plot_points[:, 0], plot_points[:, 1], fine_triangles)   # P2 nodes, 4 sub-triangles each
owner = np.empty(len(plot_points), dtype=int)
owner[fine_triangles.ravel()] = np.repeat(np.arange(len(fine_triangles)), 3)
inside = plot_points + 1e-6 * (plot_points[fine_triangles].mean(axis=1)[owner] - plot_points)

snapshots = [nearest_time(biot, years * year) for years in (2, 5, 10)]
heads = [at(biot, model.pore_pressure, inside, t) for t in snapshots]
settlements = [-at(biot, model.displacement, inside, t)[:, 1] for t in snapshots]
head_T = at(terzaghi, terzaghi["model"].head, inside, T)
grid_x = np.linspace(1.0, width - 1.0, 261)
columns = terzaghi_columns(grid_x, T)

fig = plt.figure(figsize=(18, 11.5), constrained_layout=True)
layout = fig.add_gridspec(3, 12, height_ratios=(0.8, 0.8, 1.25))
ty = np.linspace(-depth, 0.0, 221)
TX, TY = np.meshgrid(grid_x, ty, indexing="ij")
TU = np.array([np.interp(ty, ys, -uy, left=np.nan) for ys, uy in columns])   # Terzaghi settlement, 10 years
TU[(TX > x_step) & (TY < y_step)] = np.nan


def section_axes(ax, title):
    ax.add_patch(plt.Rectangle((x_step, -depth), width - x_step, depth + y_step, facecolor="#D5D7C8",
                               edgecolor=INK, lw=0.6, hatch="//"))
    ax.axhline(y_conf_top, color=INK, lw=0.4, ls=":")
    ax.axhline(y_conf_bottom, color=INK, lw=0.4, ls=":")
    ax.set_aspect(5.0)
    ax.set(xlim=(0, width), ylim=(-depth, 0), title=title, xlabel="x / m", ylabel="y / m")


for k, t in enumerate(snapshots):
    ax = fig.add_subplot(layout[0, 4 * k:4 * k + 4])
    levels_h = np.linspace(heads[k].min(), 0.0, 25)
    filled = ax.tricontourf(section, heads[k], levels=levels_h, cmap=FIELD_CMAP)
    ax.tricontour(section, heads[k], levels=levels_h[::3], colors=INK, linewidths=0.4)
    if k == 2:
        ax.tricontour(section, head_T, levels=levels_h[::3], colors=COPPER, linewidths=1.0, linestyles="--")
    section_axes(ax, f"Head change, t = {t / year:.0f} yr" + (" (dashed: Terzaghi method)" if k == 2 else ""))
    fig.colorbar(filled, ax=ax, label="head change / m", shrink=0.7, format="%.0f")
for k, t in enumerate(snapshots):
    ax = fig.add_subplot(layout[1, 4 * k:4 * k + 4])
    levels_u = np.linspace(0.0, settlements[k].max(), 25)
    lines_u = np.linspace(0.0, settlements[k].max(), 9)[1:-1]
    filled = ax.tricontourf(section, settlements[k], levels=levels_u, cmap=FIELD_CMAP, extend="min")
    ax.tricontour(section, settlements[k], levels=lines_u, colors=INK, linewidths=0.5)
    if k == 2:
        ax.contour(TX, TY, TU, levels=lines_u, colors=COPPER, linewidths=1.0, linestyles="--")
    section_axes(ax, f"Settlement (Biot), t = {t / year:.0f} yr" + (" (dashed: Terzaghi)" if k == 2 else ""))
    fig.colorbar(filled, ax=ax, label="settlement / m", shrink=0.7, format="%.2f")

ax = fig.add_subplot(layout[2, 0:3])
xs_line = np.linspace(1.0, width - 1.0, 521)
for t, style in zip(snapshots, (":", "--", "-")):
    ax.plot(xs_line, -at(biot, model.displacement, np.column_stack((xs_line, np.full_like(xs_line, -0.01))), t)[:, 1],
            color=GREEN, ls=style, lw=1.5, label=f"Biot surface, {t / year:.0f} yr")
ax.plot(grid_x, [-uy[-1] for ys, uy in columns], color=COPPER, lw=1.4, ls="--", label="Terzaghi method, 10 yr")
ax.plot([x_step - 40], [0.10], "o", color=INK, ms=7, mfc="none", label="published Biot, 10 yr")
ax.plot([x_step + 40, x_step - 40], [0.08, 0.18], "_", color=INK, ms=16, mew=2)
ax.annotate("published < 0.08 m", (x_step + 40, 0.08), xytext=(4250, 0.11), fontsize=7,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
ax.annotate("published Terzaghi > 0.18 m", (x_step - 40, 0.18), xytext=(2400, 0.13), fontsize=7,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
ax.axvline(x_step, color="0.5", lw=0.6, ls=":")
ax.invert_yaxis()
ax.set(xlabel="x / m", ylabel="settlement / m", title="Land-surface settlement")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(layout[2, 3:6])
for t, style in zip(snapshots, (":", "--", "-")):
    ax.plot(x_up[1:-1], 1e4 * surface_strain(biot, t, x_up)[1:-1], color=GREEN, ls=style, label=f"{t / year:.0f} yr")
x_down = np.linspace(2500.0, x_step, 301)
ax.plot(x_down[1:-1], 1e4 * surface_strain(biot, T, x_down)[1:-1], color=GREEN)
ax.axhline(1.0, color=COPPER, lw=1.0, ls="--", label="published: tension > 1e-4")
ax.axvline(x_step, color="0.5", lw=0.6, ls=":")
ax.set(xlabel="x / m", ylabel="horizontal strain / 1e-4", title="Horizontal strain at the land surface (Biot)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(layout[2, 6:9])
fine_zeta = np.linspace(0.0, b_conf, 201)
for (t, (h_num, h_ref)), color in zip(early.items(), (GREEN, COPPER, "#5F9175", "#A85539")):
    ax.plot(-decline_rate * ramp_consolidation(fine_zeta, t), fine_zeta, color=color, lw=1.4)
    ax.plot(h_num, zeta, "o", color=color, ms=3, label=f"t = {t:.2f} d")
ax.invert_yaxis()
ax.set(xlabel="head change / m", ylabel="depth below top of confining unit / m",
       title="Confining unit at x = 0: Terzaghi series (lines)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(layout[2, 9:12])
for name, marker in zip(list(study[levels[0]])[:3], ("o", "s", "^")):
    values = np.array([study[level][name] for level in levels])
    ax.plot(levels, 100 * (values / values[-1] - 1), marker + "-", label=name.split(" / ")[0])
ax.set_xticks(levels)
ax.set(xlabel="mesh level", ylabel="deviation from finest mesh / %", title="Mesh study (10 years)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
plt.show()


# ## Going further
#
# Raise the decline rate or extend the pumping period: the response is quasi-steady
# after a few weeks, so heads and displacements simply scale with the head decline at
# the outlet. Moving the step closer to the basin centre or making the confining unit
# stiffer changes the tensile strain above the step, the quantity that controls earth
# fissuring. With `kinematics="axisymmetric"` the same model describes a single
# pumping well in a layered aquifer.
