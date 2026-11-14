"""OneSimulation example: heating of three buried power cables.
Requires the installed, activated OneSimulation runtime.

Three power cables lie side by side in the ground and heat each other. The temperature field of the
cross-section is computed with quadratic finite elements on a curved block mesh and compared with
the IEC 60287 thermal circuit.
"""


# # Heating of buried power cables
#
# Three single-core cables of a 132 kV circuit lie in flat formation, 1 m deep and 0.3 m apart.
# Each cable has a copper conductor (1000 mm²), an XLPE insulation, a lead sheath and a
# polyethylene jacket. The conductors carry 1200 A. The ground surface and the deep ground are at
# 20 °C. The cables are long, so the temperature varies only over the cross-section:
#
# $$-\nabla\cdot\big(k\,\nabla T\big)=q,\qquad q=\frac{W}{A_c}\ \text{in each conductor}.$$
#
# The conductivity $k$ jumps by three orders of magnitude between copper and XLPE. The cables heat
# each other, so the middle one runs hottest.
#
# | layer | outer radius | thermal conductivity |
# |---|---|---|
# | copper conductor | 19.0 mm | 400 W/(m K) |
# | XLPE insulation | 35.0 mm | 1/3.5 W/(m K) |
# | lead sheath | 37.5 mm | 35 W/(m K) |
# | PE jacket | 41.5 mm | 1/3.5 W/(m K) |
# | soil | - | 1.0 W/(m K) |
#
# **Loss.** IEC 60287-1-1 gives the conductor an AC resistance of 25.8 µΩ/m at its 90 °C limit, so
# each cable releases $W=I^2R'_{ac}=37.1$ W per metre.

from dataclasses import replace
import numpy as np

from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, disc_blocks, extrude, merge_blocks, o_grid, rectangle_block
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.results.integration import mean_value
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import theme, plots

theme.set()

radii = (0.019, 0.035, 0.0375, 0.0415)
conductivities = (400.0, 1 / 3.5, 35.0, 1 / 3.5)
soil, depth, spacing, half_width, thickness = 1.0, 1.0, 0.3, 40.0, 0.02
centres = ((-spacing, -depth), (0.0, -depth), (spacing, -depth))
surface_temperature, loss = 20.0, 1200.0**2 * 25.795e-6


# ## Reference: the IEC 60287 thermal circuit
#
# The standard replaces the field by thermal resistances per metre: insulation
# $T_1=\frac{\rho_1}{2\pi}\ln\frac{r_i}{r_c}$, jacket $T_3=\frac{\rho_3}{2\pi}\ln\frac{r_e}{r_s}$ and
# soil $T_4=\frac{\rho_4}{2\pi}\ln\big(u+\sqrt{u^2-1}\big)$ with $u=L/r_e$. A neighbour at the
# distance $d$ adds $\frac{\rho_4}{2\pi}\ln\frac{d'}{d}$, where $d'$ is the distance to its mirror
# image above the ground surface (Neher and McGrath, 1957).

def iec_60287():
    """Conductor temperatures of the three cables from the thermal circuit of IEC 60287."""
    r_c, r_i, r_s, r_e = radii
    cable = 3.5 / (2 * np.pi) * (np.log(r_i / r_c) + np.log(r_e / r_s))
    ground = np.log(depth / r_e + np.sqrt((depth / r_e) ** 2 - 1)) / (2 * np.pi * soil)
    neighbour = lambda d: np.log(np.hypot(d, 2 * depth) / d) / (2 * np.pi * soil)
    mutual = np.array([neighbour(spacing) + neighbour(2 * spacing), 2 * neighbour(spacing),
                       neighbour(spacing) + neighbour(2 * spacing)])
    return surface_temperature + loss * (cable + ground + mutual)


# ## Mesh
#
# Blocks with straight and circular sides cover the cross-section: `disc_blocks` gives each cable
# rings that follow the layer interfaces exactly, `o_grid` continues them to a square of side 0.3 m,
# and eight graded rectangles fill the soil around the three squares. `merge_blocks` joins them into curved
# quadratic cells, `extrude` makes a slab one cell thick.

def cable_mesh(n):
    """Cross-section mesh with n cells per quarter of the cable circumference."""
    outer = round(3 * n / 4)
    blocks = []
    for index, centre in enumerate(centres):
        blocks += disc_blocks(centre, radii, n, (round(3 * n / 8), outer, 2, 2),
                              regions=(f"conductor_{index}", None, None, None))
        blocks += o_grid(centre, radii[-1], spacing / 2, n, outer, inner_name=None,
                         radial_grading=Geometric(outer, first=0.35 / outer))
    h, growth = spacing / n, 2.0 ** (4 / n)
    count = lambda length: max(3, int(np.ceil(np.log(1 + length * (growth - 1) / h) / np.log(growth))))
    xs = ((-half_width, half_width - 1.5 * spacing, "end"), (-1.5 * spacing, 3 * spacing, None),
          (1.5 * spacing, half_width - 1.5 * spacing, "start"))
    ys = ((-half_width, half_width - depth - spacing / 2, "end"), (-depth - spacing / 2, spacing, None),
          (-depth + spacing / 2, depth - spacing / 2, "start"))
    for i, (x0, lx, x_toward) in enumerate(xs):
        for j, (y0, ly, y_toward) in enumerate(ys):
            if (i, j) != (1, 1):
                nx, ny = (count(lx) if x_toward else 3 * n), (count(ly) if y_toward else n)
                grading = (x_toward and Geometric(nx, first=h, toward=x_toward),
                           y_toward and Geometric(ny, first=h, toward=y_toward))
                names = ("bottom" if j == 0 else None, "sides" if i == 2 else None,
                         "surface" if j == 2 else None, "sides" if i == 0 else None)
                blocks.append(rectangle_block((x0, y0), lx, ly, (nx, ny), grading, names=names))
    return extrude(merge_blocks(blocks, order=2), thickness, 1, names=("back", "front"))


# ## Model
#
# Conductivity and heat source are given region by region through the distance from the nearest
# cable axis. The ground surface and the bottom are held at 20 °C, all other faces are insulated.

distance = lambda points: np.min([np.hypot(points[:, 0] - cx, points[:, 1] - cy) for cx, cy in centres], axis=0)
conductivity = regionwise(tuple((lambda points, r=r: distance(points) < r, k) for r, k in zip(radii, conductivities)),
                          default=soil, unit="W/m/K")
heating = regionwise(((lambda points: distance(points) < radii[0], loss / (np.pi * radii[0] ** 2)),),
                     default=0.0, unit="W/m^3")
material = replace(IsotropicSolid(density=1000.0, specific_heat_capacity=1000.0, thermal_conductivity=1.0,
                                  youngs_modulus=1e9, poisson_ratio=0.3, thermal_expansion_coefficient=0.0),
                   thermal_conductivity=conductivity)


def solve(n):
    """Temperature and heat-flux field on the mesh with n cells per quarter circumference."""
    mesh = cable_mesh(n)
    faces = mesh.geometry.boundaries
    model = HeatConduction(domain=mesh.geometry.domain(), material=material, heating=heating)
    problem = Problem(name="Buried cables", geometry=mesh.geometry, physics=(model,), boundary_conditions=(
        FixedTemperature(boundary=faces.surface | faces.bottom, physics=model, value=surface_temperature),
        HeatFlux(boundary=faces.sides | faces.back | faces.front, physics=model, value=0.0)))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    return result.field(model.temperature), result.field(model.heat_flux), mesh


runs = [solve(n) for n in (4, 6, 8)]
temperature, heat_flux, mesh = runs[-1]
faces = mesh.geometry.boundaries


# ## Results
#
# Mean conductor temperatures on the three meshes against the thermal circuit. The circuit treats
# every neighbour as a line source in uniform soil, so a difference of a few hundredths of a kelvin
# remains.

conductors = np.array([[mean_value(T, region=m.geometry.domain(f"conductor_{index}")) for index in range(3)]
                       for T, _, m in runs])
circuit = iec_60287()
print("cells     left    middle     right   [°C]")
for (_, _, m), values in zip(runs, conductors):
    print(f"{m.cell_count:5d} " + " ".join(f"{value:9.4f}" for value in values))
print("IEC   " + " ".join(f"{value:9.4f}" for value in circuit))
print("difference on the finest mesh [K]: " + ", ".join(f"{value:+.3f}" for value in conductors[-1] - circuit))
leaving = (temperature.boundary_flux(faces.surface) + temperature.boundary_flux(faces.bottom)) / thickness
print(f"energy balance: generated {3 * loss:.3f} W/m, leaving {leaving:.3f} W/m")


# ## Figure
#
# The temperature of the cross-section, the middle cable with its mesh, the heat flux through the
# ground surface against the line-source formula $q(x)=\sum_n W L/[\pi((x-x_n)^2+L^2)]$, and the
# mesh study.

x = np.linspace(-3, 3, 241)
surface_flux = heat_flux.evaluate(np.column_stack((x, np.zeros_like(x), np.full_like(x, thickness / 2))))[:, 1]
line_sources = sum(loss * depth / (np.pi * ((x - cx) ** 2 + depth**2)) for cx, _ in centres)
peak = float(temperature.values.max())

fig = plots.figure(2, 2, size=(14, 10))
plots.surface(temperature, boundary=faces.back, limits=(surface_temperature, peak), contours=12, subdivisions=2,
              label="T [°C]", title="Temperature of the cross-section", xlimits=(-1.2, 1.2), ylimits=(-2.0, 0.0),
              ylabel="y [m] (ground surface at y = 0)", ax=fig[0])
plots.surface(temperature, boundary=faces.back, limits=(peak - 30, peak), contours=12, edges=True, subdivisions=1,
              label="T [°C]", title="Middle cable and its mesh", xlimits=(-0.07, 0.07),
              ylimits=(-depth - 0.07, -depth + 0.07), ax=fig[1])
plots.line(x, {"finite elements": surface_flux}, reference={"line sources and images": line_sources},
           xlabel="x [m]", ylabel="heat flux out of the ground [W/m²]", title="Heat flux through the ground surface",
           ax=fig[2])
plots.line([m.cell_count for _, _, m in runs],
           {"outer conductors": conductors[:, 0] - circuit[0], "middle conductor": conductors[:, 1] - circuit[1]},
           hlines={"IEC 60287": 0.0}, markers=True, xlabel="number of cells",
           ylabel="finite elements - IEC 60287 [K]", title="Mesh study", ax=fig[3])
plots.show()


# ## Going further
#
# Raise the current until the middle conductor reaches the 90 °C limit of XLPE: that is the rated
# current of the circuit. Put the cables into a trench with a better-conducting backfill, or make
# the soil conductivity temperature-dependent to model the drying of the soil around hot cables.
