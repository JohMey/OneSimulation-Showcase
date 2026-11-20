"""OneSimulation example: a composite thermal barrier as resolved 3D layers and as a lumped thermal resistance.
Requires the installed, activated OneSimulation runtime.

A copper heater sits on a barrier of four thin insulating layers on a cooled aluminium plate. The
stack is read from STEP files and solved in 3D twice: once with the four layers meshed, once with
the layers replaced by one lumped thermal resistance between heater and plate. Both are compared in
the steady state, with each other and with the series resistance of the layers.
"""

# # Composite thermal barrier: resolved layers and a lumped resistance
#
# Thin insulating layers are stacked to protect a part from a heat source or to set the thermal
# resistance between them. A copper heater (30 mm × 30 mm, 10 mm thick) releasing 1 W rests on a
# barrier of four layers of its footprint, each 1 mm thick, which lies on an aluminium plate
# (60 mm × 60 mm, 5 mm thick) cooled from below. Meshing such layers costs elements through their
# thickness; a common shortcut replaces them by a **lumped thermal resistance** between the faces
# they connect, the series resistance per area
#
# $$R''=\sum_i\frac{d_i}{k_i}.$$
#
# | part | thickness [mm] | $k$ [W/(m K)] | $\rho c$ [MJ/(m³ K)] |
# |---|---|---|---|
# | copper heater | 10 | 400 | 3.43 |
# | graphite foil | 1 | 5 | 0.71 |
# | glass-fibre mat | 1 | 0.04 | 0.084 |
# | ceramic paper | 1 | 0.10 | 0.25 |
# | mica sheet | 1 | 0.5 | 2.46 |
# | aluminium plate | 5 | 200 | 2.43 |
#
# | face | condition |
# |---|---|
# | bottom of the plate | forced air, $h=200$ W/(m² K), 25 °C |
# | all other faces | insulated |
#
# The resolved stack comes from `composite_thermal_barrier_lumped.step` with one region file per
# layer and for the heater; for the lumped model, `composite_thermal_barrier_lumped_plate.step` and
# `composite_thermal_barrier_lumped_heater_on_plate.step` hold the plate and the heater placed
# directly on it. The script `composite_thermal_barrier_lumped_geometry.py` writes all of them with
# the CAD objects of OneSimulation. In the lumped model the faces between heater and plate carry a
# `ThermalContact` of conductance $1/R''$: the temperature jumps across them by $q''R''$.
#
# **Reference.** The barrier is much less conductive than copper and aluminium, so the heat crosses
# it almost uniformly over the footprint $A$, and the temperature step across the barrier is close to
# $P\,R''/A$ (Incropera and DeWitt, *Fundamentals of Heat and Mass Transfer*, ch. 3.1). In the steady
# state all of the 1 W leaves through the cooled face. The resolved model is the reference for the
# lumped one.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.d3 import Assembly
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import TetrahedralMesher, interface
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux,
    HeatConduction,
    HeatFlux,
    ThermalContact,
)
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The resolved stack is one solid with the layers and the heater embedded as regions; its mesh
# follows every interface. The lumped model is an assembly of plate and heater; its mesh keeps the
# two parts apart on their common face, where the contact acts.

stack = load("composite_thermal_barrier_lumped.step")
layers = {
    "mica sheet": ("mica", 0.5, 2.46e6),
    "ceramic paper": ("ceramic_paper", 0.10, 0.25e6),
    "glass-fibre mat": ("glass_fibre", 0.04, 0.084e6),
    "graphite foil": ("graphite_foil", 5.0, 0.71e6),
}
regions = {
    name: load(f"composite_thermal_barrier_lumped_{file}.step")
    for name, (file, _, _) in layers.items()
}
heater = load("composite_thermal_barrier_lumped_heater.step")
lumped = Assembly(
    parts=(
        load("composite_thermal_barrier_lumped_plate.step").named("plate"),
        load("composite_thermal_barrier_lumped_heater_on_plate.step").named("heater"),
    )
)
meshes = {
    "resolved": TetrahedralMesher(
        target_size=0.004, embedded_regions=(*regions.values(), heater)
    ).generate(stack),
    "lumped": TetrahedralMesher(target_size=0.004).generate(lumped),
}

view = plots.Camera(elevation=25, azimuth=-60)
fig = plots.figure(1, 2, size=(15, 6))
plots.geometry(
    [part.named(name) for name, part in (("heater", heater), *regions.items())]
    + [load("composite_thermal_barrier_lumped_plate.step").named("plate")],
    labels=False,
    camera=view,
    title="Heater, four layers and plate (STEP)",
    ax=fig[0],
)
plots.mesh(
    meshes["resolved"],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Resolved layers: {meshes['resolved'].cell_count} quadratic tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Models
#
# Conductivity, heat capacity and heating are given region by region; with the density set to one,
# the specific heat capacity is the volumetric one (it matters only for transients). Both models are
# solved for the steady state.

power, film, ambient = 1.0, 200.0, 298.15
area, heater_volume = (
    heater.size[0] * heater.size[1],
    heater.size[0] * heater.size[1] * heater.size[2],
)
resistance = sum(
    r.size[2] / k for r, (_, k, _) in zip(regions.values(), layers.values())
)
solid = lambda k, c: IsotropicSolid(
    density=1.0,
    specific_heat_capacity=c,
    thermal_conductivity=k,
    youngs_modulus=7e10,
    poisson_ratio=0.33,
    thermal_expansion_coefficient=0.0,
)
resolved = HeatConduction(
    domain=stack.volume,
    material=solid(
        regionwise(
            ((heater, 400.0), *((regions[n], k) for n, (_, k, _) in layers.items())),
            default=200.0,
            unit="W/m/K",
        ),
        regionwise(
            ((heater, 3.43e6), *((regions[n], c) for n, (_, _, c) in layers.items())),
            default=2.43e6,
            unit="J/kg/K",
        ),
    ),
    heating=regionwise(((heater, power / heater_volume),), default=0.0, unit="W/m^3"),
)
contact = HeatConduction(
    domain=lumped.domain(),
    material=solid(
        regionwise({"heater": 400.0, "plate": 200.0}, unit="W/m/K"),
        regionwise({"heater": 3.43e6, "plate": 2.43e6}, unit="J/kg/K"),
    ),
    heating=regionwise({"heater": power / heater_volume, "plate": 0.0}, unit="W/m^3"),
)
problems = {
    "resolved": Problem(
        name="Resolved barrier",
        geometry=stack,
        physics=(resolved,),
        boundary_conditions=(
            ConvectiveHeatFlux(
                boundary=stack.boundaries.bottom,
                physics=resolved,
                heat_transfer_coefficient=film,
                ambient_temperature=ambient,
            ),
            HeatFlux(
                boundary=stack.boundaries.all - stack.boundaries.bottom,
                physics=resolved,
                value=0.0,
            ),
        ),
    ),
    "lumped": Problem(
        name="Lumped barrier",
        geometry=lumped,
        physics=(contact,),
        boundary_conditions=(
            ConvectiveHeatFlux(
                boundary=lumped.boundaries["plate.bottom"],
                physics=contact,
                heat_transfer_coefficient=film,
                ambient_temperature=ambient,
            ),
            HeatFlux(
                boundary=lumped.boundaries.all - lumped.boundaries["plate.bottom"],
                physics=contact,
                value=0.0,
            ),
            ThermalContact(
                boundary=interface(meshes["lumped"], "heater_plate"),
                physics=contact,
                conductance=1 / resistance,
            ),
        ),
    ),
}
models = {"resolved": resolved, "lumped": contact}
spatial = dict(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
steady = {
    name: Simulation(
        problem=problem, mesh=meshes[name], numerics=Numerics(**spatial)
    ).solve_stationary()
    for name, problem in problems.items()
}

# ## Results
#
# Temperatures at the centre of the heater's top and of the plate's top beside the heater, the step
# across the barrier, and the heat into the air.

plate_top, heater_top = 0.005, stack.size[2]
middle = stack.size[0] / 2
probes = {
    "resolved": [
        [middle, middle, heater_top],
        [middle, middle, plate_top + 4e-3],
        [middle, middle, plate_top],
    ],
    "lumped": [
        [middle, middle, plate_top + heater.size[2]],
        [middle, middle, plate_top + 1e-9],
        [middle, middle, plate_top - 1e-9],
    ],
}
values = {
    name: np.ravel(steady[name].field(models[name].temperature).evaluate(points))
    for name, points in probes.items()
}
print(
    f"series resistance of the layers R'' = {1e4 * resistance:.3f} cm² K/W; 1D estimate of the step across the barrier P R''/A = {power * resistance / area:.3f} K"
)
print(
    "steady state           heater top [°C]   barrier top [°C]   plate below the barrier [°C]   step across the barrier [K]   heat into the air [W]"
)
for name in values:
    print(
        f"{name:12s} {values[name][0] - 273.15:24.3f} {values[name][1] - 273.15:18.3f} {values[name][2] - 273.15:30.3f} {values[name][1] - values[name][2]:29.3f}"
        f" {abs(steady[name].field(models[name].temperature).boundary_flux(problems[name].boundary_conditions[0].boundary)):23.5f}"
    )
print(
    f"lumped against resolved: heater top {values['lumped'][0] - values['resolved'][0]:+.1e} K "
    f"({100 * (values['lumped'][0] - values['resolved'][0]) / (values['resolved'][0] - ambient):+.1e} % of the rise)"
)

# ## Figure
#
# Left: the steady temperature on a vertical line through the middle of the stack, resolved and
# lumped (the heater of the lumped model is drawn above the barrier it replaces). Right: the steady temperature of the resolved stack on its vertical mid-plane.

z = np.linspace(0.0, heater_top, 400)
z_lumped = np.linspace(0.0, plate_top + heater.size[2], 300)
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    y={
        "resolved layers": (
            steady["resolved"]
            .field(resolved.temperature)
            .evaluate(
                np.column_stack((np.full_like(z, middle), np.full_like(z, middle), z))
            )
            .ravel()
            - 273.15,
            1e3 * z,
        ),
        "lumped resistance": (
            steady["lumped"]
            .field(contact.temperature)
            .evaluate(
                np.column_stack(
                    (
                        np.full_like(z_lumped, middle),
                        np.full_like(z_lumped, middle),
                        z_lumped,
                    )
                )
            )
            .ravel()
            - 273.15,
            1e3 * np.where(z_lumped > plate_top, z_lumped + 4e-3, z_lumped),
        ),
    },
    hlines={"barrier": 5.0, "": 9.0},
    xlabel="T [°C]",
    ylabel="height in the stack [mm]",
    title="Steady temperature through the middle",
    legend="lower right",
    ax=fig[0],
)
plots.section(
    steady["resolved"].field(resolved.temperature),
    plane=Plane(origin=(0, middle, 0), normal=(0, 1, 0)),
    component=lambda T, p: np.ravel(T) - 273.15,
    limits=(25, 70),
    contours=10,
    subdivisions=2,
    label="T [°C]",
    title="Resolved layers: temperature on the mid-plane",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Give the barrier a larger footprint than the heater in `composite_thermal_barrier_lumped_geometry.py`:
# heat then spreads sideways in the graphite foil, which the lumped resistance cannot do, and the two
# models drift apart. Swapping two layers changes neither model in the steady state.
