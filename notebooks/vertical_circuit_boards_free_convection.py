"""OneSimulation example: free-convection cooling of vertical circuit boards (2D).
Requires the installed, activated OneSimulation runtime.

Vertical circuit boards in a rack are cooled by the air that their own heat sets in motion: the
chips warm the air in the channel between two boards, it rises and draws cool air in from below.
The air channel is read from a STEP file and cut to its cross-section; the Navier-Stokes equations
with the Boussinesq buoyancy force and the heat transport are solved together in the air. The heat
balance is checked against the enthalpy leaving the channel, the chip temperatures against the
textbook correlation for vertical channels, and the solution against a second mesh.
"""

# # Free-convection cooling of vertical circuit boards
#
# Two vertical circuit boards (1.6 mm thick, 100 mm high) stand 10 mm apart in a rack. The left
# board carries three chips (2 mm thick, 12 mm high) on the side facing the channel; each chip
# gives off 3 W per metre of board depth through its three faces in the air. Without a fan the
# air (25 °C) is driven by buoyancy alone. The steady Boussinesq equations hold in the air,
#
# $$\rho(\mathbf u\cdot\nabla)\mathbf u=-\nabla p+\mu\nabla^2\mathbf u-\rho\beta(T-T_0)\mathbf g,
#   \qquad \nabla\cdot\mathbf u=0,\qquad \rho c_p\,\mathbf u\cdot\nabla T=k\nabla^2T ,$$
#
# for air at 25 °C ($\rho=1.184$ kg/m³, $\mu=1.849\cdot10^{-5}$ Pa s, $k=0.02551$ W/(m K),
# $c_p=1007$ J/(kg K), $\beta=1/T_0$). The board faces take up no heat (their other sides face
# neighbouring channels of the same rack). The model includes 20 mm of air below and 30 mm above the
# boards, so that the air enters and leaves the channel freely.
#
# The geometry comes from the CAD files `vertical_circuit_boards_free_convection*.step`, written by
# `vertical_circuit_boards_free_convection_geometry.py` with the CAD objects of OneSimulation: the
# air channel (`_air.step`, with the faces of every board and chip named after them), and the
# boards and chips as parts of their own.
#
# | face | flow | heat |
# |---|---|---|
# | `inlet` (bottom) | open, no traction | $T=25$ °C |
# | `outlet` (top) | open, no traction (backflow damped) | no conduction |
# | `sides` | free slip | no flux |
# | chip faces | no slip | heat flux 3 W/m per chip |
# | board faces | no slip | no flux |
#
# **Reference.** In the steady state the 9 W/m of the chips leave with the air: the enthalpy flow
# through the top plus the heat conducted back through the inlet. For a vertical channel with one
# wall heated at a uniform flux $q''$ and the other insulated, Bar-Cohen and Rohsenow give the wall
# temperature at the channel exit (F. P. Incropera et al., *Fundamentals of Heat and Mass Transfer*,
# Eq. 9.46 and Table 9.3),
#
# $$Nu_{S,L}=\frac{q''S}{k\,(T_{s,L}-T_0)}=\left[\frac{48}{Ra_S^*S/L}+\frac{2.51}{(Ra_S^*S/L)^{2/5}}\right]^{-1/2},
#   \qquad Ra_S^*=\frac{g\beta q''S^4}{k\alpha\nu},$$
#
# with the gap $S$, the board height $L$ and the mean flux $q''=9\,\mathrm{W/m}/L$. The chips
# concentrate the heat, so the correlation estimates the temperature of the top chip.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import ThermalFluid
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.coupled_systems import StationaryPartitionedCoupling
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange, TaylorHood
from onesim.numerics.field_transfer import FieldInterpolation
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import FixedPoint, Newton
from onesim.physics.heat_transfer import (
    BoussinesqBuoyancy,
    ConvectiveHeatTransfer,
    FixedTemperature,
    HeatFlux,
)
from onesim.physics.hydrodynamics import (
    BackflowTraction,
    FreeSlipLaw,
    IncompressibleNavierStokes,
    NoSlipWall,
    OpenBoundaryTraction,
    WallShear,
)
from onesim.results.derived import derived_field
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The air channel is cut at half depth; its outline keeps the face names, so each chip is the group
# of its three faces in the air. The triangles are 0.5 mm along boards and chips, where the thermal
# boundary layers sit, and grow to 2.5 mm in the open air; a second mesh is coarser by a factor 1.4.

slug = "vertical_circuit_boards_free_convection"
parts = {
    name: load(f"{slug}_{name}.step")
    for name in ("air", "board_left", "board_right", "chip_1", "chip_2", "chip_3")
}
air = parts["air"].section((0, 0, 1), origin=(0, 0, parts["air"].size[2] / 2))
faces = air.boundaries
chips = [
    faces[f"chip_{index}.x_max"]
    | faces[f"chip_{index}.y_min"]
    | faces[f"chip_{index}.y_max"]
    for index in (1, 2, 3)
]
meshes = [
    TriangularMesher(
        target_size=2.5e-3 * scale, local_sizes={"solids": 5e-4 * scale}, grading=0.25
    ).generate(air)
    for scale in (1.4, 1.0)
]

fig = plots.figure(1, 2, size=(12, 8))
plots.geometry(
    parts["air"],
    selection={
        "inlet": parts["air"].boundaries.inlet,
        "outlet": parts["air"].boundaries.outlet,
        "boards and chips": parts["air"].boundaries.solids,
    },
    labels=False,
    camera=plots.Camera(elevation=60, azimuth=-75),
    title="Air channel between the boards",
    ax=fig[0],
)
plots.mesh(
    meshes[-1],
    groups=("all",),
    legend=False,
    title=f"Finer mesh: {meshes[-1].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The flow (Taylor-Hood elements) and the heat transport (quadratic elements) share each mesh.
# `BoussinesqBuoyancy` turns the temperature into the body force, with $\rho$ and $\beta$ from the
# `ThermalFluid`; the solved velocity carries the heat, and `StationaryPartitionedCoupling` iterates
# the two to their common steady state. The sweeps start with the heat transport in an assumed
# uniform updraught of 5 cm/s: a first heat solve in still air would have to conduct the whole chip
# power down to the inlet and would start the iteration from absurd temperatures. Inlet and outlet are open boundaries; `BackflowTraction` controls the energy
# of any air re-entering through them.

ambient, gravity, power = 298.15, 9.81, 3.0
chip_faces = 12e-3 + 2 * 2e-3
air_properties = ThermalFluid(
    density=1.184,
    dynamic_viscosity=1.849e-5,
    thermal_conductivity=0.02551,
    specific_heat_capacity=1007.0,
    thermal_expansion_coefficient=1 / ambient,
)
flow = IncompressibleNavierStokes(domain=air.domain(), material=air_properties)
heat = ConvectiveHeatTransfer(
    domain=air.domain(), material=air_properties, velocity=None
)
problem = Problem(
    name="Vertical circuit boards",
    geometry=air,
    physics=(flow, heat),
    boundary_conditions=(
        OpenBoundaryTraction(
            physics=flow,
            boundary=faces.inlet | faces.outlet,
            value=0.0,
            law=BackflowTraction(),
        ),
        WallShear(physics=flow, boundary=faces.sides, law=FreeSlipLaw()),
        NoSlipWall(physics=flow, boundary=faces.solids),
        FixedTemperature(physics=heat, boundary=faces.inlet, value=ambient),
        HeatFlux(
            physics=heat,
            boundary=chips[0] | chips[1] | chips[2],
            value=-power / chip_faces,
        ),
        HeatFlux(
            physics=heat,
            boundary=faces.outlet
            | faces.sides
            | (faces.solids - chips[0] - chips[1] - chips[2]),
            value=0.0,
        ),
    ),
    initial_conditions=(
        InitialValue(heat.temperature, ambient),
        InitialValue(flow.velocity, (0.0, 0.05)),
    ),
    couplings=(
        FieldCoupling(
            source=heat.temperature,
            target=flow.field("body_force"),
            transfer=BoussinesqBuoyancy(
                gravity=(0, -gravity), reference_temperature=ambient
            ),
        ),
        FieldCoupling(
            source=flow.velocity,
            target=heat.advecting_velocity,
            transfer=FieldInterpolation(),
        ),
    ),
)
numerics = {
    flow: Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        initial_guess={flow.velocity: (0.0, 0.0)},
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(),
            absolute_tolerance=1e-12,
            relative_tolerance=1e-9,
            maximum_iterations=50,
        ),
    ),
    heat: Numerics(
        spatial=FiniteElement(
            element=Lagrange(order=2), convection=GalerkinAdvection()
        ),
        linear_solver=SparseDirect(),
    ),
}
results = [
    Simulation(
        problem=problem,
        mesh={flow: mesh, heat: mesh},
        numerics=numerics,
        coupling_solver=StationaryPartitionedCoupling(
            stationary_systems=(flow, heat),
            order=(heat, flow),
            iteration=FixedPoint(
                relative_tolerance=1e-6,
                absolute_tolerance=1e-9,
                maximum_iterations=250,
                relaxation=0.4,
            ),
        ),
    ).solve_stationary()
    for mesh in meshes
]

# ## Heat balance and chip temperatures
#
# `derived_field` forms the enthalpy flux $\rho c_p v\,(T-T_0)$ of the air, integrated over the
# outlet; the chip temperature is the largest value on the chip face towards the channel.

gap, height = 10e-3, 0.1
nu, alpha = (
    air_properties.dynamic_viscosity / air_properties.density,
    air_properties.thermal_conductivity
    / (air_properties.density * air_properties.specific_heat_capacity),
)
flux = 3 * power / height
rayleigh = (
    gravity
    * air_properties.thermal_expansion_coefficient
    * flux
    * gap**4
    / (air_properties.thermal_conductivity * alpha * nu)
    * gap
    / height
)
exit_rise = (
    flux
    * gap
    / air_properties.thermal_conductivity
    * np.sqrt(48 / rayleigh + 2.51 / rayleigh**0.4)
)
hottest = []
for name, mesh, result in zip(("coarser", "finer"), meshes, results):
    temperature, velocity = result.field(heat.temperature), result.field(flow.velocity)
    enthalpy = derived_field(
        velocity,
        lambda u, points, T=temperature: air_properties.density
        * air_properties.specific_heat_capacity
        * u[:, 1]
        * (np.ravel(T.evaluate(points)) - ambient),
        name="enthalpy_flux",
        unit="W/m^2",
    )
    carried, back = enthalpy.integrate(
        boundary=faces.outlet
    ).value, temperature.boundary_flux(faces.inlet)
    inflow = (
        derived_field(
            velocity, lambda u, points: u[:, 1], name="vertical_velocity", unit="m/s"
        )
        .integrate(boundary=faces.inlet)
        .value
    )
    hottest.append(
        [
            temperature.evaluate(
                np.column_stack(
                    (
                        np.full(61, box[1][0] + 1e-7),
                        np.linspace(box[0][1], box[1][1], 61),
                    )
                )
            ).max()
            - 273.15
            for box in (parts[f"chip_{index}"].bounding_box for index in (1, 2, 3))
        ]
    )
    print(
        f"{name} mesh ({mesh.cell_count} triangles, {len(result.result_set.solver_report['coupling']['scaled_changes'])} coupling sweeps)"
    )
    print(
        f"  chip power {3 * power:.3f} W/m; enthalpy through the outlet {carried:.4f} W/m, conducted back through the inlet "
        f"{back:.4f} W/m; balance {100 * (carried + back - 3 * power) / (3 * power):+.3f} %"
    )
    print(
        f"  air drawn through the channel {1e3 * inflow:.4f} l/(s m), mean speed in the gap {1e3 * inflow / gap:.1f} mm/s, mixed outlet "
        f"temperature rise {3 * power / (air_properties.density * air_properties.specific_heat_capacity * inflow):.2f} K"
    )
    print(
        "  chip temperatures (bottom to top): "
        + ", ".join(f"{value:.2f} °C" for value in hottest[-1])
    )
print(
    f"\nRa*_S S/L = {rayleigh:.1f}; channel correlation: wall {exit_rise:.2f} K above ambient at the channel exit"
)
print(
    f"top chip {hottest[-1][-1] - 25:.2f} K above ambient ({100 * ((hottest[-1][-1] - 25) / exit_rise - 1):+.1f} % from the correlation); "
    f"change from the coarser mesh {hottest[-1][-1] - hottest[0][-1]:+.3f} K"
)

# ## Figure
#
# Temperature and speed of the air (the chips on the left board are the hot spots), and the
# temperature along the channel face of the left board and of the right board.

temperature, velocity = results[-1].field(heat.temperature), results[-1].field(
    flow.velocity
)
(bx0, by0, _), (bx1, by1, _) = parts["board_left"].bounding_box
right_face = parts["board_right"].bounding_box[0][0]
along = np.linspace(by0, by1, 500)
fig = plots.figure(
    layout=[["temperature", "speed", "profiles", "profiles"]], size=(15, 9)
)
plots.surface(
    temperature,
    component=lambda T, position: T - 273.15,
    limits=(25.0, np.ceil(max(hottest[-1]))),
    contours=12,
    subdivisions=2,
    label="T [°C]",
    xlabel="x [m]",
    ylabel="y [m]",
    title="Air temperature",
    ax=fig["temperature"],
)
plots.surface(
    velocity,
    component=lambda u, position: 1e3 * np.linalg.norm(u, axis=1),
    contours=8,
    subdivisions=2,
    label="speed [mm/s]",
    xlabel="x [m]",
    ylabel="y [m]",
    title="Speed of the air",
    ax=fig["speed"],
)
plots.line(
    y={
        "left board with chips": (
            temperature.evaluate(
                np.column_stack((np.full_like(along, bx1 + 2.1e-3), along))
            ).ravel()
            - 273.15,
            1e3 * (along - by0),
        ),
        "right board": (
            temperature.evaluate(
                np.column_stack((np.full_like(along, right_face - 1e-7), along))
            ).ravel()
            - 273.15,
            1e3 * (along - by0),
        ),
    },
    points={"correlation, channel exit": ([25.0 + exit_rise], [1e3 * (by1 - by0)])},
    vlines={"ambient": 25.0},
    xlabel="temperature [°C]",
    ylabel="height along the board [mm]",
    title="Temperature 2.1 mm off the left board and on the right board",
    legend="lower right",
    ax=fig["profiles"],
)
plots.show()

# ## Going further
#
# Change the board spacing in `vertical_circuit_boards_free_convection_geometry.py` and look for
# the spacing with the coolest chips (too narrow chokes the flow, too wide wastes rack space); move
# the hottest chip to the bottom of the board; or impose an inflow at the inlet, as a fan would, to
# compare free and forced cooling.
