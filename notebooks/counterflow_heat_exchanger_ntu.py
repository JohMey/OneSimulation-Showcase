"""OneSimulation example: counterflow heat exchanger against the effectiveness-NTU relation.
Requires the installed, activated OneSimulation runtime.

A hot and a cold water stream flow in opposite directions through two thin channels separated by a
plastic wall. The channel is read from a STEP file; the simulated outlet temperatures and
temperature profiles are compared with the closed-form effectiveness-NTU solution.
"""

# # Counterflow heat exchanger
#
# Two water streams flow in opposite directions through parallel channels (1 mm gap, 2 mm wide,
# 45 mm long) separated by a 1.2 mm polypropylene wall. Heat crosses the wall from the hot to the
# cold stream; the outer faces are insulated. Each channel obeys
#
# $$\rho c_p\,\mathbf u\cdot\nabla T=k\,\nabla^2 T ,$$
#
# and exchanges the heat flux $h_w\,(T_h-T_c)$ with the other one through the wall conductance
# $h_w=k_w/t_w$.
#
# Both streams use the channel from the CAD file `counterflow_heat_exchanger_ntu.step` with the
# faces `x_min` and `x_max` (the two ends), `y_min` and `y_max` (across the gap) and the group
# `ends` across the width. The script `counterflow_heat_exchanger_ntu_geometry.py` builds it with
# the CAD objects of OneSimulation.
#
# **Reference.** With the capacity rates $C=\rho c_p u\,t\,w$ of the two streams,
# $C_r=C_\min/C_\max$ and $\mathrm{NTU}=UA/C_\min$, the effectiveness of a counterflow exchanger is
#
# $$\varepsilon=\frac{1-e^{-\mathrm{NTU}(1-C_r)}}{1-C_r\,e^{-\mathrm{NTU}(1-C_r)}} ,$$
#
# and the temperature difference between the streams decays as $e^{-mx}$ with
# $m=Uw\,(1/C_h-1/C_c)$. The overall coefficient combines the wall with the two fluid films,
# $1/U=t_w/k_w+2t/(3k_f)$.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux,
    ConvectiveHeatTransfer,
    FixedTemperature,
    HeatFlux,
)
from onesim.results.integration import mean_value
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# One quadratic element across the gap and 80 along the flow.

duct = load("counterflow_heat_exchanger_ntu.step")
faces = duct.boundaries
length, gap, width = duct.size
mesh = StructuredHexahedralMesher(cells=(80, 1, 1)).generate(duct)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(
    duct,
    selection={"wall to the cold stream": faces.y_max, "hot inlet": faces.x_min},
    labels=False,
    title="Channel of the hot stream",
    ax=fig[0],
)
plots.mesh(
    mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} hexahedra", ax=fig[1]
)
plots.show()

# ## Model
#
# `solve` computes one stream for a given wall temperature of the other stream. The two streams are
# solved in turn until the temperatures settle.

water = IsotropicSolid(
    density=1000.0,
    specific_heat_capacity=4180.0,
    thermal_conductivity=0.6,
    youngs_modulus=2e9,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=1e-5,
)
wall_conductance = 0.12 / 1.2e-3
hot_inlet, cold_inlet, ntu, capacity_ratio = 350.0, 290.0, 1.5, 0.6
overall = 1 / (1 / wall_conductance + 2 * gap / (3 * water.thermal_conductivity))
hot_capacity = overall * length * width / ntu
cold_capacity = hot_capacity / capacity_ratio
speed = lambda capacity: capacity / (
    water.density * water.specific_heat_capacity * gap * width
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2), convection=GalerkinAdvection()),
    linear_solver=SparseDirect(),
)


def solve(velocity, inlet, outlet, wall, outer, inlet_temperature, other_stream):
    """Temperature of one stream; `other_stream(x)` is the wall temperature of the other one."""
    model = ConvectiveHeatTransfer(
        domain=duct.volume, material=water, velocity=(velocity, 0.0, 0.0)
    )
    problem = Problem(
        name="Exchanger channel",
        geometry=duct,
        physics=(model,),
        boundary_conditions=(
            FixedTemperature(boundary=inlet, physics=model, value=inlet_temperature),
            HeatFlux(boundary=outlet | outer | faces.ends, physics=model, value=0.0),
            ConvectiveHeatFlux(
                boundary=wall,
                physics=model,
                heat_transfer_coefficient=wall_conductance,
                ambient_temperature=other_stream,
            ),
        ),
    )
    return (
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .solve_stationary()
        .field(model.temperature)
    )


def at_wall(field, y):
    """Temperature of a stream on its wall face as a function of x."""
    return lambda x: field.evaluate(
        np.column_stack(
            (np.ravel(x), np.full(np.size(x), y), np.full(np.size(x), width / 2))
        )
    ).reshape(np.shape(x))


cold_wall = lambda x: cold_inlet + 0.0 * np.asarray(x)
for sweep in range(12):
    hot = solve(
        speed(hot_capacity),
        faces.x_min,
        faces.x_max,
        faces.y_max,
        faces.y_min,
        hot_inlet,
        cold_wall,
    )
    cold = solve(
        -speed(cold_capacity),
        faces.x_max,
        faces.x_min,
        faces.y_min,
        faces.y_max,
        cold_inlet,
        at_wall(hot, gap),
    )
    cold_wall = at_wall(cold, 0.0)

# ## Results
#
# The bulk temperature of each stream is its mean over the channel cross-section. The
# effectiveness-NTU solution gives the heat flow, the decay rate $m$ and from them both bulk
# temperatures along the exchanger.

x = length * np.linspace(1e-6, 1 - 1e-6, 31)
hot_bulk, cold_bulk = (
    np.array(
        [
            mean_value(field, plane=Plane(origin=(position, 0, 0), normal=(1, 0, 0)))
            for position in x
        ]
    )
    for field in (hot, cold)
)
decay = np.exp(-ntu * (1 - capacity_ratio))
effectiveness = (1 - decay) / (1 - capacity_ratio * decay)
m = overall * width * (1 / hot_capacity - 1 / cold_capacity)
difference = (
    hot_inlet
    - cold_inlet
    - effectiveness * hot_capacity / cold_capacity * (hot_inlet - cold_inlet)
)
hot_exact = (
    hot_inlet - overall * width / hot_capacity * difference * (1 - np.exp(-m * x)) / m
)
cold_exact = hot_exact - difference * np.exp(-m * x)
power = hot_capacity * (hot_inlet - hot_bulk[-1])

print(
    f"hot outlet    simulation {hot_bulk[-1]:.2f} K   effectiveness-NTU {hot_exact[-1]:.2f} K"
)
print(
    f"cold outlet   simulation {cold_bulk[0]:.2f} K   effectiveness-NTU {cold_exact[0]:.2f} K"
)
print(
    f"effectiveness simulation {power / (hot_capacity * (hot_inlet - cold_inlet)):.4f}     effectiveness-NTU {effectiveness:.4f}"
)
print(
    f"heat given off by the hot stream {power:.4f} W, taken up by the cold stream "
    f"{cold_capacity * (cold_bulk[0] - cold_inlet):.4f} W"
)

# ## Figure

plots.line(
    1e3 * x,
    points={
        "hot stream (flows to the right)": hot_bulk,
        "cold stream (flows to the left)": cold_bulk,
    },
    reference={
        "hot stream (flows to the right)": hot_exact,
        "cold stream (flows to the left)": cold_exact,
    },
    xlabel="position along the exchanger [mm]",
    ylabel="bulk temperature [K]",
    legend="upper right",
    title="Counterflow heat exchanger: simulation (markers) and effectiveness-NTU (dashed)",
)
plots.show()

# ## Going further
#
# Make the two capacity rates equal: the temperature difference is then uniform and both profiles
# become straight lines. Reverse the cold stream to obtain a parallel-flow exchanger, whose
# effectiveness is lower at the same NTU. Lengthen the channel in
# `counterflow_heat_exchanger_ntu_geometry.py` and write the STEP file again to raise the NTU.
