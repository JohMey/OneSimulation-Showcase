"""OneSimulation example: counterflow heat exchanger against the effectiveness-NTU relation.
Requires the installed, activated OneSimulation runtime.

A hot and a cold water stream flow in opposite directions through two thin channels separated by a
plastic wall. The simulated outlet temperatures and temperature profiles are compared with the
closed-form effectiveness-NTU solution.
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
from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, ConvectiveHeatTransfer, FixedTemperature, HeatFlux
from onesim.results.integration import mean_value
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()

length, gap, width = 0.045, 1.0e-3, 2.0e-3
water = IsotropicSolid(density=1000.0, specific_heat_capacity=4180.0, thermal_conductivity=0.6,
                       youngs_modulus=2e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5)
wall_conductance = 0.12 / 1.2e-3
hot_inlet, cold_inlet, ntu, capacity_ratio = 350.0, 290.0, 1.5, 0.6
overall = 1 / (1 / wall_conductance + 2 * gap / (3 * water.thermal_conductivity))
hot_capacity = overall * length * width / ntu
cold_capacity = hot_capacity / capacity_ratio
speed = lambda capacity: capacity / (water.density * water.specific_heat_capacity * gap * width)


def ntu_solution(x):
    """Bulk temperatures of the hot and the cold stream along the exchanger (effectiveness-NTU)."""
    decay = np.exp(-ntu * (1 - capacity_ratio))
    power = (1 - decay) / (1 - capacity_ratio * decay) * hot_capacity * (hot_inlet - cold_inlet)
    m = overall * width * (1 / hot_capacity - 1 / cold_capacity)
    difference = hot_inlet - (cold_inlet + power / cold_capacity)
    hot = hot_inlet - overall * width / hot_capacity * difference * (1 - np.exp(-m * x)) / m
    return hot, hot - difference * np.exp(-m * x)


# ## Model
#
# Both channels use the same slender box with one quadratic element across the gap and 80 along
# the flow. `solve` computes one stream for a given wall temperature of the other stream. The two
# streams are solved in turn until the temperatures settle.

duct = Box(length=length, width=gap, height=width)
faces = duct.boundaries
mesh = StructuredHexahedralMesher(cells=(80, 1, 1)).generate(duct)
numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2), convection=GalerkinAdvection()), linear_solver=SparseDirect())


def solve(velocity, inlet, outlet, wall, outer, inlet_temperature, other_stream):
    """Temperature of one stream; `other_stream(x)` is the wall temperature of the other one."""
    model = ConvectiveHeatTransfer(domain=duct.volume, material=water, velocity=(velocity, 0.0, 0.0))
    problem = Problem(name="Exchanger channel", geometry=duct, physics=(model,), boundary_conditions=(
        FixedTemperature(boundary=inlet, physics=model, value=inlet_temperature),
        HeatFlux(boundary=outlet | outer | faces.z_min | faces.z_max, physics=model, value=0.0),
        ConvectiveHeatFlux(boundary=wall, physics=model, heat_transfer_coefficient=wall_conductance,
                           ambient_temperature=other_stream)))
    return Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.temperature)


def at_wall(field, y):
    """Temperature of a stream on its wall face as a function of x."""
    return lambda x: field.evaluate(np.column_stack((np.ravel(x), np.full(np.size(x), y), np.full(np.size(x), width / 2)))).reshape(np.shape(x))


cold_wall = lambda x: cold_inlet + 0.0 * np.asarray(x)
for sweep in range(12):
    hot = solve(speed(hot_capacity), faces.x_min, faces.x_max, faces.y_max, faces.y_min, hot_inlet, cold_wall)
    cold = solve(-speed(cold_capacity), faces.x_max, faces.x_min, faces.y_min, faces.y_max, cold_inlet, at_wall(hot, gap))
    cold_wall = at_wall(cold, 0.0)


# ## Results
#
# The bulk temperature of each stream is its mean over the channel cross-section.

x = length * np.linspace(1e-6, 1 - 1e-6, 31)
bulk = lambda field: np.array([mean_value(field, plane=Plane(origin=(position, 0, 0), normal=(1, 0, 0))) for position in x])
hot_bulk, cold_bulk = bulk(hot), bulk(cold)
hot_exact, cold_exact = ntu_solution(x)
power = hot_capacity * (hot_inlet - hot_bulk[-1])
print(f"hot outlet    simulation {hot_bulk[-1]:.2f} K   effectiveness-NTU {hot_exact[-1]:.2f} K")
print(f"cold outlet   simulation {cold_bulk[0]:.2f} K   effectiveness-NTU {cold_exact[0]:.2f} K")
print(f"effectiveness simulation {power / (hot_capacity * (hot_inlet - cold_inlet)):.4f}     "
      f"effectiveness-NTU {(hot_inlet - hot_exact[-1]) / (hot_inlet - cold_inlet):.4f}")
print(f"heat given off by the hot stream {power:.4f} W, taken up by the cold stream "
      f"{cold_capacity * (cold_bulk[0] - cold_inlet):.4f} W")


# ## Figure

plots.line(1e3 * x, points={"hot stream (flows to the right)": hot_bulk, "cold stream (flows to the left)": cold_bulk},
           reference={"hot stream (flows to the right)": hot_exact, "cold stream (flows to the left)": cold_exact},
           xlabel="position along the exchanger [mm]", ylabel="bulk temperature [K]", legend="center right",
           title="Counterflow heat exchanger: simulation (markers) and effectiveness-NTU (dashed)")
plots.show()


# ## Going further
#
# Make the two capacity rates equal: the temperature difference is then uniform and both profiles
# become straight lines. Reverse the cold stream to obtain a parallel-flow exchanger, whose
# effectiveness is lower at the same NTU.
