"""OneSimulation example: steady conduction through a two-layer composite wall.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the series-resistance solution.
"""


# # Conduction through a composite wall
#
# A plane wall is built from two bonded layers in series: layer 1 of thickness
# $L_1$ and conductivity $k_1$, layer 2 of thickness $L_2$ and conductivity
# $k_2$. The outer face of layer 1 is held at $T_a$, the outer face of layer 2 at
# $T_b$, and the lateral faces are insulated, so the temperature varies only
# along $x$.
#
# OneSimulation solves the stationary heat equation with a position-dependent
# conductivity $k(x)$ on a real 3D mesh with finite elements (Lagrange order 1);
# the mesh is aligned to the material interface so each element sees a single
# conductivity.
#
# **Reference.** In steady state the heat flux $q''$ is the same through both
# layers. With the thermal resistances $R_1 = L_1/k_1$ and $R_2 = L_2/k_2$,
#
# $$q'' = \frac{T_a - T_b}{R_1 + R_2},\qquad
#   T_i = T_a - q''\,R_1,$$
#
# where $T_i$ is the interface temperature. The profile is linear in each layer.

from dataclasses import dataclass, replace
from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Layer 1 is a good conductor (metal), layer 2 an insulation layer. Most of the
# temperature drop falls across the layer with the larger resistance.

L1, L2 = 0.05, 0.05       # m, layer thicknesses (x direction)
L = L1 + L2
cross = 0.02              # m, lateral size (insulated)
k1, k2 = 40.0, 2.0        # W/(m·K), layer conductivities
T_a, T_b = 400.0, 300.0   # K, outer face temperatures

R1, R2 = L1 / k1, L2 / k2
flux_ref = (T_a - T_b) / (R1 + R2)            # W/m^2, steady heat flux
T_interface_ref = T_a - flux_ref * R1         # K, interface temperature
print(f"Analytic heat flux: {flux_ref:.2f} W/m^2,  interface temperature: {T_interface_ref:.3f} K")


# ## Model
#
# The conductivity is supplied as a position-dependent material coefficient that
# returns $k_1$ in layer 1 and $k_2$ in layer 2.

@dataclass(frozen=True)
class LayeredConductivity:
    unit: str = "W/m/K"

    def evaluate(self, *, x, **state):
        return np.where(np.asarray(x) < L1, k1, k2)

geometry = Box(length=L, width=cross, height=cross)
base = IsotropicSolid(
    density=1000.0, specific_heat_capacity=500.0, thermal_conductivity=k1,
    youngs_modulus=100e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
material = replace(base, thermal_conductivity=LayeredConductivity())
# Even cell count along x so an element boundary lands exactly on the interface.
mesh = StructuredHexahedralMesher(cells=(40, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Composite wall conduction",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.x_min, physics=model, value=T_a),
        FixedTemperature(boundary=geometry.boundaries.x_max, physics=model, value=T_b),
        # laterally insulated (zero flux) -> 1D behaviour
        HeatFlux(
            boundary=(geometry.boundaries.y_min | geometry.boundaries.y_max
                      | geometry.boundaries.z_min | geometry.boundaries.z_max),
            physics=model, value=0.0,
        ),
    ),
)
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=1)),
        linear_solver=SparseDirect(),
    ),
).solve_stationary()


# ## Comparison with the series-resistance solution
#
# The reference profile is linear in each layer and passes through the interface
# temperature $T_i$.

def analytic(x):
    x = np.asarray(x, dtype=float)
    in_layer1 = x < L1
    t1 = T_a - flux_ref * x / k1
    t2 = T_interface_ref - flux_ref * (x - L1) / k2
    return np.where(in_layer1, t1, t2)

sample_x = np.linspace(0.0, L, 51)
line = np.column_stack((sample_x, np.full_like(sample_x, cross / 2), np.full_like(sample_x, cross / 2)))
num = np.asarray(result.field(model.temperature).evaluate(line)).reshape(-1)
ana = analytic(sample_x)
max_abs_error = float(np.max(np.abs(num - ana)))
rel_error = max_abs_error / (T_a - T_b)
print(f"Maximum absolute error against the series solution: {max_abs_error:.4f} K  ({100*rel_error:.3f} % of Ta-Tb)")
assert rel_error < 0.01, "Deviation larger than the example tolerance (1 % of Ta-Tb)."

# Interface temperature and the steeper drop in the insulating layer.
T_interface = float(np.asarray(result.field(model.temperature).evaluate([[L1, cross / 2, cross / 2]])).reshape(-1)[0])
assert abs(T_interface - T_interface_ref) < 0.01 * (T_a - T_b), "Interface temperature off."
drop1 = T_a - T_interface
drop2 = T_interface - T_b
assert drop2 > drop1, "The insulating layer must carry the larger temperature drop."
print(f"Interface temperature: FEM {T_interface:.3f} K,  analytic {T_interface_ref:.3f} K")
print(f"Temperature drop: layer 1 {drop1:.2f} K,  layer 2 {drop2:.2f} K")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(sample_x * 1000, ana, color="#214D43", lw=1.8, label="Series-resistance solution")
ax.plot(sample_x * 1000, num, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.axvline(L1 * 1000, color="#8a7a63", ls="--", lw=1.0)
ax.text(L1 * 1000, T_b + 5, "  interface", color="#8a7a63", fontsize=9, va="bottom")
ax.set(xlabel="Position x / mm", ylabel="Temperature / K",
       title="Composite wall: temperature profile (k1 = %g, k2 = %g W/m·K)" % (k1, k2))
ax.legend()
plt.show()


# ## Going further
#
# Change the conductivity ratio $k_1/k_2$ and watch the kink at the interface
# sharpen; the drop across each layer is proportional to its resistance
# $L/k$. Add a third layer by extending the conductivity function and the mesh,
# or add a contact resistance at the interface to model an imperfect bond.
