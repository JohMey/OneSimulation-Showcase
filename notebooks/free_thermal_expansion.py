"""OneSimulation example: free thermal expansion of a heated bar.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the 1D thermal-elongation formula.
"""


# # Free thermal expansion of a bar
#
# A slender bar (length $L$, square cross-section $h\times h$) is held only at its
# left end face and then heated uniformly by $\Delta T$. With nothing resisting the
# growth, the bar stretches freely and no stress is generated.
#
# OneSimulation solves two coupled stationary problems: a heat conduction problem
# that fixes the temperature to $T_0+\Delta T$ on every face, and the 3D elastic
# equilibrium $-\nabla\!\cdot\boldsymbol\sigma=0$ with an isotropic Hooke law and a
# temperature-driven eigenstrain $\boldsymbol\varepsilon_\text{th}=\alpha\,(T-T_0)\mathbf I$.
# The temperature field is passed into the mechanics through a field coupling.
#
# **Reference.** An unconstrained bar heated by $\Delta T$ elongates by
#
# $$\Delta L=\alpha\,\Delta T\,L,$$
#
# and the stress stays zero everywhere (pure stress-free eigenstrain).

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.field_transfer import FiniteElementEvaluation
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction
from onesim.physics.solid_mechanics import (
    FixedDisplacement, IsotropicThermalExpansion, LinearElasticity, TractionFree,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

length, cross = 0.200, 0.010                    # m, slender square bar
E, nu, alpha = 200e9, 0.30, 12e-6               # Pa, -, 1/K (structural steel)
reference_temperature = 293.15                  # K
delta_T = 80.0                                  # K, uniform temperature rise
elongation_reference = alpha * delta_T * length # m, 1D free-expansion elongation
stress_scale = E * alpha * delta_T              # Pa, scale for the stress tolerance

geometry = Box(length=length, width=cross, height=cross)
material = IsotropicSolid(
    density=7850.0, specific_heat_capacity=500.0, thermal_conductivity=45.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=alpha,
)
heat = HeatConduction(domain=geometry.volume, material=material)
solid = LinearElasticity(
    domain=geometry.volume, material=material,
    thermal_strain=IsotropicThermalExpansion(reference_temperature=reference_temperature),
)

boundaries = geometry.boundaries
problem = Problem(
    name="Bar held at one end under uniform heating",
    geometry=geometry, physics=(solid, heat),
    boundary_conditions=(
        FixedTemperature(boundary=boundaries.all, physics=heat,
                         value=reference_temperature + delta_T),
        FixedDisplacement(boundary=boundaries.x_min, physics=solid, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=boundaries.all - boundaries.x_min, physics=solid),
    ),
    couplings=(
        FieldCoupling(source=heat.temperature, target=solid.temperature,
                      transfer=FiniteElementEvaluation()),
    ),
)
mesh = StructuredHexahedralMesher(cells=(40, 3, 3)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh={heat: mesh, solid: mesh},
    numerics={
        heat: Numerics(spatial=FiniteElement(element=Lagrange(order=1)), linear_solver=SparseDirect()),
        solid: Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
    },
).solve_stationary()


# ## Comparison with the 1D formula

displacement = result.field(solid.displacement)
elongation = float(np.asarray(displacement.evaluate([[length, cross / 2, cross / 2]]))[0, 0])
rel_error = abs(elongation - elongation_reference) / elongation_reference
print(f"Elongation: FEM {elongation*1e6:.3f} um,  1D formula {elongation_reference*1e6:.3f} um")
print(f"relative deviation: {100*rel_error:.2f} %")

# Stress away from the held face should be negligible
stress = result.field(solid.stress)
bulk = np.asarray(stress.evaluate([[length / 2, cross / 2, cross / 2],
                                    [0.8 * length, cross / 2, cross / 2]]))
max_stress = float(np.max(np.abs(bulk)))
print(f"Maximum bulk stress: {max_stress:.3e} Pa  (scale E*alpha*dT = {stress_scale:.3e} Pa)")
assert elongation > 0, "Heating must elongate the bar."
assert rel_error < 0.02, "Deviation larger than the example tolerance (2 % against the 1D formula)."
assert max_stress < 1e-6 * stress_scale, "Free expansion must stay essentially stress free."

# Axial displacement sampled along the central axis
xs = np.linspace(0.0, length, 25)
line = np.column_stack((xs, np.full_like(xs, cross / 2), np.full_like(xs, cross / 2)))
ux = np.asarray(displacement.evaluate(line))[:, 0]
ux_ref = alpha * delta_T * xs


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(xs * 1000, ux_ref * 1e6, color="#214D43", lw=1.8, label="1D formula  alpha dT x")
ax.plot(xs * 1000, ux * 1e6, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.set(xlabel="Position along the bar x / mm", ylabel="Axial displacement u_x / um",
       title="Free thermal expansion: displacement under uniform heating")
ax.legend()
plt.show()


# ## Going further
#
# Clamp the right end face as well and the free growth turns into a compressive
# thermal stress instead. Change the temperature rise and watch the elongation
# scale linearly with dT, or swap the material coefficient alpha to compare how
# different alloys expand under the same heating.
