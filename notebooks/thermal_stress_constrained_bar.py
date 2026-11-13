"""OneSimulation example: thermal stress in an axially constrained bar.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the 1D thermal-stress formula.
"""


# # Thermal stress in a constrained bar
#
# A slender bar (length $L$, square cross-section $h\times h$) is clamped on both
# end faces and then heated uniformly by $\Delta T$. The clamps block the axial
# thermal growth, so a compressive axial stress builds up while the free lateral
# faces let the cross-section breathe.
#
# OneSimulation solves two coupled stationary problems: a heat conduction problem
# that fixes the temperature to $T_0+\Delta T$ on every face, and the 3D elastic
# equilibrium $-\nabla\!\cdot\boldsymbol\sigma=0$ with an isotropic Hooke law and a
# temperature-driven eigenstrain $\boldsymbol\varepsilon_\text{th}=\alpha\,(T-T_0)\mathbf I$.
# The temperature field is passed into the mechanics through a field coupling.
#
# **Reference.** For a bar that cannot elongate but is free to contract/expand
# laterally, the uniaxial stress state gives
#
# $$\sigma_{xx}=-E\,\alpha\,\Delta T,\qquad \sigma_{yy}=\sigma_{zz}=0.$$
#
# The full 3D model fixes the complete end faces, so the lateral motion is also
# blocked there; this end effect makes the computed stress slightly stiffer than
# the 1D value.

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
stress_reference = -E * alpha * delta_T         # Pa, 1D constrained-bar stress

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
    name="Bar clamped at both ends under uniform heating",
    geometry=geometry, physics=(solid, heat),
    boundary_conditions=(
        FixedTemperature(boundary=boundaries.all, physics=heat,
                         value=reference_temperature + delta_T),
        FixedDisplacement(boundary=boundaries.x_min, physics=solid, value=(0.0, 0.0, 0.0)),
        FixedDisplacement(boundary=boundaries.x_max, physics=solid, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=boundaries.all - boundaries.x_min - boundaries.x_max,
                     physics=solid),
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

stress = result.field(solid.stress)
centre = np.asarray(stress.evaluate([[length / 2, cross / 2, cross / 2]]))[0]
sigma_xx, sigma_yy, sigma_zz = centre[0], centre[1], centre[2]
rel_error = abs(sigma_xx - stress_reference) / abs(stress_reference)
print(f"Axial stress: FEM {sigma_xx/1e6:.3f} MPa,  1D formula {stress_reference/1e6:.3f} MPa")
print(f"relative deviation: {100*rel_error:.2f} %  (3D end clamping stiffens the bar slightly)")
print(f"Lateral stresses: sigma_yy {sigma_yy/1e6:.2e} MPa,  sigma_zz {sigma_zz/1e6:.2e} MPa")
assert sigma_xx < 0, "Blocked heating must create a compressive axial stress."
assert rel_error < 0.03, "Deviation larger than the example tolerance (3 % against the 1D formula)."
assert abs(sigma_yy) < 0.01 * abs(stress_reference), "Lateral faces should stay essentially stress free."
assert abs(sigma_zz) < 0.01 * abs(stress_reference), "Lateral faces should stay essentially stress free."

# Axial stress sampled along the central axis
xs = np.linspace(0.0, length, 25)
line = np.column_stack((xs, np.full_like(xs, cross / 2), np.full_like(xs, cross / 2)))
sxx_line = np.asarray(stress.evaluate(line))[:, 0]


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.axhline(stress_reference / 1e6, color="#214D43", lw=1.8, label="1D formula  -E alpha dT")
ax.plot(xs * 1000, sxx_line / 1e6, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.set(xlabel="Position along the bar x / mm", ylabel="Axial stress sigma_xx / MPa",
       title="Constrained bar: thermal stress under uniform heating")
ax.legend()
plt.show()


# ## Going further
#
# Make the bar more slender (larger L/h) and the end clamping matters less, so the
# axial stress along the bar flattens onto the 1D value. Change the temperature
# rise and watch the stress scale linearly with dT, or release one end face to a
# traction-free condition to recover stress-free thermal elongation instead.
