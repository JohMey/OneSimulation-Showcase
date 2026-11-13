"""OneSimulation example: steady temperature distribution in a cooling fin.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against one-dimensional fin theory.
"""


# # Cooling fin with convective surface loss
#
# A straight fin of length $L$ and square cross-section (side $a$) sticks out of a
# hot wall. Its base is held at $T_b$; the four lateral faces lose heat by
# convection to ambient air at $T_\infty$ with coefficient $h$; the tip is
# insulated.
#
# OneSimulation solves the stationary heat equation on a real 3D mesh with finite
# elements (Lagrange order 2) and the convective (Robin) boundary condition on the
# lateral faces.
#
# **Reference.** When the cross-section is nearly isothermal (small transverse
# Biot number), one-dimensional fin theory applies. With cross-sectional area
# $A=a^2$, perimeter $P=4a$ and fin parameter $m=\sqrt{hP/(kA)}$, the excess
# temperature $\theta=T-T_\infty$ for an insulated tip is
#
# $$\frac{\theta(x)}{\theta_b} = \frac{\cosh\!\big(m(L-x)\big)}{\cosh(mL)},\qquad
#   \theta_b = T_b - T_\infty.$$

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux,
    FixedTemperature,
    HeatConduction,
    HeatFlux,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# An aluminium fin: high conductivity and a slender cross-section keep the
# transverse Biot number tiny, so one-dimensional fin theory is an accurate
# target.

L = 0.10              # m, fin length (x direction)
a = 0.01              # m, square cross-section side
k = 200.0            # W/(m·K), thermal conductivity (aluminium)
h = 50.0             # W/(m^2·K), convective coefficient on the lateral faces
T_b = 400.0           # K, base temperature
T_inf = 300.0         # K, ambient temperature

area = a * a          # m^2, cross-sectional area
perimeter = 4 * a     # m, convective perimeter
m = np.sqrt(h * perimeter / (k * area))
theta_b = T_b - T_inf
biot_transverse = h * (a / 2) / k
print(f"Fin parameter m = {m:.3f} 1/m,  mL = {m*L:.3f},  transverse Biot = {biot_transverse:.5f}")

geometry = Box(length=L, width=a, height=a)
# Many cells along x (the fin axis), a few across the slender section.
mesh = StructuredHexahedralMesher(cells=(50, 3, 3)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=2700.0, specific_heat_capacity=900.0, thermal_conductivity=k,
    youngs_modulus=70e9, poisson_ratio=0.33, thermal_expansion_coefficient=23e-6,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Convective cooling fin",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.x_min, physics=model, value=T_b),
        # Convective loss on the four lateral faces.
        ConvectiveHeatFlux(
            boundary=(geometry.boundaries.y_min | geometry.boundaries.y_max
                      | geometry.boundaries.z_min | geometry.boundaries.z_max),
            physics=model,
            heat_transfer_coefficient=h,
            ambient_temperature=T_inf,
        ),
        # Insulated tip.
        HeatFlux(boundary=geometry.boundaries.x_max, physics=model, value=0.0),
    ),
)
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)),
        linear_solver=SparseDirect(),
    ),
).solve_stationary()


# ## Comparison with fin theory
#
# We sample the temperature along the fin axis and compare it with the
# cosh-profile of the insulated-tip fin.

def analytic(x):
    return T_inf + theta_b * np.cosh(m * (L - x)) / np.cosh(m * L)

sample_x = np.linspace(0.0, L, 41)
line = np.column_stack((sample_x, np.full_like(sample_x, a / 2), np.full_like(sample_x, a / 2)))
num = np.asarray(result.field(model.temperature).evaluate(line)).reshape(-1)
ana = analytic(sample_x)
max_abs_error = float(np.max(np.abs(num - ana)))
rel_error = max_abs_error / theta_b
print(f"Maximum absolute error against fin theory: {max_abs_error:.4f} K  ({100*rel_error:.2f} % of theta_b)")
assert rel_error < 0.02, "Deviation larger than the example tolerance (2 % of theta_b)."

# Physical sanity: the fin cools monotonically from base to tip and stays warm.
assert all(b <= a_ + 1e-6 for a_, b in zip(num, num[1:])), "Temperature must fall from base to tip."
T_tip = float(np.asarray(result.field(model.temperature).evaluate([[L, a / 2, a / 2]])).reshape(-1)[0])
print(f"Tip temperature: FEM {T_tip:.3f} K,  fin theory {analytic(L):.3f} K")
assert T_inf < T_tip < T_b, "Tip temperature must lie between ambient and base."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(sample_x * 1000, ana, color="#214D43", lw=1.8, label="1D fin theory")
ax.plot(sample_x * 1000, num, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.axhline(T_inf, color="#8a7a63", ls=":", lw=1.0)
ax.text(L * 1000 * 0.98, T_inf + 1.5, "ambient", color="#8a7a63", fontsize=9, ha="right")
ax.set(xlabel="Position along the fin x / mm", ylabel="Temperature / K",
       title="Cooling fin: temperature along the axis")
ax.legend()
plt.show()


# ## Going further
#
# Shorten the fin or raise the convective coefficient $h$ and watch the tip cool
# toward ambient; the dimensionless group $mL$ controls how much of the fin stays
# effective. Replace the insulated tip with a convective tip condition and compare
# against the corresponding corrected fin solution.
