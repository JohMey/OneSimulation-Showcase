"""OneSimulation example: steady radial conduction through a pipe wall.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form logarithmic profile.
"""


# # Radial heat conduction through a pipe wall
#
# A long pipe wall is an annular cylinder with inner radius $a$ and outer radius
# $b$. The inner surface is held at $T_a$, the outer surface at $T_b$, and the two
# ends (top and bottom) are insulated, so no heat flows axially. With a single
# conductivity $k$ and no heat source the temperature satisfies the steady heat
# equation, which in cylindrical symmetry reduces to a radial problem:
#
# $$\nabla\cdot(k\,\nabla T)=0,\qquad T(a)=T_a,\quad T(b)=T_b,\quad
#   \left.\frac{\partial T}{\partial z}\right|_{\text{ends}}=0.$$
#
# OneSimulation solves this on a real 3D annular-cylinder mesh with finite elements
# (Lagrange order 2); the top and bottom faces carry a zero heat flux, so the field
# is purely radial.
#
# **Reference.** For a cylindrical shell the closed form is the logarithmic profile
#
# $$T(r)=T_a+(T_b-T_a)\,\frac{\ln(r/a)}{\ln(b/a)},$$
#
# with the radial flux $q''(r)=k\,(T_a-T_b)/[r\ln(b/a)]$. We check the computed
# temperature along a radial cut against this formula.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import AnnularCylinder
from onesim.materials import IsotropicSolid
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

a = 0.05              # m, inner radius of the pipe wall
b = 0.08              # m, outer radius of the pipe wall
H = 0.04              # m, axial length (field-free in z)
T_a = 450.0           # K, inner surface temperature (hot fluid side)
T_b = 300.0           # K, outer surface temperature (ambient side)
k = 15.0              # W/(m·K), wall conductivity (stainless-steel order)

geometry = AnnularCylinder(inner_radius=a, outer_radius=b, length=H)
mesh = TetrahedralMesher(target_size=(b - a) / 10.0).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")

material = IsotropicSolid(
    density=8000.0, specific_heat_capacity=500.0, thermal_conductivity=k,
    youngs_modulus=200e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Radial conduction pipe wall",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.inner_wall, physics=model, value=T_a),
        FixedTemperature(boundary=geometry.boundaries.outer_wall, physics=model, value=T_b),
        # ends insulated (zero heat flux) -> purely radial field
        HeatFlux(
            boundary=geometry.boundaries.top | geometry.boundaries.bottom,
            physics=model, value=0.0,
        ),
    ),
)
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with the logarithmic solution
#
# Radial cut at half height along the x axis (there the radial direction is the
# x direction). The ends are slightly inset to avoid evaluating exactly on the
# fixed-temperature boundaries.

r = np.linspace(a + 1e-4, b - 1e-4, 40)
pts = np.column_stack((r, np.zeros_like(r), np.full_like(r, H / 2)))
T_num = np.asarray(result.field(model.temperature).evaluate(pts)).reshape(-1)

T_ana = T_a + (T_b - T_a) * np.log(r / a) / np.log(b / a)

max_abs_error = float(np.max(np.abs(T_num - T_ana)))
rel_error = max_abs_error / (T_a - T_b)
print(f"max relative temperature error: {100*rel_error:.3f} % of Ta-Tb  ({max_abs_error:.4f} K)")
assert rel_error < 0.02, "Deviation larger than the example tolerance (2 % of Ta-Tb)."
# Physical sanity: the profile drops monotonically from Ta at r=a to Tb at r=b and,
# because of the logarithm, is steeper near the inner wall than near the outer wall.
assert np.all(np.diff(T_num) < 0), "Temperature should fall monotonically outward."
drop_inner_half = T_num[0] - T_num[len(r) // 2]
drop_outer_half = T_num[len(r) // 2] - T_num[-1]
assert drop_inner_half > drop_outer_half, "The inner half of the wall should carry the larger drop."
print(f"Temperature drop: inner half {drop_inner_half:.2f} K,  outer half {drop_outer_half:.2f} K")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(r * 1000, T_ana, color="#6b442f", lw=1.8, label="Logarithmic solution")
ax.plot(r * 1000, T_num, "o", color="#c56838", ms=4, label="OneSimulation FEM")
ax.set(xlabel="Radius r / mm", ylabel="Temperature / K",
       title="Pipe wall: radial temperature profile T(r)")
ax.legend()
plt.show()


# ## Going further
#
# Change the radius ratio b/a and watch the profile bend more strongly: a thick
# wall (large b/a) concentrates most of the temperature drop near the inner
# surface. Add a volumetric heat source via the model's heating term to recover
# the self-heated pipe wall, whose reference gains a quadratic term on top of the
# logarithm.
