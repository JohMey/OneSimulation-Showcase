"""OneSimulation example: electric field between coaxial cylinders.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form coaxial solution.
"""


# # Electric field between coaxial cylinders
#
# A coaxial arrangement: an inner conductor (radius $a$) at potential $V_0$ and an
# outer conductor (radius $b$) grounded. The dielectric in between is charge-free,
# so the potential satisfies Laplace's equation:
#
# $$\nabla\cdot(\varepsilon\nabla V)=0,\qquad V(a)=V_0,\quad V(b)=0.$$
#
# OneSimulation solves this on a real 3D annular-cylinder mesh with finite elements
# (Lagrange order 2). The top and bottom have a zero normal displacement, so the
# field is purely radial.
#
# **Reference.** For the coaxial capacitor the closed form is
#
# $$V(r)=V_0\,\frac{\ln(b/r)}{\ln(b/a)},\qquad
#   E_r(r)=\frac{V_0}{r\,\ln(b/a)},\qquad
#   C'=\frac{2\pi\varepsilon}{\ln(b/a)}\ \text{(capacitance per length).}$$
#
# We check potential and field along a radial cut against these formulas.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import AnnularCylinder
from onesim.materials import VacuumPermittivity
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

a = 0.01              # m, inner radius
b = 0.03              # m, outer radius
H = 0.02              # m, height (axial, field-free in z)
V0 = 100.0            # V, potential at the inner conductor
eps0 = 8.8541878128e-12

geometry = AnnularCylinder(inner_radius=a, outer_radius=b, length=H)
mesh = TetrahedralMesher(target_size=(b - a) / 8.0).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")

model = PoissonEquation(domain=geometry.volume, permittivity=VacuumPermittivity(), charge_density=0.0)
problem = Problem(
    name="Coaxial capacitor",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedPotential(boundary=geometry.boundaries.inner_wall, physics=model, value=V0),
        FixedPotential(boundary=geometry.boundaries.outer_wall, physics=model, value=0.0),
        ZeroNormalElectricDisplacement(
            boundary=geometry.boundaries.top | geometry.boundaries.bottom, physics=model),
    ),
)
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with the analytic solution
#
# Radial cut at half height along the x axis (there the radial direction is the
# x direction). The ends are slightly inset to avoid evaluating exactly on the boundary.

r = np.linspace(a + 1e-4, b - 1e-4, 40)
pts = np.column_stack((r, np.zeros_like(r), np.full_like(r, H / 2)))
V_num = np.asarray(result.field(model.potential).evaluate(pts)).reshape(-1)
E_num = np.asarray(result.field(model.electric_field).evaluate(pts))
Er_num = E_num[:, 0]  # radial component = x component on the x axis

V_ana = V0 * np.log(b / r) / np.log(b / a)
Er_ana = V0 / (r * np.log(b / a))

V_rel = float(np.max(np.abs(V_num - V_ana)) / V0)
Er_rel = float(np.max(np.abs(np.abs(Er_num) - Er_ana) / Er_ana))
C_per_length_ana = 2 * np.pi * eps0 / np.log(b / a)
print(f"max relative potential error: {100*V_rel:.2f} %")
print(f"max relative field error:     {100*Er_rel:.2f} %")
print(f"analytic capacitance per length: {C_per_length_ana*1e12:.3f} pF/m")
assert V_rel < 0.02, "Potential deviation larger than the example tolerance (2 %)."
assert Er_rel < 0.06, "Field deviation larger than the example tolerance (6 %)."
assert np.all(Er_num > 0), "The field should point outward (from + to ground)."


# ## Figure

cream = "#fbf5eb"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax1, ax2):
    ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25); ax.set_xlabel("Radius r / mm")
ax1.plot(r * 1000, V_ana, color="#6b442f", label="analytic")
ax1.plot(r * 1000, V_num, "o", color="#c56838", ms=4, label="OneSimulation FEM")
ax1.set(ylabel="Potential / V", title="Potential V(r)"); ax1.legend()
ax2.plot(r * 1000, Er_ana, color="#2f5b6b", label="analytic")
ax2.plot(r * 1000, np.abs(Er_num), "o", color="#3887c5", ms=4, label="OneSimulation FEM")
ax2.set(ylabel="Radial field |E_r| / (V/m)", title="Radial field E_r(r)"); ax2.legend()
plt.show()


# ## Going further
#
# Change the radius ratio b/a and see how the field enhancement at the inner
# conductor and the capacitance per length move with it. A finer mesh lowers the
# field error near r=a, where the field is most strongly curved.
