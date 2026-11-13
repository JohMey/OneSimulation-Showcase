"""OneSimulation example: the parallel-plate capacitor.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form parallel-plate result.
"""


# # Parallel-plate capacitor
#
# Two parallel conducting plates, a distance $d$ apart, face each other across a
# charge-free vacuum gap. One plate is held at potential $V_0$, the other is
# grounded. On the four side walls no charge leaves the domain, so the field stays
# one-dimensional and we recover the textbook parallel-plate capacitor without
# fringing.
#
# The potential satisfies Laplace's equation in the gap,
#
# $$\nabla\cdot(\varepsilon\nabla V)=0,\qquad V(0)=V_0,\quad V(d)=0,$$
#
# and OneSimulation solves it on a real 3D hexahedral mesh with finite elements
# (Lagrange order 2).
#
# **Reference.** For the ideal parallel-plate capacitor the closed form is
#
# $$V(x)=V_0\left(1-\frac{x}{d}\right),\qquad
#   E_x=\frac{V_0}{d},\qquad
#   C=\frac{\varepsilon_0\,A}{d},$$
#
# with $A$ the plate area. We check the potential along the gap, the uniform field,
# and the capacitance obtained from the charge flux against these formulas.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import VacuumPermittivity
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The gap runs along x, so the two plates are the x_min face (at $V_0$) and the
# x_max face (grounded). The four remaining faces carry a zero normal electric
# displacement, which keeps the field purely axial.

d = 0.002             # m, plate separation (gap along x)
W = 0.02              # m, plate width (y)
H = 0.02              # m, plate height (z)
V0 = 50.0             # V, potential on the x_min plate
eps0 = 8.8541878128e-12
A = W * H             # m^2, plate area

geometry = Box(length=d, width=W, height=H)
mesh = StructuredHexahedralMesher(cells=(12, 6, 6)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

model = PoissonEquation(domain=geometry.volume, permittivity=VacuumPermittivity(), charge_density=0.0)
problem = Problem(
    name="Parallel-plate capacitor",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedPotential(boundary=geometry.boundaries.x_min, physics=model, value=V0),
        FixedPotential(boundary=geometry.boundaries.x_max, physics=model, value=0.0),
        ZeroNormalElectricDisplacement(
            boundary=geometry.boundaries.y_min | geometry.boundaries.y_max
            | geometry.boundaries.z_min | geometry.boundaries.z_max,
            physics=model),
    ),
)
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with the analytic solution
#
# Axial cut along x at the centre of the plates. The ends are slightly inset to
# avoid evaluating exactly on the boundary.

x = np.linspace(1e-5, d - 1e-5, 40)
pts = np.column_stack((x, np.full_like(x, W / 2), np.full_like(x, H / 2)))
V_num = np.asarray(result.field(model.potential).evaluate(pts)).reshape(-1)
E_num = np.asarray(result.field(model.electric_field).evaluate(pts))
Ex_num = E_num[:, 0]  # axial component

V_ana = V0 * (1.0 - x / d)
Ex_ana = V0 / d

V_rel = float(np.max(np.abs(V_num - V_ana)) / V0)
Ex_rel = float(np.max(np.abs(Ex_num - Ex_ana)) / Ex_ana)

# Capacitance from the charge on the grounded plate: Q = eps0 * flux of E through it.
flux = result.field(model.electric_field).integrate(
    boundary=geometry.boundaries.x_max, normal_component=True)
C_num = eps0 * abs(float(flux.value)) / V0
C_ana = eps0 * A / d
C_rel = abs(C_num - C_ana) / C_ana

print(f"max relative potential error: {100*V_rel:.3f} %")
print(f"max relative field error:     {100*Ex_rel:.3f} %")
print(f"analytic capacitance: {C_ana*1e12:.4f} pF")
print(f"numeric  capacitance: {C_num*1e12:.4f} pF  ({100*C_rel:.3f} % off)")

assert V_rel < 0.01, "Potential deviation larger than the example tolerance (1 %)."
assert Ex_rel < 0.01, "Field deviation larger than the example tolerance (1 %)."
assert C_rel < 0.01, "Capacitance deviation larger than the example tolerance (1 %)."
assert np.all(Ex_num > 0), "The field should point from the live plate to ground."


# ## Figure

cream = "#fbf5eb"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax1, ax2):
    ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25); ax.set_xlabel("Position x / mm")
ax1.plot(x * 1000, V_ana, color="#6b442f", label="analytic")
ax1.plot(x * 1000, V_num, "o", color="#c56838", ms=4, label="OneSimulation FEM")
ax1.set(ylabel="Potential / V", title="Potential V(x)"); ax1.legend()
ax2.plot(x * 1000, np.full_like(x, Ex_ana), color="#2f5b6b", label="analytic")
ax2.plot(x * 1000, Ex_num, "o", color="#3887c5", ms=4, label="OneSimulation FEM")
ax2.set(ylabel="Axial field E_x / (V/m)", title="Axial field E_x(x)"); ax2.legend()
plt.show()


# ## Going further
#
# Shrink the gap d and watch the capacitance grow as 1/d while the field V0/d rises
# to match. Replacing the vacuum permittivity with a dielectric scales the
# capacitance by the relative permittivity without changing the field pattern.
