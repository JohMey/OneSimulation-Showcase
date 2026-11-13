"""OneSimulation example: electrostatic potential of a uniform space charge.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form parabolic potential.
"""


# # Electrostatics with a uniform space charge between grounded plates
#
# Two parallel plates a distance $d$ apart are both grounded, and the gap is
# filled with a **uniform space charge** of density $\rho$ (for example a region
# of ionized gas with a fixed net charge). The potential then satisfies Poisson's
# equation:
#
# $$-\nabla\cdot(\varepsilon\nabla V)=\rho,\qquad V(0)=V(d)=0.$$
#
# OneSimulation solves this on a real 3D slab with finite elements (Lagrange
# order 2). The four lateral faces carry a zero normal electric displacement, so
# the field is purely one-dimensional along $x$.
#
# **Reference.** With a constant $\rho$ the equation reduces to
# $V''(x)=-\rho/\varepsilon_0$, whose solution for two grounded plates is the
# parabola
#
# $$V(x)=\frac{\rho}{2\varepsilon_0}\,x\,(d-x),\qquad
#   E_x(x)=-V'(x)=\frac{\rho}{2\varepsilon_0}\,(2x-d).$$
#
# The potential peaks at the centre ($V_\max=\rho d^2/8\varepsilon_0$) and the
# field is linear, vanishing at the centre and largest at the plates. We check
# both against these formulas.

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
# The slab lies along $x$; its width and height are small and only form a slender
# 3D mesh. $\rho$ is the net space-charge density in the gap.

d = 0.02              # m, plate separation (x direction)
cross = 0.004         # m, lateral size (field-free in y, z)
rho = 1.0e-5          # C/m^3, uniform space-charge density
eps0 = 8.8541878128e-12

geometry = Box(length=d, width=cross, height=cross)
# Structured hexahedral mesh: many cells along x (the 1D direction), few across.
mesh = StructuredHexahedralMesher(cells=(60, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

model = PoissonEquation(domain=geometry.volume, permittivity=VacuumPermittivity(), charge_density=rho)
problem = Problem(
    name="Uniform space charge between grounded plates",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedPotential(boundary=geometry.boundaries.x_min, physics=model, value=0.0),
        FixedPotential(boundary=geometry.boundaries.x_max, physics=model, value=0.0),
        # lateral faces: no normal field -> 1D behaviour along x
        ZeroNormalElectricDisplacement(
            boundary=(geometry.boundaries.y_min | geometry.boundaries.y_max
                      | geometry.boundaries.z_min | geometry.boundaries.z_max),
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
# Centre-line cut along x (the ends are slightly inset to avoid evaluating exactly
# on the boundary).

x = np.linspace(1e-4, d - 1e-4, 41)
pts = np.column_stack((x, np.full_like(x, cross / 2), np.full_like(x, cross / 2)))
V_num = np.asarray(result.field(model.potential).evaluate(pts)).reshape(-1)
E_num = np.asarray(result.field(model.electric_field).evaluate(pts))
Ex_num = E_num[:, 0]  # field component along the slab

V_ana = (rho / (2 * eps0)) * x * (d - x)
Ex_ana = (rho / (2 * eps0)) * (2 * x - d)

V_peak = rho * d * d / (8 * eps0)
V_rel = float(np.max(np.abs(V_num - V_ana)) / V_peak)
E_scale = (rho / (2 * eps0)) * d
E_rel = float(np.max(np.abs(Ex_num - Ex_ana)) / E_scale)
print(f"peak potential (analytic):   {V_peak:.2f} V")
print(f"max relative potential error: {100*V_rel:.2f} %")
print(f"max relative field error:     {100*E_rel:.2f} %")
assert V_rel < 0.01, "Potential deviation larger than the example tolerance (1 %)."
assert E_rel < 0.03, "Field deviation larger than the example tolerance (3 %)."
# Physical sanity: potential is non-negative and the field reverses at the centre.
assert np.all(V_num > -1e-9), "A positive space charge lifts the potential above ground."
assert Ex_num[0] < 0 < Ex_num[-1], "The field points inward from each grounded plate."


# ## Figure

cream = "#fbf5eb"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax1, ax2):
    ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25); ax.set_xlabel("Position x / mm")
ax1.plot(x * 1000, V_ana, color="#6b442f", label="analytic")
ax1.plot(x * 1000, V_num, "o", color="#c56838", ms=4, label="OneSimulation FEM")
ax1.set(ylabel="Potential / V", title="Potential V(x)"); ax1.legend()
ax2.plot(x * 1000, Ex_ana, color="#2f5b6b", label="analytic")
ax2.plot(x * 1000, Ex_num, "o", color="#3887c5", ms=4, label="OneSimulation FEM")
ax2.set(ylabel="Field E_x / (V/m)", title="Electric field E_x(x)"); ax2.legend()
plt.show()


# ## Going further
#
# Raise the charge density or widen the gap and watch the peak potential grow as
# $\rho d^2$. Ground only one plate and fix the other at a voltage to superpose a
# linear drift on the parabola; the reference then gains the corresponding linear
# term.
