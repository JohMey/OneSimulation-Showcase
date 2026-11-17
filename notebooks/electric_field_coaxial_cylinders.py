"""OneSimulation example: electric field between coaxial cylinders.
Requires the installed, activated OneSimulation runtime.

The dielectric of a coaxial capacitor is read from a STEP file, meshed and solved for the
potential; potential and field are checked against the closed-form coaxial solution.
"""


# # Electric field between coaxial cylinders
#
# A coaxial arrangement: an inner conductor (radius $a$) at potential $V_0$ and an
# outer conductor (radius $b$) grounded. The dielectric in between is charge-free,
# so the potential satisfies Laplace's equation:
#
# $$\nabla\cdot(\varepsilon\nabla V)=0,\qquad V(a)=V_0,\quad V(b)=0.$$
#
# The dielectric comes from the CAD file `electric_field_coaxial_cylinders.step` with the named
# faces `inner_wall`, `outer_wall`, `bottom` and `top`; the script
# `electric_field_coaxial_cylinders_geometry.py` builds it with the CAD objects of OneSimulation.
# Top and bottom carry no normal displacement, so the field is purely radial.
#
# **Reference.** For the coaxial capacitor the closed form is
#
# $$V(r)=V_0\,\frac{\ln(b/r)}{\ln(b/a)},\qquad
#   E_r(r)=\frac{V_0}{r\,\ln(b/a)}.$$

import numpy as np

from onesim.cad.exchange import load
from onesim.materials import VacuumPermittivity
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()

# ## Geometry and mesh

dielectric = load("electric_field_coaxial_cylinders.step")
faces = dielectric.boundaries
mesh = TetrahedralMesher(target_size=0.0025, geometry_order=2).generate(dielectric)

fig = plots.figure(1, 2, size=(13, 5))
plots.geometry(dielectric, selection={"inner_wall": faces.inner_wall, "outer_wall": faces.outer_wall}, labels=False,
               title="Dielectric with its electrodes", ax=fig[0])
plots.mesh(mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} tetrahedra", ax=fig[1])
plots.show()

# ## Model

a, b, height, voltage = 0.01, 0.03, 0.02, 100.0
model = PoissonEquation(domain=dielectric.volume, permittivity=VacuumPermittivity(), charge_density=0.0)
problem = Problem(name="Coaxial capacitor", geometry=dielectric, physics=(model,), boundary_conditions=(
    FixedPotential(boundary=faces.inner_wall, physics=model, value=voltage),
    FixedPotential(boundary=faces.outer_wall, physics=model, value=0.0),
    ZeroNormalElectricDisplacement(boundary=faces.top | faces.bottom, physics=model)))
numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()

# ## Comparison with the closed form
#
# Radial cut at half height along the $x$ axis, slightly inset from the two electrodes.

r = np.linspace(a + 1e-4, b - 1e-4, 40)
cut = np.column_stack((r, np.zeros_like(r), np.full_like(r, height / 2)))
potential = result.field(model.potential).evaluate(cut).reshape(-1)
field = np.abs(result.field(model.electric_field).evaluate(cut)[:, 0])
exact_potential = voltage * np.log(b / r) / np.log(b / a)
exact_field = voltage / (r * np.log(b / a))

print(f"largest potential error: {100 * np.max(np.abs(potential - exact_potential)) / voltage:.2f} % of the voltage")
print(f"largest field error:     {100 * np.max(np.abs(field / exact_field - 1)):.2f} %")

# ## Figure

fig = plots.figure(1, 2, size=(11, 4.5))
plots.line(1e3 * r, points={"simulation": potential}, reference={"closed form": exact_potential},
           xlabel="radius r [mm]", ylabel="potential [V]", title="Potential V(r)", ax=fig[0])
plots.line(1e3 * r, points={"simulation": field}, reference={"closed form": exact_field},
           xlabel="radius r [mm]", ylabel="radial field [V/m]", title="Radial field E_r(r)", ax=fig[1])
plots.show()

# ## Going further
#
# Change the radii in `electric_field_coaxial_cylinders_geometry.py`, write the STEP file again and
# see how the field at the inner conductor moves with the ratio $b/a$. A finer mesh lowers the
# field error near $r=a$, where the field is most strongly curved.
