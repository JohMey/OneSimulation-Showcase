"""OneSimulation example: magnetic field of a Helmholtz coil.
Requires the installed, activated OneSimulation runtime.

Two coaxial circular coils, spaced one radius apart, create a highly uniform field between them.
The 3D field is computed with edge elements and compared with the closed-form field on the axis.
"""


# # Magnetic field of a Helmholtz coil
#
# Two circular coils of radius $R=0.4$ m with 10 turns each sit at $z=\pm R/2$ and carry the same
# current of 0.25 mA. At this spacing the first three derivatives of the axial field vanish at the
# centre, so the field between the coils is very uniform.
#
# The model solves $\nabla\times(\mu_0^{-1}\nabla\times\mathbf A)=\mathbf J$ with second-order
# Nédélec edge elements and a Coulomb gauge. The windings have a square cross-section of 5 cm and
# a uniform current density. The air is a sphere of radius $a=2$ m with magnetic insulation.
#
# **Reference.** On the axis the field of the two loops is
#
# $$B_z(z)=\frac{\mu_0NIR^2}{2}\left[\big(R^2+(z-\tfrac R2)^2\big)^{-3/2}+\big(R^2+(z+\tfrac R2)^2\big)^{-3/2}\right]
#   -\frac{\mu_0NIR^2}{a^3} ,$$
#
# with the centre value $(4/5)^{3/2}\mu_0NI/R$. The last term is the image field of the insulated
# sphere, which sees the coil pair as a dipole; it lowers the field by 1.1 %.

import numpy as np

from onesim.cad.d2 import Plane
from onesim.cad.d3 import Sphere
from onesim.materials import VacuumPermeability
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import CoulombGauge, MagneticInsulation, MagneticVectorPotential, Magnetostatics, SolenoidCurrent
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()

radius, width, turns, current, air_radius = 0.4, 0.05, 10, 0.25e-3, 2.0
mu0 = float(VacuumPermeability().evaluate())
centre_field = 0.8**1.5 * mu0 * turns * current / radius


def axis_field(z):
    """Axial field of the two loops inside the insulated air sphere."""
    loops = sum((radius**2 + (z - offset) ** 2) ** -1.5 for offset in (-radius / 2, radius / 2))
    return mu0 * turns * current * radius**2 * (loops / 2 - 1 / air_radius**3)


# ## Model
#
# The air sphere with the two windings as embedded regions, meshed finely around the coils.

air = Sphere(radius=air_radius)
coils = tuple(SolenoidCurrent(inner_radius=radius - width / 2, outer_radius=radius + width / 2, length=width, turns=turns,
                              current=current, center=(0.0, 0.0, offset)) for offset in (-radius / 2, radius / 2))
model = Magnetostatics(domain=air.volume, permeability=VacuumPermeability(),
                       formulation=MagneticVectorPotential(gauge=CoulombGauge()), current_sources=coils)
problem = Problem(name="Helmholtz coil", geometry=air, physics=(model,),
                  boundary_conditions=(MagneticInsulation(boundary=air.boundaries.all, physics=model),))
mesh = TetrahedralMesher(target_size=0.6, embedded_regions=tuple(coil.support for coil in coils), refinements=(
    *(LocalRefinement(region=coil.support, target_size=0.05) for coil in coils),
    LocalRefinement(region=Sphere(radius=0.55), target_size=0.14))).generate(air)
numerics = Numerics(spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=SparseDirect())
flux_density = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.magnetic_flux_density)


# ## Results
#
# The field on the axis against the closed form, and the size of the region in which the field
# stays within 0.1 % of its centre value.

z = np.linspace(0.0, 0.9, 46)
on_axis = flux_density.evaluate(np.column_stack((np.zeros_like(z), np.zeros_like(z), z)))[:, 2]
uniform = z[np.argmax(np.abs(on_axis / on_axis[0] - 1) > 1e-3)]
print(f"{mesh.cell_count} tetrahedra")
print(f"field at the centre: simulation {1e9 * on_axis[0]:.4f} nT, closed form {1e9 * axis_field(0.0):.4f} nT "
      f"({100 * (on_axis[0] / axis_field(0.0) - 1):+.2f} %); thin loops in free space {1e9 * centre_field:.4f} nT")
print(f"largest deviation on the axis up to z = 0.9 m: {100 * np.abs(on_axis - axis_field(z)).max() / centre_field:.2f} % of the centre field")
print(f"the axial field stays within 0.1 % of its centre value up to z = {uniform:.2f} m")


# ## Figure
#
# The field strength in a plane through the axis, and the axial field against the closed form.

strength = lambda B, position: np.linalg.norm(B, axis=1) / centre_field
fine = np.linspace(0.0, 0.9, 200)
fig = plots.figure(1, 2, size=(14, 6))
plots.section(flux_density, plane=Plane(origin=(0, 0, 0), normal=(0, 1, 0)), component=strength, limits=(0, 2.2),
              contours=11, subdivisions=2, label="|B| / centre field of the thin loops", title="Field strength",
              xlimits=(-0.9, 0.9), ylimits=(-0.75, 0.75), xlabel="x [m]", ylabel="z [m]", ax=fig[0])
plots.line(points={"simulation": (z, on_axis / centre_field)}, reference={"closed form": (fine, axis_field(fine) / centre_field)},
           vlines={"coil plane": radius / 2}, xlabel="z on the axis [m]", ylabel="B_z / centre field of the thin loops",
           title="Axial field", ax=fig[1])
plots.show()


# ## Going further
#
# Move the coils further apart or closer together and the plateau of the axial field turns into a
# dip or a peak. A third coil pair (Maxwell coil) widens the uniform region further, and opposite
# currents give the linear gradient field of an anti-Helmholtz pair.
