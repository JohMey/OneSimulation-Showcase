"""OneSimulation example: interior field of a long solenoid (magnetostatics).
Requires the installed, activated OneSimulation runtime.

The air cylinder around the coil is read from a STEP file. The 3D FEM solution is checked against
the ideal infinite-solenoid interior value.
"""

# # Interior field of a long solenoid
#
# A single homogenised solenoid (mean radius $\approx 12.5$ mm, length $200$ mm, so
# length $\gg$ radius) carries $1\,\mathrm{A}$ through $400$ turns. Its current is
# modelled as an azimuthal volume current density on one explicitly meshed winding
# region. OneSimulation solves
# $\nabla\times(\mu^{-1}\nabla\times\mathbf A)=\mathbf J$ with a weak Coulomb gauge
# and reconstructs $\mathbf B=\nabla\times\mathbf A$.
#
# The air cylinder comes from the CAD file `long_solenoid_interior_field.step`; the script
# `long_solenoid_interior_field_geometry.py` builds it with the CAD objects of OneSimulation. The
# winding is declared as a current source, and the mesher embeds its volume in the air.
#
# **Reference.** Deep inside an infinitely long solenoid the field is axial and uniform,
#
# $$B=\mu_0\,n\,I,\qquad n=\frac{\text{turns}}{\text{length}}.$$
#
# The coil here is long but finite, so the formula is approached from below (the ends
# let a little flux escape).

import numpy as np
from onesim.cad.d3 import AnnularCylinder
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    CoulombGauge,
    MagneticInsulation,
    MagneticVectorPotential,
    Magnetostatics,
    SolenoidCurrent,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The winding sits in the middle of the air cylinder and is meshed more finely than the air.

air = load("long_solenoid_interior_field.step")
centre = air.size[2] / 2
coil_length, turns, current = 0.20, 400, 1.0
coil = SolenoidCurrent(
    inner_radius=0.010,
    outer_radius=0.015,
    length=coil_length,
    center=(0.0, 0.0, centre),
    turns=turns,
    current=current,
    support_factory=AnnularCylinder,
)
mesh = TetrahedralMesher(
    target_size=0.02,
    embedded_regions=(coil.support,),
    refinements=(LocalRefinement(region=coil.support, target_size=0.006),),
).generate(air)

fig = plots.figure(1, 2, size=(11, 6))
plots.geometry([air, coil.support], title="Air cylinder with the winding", ax=fig[0])
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"{mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model

model = Magnetostatics(
    domain=air.volume,
    permeability=VacuumPermeability(),
    formulation=MagneticVectorPotential(gauge=CoulombGauge()),
    current_sources=(coil,),
)
problem = Problem(
    name="Long solenoid interior field",
    geometry=air,
    physics=(model,),
    boundary_conditions=(
        MagneticInsulation(boundary=air.boundaries.all, physics=model),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=1)), linear_solver=SparseDirect()
)
flux_density = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(model.magnetic_flux_density)
)

# ## Centre field and axial uniformity
#
# The axial field is sampled along the axis over the central half of the coil, where an infinite
# solenoid would be perfectly uniform.

ideal = VacuumPermeability().evaluate() * turns / coil_length * current
z = np.linspace(centre - coil_length / 4, centre + coil_length / 4, 21)
axial = flux_density.evaluate(np.column_stack((np.zeros_like(z), np.zeros_like(z), z)))[
    :, 2
]

print(
    f"centre axial field: simulation {1e3 * axial[10]:.4f} mT, ideal mu0 n I {1e3 * ideal:.4f} mT, "
    f"deviation {100 * (axial[10] / ideal - 1):+.2f} % (finite length lowers the field)"
)
print(
    f"axial non-uniformity over the central half: {100 * (axial.max() - axial.min()) / axial.mean():.2f} %"
)

# ## Figure

plots.line(
    1e3 * (z - centre),
    {"simulation": 1e3 * axial},
    hlines={"ideal mu_0 n I": 1e3 * ideal},
    markers=True,
    xlabel="axial position from centre z [mm]",
    ylabel="axial flux density B_z [mT]",
    title="Long solenoid: interior axial field",
)
plots.show()

# ## Going further
#
# Lengthen the coil and watch the centre field climb towards $\mu_0 n I$ as the end effects fade.
# Enlarge the air cylinder in `long_solenoid_interior_field_geometry.py` and write the STEP file
# again: the magnetic insulation then sits farther from the winding, with the same effect.
