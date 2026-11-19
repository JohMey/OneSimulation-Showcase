"""OneSimulation example: the on-axis field of a finite solenoid.
Requires the installed, activated OneSimulation runtime.

The air cylinder around the coil is read from a STEP file. The computed axial flux density is
checked against the finite-solenoid formula.
"""

# # Magnetostatics: the on-axis field of a finite solenoid
#
# A single short, thin-walled solenoid carries a steady current. OneSimulation
# solves the full 3-D magnetostatic problem
#
# $$\nabla\times\big(\mu^{-1}\nabla\times\mathbf A\big)=\mathbf J,\qquad
#   \mathbf B=\nabla\times\mathbf A,$$
#
# with first-order Nedelec (edge) elements and a weak Coulomb gauge, and
# reconstructs the magnetic flux density $\mathbf B$ on a tetrahedral mesh.
#
# The air cylinder comes from the CAD file `solenoid_on_axis_field.step`; the script
# `solenoid_on_axis_field_geometry.py` builds it with the CAD objects of OneSimulation. The winding
# is declared as a current source, and the mesher embeds its volume in the air. The outer surface
# carries a magnetic-insulation condition; the air cylinder is several coil radii wide and long so
# that this boundary barely affects the central field.
#
# **Reference.** On the axis of a finite solenoid of radius $R$, length $L$, with
# $N$ turns carrying current $I$, the axial flux density has the closed form
#
# $$B_z(z)=\frac{\mu_0 N I}{2L}
#   \left[\frac{z-z_a}{\sqrt{(z-z_a)^2+R^2}}
#        -\frac{z-z_b}{\sqrt{(z-z_b)^2+R^2}}\right],$$
#
# where $z_a,z_b$ are the axial positions of the two coil ends. The winding is thin, so the
# formula is accurate to well under a percent.

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
# The coil is a thin cylindrical shell at the centre of the air cylinder, meshed more finely than
# the air.

air = load("solenoid_on_axis_field.step")
centre = air.size[2] / 2
inner_radius, outer_radius, coil_length, turns, current = 0.0235, 0.0265, 0.03, 200, 1.0
coil = SolenoidCurrent(
    inner_radius=inner_radius,
    outer_radius=outer_radius,
    length=coil_length,
    center=(0.0, 0.0, centre),
    turns=turns,
    current=current,
    support_factory=AnnularCylinder,
)
mesh = TetrahedralMesher(
    target_size=0.032,
    embedded_regions=(coil.support,),
    refinements=(LocalRefinement(region=coil.support, target_size=0.0045),),
).generate(air)

fig = plots.figure(1, 2, size=(13, 6))
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
    name="On-axis field of a finite solenoid",
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

# ## Comparison with the finite-solenoid formula
#
# First-order edge elements give a flux density that is constant per tetrahedron. To read off the
# smooth bore field, $B_z$ is averaged at each axial station over the axis point and twelve points
# on a coaxial circle of radius 6 mm, well inside the bore.

z = centre + np.linspace(-0.045, 0.045, 31)
angles = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
ring = np.column_stack(
    (np.append(0.0, 0.006 * np.cos(angles)), np.append(0.0, 0.006 * np.sin(angles)))
)
stations = np.column_stack((np.tile(ring, (len(z), 1)), np.repeat(z, len(ring))))
axial = flux_density.evaluate(stations)[:, 2].reshape(len(z), len(ring)).mean(axis=1)
radius, ends = (inner_radius + outer_radius) / 2, (
    centre - coil_length / 2,
    centre + coil_length / 2,
)
formula = (
    VacuumPermeability().evaluate()
    * turns
    * current
    / (2 * coil_length)
    * (
        (z - ends[0]) / np.hypot(z - ends[0], radius)
        - (z - ends[1]) / np.hypot(z - ends[1], radius)
    )
)
inside = np.abs(z - centre) <= coil_length / 2 + 1e-12

print(
    f"centre B_z: simulation {1e6 * axial[15]:.2f} µT, formula {1e6 * formula[15]:.2f} µT, "
    f"deviation {100 * (axial[15] / formula[15] - 1):+.2f} %"
)
print(
    f"largest deviation along the axis inside the coil: {100 * np.max(np.abs(axial[inside] / formula[inside] - 1)):.2f} %"
)

# ## Figure

plots.line(
    1e3 * (z - centre),
    points={"simulation": 1e6 * axial},
    reference={"finite-solenoid formula": 1e6 * formula},
    vlines={"coil ends": -1e3 * coil_length / 2, "": 1e3 * coil_length / 2},
    xlabel="axial position z from the coil centre [mm]",
    ylabel="axial flux density B_z [µT]",
    title="Finite solenoid: on-axis field",
)
plots.show()

# ## Going further
#
# Refine the mesh (smaller `target_size`), or widen the air cylinder in
# `solenoid_on_axis_field_geometry.py` and write the STEP file again: the finite domain and the
# magnetic insulation set the small residual offset from the free-space formula. Make the coil
# longer relative to its radius and the bore field flattens towards the ideal $\mu_0 n I$.
