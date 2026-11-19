"""OneSimulation example: bending of a clamped beam (linear elasticity).
Requires the installed, activated OneSimulation runtime.

The beam is read from a STEP file, meshed with hexahedra and solved for the displacement; the
deflection is checked against Euler-Bernoulli beam theory.
"""

# # Bending of a clamped beam
#
# A beam (length $L$, width $b$, height $h$) fully clamped on the left carries a
# uniformly distributed transverse load with total force $F$ pointing down on the right.
#
# The beam comes from the CAD file `cantilever_beam_bending.step`; its faces are named `x_min`,
# `x_max`, `y_min`, `y_max`, `z_min` and `z_max`. The script `cantilever_beam_bending_geometry.py`
# builds it with the CAD objects of OneSimulation. OneSimulation solves the stationary 3D
# equilibrium $-\nabla\!\cdot\boldsymbol\sigma=0$ with an isotropic Hooke law on quadratic
# hexahedra.
#
# **Reference.** Euler-Bernoulli beam theory gives the deflection line
#
# $$u_z(x)=-\frac{F\,(3Lx^2-x^3)}{6EI},\qquad I=\frac{b h^3}{12}.$$
#
# It neglects shear and 3D clamping effects and is therefore an approximate target value.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import CadHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacement,
    LinearElasticity,
    Traction,
    TractionFree,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh

beam = load("cantilever_beam_bending.step")
faces = beam.boundaries
length, width, height = beam.size
mesh = CadHexahedralMesher(cells=(24, 4, 4)).generate(beam)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(
    beam,
    selection={"clamped": faces.x_min, "loaded": faces.x_max},
    labels=False,
    title="Beam with its clamped and loaded face",
    ax=fig[0],
)
plots.mesh(
    mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} hexahedra", ax=fig[1]
)
plots.show()

# ## Model

force, youngs_modulus = 30.0, 70e9
aluminium = IsotropicSolid(
    density=2700.0,
    specific_heat_capacity=900.0,
    thermal_conductivity=200.0,
    youngs_modulus=youngs_modulus,
    poisson_ratio=0.30,
    thermal_expansion_coefficient=23e-6,
)
model = LinearElasticity(domain=beam.volume, material=aluminium)
problem = Problem(
    name="Clamped beam under end-face load",
    geometry=beam,
    physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=faces.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        Traction(
            boundary=faces.x_max,
            physics=model,
            value=(0.0, 0.0, -force / (width * height)),
        ),
        TractionFree(boundary=faces.all - faces.x_min - faces.x_max, physics=model),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
displacement = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(model.displacement)
)

# ## Comparison with beam theory

x = np.linspace(0.0, length, 25)
deflection = displacement.evaluate(
    np.column_stack((x, np.full_like(x, width / 2), np.full_like(x, height / 2)))
)[:, 2]
beam_theory = (
    -force * (3 * length * x**2 - x**3) / (6 * youngs_modulus * width * height**3 / 12)
)

print(
    f"tip deflection: simulation {1e3 * deflection[-1]:.4f} mm, Euler-Bernoulli {1e3 * beam_theory[-1]:.4f} mm, "
    f"deviation {100 * (deflection[-1] / beam_theory[-1] - 1):+.2f} %"
)

# ## Figure

plots.line(
    1e3 * x,
    points={"simulation": 1e3 * deflection},
    reference={"Euler-Bernoulli": 1e3 * beam_theory},
    xlabel="position along the beam x [mm]",
    ylabel="deflection u_z [mm]",
    title="Clamped beam: deflection line",
)
plots.show()

# ## Going further
#
# Change the cross-section in `cantilever_beam_bending_geometry.py` and write the STEP file again;
# for stubby beams the shear contribution grows and beam theory becomes less accurate. More cells
# through the height bring the 3D result closer to its own limit.
