"""OneSimulation example: fundamental bending frequency of a clamped beam (modal analysis).
Requires the installed, activated OneSimulation runtime.

The beam is read from a STEP file and meshed with hexahedra; the natural frequencies from the
modal analysis are checked against Euler-Bernoulli beam theory.
"""

# # Natural frequency of a clamped beam
#
# A prismatic beam (length $L$, square cross-section $b\times h$) is fully clamped on
# the left and free on every other face. It comes from the CAD file
# `cantilever_natural_frequency.step` with the face names `x_min` to `z_max`; the script
# `cantilever_natural_frequency_geometry.py` builds it with the CAD objects of OneSimulation.
# OneSimulation assembles the 3D linear-elastic stiffness and mass operators on quadratic
# hexahedra and solves the generalised eigenproblem
# $K\boldsymbol\phi=\omega^2 M\boldsymbol\phi$ for the natural modes and frequencies.
#
# **Reference.** Euler-Bernoulli beam theory gives the bending natural frequencies
#
# $$f_n=\frac{\beta_n^2}{2\pi}\sqrt{\frac{E I}{\rho A L^4}},\qquad
#   I=\frac{b h^3}{12},\quad A=b h,$$
#
# with $\beta_1=1.875104$ for the first clamped-free mode. Beam theory neglects shear
# deformation and rotary inertia and is therefore an approximate target value for a
# three-dimensional solid of finite slenderness.

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
    TractionFree,
    structural_modes,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh

beam = load("cantilever_natural_frequency.step")
faces = beam.boundaries
length, side, _ = beam.size
mesh = CadHexahedralMesher(cells=(24, 3, 3)).generate(beam)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(
    beam,
    selection={"clamped": faces.x_min},
    labels=False,
    title="Beam with its clamped face",
    ax=fig[0],
)
plots.mesh(
    mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} hexahedra", ax=fig[1]
)
plots.show()

# ## Model

youngs_modulus, density = 210e9, 7850.0
steel = IsotropicSolid(
    density=density,
    specific_heat_capacity=460.0,
    thermal_conductivity=50.0,
    youngs_modulus=youngs_modulus,
    poisson_ratio=0.30,
    thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=beam.volume, material=steel)
problem = Problem(
    name="Clamped beam natural modes",
    geometry=beam,
    physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=faces.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=faces.all - faces.x_min, physics=model),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()

# ## Modal analysis and comparison with beam theory
#
# The square cross-section makes bending about the two transverse axes degenerate, so
# the first two computed frequencies form one bending pair.

modes = structural_modes(result, model, mode_count=4)
beam_theory = (
    1.875104**2
    / (2 * np.pi)
    * np.sqrt(youngs_modulus * side**4 / 12 / (density * side**2 * length**4))
)

print(
    f"fundamental frequency: simulation {modes.frequencies[0]:.2f} Hz, Euler-Bernoulli {beam_theory:.2f} Hz, "
    f"deviation {100 * (modes.frequencies[0] / beam_theory - 1):+.2f} %"
)
print("first four frequencies [Hz]:", [round(f, 2) for f in modes.frequencies])

# ## Figure

plots.bars(
    ["Mode 1\n(bending)", "Mode 2\n(bending)", "Mode 3", "Mode 4"],
    {"simulation": modes.frequencies},
    reference={"Euler-Bernoulli": [beam_theory, beam_theory, np.nan, np.nan]},
    ylabel="natural frequency [Hz]",
    title="Clamped beam: natural frequencies",
)
plots.show()

# ## Going further
#
# Change the slenderness in `cantilever_natural_frequency_geometry.py` and write the STEP file
# again; for stubby beams shear and rotary inertia grow and beam theory increasingly
# overestimates the higher bending frequencies.
