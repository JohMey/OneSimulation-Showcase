"""OneSimulation example: fundamental bending frequency of a clamped beam (modal analysis).
Requires the installed, activated OneSimulation runtime.

The OneSimulation modal solution is checked against Euler-Bernoulli beam theory.
"""


# # Natural frequency of a clamped beam
#
# A prismatic beam (length $L$, square cross-section $b\times h$) is fully clamped on
# the left and free on every other face. OneSimulation assembles the 3D linear-elastic
# stiffness and mass operators (Q2 hexahedra) and solves the generalised eigenproblem
# $K\boldsymbol\phi=\omega^2 M\boldsymbol\phi$ on the free degrees of freedom to obtain
# the natural modes and frequencies.
#
# **Reference.** Euler-Bernoulli beam theory gives the bending natural frequencies
#
# $$f_n=\frac{\beta_n^2}{2\pi}\sqrt{\frac{E I}{\rho A L^4}},\qquad
#   I=\frac{b h^3}{12},\quad A=b h,$$
#
# with $\beta_1=1.875104$ for the first clamped-free mode. Beam theory neglects shear
# deformation and rotary inertia and is therefore an approximate target value for a
# three-dimensional solid of finite slenderness.

from math import pi, sqrt
import numpy as np

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, TractionFree, structural_modes
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()


# ## Parameters and model

length, side = 0.120, 0.012
E, nu, density = 210e9, 0.30, 7850.0
inertia = side * side**3 / 12.0
area = side * side
beta_1 = 1.875104
frequency_reference = beta_1**2 / (2.0 * pi) * sqrt(E * inertia / (density * area * length**4))

geometry = Box(length=length, width=side, height=side)
material = IsotropicSolid(
    density=density, specific_heat_capacity=460.0, thermal_conductivity=50.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)
problem = Problem(
    name="Clamped beam natural modes",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=geometry.boundaries.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=geometry.boundaries.all - geometry.boundaries.x_min, physics=model),
    ),
)
mesh = StructuredHexahedralMesher(cells=(24, 3, 3)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Modal analysis and comparison with beam theory
#
# The square cross-section makes bending about the two transverse axes degenerate, so
# the first two computed frequencies form one bending pair. The fundamental is the
# lowest of them.

modes = structural_modes(result, model, mode_count=4)
fundamental = modes.frequencies[0]
rel_error = abs(fundamental - frequency_reference) / frequency_reference
print(f"Fundamental frequency: modal {fundamental:.2f} Hz,  Euler-Bernoulli {frequency_reference:.2f} Hz")
print(f"relative deviation: {100*rel_error:.2f} %  (3D solid is slightly stiffer than beam theory)")
print("First four frequencies / Hz:", [round(f, 2) for f in modes.frequencies])


# ## Figure

plots.bars(["Mode 1\n(bending)", "Mode 2\n(bending)", "Mode 3", "Mode 4"], {"OneSimulation": modes.frequencies},
           reference={"Euler-Bernoulli": [frequency_reference, frequency_reference, np.nan, np.nan]},
           ylabel="natural frequency [Hz]", title="Clamped beam: natural frequencies")
plots.show()


# ## Going further
#
# Refine the mesh and watch the computed fundamental frequency descend towards the
# Euler-Bernoulli value from above, as the discrete model becomes less stiff. Change
# the slenderness L/h; for stubby beams shear and rotary inertia grow and beam theory
# increasingly overestimates the higher bending frequencies.
