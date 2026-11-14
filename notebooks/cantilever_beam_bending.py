"""OneSimulation example: bending of a clamped beam (linear elasticity).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against Euler-Bernoulli beam theory.
"""


# # Bending of a clamped beam
#
# A beam (length $L$, width $b$, height $h$) fully clamped on the left carries a
# uniformly distributed transverse load with total force $F$ pointing down on the right.
#
# OneSimulation solves the stationary 3D equilibrium
# $-\nabla\!\cdot\boldsymbol\sigma=0$ with an isotropic Hooke law (Q2 hexahedra) and
# computes displacement and stress.
#
# **Reference.** Euler-Bernoulli beam theory gives the tip deflection
#
# $$u_z(L)=-\frac{F L^3}{3EI},\qquad I=\frac{b h^3}{12}.$$
#
# It neglects shear and 3D clamping effects and is therefore an approximate target value.

import numpy as np

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()


# ## Parameters and model

length, width, height = 0.120, 0.018, 0.012
E, nu = 70e9, 0.30
force = 30.0
inertia = width * height**3 / 12.0
tip_reference = -force * length**3 / (3.0 * E * inertia)

geometry = Box(length=length, width=width, height=height)
material = IsotropicSolid(
    density=2700.0, specific_heat_capacity=900.0, thermal_conductivity=200.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=23e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)
problem = Problem(
    name="Clamped beam under end-face load",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=geometry.boundaries.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        Traction(boundary=geometry.boundaries.x_max, physics=model,
                 value=(0.0, 0.0, -force / (width * height))),
        TractionFree(boundary=geometry.boundaries.all - geometry.boundaries.x_min - geometry.boundaries.x_max,
                     physics=model),
    ),
)
mesh = StructuredHexahedralMesher(cells=(24, 4, 4)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with beam theory

displacement = result.field(model.displacement)
tip = float(displacement.evaluate([[length, width / 2, height / 2]])[0, 2])
rel_error = abs(tip - tip_reference) / abs(tip_reference)
print(f"Tip deflection: FEM {tip*1e3:.4f} mm,  Euler-Bernoulli {tip_reference*1e3:.4f} mm")
print(f"relative deviation: {100*rel_error:.2f} %  (3D FEM is slightly lower due to shear/clamping)")

xs = np.linspace(0.0, length, 25)
line = np.column_stack((xs, np.full_like(xs, width / 2), np.full_like(xs, height / 2)))
uz = np.asarray(displacement.evaluate(line))[:, 2]
uz_ref = -force * (3 * length * xs**2 - xs**3) / (6 * E * inertia)


# ## Figure

plots.line(xs * 1000, points={"OneSimulation 3D FEM": uz * 1e3}, reference={"Euler-Bernoulli": uz_ref * 1e3},
           xlabel="position along the beam x [mm]", ylabel="deflection u_z [mm]", title="Clamped beam: deflection line")
plots.show()


# ## Going further
#
# Refine the mesh (more cells through the height) and watch the FEM deflection
# approach the Euler-Bernoulli value, or slightly exceed it because of shear.
# Change the L/h ratio; for stubby beams the shear contribution grows and beam
# theory becomes less accurate.
