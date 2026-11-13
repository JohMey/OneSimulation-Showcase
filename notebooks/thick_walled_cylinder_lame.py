"""OneSimulation example: thick-walled cylinder under internal pressure.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form Lame solution.
"""


# # Thick-walled cylinder under internal pressure
#
# A hollow cylinder (inner radius $a$, outer radius $b$, length $\ell$) carries a
# uniform internal pressure $p$ on its bore; the outer wall is free. Both end
# faces are clamped, so a long cylinder reaches a plane-strain state
# ($u_z=0$) in its central cross section, away from the end boundary layers.
#
# OneSimulation solves the stationary 3D equilibrium with an isotropic Hooke law
# on curved (Q2 geometry) annular hexahedra.
#
# **Reference.** The Lame solution for internal pressure only is
#
# $$\sigma_r(r)=\frac{a^2 p}{b^2-a^2}\Bigl(1-\frac{b^2}{r^2}\Bigr),\qquad
#   \sigma_\theta(r)=\frac{a^2 p}{b^2-a^2}\Bigl(1+\frac{b^2}{r^2}\Bigr),$$
#
# and, in plane strain, the radial displacement
#
# $$u_r(r)=\frac{(1+\nu)\,a^2 p}{E\,(b^2-a^2)}
#          \Bigl[(1-2\nu)\,r+\frac{b^2}{r}\Bigr].$$

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import AnnularCylinder
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredAnnularHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

a, b, length = 0.050, 0.100, 0.500      # m
E, nu = 200e9, 0.30                     # Pa, -
pressure = 50e6                         # Pa, internal pressure

# Closed-form Lame fields.
def sigma_r_ref(r):
    return a**2 * pressure / (b**2 - a**2) * (1.0 - b**2 / r**2)

def sigma_theta_ref(r):
    return a**2 * pressure / (b**2 - a**2) * (1.0 + b**2 / r**2)

def u_r_ref(r):
    return (1.0 + nu) * a**2 * pressure / (E * (b**2 - a**2)) * ((1.0 - 2.0 * nu) * r + b**2 / r)

# Internal pressure as an outward radial traction p * e_r on the bore.
def inner_pressure(x, y, z):
    r = np.sqrt(x**2 + y**2)
    return pressure * np.stack((x / r, y / r, np.zeros_like(x)), axis=-1)

geometry = AnnularCylinder(inner_radius=a, outer_radius=b, length=length)
material = IsotropicSolid(
    density=7850.0, specific_heat_capacity=460.0, thermal_conductivity=50.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)
problem = Problem(
    name="Thick-walled cylinder under internal pressure",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        Traction(boundary=geometry.boundaries.inner_wall, physics=model, value=inner_pressure),
        TractionFree(boundary=geometry.boundaries.outer_wall, physics=model),
        FixedDisplacement(boundary=geometry.boundaries.bottom, physics=model, value=(0.0, 0.0, 0.0)),
        FixedDisplacement(boundary=geometry.boundaries.top, physics=model, value=(0.0, 0.0, 0.0)),
    ),
)
mesh = StructuredAnnularHexahedralMesher(
    radial_cells=4, azimuthal_cells=16, axial_cells=8, geometry_order=2
).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with the Lame solution (mid-height cross section)

displacement = result.field(model.displacement)
stress = result.field(model.stress)

# Sample along the +x axis at mid-height; there e_r = e_x, so
# sigma_xx = sigma_r and sigma_yy = sigma_theta.
radii = np.linspace(a + 0.08 * (b - a), b - 0.08 * (b - a), 9)
points = np.column_stack((radii, np.zeros_like(radii), np.full_like(radii, length / 2)))

s = np.asarray(stress.evaluate(points))
sigma_r_fem, sigma_theta_fem = s[:, 0], s[:, 1]
u_r_fem = np.asarray(displacement.evaluate(points))[:, 0]

err_sr = np.max(np.abs(sigma_r_fem - sigma_r_ref(radii))) / pressure
err_st = np.max(np.abs(sigma_theta_fem - sigma_theta_ref(radii))) / pressure
err_ur = np.max(np.abs(u_r_fem - u_r_ref(radii))) / np.max(np.abs(u_r_ref(radii)))

print(f"max |sigma_r - Lame|   / p = {100*err_sr:.3f} %")
print(f"max |sigma_theta-Lame| / p = {100*err_st:.3f} %")
print(f"max |u_r - Lame| / max|u_r| = {100*err_ur:.3f} %")
print(f"sigma_r at bore: FEM {sigma_r_fem[0]/1e6:.2f} MPa (Lame {sigma_r_ref(radii[0])/1e6:.2f} MPa)")

assert err_sr < 3e-2, "Radial stress deviates more than 3 % of p from the Lame solution."
assert err_st < 3e-2, "Hoop stress deviates more than 3 % of p from the Lame solution."
assert err_ur < 3e-2, "Radial displacement deviates more than 3 % from the Lame solution."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
rr = np.linspace(a, b, 100)
ax.plot(rr * 1000, sigma_theta_ref(rr) / 1e6, color="#214D43", lw=1.8, label="Lame  sigma_theta")
ax.plot(rr * 1000, sigma_r_ref(rr) / 1e6, color="#2f6f5e", lw=1.8, ls="--", label="Lame  sigma_r")
ax.plot(radii * 1000, sigma_theta_fem / 1e6, "o", color="#A85539", ms=5, label="FEM  sigma_theta")
ax.plot(radii * 1000, sigma_r_fem / 1e6, "s", color="#c77b4a", ms=5, label="FEM  sigma_r")
ax.set(xlabel="Radius r / mm", ylabel="Stress / MPa",
       title="Thick-walled cylinder: stresses at mid-height")
ax.legend()
plt.show()


# ## Going further
#
# Increase the length-to-thickness ratio and the axial cell count to shrink the
# clamped-end boundary layer, so the mid-height state matches plane strain even
# more closely. Add an external pressure term to recover the full Lame solution
# with pressure on both walls.
