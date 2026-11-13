"""OneSimulation example: stresses in a rotating solid shaft.
Requires the installed, activated OneSimulation runtime.

The 3D elasticity solution is checked against the classical plane-strain
closed form for a long solid cylinder spinning at constant angular velocity.
"""


# # Rotating solid shaft
#
# A long solid shaft of radius $a$ spins at a constant angular velocity
# $\omega$ about its axis. In the co-rotating frame the inertia of the material
# appears as the centrifugal body force
#
# $$\mathbf f=\rho\,\omega^2\,r\,\hat{\mathbf e}_r=\rho\,\omega^2\,(x,\,y,\,0),$$
#
# while the lateral surface is free. Both end faces are held, so the central
# part of a long shaft is in a plane-strain state ($u_z=0$); the clamped-end
# boundary layers decay over a distance of about one radius.
#
# **Reference.** For a long cylinder in plane strain the radial and hoop
# stresses are
#
# $$\sigma_r(r)=\frac{3-2\nu}{8\,(1-\nu)}\,\rho\,\omega^2\,(a^2-r^2),\qquad
#   \sigma_\theta(r)=\frac{\rho\,\omega^2}{8\,(1-\nu)}\,
#     \bigl[(3-2\nu)\,a^2-(1+2\nu)\,r^2\bigr],$$
#
# which coincide on the axis and give $\sigma_r(a)=0$ at the free surface. The
# axial stress follows from the plane-strain constraint,
# $\sigma_z=\nu\,(\sigma_r+\sigma_\theta)$.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Cylinder
from onesim.materials import IsotropicSolid
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import BodyForce, FixedDisplacement, LinearElasticity, TractionFree
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and reference values

a = 0.050              # m, shaft radius
length = 0.300         # m, shaft length (six radii, clamped at both ends)
E, nu = 200e9, 0.30    # Pa, steel-like
rho = 7800.0           # kg/m^3
omega = 1000.0         # rad/s (about 9550 rpm)


def sigma_r_ref(r):
    return (3.0 - 2.0 * nu) / (8.0 * (1.0 - nu)) * rho * omega**2 * (a**2 - r**2)


def sigma_theta_ref(r):
    return rho * omega**2 / (8.0 * (1.0 - nu)) * ((3.0 - 2.0 * nu) * a**2 - (1.0 + 2.0 * nu) * r**2)


def sigma_z_ref(r):
    return nu * (sigma_r_ref(r) + sigma_theta_ref(r))


sigma_axis = sigma_r_ref(0.0)
print(f"rho omega^2 a^2 = {rho*omega**2*a**2/1e6:.2f} MPa,  "
      f"stress on the axis = {sigma_axis/1e6:.2f} MPa,  "
      f"hoop stress at the surface = {sigma_theta_ref(a)/1e6:.2f} MPa")


# ## Geometry, model and boundary conditions
#
# The shaft axis is the z axis. The centrifugal load enters as a body force
# evaluated at the quadrature points; the lateral surface carries no traction
# and both end faces are held fixed.

geometry = Cylinder(radius=a, length=length)
material = IsotropicSolid(
    density=rho, specific_heat_capacity=460.0, thermal_conductivity=50.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6,
)


def centrifugal(x, y, z):
    # f = rho omega^2 (x, y, 0): one vector (fx, fy, fz) per point.
    return rho * omega**2 * np.stack((x, y, np.zeros_like(x)), axis=-1)


model = LinearElasticity(domain=geometry.volume, material=material,
                         loading=BodyForce(value=centrifugal))

problem = Problem(
    name="Rotating solid shaft",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=geometry.boundaries.bottom, physics=model, value=(0.0, 0.0, 0.0)),
        FixedDisplacement(boundary=geometry.boundaries.top, physics=model, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=geometry.boundaries.outer_wall, physics=model),
    ),
)
# Curved quadratic tetrahedra, a few elements across the radius.
mesh = TetrahedralMesher(target_size=a / 3.0, geometry_order=2).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)),
                      linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with the plane-strain solution (mid-length cross section)
#
# Along the +x axis the radial direction is e_x, so sigma_xx = sigma_r and
# sigma_yy = sigma_theta. The stress components are ordered
# (xx, yy, zz, xy, yz, xz).

stress = result.field(model.stress)
radii = np.linspace(0.0, 0.9 * a, 10)
points = np.column_stack((radii, np.zeros_like(radii), np.full_like(radii, length / 2)))
s = np.asarray(stress.evaluate(points))
sigma_r_fem, sigma_theta_fem, sigma_z_fem = s[:, 0], s[:, 1], s[:, 2]

err_sr = float(np.max(np.abs(sigma_r_fem - sigma_r_ref(radii))) / sigma_axis)
err_st = float(np.max(np.abs(sigma_theta_fem - sigma_theta_ref(radii))) / sigma_axis)
err_sz = float(np.max(np.abs(sigma_z_fem - sigma_z_ref(radii))) / sigma_axis)
print("Stresses at mid-length:")
for r, sr, st in zip(radii, sigma_r_fem, sigma_theta_fem):
    print(f"  r={r*1e3:5.1f} mm  sigma_r FEM {sr/1e6:5.2f} MPa (theory {sigma_r_ref(r)/1e6:5.2f})"
          f"  sigma_theta FEM {st/1e6:5.2f} MPa (theory {sigma_theta_ref(r)/1e6:5.2f})")
print(f"max |sigma_r - theory| / sigma_axis     = {100*err_sr:.2f} %")
print(f"max |sigma_theta - theory| / sigma_axis = {100*err_st:.2f} %")
print(f"max |sigma_z - theory| / sigma_axis     = {100*err_sz:.2f} %")
assert err_sr < 0.05, "Radial stress deviates more than the example tolerance (5 % of the axis stress)."
assert err_st < 0.05, "Hoop stress deviates more than the example tolerance (5 % of the axis stress)."
assert err_sz < 0.05, "Axial stress deviates more than the example tolerance (5 % of the axis stress)."

# The radial displacement must grow outward under the centrifugal load.
displacement = result.field(model.displacement)
u_r_fem = np.asarray(displacement.evaluate(points))[:, 0]
assert u_r_fem[-1] > 0, "The surface must move outward under the centrifugal load."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
rr = np.linspace(0.0, a, 200)
ax.plot(rr * 1000, sigma_theta_ref(rr) / 1e6, color="#214D43", lw=1.8, label="plane strain  sigma_theta")
ax.plot(rr * 1000, sigma_r_ref(rr) / 1e6, color="#2f6f5e", lw=1.8, ls="--", label="plane strain  sigma_r")
ax.plot(rr * 1000, sigma_z_ref(rr) / 1e6, color="#6b8f84", lw=1.8, ls=":", label="plane strain  sigma_z")
ax.plot(radii * 1000, sigma_theta_fem / 1e6, "o", color="#A85539", ms=5, label="FEM  sigma_theta")
ax.plot(radii * 1000, sigma_r_fem / 1e6, "s", color="#c77b4a", ms=5, label="FEM  sigma_r")
ax.plot(radii * 1000, sigma_z_fem / 1e6, "^", color="#d9a066", ms=5, label="FEM  sigma_z")
ax.set(xlabel="Radius r / mm", ylabel="Stress / MPa",
       title="Rotating solid shaft: stresses at mid-length")
ax.legend()
plt.show()


# ## Going further
#
# Bore a central hole (AnnularCylinder) and compare against the rotating
# hollow cylinder, whose hoop stress at the bore doubles relative to the solid
# shaft as the hole shrinks. Shorten the shaft to see the clamped ends reach
# the mid-section and the state drift away from plane strain.
