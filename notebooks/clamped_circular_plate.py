"""OneSimulation example: clamped circular plate under uniform pressure.
Requires the installed, activated OneSimulation runtime.

The 3D elasticity solution is checked against the Kirchhoff thin-plate
closed form for a clamped circular plate.
"""


# # Clamped circular plate under uniform pressure
#
# A thin circular plate of radius $a$ and thickness $h\ll a$ is clamped along
# its outer edge and carries a uniform pressure $p$ on its top face. The bottom
# face is free. OneSimulation solves the full 3D equilibrium with an isotropic
# Hooke law on curved (quadratic) tetrahedra; no plate kinematics are assumed.
#
# **Reference.** Kirchhoff thin-plate theory gives the deflection surface
#
# $$w(r)=\frac{p\,a^4}{64\,D}\Bigl(1-\frac{r^2}{a^2}\Bigr)^2,\qquad
#   D=\frac{E\,h^3}{12\,(1-\nu^2)},$$
#
# with the centre deflection $w_0=p a^4/(64 D)$ and the clamped-edge radial
# bending stress $\sigma_r(a)=\pm\frac{3}{4}\,p\,(a/h)^2$ on the plate faces.
#
# With $a/h=20$ the 3D model differs from thin-plate theory by a small
# transverse-shear contribution (about one percent for this ratio), while the
# fully clamped lateral face is marginally stiffer than the ideal Kirchhoff
# clamp; both effects are well inside the tolerance used below.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Cylinder
from onesim.materials import IsotropicSolid
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and reference values

a = 0.100            # m, plate radius
h = 0.005            # m, plate thickness (a/h = 20)
E, nu = 200e9, 0.30  # Pa, steel-like
pressure = 1.0e5     # Pa, uniform pressure on the top face

D = E * h**3 / (12.0 * (1.0 - nu**2))   # N·m, flexural rigidity
w0_reference = pressure * a**4 / (64.0 * D)


def w_reference(r):
    return w0_reference * (1.0 - (r / a) ** 2) ** 2


print(f"D = {D:.2f} N·m,  reference centre deflection w0 = {w0_reference*1e6:.2f} µm")


# ## Geometry, model and boundary conditions
#
# The plate is a short cylinder whose axis is the z axis; the bottom face sits
# at z = 0 and the top (loaded) face at z = h. The pressure acts as a traction
# of magnitude p pointing into the plate (along -z) on the top face.

geometry = Cylinder(radius=a, length=h)
material = IsotropicSolid(
    density=7800.0, specific_heat_capacity=460.0, thermal_conductivity=50.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)

problem = Problem(
    name="Clamped circular plate under uniform pressure",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        # Clamped outer edge: the whole lateral face is held.
        FixedDisplacement(boundary=geometry.boundaries.outer_wall, physics=model,
                          value=(0.0, 0.0, 0.0)),
        # Uniform pressure on the top face, acting into the plate.
        Traction(boundary=geometry.boundaries.top, physics=model,
                 value=(0.0, 0.0, -pressure)),
        # The bottom face is unloaded.
        TractionFree(boundary=geometry.boundaries.bottom, physics=model),
    ),
)
# Quadratic (curved) tetrahedra: their quadratic displacement field captures
# bending with a single layer through the thickness, which keeps the full 3D
# plate model inexpensive.
mesh = TetrahedralMesher(target_size=1.5 * h, geometry_order=2).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)),
                      linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with thin-plate theory
#
# The deflection is read on the mid-plane z = h/2 along the +x axis. The plate
# moves in the -z direction, so w = -u_z.

displacement = result.field(model.displacement)

centre = np.array([[0.0, 0.0, h / 2]])
w0 = -float(displacement.evaluate(centre)[0, 2])
rel_w0 = abs(w0 - w0_reference) / w0_reference
print(f"Centre deflection: FEM {w0*1e6:.3f} µm,  Kirchhoff {w0_reference*1e6:.3f} µm")
print(f"relative deviation (centre deflection): {100*rel_w0:.2f} %")
assert w0 > 0, "The plate must deflect in the direction of the pressure."
assert rel_w0 < 0.04, "Centre deflection deviates more than the example tolerance (4 %)."

# Radial deflection profile, normalised by the reference centre deflection.
radii = np.linspace(0.0, 0.9 * a, 10)
profile_points = np.column_stack((radii, np.zeros_like(radii), np.full_like(radii, h / 2)))
w_fem = -np.asarray(displacement.evaluate(profile_points))[:, 2]
w_ref = w_reference(radii)
err_profile = float(np.max(np.abs(w_fem - w_ref)) / w0_reference)
print("Deflection profile w(r) on the mid-plane:")
for r, wf, wr in zip(radii, w_fem, w_ref):
    print(f"  r={r*1e3:5.1f} mm  FEM {wf*1e6:7.3f} µm  theory {wr*1e6:7.3f} µm")
print(f"max |w - w_Kirchhoff| / w0 = {100*err_profile:.2f} %")
assert err_profile < 0.04, "Deflection profile deviates more than the example tolerance (4 % of w0)."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
rr = np.linspace(0.0, a, 200)
ax.plot(rr * 1000, w_reference(rr) * 1e6, color="#214D43", lw=1.8, label="Kirchhoff plate theory")
ax.plot(radii * 1000, w_fem * 1e6, "o", color="#A85539", ms=5, label="OneSimulation 3D FEM")
ax.set(xlabel="Radius r / mm", ylabel="Deflection w / µm",
       title="Clamped circular plate: mid-plane deflection")
ax.legend()
plt.show()


# ## Going further
#
# Halve the thickness (a/h = 40) and watch the 3D solution approach the
# Kirchhoff curve even more closely as transverse shear fades. Replace the
# clamped lateral face by a simply supported edge and compare against
# w_0 = (5 + nu)/(1 + nu) times the clamped value.
