"""OneSimulation example: gravity-driven laminar film on an inclined plane.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form Nusselt film
profile (half-parabola with a shear-free surface).
"""


# # Gravity-driven film flow down an inclined plane (Nusselt film)
#
# A thin layer of a viscous liquid of thickness $h$ runs down a plane inclined
# at the angle $\theta$. The liquid sticks to the plane (no slip) while its free
# surface is in contact with air and carries no shear stress. Gravity supplies
# the streamwise body force $\rho g \sin\theta$ per unit volume. Once the film
# is fully developed the velocity depends only on the wall-normal coordinate
# $y$, and the incompressible Navier-Stokes equations reduce to
#
# $$\mu\,\frac{d^2 u}{dy^2} + \rho g\sin\theta = 0,\qquad
#   u(0)=0,\qquad \mu\,\frac{du}{dy}\Big|_{y=h}=0 .$$
#
# OneSimulation solves the full 3D stationary Navier-Stokes system with
# Taylor-Hood finite elements (quadratic velocity, linear pressure) and a Newton
# solver. The free surface is modelled as a flat, traction-free boundary
# (zero Cauchy traction $\sigma\cdot n = 0$), the plane as a no-slip wall.
#
# **Reference.** With the kinematic viscosity $\nu=\mu/\rho$ the Nusselt
# solution is the half-parabola
#
# $$u(y)=\frac{g\sin\theta}{2\nu}\,(2hy-y^2),\qquad
#   u_s=\frac{g\sin\theta\,h^2}{2\nu},\qquad
#   q=\int_0^h u\,dy=\frac{g\sin\theta\,h^3}{3\nu},$$
#
# where $u_s$ is the surface speed and $q$ the flow rate per unit width.
# We check the computed profile, the surface speed and the flow rate.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})
from scipy.integrate import simpson

from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import (
    FixedVelocity, FlowTraction, IncompressibleNavierStokes, NoSlipWall,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions.values import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The film thickness lies along $y$; the streamwise length ($x$) and spanwise
# width ($z$) only serve to form a short 3D slab of liquid. The flow is fully
# developed, so the solution is independent of $x$ and $z$. The liquid is a
# viscous oil with kinematic viscosity $\nu = 10^{-3}\,\mathrm{m^2/s}$; the
# density only scales the gravitational body force and the (vanishing) inertia.

length = 0.020        # m, streamwise extent (x)
h = 0.002             # m, film thickness (y), the only physically resolved direction
span = 0.005          # m, spanwise extent (z)
rho = 1.0             # kg/m^3, density (unit value: inertia vanishes in a developed film)
mu = 1.0e-3           # Pa·s, dynamic viscosity  ->  nu = mu/rho = 1e-3 m^2/s (viscous oil)
nu = mu / rho
g = 9.81              # m/s^2
theta = np.deg2rad(30.0)
g_parallel = g * np.sin(theta)        # m/s^2, gravity component along the plane
body_force = rho * g_parallel         # N/m^3, streamwise body force per unit volume

u_surface_reference = g_parallel * h**2 / (2.0 * nu)
flow_rate_reference = g_parallel * h**3 / (3.0 * nu)   # m^2/s (per unit width)
reynolds = flow_rate_reference / nu
print(f"g sin(theta) = {g_parallel:.3f} m/s^2,  surface speed = {1000*u_surface_reference:.3f} mm/s,  "
      f"flow rate q = {1e6*flow_rate_reference:.3f} mm^2/s,  Re = q/nu = {reynolds:.4f} (laminar film)")


# ## Model
#
# The film thickness $h$ is the box width (the $y$ extent). The plane (y_min) is
# a no-slip wall and the free surface (y_max) is traction-free; the upstream,
# downstream and spanwise faces carry the fully developed Nusselt trace, so each
# of the six boundary faces has an explicit condition. The traction-free face
# also fixes the pressure level (zero at the surface), hence no pressure gauge
# constraint is needed.

geometry = Box(length=length, width=h, height=span)
mesh = StructuredTetrahedralMesher(cells=(6, 8, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")

fluid = NewtonianFluid(density=rho, dynamic_viscosity=mu)


def gravity(y):
    # Constant streamwise body force (per unit volume); y only fixes the array shape.
    return np.column_stack((np.full_like(y, body_force), np.zeros_like(y), np.zeros_like(y)))


def nusselt_profile(y):
    u = g_parallel / (2.0 * nu) * (2.0 * h * y - y**2)
    return np.column_stack((u, np.zeros_like(y), np.zeros_like(y)))


def traction_free(y):
    # Zero Cauchy traction on the free surface (no shear, zero gauge pressure).
    return np.zeros((len(y), 3))


model = IncompressibleNavierStokes(domain=geometry.volume, material=fluid, body_force=gravity)
developed = (geometry.boundaries.x_min | geometry.boundaries.x_max
             | geometry.boundaries.z_min | geometry.boundaries.z_max)
problem = Problem(
    name="Inclined plane film flow",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        NoSlipWall(boundary=geometry.boundaries.y_min, physics=model),
        FlowTraction(boundary=geometry.boundaries.y_max, physics=model, value=traction_free),
        FixedVelocity(boundary=developed, physics=model, value=nusselt_profile),
    ),
    initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
)
simulation = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(), absolute_tolerance=1e-11, relative_tolerance=0.0,
        ),
        initial_guess={model.velocity: (0.0, 0.0, 0.0)},
    ),
)
result = simulation.solve_stationary()


# ## Comparison with the Nusselt profile
#
# We sample the streamwise velocity along a wall-normal line at mid-slab and
# compare with the closed-form half-parabola. Taylor-Hood velocities are
# quadratic, so the exact profile is represented essentially to solver precision.

y_samples = np.linspace(0.0, h, 41)
line = np.column_stack((np.full_like(y_samples, length / 2),
                        y_samples,
                        np.full_like(y_samples, span / 2)))
velocity = np.asarray(result.field(model.velocity).evaluate(line))
u_numeric = velocity[:, 0]
u_exact = g_parallel / (2.0 * nu) * (2.0 * h * y_samples - y_samples**2)

max_abs_error = float(np.max(np.abs(u_numeric - u_exact)))
rel_error = max_abs_error / u_surface_reference
u_surface_numeric = float(u_numeric[-1])
print(f"Surface speed: FEM {1000*u_surface_numeric:.4f} mm/s,  reference {1000*u_surface_reference:.4f} mm/s")
print(f"Maximum absolute profile error: {max_abs_error:.3e} m/s  ({100*rel_error:.2e} % of u_s)")

# Flow rate per unit width from the sampled profile (Simpson on 41 points).
flow_rate_numeric = float(simpson(u_numeric, x=y_samples))
print(f"Flow rate per unit width: FEM {1e6*flow_rate_numeric:.4f} mm^2/s,  "
      f"reference {1e6*flow_rate_reference:.4f} mm^2/s")

# Surface shear stress: central difference on the quadratic profile near y = h.
shear_surface = mu * (u_numeric[-1] - u_numeric[-2]) / (y_samples[-1] - y_samples[-2])
shear_wall = mu * (u_numeric[1] - u_numeric[0]) / (y_samples[1] - y_samples[0])
shear_wall_reference = rho * g_parallel * h
print(f"Shear stress: surface {shear_surface:.3e} Pa (should vanish),  wall {shear_wall:.4e} Pa "
      f"(reference rho g sin(theta) h = {shear_wall_reference:.4e} Pa)")

# Gauge pressure is zero at the traction-free surface and uniform across the film.


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
ax.grid(alpha=0.25)
ax.plot(u_exact * 1000, y_samples * 1000, color="#214D43", lw=1.8, label="Nusselt half-parabola")
ax.plot(u_numeric * 1000, y_samples * 1000, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.axhline(0.0, color="#555", lw=2)
ax.axhline(h * 1000, color="#5F9175", lw=1.5, ls="--")
ax.text(0.02 * 1000 * u_surface_reference, h * 1000 * 1.02, "free surface (shear-free)", color="#5F9175", fontsize=9)
ax.text(0.02 * 1000 * u_surface_reference, -0.09 * h * 1000, "inclined plane (no slip)", color="#555", fontsize=9)
ax.set(xlabel="Streamwise velocity u / (mm/s)", ylabel="Distance from the plane y / mm",
       title=f"Gravity-driven film on a {np.rad2deg(theta):.0f}° incline: velocity profile")
ax.legend(loc="center right")
plt.show()


# ## Going further
#
# Change the inclination angle or the film thickness: the surface speed scales
# with $h^2\sin\theta$ and the flow rate with $h^3\sin\theta$. Add a prescribed
# air shear stress on the free surface through the traction value (co-current or
# counter-current) and recover the superposition of the Nusselt parabola with a
# linear Couette profile. Increase the flow rate until the film Reynolds number
# approaches the long-wave instability threshold $\mathrm{Re}_c=\tfrac{5}{6}\cot\theta$,
# where a flat developed film ceases to be the physically observed state.
