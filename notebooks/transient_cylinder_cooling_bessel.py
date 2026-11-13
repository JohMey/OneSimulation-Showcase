"""OneSimulation example: transient cooling of a long solid cylinder (heat conduction).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the Bessel-function series solution.
"""


# # Transient cooling of a long solid cylinder
#
# A long solid cylinder of radius $R$ is initially at a uniform temperature $T_i$.
# At $t=0$ its lateral surface is suddenly brought to $T_s$ and held there, while
# the two end faces are insulated, so the problem is purely radial:
#
# $$\frac{\partial T}{\partial t}=\alpha\,\frac{1}{r}\frac{\partial}{\partial r}
#   \Big(r\frac{\partial T}{\partial r}\Big),\qquad T(R,t)=T_s,\quad T(r,0)=T_i,
#   \qquad \alpha=\frac{k}{\rho c_p}.$$
#
# OneSimulation solves this on a real 3D tetrahedral mesh of the cylinder with
# Lagrange elements (order 2) and a second-order BDF time integrator.
#
# **Reference.** With $\theta=(T-T_s)/(T_i-T_s)$ the classical series solution is
#
# $$\theta(r,t)=\sum_{n=1}^{\infty}\frac{2}{\lambda_n J_1(\lambda_n)}\,
#   J_0\!\Big(\lambda_n\frac{r}{R}\Big)\,
#   \exp\!\Big(-\lambda_n^2\,\frac{\alpha t}{R^2}\Big),\qquad J_0(\lambda_n)=0,$$
#
# where $J_0$, $J_1$ are Bessel functions of the first kind and $\lambda_n$ are the
# positive roots of $J_0$. We check the radial temperature profile against this
# series at several stored times.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jn_zeros

from onesim.cad.d3 import Cylinder
from onesim.materials import IsotropicSolid
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions.values import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The axial length only serves to form a 3D mesh; with insulated ends the field does
# not depend on $z$. The characteristic time $R^2/\alpha$ sets how fast the core cools.

R = 0.020             # m, cylinder radius
H = 0.008             # m, axial length (field-free in z)
k = 50.0              # W/(m·K), thermal conductivity
rho = 7800.0          # kg/m^3
cp = 460.0            # J/(kg·K)
T_i = 600.0           # K, initial temperature
T_s = 300.0           # K, surface temperature from t=0 on
alpha = k / (rho * cp)
tau = R * R / alpha
end_time = 0.5 * tau
time_step = end_time / 40.0
print(f"alpha = {alpha:.3e} m^2/s,  characteristic time R^2/alpha = {tau:.1f} s,  end time = {end_time:.1f} s")

geometry = Cylinder(radius=R, length=H)
mesh = TetrahedralMesher(target_size=R / 8.0).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")

material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
    youngs_modulus=200e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Transient cooling of a solid cylinder",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.outer_wall, physics=model, value=T_s),
        # insulated end faces (zero heat flux) -> purely radial transient
        HeatFlux(boundary=geometry.boundaries.top | geometry.boundaries.bottom, physics=model, value=0.0),
    ),
    initial_conditions=(InitialValue(model.temperature, T_i),),
)
simulation = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)),
        linear_solver=SparseDirect(),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
    ),
)
simulation.validate().require_valid()
result = simulation.run(start_time=0.0, end_time=end_time, time_step=time_step)


# ## Comparison with the Bessel series
#
# The computed temperature is evaluated along a radius at mid-height and compared
# with the series truncated after 60 roots of $J_0$. The sudden surface quench
# produces a step at $t=0$ whose steep front needs a few time steps to spread over
# more than one mesh cell, so the comparison starts a few stored times after the
# quench and stays inside the cylinder (up to $0.9R$).

roots = jn_zeros(0, 60)

def analytic(r, t):
    theta = np.zeros_like(r, dtype=float)
    for lam in roots:
        theta += (2.0 / (lam * j1(lam))) * j0(lam * r / R) * np.exp(-(lam**2) * alpha * t / R**2)
    return T_s + (T_i - T_s) * theta

sample_r = np.linspace(0.0, 0.9 * R, 36)
sample_points = np.column_stack((sample_r, np.zeros_like(sample_r), np.full_like(sample_r, H / 2)))
stored_times = np.asarray([float(t) for t in result.times])
positive = stored_times[stored_times > 0.0]
probe_times = positive[np.linspace(3, len(positive) - 1, 6).round().astype(int)]
max_abs_error = 0.0
profiles = {}
print("Error per time (K):")
for t in probe_times:
    num = np.asarray(result.field(model.temperature).evaluate(sample_points, time=float(t))).reshape(-1)
    ana = analytic(sample_r, float(t))
    err = float(np.max(np.abs(num - ana)))
    print(f"  t={t:6.2f} s  max|dT|={err:6.3f} K")
    profiles[float(t)] = (num, ana)
    max_abs_error = max(max_abs_error, err)

rel_error = max_abs_error / (T_i - T_s)
print(f"Maximum absolute error against the Bessel series: {max_abs_error:.3f} K  ({100*rel_error:.2f} % of Ti-Ts)")
assert rel_error < 0.03, "Deviation larger than the example tolerance (3 % of Ti-Ts)."
# Physical sanity: the axis cools monotonically and stays between T_s and T_i, and
# every profile falls from the axis towards the quenched surface.
axis_T = [profiles[float(t)][0][0] for t in probe_times]
assert all(b <= a + 1e-6 for a, b in zip(axis_T, axis_T[1:])), "The axis should cool monotonically."
assert T_s < axis_T[-1] < T_i
for t in probe_times:
    num, _ = profiles[float(t)]
    assert num[0] > num[-1], "Each profile should be hottest on the axis."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
colors = plt.cm.copper(np.linspace(0.25, 0.85, len(probe_times)))
for (t, (num, ana)), c in zip(profiles.items(), colors):
    ax.plot(sample_r * 1000, ana, color=c, lw=1.6)
    ax.plot(sample_r * 1000, num, "o", color=c, ms=3)
    ax.text(1.0, ana[0] + 4, f"t={t:.1f}s", color=c, fontsize=8, ha="left")
ax.set(xlabel="Radius r / mm", ylabel="Temperature / K",
       title="Cooling cylinder — lines: Bessel series, dots: OneSimulation FEM")
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Replace the fixed surface temperature by a convective boundary condition with a
# finite Biot number; the roots $\lambda_n$ then follow from
# $\lambda_n J_1(\lambda_n)=\mathrm{Bi}\,J_0(\lambda_n)$ instead of $J_0(\lambda_n)=0$.
# Refining the mesh near the wall or shortening the time step sharpens the early
# profiles, where the quench front is steepest.
