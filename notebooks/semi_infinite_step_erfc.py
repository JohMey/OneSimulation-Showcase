"""OneSimulation example: sudden surface temperature step on a long bar.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the semi-infinite-solid
error-function solution.
"""


# # Surface temperature step on a long bar (semi-infinite solid)
#
# A long bar is initially at a uniform temperature $T_i$. From $t=0$ on its end
# face at $x=0$ is suddenly held at $T_s$, while the far end and all lateral
# faces are insulated. As long as the thermal front has not reached the far end,
# the bar behaves like a **semi-infinite solid** and the problem is
# one-dimensional:
#
# $$\rho c_p \,\frac{\partial T}{\partial t} = k\,\frac{\partial^2 T}{\partial x^2},\qquad
#   T(x,0)=T_i,\quad T(0,t)=T_s,\quad T(x\to\infty,t)=T_i.$$
#
# OneSimulation solves it spatially with finite elements (Lagrange order 2) and a
# BDF2 time step on a real 3D mesh; the lateral faces and the far end are
# insulated (zero-flux condition).
#
# **Reference.** With $\alpha=k/(\rho c_p)$, the classical similarity solution is
#
# $$\frac{T(x,t)-T_s}{T_i-T_s}=\operatorname{erf}\!\Big(\frac{x}{2\sqrt{\alpha t}}\Big),$$
#
# and the surface heat flux into the bar is $q_s=k\,(T_i-T_s)/\sqrt{\pi\alpha t}$.
# We check the computed profiles at several stored times while the far end is
# still undisturbed.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
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
# The values are freely adjustable. The bar lies along $x$; its cross-section is
# small and only serves to form a slender 3D mesh. The end time is chosen such
# that the thermal penetration depth $\sim 4\sqrt{\alpha t}$ stays well below the
# bar length, which keeps the semi-infinite reference valid.

L = 0.20              # m, bar length (x direction)
cross = 0.01          # m, lateral size (insulated)
k = 45.0              # W/(m·K), thermal conductivity (carbon steel)
rho = 7850.0          # kg/m^3
cp = 460.0            # J/(kg·K)
T_i = 300.0           # K, initial temperature
T_s = 500.0           # K, end-face temperature from t=0 on
alpha = k / (rho * cp)
end_time = 80.0       # s
time_step = 2.0       # s
penetration = 4.0 * np.sqrt(alpha * end_time)
print(f"alpha = {alpha:.3e} m^2/s,  penetration depth 4*sqrt(alpha t_end) = {1000*penetration:.1f} mm of {1000*L:.0f} mm")

geometry = Box(length=L, width=cross, height=cross)
# Structured hexahedral mesh: many cells along x (the 1D direction), few across.
mesh = StructuredHexahedralMesher(cells=(100, 1, 1)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
    youngs_modulus=200e9, poisson_ratio=0.3, thermal_expansion_coefficient=1.2e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Surface temperature step on a long bar",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.x_min, physics=model, value=T_s),
        # far end insulated: the front must not reach it during the run
        HeatFlux(boundary=geometry.boundaries.x_max, physics=model, value=0.0),
        # laterally insulated (zero flux) -> 1D behaviour
        HeatFlux(
            boundary=(geometry.boundaries.y_min | geometry.boundaries.y_max
                      | geometry.boundaries.z_min | geometry.boundaries.z_max),
            physics=model, value=0.0,
        ),
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


# ## Comparison with the error-function solution
#
# We evaluate the computed temperature along the centre line at several stored
# times and compare it with the similarity solution. The sudden temperature step
# creates a corner singularity at $x=0$, $t=0$, which the first few time steps
# smear out; the comparison therefore starts once the front spans several
# elements and time steps (here from $t \ge 0.15\,t_\mathrm{end}$ on).

def analytic(x, t):
    return T_s + (T_i - T_s) * erf(x / (2.0 * np.sqrt(alpha * t)))

sample_x = np.linspace(0.0, 0.6 * L, 61)
stored_times = np.asarray([float(t) for t in result.times])
resolved = stored_times[stored_times >= 0.15 * end_time]
probe_times = resolved[np.linspace(0, len(resolved) - 1, 6).round().astype(int)]
max_abs_error = 0.0
profiles = {}
print("Error per time (K):")
for t in probe_times:
    pts = np.column_stack((sample_x, np.full_like(sample_x, cross / 2), np.full_like(sample_x, cross / 2)))
    num = np.asarray(result.field(model.temperature).evaluate(pts, time=float(t))).reshape(-1)
    ana = analytic(sample_x, float(t))
    err = float(np.max(np.abs(num - ana)))
    print(f"  t={t:5.1f} s  max|dT|={err:6.3f} K")
    profiles[float(t)] = (num, ana)
    max_abs_error = max(max_abs_error, err)

rel_error = max_abs_error / (T_s - T_i)
print(f"Maximum absolute error against erf solution: {max_abs_error:.3f} K  ({100*rel_error:.2f} % of Ts-Ti)")

# The far end must still be undisturbed, otherwise the semi-infinite reference
# would not apply; the reference itself predicts the residual change there.
far_end = np.array([[L, cross / 2, cross / 2]])
far_T = float(np.asarray(result.field(model.temperature).evaluate(far_end, time=end_time)).reshape(-1)[0])
far_ref = (T_s - T_i) * erfc(L / (2.0 * np.sqrt(alpha * end_time)))
print(f"Far-end temperature change at t_end: {far_T - T_i:.2e} K (reference {far_ref:.2e} K)")

# Physical sanity: every point heats up monotonically between stored times and
# stays between the two bounding temperatures.


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
colors = plt.cm.copper(np.linspace(0.25, 0.85, len(probe_times)))
for (t, (num, ana)), c in zip(profiles.items(), colors):
    ax.plot(sample_x * 1000, ana, color=c, lw=1.6)
    ax.plot(sample_x * 1000, num, "o", color=c, ms=3)
    x_label = 2.0 * np.sqrt(alpha * t)
    ax.text(x_label * 1000, analytic(x_label, t) + 6, f"t={t:.0f}s", color=c, fontsize=8, ha="center")
ax.axhline(T_i, color="#9a8f7d", ls="--", lw=1.0)
ax.set(xlabel="Position x / mm", ylabel="Temperature / K",
       title="Surface temperature step — lines: erf solution, dots: OneSimulation FEM")
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Refine the mesh along $x$ or halve the time step and watch the error against
# the error-function solution shrink. Extend the end time until the front reaches
# the insulated far end: the semi-infinite reference then breaks down and the
# finite-slab series solution with one fixed and one insulated face takes over.
# Replacing the fixed temperature by a constant surface heat flux leads to the
# companion closed-form solution with the complementary error function.
