"""OneSimulation example: constant surface heat flux on a semi-infinite solid.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form semi-infinite
constant-flux solution (square-root surface rise and erfc profile).
"""


# # Constant surface heat flux on a long bar (semi-infinite solid)
#
# A long bar is initially at a uniform temperature $T_i$. From $t=0$ on, a
# **constant heat flux** $q_0$ enters its end face at $x=0$; the far end and all
# lateral faces are insulated. As long as the thermal front has not reached the
# far end, the bar behaves like a **semi-infinite solid** and the problem is
# one-dimensional:
#
# $$\rho c_p \,\frac{\partial T}{\partial t} = k\,\frac{\partial^2 T}{\partial x^2},\qquad
#   T(x,0)=T_i,\quad -k\,\partial_x T\big|_{x=0}=q_0,\quad T(x\to\infty,t)=T_i.$$
#
# This is the early-phase physics of a frictionally heated brake-disc surface:
# the pad delivers a nearly constant heat flux and the disc body is still cold.
# OneSimulation solves it spatially with finite elements (Lagrange order 2) and a
# BDF2 time step on a real 3D mesh; the lateral faces and the far end are
# insulated (zero-flux condition).
#
# **Reference.** With $\alpha=k/(\rho c_p)$, the classical solution is
#
# $$T(x,t)-T_i=\frac{2q_0}{k}\sqrt{\frac{\alpha t}{\pi}}\,
#   \exp\!\Big(-\frac{x^2}{4\alpha t}\Big)
#   -\frac{q_0 x}{k}\,\operatorname{erfc}\!\Big(\frac{x}{2\sqrt{\alpha t}}\Big),$$
#
# so the surface temperature rises as $T_s(t)-T_i=(2q_0/k)\sqrt{\alpha t/\pi}$.
# We check the surface history and the profiles at several stored times while
# the far end is still undisturbed.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.heat_transfer import HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions.values import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Grey cast iron (brake-disc material) under a friction heat flux of 1 MW/m²,
# typical for a hard stop. The bar lies along $x$; its cross-section is small and
# only serves to form a slender 3D mesh. The end time is chosen such that the
# thermal penetration depth $\sim 4\sqrt{\alpha t}$ stays well below the bar
# length, which keeps the semi-infinite reference valid.

L = 0.06              # m, bar length (x direction)
cross = 0.005         # m, lateral size (insulated)
k = 52.0              # W/(m·K), thermal conductivity (grey cast iron)
rho = 7200.0          # kg/m^3
cp = 500.0            # J/(kg·K)
T_i = 300.0           # K, initial temperature
q0 = 1.0e6            # W/m^2, heat flux entering the surface from t=0 on
alpha = k / (rho * cp)
end_time = 5.0        # s
time_step = 0.1       # s
penetration = 4.0 * np.sqrt(alpha * end_time)
dT_surface_end = 2.0 * q0 / k * np.sqrt(alpha * end_time / np.pi)
print(f"alpha = {alpha:.3e} m^2/s,  penetration depth 4*sqrt(alpha t_end) = {1000*penetration:.1f} mm of {1000*L:.0f} mm")
print(f"Expected surface rise at t_end: {dT_surface_end:.1f} K")
assert penetration < 0.75 * L, "The far end must stay undisturbed for the semi-infinite reference."

geometry = Box(length=L, width=cross, height=cross)
# Structured hexahedral mesh: many cells along x (the 1D direction), few across.
mesh = StructuredHexahedralMesher(cells=(120, 1, 1)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
    youngs_modulus=110e9, poisson_ratio=0.26, thermal_expansion_coefficient=1.1e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Constant surface heat flux on a long bar",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        # heated end: HeatFlux is the outward flux, so heat entering is negative
        HeatFlux(boundary=geometry.boundaries.x_min, physics=model, value=-q0),
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


# ## Comparison with the closed-form solution
#
# The surface temperature history is compared with the square-root law at every
# stored time, and the centre-line profiles with the erfc solution at several
# stored times. The surface temperature has an infinite slope at $t=0$, which the
# first time steps resolve only approximately; the history comparison therefore
# starts after the first few steps.

def analytic(x, t):
    s = 2.0 * np.sqrt(alpha * t)
    return T_i + 2.0 * q0 / k * np.sqrt(alpha * t / np.pi) * np.exp(-(x / s) ** 2) - q0 * x / k * erfc(x / s)

stored_times = np.asarray([float(t) for t in result.times])
positive = stored_times[stored_times > 0.0]
surface = np.array([[0.0, cross / 2, cross / 2]])
surface_T = np.array([float(np.asarray(result.field(model.temperature).evaluate(surface, time=float(t))).reshape(-1)[0])
                      for t in positive])
surface_ref = T_i + 2.0 * q0 / k * np.sqrt(alpha * positive / np.pi)
history = positive >= 5.0 * time_step
history_error = float(np.max(np.abs(surface_T[history] - surface_ref[history])))
print(f"Surface history: max|dT| = {history_error:.3f} K from t = {5*time_step:.1f} s on "
      f"({100*history_error/dT_surface_end:.2f} % of the final surface rise)")
assert history_error < 0.02 * dT_surface_end, "Surface history deviates by more than 2 % of the final rise."

sample_x = np.linspace(0.0, 0.6 * L, 61)
probe_times = positive[np.linspace(len(positive) // 6, len(positive) - 1, 6).round().astype(int)]
max_abs_error = 0.0
profiles = {}
print("Profile error per time (K):")
for t in probe_times:
    pts = np.column_stack((sample_x, np.full_like(sample_x, cross / 2), np.full_like(sample_x, cross / 2)))
    num = np.asarray(result.field(model.temperature).evaluate(pts, time=float(t))).reshape(-1)
    ana = analytic(sample_x, float(t))
    err = float(np.max(np.abs(num - ana)))
    print(f"  t={t:4.1f} s  max|dT|={err:6.3f} K")
    profiles[float(t)] = (num, ana)
    max_abs_error = max(max_abs_error, err)

rel_error = max_abs_error / dT_surface_end
print(f"Maximum profile error against the erfc solution: {max_abs_error:.3f} K  ({100*rel_error:.2f} % of the final surface rise)")
assert rel_error < 0.02, "Deviation larger than the example tolerance (2 % of the final surface rise)."

# The far end must still be undisturbed, otherwise the semi-infinite reference
# would not apply; the reference itself predicts the residual change there.
far_end = np.array([[L, cross / 2, cross / 2]])
far_T = float(np.asarray(result.field(model.temperature).evaluate(far_end, time=end_time)).reshape(-1)[0])
far_ref = analytic(L, end_time) - T_i
print(f"Far-end temperature change at t_end: {far_T - T_i:.2e} K (reference {far_ref:.2e} K)")
assert abs(far_T - T_i) < 0.01 * dT_surface_end, "The thermal front reached the far end."

# Physical sanity: the surface heats monotonically and the deposited energy
# equals q0 * t (energy balance of the insulated bar).
assert np.all(np.diff(surface_T) > 0.0), "The surface should heat up monotonically."
x_fine = np.linspace(0.0, L, 1201)
pts = np.column_stack((x_fine, np.full_like(x_fine, cross / 2), np.full_like(x_fine, cross / 2)))
T_end = np.asarray(result.field(model.temperature).evaluate(pts, time=end_time)).reshape(-1)
stored_energy = rho * cp * np.trapezoid(T_end - T_i, x_fine)
print(f"Stored energy per unit area at t_end: {stored_energy:.4e} J/m^2  (supplied q0*t = {q0*end_time:.4e} J/m^2)")
assert abs(stored_energy - q0 * end_time) < 0.01 * q0 * end_time, "Energy balance violated by more than 1 %."


# ## Figure

cream = "#fbf5eb"
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
colors = plt.cm.copper(np.linspace(0.25, 0.85, len(probe_times)))
ax.set_facecolor("#fffaf2")
for (t, (num, ana)), c in zip(profiles.items(), colors):
    ax.plot(sample_x * 1000, ana, color=c, lw=1.6)
    ax.plot(sample_x * 1000, num, "o", color=c, ms=3)
    ax.text(0.3, ana[0] + 3, f"t={t:.1f}s", color=c, fontsize=8, ha="left")
ax.axhline(T_i, color="#9a8f7d", ls="--", lw=1.0)
ax.set(xlabel="Depth x / mm", ylabel="Temperature / K",
       title="Profiles — lines: erfc solution, dots: FEM")
ax.grid(alpha=0.25)
ax2.set_facecolor("#fffaf2")
t_dense = np.linspace(0.0, end_time, 300)
ax2.plot(t_dense, T_i + 2.0 * q0 / k * np.sqrt(alpha * t_dense / np.pi), color="#8c5a2b", lw=1.6, label=r"$(2q_0/k)\sqrt{\alpha t/\pi}$")
ax2.plot(positive, surface_T, "o", color="#b7792f", ms=3, label="OneSimulation FEM (surface)")
ax2.set(xlabel="Time / s", ylabel="Surface temperature / K", title="Surface temperature history")
ax2.legend(loc="lower right")
ax2.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Switch the flux off after a braking time $t_b$: by superposition the surface
# then cools as $\sqrt{t}-\sqrt{t-t_b}$. Replace the constant flux by a
# convective condition to obtain the third classical semi-infinite solution, or
# extend the end time until the front reaches the insulated far end, where the
# finite-slab series with one heated and one insulated face takes over.
