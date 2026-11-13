"""OneSimulation example: spreading of a Gaussian heat pulse on an insulated bar.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the self-similar Gaussian solution.
"""


# # Spreading of a Gaussian heat pulse (1D)
#
# A long, thermally insulated bar starts with a localized temperature bump: on top
# of a uniform baseline $T_0$ there is a Gaussian hump of amplitude $A$ and width
# $\sigma_0$ centred at $x_0$. From $t=0$ on the bar simply conducts heat, with no
# source and no loss anywhere (every face carries zero heat flux). The hump then
# flattens and widens while conserving its total energy. This is the 1D heat
# equation
#
# $$\frac{\partial T}{\partial t}=\alpha\,\frac{\partial^2 T}{\partial x^2},\qquad
#   T(x,0)=T_0+A\,\exp\!\Big[-\frac{(x-x_0)^2}{2\sigma_0^2}\Big],\qquad
#   \alpha=\frac{k}{\rho c_p},$$
#
# with insulated ends. OneSimulation solves it on a real 3D slender mesh with finite
# elements (Lagrange order 2) and a BDF2 time step; the four lateral faces and the
# two ends are all insulated, so the behaviour is one-dimensional.
#
# **Reference.** On an (effectively) infinite bar a Gaussian stays Gaussian. Its
# variance grows linearly and its peak decays so that the area is preserved:
#
# $$T(x,t)=T_0+A\,\frac{\sigma_0}{\sqrt{\sigma_0^2+2\alpha t}}\,
#   \exp\!\Big[-\frac{(x-x_0)^2}{2(\sigma_0^2+2\alpha t)}\Big],\qquad
#   \sigma^2(t)=\sigma_0^2+2\alpha t.$$
#
# The bar is made long enough that the hump never reaches the ends over the
# simulated window, so the insulated-end reflection stays negligible.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

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
# The hump sits at the centre of the bar. $\alpha$ and the end time are chosen so
# the hump roughly doubles its width yet stays more than five standard deviations
# away from either end, where the solution is already vanishingly small.

L = 1.0               # m, bar length (x direction)
cross = 0.02          # m, lateral size (insulated)
x0 = L / 2            # m, centre of the initial hump
sigma0 = 0.05         # m, initial Gaussian width
A = 80.0              # K, initial peak rise above the baseline
T0 = 300.0            # K, uniform baseline temperature
k = 40.0              # W/(m·K)
rho = 8000.0          # kg/m^3
cp = 500.0            # J/(kg·K)
alpha = k / (rho * cp)
end_time = 300.0      # s
time_step = end_time / 60.0
sigma_end = np.sqrt(sigma0**2 + 2 * alpha * end_time)
print(f"alpha = {alpha:.3e} m^2/s,  initial width = {sigma0*1000:.0f} mm,  final width = {sigma_end*1000:.0f} mm")
print(f"distance centre-to-end = {L/2/sigma_end:.1f} final standard deviations")

geometry = Box(length=L, width=cross, height=cross)
# Structured hexahedral mesh: many cells along x (the 1D direction), few across.
mesh = StructuredHexahedralMesher(cells=(200, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
    youngs_modulus=200e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Gaussian heat pulse on an insulated bar",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        # every face insulated (zero heat flux): energy is conserved, 1D behaviour
        HeatFlux(
            boundary=(geometry.boundaries.x_min | geometry.boundaries.x_max
                      | geometry.boundaries.y_min | geometry.boundaries.y_max
                      | geometry.boundaries.z_min | geometry.boundaries.z_max),
            physics=model, value=0.0,
        ),
    ),
    initial_conditions=(
        InitialValue(model.temperature,
                     lambda x, y, z: T0 + A * np.exp(-((x - x0) ** 2) / (2 * sigma0 ** 2))),
    ),
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


# ## Comparison with the self-similar solution
#
# We evaluate the computed temperature along the centre line at several stored
# times and compare it with the spreading Gaussian. The sampling window stays in
# the interior so the (negligible) end reflection plays no role.

def analytic(x, t):
    var = sigma0 ** 2 + 2 * alpha * t
    return T0 + A * (sigma0 / np.sqrt(var)) * np.exp(-((x - x0) ** 2) / (2 * var))

sample_x = np.linspace(0.15 * L, 0.85 * L, 61)
stored_times = np.asarray([float(t) for t in result.times])
positive = stored_times[stored_times > 0.0]
probe_times = positive[np.linspace(1, len(positive) - 1, 5).round().astype(int)]
max_abs_error = 0.0
profiles = {}
print("Error per time (K):")
for t in probe_times:
    pts = np.column_stack((sample_x, np.full_like(sample_x, cross / 2), np.full_like(sample_x, cross / 2)))
    num = np.asarray(result.field(model.temperature).evaluate(pts, time=float(t))).reshape(-1)
    ana = analytic(sample_x, float(t))
    err = float(np.max(np.abs(num - ana)))
    print(f"  t={t:6.1f} s  max|dT|={err:6.3f} K")
    profiles[float(t)] = (num, ana)
    max_abs_error = max(max_abs_error, err)

rel_error = max_abs_error / A
print(f"Maximum absolute error against the Gaussian: {max_abs_error:.3f} K  ({100*rel_error:.2f} % of A)")
assert rel_error < 0.03, "Deviation larger than the example tolerance (3 % of the initial amplitude A)."

# Physical sanity: the peak at the centre decays like sigma0/sqrt(sigma0^2+2*alpha*t)
# and the baseline far from the centre stays at T0 (no energy escapes).
center = np.array([[x0, cross / 2, cross / 2]])
peak = [float(np.asarray(result.field(model.temperature).evaluate(center, time=float(t))).reshape(-1)[0]) for t in probe_times]
assert all(b < a + 1e-6 for a, b in zip(peak, peak[1:])), "The peak should only decay."
peak_ana = [analytic(np.array([x0]), float(t))[0] for t in probe_times]
assert np.max(np.abs(np.array(peak) - np.array(peak_ana))) < 0.03 * A, "Peak decay off the self-similar law."
print(f"Centre temperature: start {T0 + A:.1f} K -> end {peak[-1]:.1f} K  (analytic {peak_ana[-1]:.1f} K)")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
colors = plt.cm.inferno(np.linspace(0.2, 0.8, len(probe_times)))
for (t, (num, ana)), c in zip(profiles.items(), colors):
    ax.plot(sample_x * 1000, ana, color=c, lw=1.6)
    ax.plot(sample_x * 1000, num, "o", color=c, ms=3)
    ax.text(x0 * 1000, analytic(np.array([x0]), t)[0] + 1.0, f"t={t:.0f}s", color=c, fontsize=8, ha="center")
ax.set(xlabel="Position x / mm", ylabel="Temperature / K",
       title="Spreading heat pulse — lines: self-similar Gaussian, dots: OneSimulation FEM")
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Narrow the initial hump (smaller sigma0) and refine the mesh together: a sharper
# start needs more cells per width to keep the early-time error small. Shorten the
# bar until the tails reach the ends within the window and watch the insulated
# boundaries fold the profile back, so the simple Gaussian reference no longer
# applies and an image-source sum is needed instead.
