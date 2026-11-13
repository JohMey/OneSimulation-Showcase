"""OneSimulation example: transient heat conduction in a finite slab.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the known closed-form series solution.
"""


# # Heat conduction in a finite slab
#
# A flat slab of thickness $L$ is initially at a uniform temperature $T_i$.
# From $t=0$ on, **both surfaces** are held at $T_s$, and the slab cools by pure
# heat conduction. The sides are insulated, so the problem is one-dimensional.
#
# The transient heat equation is solved:
#
# $$\rho c_p \,\frac{\partial T}{\partial t} = k\,\frac{\partial^2 T}{\partial x^2},\qquad
#   T(x,0)=T_i,\quad T(0,t)=T(L,t)=T_s.$$
#
# OneSimulation solves it spatially with finite elements (Lagrange order 2) and a
# BDF2 time step on a real 3D mesh; the lateral faces are kept insulated
# (zero-flux condition).
#
# **Reference.** With $\theta=(T-T_s)/(T_i-T_s)$ and $\alpha=k/(\rho c_p)$, the
# classical Fourier series holds:
#
# $$\theta(x,t)=\sum_{n=1,3,5,\dots}\frac{4}{n\pi}\,
#   \sin\!\Big(\frac{n\pi x}{L}\Big)\,
#   \exp\!\Big[-\Big(\frac{n\pi}{L}\Big)^2\alpha t\Big].$$
#
# We check the computed solution against this series.

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
# The values are freely adjustable. The slab lies along $x$; its width and height
# are small and only serve to form a slender 3D mesh. $\alpha$ and the
# characteristic time $L^2/\alpha$ set how fast it cools down.

L = 0.10              # m, slab thickness (x direction)
cross = 0.02          # m, lateral size (insulated)
k = 50.0             # W/(m·K), thermal conductivity
rho = 1000.0          # kg/m^3
cp = 500.0            # J/(kg·K)
T_i = 600.0           # K, initial temperature
T_s = 300.0           # K, surface temperature from t=0 on
alpha = k / (rho * cp)
tau = L * L / alpha
end_time = 0.6 * tau
time_step = end_time / 40.0
print(f"alpha = {alpha:.3e} m^2/s,  characteristic time L^2/alpha = {tau:.1f} s,  end time = {end_time:.1f} s")

geometry = Box(length=L, width=cross, height=cross)
# Structured hexahedral mesh: many cells along x (the 1D direction), few across.
mesh = StructuredHexahedralMesher(cells=(60, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
    youngs_modulus=100e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
problem = Problem(
    name="Heat conduction in a finite slab",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.x_min, physics=model, value=T_s),
        FixedTemperature(boundary=geometry.boundaries.x_max, physics=model, value=T_s),
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


# ## Comparison with the series solution
#
# We evaluate the computed temperature along the centre line at several times and
# compare it with the truncated Fourier series (301 terms).

def analytic(x, t, terms=301):
    theta = np.zeros_like(x, dtype=float)
    for n in range(1, terms, 2):
        theta += (4.0 / (n * np.pi)) * np.sin(n * np.pi * x / L) * np.exp(-((n * np.pi / L) ** 2) * alpha * t)
    return T_s + (T_i - T_s) * theta

# Compare the interior (the sudden surface boundary condition creates a corner
# singularity at t=0; right at the boundary at the earliest times neither the
# series nor the mesh is meaningfully resolved). The comparison starts at the
# second stored time, in the resolved region.
sample_x = np.linspace(0.06 * L, 0.94 * L, 37)
stored_times = np.asarray([float(t) for t in result.times])
positive = stored_times[stored_times > 0.0]
probe_times = positive[np.linspace(1, len(positive) - 1, 6).round().astype(int)]
max_abs_error = 0.0
profiles = {}
print("Error per time (K):")
for t in probe_times:
    pts = np.column_stack((sample_x, np.full_like(sample_x, cross / 2), np.full_like(sample_x, cross / 2)))
    num = np.asarray(result.field(model.temperature).evaluate(pts, time=float(t))).reshape(-1)
    ana = analytic(sample_x, float(t))
    err = float(np.max(np.abs(num - ana)))
    print(f"  t={t:6.2f} s  max|dT|={err:6.3f} K")
    profiles[float(t)] = (num, ana)
    max_abs_error = max(max_abs_error, err)

rel_error = max_abs_error / (T_i - T_s)
print(f"Maximum absolute error against the series: {max_abs_error:.3f} K  ({100*rel_error:.2f} % of Ti-Ts)")
# Physical sanity: the centre cools monotonically, the surface stays at T_s.


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
colors = FIELD_CMAP(np.linspace(0.25, 0.85, len(probe_times)))
for (t, (num, ana)), c in zip(profiles.items(), colors):
    ax.plot(sample_x * 1000, ana, color=c, lw=1.6)
    ax.plot(sample_x * 1000, num, "o", color=c, ms=3)
    ax.text(L * 1000 * 0.5, ana[len(ana) // 2] + 4, f"t={t:.0f}s", color=c, fontsize=8, ha="center")
ax.set(xlabel="Position x / mm", ylabel="Temperature / K",
       title="Cooling slab — lines: Fourier series, dots: OneSimulation FEM")
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Refine the mesh or halve the time step and watch the error against the series
# shrink. Replace the two-sided fixed temperature with a one-sided fixed
# temperature and an insulated back face; the series solution must then be
# replaced by the corresponding odd-multiple series.
