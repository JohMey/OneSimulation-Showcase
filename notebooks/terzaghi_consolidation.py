"""OneSimulation example: one-dimensional Terzaghi consolidation of a soil column.
Requires the installed, activated OneSimulation runtime.

The OneSimulation Biot solution is checked against the classical Terzaghi series
solution for the excess pore pressure and the degree of consolidation.
"""


# # One-dimensional Terzaghi consolidation
#
# A saturated soil column of height $H$ rests on a rigid, impermeable base and is
# laterally confined (one-dimensional, oedometer-like compression). At $t=0$ a
# constant surface load is applied at the top. Because the pore water cannot
# escape instantly, the load is first carried by an almost uniform **excess pore
# pressure** $u_0$. The top surface is free-draining, so the water slowly
# escapes upward and the excess pressure dissipates while the skeleton takes up
# the load.
#
# OneSimulation solves the fully coupled Biot poroelasticity equations (solid
# displacement + Darcy flow) with linear finite elements and a backward-Euler
# step.
#
# **Reference.** With $Z=z/H$ measured from the drained top, the consolidation
# coefficient $c_v=k/(\mu S_m)$ and the time factor $T_v=c_v t/H^2$, Terzaghi's
# classical series holds for the excess pore pressure
#
# $$\frac{u(Z,T_v)}{u_0}=\sum_{m=0}^{\infty}\frac{4}{(2m+1)\pi}\,
#   \sin\!\Big(\frac{(2m+1)\pi}{2}Z\Big)\,
#   \exp\!\Big[-\Big(\frac{(2m+1)\pi}{2}\Big)^2 T_v\Big],$$
#
# and for the average degree of consolidation
#
# $$U(T_v)=1-\sum_{m=0}^{\infty}\frac{8}{(2m+1)^2\pi^2}\,
#   \exp\!\Big[-\Big(\frac{(2m+1)\pi}{2}\Big)^2 T_v\Big].$$
#
# We check both the pore-pressure profile and $U(T_v)$ against these series.

from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
from onesim.cad.geometry.primitives import Rectangle
from onesim.meshing.meshes.api import cartesian
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, Nitsche
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BackwardEuler
from onesim.physics.geotechnics import (
    BiotConsolidation, BiotFixedComponent, BiotNormalTraction,
    PoreFluidFlux, PrescribedPorePressure,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The column is tall and thin. A nearly incompressible skeleton with a small
# specific storage makes the instantaneous (undrained) response an almost uniform
# excess pore pressure $u_0$, which is the premise of Terzaghi's derivation. The
# surface load is unit compression. Drainage is at the top only, so the drainage
# path length equals the full height $H$.

H = 1.0                 # m, column height (z = y direction)
width_x = 0.2           # m, lateral width (confined, 1D compression)
load = 1.0              # Pa, surface load (compression -> negative traction)
E = 1.0                 # Pa, Young's modulus
nu = 0.49               # near-incompressible skeleton -> uniform undrained u0
alpha = 1.0             # Biot coefficient
storage = 1.0e-3        # 1/Pa, specific storage
permeability = 0.15     # mobility k/mu units
viscosity = 1.0

modulus = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))   # oedometric (constrained) modulus
mass = storage + alpha**2 / modulus                   # 1D storage coefficient S_m
cv = permeability / (viscosity * mass)                # consolidation coefficient
u0 = alpha * load / (modulus * mass)                  # undrained excess pore pressure
end_time = 0.5
time_step = 0.01
print(f"constrained modulus = {modulus:.4f} Pa,  c_v = {cv:.4f} m^2/s")
print(f"undrained excess pore pressure u0 = {u0:.4f} Pa,  final Tv = {cv*end_time/H**2:.3f}")

rectangle = Rectangle(length=width_x, width=H)
# Structured quadrilateral mesh: a few cells across, many up the column (z).
mesh = cartesian(geometry=rectangle, shape=(5, 60), extents=(width_x, H)).build()

model = BiotConsolidation(
    domain=rectangle.domain(),
    youngs_modulus=E,
    poisson_ratio=nu,
    biot_coefficient=alpha,
    specific_storage=storage,
    permeability=permeability,
    fluid_viscosity=viscosity,
)

# Every boundary face gets an explicit mechanical and hydraulic condition:
#   mechanics: lateral sides roll (fixed x), base is fixed (y), top is loaded;
#   hydraulics: sides and base are impermeable (zero pore-fluid flux), top drains.
problem = Problem(
    name="One-dimensional Terzaghi consolidation",
    geometry=rectangle,
    physics=(model,),
    boundary_conditions=(
        # --- mechanical ---
        BiotFixedComponent(physics=model, boundary=rectangle.boundaries.x_min, component="x"),
        BiotFixedComponent(physics=model, boundary=rectangle.boundaries.x_max, component="x"),
        BiotFixedComponent(physics=model, boundary=rectangle.boundaries.y_min, component="y"),
        BiotNormalTraction(physics=model, boundary=rectangle.boundaries.y_max, value=-load),
        # --- hydraulic ---
        PoreFluidFlux(physics=model, boundary=rectangle.boundaries.x_min, flux=0.0),
        PoreFluidFlux(physics=model, boundary=rectangle.boundaries.x_max, flux=0.0),
        PoreFluidFlux(physics=model, boundary=rectangle.boundaries.y_min, flux=0.0),
        PrescribedPorePressure(physics=model, boundary=rectangle.boundaries.y_max, value=0.0),
    ),
    initial_conditions=(InitialValue(model.fluid_content, 0.0),),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(1), boundary_enforcement=Nitsche(penalty=16)),
    linear_solver=SparseDirect(),
    time_integrator=BackwardEuler(),
    nonlinear_solver=Newton(
        linear_solver=SparseDirect(), jacobian="analytic", absolute_tolerance=1e-11
    ),
)
simulation = Simulation(problem=problem, mesh=mesh, numerics=numerics)
simulation.validate().require_valid()
result = simulation.run(start_time=0.0, end_time=end_time, time_step=time_step)


# ## Comparison with the Terzaghi series
#
# We read the excess pore pressure along the drained-to-undrained column axis and
# compare the profile at a representative time with the series. $Z=z/H$ is the
# depth measured downward from the drained top, so $Z=0$ at the drain and $Z=1$ at
# the impermeable base.

op = result.operators[model.model_id]
points = op.space.dof_coordinates
axis = np.flatnonzero(np.isclose(points[:, 0], 0.0))
axis = axis[np.argsort(points[axis, 1])]
y = points[axis, 1]
Z = (H - y) / H         # depth from the drained top

def pressure_series(Z, Tv, terms=200):
    m = np.arange(terms)
    coeff = (2 * m + 1) * np.pi / 2
    amp = (4.0 / ((2 * m + 1) * np.pi)) * np.exp(-(coeff**2) * Tv)
    return np.sin(np.outer(Z, coeff)) @ amp

def consolidation_series(Tv, terms=200):
    m = np.arange(terms)
    coeff = (2 * m + 1) * np.pi / 2
    return 1.0 - (8.0 / np.pi**2) * (np.exp(-np.outer(Tv, coeff**2)) @ (1.0 / (2 * m + 1) ** 2))

times = np.asarray([float(t) for t in result.times])
# Representative profile at an intermediate time factor (Tv ~ 0.2).
profile_time = times[np.argmin(np.abs(cv * times / H**2 - 0.2))]
Tv_profile = cv * profile_time / H**2
num_profile = result.field(model.pore_pressure).at(profile_time).values[axis]
ref_profile = u0 * pressure_series(Z, Tv_profile)
profile_rel = np.linalg.norm(num_profile - ref_profile) / np.linalg.norm(ref_profile)
print(f"pore-pressure profile at Tv = {Tv_profile:.3f}:  relative error = {100*profile_rel:.2f} %")

# Degree of consolidation over time from the mass-weighted average excess pressure.
mass_matrix = op.kernel.assembly.mass
column_volume = float(mass_matrix.sum())
positive = times[times > 0]
Tv = cv * positive / H**2
num_U, ref_U = [], []
for t in positive:
    avg = float(np.sum(mass_matrix @ result.field(model.pore_pressure).at(t).values)) / column_volume
    num_U.append(1.0 - avg / u0)
num_U = np.asarray(num_U)
ref_U = consolidation_series(Tv)
U_max_error = float(np.max(np.abs(num_U - ref_U)))
print(f"degree of consolidation U(Tv): max absolute error = {U_max_error:.4f}")

# Physical sanity and tolerances against the closed-form series.
assert profile_rel < 0.05, "Pore-pressure profile deviates more than 5 % from the Terzaghi series."
assert U_max_error < 0.03, "Degree of consolidation deviates more than 0.03 from the Terzaghi series."
assert np.all(np.diff(num_U) > -1e-6), "Consolidation must advance monotonically."
assert num_U[-1] > num_U[0], "The column must consolidate over time."


# ## Figure

cream = "#fbf5eb"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)

ax = axes[0]
ax.set_facecolor("#fffaf2")
snap_times = positive[np.linspace(0, len(positive) - 1, 5).round().astype(int)]
colors = plt.cm.copper(np.linspace(0.25, 0.85, len(snap_times)))
for t, c in zip(snap_times, colors):
    Tv_t = cv * t / H**2
    ax.plot(u0 * pressure_series(Z, Tv_t), Z, color=c, lw=1.6)
    ax.plot(result.field(model.pore_pressure).at(t).values[axis], Z, "o", color=c, ms=3)
    ax.text(0.05 * u0, 0.02 + np.interp(0.05 * u0, u0 * pressure_series(Z, Tv_t)[::-1], Z[::-1]),
            f"Tv={Tv_t:.2f}", color=c, fontsize=8)
ax.invert_yaxis()
ax.set(xlabel="Excess pore pressure u / Pa", ylabel="Depth from drain  Z = z/H",
       title="Profiles — lines: Terzaghi series, dots: OneSimulation")
ax.grid(alpha=0.25)

ax = axes[1]
ax.set_facecolor("#fffaf2")
fine = np.linspace(Tv.min(), Tv.max(), 200)
ax.plot(fine, 100 * consolidation_series(fine), color="#39715e", lw=1.8, label="Terzaghi series")
ax.plot(Tv, 100 * num_U, "o", color="#b4572b", ms=4, label="OneSimulation")
ax.invert_yaxis()
ax.set(xlabel="Time factor Tv = cv t / H^2", ylabel="Degree of consolidation U / %",
       title="Degree of consolidation")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Lower the permeability or extend `end_time` to follow the column all the way to
# full consolidation ($U\to 1$). Draining both the top and the base halves the
# drainage path to $H/2$ and quarters the consolidation time; the same series then
# applies with $H\to H/2$.
