"""OneSimulation example: periodic surface temperature wave in a semi-infinite solid.
Requires the installed, activated OneSimulation runtime.

The daily temperature cycle at the ground surface travels into the soil as a damped thermal wave.
The soil column is read from a STEP file; the simulated amplitude and phase lag at every depth are
compared with the closed-form solution.
"""

# # Periodic surface temperature wave in a semi-infinite solid
#
# The surface of a thick body, for example the ground under the daily temperature cycle,
# oscillates about a mean value, $T_s(t)=T_m+A\sin(\omega t)$. Heat conduction carries this
# oscillation into the solid as a **damped thermal wave**: with depth the amplitude decays and the
# oscillation lags behind the surface.
#
# $$\rho c_p\,\frac{\partial T}{\partial t}=k\,\frac{\partial^2 T}{\partial x^2},\qquad
#   T(0,t)=T_m+A\sin(\omega t),\qquad T(x\to\infty,t)=T_m .$$
#
# The soil column comes from the CAD file `periodic_surface_temperature_wave.step`; its surface is
# named `x_min`, its deep end `x_max`, and the four lateral faces form the group `sides`. The
# script `periodic_surface_temperature_wave_geometry.py` builds it with the CAD objects of
# OneSimulation, seven penetration depths long.
#
# **Reference.** With the diffusivity $\alpha=k/(\rho c_p)$ the periodic solution is
#
# $$T(x,t)-T_m=A\,e^{-x/d}\,\sin\!\big(\omega t-x/d\big),\qquad d=\sqrt{\frac{2\alpha}{\omega}} .$$
#
# The amplitude falls to $1/e$ and the phase lags by one radian per penetration depth $d$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh

column = load("periodic_surface_temperature_wave.step")
faces = column.boundaries
length, cross, _ = column.size
mesh = StructuredHexahedralMesher(cells=(64, 1, 1)).generate(column)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(
    column,
    selection={"surface": faces.x_min, "deep end": faces.x_max},
    labels=False,
    title="Soil column with its surface and deep end",
    ax=fig[0],
)
plots.mesh(
    mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} hexahedra", ax=fig[1]
)
plots.show()

# ## Model
#
# Moist soil under a daily cycle of ±10 K. The surface follows the sinusoid, the deep end stays at
# the mean temperature and the sides are insulated. Quadratic elements in space, the second-order
# BDF method with 40 steps per day in time.

soil = IsotropicSolid(
    density=2000.0,
    specific_heat_capacity=1000.0,
    thermal_conductivity=1.0,
    youngs_modulus=1e9,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=1e-5,
)
mean, swing, period = 288.0, 10.0, 86400.0
omega = 2 * np.pi / period
depth = np.sqrt(
    2 * soil.thermal_conductivity / (soil.density * soil.specific_heat_capacity) / omega
)

model = HeatConduction(domain=column.volume, material=soil, heating=0.0)
problem = Problem(
    name="Periodic surface temperature wave",
    geometry=column,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(
            boundary=faces.x_min,
            physics=model,
            value=lambda t: mean + swing * np.sin(omega * t),
        ),
        FixedTemperature(boundary=faces.x_max, physics=model, value=mean),
        HeatFlux(boundary=faces.sides, physics=model, value=0.0),
    ),
    initial_conditions=(InitialValue(model.temperature, mean),),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=SparseDirect(),
    time_integrator=BDF(order=2, startup=BackwardEuler()),
)
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).run(
    start_time=0.0, end_time=6 * period, time_step=period / 40
)
temperature = result.field(model.temperature)

# ## Amplitude and phase lag
#
# The temperature along the column is recorded over the sixth day, when the start-up has died
# out. Projecting each depth's signal onto $e^{-i\omega t}$ gives its amplitude and phase lag.

x = np.linspace(0, 4 * depth, 81)
times, signals = temperature.time_series(
    np.column_stack((x, np.full_like(x, cross / 2), np.full_like(x, cross / 2)))
)
times, signals = times[-40:], signals[-40:] - mean
wave = 2j * np.mean(signals * np.exp(-1j * omega * times)[:, None], axis=0)
amplitude, lag = np.abs(wave), -np.unwrap(np.angle(wave))

print(
    f"penetration depth d = {100 * depth:.2f} cm, column length {length / depth:.1f} d"
)
print("depth    amplitude [K]   theory   phase lag [rad]   theory")
for index in (0, 10, 20, 40, 60):
    print(
        f"{x[index] / depth:4.1f} d {amplitude[index]:14.3f} {swing * np.exp(-x[index] / depth):8.3f}"
        f" {lag[index]:17.3f} {x[index] / depth:8.3f}"
    )
print(
    f"penetration depth from the amplitude decay: {-100 / np.polyfit(x, np.log(amplitude), 1)[0]:.2f} cm, "
    f"from the phase lag: {100 / np.polyfit(x, lag, 1)[0]:.2f} cm"
)

# ## Figure
#
# Amplitude and phase lag against depth, and the temperature over the sixth day at four depths.

hours = (times - times[0]) / 3600
picks = {"surface": 0, "x = 0.5 d": 10, "x = d": 20, "x = 2 d": 40}
theory = {
    label: mean + swing * np.exp(-x[i] / depth) * np.sin(omega * times - x[i] / depth)
    for label, i in picks.items()
}

fig = plots.figure(1, 3, size=(16, 4.8))
plots.line(
    100 * x,
    points={"simulation": (100 * x[::2], amplitude[::2])},
    reference={"A exp(-x/d)": swing * np.exp(-x / depth)},
    vlines={"penetration depth d": 100 * depth},
    xlabel="depth [cm]",
    ylabel="temperature amplitude [K]",
    title="Amplitude",
    ax=fig[0],
)
plots.line(
    100 * x,
    points={"simulation": (100 * x[::2], lag[::2])},
    reference={"x/d": x / depth},
    xlabel="depth [cm]",
    ylabel="phase lag [rad]",
    title="Phase lag",
    ax=fig[1],
)
plots.line(
    hours,
    points={label: mean + signals[:, i] for label, i in picks.items()},
    reference=theory,
    xlabel="time within the sixth day [h]",
    ylabel="temperature [K]",
    title="Simulation (markers) and theory (dashed)",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Use the yearly cycle instead of the daily one: the penetration depth grows by $\sqrt{365}$ to
# about two metres, so lengthen the column in `periodic_surface_temperature_wave_geometry.py` and
# write the STEP file again. Add a second soil layer with a different conductivity, or replace the
# fixed surface temperature by convection to the air.
