"""OneSimulation example: laser heating of a rotating silicon wafer.
Requires the installed, activated OneSimulation runtime.

A laser spot sweeps in and out over a silicon wafer that turns on its stage, so the heated spot
writes a spiral on the wafer. The wafer is read from a STEP file; the moving spot is a heat flux
that depends on position and time. The stored heat is compared with the laser energy, and the
temperature pattern shows the trace of the spot.
"""

# # Laser heating of a rotating silicon wafer
#
# In rapid thermal processing a laser heats a wafer locally. Here a wafer of 100 mm diameter and
# 0.5 mm thickness turns at 6 revolutions per minute while the laser moves radially between 10 and
# 40 mm from the centre and back within the 10 s of the run. Seen from the wafer, the spot
# travels on a spiral:
#
# $$\mathbf p(t)=r_l(t)\,\big(\cos\omega t,\,-\sin\omega t\big),\qquad
#   r_l(t)=25\ \mathrm{mm}-15\ \mathrm{mm}\cdot\cos\frac{2\pi t}{10\ \mathrm s},\qquad \omega=0.2\pi\ \mathrm{1/s}.$$
#
# The absorbed laser power $P=50$ W arrives as a Gaussian surface flux of radius $w=3$ mm,
#
# $$q(\mathbf x,t)=\frac{P}{\pi w^2}\,e^{-|\mathbf x-\mathbf p(t)|^2/w^2},\qquad
#   \rho c\,\frac{\partial T}{\partial t}=\nabla\cdot(k\nabla T),$$
#
# with silicon $k=148$ W/(m K), $\rho=2330$ kg/m³, $c=712$ J/(kg K).
#
# | face | condition |
# |---|---|
# | `top` | laser flux $q(\mathbf x,t)$ |
# | `insulated`: bottom and rim | insulated (short run in vacuum, radiation neglected) |
# | initial state | 20 °C |
#
# The wafer comes from the CAD file `laser_heating_rotating_wafer.step`; the script
# `laser_heating_rotating_wafer_geometry.py` builds it with the CAD objects of OneSimulation.
#
# **Reference.** With all losses neglected the energy balance is exact: the heat stored in the
# wafer equals the laser energy $Pt$, and the mean temperature rises linearly as $Pt/(\rho cV)$.
# The spot stays well inside the wafer, so no part of the Gaussian is cut off.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, disc_blocks, extrude, merge_blocks
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.heat_transfer import HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the wafer with its named faces. `disc_blocks` meshes its face with a square core
# and a ring of quadrilaterals, about 2.5 mm in size; one layer of quadratic hexahedra spans the
# thickness, through which the temperature hardly changes.

wafer = load("laser_heating_rotating_wafer.step")
faces = wafer.boundaries
radius, thickness = wafer.size[0] / 2, wafer.size[2]
mesh = extrude(
    merge_blocks(disc_blocks((0, 0), (radius,), 24, (14,), core=0.5), order=2),
    thickness,
    Geometric(1, ratio=1.0),
    geometry=wafer,
)

view = plots.Camera(elevation=40, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 5.5))
plots.geometry(
    wafer,
    selection={"top, under the laser": faces.top, "insulated": faces.insulated},
    labels=False,
    camera=view,
    title="Wafer with its named faces",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} quadratic hexahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The laser is a `HeatFlux` whose value depends on position and time (a positive flux leaves the
# wafer, so the laser enters with a negative sign). Second-order BDF steps of 0.1 s move the spot
# by at most 2.6 mm, less than its radius.

power, spot, initial, duration = 50.0, 0.003, 293.15, 10.0
path_radius = lambda t: 0.025 - 0.015 * np.cos(2 * np.pi * t / duration)
path = lambda t: (
    path_radius(t) * np.cos(0.2 * np.pi * t),
    -path_radius(t) * np.sin(0.2 * np.pi * t),
)
silicon = IsotropicSolid(
    density=2330.0,
    specific_heat_capacity=712.0,
    thermal_conductivity=148.0,
    youngs_modulus=1.3e11,
    poisson_ratio=0.28,
    thermal_expansion_coefficient=2.6e-6,
)
model = HeatConduction(domain=wafer.volume, material=silicon, heating=0.0)
problem = Problem(
    name="Laser on a rotating wafer",
    geometry=wafer,
    physics=(model,),
    boundary_conditions=(
        HeatFlux(
            boundary=faces.top,
            physics=model,
            value=lambda x, y, z, t: -power
            / (np.pi * spot**2)
            * np.exp(-((x - path(t)[0]) ** 2 + (y - path(t)[1]) ** 2) / spot**2),
        ),
        HeatFlux(boundary=faces.insulated, physics=model, value=0.0),
    ),
    initial_conditions=(InitialValue(model.temperature, initial),),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=SparseDirect(),
    time_integrator=BDF(order=2, startup=BackwardEuler()),
)
temperature = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .run(start_time=0.0, end_time=duration, time_step=0.1)
    .field(model.temperature)
)

# ## Results
#
# Stored heat and mean temperature against the laser energy, and the temperature under the spot.

capacity, volume = (
    silicon.density * silicon.specific_heat_capacity,
    np.pi * radius**2 * thickness,
)
times = temperature.times
under_spot = np.array(
    [np.ravel(temperature.evaluate([[*path(t), thickness]], time=t))[0] for t in times]
)
print(
    "time [s]   stored heat [J]   laser energy [J]   mean T [°C]   expected [°C]   T under the spot [°C]"
)
for t in (1.0, 2.5, 5.0, 7.5, 10.0):
    index = np.argmin(np.abs(times - t))
    stored = capacity * (
        temperature.integrate(time=times[index]).value - initial * volume
    )
    print(
        f"{times[index]:8.2f} {stored:17.3f} {power * times[index]:18.3f} {stored / (capacity * volume) + initial - 273.15:13.3f} "
        f"{power * times[index] / (capacity * volume) + initial - 273.15:15.3f} {under_spot[index] - 273.15:21.2f}"
    )

# ## Figure
#
# The temperature of the top face at three instants, and the
# temperature under the spot and the mean temperature over time.

fig = plots.figure(2, 2, size=(15, 12))
for panel, instant in zip(range(3), (2.5, 5.0, 10.0)):
    plots.surface(
        temperature,
        time=instant,
        boundary=faces.top,
        component=lambda T, position: T - 273.15,
        limits=(20, 300),
        contours=8,
        subdivisions=2,
        xlabel="x [m]",
        ylabel="y [m]",
        label="T [°C]",
        title=f"Top face after {instant:.1f} s",
        ax=fig[panel],
    )
plots.line(
    y={
        "under the moving spot": (times, under_spot - 273.15),
        "mean of the wafer": (
            times,
            np.array([temperature.integrate(time=t).value / volume for t in times])
            - 273.15,
        ),
    },
    reference={
        "mean expected from P t / (ρ c V)": (
            times,
            power * times / (capacity * volume) + initial - 273.15,
        )
    },
    xlabel="time [s]",
    ylabel="T [°C]",
    title="Temperature under the spot and mean temperature",
    legend="upper left",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Add the radiation of both faces (`RadiativeHeatFlux`), spin the wafer faster and compare the
# uniformity of the heating, or replace the spiral by a raster scan of the laser.
