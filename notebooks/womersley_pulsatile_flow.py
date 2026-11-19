"""OneSimulation example: Womersley pulsatile flow between parallel plates.
Requires the installed, activated OneSimulation runtime.

An oscillating pressure gradient drives the flow between two plates. The channel is read from a
STEP file; the velocity profiles of the time-dependent Navier-Stokes simulation are compared with
the closed-form Womersley solution.
"""

# # Womersley pulsatile flow between parallel plates
#
# A viscous fluid fills the gap between two fixed plates a distance $2h$ apart. A pressure gradient
# that oscillates in time, $-\partial p/\partial x = G\cos\omega t$, drives the flow $u(y,t)$:
#
# $$\rho\,\frac{\partial u}{\partial t}=G\cos\omega t+\mu\,\frac{\partial^2 u}{\partial y^2},
#   \qquad u(\pm h,t)=0 .$$
#
# The channel comes from the CAD file `womersley_pulsatile_flow.step`, a thin slab between the
# plates `y_min` and `y_max`. The script `womersley_pulsatile_flow_geometry.py` builds it with the
# CAD objects of OneSimulation.
#
# **Reference.** The periodic state is the Womersley solution $u=\operatorname{Re}\{U(y)\,e^{i\omega t}\}$,
#
# $$U(y)=\frac{G}{i\rho\omega}\left[1-\frac{\cosh(k y)}{\cosh(k h)}\right],\qquad k=\sqrt{\frac{i\omega}{\nu}} .$$
#
# It is governed by the Womersley number $\alpha=h\sqrt{\omega/\nu}$. For small $\alpha$ the profile
# is a parabola in phase with the forcing. For $\alpha=3$, as here, the core is flattened and lags
# the pressure gradient, while the layers near the walls respond first.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import PeriodicPair, StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.hydrodynamics import IncompressibleNavierStokes, NoSlipWall
from onesim.simulation import PeriodicBoundary, Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The slab is periodic along the flow and across the span; the mesher joins the opposite faces.

channel = load("womersley_pulsatile_flow.step")
faces = channel.boundaries
length, gap, span = channel.size
half_gap = gap / 2
along, across = PeriodicPair(faces.x_min, faces.x_max), PeriodicPair(
    faces.z_min, faces.z_max
)
mesh = StructuredTetrahedralMesher(
    cells=(1, 12, 1), periodic_pairs=(along, across)
).generate(channel)

fig = plots.figure(1, 2, size=(12, 5.5))
plots.geometry(
    channel,
    selection={"plates": faces.y_min | faces.y_max},
    labels=False,
    title="Slab of the channel with its two plates",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"{mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The pressure gradient acts as a body force. Taylor-Hood elements in space and the second-order
# BDF method with 40 steps per period in time; the flow starts from rest.

forcing, womersley_number = 1.0, 3.0
water = NewtonianFluid(density=1000.0, dynamic_viscosity=1.0e-3)
viscosity = water.dynamic_viscosity / water.density
omega = (womersley_number / half_gap) ** 2 * viscosity
period = 2 * np.pi / omega


def womersley(y, t):
    """Velocity of the Womersley solution at the distance y from the channel centre."""
    k = np.sqrt(1j * omega / viscosity)
    return np.real(
        forcing
        / (1j * water.density * omega)
        * (1 - np.cosh(k * y) / np.cosh(k * half_gap))
        * np.exp(1j * omega * t)
    )


model = IncompressibleNavierStokes(
    domain=channel.volume,
    material=water,
    body_force=lambda x, t: np.column_stack(
        (np.full_like(x, forcing * np.cos(omega * t)), 0 * x, 0 * x)
    ),
)
problem = Problem(
    name="Womersley pulsatile channel flow",
    geometry=channel,
    physics=(model,),
    boundary_conditions=(
        NoSlipWall(boundary=faces.y_min | faces.y_max, physics=model),
        PeriodicBoundary(pair=along, physics=model),
        PeriodicBoundary(pair=across, physics=model),
    ),
    constraints=(MeanValue(model.pressure, 0.0),),
    initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
)
numerics = Numerics(
    spatial=FiniteElement(element=TaylorHood()),
    time_integrator=BDF(order=2, startup=BackwardEuler()),
    linear_solver=SparseDirect(),
    nonlinear_solver=Newton(
        linear_solver=SparseDirect(), absolute_tolerance=1e-12, relative_tolerance=0.0
    ),
)
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).run(
    start_time=0.0, end_time=4 * period, time_step=period / 40
)
velocity = result.field(model.velocity)

# ## Results
#
# The velocity across the gap is recorded over the fourth period, when the start-up has died out.

y = np.linspace(-half_gap, half_gap, 25)
times, history = velocity.time_series(
    np.column_stack(
        (np.full_like(y, length / 2), y + half_gap, np.full_like(y, span / 2))
    )
)
times, u = times[-41:], history[-41:, :, 0]
scale = np.abs(womersley(0.0, times)).max()
print("phase of the forcing   largest deviation from the Womersley profile")
for step in (0, 10, 20, 30):
    print(
        f"{9 * step:14d}°        {100 * np.abs(u[step] - womersley(y, times[step])).max() / scale:.2f} % of the centre-line amplitude"
    )
print(
    f"centre-line amplitude: simulation {1e3 * np.abs(u[:, 12]).max():.4f} mm/s, Womersley {1e3 * scale:.4f} mm/s, "
    f"quasi-steady parabola {1e3 * forcing * half_gap**2 / (2 * water.dynamic_viscosity):.4f} mm/s"
)

# ## Figure
#
# Velocity profiles at four phases of the forcing, and the centre-line velocity over one period
# against the Womersley solution and the quasi-steady response.

fine_y, fine_t = np.linspace(-half_gap, half_gap, 121), np.linspace(
    times[0], times[-1], 200
)
phases = {f"ωt = {9 * step}°": step for step in (0, 10, 20, 30)}

fig = plots.figure(1, 2, size=(12, 5))
plots.line(
    points={label: (1e3 * u[step], 1e3 * y) for label, step in phases.items()},
    reference={
        label: (1e3 * womersley(fine_y, times[step]), 1e3 * fine_y)
        for label, step in phases.items()
    },
    xlabel="velocity [mm/s]",
    ylabel="distance from the centre [mm]",
    title="Profiles: simulation (markers), Womersley (dashed)",
    ax=fig[0],
)
plots.line(
    points={"simulation": (times / period, 1e3 * u[:, 12])},
    reference={
        "Womersley": (fine_t / period, 1e3 * womersley(0.0, fine_t)),
        "quasi-steady response": (
            fine_t / period,
            1e3
            * forcing
            * half_gap**2
            / (2 * water.dynamic_viscosity)
            * np.cos(omega * fine_t),
        ),
    },
    xlabel="time [periods]",
    ylabel="centre-line velocity [mm/s]",
    title="Centre line: smaller amplitude and phase lag",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Lower the Womersley number to 0.5 and the profile becomes a parabola in phase with the forcing;
# raise it to 10 and the core moves as a plug with thin Stokes layers at the walls. Change the gap
# in `womersley_pulsatile_flow_geometry.py` and write the STEP file again; the frequency follows
# from the Womersley number. A circular pipe gives the classical solution with Bessel functions.
