"""OneSimulation example: the shallow-water equations - dam breaks and a lake at rest.
Requires the installed, activated OneSimulation runtime.

The one-dimensional shallow-water equations are solved with a well-balanced finite-volume scheme,
first order and with MUSCL reconstruction. Two dam breaks are checked against the exact solutions
of Stoker (wet bed) and Ritter (dry bed) on four grids, and a lake at rest around an island against
the exact still water.
"""

# # Shallow-water equations: dam breaks and a lake at rest
#
# Where water is shallow compared with the length of its waves, the pressure is hydrostatic and
# the velocity nearly uniform over the depth. Mass and momentum balances over the water column
# give the shallow-water (Saint-Venant) equations for the depth $h$ and the discharge $q=hu$ over a
# bed of elevation $b(x)$:
#
# $$\frac{\partial h}{\partial t}+\frac{\partial q}{\partial x}=0,\qquad
#   \frac{\partial q}{\partial t}+\frac{\partial}{\partial x}\Big(\frac{q^2}{h}+\frac{g h^2}{2}\Big)=-g\,h\,\frac{\partial b}{\partial x}.$$
#
# They are used for rivers, floods, tides and tsunamis. Two classic problems test a solver:
#
# * **Dam break.** A channel of length 10 m holds water 1 m deep behind a gate at $x_0=5$ m and
#   water 0.1 m deep (or none) in front of it. When the gate is removed, a rarefaction runs back
#   into the reservoir and a bore (or, on a dry bed, a thin front) runs downstream.
# * **Lake at rest.** Still water over an uneven bed, with an island that rises above the
#   surface, must stay exactly still. A scheme that does not balance the pressure gradient and the
#   bed slope discretely produces spurious currents.
#
# | end | condition |
# |---|---|
# | $x=0$ and $x=10$ m | reflecting wall, $q=0$ (the waves do not reach them) |
#
# **Reference.** J. J. Stoker, *Water Waves*, Interscience 1957, ch. 10 (dam break on a wet bed:
# rarefaction, constant state $h_m$, bore); A. Ritter, *Z. Vereines Deutscher Ingenieure* 36, 947
# (1892) (dry bed: $h=(2c_0-\xi)^2/(9g)$ with $\xi=(x-x_0)/t$, $c_0=\sqrt{g h_0}$); E. Audusse,
# F. Bouchut, M.-O. Bristeau, R. Klein, B. Perthame, *SIAM J. Sci. Comput.* 25, 2050-2065 (2004)
# for the hydrostatic reconstruction that keeps the lake at rest.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fvm import (
    MUSCL,
    ConvexInvariantDomain,
    FiniteVolume,
    HydrostaticReconstruction,
    MinmodLimiter,
    Rusanov,
)
from onesim.numerics.time import TimeMethod, TimeStepLimits
from onesim.physics.hydrodynamics import TopographicShallowWater
from onesim.simulation import Problem, ReflectingWall, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## Exact dam-break solution
#
# On a wet bed of depth $h_1$ the middle depth $h_m$ joins the rarefaction from the reservoir to the
# bore: $2(c_0-c_m)=(h_m-h_1)\sqrt{g(h_m+h_1)/(2h_mh_1)}$. On a dry bed $h_m=0$ and the solution
# is Ritter's.

g, gate, deep, end = 9.81, 5.0, 1.0, 0.6


def dam_break(x, t, shallow):
    """Exact depth of the dam break at time t (Stoker; Ritter for a dry bed)."""
    c0 = np.sqrt(g * deep)
    middle = (
        brentq(
            lambda h: 2 * (c0 - np.sqrt(g * h))
            - (h - shallow) * np.sqrt(g * (h + shallow) / (2 * h * shallow)),
            shallow,
            deep,
        )
        if shallow > 0
        else 0.0
    )
    speed = 2 * (c0 - np.sqrt(g * middle))
    front = middle * speed / (middle - shallow) if shallow > 0 else speed
    xi = (x - gate) / t
    return np.select(
        [xi < -c0, xi < speed - np.sqrt(g * middle), xi < front],
        [deep, (2 * c0 - xi) ** 2 / (9 * g), middle],
        shallow,
    )


# ## Model
#
# One problem for each downstream depth, solved with two finite-volume schemes on four grids.
# Both use the Rusanov flux, the hydrostatic reconstruction for the bed and a two-stage
# strong-stability-preserving Runge-Kutta method with a CFL-limited time step; the second scheme
# adds MUSCL reconstruction with the minmod limiter and a positivity limiter for the dry front.

channel = Line(start=(0, 0, 0), end=(10, 0, 0))
model = TopographicShallowWater(domain=channel.line, bed_elevation=0.0)
released = {
    0.1: lambda x: np.where(x < gate, deep, 0.1),
    0.0: lambda x: np.where(x < gate, deep, 0.0),
}
problems = {
    shallow: Problem(
        name="Dam break",
        geometry=channel,
        physics=(model,),
        boundary_conditions=(
            ReflectingWall(physics=model, boundary=channel.boundaries.all),
        ),
        initial_conditions=(
            InitialValue(model.depth, initial),
            InitialValue(model.discharge, 0.0),
        ),
    )
    for shallow, initial in released.items()
}
schemes = {
    "first order": FiniteVolume(
        convection=Rusanov(),
        provider=HydrostaticReconstruction(),
        positivity=ConvexInvariantDomain(),
        cell_quadrature_order=4,
    ),
    "MUSCL, minmod": FiniteVolume(
        convection=Rusanov(),
        provider=HydrostaticReconstruction(),
        reconstruction=MUSCL(),
        limiter=MinmodLimiter(),
        positivity=ConvexInvariantDomain(),
        cell_quadrature_order=4,
    ),
}
grids = (100, 200, 400)
depths = {
    (shallow, scheme, cells): Simulation(
        problem=problems[shallow],
        mesh=StructuredLineMesher(cells=cells).generate(channel),
        numerics=Numerics(
            spatial=spatial,
            time_integrator=TimeMethod("ssprk2"),
            step_limits=TimeStepLimits(stability_safety_factor=0.8),
        ),
    )
    .run(start_time=0.0, end_time=end, time_step=0.05)
    .field(model.depth)
    for shallow in problems
    for scheme, spatial in schemes.items()
    for cells in grids
}

# ## Results
#
# The error is the mean absolute depth error over the channel at $t=0.6$ s; the water volume must
# stay $5.5$ m³ per metre width (wet bed) and $5.0$ m³ (dry bed).

x = np.linspace(0.005, 9.995, 2000)
points = np.column_stack((x, 0 * x, 0 * x))
errors = {
    key: np.mean(np.abs(np.ravel(depth.evaluate(points)) - dam_break(x, end, key[0])))
    for key, depth in depths.items()
}
for shallow, label in (
    (0.1, "wet bed, h1 = 0.1 m (Stoker)"),
    (0.0, "dry bed (Ritter)"),
):
    print(label)
    print(
        "  cells    first order    MUSCL      observed order (MUSCL)   volume change (MUSCL)"
    )
    for index, cells in enumerate(grids):
        second = errors[(shallow, "MUSCL, minmod", cells)]
        order = (
            np.log2(errors[(shallow, "MUSCL, minmod", grids[index - 1])] / second)
            if index
            else np.nan
        )
        volume = depths[(shallow, "MUSCL, minmod", cells)].integrate().value - 5 * (
            deep + shallow
        )
        print(
            f"  {cells:5d} {errors[(shallow, 'first order', cells)]:13.2e} {second:10.2e} {order:16.2f} {volume:22.1e} m³"
        )

# ## Lake at rest around an island
#
# A Gaussian hill 0.8 m high under water whose surface lies at 0.6 m: the hilltop is dry. The run
# lasts two seconds; the exact solution is the initial state, so depth and discharge must not
# change.

hill = lambda x: 0.8 * np.exp(-(((x - 5.0) / 1.2) ** 2))
lake = TopographicShallowWater(domain=channel.line, bed_elevation=hill)
still = Simulation(
    problem=Problem(
        name="Lake at rest",
        geometry=channel,
        physics=(lake,),
        boundary_conditions=(
            ReflectingWall(physics=lake, boundary=channel.boundaries.all),
        ),
        initial_conditions=(
            InitialValue(lake.depth, lambda x: np.maximum(0.6 - hill(x), 0.0)),
            InitialValue(lake.discharge, 0.0),
        ),
    ),
    mesh=StructuredLineMesher(cells=200).generate(channel),
    numerics=Numerics(
        spatial=schemes["MUSCL, minmod"],
        time_integrator=TimeMethod("ssprk2"),
        step_limits=TimeStepLimits(stability_safety_factor=0.8),
    ),
).run(start_time=0.0, end_time=2.0, time_step=0.1)
lake_depth = still.field(lake.depth)
level = np.ravel(still.field(lake.free_surface).evaluate(points))
wet = np.ravel(lake_depth.evaluate(points)) > 0
print(
    f"\nlake at rest after 2 s: largest change of the depth {np.abs(lake_depth.values - lake_depth.at(0.0).values).max():.1e} m, "
    f"largest discharge {np.abs(still.field(lake.discharge).values).max():.1e} m²/s, "
    f"dry cells {np.count_nonzero(lake_depth.values == 0)} (island)"
)

# ## Figure
#
# Depth profiles of both dam breaks at $t=0.6$ s on the 200-cell grid against the exact solutions,
# the grid study, and the lake at rest.

fig = plots.figure(2, 2, size=(15, 10))
for panel, shallow, title in (
    (fig[0], 0.1, "Wet bed: rarefaction and bore"),
    (fig[1], 0.0, "Dry bed: rarefaction and front"),
):
    plots.line(
        x,
        {
            scheme: np.ravel(depths[(shallow, scheme, 200)].evaluate(points))
            for scheme in schemes
        },
        reference={
            "exact (Stoker)" if shallow else "exact (Ritter)": dam_break(
                x, end, shallow
            )
        },
        xlabel="x [m]",
        ylabel="depth h [m]",
        ylimits=(-0.02, 1.05),
        title=f"{title}, t = {end} s, 200 cells",
        ax=panel,
    )
plots.convergence(
    [10 / cells for cells in grids],
    {
        f"{scheme}, {'wet' if shallow else 'dry'} bed": [
            errors[(shallow, scheme, cells)] for cells in grids
        ]
        for shallow in problems
        for scheme in schemes
    },
    order=1,
    xlabel="cell width [m]",
    ylabel="mean depth error [m]",
    title="Grid study",
    ax=fig[2],
)
plots.line(
    x,
    {"water surface after 2 s": np.where(wet, level, np.nan), "bed": hill(x)},
    hlines={"initial surface": 0.6},
    xlabel="x [m]",
    ylabel="elevation [m]",
    ylimits=(0.0, 1.0),
    title="Lake at rest around an island",
    legend="upper right",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Give the bed a slope with `bed_elevation` and release the dam onto it, or add a rain source
# with `water_source`. The same model on a periodic channel carries a small wave, whose speed
# $\sqrt{gh}$ the example on shallow-water wave speed checks.
