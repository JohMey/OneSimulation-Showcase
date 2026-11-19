"""OneSimulation example: Tesla microvalve - diodicity over the Reynolds number.
Requires the installed, activated OneSimulation runtime.

A Tesla valve has no moving parts, yet it resists flow in one direction more than in the other. The
valve is read from a STEP file and cut to its cross-section; the laminar flow is solved in both
directions with Taylor-Hood elements over a sweep of the Reynolds number. The diodicity is checked
against the exact Stokes limit and the published range, with a mesh study.
"""

# # Tesla microvalve: diodicity over the Reynolds number
#
# Nikola Tesla's "valvular conduit" (US patent 1,329,559, 1920) is a channel with side loops. In
# the forward direction the fluid passes straight through. In the reverse direction part of the
# stream is caught by the loops, turned around and fed back against the main stream, which costs
# pressure. Micropumps use such valves because they cannot wear or stick. Their figure of merit is
# the **diodicity**, the ratio of the pressure drops at the same flow rate:
#
# $$Di=\frac{\Delta p_\text{reverse}}{\Delta p_\text{forward}} .$$
#
# The valve comes from the CAD file `tesla_valve_diodicity.step`, written by
# `tesla_valve_diodicity_geometry.py`: a straight channel of width $w=1$ mm with one loop whose
# legs leave the channel at 30°, leaning against the forward flow. The model is its cross-section
# and solves the steady Navier-Stokes equations for water; the Reynolds number is $Re=\bar U w/\nu$
# with the mean inflow speed $\bar U=0.1$ m/s, and it is varied through the viscosity.
#
# | side | forward flow | reverse flow |
# |---|---|---|
# | `channel.x_min` | parabolic inflow, mean speed $\bar U$ | traction free |
# | `channel.x_max` | traction free | parabolic inflow, mean speed $\bar U$ |
# | all walls | no slip | no slip |
#
# **Reference.** Creeping (Stokes) flow is reversible: reversing the inflow reverses the velocity
# and the pressure drop, so $Di\to1$ for $Re\to0$ (up to the small difference between the inflow
# and the outflow condition at the two ends). Inertia makes the valve work: published measurements
# and simulations of Tesla-type valves show the diodicity rising from 1 with the Reynolds number,
# by a few per cent to some tens of per cent per stage in the laminar range, and multiplying over
# several stages (F. K. Forster, R. L. Bardell, M. A. Afromowitz, N. R. Sharma, A. Blanchard,
# ASME FED 234, 39-44, 1995; A. R. Gamboa, C. J. Morris, F. K. Forster, *J. Fluids Eng.* 127,
# 339-346, 2005).

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import (
    FixedVelocity,
    FlowTraction,
    NoSlipWall,
    PlanarIncompressibleNavierStokes,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `section` cuts the slab at half thickness; the faces keep their names. The triangles are an
# eighth of the channel width; the mesh study uses a sixth and an eleventh.

part = load("tesla_valve_diodicity.step")
valve = part.section((0, 0, 1), origin=(0, 0, part.size[2] / 2))
faces = valve.boundaries
left, right = faces["channel.x_min"], faces["channel.x_max"]
walls = faces.all - left - right
width, speed = 1e-3, 0.1
meshes = {
    cells: TriangularMesher(target_size=width / cells).generate(valve)
    for cells in (6, 8, 11)
}
mesh = meshes[8]

fig = plots.figure(layout=[["geometry"], ["mesh"]], size=(14, 9))
plots.geometry(
    part,
    selection={
        "forward inlet": part.boundaries["channel.x_min"],
        "forward outlet": part.boundaries["channel.x_max"],
    },
    labels=False,
    camera=plots.Camera(elevation=45, azimuth=-70),
    title="Tesla valve with one loop",
    ax=fig["geometry"],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"{mesh.cell_count} triangles",
    ax=fig["mesh"],
)
plots.show()

# ## Model
#
# For each direction the Reynolds number is raised step by step; every Newton solve starts from the
# previous solution. The first step, $Re=0.01$, is practically creeping flow. The coarser and the
# finer mesh repeat the steps up to $Re=300$ for the mesh study.

reynolds_numbers = (0.01, 100, 200, 300, 400, 500)
profile = lambda y: 6 * speed * y * (width - y) / width**2
directions = {"forward": (left, right, 1.0), "reverse": (right, left, -1.0)}
runs = [(8, reynolds_numbers), (6, (0.01, 100, 200, 300)), (11, (0.01, 100, 200, 300))]
results = {}
for cells, levels in runs:
    for direction, (inlet, outlet, sign) in directions.items():
        guess = (0.0, 0.0)
        for reynolds in levels:
            model = PlanarIncompressibleNavierStokes(
                domain=valve.domain(),
                material=NewtonianFluid(
                    density=1000.0, dynamic_viscosity=1000.0 * speed * width / reynolds
                ),
            )
            problem = Problem(
                name=f"Tesla valve, {direction}",
                geometry=valve,
                physics=(model,),
                boundary_conditions=(
                    FixedVelocity(
                        boundary=inlet,
                        physics=model,
                        value=lambda y: np.column_stack((sign * profile(y), 0 * y)),
                    ),
                    NoSlipWall(boundary=walls, physics=model),
                    FlowTraction(boundary=outlet, physics=model, value=(0.0, 0.0)),
                ),
            )
            numerics = Numerics(
                spatial=FiniteElement(element=TaylorHood()),
                initial_guess={model.velocity: guess},
                nonlinear_solver=Newton(
                    linear_solver=SparseDirect(),
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-9,
                    maximum_iterations=25,
                ),
            )
            solved = Simulation(
                problem=problem, mesh=meshes[cells], numerics=numerics
            ).solve_stationary()
            results[direction, reynolds, cells] = solved.field(
                model.velocity
            ), solved.field(model.pressure)
            previous = results[direction, reynolds, cells][0]
            guess = lambda x, y: previous.evaluate(np.column_stack((x, y)))

# ## Results
#
# The pressure drop is the difference of the mean pressures across the inlet and the outlet. The share
# of the flow that takes the loop is the flow rate through the top of the loop, at $x=6.8$ mm.

ends = {
    "forward": (-4e-3 + 1e-9, 22e-3 - 1e-9),
    "reverse": (22e-3 - 1e-9, -4e-3 + 1e-9),
}
across_channel = np.linspace(0.0, width, 101)
mean_pressure = lambda pressure, x: np.mean(
    pressure.evaluate(
        np.column_stack((np.full_like(across_channel, x), across_channel))
    )
)
pressure_drops = {
    key: mean_pressure(pressure, ends[key[0]][0])
    - mean_pressure(pressure, ends[key[0]][1])
    for key, (velocity, pressure) in results.items()
}
diodicity = {
    key[1:]: pressure_drops["reverse", *key[1:]] / pressure_drops[key]
    for key in pressure_drops
    if key[0] == "forward"
}
across = np.linspace(3e-3, 4e-3, 201)
loop_share = {
    key: abs(
        np.trapezoid(
            velocity.evaluate(np.column_stack((np.full_like(across, 6.8e-3), across)))[
                :, 0
            ],
            across,
        )
    )
    / (speed * width)
    for key, (velocity, pressure) in results.items()
}

print(
    "    Re   Δp forward [Pa]   Δp reverse [Pa]   diodicity   loop share forward   loop share reverse"
)
for reynolds in reynolds_numbers:
    print(
        f"{reynolds:6.2f} {pressure_drops['forward', reynolds, 8]:16.4f} {pressure_drops['reverse', reynolds, 8]:17.4f} {diodicity[reynolds, 8]:11.4f}"
        f" {100 * loop_share['forward', reynolds, 8]:18.1f} % {100 * loop_share['reverse', reynolds, 8]:18.1f} %"
    )
print(
    f"\nStokes limit: diodicity at Re = 0.01 differs from 1 by {abs(diodicity[0.01, 8] - 1):.1e}"
)
for reynolds in (100, 300):
    print(
        f"mesh study at Re = {reynolds}: diodicity "
        + ", ".join(
            f"{diodicity[reynolds, cells]:.4f} ({meshes[cells].cell_count} triangles)"
            for cells in (6, 8, 11)
        )
    )

# ## Figure
#
# The speed at $Re=500$ in both directions - forward the stream stays in the main channel, reverse
# the loop jet crosses it - and the diodicity over the Reynolds number.

fig = plots.figure(layout=[["forward"], ["reverse"], ["diodicity"]], size=(15, 13))
for direction in directions:
    plots.surface(
        results[direction, 500, 8][0],
        component=lambda u, position: np.linalg.norm(u, axis=1),
        limits=(0.0, 0.2),
        contours=6,
        subdivisions=2,
        label="speed [m/s]",
        ax=fig[direction],
        title=f"{direction.capitalize()} flow, Re = 500: {100 * loop_share[direction, 500, 8]:.0f} % of the flow takes the loop",
    )
plots.line(
    y={
        "simulation": (
            list(reynolds_numbers),
            [diodicity[reynolds, 8] for reynolds in reynolds_numbers],
        )
    },
    points={
        "coarser mesh": (
            [0.01, 100, 200, 300],
            [diodicity[reynolds, 6] for reynolds in (0.01, 100, 200, 300)],
        ),
        "finer mesh": (
            [0.01, 100, 200, 300],
            [diodicity[reynolds, 11] for reynolds in (0.01, 100, 200, 300)],
        ),
    },
    hlines={"Stokes limit Di = 1": 1.0},
    markers=True,
    xlabel="Reynolds number Re = U w / ν",
    ylabel="diodicity Δp_reverse / Δp_forward",
    title="Diodicity of a single-stage Tesla valve",
    legend="upper left",
    ax=fig["diodicity"],
)
plots.show()

# ## Going further
#
# Change the lean of the legs or the rounding of the loop in `tesla_valve_diodicity_geometry.py`,
# write the STEP file again and compare the diodicity: this is the design study behind optimized
# valves. Put several loops in a row, alternating above and below the channel, to multiply the
# effect; a pump with two such valves rectifies an oscillating flow.
