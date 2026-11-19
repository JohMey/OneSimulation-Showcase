"""OneSimulation example: heat transfer in a surface-mount package for a silicon chip.
Requires the installed, activated OneSimulation runtime.

A voltage regulator heats a circuit board; one lead of a 16-pin package picks the heat up from a
shared copper plane while the silicon die inside dissipates its own power. The 3D conduction
model gives the die temperature, which is compared with the published value.
"""

# # Heat transfer in a surface-mount package
#
# Circuit-board designers keep heat-sensitive chips away from heat sources. Here the footprint of a
# voltage regulator is held at 50 °C. It shares a copper ground plane with lead 1 of a small-outline
# package (SOIC-16), and a thin copper strip connects that lead to the silicon die, which
# dissipates 20 mW itself. How hot does the die get?
#
# | part (dimensions in mm) | conductivity [W/(m K)] |
# |---|---|
# | circuit board, $20\times10\times1$ | 0.3 |
# | copper ground plane $4\times4\times0.1$ and interconnect strip, 5 µm thick | 400 |
# | plastic package body $9.9\times3.9\times1.5$ with tapered side walls | 0.2 |
# | silicon die $1\times1\times0.1$ | 130 |
# | 16 aluminium gull-wing leads, $0.4\times0.2$ cross-section | 238 |
#
# The heat equation $\nabla\cdot(k\nabla T)+Q=0$ holds with $Q=2\times10^8$ W/m³ in the die. The
# regulator footprint is fixed at 50 °C; every other outer face is cooled by air at 30 °C with
# $h=50$ W/(m² K).
#
# **Reference.** The published maximum die temperature of this benchmark is 47.6 °C. The energy
# balance and a mesh study accompany the comparison.

from dataclasses import replace

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import ExplicitGeometry, VolumeBlock, merge_volume_blocks
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux,
    FixedTemperature,
    HeatConduction,
)
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()


# ## Parameters and materials
#
# Regions are axis-parallel boxes; `regionwise` assigns the conductivity and the heat source by
# region.

mm = 1e-3
h_air, T_air, T_regulator, T_published = 50.0, 303.15, 323.15, 47.6
pitch, lead_width, first_lead = 1.27, 0.4, -4.645
box = lambda x0, x1, y0, y1, z0, z1: Box(
    length=(x1 - x0) * mm,
    width=(y1 - y0) * mm,
    height=(z1 - z0) * mm,
    origin=(x0 * mm, y0 * mm, z0 * mm),
)
die = box(-0.5, 0.5, -0.5, 0.5, -0.05, 0.05)
strip = box(-4.645, -4.245, -1.95, 0.2, -0.0025, 0.0025) | box(
    -4.245, -0.5, -0.2, 0.2, -0.0025, 0.0025
)
ground_plane, board = box(-8.0, -4.0, -5.0, -1.0, -1.0, -0.9), box(
    -10.0, 10.0, -5.0, 5.0, -1.9, -0.9
)
in_leads = lambda points: (points[:, 2] > -0.9 * mm) & (
    np.abs(points[:, 1]) > 1.95 * mm
)
conductivity = regionwise(
    (
        (die, 130.0),
        (strip, 400.0),
        (ground_plane, 400.0),
        (in_leads, 238.0),
        (board, 0.3),
    ),
    default=0.2,
    unit="W/m/K",
)
materials = replace(
    IsotropicSolid(
        density=1000.0,
        specific_heat_capacity=1000.0,
        thermal_conductivity=1.0,
        youngs_modulus=1e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=0.0,
    ),
    thermal_conductivity=conductivity,
)
heating = regionwise(((die, 2e8),), default=0.0, unit="W/m^3")
assembly = ExplicitGeometry(
    dimension=3,
    boundary_names=("regulator", "air"),
    classifier=lambda point, normal: (
        "regulator"
        if abs(point[2] + 0.9 * mm) < 1e-9
        and point[0] < -8.0 * mm
        and point[1] < -1.0 * mm
        else "air"
    ),
)
faces = assembly.boundaries


# ## Geometry and mesh
#
# Three kinds of hexahedral blocks share their nodes: a graded block for the board, a tapered block
# for the package, and one swept block per lead that follows the gull-wing path through both
# 90° bends. All cells are quadratic, so the bends are curved.


def lead_path(u, t):
    """(y, z) in mm of a lead on the side y < 0: u in [0, 5] runs along stub, bend, leg, bend and foot,
    t in [0, 0.2] across the thickness."""
    u, t = np.broadcast_arrays(np.asarray(u, float), np.asarray(t, float))
    part, s = np.minimum(np.floor(u).astype(int), 4), u - np.minimum(
        np.floor(u).astype(int), 4
    )
    outer, inner = 0.14 + t, 0.34 - t
    y = np.select(
        [part == 0, part == 1, part == 2, part == 3],
        [
            -1.95 - 0.26 * s,
            -2.21 - outer * np.sin(0.5 * np.pi * s),
            -2.35 - t,
            -2.69 + inner * np.cos(0.5 * np.pi * s),
        ],
        -2.69 - 0.16 * s,
    )
    z = np.select(
        [part == 0, part == 1, part == 2, part == 3],
        [
            -0.1 + t,
            -0.24 + outer * np.cos(0.5 * np.pi * s),
            -0.24 - 0.32 * s,
            -0.56 - inner * np.sin(0.5 * np.pi * s),
        ],
        -0.9 + t,
    )
    return y, z


def package_mesh(level):
    """Board, tapered package and 16 swept leads as conforming 27-node hexahedra."""

    def nodes(breaks, size):
        """Grid nodes through all break points with cells no larger than size(midpoint)."""
        out = [breaks[0]]
        for a, b in zip(breaks[:-1], breaks[1:]):
            count = max(1, int(np.ceil((b - a) / size(0.5 * (a + b)) - 1e-9)))
            out.extend(a + (b - a) * np.arange(1, count + 1) / count)
        return np.array(out)

    h = 0.45 / level
    lead_edges = sorted(
        {first_lead + pitch * k + d for k in range(8) for d in (0.0, lead_width)}
    )
    near_die = lambda v: min(h, (0.08 + 0.5 * abs(abs(v) - 0.5)) / level)
    px = nodes(sorted({-4.95, -0.5, 0.5, 4.95, *lead_edges}), near_die)
    py = nodes([-1.95, -0.5, -0.2, 0.2, 0.5, 1.95], near_die)
    pz = nodes(
        [-0.75, -0.1, -0.05, -0.0025, 0.0025, 0.05, 0.1, 0.75],
        lambda z: (
            0.05 / level if abs(z) < 0.1 else (0.05 + 0.35 * (abs(z) - 0.1)) / level
        ),
    )
    in_lead = lambda x: any(
        first_lead + pitch * k - 1e-9 <= x <= first_lead + pitch * k + lead_width + 1e-9
        for k in range(8)
    )
    foot_cells = int(np.ceil(0.16 / h - 1e-9))
    foot = list(-2.69 - 0.16 * np.arange(foot_cells + 1) / foot_cells)
    bx = nodes(
        sorted({-10.0, -8.0, -6.5, -4.0, 7.0, 10.0, *(x for x in px if in_lead(x))}),
        lambda x: h * (1.0 + 0.8 * max(abs(x) - 4.95, 0.0)),
    )
    by = nodes(
        sorted({-5.0, *foot, -1.0, 0.0, *(-v for v in foot), 5.0}),
        lambda y: (
            h * (1.0 + 0.8 * max(abs(y) - 2.85, 0.0)) if abs(y) > 2.69 else 1.5 * h
        ),
    )
    bz = nodes([-1.9, -1.0, -0.9], lambda z: 0.3 / level)
    half_width = lambda z: 1.95 - 0.23 * np.clip((np.abs(z) - 0.1) / 0.65, 0.0, 1.0)
    blocks = [
        VolumeBlock(
            lambda u, v, w: (u, v, w), bx * mm, by * mm, bz * mm, region="board"
        ),
        VolumeBlock(
            lambda u, v, w: (u, v * half_width(w / mm) / 1.95, w),
            px * mm,
            py * mm,
            pz * mm,
            region="package",
        ),
    ]
    along = [
        max(1, int(np.ceil(length / (0.2 / level) - 1e-9)))
        for length in (0.26, 0.377, 0.32, 0.377)
    ] + [foot_cells]
    u = np.concatenate(
        [[0.0]] + [k + np.arange(1, n + 1) / n for k, n in enumerate(along)]
    )
    across = pz[np.abs(pz) <= 0.1 + 1e-9] + 0.1
    for k in range(8):
        x0 = first_lead + pitch * k
        lead_x = px[(px >= x0 - 1e-9) & (px <= x0 + lead_width + 1e-9)]
        for side in (-1, 1):
            lead = lambda u, x, t, side=side: (
                x,
                side * lead_path(u, t)[0] * mm,
                lead_path(u, t)[1] * mm,
            )
            blocks.append(VolumeBlock(lead, u, lead_x * mm, across, region="leads"))
    return merge_volume_blocks(blocks, assembly, order=2)


# ## Model and solution
#
# Quadratic Lagrange elements on three meshes.


def solve(level):
    """Temperature field on the mesh of the given refinement level."""
    mesh = package_mesh(level)
    model = HeatConduction(
        domain=assembly.domain(), material=materials, heating=heating
    )
    problem = Problem(
        name="SOIC-16 package next to a voltage regulator",
        geometry=assembly,
        physics=(model,),
        boundary_conditions=(
            FixedTemperature(
                boundary=faces.regulator, physics=model, value=T_regulator
            ),
            ConvectiveHeatFlux(
                boundary=faces.air,
                physics=model,
                heat_transfer_coefficient=h_air,
                ambient_temperature=T_air,
            ),
        ),
    )
    numerics = Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
    )
    return (
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .solve_stationary()
        .field(model.temperature),
        mesh,
    )


levels = (0.5, 0.7, 1.0)
fields, meshes = zip(*(solve(level) for level in levels))
field = fields[-1]


# ## Results

g = np.linspace(-0.5, 0.5, 21) * mm
die_points = (
    np.array(np.meshgrid(g, g, np.linspace(-0.05, 0.05, 5) * mm, indexing="ij"))
    .reshape(3, -1)
    .T
)
T_die = [float(np.max(f.evaluate(die_points))) - 273.15 for f in fields]
print(
    "hexahedra   maximum die temperature [°C]   deviation from the published 47.6 °C [K]"
)
for mesh, T in zip(meshes, T_die):
    print(f"{mesh.cell_count:9d} {T:22.3f} {T - T_published:+31.3f}")
heat_in = -field.boundary_flux(faces.regulator) + 2e8 * 1.0 * 1.0 * 0.1 * mm**3
heat_out = field.boundary_flux(faces.air)
print(
    f"energy balance: regulator and die supply {1e3 * heat_in:.3f} mW, the air removes {1e3 * heat_out:.3f} mW "
    f"(difference {abs(heat_in / heat_out - 1):.1e})"
)


# ## Figure
#
# The surface temperature, the section through the copper strip plane ($z=0$), the vertical
# section through lead 1, and the mesh study of the die temperature.

view = dict(
    component=lambda T, position: T - 273.15,
    limits=(30.0, 50.0),
    subdivisions=2,
    label="T [°C]",
)
fig = plots.figure(2, 2, size=(17, 11))
plots.surface(
    field,
    boundary=faces.air | faces.regulator,
    camera=plots.Camera(elevation=30, azimuth=-60, distance=0.8),
    title="Surface temperature",
    ax=fig[0],
    **view,
)
plots.section(
    field,
    plane=Plane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
    contours=20,
    title="Plane z = 0: die, interconnect strip and lead stubs",
    xlimits=(-6 * mm, 6 * mm),
    ylimits=(-2.4 * mm, 2.4 * mm),
    ax=fig[1],
    **view,
)
plots.section(
    field,
    plane=Plane(origin=(-4.445 * mm, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
    contours=20,
    title="Section through lead 1",
    xlimits=(-5 * mm, 0.6 * mm),
    ylimits=(-1.9 * mm, 0.8 * mm),
    ax=fig[2],
    **view,
)
plots.line(
    [mesh.cell_count for mesh in meshes],
    {"3D model": T_die},
    hlines={"published value": T_published},
    markers=True,
    xlabel="hexahedra",
    ylabel="maximum die temperature [°C]",
    title="Mesh study",
    ax=fig[3],
)
plots.show()


# ## Going further
#
# Move the heat path: connect the strip to a lead on the far side of the package, or widen the
# ground plane, and compare the die temperature. Without the die power (`heating = 0`) the
# package only conducts regulator heat to the air and the die settles between 30 and 50 °C.
