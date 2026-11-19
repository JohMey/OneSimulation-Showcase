"""OneSimulation example: temperature field in a glass cooling flange.
Requires the installed, activated OneSimulation runtime.

A hot fluid heats the bore of a glass pipe with three cooling flanges; natural convection that
depends on the angle and on the surface temperature cools the outside. The flange is read from a
STEP file; the 3D temperature field is checked by the energy balance and a mesh study with
Richardson extrapolation.
"""

# # Temperature field in a cooling flange
#
# Reaction fluids are cooled by passing them through a glass pipe with cooling flanges. The fluid
# heats the bore, the glass conducts the heat into the flanges, and room air removes it by
# **natural convection**. The pipe axis is horizontal ($x$), gravity points along $-y$.
#
# One half of the flange comes from the CAD file `cooling_flange_temperature.step`: a pipe of 16 mm
# bore and 4 mm wall with three fins, 4 mm wide and 10 mm tall, with rounded shoulders and tips,
# and a 12 mm connecting pipe with a 3 mm wall. The script `cooling_flange_temperature_geometry.py`
# builds it as a solid of revolution with the CAD objects of OneSimulation and names its faces.
#
# The glass obeys the stationary heat equation $\nabla\cdot(k\nabla T)=0$ with
#
# | face | condition |
# |---|---|
# | `bore` | forced convection, $h_i=15$ W/(m² K), fluid at 363 K |
# | `outer` | natural convection $q=C(\theta)\,(T-T_a)^{5/4}$ to air at 298 K |
# | `pipe_end`, `flange_end`, `symmetry` | no heat flux |
#
# The air-side law is the laminar correlation for a horizontal cylinder (B. Sundén, *Kompendium i
# värmeöverföring*, 2003): $h_a=(k_\mathrm{air}/D)\,f(\theta)\,Gr^{1/4}$ with the Grashof number
# $Gr=g\,(T-T_a)\,D^3/(T_a\nu^2)$ and a tabulated factor $f(\theta)$ of the angle from the lowest
# point. It makes the problem nonlinear and three-dimensional: the bottom of the flange cools
# better than the top.
#
# **Reference.** No closed-form solution exists. The check is the energy balance (heat through the
# bore equals heat to the air) and a three-level mesh study.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import Block, Geometric, arc, bezier, line, merge_blocks, revolve
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux,
    HeatConduction,
    HeatFlux,
    NonlinearHeatFlux,
)
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the half flange with its named faces. The mesh consists of hexahedra: curved blocks
# cover the meridian section (straight sides, Bezier fillets at the shoulders, circular arcs at the
# fin tips, written as (radius, axial position)), `merge_blocks` joins them and `revolve` sweeps the
# section by 180° about the pipe axis. Bound to the loaded part, the mesh takes its boundary names
# from the faces of the STEP file.

flange = load("cooling_flange_temperature.step")
faces = flange.boundaries
mm = 1e-3
bore, pipe_length, pipe_wall, wall, fin_height, fin_width = (
    8 * mm,
    12 * mm,
    3 * mm,
    4 * mm,
    10 * mm,
    4 * mm,
)


def flange_mesh(n):
    """Hexahedral mesh of the half flange with n cells across each block unit."""
    root, neck, shoulder, top = (
        bore + wall,
        -2 * pipe_length / 3,
        bore + wall + fin_width / 2,
        bore + wall + fin_height - fin_width / 2,
    )
    blocks = [
        Block(
            line((bore, -pipe_length), (bore, neck)),
            line((bore, neck), (bore + pipe_wall, neck)),
            line((bore + pipe_wall, -pipe_length), (bore + pipe_wall, neck)),
            line((bore, -pipe_length), (bore + pipe_wall, -pipe_length)),
            cells=(n, 2 * n),
        ),
        Block(
            line((bore, neck), (bore, 0)),
            line((bore, 0), (root, 0)),
            bezier(
                (bore + pipe_wall, neck),
                (root, -pipe_length / 3),
                (root, -pipe_length / 3),
                (root, 0),
            ),
            line((bore, neck), (bore + pipe_wall, neck)),
            cells=(2 * n, 2 * n),
        ),
    ]
    for fin in range(3):
        a, b, c, d = (
            2 * fin_width * fin + offset
            for offset in (0, fin_width / 2, 3 * fin_width / 2, 2 * fin_width)
        )
        tip = lambda start, end: arc((top, a + fin_width), fin_width / 2, start, end)
        blocks += [
            Block(
                line((bore, a), (bore, b)),
                line((bore, b), (shoulder, b)),
                bezier((root, a), (root, b), (shoulder, b)),
                line((bore, a), (root, a)),
                cells=(n, 2 * n),
            ),
            Block(
                line((bore, b), (bore, c)),
                line((bore, c), (shoulder, c)),
                line((shoulder, b), (shoulder, c)),
                line((bore, b), (shoulder, b)),
                cells=(2 * n, 2 * n),
            ),
            Block(
                line((bore, c), (bore, d)),
                line((bore, d), (root, d)),
                bezier((shoulder, c), (root, c), (root, d)),
                line((bore, c), (shoulder, c)),
                cells=(n, 2 * n),
            ),
            Block(
                line((shoulder, b), (shoulder, c)),
                line((shoulder, c), (top, c)),
                line((top, b), (top, c)),
                line((shoulder, b), (top, b)),
                cells=(2 * n, 3 * n),
                grading=(None, Geometric(3 * n, growth=1.15 ** (-2 / n))),
            ),
            Block(
                line((top, b), (top, c)),
                tip(np.pi / 2, np.pi / 4),
                tip(-np.pi / 4, np.pi / 4),
                tip(-np.pi / 2, -np.pi / 4),
                cells=(2 * n, n),
            ),
        ]
    return revolve(
        merge_blocks(blocks),
        4 * n,
        angle=np.pi,
        axis=(1.0, 0.0, 0.0),
        radial=(0.0, 1.0, 0.0),
        geometry=flange,
    )


levels = (2, 3, 4)
meshes = [flange_mesh(level) for level in levels]

view = plots.Camera(elevation=14, azimuth=-58, up="y")
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    flange,
    selection={"bore": faces.bore, "outer": faces.outer, "symmetry": faces.symmetry},
    labels=False,
    camera=view,
    title="Half flange with its named faces",
    ax=fig[0],
)
plots.mesh(
    meshes[0],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Coarsest mesh: {meshes[0].cell_count} hexahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Fused silica conducts with $k=1.38$ W/(m K); air at 298 K has $k=0.0262$ W/(m K) and
# $\nu=18\times10^{-6}$ m²/s. Newton's method handles the nonlinear air-side flux, which enters as
# a `NonlinearHeatFlux` with its derivative $\partial q/\partial T$.

diameter = 2 * (bore + wall + fin_height)
fluid_film, fluid_temperature, air_temperature = 15.0, 363.0, 298.0
glass = IsotropicSolid(
    density=2203.0,
    specific_heat_capacity=703.0,
    thermal_conductivity=1.38,
    youngs_modulus=73.1e9,
    poisson_ratio=0.17,
    thermal_expansion_coefficient=5.5e-7,
)
angles = np.radians([0, 90, 100, 110, 120, 130, 140, 150, 160, 180])
factors = np.array([0.48, 0.46, 0.45, 0.435, 0.42, 0.38, 0.35, 0.28, 0.22, 0.15])
grashof_per_kelvin = 9.81 * diameter**3 / (air_temperature * 18e-6**2)


def air_flux(temperature, y, z, power=1.25):
    """Natural-convection flux C(theta) (T - T_air)^(5/4) leaving the surface."""
    coefficient = (
        0.0262
        / diameter
        * np.interp(np.arctan2(np.abs(z), -y), angles, factors)
        * grashof_per_kelvin**0.25
    )
    return coefficient * np.maximum(temperature - air_temperature, 0.0) ** power


model = HeatConduction(domain=flange.volume, material=glass, heating=0.0)
problem = Problem(
    name="Glass cooling flange",
    geometry=flange,
    physics=(model,),
    boundary_conditions=(
        ConvectiveHeatFlux(
            boundary=faces.bore,
            physics=model,
            heat_transfer_coefficient=fluid_film,
            ambient_temperature=fluid_temperature,
        ),
        NonlinearHeatFlux(
            boundary=faces.outer,
            physics=model,
            flux=air_flux,
            derivative=lambda T, y, z: 1.25 * air_flux(T, y, z, power=0.25),
        ),
        HeatFlux(
            boundary=faces.pipe_end | faces.flange_end | faces.symmetry,
            physics=model,
            value=0.0,
        ),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=SparseDirect(),
    nonlinear_solver=Newton(),
    initial_guess={model.temperature: 330.0},
)
fields = [
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(model.temperature)
    for mesh in meshes
]
field = fields[-1]

# ## Results
#
# Heat rates of the full flange (twice the computed half). `convergence` returns the observed
# order and the Richardson estimate from the three meshes.

heat_in = [-2 * f.boundary_flux(faces.bore) for f in fields]
heat_out = [2 * f.boundary_flux(faces.outer) for f in fields]
study = convergence(heat_in, [1.0 / level for level in levels])
surface = field.sample(along=faces.outer, subdivisions=2)
print("level   hexahedra   heat through the bore [W]   heat to the air [W]")
for level, mesh, q_in, q_out in zip(levels, meshes, heat_in, heat_out):
    print(f"{level:5d} {mesh.cell_count:11d} {q_in:22.5f} {q_out:21.5f}")
print(f"energy imbalance on the finest mesh: {abs(heat_in[-1] / heat_out[-1] - 1):.1e}")
print(
    f"observed order {study.order:.2f}, Richardson estimate {study.extrapolated:.4f} W "
    f"(finest mesh {100 * (heat_in[-1] / study.extrapolated - 1):+.2f} %)"
)
print(
    f"surface temperature between {surface.values.min():.1f} K and {surface.values.max():.1f} K"
)

# ## Figure
#
# The surface temperature of the complete flange, the vertical section through the axis (top of
# the flange above, bottom below), the mesh study, and the local air-side film coefficient.

positions, surface_temperature = (
    np.asarray(surface.positions),
    np.asarray(surface.values).ravel(),
)
film = air_flux(surface_temperature, positions[:, 1], positions[:, 2]) / np.maximum(
    surface_temperature - air_temperature, 1e-9
)
limits = (
    surface_temperature.min(),
    float(np.asarray(field.sample(along=faces.bore).values).max()),
)
fig = plots.figure(
    layout=[["surface", "section", "study"], ["surface", "section", "film"]],
    size=(17, 6.2),
)
plots.surface(
    field,
    boundary=faces.outer | faces.pipe_end | faces.flange_end,
    symmetry={"planes": [((0, 0, 1), (0, 0, 0))]},
    limits=limits,
    subdivisions=3,
    camera=view,
    label="T [K]",
    title="Surface temperature (full flange)",
    ax=fig["surface"],
)
plots.surface(
    field,
    boundary=faces.symmetry,
    limits=limits,
    contours=10,
    subdivisions=3,
    label="T [K]",
    title="Vertical section through the axis",
    ax=fig["section"],
)
plots.line(
    levels,
    {"through the bore": heat_in, "to the air": heat_out},
    hlines={f"Richardson estimate {study.extrapolated:.4f} W": study.extrapolated},
    markers=True,
    xlabel="mesh level",
    ylabel="heat rate [W]",
    title="Mesh study of the cooling power",
    ax=fig["study"],
)
plots.line(
    points={
        "outer surface": (
            np.degrees(np.arctan2(np.abs(positions[:, 2]), -positions[:, 1])),
            film,
        )
    },
    xlabel="angle from the bottom [deg]",
    ylabel="film coefficient [W/(m² K)]",
    title="Local natural-convection coefficient",
    legend=False,
    ax=fig["film"],
)
plots.show()

# ## Going further
#
# Change the bore radius between 6 and 10 mm in `cooling_flange_temperature_geometry.py` and write
# the STEP file again: the bore-side film resistance dominates, so a wider bore raises the cooling
# power noticeably. Add a fourth fin, or replace glass by a metal, and compare how much of the
# temperature drop sits in the wall and in the two films.
