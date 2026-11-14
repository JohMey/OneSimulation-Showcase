"""OneSimulation example: temperature field in a glass cooling flange.
Requires the installed, activated OneSimulation runtime.

A hot fluid heats the bore of a glass pipe with three cooling flanges; natural convection that
depends on the angle and on the surface temperature cools the outside. The 3D temperature field
is checked by the energy balance and a mesh study with Richardson extrapolation.
"""


# # Temperature field in a cooling flange
#
# Reaction fluids are cooled by passing them through a glass pipe with cooling flanges. The fluid
# heats the bore, the glass conducts the heat into the flanges, and room air removes it by
# **natural convection**. The pipe axis is horizontal ($x$), gravity points along $-y$.
#
# The glass obeys the stationary heat equation $\nabla\cdot(k\nabla T)=0$ with
#
# | face | condition |
# |---|---|
# | bore | forced convection, $h_i=15$ W/(m² K), fluid at 363 K |
# | all outer surfaces | natural convection $q=C(\theta)\,(T-T_a)^{5/4}$ to air at 298 K |
# | pipe end faces, vertical symmetry plane | no heat flux |
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

from onesim.materials import IsotropicSolid
from onesim.meshing import Block, Geometric, arc, bezier, line, merge_blocks, revolve
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux, NonlinearHeatFlux
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# A pipe of 16 mm bore and 4 mm wall carries three flanges, 4 mm wide and 10 mm tall, with
# rounded shoulders and tips; a 12 mm connecting pipe with a 3 mm wall joins on one side.
# Fused silica conducts with $k=1.38$ W/(m K); air at 298 K has $k=0.0262$ W/(m K) and
# $\nu=18\times10^{-6}$ m²/s.

mm = 1e-3
r_in, l_pipe, t_pipe, t_wall, h_flange, w = 8 * mm, 12 * mm, 3 * mm, 4 * mm, 10 * mm, 4 * mm
D = 2 * (r_in + t_wall + h_flange)
h_in, T_in, T_air = 15.0, 363.0, 298.0
k_air, nu_air, gravity = 0.0262, 18e-6, 9.81
glass = IsotropicSolid(density=2203.0, specific_heat_capacity=703.0, thermal_conductivity=1.38,
                       youngs_modulus=73.1e9, poisson_ratio=0.17, thermal_expansion_coefficient=5.5e-7)
angles = np.radians([0, 90, 100, 110, 120, 130, 140, 150, 160, 180])
factors = np.array([0.48, 0.46, 0.45, 0.435, 0.42, 0.38, 0.35, 0.28, 0.22, 0.15])
grashof_per_kelvin = gravity * D**3 / (T_air * nu_air**2)


def air_flux(temperature, y, z, power=1.25):
    """Natural-convection flux C(theta) (T - T_air)^(5/4) leaving the surface."""
    theta = np.arctan2(np.abs(z), -y)
    coefficient = k_air / D * np.interp(theta, angles, factors) * grashof_per_kelvin**0.25
    return coefficient * np.maximum(temperature - T_air, 0.0) ** power


# ## Geometry and mesh
#
# The flange is a solid of revolution. Curved blocks cover its meridian section: straight sides,
# Bezier fillets at the shoulders and circular arcs at the fin tips. `merge_blocks` joins them,
# and `revolve` sweeps the section by 180° about the pipe axis into hexahedra. The block sides
# name the boundaries. Section points are written as (radius, axial position).

def flange_mesh(level):
    """Hexahedral mesh of the half flange with `level` cells across each block unit."""
    n = level
    y0, y1, x1 = r_in, r_in + t_wall, -2 * l_pipe / 3
    blocks = [
        Block(line((y0, -l_pipe), (y0, x1)), line((y0, x1), (y0 + t_pipe, x1)),
              line((y0 + t_pipe, -l_pipe), (y0 + t_pipe, x1)), line((y0, -l_pipe), (y0 + t_pipe, -l_pipe)),
              cells=(n, 2 * n), names=("bore", None, "outer", "pipe_end")),
        Block(line((y0, x1), (y0, 0)), line((y0, 0), (y1, 0)),
              bezier((y0 + t_pipe, x1), (y1, -l_pipe / 3), (y1, -l_pipe / 3), (y1, 0)), line((y0, x1), (y0 + t_pipe, x1)),
              cells=(2 * n, 2 * n), names=("bore", None, "outer", None)),
    ]
    for fin in range(3):
        a, b, c, d = (2 * w * fin + offset for offset in (0, w / 2, 3 * w / 2, 2 * w))
        shoulder, top = y1 + w / 2, y1 + h_flange - w / 2
        tip = lambda start, end: arc((top, a + w), w / 2, start, end)
        blocks += [
            Block(line((y0, a), (y0, b)), line((y0, b), (shoulder, b)), bezier((y1, a), (y1, b), (shoulder, b)),
                  line((y0, a), (y1, a)), cells=(n, 2 * n), names=("bore", None, "outer", None)),
            Block(line((y0, b), (y0, c)), line((y0, c), (shoulder, c)), line((shoulder, b), (shoulder, c)),
                  line((y0, b), (shoulder, b)), cells=(2 * n, 2 * n), names=("bore", None, None, None)),
            Block(line((y0, c), (y0, d)), line((y0, d), (y1, d)), bezier((shoulder, c), (y1, c), (y1, d)),
                  line((y0, c), (shoulder, c)), cells=(n, 2 * n),
                  names=("bore", "flange_end" if fin == 2 else None, "outer", None)),
            Block(line((shoulder, b), (shoulder, c)), line((shoulder, c), (top, c)), line((top, b), (top, c)),
                  line((shoulder, b), (top, b)), cells=(2 * n, 3 * n),
                  grading=(None, Geometric(3 * n, growth=1.15 ** (-2 / n))), names=(None, "outer", None, "outer")),
            Block(line((top, b), (top, c)), tip(np.pi / 2, np.pi / 4), tip(-np.pi / 4, np.pi / 4), tip(-np.pi / 2, -np.pi / 4),
                  cells=(2 * n, n), names=(None, "outer", "outer", "outer")),
        ]
    return revolve(merge_blocks(blocks), 4 * n, angle=np.pi, axis=(1.0, 0.0, 0.0), radial=(0.0, 1.0, 0.0),
                   names=("symmetry", "symmetry"))


# ## Model and solution
#
# Quadratic Lagrange elements; Newton's method handles the nonlinear air-side flux, which enters
# as a `NonlinearHeatFlux` with its derivative $\partial q/\partial T$.

def solve(level):
    """Temperature field of the half flange on the mesh of the given level."""
    mesh = flange_mesh(level)
    b = mesh.geometry.boundaries
    model = HeatConduction(domain=mesh.geometry.domain(), material=glass, heating=0.0)
    problem = Problem(name="Glass cooling flange", geometry=mesh.geometry, physics=(model,), boundary_conditions=(
        ConvectiveHeatFlux(boundary=b.bore, physics=model, heat_transfer_coefficient=h_in, ambient_temperature=T_in),
        NonlinearHeatFlux(boundary=b.outer, physics=model, flux=air_flux,
                          derivative=lambda T, y, z: 1.25 * air_flux(T, y, z, power=0.25)),
        HeatFlux(boundary=b.pipe_end | b.flange_end | b.symmetry, physics=model, value=0.0)))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect(),
                        nonlinear_solver=Newton(), initial_guess={model.temperature: 330.0})
    return Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.temperature), mesh


levels = (2, 3, 4)
fields, meshes = zip(*(solve(level) for level in levels))
field, b = fields[-1], meshes[-1].geometry.boundaries


# ## Results
#
# Heat rates of the full flange (twice the computed half). `convergence` returns the observed
# order and the Richardson estimate from the three meshes.

Q_in = [-2 * f.boundary_flux(m.geometry.boundaries.bore) for f, m in zip(fields, meshes)]
Q_out = [2 * f.boundary_flux(m.geometry.boundaries.outer) for f, m in zip(fields, meshes)]
study = convergence(Q_in, [1.0 / level for level in levels])
surface = field.sample(along=b.outer, subdivisions=2)
print("level   hexahedra   heat through the bore [W]   heat to the air [W]")
for level, mesh, q_in, q_out in zip(levels, meshes, Q_in, Q_out):
    print(f"{level:5d} {mesh.cell_count:11d} {q_in:22.5f} {q_out:21.5f}")
print(f"energy imbalance on the finest mesh: {abs(Q_in[-1] / Q_out[-1] - 1):.1e}")
print(f"observed order {study.order:.2f}, Richardson estimate {study.extrapolated:.4f} W "
      f"(finest mesh {100 * (Q_in[-1] / study.extrapolated - 1):+.2f} %)")
print(f"surface temperature between {surface.values.min():.1f} K and {surface.values.max():.1f} K")


# ## Figure
#
# The surface temperature of the complete flange, the vertical section through the axis (top of
# the flange above, bottom below), the mesh study, and the local air-side film coefficient.

positions, T_surface = np.asarray(surface.positions), np.asarray(surface.values).ravel()
film = air_flux(T_surface, positions[:, 1], positions[:, 2]) / np.maximum(T_surface - T_air, 1e-9)
limits = (T_surface.min(), float(np.asarray(field.sample(along=b.bore).values).max()))
fig = plots.figure(layout=[["surface", "section", "study"], ["surface", "section", "film"]], size=(17, 6.2))
plots.surface(field, boundary=b.outer | b.pipe_end | b.flange_end, symmetry={"planes": [((0, 0, 1), (0, 0, 0))]},
              limits=limits, subdivisions=3, camera=plots.Camera(elevation=14, azimuth=-58, up="y"), label="T [K]",
              title="Surface temperature (full flange)", ax=fig["surface"])
plots.surface(field, boundary=b.symmetry, limits=limits, contours=10, subdivisions=3, label="T [K]",
              title="Vertical section through the axis", ax=fig["section"])
plots.line(levels, {"through the bore": Q_in, "to the air": Q_out},
           hlines={f"Richardson estimate {study.extrapolated:.4f} W": study.extrapolated}, markers=True,
           xlabel="mesh level", ylabel="heat rate [W]", title="Mesh study of the cooling power", ax=fig["study"])
plots.line(points={"outer surface": (np.degrees(np.arctan2(np.abs(positions[:, 2]), -positions[:, 1])), film)},
           xlabel="angle from the bottom [deg]", ylabel="film coefficient [W/(m² K)]",
           title="Local natural-convection coefficient", legend=False, ax=fig["film"])
plots.show()


# ## Going further
#
# Sweep the bore radius between 6 and 10 mm: the bore-side film resistance dominates, so a wider
# bore raises the cooling power noticeably. Add a fourth flange, or replace glass by a metal, and
# compare how much of the temperature drop sits in the wall and in the two films.
