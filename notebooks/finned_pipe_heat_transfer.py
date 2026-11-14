"""OneSimulation example: heat transfer through a finned pipe.
Requires the installed, activated OneSimulation runtime.

Warm water flows through a copper pipe with a grooved bore; air cools thin annular fins on the
outside. The fin efficiency of the 3D model is compared with the Bessel-function solution of the
one-dimensional fin equation, together with the energy balance and a mesh study.
"""


# # Heat transfer through a finned pipe
#
# Warm water flows through a copper pipe, air blows across the outside. Thin annular **disk fins**
# enlarge the air-side area, twelve axial **grooves** the water-side area. The arrangement repeats
# with the fin pitch and every 30° around the axis, so one fin and a 15° wedge between two mirror
# planes represent the whole pipe.
#
# The copper obeys the stationary heat equation $\nabla\cdot(k\nabla T)=0$ with
#
# | face | condition |
# |---|---|
# | grooved bore | convection to the water, $-k\,\partial_n T=h_w\,(T-T_w)$ |
# | fin and bare pipe surface | convection to the air, $-k\,\partial_n T=h_a\,(T-T_a)$ |
# | end planes and wedge planes | no heat flux (periodicity and mirror symmetry) |
#
# **Reference.** A thin annular fin between the radii $r_1$ and $r_2$ follows the fin equation with
# the Bessel-function solution $\theta(r)=A\,I_0(mr)+B\,K_0(mr)$, $m=\sqrt{2h/(kt)}$ (K. A. Gardner,
# *Trans. ASME* 67, 1945). With the base temperature fixed and a convective tip it gives the
# temperature along the fin and the **fin efficiency**, the heat the fin gives off relative to a
# fin at base temperature throughout.

import numpy as np
from scipy.special import i0, i1, k0, k1

from onesim.cad.d2 import Plane
from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, mapped_mesh, merge_blocks, rectangle_block, revolve
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.results.comparisons import convergence
from onesim.results.integration import measure
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# Pipe: 20 mm bore, 2 mm wall, grooves 0.8 mm deep. Fins: 0.6 mm thick, 35 mm outer radius, 5 mm
# pitch. Water at 30 °C flows at 0.5 m/s (Dittus-Boelter gives the film coefficient), air at 20 °C
# is taken with 80 W/(m² K).

mm = 1e-3
r_bore, r_wall, r_fin, t_fin, pitch = 10 * mm, 12 * mm, 35 * mm, 0.6 * mm, 5 * mm
n_grooves, groove_depth = 12, 0.8 * mm
wedge = np.pi / n_grooves
T_water, T_air, h_air = 303.15, 293.15, 80.0
reynolds = 0.5 * 2 * r_bore / 0.801e-6
h_water = 0.023 * reynolds**0.8 * 5.42**0.4 * 0.615 / (2 * r_bore)
copper = IsotropicSolid(density=8933.0, specific_heat_capacity=385.0, thermal_conductivity=401.0,
                        youngs_modulus=110e9, poisson_ratio=0.34, thermal_expansion_coefficient=17e-6)


def annular_fin(r1, r2, t, h, k):
    """Efficiency and temperature profile theta(r)/theta_base of an annular fin with a convective tip."""
    m = np.sqrt(2 * h / (k * t))
    A, B = np.linalg.solve([[i0(m * r1), k0(m * r1)],
                            [k * m * i1(m * r2) + h * i0(m * r2), -k * m * k1(m * r2) + h * k0(m * r2)]], [1.0, 0.0])
    base_flux = -k * m * (A * i1(m * r1) - B * k1(m * r1))
    surface = 2 * np.pi * (r2**2 - r1**2) + 2 * np.pi * r2 * t
    return base_flux * 2 * np.pi * r1 * t / (h * surface), lambda r: A * i0(m * r) + B * k0(m * r)


# ## Geometry and mesh
#
# Four graded rectangular blocks form the section (wall below, in and above the fin, and the fin).
# `revolve` sweeps it over the 15° wedge, and `mapped_mesh` moves the bore inwards where a groove
# ridge sits, $r_\mathrm{bore}(\theta)=r_0+\tfrac14 d\,(1+\cos 12\theta)^2$.

def finned_pipe_mesh(level):
    """Tetrahedral mesh of one fin pitch and a 15 degree wedge."""
    n_gap, n_thick, n_wall, n_radial = 3 * level, 2 * level, 3 * level, 10 * level
    grow = lambda count, factor, toward: Geometric(count, growth=factor ** (2 / level), toward=toward)
    wall, gap = grow(n_wall, 1.15, "end"), pitch / 2 - t_fin / 2
    section = merge_blocks([
        rectangle_block((r_bore, -pitch / 2), r_wall - r_bore, gap, (n_wall, n_gap), (wall, grow(n_gap, 1.25, "end")),
                        names=("end_planes", "pipe_surface", None, "bore")),
        rectangle_block((r_bore, -t_fin / 2), r_wall - r_bore, t_fin, (n_wall, n_thick), (wall, None),
                        names=(None, None, None, "bore")),
        rectangle_block((r_bore, t_fin / 2), r_wall - r_bore, gap, (n_wall, n_gap), (wall, grow(n_gap, 1.25, "start")),
                        names=(None, "pipe_surface", "end_planes", "bore")),
        rectangle_block((r_wall, -t_fin / 2), r_fin - r_wall, t_fin, (n_radial, n_thick), (grow(n_radial, 1.12, "start"), None),
                        names=("fin_surface", "fin_surface", "fin_surface", None)),
    ]).triangulated()

    def grooved(points):
        r, theta = np.hypot(points[:, 1], points[:, 2]), np.arctan2(points[:, 2], points[:, 1])
        r_inner = r_bore + groove_depth * 0.25 * (1.0 + np.cos(n_grooves * theta)) ** 2
        r_new = np.where(r <= r_wall + 1e-12, r_inner + (r - r_bore) * (r_wall - r_inner) / (r_wall - r_bore), r)
        points[:, 1], points[:, 2] = r_new * np.cos(theta), r_new * np.sin(theta)
        return points

    wedge_mesh = revolve(section, 3 * level, angle=wedge, axis=(1.0, 0.0, 0.0), radial=(0.0, 1.0, 0.0),
                         names=("wedge_planes", "wedge_planes"))
    return mapped_mesh(wedge_mesh, grooved)


# ## Model and solution
#
# Quadratic Lagrange elements on three meshes with cell counts in the ratio 2 : 3 : 4.

def solve(level):
    """Temperature field of the periodic element on the mesh of the given level."""
    mesh = finned_pipe_mesh(level)
    b = mesh.geometry.boundaries
    model = HeatConduction(domain=mesh.geometry.domain(), material=copper, heating=0.0)
    problem = Problem(name="Finned pipe periodic element", geometry=mesh.geometry, physics=(model,), boundary_conditions=(
        ConvectiveHeatFlux(boundary=b.bore, physics=model, heat_transfer_coefficient=h_water, ambient_temperature=T_water),
        ConvectiveHeatFlux(boundary=b.fin_surface | b.pipe_surface, physics=model, heat_transfer_coefficient=h_air,
                           ambient_temperature=T_air),
        HeatFlux(boundary=b.end_planes | b.wedge_planes, physics=model, value=0.0)))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    return Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.temperature), mesh


levels = (2, 3, 4)
fields, meshes = zip(*(solve(level) for level in levels))
field, b = fields[-1], meshes[-1].geometry.boundaries


# ## Results
#
# The fin efficiency is the heat leaving the fin divided by $h_a A_\mathrm{fin}(T_b-T_a)$, with
# the base temperature $T_b$ averaged over the fin root. Heat rates are given for one pitch of the
# full circumference (24 wedges).

x_root, angle_root = np.meshgrid((np.arange(16) + 0.5) / 16 * t_fin - t_fin / 2, (np.arange(48) + 0.5) / 48 * wedge)
root = np.column_stack((x_root.ravel(), r_wall * np.cos(angle_root.ravel()), r_wall * np.sin(angle_root.ravel())))
T_base = [float(np.mean(f.evaluate(root))) for f in fields]
efficiency = [f.boundary_flux(m.geometry.boundaries.fin_surface)
              / (h_air * measure(f, boundary=m.geometry.boundaries.fin_surface) * (T - T_air))
              for f, m, T in zip(fields, meshes, T_base)]
eta_exact, fin_profile = annular_fin(r_wall, r_fin, t_fin, h_air, copper.thermal_conductivity)
study = convergence(efficiency, [1.0 / level for level in levels])
Q_water, Q_air = -24 * field.boundary_flux(b.bore), 24 * field.boundary_flux(b.fin_surface | b.pipe_surface)
print("level   tetrahedra   fin efficiency")
for level, mesh, eta in zip(levels, meshes, efficiency):
    print(f"{level:5d} {mesh.cell_count:12d} {eta:16.5f}")
print(f"Richardson estimate {study.extrapolated:.5f} (order {study.order:.2f}),  Bessel solution {eta_exact:.5f},  "
      f"deviation of the finest mesh {100 * (efficiency[-1] / eta_exact - 1):+.3f} %")
print(f"heat from the water {Q_water:.5f} W, heat to the air {Q_air:.5f} W, imbalance {abs(Q_water / Q_air - 1):.1e}")


# ## Figure
#
# The surface temperature of four pitches (completed by the mirror planes and the 12-fold
# rotation), the section through the fin mid-plane near the grooved bore, the temperature along
# the fin against the Bessel solution, and the mesh study.

surface_T = np.asarray(field.sample(along=b.fin_surface | b.bore).values)
limits = (float(surface_T.min()), float(surface_T.max()))
mirror, repeat = [((0, 0, 1), (0, 0, 0))], [((1, 0, 0), (x, 0, 0)) for x in (pitch / 2, -pitch / 2, 1.5 * pitch)]
r_line = np.linspace(r_wall, r_fin * (1 - 1e-6), 80)
along_fin = {f"3D model, {np.degrees(angle):.0f}° from a groove ridge":
             np.column_stack((np.zeros_like(r_line), r_line * np.cos(angle), r_line * np.sin(angle))) for angle in (0.0, wedge)}
fig = plots.figure(layout=[["surface", "surface", "cut", "fin"], ["surface", "surface", "cut", "study"]], size=(17, 6.4))
plots.surface(field, boundary=b.fin_surface | b.pipe_surface | b.bore,
              symmetry={"planes": mirror + repeat, "sector": ((1, 0, 0), n_grooves)}, limits=limits, subdivisions=1,
              camera=plots.Camera(elevation=25, azimuth=-62, distance=0.85), label="T [K]",
              title="Surface temperature, four pitches", ax=fig["surface"])
plots.section(field, plane=Plane(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
              symmetry={"planes": mirror, "sector": ((1, 0, 0), n_grooves)}, subdivisions=3, contours=12, limits=limits,
              label="T [K]", title="Fin mid-plane near the grooved bore", xlimits=(-14 * mm, 14 * mm),
              ylimits=(-14 * mm, 14 * mm), ax=fig["cut"])
plots.line(points={label: (r_line[::4] / mm, ((np.asarray(field.evaluate(points)).ravel() - T_air) / (T_base[-1] - T_air))[::4])
                   for label, points in along_fin.items()},
           reference={"Bessel solution": (r_line / mm, fin_profile(r_line))}, xlabel="radius [mm]",
           ylabel="(T - T_air) / (T_base - T_air)", title="Temperature along the fin", legend="lower left", ax=fig["fin"])
plots.line(levels, {"3D model": efficiency}, hlines={"Bessel solution": eta_exact}, markers=True, xlabel="mesh level",
           ylabel="fin efficiency", title="Mesh study", ax=fig["study"])
plots.show()


# ## Going further
#
# Halve the fin thickness or double the air-side coefficient and watch the fin efficiency fall:
# the fin tip then runs colder and adds less. Remove the grooves (`groove_depth = 0`) to see how
# much of the total resistance sits on the water side.
