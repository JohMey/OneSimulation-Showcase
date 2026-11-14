"""OneSimulation example: temperature field in a glass cooling flange.
Requires the installed, activated OneSimulation runtime.

A hot process fluid flows through a glass pipe with three cooling flanges. Forced
convection heats the bore, angle- and temperature-dependent natural convection cools the
outside. The 3D temperature field is checked by the global energy balance, a three-level
mesh study with Richardson extrapolation, and a 1D radial-wall estimate near the bore.
"""


# # Temperature field in a cooling flange
#
# Reaction fluids are often cooled by passing them through a glass pipe with cooling
# flanges: the fluid heats the bore, the heat is conducted through the glass into the
# flanges, and the surrounding room air carries it away by **natural convection**. The
# pipe axis is horizontal ($x$), gravity points along $-y$.
#
# The glass is described by the stationary heat equation
#
# $$\nabla\cdot(k\nabla T)=0 .$$
#
# The bore sees forced convection to the process fluid with a constant coefficient,
# $-k\,\partial_n T=h_i\,(T-T_i)$. On all outer surfaces the free-convection film
# coefficient is taken from the laminar correlation for a horizontal cylinder of
# diameter $D$ (the flange outer diameter), with a local factor $f(\theta)$ that depends
# on the angle $\theta$ measured from the lowest point of the cylinder (B. Sundén,
# *Kompendium i värmeöverföring*, Sec. 10-3, Lund Institute of Technology, 2003):
#
# $$h_a=\frac{k_\mathrm{air}}{D}\,f(\theta)\,Gr^{1/4},\qquad
#   Gr=\frac{g\,\beta\,(T-T_a)\,D^3}{\nu^2},\qquad \beta=1/T_a ,$$
#
# so the outward flux $h_a(T-T_a)\propto f(\theta)\,(T-T_a)^{5/4}$ is nonlinear in the
# surface temperature and varies around the circumference. The temperature therefore
# depends on all three coordinates: axially (pipe, flanges), radially (wall, fins) and
# around the axis (stronger cooling at the bottom than at the top).
#
# OneSimulation solves the problem with quadratic (Q2) Lagrange finite elements on a
# block-structured hexahedral mesh of the half flange $z\ge 0$; the vertical plane through the
# axis is a plane of symmetry because gravity lies in it. Boundary conditions on every
# face:
#
# | face | condition |
# |---|---|
# | bore | forced convection to the process fluid, $h_i=15$ W/(m² K), $T_i=363$ K |
# | all outer surfaces (pipe, flange shoulders, fins, fin tips) | natural convection $h_a(\theta, T)$ to air at $T_a=298$ K |
# | the two annular end faces of the pipe | thermally insulated (zero flux) |
# | vertical symmetry plane $z=0$ | symmetry (zero flux) |
#
# The nonlinear, angle-dependent air-side law is a `NonlinearHeatFlux`: the outward flux
# is given directly as a function of the surface temperature and position, together with
# its slope $\partial q/\partial T$, and the solver applies it with Newton steps.
#
# **Reference.** There is no closed-form solution for this geometry. The notebook
# uses (i) the global energy balance (heat entering through the bore equals heat leaving
# to the air), (ii) a three-level mesh-convergence study of the total heat rate and of
# the peak and minimum temperatures with observed order and Richardson extrapolation,
# and (iii) as a sanity bound only, the 1D radial conduction estimate through the pipe
# wall near the insulated pipe end, where the field is closest to one-dimensional.

import time
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
# Flange geometry: a glass pipe of 16 mm bore with a 4 mm wall carries three flanges,
# 4 mm wide and 10 mm tall, with filleted shoulders and rounded tips; a 12 mm long
# connecting pipe with a 3 mm wall joins on one side and widens smoothly to the flange
# wall. Fused silica: $k=1.38$ W/(m K) (Incropera et al., *Fundamentals of Heat and Mass
# Transfer*, Table A.3, 300 K). Air at 298 K: $k_\mathrm{air}=0.0262$ W/(m K),
# $\nu=18\times10^{-6}$ m²/s.

mm = 1e-3
r_in = 8.0 * mm
l_pipe = 12.0 * mm
t_pipe = 3.0 * mm
t_wall = 4.0 * mm
h_flange = 10.0 * mm
w_flange = 4.0 * mm
n_flanges = 3
D = 2 * (r_in + t_wall + h_flange)

k_glass = 1.38
h_in, T_in = 15.0, 363.0
T_air = 298.0
k_air, nu_air, g = 0.0262, 18e-6, 9.81
theta_table = np.radians([0, 90, 100, 110, 120, 130, 140, 150, 160, 180])
f_table = np.array([0.48, 0.46, 0.45, 0.435, 0.42, 0.38, 0.35, 0.28, 0.22, 0.15])
grashof_per_kelvin = g * (1.0 / T_air) * D**3 / nu_air**2
print(f"outer diameter D = {D / mm:.0f} mm,  Gr = {grashof_per_kelvin:.3e} * (T - T_air)/K")


def incidence(y, z):
    """Angle from the lowest point of the horizontal cylinder (gravity along -y)."""
    return np.arctan2(np.abs(z), -y)


def air_prefactor(y, z):
    """C(theta) in q = C(theta) (T - T_air)^(5/4)."""
    return k_air / D * np.interp(incidence(y, z), theta_table, f_table) * grashof_per_kelvin**0.25


def air_flux(temperature, y, z):
    return air_prefactor(y, z) * np.maximum(temperature - T_air, 0.0) ** 1.25


def air_flux_slope(temperature, y, z):
    return 1.25 * air_prefactor(y, z) * np.maximum(temperature - T_air, 0.0) ** 0.25


# ## Geometry and mesh
#
# The flange is a solid of revolution. Its meridian section is covered by curved
# quadrilateral blocks: straight sides, Bezier fillets at the flange shoulders and the
# pipe transition, and circular arcs at the fin tips. `merge_blocks` joins them into one
# planar quadrilateral mesh, and `revolve` sweeps it by $180^\circ$ about the pipe axis
# $x$ into hexahedra. (Fillets that end tangent to a block side would give curved
# 27-node cells a vanishing Jacobian at that corner, so the cells are trilinear; the
# temperature is still approximated quadratically.)
# The block sides name the boundaries (bore, pipe end, flange end, outer surface), the
# two ends of the half revolution form the symmetry plane.
#
# `revolve` takes the section in (radius, axial position) coordinates; the helper `P`
# writes a section point given as (x, r) in that order.

def P(x, r):
    return (r, x)


def section_blocks(level):
    """Curved blocks of the meridian section; neighbouring blocks share their sides."""
    L = level
    fin = Geometric(3 * L, growth=1.15 ** (-2 / L))
    y0, y1 = r_in, r_in + t_wall
    x1 = -2 * l_pipe / 3
    blocks = [
        Block(line(P(-l_pipe, y0), P(x1, y0)), line(P(x1, y0), P(x1, y0 + t_pipe)),
              line(P(-l_pipe, y0 + t_pipe), P(x1, y0 + t_pipe)), line(P(-l_pipe, y0), P(-l_pipe, y0 + t_pipe)),
              cells=(L, 2 * L), names=("bore", None, "outer", "pipe_end")),
        Block(line(P(x1, y0), P(0, y0)), line(P(0, y0), P(0, y1)),
              bezier(P(x1, y0 + t_pipe), P(-l_pipe / 3, y1), P(-l_pipe / 3, y1), P(0, y1)),
              line(P(x1, y0), P(x1, y0 + t_pipe)),
              cells=(2 * L, 2 * L), names=("bore", None, "outer", None)),
    ]
    w = w_flange
    for n in range(n_flanges):
        x0 = 2 * w * n
        shoulder, top = y1 + w / 2, y1 + h_flange - w / 2
        last = n == n_flanges - 1
        a, b, c, d = x0, x0 + w / 2, x0 + 3 * w / 2, x0 + 2 * w
        tip = lambda start, end: arc(P(x0 + w, top), w / 2, np.pi / 2 - start, np.pi / 2 - end)
        blocks += [
            Block(line(P(a, y0), P(b, y0)), line(P(b, y0), P(b, shoulder)),
                  bezier(P(a, y1), P(b, y1), P(b, shoulder)), line(P(a, y0), P(a, y1)),
                  cells=(L, 2 * L), names=("bore", None, "outer", None)),
            Block(line(P(b, y0), P(c, y0)), line(P(c, y0), P(c, shoulder)),
                  line(P(b, shoulder), P(c, shoulder)), line(P(b, y0), P(b, shoulder)),
                  cells=(2 * L, 2 * L), names=("bore", None, None, None)),
            Block(line(P(c, y0), P(d, y0)), line(P(d, y0), P(d, y1)),
                  bezier(P(c, shoulder), P(c, y1), P(d, y1)), line(P(c, y0), P(c, shoulder)),
                  cells=(L, 2 * L), names=("bore", "flange_end" if last else None, "outer", None)),
            Block(line(P(b, shoulder), P(c, shoulder)), line(P(c, shoulder), P(c, top)),
                  line(P(b, top), P(c, top)), line(P(b, shoulder), P(b, top)),
                  cells=(2 * L, 3 * L), grading=(None, fin), names=(None, "outer", None, "outer")),
            Block(line(P(b, top), P(c, top)), tip(0.0, np.pi / 4),
                  tip(3 * np.pi / 4, np.pi / 4), tip(np.pi, 3 * np.pi / 4),
                  cells=(2 * L, L), names=(None, "outer", "outer", "outer")),
        ]
    return blocks


def build_mesh(level):
    section = merge_blocks(section_blocks(level))
    return revolve(section, 4 * level, angle=np.pi, axis=(1.0, 0.0, 0.0), radial=(0.0, 1.0, 0.0),
                   names=("symmetry", "symmetry"))


# ## Model
#
# Glass is homogeneous and isotropic. The bore carries a `ConvectiveHeatFlux`, the end
# faces and the symmetry plane an explicit zero `HeatFlux`, and the whole outer surface
# the `NonlinearHeatFlux` $q=C(\theta)\,(T-T_a)^{5/4}$ with its slope
# $\partial q/\partial T=\tfrac54\,C(\theta)\,(T-T_a)^{1/4}$.

glass = IsotropicSolid(density=2203.0, specific_heat_capacity=703.0, thermal_conductivity=k_glass,
                       youngs_modulus=73.1e9, poisson_ratio=0.17, thermal_expansion_coefficient=5.5e-7)


def solve_level(mesh):
    geometry = mesh.geometry
    b = geometry.boundaries
    model = HeatConduction(domain=geometry.domain(), material=glass, heating=0.0)
    problem = Problem(name="Glass cooling flange", geometry=geometry, physics=(model,), boundary_conditions=(
        ConvectiveHeatFlux(boundary=b.bore, physics=model, heat_transfer_coefficient=h_in, ambient_temperature=T_in),
        NonlinearHeatFlux(boundary=b.outer, physics=model, flux=air_flux, derivative=air_flux_slope),
        HeatFlux(boundary=b.pipe_end | b.flange_end | b.symmetry, physics=model, value=0.0),
    ))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect(),
                        nonlinear_solver=Newton(), initial_guess={model.temperature: 330.0})
    field = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.temperature)
    nodal = np.asarray(field.evaluate(mesh.coordinates)).reshape(-1)
    return dict(field=field, b=b, Q_in=-2 * field.boundary_flux(b.bore), Q_out=2 * field.boundary_flux(b.outer),
                T_max=float(nodal.max()), T_min=float(nodal.min()))


# ## Solve
#
# Three meshes with cell counts in the ratio 2 : 3 : 4 in every direction.

levels = (2, 3, 4)
runs = []
for level in levels:
    start = time.time()
    mesh = build_mesh(level)
    out = solve_level(mesh)
    runs.append(out)
    print(f"level {level}: {mesh.cell_count:6d} hexahedra, Q = {out['Q_in']:.5f} W, "
          f"T in [{out['T_min']:.3f}, {out['T_max']:.3f}] K  ({time.time() - start:.0f} s)")
out = runs[-1]
field, b = out["field"], out["b"]


# ## Comparison with the reference
#
# **Energy balance and mesh convergence.** For three meshes with sizes $h\propto 1/\ell$
# `convergence` returns the observed order $p$ and the Richardson estimate of each
# quantity.

print("Energy balance on the finest mesh (full flange)")
print(f"  heat entering through the bore   Q_in  = {out['Q_in']:.5f} W")
print(f"  heat leaving to the air          Q_out = {out['Q_out']:.5f} W")
print(f"  relative imbalance               {abs(out['Q_in'] - out['Q_out']) / out['Q_in']:.2e}")
print("Mesh study (h ~ 1/level)")
study = {}
for label, key in (("Q_in / W", "Q_in"), ("Q_out / W", "Q_out"), ("T_max / K", "T_max"), ("T_min / K", "T_min")):
    vals = [o[key] for o in runs]
    study[key] = estimate = convergence(vals, [1.0 / level for level in levels])
    order = f"{estimate.order:.2f}" if estimate.order is not None else "n/a"
    extrapolated = f"{estimate.extrapolated:.6f}" if estimate.extrapolated is not None else "n/a"
    print(f"  {label:10s} " + "  ".join(f"{v:.6f}" for v in vals)
          + f"   order {order}, extrapolated {extrapolated}, change last two meshes {abs(vals[-1] - vals[-2]):.1e}"
          + (f" ({100 * abs(vals[-1] / vals[-2] - 1):.3f} %)" if key.startswith("Q") else " K"))
Q_ext = study["Q_in"].extrapolated
if Q_ext is not None:
    print(f"  finest-mesh heat rate vs Richardson estimate: {100 * (out['Q_in'] / Q_ext - 1):+.3f} %")

# **1D radial wall estimate (sanity bound).** On the insulated end face of the pipe the
# axial temperature gradient vanishes, so this is where the wall comes closest to pure
# radial conduction. For radial conduction the
# local bore flux $q_i$ fixes the drop across the wall,
# $\Delta T_\mathrm{1D}=q_i\,r_i\ln(r_o/r_i)/k$, and the outer flux must be
# $q_o=q_i\,r_i/r_o$. The connecting pipe is only 12 mm long, shorter than the axial
# conduction length $\sqrt{k\,t/(h_i+h_a)}\approx 15$ mm of the glass wall, so part of
# the heat entering the bore there still flows axially into the flanges: the 3D wall
# drop and outer flux must lie *below* the 1D values. The comparison is a plausibility
# bound, not the main reference.
r_out = r_in + t_pipe
print("1D radial estimate at the pipe end, x = -11.9 mm")
for theta_deg in (0.0, 90.0, 180.0):
    th = np.radians(theta_deg)
    direction = np.array([0.0, -np.cos(th), np.sin(th)])
    inner = np.array([-11.9 * mm, 0.0, 0.0]) + r_in * (1 + 1e-6) * direction
    outer = np.array([-11.9 * mm, 0.0, 0.0]) + r_out * (1 - 1e-6) * direction
    T_i, T_o = (float(np.asarray(field.evaluate([p]))[0]) for p in (inner, outer))
    q_i = h_in * (T_in - T_i)
    dT_1d = q_i * r_in * np.log(r_out / r_in) / k_glass
    q_o = float(air_flux(T_o, outer[1], outer[2]))
    print(f"  theta = {theta_deg:5.1f} deg: wall drop 3D {T_i - T_o:.4f} K vs 1D {dT_1d:.4f} K "
          f"({100 * ((T_i - T_o) / dT_1d - 1):+.1f} %);  outer flux 3D {q_o:.2f} vs 1D {q_i * r_in / r_out:.2f} W/m^2 "
          f"({100 * (q_o / (q_i * r_in / r_out) - 1):+.1f} %)")


# ## Figure
#
# Left: surface temperature of the complete flange (the computed half mirrored about
# the symmetry plane), seen obliquely from the front; gravity points down. Middle: the
# vertical section through the axis, which is the symmetry plane of the model: upper half
# at the top of the flange ($\theta=180^\circ$), lower half at the bottom ($\theta=0$).
# Right: total heat rate on the three meshes with its Richardson estimate, and the local
# air-side film coefficient over the outer surface.

limits = (out["T_min"], out["T_max"])
outer = field.sample(along=b.outer, subdivisions=2)
p, T_surface = np.asarray(outer.positions), np.asarray(outer.values).reshape(-1)
h_local = air_flux(T_surface, p[:, 1], p[:, 2]) / np.maximum(T_surface - T_air, 1e-9)
fig = plots.figure(layout=[["surface", "section", "study"], ["surface", "section", "film"]], size=(17, 6.2))
plots.surface(field, boundary=b.outer | b.pipe_end | b.flange_end, symmetry={"planes": [((0, 0, 1), (0, 0, 0))]},
              limits=limits, subdivisions=3, camera=plots.Camera(elevation=14, azimuth=-58, up="y"), label="T [K]",
              title="Surface temperature (full flange)", ax=fig["surface"])
plots.surface(field, boundary=b.symmetry, limits=limits, contours=10, subdivisions=3, label="T [K]",
              title="Vertical section through the axis (top: theta = 180 deg, bottom: theta = 0)", ax=fig["section"])
plots.line(levels, {"Q through the bore": study["Q_in"].values, "Q to the air": study["Q_out"].values},
           hlines=None if Q_ext is None else {f"Richardson estimate {Q_ext:.4f} W": Q_ext}, markers=True,
           xlabel="mesh level", ylabel="heat rate [W]", title="Mesh study of the cooling power", ax=fig["study"])
plots.line(points={"outer surface": (np.degrees(incidence(p[:, 1], p[:, 2])), h_local)},
           xlabel="angle from the bottom theta [deg]", ylabel="h_air [W/(m² K)]",
           title="Local natural-convection coefficient", legend=False, ax=fig["film"])
plots.show()


# ## Going further
#
# Sweep the bore radius between 6 and 10 mm at fixed flange dimensions: the bore-side
# film resistance dominates, so a wider bore raises the cooling power noticeably. Add
# a fourth flange, or replace glass by a metal, and compare how much of the temperature
# drop then sits in the wall versus the two films.
