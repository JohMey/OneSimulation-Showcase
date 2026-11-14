"""OneSimulation example: heat transfer in a surface-mount package next to a hot voltage regulator.
Requires the installed, activated OneSimulation runtime.

A silicon die in a 16-lead plastic gull-wing package (SOIC-16) sits on an FR-4 circuit board. It
dissipates 20 mW and is connected through a thin copper interconnect, an aluminium lead and a
copper ground plane to a voltage regulator at 50 degC. The steady 3D temperature field is computed
with isoparametric tensor-product finite elements on a curved, multi-block hexahedral mesh
(quadratic and linear elements compared) and checked against the published peak die temperature,
the global energy balance, a thermal-resistance network and mesh studies.
"""


# # Heat transfer in a surface-mount package for a silicon chip
#
# Designers of printed-circuit boards have to keep heat-sensitive integrated circuits away from
# heat sources. Here a voltage regulator, whose footprint on the board is held at 50 °C, shares
# a copper ground plane with one lead of a small-outline package. Inside the package a silicon
# die dissipates its own power. The question is how hot the die gets.
#
# **Geometry** (all dimensions in mm; $x$ along the package, $z$ normal to the board):
#
# | part | geometry | conductivity / (W/(m K)) |
# |---|---|---|
# | circuit board (FR-4) | $20\times10\times1$, $-1.9\le z\le-0.9$ | 0.3 |
# | ground plane (copper) | $4\times4\times0.1$ in the board top, $-8\le x\le-4$, $-5\le y\le-1$ | 400 |
# | regulator footprint | $-10\le x\le-8$, $-5\le y\le-1$ on the board top, held at 50 °C | - |
# | package body (plastic) | $9.9\times3.9$, $\lvert z\rvert\le0.75$; side walls vertical for $\lvert z\rvert\le0.1$, then tapered to a half-width of 1.72 at the top and bottom faces | 0.2 |
# | silicon die | $1\times1\times0.1$, centred at the origin | 130 |
# | 16 gull-wing leads (aluminium) | $0.4\times0.2$ cross-section, pitch 1.27; stub 0.26 long, a 90° bend (inner radius 0.14), a vertical leg of 0.32, a second 90° bend, a 0.16 long foot on the board | 238 |
# | interconnect (copper) | 5 µm thick L-shaped strip in the plane $z=0$, 0.4 wide, from lead 1 ($x=-4.445$) along $y$ to $y=0$, then along $x$ to the die | 400 |
#
# These are the dimensions of a narrow-body SOIC-16 (JEDEC MS-012: body 9.9 x 3.9 mm, lead pitch
# 1.27 mm, lead width 0.31-0.51 mm, lead thickness 0.19-0.25 mm). The body stands 0.15 mm above
# the board, so the air gap under it is part of the cooled surface.
#
# **Physics.** The steady heat equation $\nabla\cdot(k\nabla T)+Q=0$ with the piecewise constant
# conductivities above and a uniform source $Q=2\times10^8$ W/m³ in the die, i.e.
# $P=20$ mW. Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | regulator footprint on the board top | $T=50$ °C |
# | every other exterior face (board, package, leads, air gap) | forced convection $-k\,\partial_nT=h\,(T-T_\infty)$, $h=50$ W/(m² K), $T_\infty=30$ °C |
#
# The thin copper layers (5 µm interconnect, 0.1 mm ground plane) are resolved as true 3D
# layers of cells, not lumped into a shell model. The field is genuinely three-dimensional:
# the heat flows from the regulator through the ground plane, up lead 1, along the
# interconnect into the package, and from the die through the plastic into the air and the
# leads.
#
# **Numerical method.** Continuous Galerkin finite elements with tensor-product Lagrange
# elements on hexahedra: quadratic (Q2, 27 nodes) and, for comparison, trilinear (Q1, 8 nodes).
# The mesh is a conforming multi-block hexahedral mesh built here in numpy: a graded
# tensor-product block for the board, a tapered block for the package (the taper is an exact
# bilinear map), and sixteen swept blocks for the leads. Every cell carries 27 geometry nodes,
# so the circular lead bends are represented by curved (isoparametric) quadratic cells. Both
# element orders use the same curved cells. The problem is linear and steady: one sparse direct
# LU solve per mesh, no time integration.
#
# **Reference.** (i) The published value for this set-up, a maximum silicon temperature of
# 47.6 °C (obtained with a shell approximation for the copper layers on a free tetrahedral mesh),
# (ii) the global energy balance: heat entering through the regulator footprint plus the die power
# equals the heat convected to the air, (iii) a thermal-resistance network as an independent
# plausibility estimate, and (iv) three-level mesh studies of the maximum die temperature
# for both element orders, with observed order and Richardson extrapolation.

from dataclasses import replace
import time
import numpy as np

from onesim.cad.d2 import Plane
from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import ExplicitGeometry, VolumeBlock, merge_volume_blocks
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, FixedTemperature, HeatConduction
from onesim.results.comparisons import convergence
from onesim.results.integration import measure
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

mm = 1e-3
K_FR4, K_CU, K_AL, K_SI, K_PLASTIC = 0.3, 400.0, 238.0, 130.0, 0.2
Q_DIE = 2e8
V_DIE = 1.0 * 1.0 * 0.1 * mm**3
P_DIE = Q_DIE * V_DIE
h_air, T_air, T_reg = 50.0, 303.15, 323.15
T_published = 47.6
PITCH, LEAD_W, LEAD_X0 = 1.27, 0.4, -4.645
T_IC = 0.0025
DIE_SIZE, DIE_GROWTH = 0.08, 0.5
print(f"die power P = {P_DIE * 1e3:.1f} mW")


# ## Geometry and mesh
#
# **Blocks.** The board and the package are separate tensor-product blocks whose grid lines pass
# through every material boundary: in the package the lead edges, the die and the interconnect
# strip, in the board the lead edges (where the feet stand), the ground plane and the regulator
# footprint. In $z$ the package grid contains the die faces ($\pm0.05$ mm), the two faces of the
# 5 µm interconnect and the start of the taper ($\pm0.1$ mm). Above
# $\lvert z\rvert=0.1$ mm the package nodes are moved inward linearly in $y$ to follow the
# tapered side walls. Each lead is a swept block: its cross-section grid is the package grid on
# the side wall ($x$ lines of the lead, $z$ lines between $\pm0.1$ mm), and it is swept along
# the lead centre-plane path, stub, quarter circle, leg, quarter circle, foot. The foot meets
# the board top on board grid lines and the stub meets the package wall on package grid lines,
# so all blocks share their nodes. Each block is a `VolumeBlock` (a map of a parameter box
# onto the part), and `merge_volume_blocks` joins them by node position into one mesh.
#
# **Curved cells.** Every cell is a 27-node hexahedron: its edge midpoints, face centres and
# centre are sampled from the exact block maps. In the bends the cells therefore follow the
# circular arcs to third order in the cell size.
#
# **Grading and refinement level.** The temperature gradients are steepest in the plastic at the
# die edges, where the conductivity jumps from 130 to 0.2 W/(m K). The package grid is therefore
# graded: in the package plane the cell size is 0.08 mm at the die edges and grows by 0.5 mm per
# mm of distance up to 0.45 mm; across the tapered plastic it grows from 0.05 mm at the die level
# to about 0.28 mm at the package faces. A level $\ell$ divides all these sizes by $\ell$, as well
# as the 0.05 mm through the die, 0.3 mm through the board and 0.2 mm along the leads; the board
# grid grows away from the package. Material interfaces
# stay on cell faces at every level, so each quadrature point lies inside one material.

def subdivide(breaks, size):
    """Nodes through all break points; each interval split into cells no larger than size(midpoint)."""
    out = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(np.ceil((b - a) / size(0.5 * (a + b)) - 1e-9)))
        out.extend(a + (b - a) * np.arange(1, n + 1) / n)
    return np.array(out)


def grid_lines(level):
    """Grid lines (mm) of the package block, the board block and the lead sweep at a refinement level."""
    h = 0.45 / level
    leads = sorted({LEAD_X0 + PITCH * k + d for k in range(8) for d in (0.0, LEAD_W)})
    def near_die(v):
        return min(h, (DIE_SIZE + DIE_GROWTH * abs(abs(v) - 0.5)) / level)
    px = subdivide(sorted({-4.95, -0.5, 0.5, 4.95, *leads}), near_die)
    py = subdivide([-1.95, -0.5, -0.2, 0.2, 0.5, 1.95], near_die)
    pz = subdivide([-0.75, -0.1, -0.05, -T_IC, T_IC, 0.05, 0.1, 0.75],
                   lambda z: 0.05 / level if abs(z) < 0.1 else (0.05 + 0.35 * (abs(z) - 0.1)) / level)
    lead_lines = [x for x in px if any(LEAD_X0 + PITCH * k - 1e-9 <= x <= LEAD_X0 + PITCH * k + LEAD_W + 1e-9
                                         for k in range(8))]
    bx = subdivide(sorted({-10.0, -8.0, -6.5, -4.0, 7.0, 10.0, *lead_lines}),
                   lambda x: h * (1.0 + 0.8 * max(abs(x) - 4.95, 0.0)))
    n_foot = int(np.ceil(0.16 / h - 1e-9))
    foot = list(-2.69 - 0.16 * np.arange(n_foot + 1) / n_foot)
    by = subdivide(sorted({-5.0, *foot, -1.0, 0.0, *(-v for v in foot), 5.0}),
                   lambda y: h * (1.0 + 0.8 * max(abs(y) - 2.85, 0.0)) if abs(y) > 2.69 else 1.5 * h)
    bz = subdivide([-1.9, -1.0, -0.9], lambda z: 0.3 / level)
    n_path = [max(1, int(np.ceil(length / (0.2 / level) - 1e-9))) for length in (0.26, 0.377, 0.32, 0.377)]
    return (px, py, pz), (bx, by, bz), n_path + [n_foot]


def half_width(z):
    """Package half-width: 1.95 mm for |z| <= 0.1 mm, tapering to 1.72 mm at |z| = 0.75 mm."""
    return 1.95 - 0.23 * np.clip((np.abs(z) - 0.1) / 0.65, 0.0, 1.0)


def lead_path(u, t):
    """(y, z) of a lead on the y < 0 side; u in [0, 5] runs over stub, bend, leg, bend, foot,
    t in [0, 0.2] across the thickness (t = 0 is the lower face of the stub)."""
    u, t = np.broadcast_arrays(np.asarray(u, float), np.asarray(t, float))
    segment = np.minimum(np.floor(u).astype(int), 4)
    s = u - segment
    y, z = np.empty(u.shape), np.empty(u.shape)
    m = segment == 0
    y[m], z[m] = -1.95 - 0.26 * s[m], -0.1 + t[m]
    m = segment == 1
    r, phi = 0.14 + t[m], 0.5 * np.pi * s[m]
    y[m], z[m] = -2.21 - r * np.sin(phi), -0.24 + r * np.cos(phi)
    m = segment == 2
    y[m], z[m] = -2.35 - t[m], -0.24 - 0.32 * s[m]
    m = segment == 3
    d, psi = 0.34 - t[m], 0.5 * np.pi * s[m]
    y[m], z[m] = -2.69 + d * np.cos(psi), -0.56 - d * np.sin(psi)
    m = segment == 4
    y[m], z[m] = -2.69 - 0.16 * s[m], -0.9 + t[m]
    return y, z


def on_regulator(point, normal):
    """Boundary group of a face: the regulator footprint on the board top, or air."""
    x, y, z = np.asarray(point) / mm
    return "regulator" if abs(z + 0.9) < 1e-6 and x < -8.0 and y < -1.0 else "air"


assembly = ExplicitGeometry(dimension=3, boundary_names=("regulator", "air"), classifier=on_regulator)


def build_mesh(level):
    """Board, tapered package and 16 swept leads as conforming 27-node hexahedra."""
    (px, py, pz), (bx, by, bz), n_path = grid_lines(level)
    blocks = [
        VolumeBlock(lambda u, v, w: (u, v, w), bx * mm, by * mm, bz * mm, region="board"),
        VolumeBlock(lambda u, v, w: (u, v * half_width(w / mm) / 1.95, w), px * mm, py * mm, pz * mm,
                    region="package"),
    ]
    t_lines = pz[np.abs(pz) <= 0.1 + 1e-9] + 0.1
    u = np.concatenate([[0.0]] + [k + np.arange(1, n + 1) / n for k, n in enumerate(n_path)])
    for k in range(8):
        x0 = LEAD_X0 + PITCH * k
        lx = px[(px >= x0 - 1e-9) & (px <= x0 + LEAD_W + 1e-9)]
        for side in (-1, 1):
            def lead(u, x, t, side=side):
                y, z = lead_path(u, t)
                return x, side * y * mm, z * mm
            blocks.append(VolumeBlock(lead, u, lx * mm, t_lines, region="leads"))
    return merge_volume_blocks(blocks, assembly, order=2)


# ## Model
#
# One `HeatConduction` problem serves both element orders. The conductivity is region-wise
# (first match wins): the silicon die, the copper interconnect strip, the copper ground plane
# under the regulator, the aluminium leads, the FR-4 board and the plastic everywhere else.
# The die source is a region-wise volumetric heating. Material interfaces lie on cell faces,
# so every quadrature point sees one material.

def box_mm(x0, x1, y0, y1, z0, z1):
    return Box(length=(x1 - x0) * mm, width=(y1 - y0) * mm, height=(z1 - z0) * mm, origin=(x0 * mm, y0 * mm, z0 * mm))


die = box_mm(-0.5, 0.5, -0.5, 0.5, -0.05, 0.05)
strip = box_mm(-4.645, -4.245, -1.95, 0.2, -T_IC, T_IC) | box_mm(-4.245, -0.5, -0.2, 0.2, -T_IC, T_IC)
ground_plane = box_mm(-8.0, -4.0, -5.0, -1.0, -1.0, -0.9)
board = box_mm(-10.0, 10.0, -5.0, 5.0, -1.9, -0.9)


def in_leads(points):
    return (points[:, 2] > -0.9 * mm) & (np.abs(points[:, 1]) > 1.95 * mm)


conductivity = regionwise(((die, K_SI), (strip, K_CU), (ground_plane, K_CU), (in_leads, K_AL), (board, K_FR4)),
                          default=K_PLASTIC, unit="W/m/K")
materials = replace(IsotropicSolid(density=1000.0, specific_heat_capacity=1000.0, thermal_conductivity=1.0,
                                   youngs_modulus=1e9, poisson_ratio=0.3, thermal_expansion_coefficient=0.0),
                    thermal_conductivity=conductivity)
die_heating = regionwise(((die, Q_DIE),), default=0.0, unit="W/m^3")
b = assembly.boundaries


def solve(level, order):
    start = time.perf_counter()
    mesh = build_mesh(level)
    model = HeatConduction(domain=assembly.domain(), material=materials, heating=die_heating)
    problem = Problem(name="SOIC-16 package next to a voltage regulator", geometry=assembly, physics=(model,),
                      boundary_conditions=(
                          FixedTemperature(boundary=b.regulator, physics=model, value=T_reg),
                          ConvectiveHeatFlux(boundary=b.air, physics=model, heat_transfer_coefficient=h_air,
                                             ambient_temperature=T_air)))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=order)), linear_solver=SparseDirect())).solve_stationary()
    field = result.field(model.temperature)
    g = np.linspace(-0.5, 0.5, 21) * mm
    grid = np.array(np.meshgrid(g, g, np.linspace(-0.05, 0.05, 5) * mm, indexing="ij")).reshape(3, -1).T
    T_die = np.asarray(field.evaluate(grid)).ravel()
    q_out = h_air * (float(field.integrate(boundary=b.air).value) - T_air * measure(field, boundary=b.air))
    return dict(level=level, order=order, mesh=mesh, field=field, dofs=result.operators[model.model_id].problem_size,
                T_die=float(T_die.max()) - 273.15, q_out=q_out, q_reg=-field.boundary_flux(b.regulator),
                seconds=time.perf_counter() - start)


# ## Solve
#
# Quadratic and linear elements on the same three meshes ($\ell=0.5$, 0.7 and 1.0). The levels
# are not nested (the material interfaces fix many grid lines), so each mesh is characterised by
# its representative cell size $h=(V/N)^{1/3}$, as in the grid-convergence procedure of
# Celik et al. (*J. Fluids Eng.* 130, 078001, 2008).

LEVELS = {2: (0.5, 0.7, 1.0), 1: (0.5, 0.7, 1.0)}
runs = {2: [], 1: []}
for order, levels in LEVELS.items():
    for level in levels:
        run = solve(level, order)
        run["h"] = (run["mesh"].cell_volumes.sum() / run["mesh"].cell_count) ** (1 / 3)
        runs[order].append(run)
        print(f"Q{order} level {level}: {run['mesh'].cell_count:6d} cells, {run['dofs']:6d} unknowns, "
              f"h = {run['h'] / mm * 1e3:.0f} um, max die temperature {run['T_die']:.4f} degC, "
              f"convected {run['q_out'] * 1e3:.4f} mW  ({run['seconds']:.0f} s)")


# ## Comparison with the reference
#
# Finite elements are too stiff on coarse meshes: here that means too conductive, so both element
# orders approach the die temperature from below. The edges of the die and of the 5 µm copper strip,
# where the conductivity jumps by more than three orders of magnitude, are singular lines of the
# temperature gradient; they limit the observed order of convergence to well below the smooth-field
# value (3 for the quadratic elements). The heat convected to the air is dominated by the heat
# entering through the regulator footprint, whose flux is singular along the footprint edge.

q2, q1 = runs[2][-1], runs[1][-1]
print("Maximum silicon temperature, finest mesh of each element order")
print(f"   published value                   {T_published:8.2f} degC")
for run in (q2, q1):
    print(f"   Q{run['order']} ({run['dofs']:6d} unknowns)          {run['T_die']:8.3f} degC   "
          f"deviation {run['T_die'] - T_published:+.3f} K ({100 * (run['T_die'] - T_published) / (T_published - 30.0):+.2f} % "
          f"of the temperature rise above the air)")

print("Mesh studies (representative cell sizes " + ", ".join(f"{r['h'] / mm * 1e3:.0f}" for r in runs[2]) + " um)")
def trend(values, sizes, unit, digits):
    """Observed order and Richardson value, or the reason why none is defined."""
    estimate = convergence(values, sizes)
    if estimate.order is not None:
        return f"observed order {estimate.order:.2f}, Richardson {estimate.extrapolated:.{digits}f} {unit}", estimate.extrapolated
    return estimate.reason, np.nan


extrapolated_T = {}
for order in (2, 1):
    sizes = [r["h"] for r in runs[order]]
    vals = [r["T_die"] for r in runs[order]]
    text, extrapolated_T[order] = trend(vals, sizes, "degC", 3)
    print(f"   Q{order} maximum die temperature: " + "  ".join(f"{v:.4f}" for v in vals) + f" degC; {text}; "
          f"change between the last two meshes {abs(vals[-1] - vals[-2]):.3f} K")
    heat = [r["q_out"] * 1e3 for r in runs[order]]
    text, _ = trend(heat, sizes, "mW", 3)
    print(f"   Q{order} heat convected to the air: " + "  ".join(f"{v:.3f}" for v in heat) + f" mW; {text}; "
          f"change between the last two meshes {100 * abs(heat[-1] / heat[-2] - 1):.2f} %")
reference_T = extrapolated_T[2] if np.isfinite(extrapolated_T[2]) else q2["T_die"]
print(f"   published value {T_published} degC against the Q2 estimate {reference_T:.2f} degC: "
      f"{T_published - reference_T:+.2f} K ({100 * (T_published - reference_T) / (reference_T - 30.0):+.1f} % "
      f"of the temperature rise)")

print("Energy balance on the finest mesh (heat in = heat out)")
for run in (q2, q1):
    heat_in = run["q_reg"] + P_DIE
    print(f"   Q{run['order']}: regulator footprint {run['q_reg'] * 1e3:.4f} mW + die {P_DIE * 1e3:.1f} mW = "
          f"{heat_in * 1e3:.4f} mW; convected to the air {run['q_out'] * 1e3:.4f} mW; "
          f"relative difference {abs(heat_in - run['q_out']) / run['q_out']:.1e}")


# **Thermal-resistance network** (an independent estimate by hand). Heat leaves the die
# through the plastic above and below it: a column that spreads at 45° from the $1\times1$ mm
# die over the 0.7 mm to the package faces has $R=t/(k\,a\,(a+2t))$; each face then loses it by
# convection over its area. The interconnect is a $5$ µm copper strip of 0.4 mm width and
# 2.15 + 3.75 mm length, $R=\ell/(k\,d\,w)$, connecting the die to lead 1, which sits on the
# ground plane at nearly 50 °C. The die node then follows from
# $T_j=(T_\infty/R_a+T_\mathrm{reg}/R_i+P)/(1/R_a+1/R_i)$. The network ignores conduction
# into the other leads and the board, so it is expected to over-estimate the die temperature
# by some kelvin.

a_die, t_mold = 1.0 * mm, 0.70 * mm
R_spread = t_mold / (K_PLASTIC * a_die * (a_die + 2 * t_mold))
A_face = 9.9 * mm * 2 * 1.835 * mm
R_side = R_spread + 1.0 / (h_air * A_face)
R_air = R_side / 2
R_ic = (2.15 * mm + 3.745 * mm) / (K_CU * 2 * T_IC * mm * LEAD_W * mm)
T_network = (T_air / R_air + T_reg / R_ic + P_DIE) / (1 / R_air + 1 / R_ic) - 273.15
print(f"Network: R_spread = {R_spread:.0f} K/W per face, R_die-air = {R_air:.0f} K/W, "
      f"R_interconnect = {R_ic:.0f} K/W")
print(f"   die temperature, network {T_network:.2f} degC; Q2 {q2['T_die']:.2f} degC; "
      f"difference {T_network - q2['T_die']:+.2f} K")


# ## Figure
#
# Top: the surface temperature of the whole assembly, and the horizontal section through the
# die mid-plane with the interconnect strip and the lead stubs. Bottom: the section through
# lead 1 and the interconnect riser ($x=-4.445$ mm), the section $y=0$ through the die and the
# interconnect strip, the temperature on the lower face of the die, and the mesh convergence.

fine = runs[2][-1]
field = fine["field"]
celsius = lambda T, p: T - 273.15
g = np.linspace(-0.5, 0.5, 41) * mm
die_face = np.asarray(field.evaluate(np.column_stack((*(c.ravel() for c in np.meshgrid(g, g)), np.full(g.size**2, -0.05 * mm))))) - 273.15
estimates = {f"published value {T_published} °C": T_published}
if np.isfinite(extrapolated_T[2]):
    estimates[f"Q2 Richardson estimate {extrapolated_T[2]:.2f} °C"] = extrapolated_T[2]
view = dict(component=celsius, limits=(30.0, 50.0), subdivisions=2, label="T [°C]")
fig = plots.figure(layout=[["surface", "surface", "strip", "strip"], ["lead", "die", "face", "study"]], size=(22, 12))
plots.surface(field, boundary=b.air | b.regulator, camera=plots.Camera(elevation=30, azimuth=-60, distance=0.8),
              title=f"Surface temperature (Q2, {fine['dofs']} unknowns)", ax=fig["surface"], **view)
plots.section(field, plane=Plane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)), contours=20,
              title="Section z = 0: die, interconnect strip and lead stubs", xlimits=(-6 * mm, 6 * mm),
              ylimits=(-2.4 * mm, 2.4 * mm), ax=fig["strip"], **view)
plots.section(field, plane=Plane(origin=(-4.445 * mm, 0.0, 0.0), normal=(1.0, 0.0, 0.0)), contours=20,
              title="Section x = -4.445 mm through lead 1", xlimits=(-5 * mm, 0.6 * mm), ylimits=(-1.9 * mm, 0.8 * mm),
              ax=fig["lead"], **view)
plots.section(field, plane=Plane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0)), contours=20,
              title="Section y = 0 through the die and the strip", xlimits=(-6 * mm, 2 * mm), ylimits=(-1.9 * mm, 0.8 * mm),
              ax=fig["die"], **view)
plots.section(field, plane=Plane(origin=(0.0, 0.0, -0.05 * mm), normal=(0.0, 0.0, 1.0)), component=celsius,
              limits=(float(die_face.min()), float(die_face.max())), contours=10, subdivisions=3, label="T [°C]",
              title=f"Lower face of the die (max {die_face.max():.2f} °C)", xlimits=(-0.5 * mm, 0.5 * mm),
              ylimits=(-0.5 * mm, 0.5 * mm), ax=fig["face"])
plots.line(y={f"Q{order} elements": ([run["h"] / mm * 1e3 for run in runs[order]], [run["T_die"] for run in runs[order]])
              for order in (2, 1)},
           hlines=estimates, markers=True, xlabel="representative cell size (V/N)^(1/3) [µm]",
           ylabel="maximum die temperature [°C]", title="Mesh convergence of the die temperature", ax=fig["study"])
plots.show()


# ## Going further
#
# Raise the die power or move the regulator closer and watch the share of the regulator heat
# in the die temperature; replace the aluminium leads by copper alloy ($k\approx260$ W/(m K))
# or the plastic by a filled moulding compound ($k\approx0.8$ W/(m K)); add the shell model
# of the copper layers as a second variant and compare it with the resolved 5 µm layer; or make
# the problem transient (`solve_transient`) to see how fast the die follows a power step.
