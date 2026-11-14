"""OneSimulation example: three-dimensional lid-driven cubic cavity at Re = 100 and 400.
Requires the installed, activated OneSimulation runtime.

The lid of a closed cube slides at constant speed. The genuinely three-dimensional flow - a
primary vortex that is weakened and bent by the end walls, with secondary corner vortices and a
spiralling motion along the vortex axis - is computed with Taylor-Hood Q2/Q1 finite elements on
graded hexahedra (plus P2/P1 tetrahedra for comparison) and graded against published 3D cube
results, with a mesh study, mass conservation and the comparison with the 2D cavity of Ghia et al.
"""


# # Lid-driven cubic cavity in three dimensions
#
# A cube of side $L$ is filled with an incompressible Newtonian fluid. Its top face (the lid,
# $y=L$) slides in the $x$-direction with constant speed $U$; the other five faces are fixed walls.
# The Reynolds number is $Re=\rho UL/\mu$. In two dimensions this is the classical driven cavity
# (Ghia, Ghia & Shin 1982). In a real, closed cube the two **end walls** $z=0$ and $z=L$ brake the
# primary vortex: the flow depends on all three coordinates, fluid spirals along the vortex axis
# from the end walls towards the mid-plane, and corner vortices form where the end walls meet the
# bottom and the downstream wall. How strongly the end walls change the flow depends on $Re$ -
# that is the "3D end-wall effect" this notebook quantifies at $Re=100$ and $Re=400$.
#
# **Numerical method.**
#
# * Spatial: continuous Galerkin finite elements, **Taylor-Hood Q2/Q1 on hexahedra** - 27-node
#   triquadratic velocity, 8-node trilinear pressure, inf-sup stable. For comparison the same grid
#   is cut into six tetrahedra per hexahedron and solved with Taylor-Hood P2/P1 tetrahedra.
# * Mesh: tensor-product grid graded towards all walls with a tanh stretching (the wall cells are
#   about 2.8 times thinner than the centre cells). The flow is mirror-symmetric about the
#   mid-plane $z=L/2$, so only the half cube $0\le z\le L/2$ is meshed and the mid-plane carries a
#   symmetry (free-slip) condition.
# * Solver: stationary Navier-Stokes, Newton iteration with a sparse direct LU solver; $Re=400$ is
#   started from the $Re=100$ solution, each finer mesh from the coarser one.
#
# **Reference.** Published three-dimensional results for the cube: the extrema of the horizontal
# velocity on the vertical centreline and of the vertical velocity on the horizontal centreline
# (both in the mid-plane) and their positions, from
# F. Oueslati, B. Ben Beya, T. Lili, "Aspect ratio effects on three-dimensional incompressible
# flow in a two-sided non-facing lid-driven parallelepiped cavity", *C. R. Mecanique* 339,
# 655-665 (2011), Table 1 (non-uniform staggered $48^3$ finite volumes, validated against
# S. Albensoeder and H. C. Kuhlmann, *J. Comput. Phys.* 206, 536-558, 2005), and from
# H. Ding, C. Shu, K. S. Yeo, D. Xu, *Comput. Methods Appl. Mech. Engrg.* 195, 516-533 (2006)
# ($41^3$ local multiquadric differential quadrature). The two references differ by up to 4 %,
# which bounds what "agreement" can mean here. In addition: a three-level mesh study with
# observed order and Richardson estimate, the mass balance, and the comparison with the
# two-dimensional benchmark of U. Ghia, K. N. Ghia, C. T. Shin, *J. Comput. Phys.* 48, 387-411
# (1982), Table I, which shows the end-wall effect directly.

from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import gc
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq, minimize_scalar

from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import StructuredHexahedralMesher, StructuredTetrahedralMesher
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.backends.structured.volume.tetrahedral import StructuredTetrahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import trilinear_hexahedral_geometry
from onesim.meshing.meshes.models.geometry.discretization_data import CoordinateSet
from onesim.meshing.meshes.models.topology.topology import Topology
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import tetrahedral_geometry
from onesim.simulation.problem_definition.values.artifacts.artifact_reference import ArtifactReference
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FixedVelocity, IncompressibleNavierStokes, NoSlipWall, WallShear
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions.values import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters

L = 1.0                    # m, cube side
U_lid = 1.0                # m/s, lid speed
rho = 1.0                  # kg/m^3
reynolds_numbers = (100.0, 400.0)
levels = (8, 11, 14)       # cells per cube side (the half cube has half as many across z)
grading = 1.2              # tanh stretching parameter (0 = uniform)
print(f"cube side {L} m, lid speed {U_lid} m/s, Re = {reynolds_numbers}")

# 3D reference values (mid-plane z = L/2): u_min on the vertical centreline and its height,
# v_min on the horizontal centreline and its x position
reference_3d = {
    100.0: {"Oueslati et al. 2011": (-0.21361, 0.47702, -0.24971, 0.81026),
            "Ding et al. 2006": (-0.205, 0.48, -0.246, 0.80)},
    400.0: {"Oueslati et al. 2011": (-0.22979, 0.22433, -0.37685, 0.87127),
            "Ding et al. 2006": (-0.231, 0.24, -0.371, 0.88)},
}
# 2D cavity, Ghia et al. (1982), Table I: u on the vertical centreline
ghia_y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813,
                   0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0])
ghia_u = {
    100.0: np.array([1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581,
                     -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0]),
    400.0: np.array([1.0, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477,
                     -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.0]),
}
# 2D cavity, Ghia et al. (1982), Table II: extremum of v on the horizontal centreline
ghia_vmin = {100.0: (-0.24533, 0.8047), 400.0: (-0.44993, 0.8594)}


# ## Geometry and mesh
#
# The computational box is the half cube $[0,L]\times[0,L]\times[0,L/2]$. The structured mesher
# produces a uniform grid, which is then stretched with
# $s\mapsto\tfrac12\left[1+\tanh(\beta(2s-1))/\tanh\beta\right]$ in $x$ and $y$ (clustering at
# both walls) and with the half of that map in $z$ (clustering at the end wall $z=0$, the widest
# cells at the mid-plane).
#
# Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | lid $y=L$ | moving wall, $\mathbf u=(U,0,0)$ |
# | bottom $y=0$, upstream $x=0$, downstream $x=L$, end wall $z=0$ | no slip, $\mathbf u=0$ |
# | mid-plane $z=L/2$ | symmetry: $w=0$ and no shear stress |
#
# The lid speed jumps to zero where the lid meets the fixed walls, a singularity of the
# continuum problem. As usual in discrete cavity computations, the lid velocity is $U$ everywhere except
# in the wall-adjacent cell along these edges, where it falls linearly to zero; the transition
# shrinks with the mesh. The pressure level is fixed by a zero mean value.

def stretch(s, beta=grading):
    return 0.5 * (1 + np.tanh(beta * (2 * s - 1)) / np.tanh(beta))


def graded(points):
    p = np.array(points, dtype=float)
    p[:, 0] = stretch(p[:, 0] / L) * L
    p[:, 1] = stretch(p[:, 1] / L) * L
    p[:, 2] = stretch(p[:, 2] / L) * L      # z in [0, L/2]: half of the symmetric map
    return p


class GradedHexMesh(StructuredHexahedralMesh):
    """Structured hexahedra moved onto the graded tensor grid (topology and face groups unchanged)."""

    def __init__(self, geometry, cells):
        super().__init__(geometry, cells)
        data = trilinear_hexahedral_geometry(graded(self.coordinates), self.cells)
        self.coordinates, self.cell_volumes = data.coordinates, data.volumes
        self.mesh_id = "graded-" + self.mesh_id + sha256(self.coordinates.tobytes()).hexdigest()[:16]
        self._carrier = self._build_carrier()


class GradedTetMesh(StructuredTetrahedralMesh):
    """Kuhn tetrahedra (six per hexahedron) moved onto the same graded tensor grid."""

    def __init__(self, geometry, cells, periodic_pairs=()):
        super().__init__(geometry, cells, periodic_pairs)
        data = tetrahedral_geometry(graded(self.coordinates), self.cells)
        self.coordinates, self.cell_volumes = data.coordinates, data.volumes
        self.mesh_id = "graded-" + self.mesh_id + sha256(self.coordinates.tobytes()).hexdigest()[:16]
        self._carrier = replace(
            self._carrier, carrier_id=self.mesh_id,
            topology=Topology(3, self.cell_count, "cell", self.mesh_id),
            coordinates=CoordinateSet(tuple(map(tuple, self.coordinates)), unit="m"),
            measures=tuple(self.cell_volumes), source_mesh=ArtifactReference(self.mesh_id, "1"))


box = Box(length=L, width=L, height=L / 2)
for n in levels:
    edges = stretch(np.linspace(0, 1, n + 1)) * L
    print(f"{n:2d} cells per side: {n * n * (n // 2):5d} hexahedra in the half cube, wall cell {edges[1]:.4f} m, "
          f"centre cell {edges[n // 2 + 1] - edges[n // 2]:.4f} m")


# ## Model

@dataclass(frozen=True)
class FreeSlip:
    """Wall-shear law with zero tangential traction: together with WallShear a symmetry plane."""

    def traction_and_slope(self, speed, distance, density, viscosity):
        zero = np.zeros_like(np.asarray(speed, dtype=float))
        return zero, zero


def solve(n, reynolds, element="hex", start=None):
    begin = time.perf_counter()
    if element == "hex":
        mesh = StructuredHexahedralMesher(cells=(n, n, n // 2), mesh_factory=GradedHexMesh).generate(box)
    else:
        mesh = StructuredTetrahedralMesher(cells=(n, n, n // 2), mesh_factory=GradedTetMesh).generate(box)
    edge_cell = stretch(1.0 / n) * L

    def lid(x, y, z):
        ramp = np.clip(np.minimum.reduce([x, L - x, z]) / edge_cell, 0.0, 1.0)
        return np.column_stack((U_lid * ramp, np.zeros_like(x), np.zeros_like(x)))

    fluid = NewtonianFluid(density=rho, dynamic_viscosity=rho * U_lid * L / reynolds)
    model = IncompressibleNavierStokes(domain=box.volume, material=fluid)
    b = box.boundaries
    problem = Problem(
        name="Lid-driven cube", geometry=box, physics=(model,),
        boundary_conditions=(
            FixedVelocity(boundary=b.y_max, physics=model, value=lid),
            NoSlipWall(boundary=b.x_min | b.x_max | b.y_min | b.z_min, physics=model),
            WallShear(boundary=b.z_max, physics=model, law=FreeSlip()),
        ),
        constraints=(MeanValue(model.pressure, 0.0),),
        initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
    )
    guess = (0.0, 0.0, 0.0) if start is None else start
    numerics = Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        nonlinear_solver=Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-9,
                                relative_tolerance=1e-9, maximum_iterations=30),
        initial_guess={model.velocity: guess},
    )
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    run = dict(n=n, reynolds=reynolds, element=element, cells=mesh.cell_count,
               velocity=result.field(model.velocity), pressure=result.field(model.pressure))
    run.update(centreline_values(run))
    run["seconds"] = time.perf_counter() - begin
    return run


# ## Post-processing
#
# All profiles are taken in the mid-plane $z=L/2$: $u(y)$ on the vertical centreline $x=L/2$ and
# $v(x)$ on the horizontal centreline $y=L/2$. The extrema are located by a bounded scalar
# minimisation of the finite-element field. The **mass balance** is checked by the net volume flux
# through the vertical mid-section $x=L/2$, which must vanish in a closed cavity; it is integrated
# with 4 x 4 Gauss points per grid cell and compared with the flux of $|u|$.

def velocity_at(run, points):
    return np.asarray(run["velocity"].evaluate(np.atleast_2d(points)))


def centreline_values(run):
    ys = np.linspace(0, L, 401)
    u_line = velocity_at(run, np.column_stack((np.full_like(ys, L / 2), ys, np.full_like(ys, L / 2))))[:, 0]
    k = np.argmin(u_line)
    u_min = minimize_scalar(lambda y: velocity_at(run, [L / 2, y, L / 2])[0, 0],
                            bounds=(ys[max(k - 2, 0)], ys[min(k + 2, 400)]), method="bounded",
                            options={"xatol": 1e-6})
    v_line = velocity_at(run, np.column_stack((ys, np.full_like(ys, L / 2), np.full_like(ys, L / 2))))[:, 1]
    k = np.argmin(v_line)
    v_min = minimize_scalar(lambda x: velocity_at(run, [x, L / 2, L / 2])[0, 1],
                            bounds=(ys[max(k - 2, 0)], ys[min(k + 2, 400)]), method="bounded",
                            options={"xatol": 1e-6})
    # net flux through x = L/2 over the half section (y in [0, L], z in [0, L/2])
    gp, gw = np.polynomial.legendre.leggauss(4)
    gp, gw = (gp + 1) / 2, gw / 2
    yb = stretch(np.linspace(0, 1, run["n"] + 1)) * L
    zb = stretch(np.linspace(0, 1, run["n"] // 2 + 1) / 2) * L
    yq, wy = (yb[:-1, None] + np.diff(yb)[:, None] * gp).ravel(), (np.diff(yb)[:, None] * gw).ravel()
    zq, wz = (zb[:-1, None] + np.diff(zb)[:, None] * gp).ravel(), (np.diff(zb)[:, None] * gw).ravel()
    Yq, Zq = np.meshgrid(yq, zq, indexing="ij")
    uq = velocity_at(run, np.column_stack((np.full(Yq.size, L / 2), Yq.ravel(), Zq.ravel())))[:, 0].reshape(Yq.shape)
    W = np.outer(wy, wz)
    return dict(ys=ys, u_line=u_line, v_line=v_line, u_min=float(u_min.fun), y_umin=float(u_min.x),
                v_min=float(v_min.fun), x_vmin=float(v_min.x),
                net_flux=float(np.sum(W * uq)), abs_flux=float(np.sum(W * np.abs(uq))),
                w_mid=float(np.max(np.abs(velocity_at(run, np.column_stack((ys, ys, np.full_like(ys, L / 2))))[:, 2]))))


# ## Solve
#
# Re = 100 on the three hexahedral meshes (each started from the previous one), the tetrahedral
# P2/P1 comparison on the middle grid, and Re = 400 on the two finer meshes (the middle one
# started from the Re = 100 solution, the finest from the Re = 400 solution on the middle mesh).
# Starting values are passed on as velocities sampled on a regular grid, so the solver objects of
# a finished run can be released and the memory footprint stays at that of a single solve.

def report(r, label):
    print(f"Re = {r['reynolds']:3.0f}, {label}, {r['n']:2d} cells/side ({r['cells']:5d} cells): u_min = "
          f"{r['u_min']:.5f} at y = {r['y_umin']:.4f}, v_min = {r['v_min']:.5f} at x = {r['x_vmin']:.4f}  "
          f"({r['seconds']:.0f} s)")


def sampled_start(run):
    """Newton starting value: the velocity sampled on a 33 x 33 x 17 grid, interpolated trilinearly."""
    ax_xy, ax_z = np.linspace(0, L, 33), np.linspace(0, L / 2, 17)
    P = np.stack(np.meshgrid(ax_xy, ax_xy, ax_z, indexing="ij"), axis=-1).reshape(-1, 3)
    table = RegularGridInterpolator((ax_xy, ax_xy, ax_z), velocity_at(run, P).reshape(33, 33, 17, 3),
                                    bounds_error=False, fill_value=None)
    return lambda x, y, z: table(np.column_stack((x, y, z)))


def release(run):
    """Keep the sampled profiles, drop the solver objects."""
    run.pop("velocity", None)
    run.pop("pressure", None)
    gc.collect()


runs = {100.0: [], 400.0: []}
starts = {}
previous = None
for n in levels[:2]:
    runs[100.0].append(solve(n, 100.0, start=previous))
    report(runs[100.0][-1], "Q2/Q1 hex")
    starts[(100.0, n)] = previous = sampled_start(runs[100.0][-1])
    release(runs[100.0][-1])
tet = solve(levels[1], 100.0, element="tet", start=starts[(100.0, levels[1])])
report(tet, "P2/P1 tet")
release(tet)
runs[400.0].append(solve(levels[1], 400.0, start=starts[(100.0, levels[1])]))
report(runs[400.0][-1], "Q2/Q1 hex")
starts[(400.0, levels[1])] = sampled_start(runs[400.0][-1])
release(runs[400.0][-1])
runs[100.0].append(solve(levels[2], 100.0, start=starts[(100.0, levels[1])]))
report(runs[100.0][-1], "Q2/Q1 hex")
release(runs[100.0][-1])
runs[400.0].append(solve(levels[2], 400.0, start=starts[(400.0, levels[1])]))
report(runs[400.0][-1], "Q2/Q1 hex")


# ## Comparison with the reference
#
# **Centreline extrema** on the finest mesh against the published 3D results for the cube, and
# against the 2D cavity (Ghia et al.; the tabulated grid point nearest to the extremum).

print(f"{'':6s}{'quantity':16s}{'OneSimulation':>14s}" + "".join(f"{name:>24s}" for name in reference_3d[100.0])
      + f"{'2D cavity (Ghia)':>20s}")
for reynolds in reynolds_numbers:
    r = runs[reynolds][-1]
    refs = reference_3d[reynolds]
    k = np.argmin(ghia_u[reynolds])
    rows = (("u_min / U", r["u_min"], 0, ghia_u[reynolds][k]),
            ("y(u_min) / L", r["y_umin"], 1, ghia_y[k]),
            ("v_min / U", r["v_min"], 2, ghia_vmin[reynolds][0]),
            ("x(v_min) / L", r["x_vmin"], 3, ghia_vmin[reynolds][1]))
    for name, value, i, two_d in rows:
        cells = "".join(f"{ref[i]:12.5f} ({value / ref[i] - 1:+6.1%})" for ref in refs.values())
        print(f"Re={reynolds:<3.0f} {name:16s}{value:14.5f}{cells}{two_d:12.5f} ({value / two_d - 1:+6.1%})")

# **Mesh study** at Re = 100 (cell size $h\propto1/n$): change between the two finest meshes and,
# where the three values are monotone, the observed order and the Richardson estimate; at
# Re = 400 the change between the two finer meshes. The convergence of the extrema is slowed by
# the lid-edge singularity: the transition of the lid speed to zero shrinks with the mesh, so the
# boundary data themselves converge only like $h$. The tetrahedral P2/P1 solution is compared with
# the hexahedral Q2/Q1 solutions on the same grid and on the finest grid.

h = 1.0 / np.array(levels, float)


def richardson(values):
    f1, f2, f3 = values
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda q: (h[0]**q - h[1]**q) / (h[1]**q - h[2]**q) - ratio, 0.3, 12.0)
    except ValueError:
        return np.nan, np.nan
    return p, f3 + (f3 - f2) / ((h[1] / h[2]) ** p - 1)


print("\nMesh study, Re = 100 (Q2/Q1 hexahedra, " + ", ".join(f"{n}" for n in levels) + " cells per side)")
for key in ("u_min", "v_min", "y_umin", "x_vmin"):
    values = [r[key] for r in runs[100.0]]
    p, rich = richardson(values)
    line = f"  {key:7s} " + "  ".join(f"{v:.5f}" for v in values) + \
        f"   change last two meshes {abs(values[2] / values[1] - 1):.1e}"
    line += f", observed order {p:.2f}, Richardson {rich:.5f}" if np.isfinite(p) else ", not monotone (no Richardson estimate)"
    print(line)
print("Re = 400, change between the two finer meshes: "
      + ", ".join(f"{key} {abs(runs[400.0][1][key] / runs[400.0][0][key] - 1):.1e}"
                  for key in ("u_min", "v_min", "y_umin", "x_vmin")))
print(f"P2/P1 tetrahedra ({levels[1]} cells/side): u_min {tet['u_min']:.5f}, v_min {tet['v_min']:.5f};  "
      f"Q2/Q1 hexahedra: {runs[100.0][1]['u_min']:.5f}, {runs[100.0][1]['v_min']:.5f} ({levels[1]} cells/side), "
      f"{runs[100.0][2]['u_min']:.5f}, {runs[100.0][2]['v_min']:.5f} ({levels[2]} cells/side)")

# **Mass balance and symmetry.** Net flux through the mid-section $x=L/2$ relative to the flux
# of $|u|$, and the largest $|w|$ on the symmetry plane (weakly enforced, should be close to zero).

for reynolds in reynolds_numbers:
    for r in runs[reynolds]:
        print(f"Re = {reynolds:3.0f}, {r['n']:2d} cells/side: net flux / |flux| through x = L/2: "
              f"{r['net_flux'] / r['abs_flux']:+.1e},  max |w| on the mid-plane: {r['w_mid']:.1e} U")

# **End-wall effect.** Ratio of the 3D to the 2D extremum of the centreline velocity:

for reynolds in reynolds_numbers:
    r = runs[reynolds][-1]
    print(f"Re = {reynolds:3.0f}: u_min(3D) / u_min(2D) = {r['u_min'] / ghia_u[reynolds].min():.3f}, "
          f"v_min(3D) / v_min(2D) = {r['v_min'] / ghia_vmin[reynolds][0]:.3f}")


# ## Figure
#
# Top, from left: 3D view at Re = 400 - the speed on the mid-plane and
# streamlines that spiral along the vortex axis from the end wall towards the mid-plane; speed and
# streamlines in the mid-plane; the secondary flow in the cross-section $x=L/2$ (colour: the
# through-flow velocity $u$, lines: in-plane streamlines of $(w, v)$ with the end-wall vortices).
# Bottom: centreline profiles at Re = 100 and 400 against the 3D references and the 2D cavity,
# and the mesh convergence of the extrema, normalised by the values of Oueslati et al. (the open
# symbols at the left are the values of Ding et al., which marks the spread of the references).

final = runs[400.0][-1]
n_s = 41
gx = stretch(np.linspace(0, 1, n_s)) * L
gz = stretch(np.linspace(0, 1, 2 * n_s - 1)) * L           # full cube, mirrored from the half
GX, GY, GZ = np.meshgrid(gx, gx, gz[: n_s], indexing="ij")
half_field = velocity_at(final, np.column_stack((GX.ravel(), GY.ravel(), np.minimum(GZ.ravel(), L / 2))))
half_field = half_field.reshape(GX.shape + (3,))
mirror = half_field[:, :, -2::-1].copy()
mirror[..., 2] *= -1
full_field = np.concatenate((half_field, mirror), axis=2)
interp = RegularGridInterpolator((gx, gx, gz), full_field, bounds_error=False, fill_value=0.0)
speed = np.linalg.norm(full_field, axis=3)


def streamline(seed, t_end, dt=0.02):
    path = [np.asarray(seed, float)]
    for _ in range(int(t_end / dt)):
        p = path[-1]
        k1 = interp(p)[0]
        k2 = interp(p + 0.5 * dt * k1)[0]
        k3 = interp(p + 0.5 * dt * k2)[0]
        k4 = interp(p + dt * k3)[0]
        path.append(np.clip(p + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0.0, L))
    return np.array(path)


fig = plt.figure(figsize=(18, 10.5), constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 0.9))
norm = Normalize(0, U_lid)

ax = fig.add_subplot(grid[0, 0], projection="3d")
ax.computed_zorder = False
k_mid = n_s - 1
Xm, Ym = np.meshgrid(gx, gx, indexing="ij")
ax.plot_surface(Xm, np.full_like(Xm, L / 2), Ym, facecolors=FIELD_CMAP(norm(speed[:, :, k_mid])),
                rstride=1, cstride=1, shade=False, alpha=0.55, linewidth=0, zorder=1)
for seed, colour in (((0.62, 0.62, 0.04), COPPER), ((0.55, 0.70, 0.08), COPPER), ((0.62, 0.62, 0.96), GREEN),
                     ((0.55, 0.70, 0.92), GREEN)):
    path = streamline(seed, t_end=45.0)
    ax.plot(path[:, 0], path[:, 2], path[:, 1], color=colour, lw=0.9, zorder=5)
for a in (0, L):
    for b in (0, L):
        ax.plot([0, L], [a, a], [b, b], color=INK, lw=0.6, zorder=6)
        ax.plot([a, a], [0, L], [b, b], color=INK, lw=0.6, zorder=6)
        ax.plot([a, a], [b, b], [0, L], color=INK, lw=0.6, zorder=6)
ax.plot([0, L], [L / 2, L / 2], [L, L], color=COPPER, lw=2.5, zorder=7)
ax.text(0.05, L / 2, 1.08 * L, "lid  ->", color=COPPER, fontsize=9)
ax.set_zlabel("y / m", labelpad=2)
ax.set(xlabel="x / m", ylabel="z / m", xlim=(0, L), ylim=(0, L), zlim=(0, L),
       title="Re = 400: speed on the mid-plane z = L/2 and streamlines\nspiralling from the end walls towards the mid-plane")
ax.view_init(elev=22, azim=-50)
ax.set_box_aspect((1, 1, 1))

ax = fig.add_subplot(grid[0, 1])
Xg, Yg = np.meshgrid(gx, gx, indexing="ij")
filled = ax.contourf(Xg, Yg, speed[:, :, k_mid], levels=np.linspace(0, U_lid, 41), cmap=FIELD_CMAP, extend="max")
ue = np.linspace(0, L, 161)
UI = interp(np.stack(np.meshgrid(ue, ue, [L / 2], indexing="ij"), axis=-1).reshape(-1, 3)).reshape(161, 161, 3)
ax.streamplot(ue, ue, UI[:, :, 0].T, UI[:, :, 1].T, density=1.8, color=INK, linewidth=0.55, arrowsize=0.6)
ax.set_aspect("equal")
ax.set(xlim=(0, L), ylim=(0, L), xlabel="x / m", ylabel="y / m", title="Re = 400: mid-plane z = L/2, speed and streamlines")
fig.colorbar(filled, ax=ax, label="|u| / U", shrink=0.85, format="%.1f")

ax = fig.add_subplot(grid[0, 2])
cross = interp(np.stack(np.meshgrid([L / 2], ue, ue, indexing="ij"), axis=-1).reshape(-1, 3)).reshape(161, 161, 3)
limit = np.abs(cross[:, :, 0]).max()
filled = ax.contourf(ue, ue, cross[:, :, 0], levels=np.linspace(-limit, limit, 41), cmap=DIVERGING_CMAP,
                     norm=TwoSlopeNorm(0.0, -limit, limit))
ax.streamplot(ue, ue, cross[:, :, 2], cross[:, :, 1], density=2.2, color=INK, linewidth=0.5, arrowsize=0.6)
ax.set_aspect("equal")
ax.set(xlim=(0, L), ylim=(0, L), xlabel="z / m (end walls at z = 0 and z = L)", ylabel="y / m",
       title="Re = 400: cross-section x = L/2, u (colour) and secondary flow (w, v)")
fig.colorbar(filled, ax=ax, label="u / U", shrink=0.85, format="%.2f")

ax = fig.add_subplot(grid[1, 0])
for reynolds, colour in zip(reynolds_numbers, (GREEN, COPPER)):
    r = runs[reynolds][-1]
    ax.plot(r["u_line"], r["ys"], "-", color=colour, lw=1.8, label=f"3D cube, Re = {reynolds:.0f} (Q2/Q1)")
    ax.plot(ghia_u[reynolds], ghia_y, "o", mfc="none", color=colour, ms=5, label=f"2D cavity, Ghia et al., Re = {reynolds:.0f}")
    for (name, ref), marker in zip(reference_3d[reynolds].items(), ("*", "X")):
        ax.plot(ref[0], ref[1], marker, color=INK, ms=10 if marker == "*" else 7,
                label=f"3D: {name}" if reynolds == 100.0 else None)
ax.plot(tet["u_line"], tet["ys"], "--", color="#8B9A8D", lw=1.2, label=f"3D, Re = 100, P2/P1 tetrahedra ({levels[1]} cells/side)")
ax.axvline(0, color="#999", lw=0.8)
ax.set(xlabel="u / U on the vertical centreline (x = L/2, z = L/2)", ylabel="y / L",
       title="Horizontal velocity on the vertical centreline")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="center right")

ax = fig.add_subplot(grid[1, 1])
for reynolds, colour in zip(reynolds_numbers, (GREEN, COPPER)):
    r = runs[reynolds][-1]
    ax.plot(r["ys"], r["v_line"], "-", color=colour, lw=1.8, label=f"3D cube, Re = {reynolds:.0f}")
    ax.plot(*ghia_vmin[reynolds][::-1], "o", mfc="none", color=colour, ms=7, label=f"2D cavity minimum, Re = {reynolds:.0f}")
    for (name, ref), marker in zip(reference_3d[reynolds].items(), ("*", "X")):
        ax.plot(ref[3], ref[2], marker, color=INK, ms=10 if marker == "*" else 7,
                label=f"3D: {name}" if reynolds == 100.0 else None)
ax.axhline(0, color="#999", lw=0.8)
ax.set(xlabel="x / L", ylabel="v / U on the horizontal centreline (y = L/2, z = L/2)",
       title="Vertical velocity on the horizontal centreline")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower left")

ax = fig.add_subplot(grid[1, 2])
for key, index, marker, name in (("u_min", 0, "o", "u_min"), ("v_min", 2, "s", "v_min")):
    for reynolds, colour in zip(reynolds_numbers, (GREEN, COPPER)):
        ref = reference_3d[reynolds]["Oueslati et al. 2011"][index]
        ns = [r["n"] for r in runs[reynolds]]
        ax.plot(1.0 / np.array(ns), [r[key] / ref for r in runs[reynolds]], marker + ("-" if key == "u_min" else "--"),
                color=colour, mfc="none" if key == "v_min" else colour, label=f"{name}, Re = {reynolds:.0f}")
        ding = reference_3d[reynolds]["Ding et al. 2006"][index] / ref
        ax.plot([0.0], [ding], marker, color=colour, ms=9, mfc="none", mew=0.8, alpha=0.6)
ax.axhline(1.0, color=INK, lw=0.8, ls=":")
ax.set_xticks([0.0] + [1.0 / n for n in levels], ["Ding\n(41^3)"] + [f"1/{n}" for n in levels])
ax.set(xlim=(-0.012, 0.135), xlabel="cell size (1 / cells per side)", ylabel="value / value of Oueslati et al. (2011)",
       title="Mesh convergence of the centreline extrema")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=2)
plt.show()


# ## Going further
#
# Raise the Reynolds number to 1000: the primary vortex moves to the cavity centre, the
# end-wall vortices grow and, near $Re\approx1900$, the steady flow becomes unstable to
# travelling waves (Feldman & Gelfgat 2010) - a time-dependent run with `Simulation.run` and a
# BDF time integrator. Change the spanwise aspect ratio of the box to see how the mid-plane flow
# approaches the 2D cavity for long cavities, or make the end walls periodic to recover the
# two-dimensional benchmark exactly. Replacing the symmetry plane by the full cube lets
# symmetry-breaking states appear at higher Reynolds numbers.
