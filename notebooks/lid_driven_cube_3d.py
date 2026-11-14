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

import gc
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import StructuredHexahedralMesher, StructuredTetrahedralMesher, Tanh
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FixedVelocity, FreeSlipLaw, IncompressibleNavierStokes, NoSlipWall, WallShear
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

L = 1.0
U_lid = 1.0
rho = 1.0
reynolds_numbers = (100.0, 400.0)
levels = (8, 11, 14)
grading = 1.2
print(f"cube side {L} m, lid speed {U_lid} m/s, Re = {reynolds_numbers}")

# 3D reference values (mid-plane z = L/2): u_min on the vertical centreline and its height,
# v_min on the horizontal centreline and its x position
reference_3d = {
    100.0: {"Oueslati et al. 2011": (-0.21361, 0.47702, -0.24971, 0.81026),
            "Ding et al. 2006": (-0.205, 0.48, -0.246, 0.80)},
    400.0: {"Oueslati et al. 2011": (-0.22979, 0.22433, -0.37685, 0.87127),
            "Ding et al. 2006": (-0.231, 0.24, -0.371, 0.88)},
}
ghia_y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813,
                   0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0])
ghia_u = {
    100.0: np.array([1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581,
                     -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0]),
    400.0: np.array([1.0, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477,
                     -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.0]),
}
ghia_vmin = {100.0: (-0.24533, 0.8047), 400.0: (-0.44993, 0.8594)}


# ## Geometry and mesh
#
# The computational box is the half cube $[0,L]\times[0,L]\times[0,L/2]$. The structured mesher
# grades its grid with the hyperbolic-tangent clustering (`Tanh`)
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


def cavity_grading(n):
    """Cells clustered at both walls in x and y (tanh), at the end wall z = 0 in the half depth."""
    return Tanh(n, grading), Tanh(n, grading), Tanh(n // 2, grading, toward="start")


box = Box(length=L, width=L, height=L / 2)
for n in levels:
    edges = stretch(np.linspace(0, 1, n + 1)) * L
    print(f"{n:2d} cells per side: {n * n * (n // 2):5d} hexahedra in the half cube, wall cell {edges[1]:.4f} m, "
          f"centre cell {edges[n // 2 + 1] - edges[n // 2]:.4f} m")


# ## Model

def solve(n, reynolds, element="hex", start=None):
    begin = time.perf_counter()
    if element == "hex":
        mesh = StructuredHexahedralMesher(cells=(n, n, n // 2), grading=cavity_grading(n)).generate(box)
    else:
        mesh = StructuredTetrahedralMesher(cells=(n, n, n // 2), grading=cavity_grading(n)).generate(box)
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
            WallShear(boundary=b.z_max, physics=model, law=FreeSlipLaw()),
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
    estimate = convergence(values, h)
    return (estimate.order, estimate.extrapolated) if estimate.order is not None else (np.nan, np.nan)


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
gz = stretch(np.linspace(0, 1, 2 * n_s - 1)) * L
GX, GY, GZ = np.meshgrid(gx, gx, gz[:n_s], indexing="ij")
half_field = velocity_at(final, np.column_stack((GX.ravel(), GY.ravel(), np.minimum(GZ.ravel(), L / 2)))).reshape(GX.shape + (3,))
mirror = half_field[:, :, -2::-1].copy()
mirror[..., 2] *= -1
interp = RegularGridInterpolator((gx, gx, gz), np.concatenate((half_field, mirror), axis=2), bounds_error=False, fill_value=0.0)
ue = np.linspace(0, L, 161)
mid = interp(np.stack(np.meshgrid(ue, ue, [L / 2], indexing="ij"), axis=-1).reshape(-1, 3)).reshape(161, 161, 3)
cross = interp(np.stack(np.meshgrid([L / 2], ue, ue, indexing="ij"), axis=-1).reshape(-1, 3)).reshape(161, 161, 3)
u_curves, u_points, v_curves, v_points, study, study_points = {}, {}, {}, {}, {}, {}
for reynolds in reynolds_numbers:
    r = runs[reynolds][-1]
    u_curves[f"3D cube, Re = {reynolds:.0f} (Q2/Q1)"] = (r["u_line"], r["ys"])
    v_curves[f"3D cube, Re = {reynolds:.0f}"] = (r["ys"], r["v_line"])
    u_points[f"2D cavity, Ghia et al., Re = {reynolds:.0f}"] = (ghia_u[reynolds], ghia_y)
    v_points[f"2D cavity minimum, Re = {reynolds:.0f}"] = ([ghia_vmin[reynolds][1]], [ghia_vmin[reynolds][0]])
    for name, ref in reference_3d[reynolds].items():
        u_points[f"3D: {name}, Re = {reynolds:.0f}"] = ([ref[0]], [ref[1]])
        v_points[f"3D: {name}, Re = {reynolds:.0f}"] = ([ref[3]], [ref[2]])
    for key, index in (("u_min", 0), ("v_min", 2)):
        published = reference_3d[reynolds]["Oueslati et al. 2011"][index]
        study[f"{key}, Re = {reynolds:.0f}"] = (1.0 / np.array([run["n"] for run in runs[reynolds]]),
                                              [run[key] / published for run in runs[reynolds]])
        study_points[f"{key}, Re = {reynolds:.0f}"] = ([0.0], [reference_3d[reynolds]["Ding et al. 2006"][index] / published])
u_curves[f"3D, Re = 100, P2/P1 tetrahedra ({levels[1]} cells/side)"] = (tet["u_line"], tet["ys"])
fig = plots.figure(layout=[["speed"] * 3 + ["flow"] * 3 + ["cross"] * 3 + ["secondary"] * 3,
                           ["u"] * 4 + ["v"] * 4 + ["study"] * 4], size=(18, 10.5))
plots.heatmap(ue, ue, np.linalg.norm(mid, axis=2), limits=(0, U_lid), label="|u| / U", xlabel="x [m]", ylabel="y [m]",
              title="Re = 400: speed on the mid-plane z = L/2", ax=fig["speed"])
plots.streamlines(ue, ue, mid[:, :, 0], mid[:, :, 1], density=1.8, label="|u| / U", xlabel="x [m]", ylabel="y [m]",
                  title="Re = 400: streamlines on the mid-plane", ax=fig["flow"])
plots.heatmap(ue, ue, cross[:, :, 0].T, colormap="diverging", label="u / U", xlabel="z [m] (end walls at z = 0 and z = L)",
              ylabel="y [m]", title="Re = 400: u on the cross-section x = L/2", ax=fig["cross"])
plots.streamlines(ue, ue, cross[:, :, 2].T, cross[:, :, 1].T, density=2.2, label="speed / U", xlabel="z [m]", ylabel="y [m]",
                  title="Re = 400: secondary flow (w, v) on x = L/2", ax=fig["secondary"])
plots.line(y=u_curves, points=u_points, styles={f"3D, Re = 100, P2/P1 tetrahedra ({levels[1]} cells/side)": "dashed"},
           vlines=0.0, xlabel="u / U on the vertical centreline (x = L/2, z = L/2)", ylabel="y / L",
           title="Horizontal velocity on the vertical centreline", legend="center right", ax=fig["u"])
plots.line(y=v_curves, points=v_points, hlines=0.0, xlabel="x / L",
           ylabel="v / U on the horizontal centreline (y = L/2, z = L/2)",
           title="Vertical velocity on the horizontal centreline", legend="lower left", ax=fig["v"])
plots.line(y=study, points=study_points, hlines=1.0, markers=True, xlimits=(-0.012, 0.135),
           xlabel="cell size (1 / cells per side); open markers at 0: Ding et al. 2006 (41³)",
           ylabel="value / value of Oueslati et al. (2011)", title="Mesh convergence of the centreline extrema", ax=fig["study"])
plots.show()


# ## Going further
#
# Raise the Reynolds number to 1000: the primary vortex moves to the cavity centre, the
# end-wall vortices grow and, near $Re\approx1900$, the steady flow becomes unstable to
# travelling waves (Feldman & Gelfgat 2010) - a time-dependent run with `Simulation.run` and a
# BDF time integrator. Change the spanwise aspect ratio of the box to see how the mid-plane flow
# approaches the 2D cavity for long cavities, or make the end walls periodic to recover the
# two-dimensional benchmark exactly. Replacing the symmetry plane by the full cube lets
# symmetry-breaking states appear at higher Reynolds numbers.
