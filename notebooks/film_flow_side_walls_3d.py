"""OneSimulation example: gravity-driven film flow down an inclined open channel with side walls.
Requires the installed, activated OneSimulation runtime.

A liquid film of depth H runs down an open rectangular channel of width W inclined at an angle
beta. The bottom and both side walls are no-slip, the flat free surface is stress-free. The
cross-section velocity field u(y, z) is computed in 3D with Taylor-Hood finite elements on a
streamwise-periodic slab and, as a second method, with staggered (MAC) finite differences. Both
are graded against the exact Fourier-series solution, the flow rate against its closed form as a
function of the aspect ratio W/H, and the centre plane against the Nusselt film for wide channels.
"""


# # Film flow down an inclined channel with side walls
#
# A Newtonian liquid (viscosity $\mu$, density $\rho$) flows down an open channel of width $W$
# that is inclined at the angle $\beta$ to the horizontal. The film has the depth $H$, its free
# surface is flat and free of shear stress, and the liquid sticks to the bottom and to the two
# side walls. Far from the inlet the flow is fully developed: it no longer changes along the
# channel, but it varies across the whole cross-section, because the side walls slow down the
# liquid next to them. The classical Nusselt film $u_N(z)=\frac{\rho g\sin\beta}{\mu}(Hz-z^2/2)$
# only describes the middle of a very wide channel.
#
# Coordinates: $x$ runs down the channel, $y\in[0,W]$ across it, $z\in[0,H]$ from the bottom to
# the free surface. Gravity has the components $(\rho g\sin\beta,\,0,\,-\rho g\cos\beta)$.
#
# **Exact solution.** For the fully developed flow the momentum balance reduces to
# $\mu\nabla^2u=-\rho g\sin\beta$ on the cross-section, with $u=0$ on the walls and
# $\partial u/\partial z=0$ on the free surface. The free surface acts like a mirror plane, so the
# film is exactly one half of a closed duct of depth $2H$ (the classical rectangular-duct
# solution; F. M. White, *Viscous Fluid Flow*, 3rd ed., McGraw-Hill 2006, Sec. 3-3.3;
# R. K. Shah and A. L. London, *Laminar Flow Forced Convection in Ducts*, Academic Press 1978,
# Ch. VII). With $\lambda_n=(2n+1)\pi/(2H)$:
#
# $$u(y,z)=\frac{\rho g\sin\beta}{\mu}\left[Hz-\frac{z^2}{2}-\sum_{n=0}^{\infty}\frac{2}{H\lambda_n^3}
#   \sin(\lambda_n z)\,\frac{\cosh\lambda_n(y-W/2)}{\cosh(\lambda_n W/2)}\right],$$
#
# $$\frac{Q}{W\,q_N}=1-\frac{12H}{W}\sum_{n=0}^{\infty}\frac{\tanh(\lambda_n W/2)}{(\lambda_n H)^5},
#   \qquad q_N=\frac{\rho g\sin\beta\,H^3}{3\mu},$$
#
# where $q_N$ is the Nusselt flow rate per unit width. For wide channels
# $Q/(Wq_N)\to 1-1.2605\,H/W$: each side wall removes the flow of a strip about $0.63H$ wide.
#
# **Numerical methods.** Both methods solve the full 3D incompressible Navier-Stokes equations
# for $(u,v,w,p)$ on a short slab of the channel whose upstream and downstream faces are joined
# by a periodic condition, so that nothing about the cross-section flow is assumed.
#
# * Method 1 - continuous Galerkin finite elements, Taylor-Hood P2/P1 tetrahedra (quadratic
#   velocity, linear pressure), stationary, Newton iteration with a sparse direct LU solver.
# * Method 2 - finite differences on a staggered marker-and-cell (MAC) grid (F. H. Harlow and
#   J. E. Welch, Phys. Fluids 8, 2182, 1965): pressure in the cell centres, each velocity component
#   on the faces normal to it, second-order central differences, stationary, Newton iteration with
#   a sparse direct solver for the complete saddle-point system.
#
# **Reference.** The exact Fourier-series solution of the fully developed problem (field, surface
# velocity, flow rate), the Nusselt profile as the wide-channel limit of the centre plane, the
# hydrostatic pressure, and a three-level mesh study for each method.

import time
import numpy as np

from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import PeriodicPair, StructuredHexahedralMesher, StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import CentralDifference, CentralLaplacian, FiniteDifference
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FlowTraction, IncompressibleNavierStokes, NoSlipWall
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation import PeriodicBoundary
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# A glycerol-water mixture flows in a channel inclined by 10 degrees. The film depth is 2 mm;
# the channel width is varied from half the depth to eight times the depth.

H = 0.002
mu = 0.10
rho = 1200.0
g = 9.81
beta = np.deg2rad(10.0)
aspects = (0.5, 1.0, 2.0, 4.0, 8.0)
cells_depth = 16

G = rho * g * np.sin(beta)
u_nusselt = G * H**2 / (2 * mu)
q_nusselt = G * H**3 / (3 * mu)
p_bottom = rho * g * np.cos(beta) * H
reynolds = rho * q_nusselt / mu
print(f"Nusselt film: surface speed {1e3*u_nusselt:.3f} mm/s, flow rate per width {1e6*q_nusselt:.4f} mm^2/s, "
      f"Re = rho q_N / mu = {reynolds:.3f} (laminar)")


def exact_velocity(y, z, W, terms=200):
    """Exact streamwise velocity u(y, z) of the fully developed film (Fourier series)."""
    y, z = np.asarray(y, float), np.asarray(z, float)
    u = H * z - z**2 / 2
    s = np.abs(y - W / 2)
    for n in range(terms):
        lam = (2 * n + 1) * np.pi / (2 * H)
        ratio = np.exp(lam * (s - W / 2)) * (1 + np.exp(-2 * lam * s)) / (1 + np.exp(-lam * W))
        u = u - 2 / (H * lam**3) * np.sin(lam * z) * ratio
    return G / mu * u


def exact_flow_rate(W, terms=400):
    k = (2 * np.arange(terms) + 1) * np.pi / 2
    return q_nusselt * W * (1 - 12 * H / W * np.sum(np.tanh(k * W / (2 * H)) / k**5))


# ## Geometry and mesh
#
# The computational domain is a box: length $L_x$ down the channel (one cell for the finite
# elements, five for the difference stencils), width $W$ and depth $H$. The finite-element mesh
# is a structured grid of cubes of edge $H/16$, each split into six tetrahedra; the MAC grid uses
# hexahedral cells. With $W/H=8$ the finite-element mesh has 128 x 16 cells across the channel.
#
# Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | bottom $z=0$ and side walls $y=0$, $y=W$ | no slip, $\mathbf u=0$ |
# | free surface $z=H$ | traction-free, $\boldsymbol\sigma\cdot\mathbf n=0$ (no shear, zero gauge pressure) |
# | upstream and downstream faces $x=0$, $x=L_x$ | periodic pair |
#
# The traction-free surface also fixes the pressure level, so no pressure gauge is needed. The
# flat surface is the fully developed state; a surface tension meniscus at the side walls is not
# modelled.

def gravity(x, y, z):
    n = len(np.asarray(x))
    return np.column_stack((np.full(n, G), np.zeros(n), np.full(n, -rho * g * np.cos(beta))))


# ## Model

fluid = NewtonianFluid(density=rho, dynamic_viscosity=mu)


def solve(method, aspect, cells_deep):
    W = aspect * H
    cells_across = int(round(aspect * cells_deep))
    cells_along = 1 if method == "fem" else 5
    box = Box(length=cells_along * W / cells_across, width=W, height=H)
    model = IncompressibleNavierStokes(domain=box.volume, material=fluid, body_force=gravity)
    periodic = PeriodicPair(first=box.boundaries.x_min, second=box.boundaries.x_max)
    problem = Problem(
        name="Film flow in an inclined channel",
        geometry=box,
        physics=(model,),
        boundary_conditions=(
            NoSlipWall(boundary=box.boundaries.z_min | box.boundaries.y_min | box.boundaries.y_max, physics=model),
            FlowTraction(boundary=box.boundaries.z_max, physics=model, value=(0.0, 0.0, 0.0)),
            PeriodicBoundary(pair=periodic, physics=model),
        ),
        initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
    )
    newton = Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-12, relative_tolerance=1e-10,
                    maximum_iterations=20)
    if method == "fem":
        mesh = StructuredTetrahedralMesher(cells=(cells_along, cells_across, cells_deep)).generate(box)
        spatial = FiniteElement(element=TaylorHood())
    else:
        mesh = StructuredHexahedralMesher(cells=(cells_along, cells_across, cells_deep)).generate(box)
        spatial = FiniteDifference(gradient=CentralDifference(), diffusion=CentralLaplacian())
    start = time.perf_counter()
    result = Simulation(problem=problem, mesh=mesh,
                        numerics=Numerics(spatial=spatial, nonlinear_solver=newton,
                                          initial_guess={model.velocity: (0.0, 0.0, 0.0)})).solve_stationary()
    velocity, pressure = result.field(model.velocity), result.field(model.pressure)
    x_mid = box.length / 2

    gp, gw = np.polynomial.legendre.leggauss(4)
    gp, gw = (gp + 1) / 2, gw / 2
    yb, zb = np.linspace(0, W, cells_across + 1), np.linspace(0, H, cells_deep + 1)
    ys, wy = (yb[:-1, None] + np.diff(yb)[:, None] * gp).ravel(), (np.diff(yb)[:, None] * gw).ravel()
    zs, wz = (zb[:-1, None] + np.diff(zb)[:, None] * gp).ravel(), (np.diff(zb)[:, None] * gw).ravel()
    Yq, Zq = np.meshgrid(ys, zs, indexing="ij")
    Uq = np.asarray(velocity.evaluate(np.column_stack((np.full(Yq.size, x_mid), Yq.ravel(), Zq.ravel()))))
    flow_rate = float(np.einsum("i,j,ij->", wy, wz, Uq[:, 0].reshape(Yq.shape)))

    yg, zg = np.linspace(0, W, int(40 * aspect) + 1), np.linspace(0, H, 41)
    Yg, Zg = np.meshgrid(yg, zg, indexing="ij")
    Ug = np.asarray(velocity.evaluate(np.column_stack((np.full(Yg.size, x_mid), Yg.ravel(), Zg.ravel()))))
    u_surface = float(np.asarray(velocity.evaluate(np.array([[x_mid, W / 2, H]])))[0, 0])
    p_floor = float(np.asarray(pressure.evaluate(np.array([[x_mid, W / 2, 0.0]])))[0])
    elapsed = time.perf_counter() - start
    label = "Taylor-Hood FEM" if method == "fem" else "MAC finite differences"
    print(f"{label:23s} W/H = {aspect:4.1f}, {cells_across:3d} x {cells_deep:2d} cells across: "
          f"{mesh.cell_count:6d} cells, {elapsed:5.1f} s")
    return dict(W=W, aspect=aspect, h=H / cells_deep, Q=flow_rate, u_s=u_surface, p_floor=p_floor,
                Y=Yg, Z=Zg, U=Ug[:, 0].reshape(Yg.shape), V=Ug[:, 1:].reshape(Yg.shape + (2,)),
                velocity=velocity, x_mid=x_mid)


# ## Solve
#
# Taylor-Hood finite elements for five aspect ratios; for $W/H=4$ additionally two coarser
# meshes (4 and 8 cells across the depth) for the mesh study, and the MAC difference method on
# three grids (8, 16 and 32 cells across the depth).

fem = {aspect: solve("fem", aspect, cells_depth) for aspect in aspects}
fem_study = [solve("fem", 4.0, n) for n in (4, 8)] + [fem[4.0]]
mac_study = [solve("fd", 4.0, n) for n in (8, 16, 32)]


# ## Comparison with the reference
#
# **Velocity field, surface speed, flow rate and pressure** on the finest finite-element mesh for
# every aspect ratio, against the exact series solution.

print(" W/H   max|u - u_exact|/u_max   u_s FEM / exact [mm/s]    dev       Q/(W q_N) FEM / exact     dev"
      "      max|v,w|/u_max   p_bottom FEM / hydrostatic [Pa]")
for aspect, run in fem.items():
    exact_field = exact_velocity(run["Y"], run["Z"], run["W"])
    field_error = np.max(np.abs(run["U"] - exact_field)) / exact_field.max()
    us_exact = float(exact_velocity(run["W"] / 2, H, run["W"]))
    Q_exact = exact_flow_rate(run["W"])
    cross = np.max(np.abs(run["V"])) / exact_field.max()
    print(f" {aspect:3.1f}   {field_error:22.2e}   {1e3*run['u_s']:9.4f} / {1e3*us_exact:9.4f}   {run['u_s']/us_exact-1:+.1e}"
          f"   {run['Q']/(run['W']*q_nusselt):.6f} / {Q_exact/(run['W']*q_nusselt):.6f}   {run['Q']/Q_exact-1:+.1e}"
          f"   {cross:14.1e}   {run['p_floor']:8.4f} / {p_bottom:8.4f} ({run['p_floor']/p_bottom-1:+.1e})")

# **Wide-channel limit.** In the middle of a wide channel the film must become the Nusselt film:
# the centre-plane profile approaches $u_N(z)$ and the surface speed approaches $u_N(H)$, with a
# deviation that decays like $e^{-\pi W/(4H)}$.

print("\n W/H   centre-plane max|u - u_Nusselt|/u_N(H)   surface speed / Nusselt   flow rate / (W q_N)   wide-channel law 1 - 1.2605 H/W")
for aspect, run in fem.items():
    j = run["Y"].shape[0] // 2
    centre = run["U"][j]
    nusselt = G / mu * (H * run["Z"][j] - run["Z"][j] ** 2 / 2)
    wide = f"{1 - 1.2605/aspect:31.5f}" if aspect >= 2 else f"{'(not valid)':>31s}"
    print(f" {aspect:3.1f}   {np.max(np.abs(centre - nusselt))/u_nusselt:38.4f}   {run['u_s']/u_nusselt:23.5f}"
          f"   {run['Q']/(run['W']*q_nusselt):19.5f}   {wide}")

# **Mesh study and method comparison** for $W/H=4$. For three grids with spacings $h$, $h/2$,
# $h/4$ the observed order is $p=\log_2[(f_1-f_2)/(f_2-f_3)]$ and the Richardson estimate is
# $f_3+(f_3-f_2)/(2^p-1)$.

W4 = 4.0 * H
Q4, us4 = exact_flow_rate(W4), float(exact_velocity(W4 / 2, H, W4))
study = {}
for name, runs in (("Taylor-Hood FEM", fem_study), ("MAC finite differences", mac_study)):
    print(f"\n{name}, W/H = 4, cells across the depth: " + ", ".join(f"{round(H/r['h'])}" for r in runs))
    for label, key, ref in (("flow rate Q [mm^3/s]", "Q", Q4), ("surface speed u_s [mm/s]", "u_s", us4)):
        f = np.array([r[key] for r in runs])
        scale = 1e9 if key == "Q" else 1e3
        p = np.log2(abs(f[0] - f[1]) / abs(f[1] - f[2]))
        rich = f[2] + (f[2] - f[1]) / (2**p - 1)
        study[(name, key)] = (f, p, rich)
        print(f"  {label:25s} " + "  ".join(f"{scale*v:.7f}" for v in f)
              + f"   exact {scale*ref:.7f}")
        print(f"  {'':25s} deviation from exact " + "  ".join(f"{v/ref-1:+.2e}" for v in f)
              + f";  change between the last two grids {abs(f[2]/f[1]-1):.1e};  observed order {p:.2f};"
              f"  Richardson {scale*rich:.7f} ({rich/ref-1:+.1e})")


# ## Figure
#
# Top: the computed streamwise velocity over the cross-section of the channel with $W/H=4$ (the
# whole channel, colour map and isolines) and the deviation from the exact solution. Bottom: the
# centre-plane profiles for all widths against the series and the Nusselt film, the surface
# velocity across the channel, the flow rate as a function of the aspect ratio, and the error of
# both methods under mesh refinement.

run = fem[4.0]
z_line = np.linspace(0, H, 201)
aspects = np.geomspace(0.25, 20, 200)
labelled = {f"W/H = {aspect:g}": r for aspect, r in fem.items()}
studies = {"Taylor-Hood FEM": fem_study, "MAC FD": mac_study}
fig = plots.figure(layout=[["field"] * 3 + ["deviation"] * 3, ["centre", "centre", "surface", "surface", "rate", "study"]],
                   size=(16, 8.4))
plots.heatmap(1e3 * run["Y"], 1e3 * run["Z"], run["U"] / u_nusselt, limits=(0, 1), contours=9, label="u / u_N(H)",
              xlabel="y across the channel [mm]", ylabel="z above the bottom [mm]",
              title="Taylor-Hood FEM, W/H = 4: streamwise velocity u / u_N(H)", ax=fig["field"])
plots.heatmap(1e3 * run["Y"], 1e3 * run["Z"], (run["U"] - exact_velocity(run["Y"], run["Z"], run["W"])) / u_nusselt,
              colormap="diverging", label="(u - u_exact) / u_N(H)", xlabel="y across the channel [mm]",
              ylabel="z above the bottom [mm]", title="Deviation from the exact series", ax=fig["deviation"])
plots.line(points={label: (r["U"][r["Y"].shape[0] // 2][::4] / u_nusselt, 1e3 * r["Z"][r["Y"].shape[0] // 2][::4])
                   for label, r in labelled.items()},
           reference={**{label: (exact_velocity(r["W"] / 2, z_line, r["W"]) / u_nusselt, 1e3 * z_line) for label, r in labelled.items()},
                      "Nusselt film": ((H * z_line - z_line**2 / 2) * G / mu / u_nusselt, 1e3 * z_line)},
           xlabel="u / u_N(H) on the centre plane y = W/2", ylabel="z [mm]",
           title="Centre-plane profiles: FEM (markers), series (dashed)", legend="lower right", ax=fig["centre"])
plots.line(points={label: (r["Y"][::max(1, r["Y"].shape[0] // 20), -1] / r["W"], r["U"][::max(1, r["Y"].shape[0] // 20), -1] / u_nusselt)
                   for label, r in labelled.items()},
           reference={label: (np.linspace(0, 1, 201), exact_velocity(np.linspace(0, r["W"], 201), H, r["W"]) / u_nusselt)
                      for label, r in labelled.items()},
           xlabel="y / W", ylabel="surface velocity u(y, H) / u_N(H)", title="Free-surface velocity across the channel",
           legend="center", ax=fig["surface"])
plots.line(aspects, {"exact series": [exact_flow_rate(a * H) / (a * H * q_nusselt) for a in aspects]},
           reference={"1 - 1.2605 H/W": (aspects[aspects > 1.5], 1 - 1.2605 / aspects[aspects > 1.5])},
           points={"Taylor-Hood FEM": (list(fem), [r["Q"] / (r["W"] * q_nusselt) for r in fem.values()]),
                   "MAC FD": ([4.0], [mac_study[-1]["Q"] / (W4 * q_nusselt)])},
           scale="logx", ylimits=(0, 1.02), xlabel="aspect ratio W/H", ylabel="Q / (W q_N)", title="Flow rate vs width",
           legend="upper left", ax=fig["rate"])
plots.line(y={**{f"{name}: flow rate": ([r["h"] / H for r in runs], [abs(r["Q"] / Q4 - 1) for r in runs]) for name, runs in studies.items()},
              **{f"{name}: surface speed": ([r["h"] / H for r in runs], [abs(r["u_s"] / us4 - 1) for r in runs]) for name, runs in studies.items()}},
           colors={f"{name}: {kind}": n for n, name in enumerate(studies) for kind in ("flow rate", "surface speed")},
           styles={f"{name}: surface speed": "dashed" for name in studies}, markers=True, scale="loglog",
           xlabel="cell size h / H", ylabel="relative error vs series", title="Mesh study, W/H = 4", ax=fig["study"])
plots.show()


# ## Going further
#
# Tilt the channel further or make the liquid less viscous: as long as the Reynolds number stays
# small the fully developed film keeps exactly this shape, and only the scale $u_N$ changes. For
# thin films in narrow channels surface tension bends the free surface into a meniscus at the side
# walls; replacing the flat traction-free top by a curved surface shows how the meniscus changes
# the flow rate. A longer slab with an inflow profile shows the development length of the film
# behind an inlet.
