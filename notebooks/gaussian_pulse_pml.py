"""OneSimulation example: absorption of a Gaussian pressure pulse by perfectly matched layers.
Requires the installed, activated OneSimulation runtime.

A Gaussian pressure pulse released at rest in a 2D square of air spreads as a cylindrical
wave and leaves through perfectly matched layers. The computed 2D field is compared with the
exact free-space solution (a Hankel-transform integral) on three grids, and the energy
reflected back by the layers is measured against the theoretical PML reflection coefficient
for several layer thicknesses and strengths.
"""


# # Gaussian pulse absorption by perfectly matched layers
#
# A standard test for absorbing layers in the time domain (F. Q. Hu, "On absorbing boundary
# conditions for linearized Euler equations by a perfectly matched layer", *J. Comput.
# Phys.* 129, 201-219, 1996, here without mean flow): a square of still air is surrounded by
# perfectly matched layers (PMLs), and at $t=0$ a Gaussian pressure pulse is released at
# rest in its centre,
#
# $$p(\mathbf{x},0)=p_0\,\exp\!\big(-\ln 2\;|\mathbf{x}|^2/b^2\big),\qquad \mathbf{u}(\mathbf{x},0)=0 ,$$
#
# where $b$ is the half-width at half maximum. The pulse obeys the linear acoustic equations
#
# $$\partial_t p+\rho c^2\,\nabla\cdot\mathbf{u}=0,\qquad \rho\,\partial_t\mathbf{u}+\nabla p=0 .$$
#
# **Numerical method.** OneSimulation's native staggered-grid scheme on a 2D Cartesian grid:
# a second-order finite-difference / finite-volume discretisation with the pressure at cell
# centres and the normal velocities on the cell faces (no FEM, no matrix); time integration
# with the explicit classical Runge-Kutta method of order 4 at a fixed Courant number of 0.25,
# so no linear solver is involved. The PML is a split-pressure
# (Berenger-type) layer with a cubic damping profile $\sigma(\xi)=\sigma_\max(\xi/d)^3$, closed by
# a rigid wall. Its strength is set through the continuum round-trip reflection at normal
# incidence, $R_0=\exp\!\big(-\tfrac{2}{c}\int_0^d\sigma\,d\xi\big)$; at incidence angle
# $\theta$ the continuum layer reflects $R(\theta)=R_0^{\cos\theta}$, independently of its
# thickness $d$. On a grid the layer adds a discrete reflection that falls as $d$ grows.
#
# The pulse is symmetric about both axes, so one quarter of the square is computed with rigid
# (symmetry) walls on $x=0$ and $y=0$; the field is genuinely two-dimensional, $p(x,y,t)$, and
# the figure shows the full square obtained by mirroring.
#
# **Reference.** Two kinds of reference are used:
#
# 1. *Exact solution of the 2D problem.* In free space the pressure and radial velocity are the
#    Hankel-transform integrals of the initial pulse (Hu 1996, for zero Mach number)
#
#    $$p(r,t)=\frac{p_0}{2\beta}\int_0^\infty e^{-k^2/4\beta}\cos(ckt)\,J_0(kr)\,k\,dk,\qquad
#      u_r(r,t)=\frac{p_0}{2\beta\rho c}\int_0^\infty e^{-k^2/4\beta}\sin(ckt)\,J_1(kr)\,k\,dk,$$
#
#    with $\beta=\ln 2/b^2$. They are evaluated numerically by Gauss-Legendre quadrature and
#    compared with the computation at probe points and along a radial line at several times, on
#    three grids (observed order of accuracy).
# 2. *PML performance.* The field reflected by the layers is isolated as the difference to a
#    computation on a much larger grid whose walls are too far away to be reached in the time
#    window. Its energy in the physical square is compared with the prediction of the method
#    of images: each side of the square returns the exact free-space field of a mirror source,
#    weighted by the continuum reflection coefficient $R(\theta)$.

from importlib.metadata import version
import time

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
import numpy as np
from matplotlib.patches import Rectangle as RectanglePatch
from scipy.special import j0, j1, roots_legendre

from onesim.cad.d2 import Rectangle
from onesim.meshing.meshes.api import cartesian
from onesim.numerics import Numerics
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.acoustics import AcousticPML, LinearAcoustics, StaggeredAcousticDifference
from onesim.results.storage.output.retention_policy import ResultRetentionPolicy
from onesim.simulation import ExecutionOptions, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation.problem_definition.boundaries import ReflectingWall

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Air at 20 °C. The pulse has a half-width of $b=30$ mm and the physical square is
# $0.6\times0.6$ m ($a=0.3$ m $=10\,b$ from the centre to each side). The spectrum of the
# pulse, $\propto e^{-k^2b^2/4\ln2}$, has fallen to 4 % at $k=3/b$ (wavelength $2.1\,b$).
#
# * Accuracy runs: grid spacings $b/5$, $b/10$ and $b/20$ (the finest resolves that shortest
#   wavelength with 42 cells), a 60 mm layer with $R_0=10^{-4}$, until the pulse has travelled
#   $c\,t=0.42$ m and sits inside the layer.
# * PML runs: grid spacing $b/10$, layers of 10, 20, 40 and 80 cells at $R_0=10^{-4}$ and of
#   20 cells at $R_0=10^{-2},10^{-3},10^{-6}$, until $c\,t=1.02$ m $=3.4\,a$, long after the
#   pulse has left the square and late enough for the wave returning from the back of the
#   thickest layer to cross the square.
#
# Every run uses the Courant number $c\Delta t/\Delta x=0.25$ and stores a frame every 6 mm of
# travel.

c = 343.0                  # m/s, speed of sound
rho = 1.204                # kg/m^3, air density
p0 = 1.0                   # Pa, peak of the initial pulse (linear acoustics: any amplitude)
b = 0.030                  # m, half-width at half maximum of the pulse
beta = np.log(2.0) / b**2  # 1/m^2
a = 0.30                   # m, half side of the physical square
courant = 0.25
frame_travel = 0.006       # m, c * (time between stored frames)
layer_depth, layer_R0 = 0.060, 1e-4    # accuracy runs
dx = b / 10                # m, grid spacing of the PML runs
n_phys = int(round(a / dx))
print(f"PML runs: grid spacing {dx * 1e3:.1f} mm, {n_phys} x {n_phys} physical cells per quarter, "
      f"time step {courant * dx / c * 1e6:.3f} us")


# ## Geometry and mesh
#
# The computed quarter is $[0, a+d]^2$: the physical region $[0,a]^2$ plus the layer of
# thickness $d=N\Delta x$ on the two outer sides. The layer cells are part of the Cartesian
# grid; the grid nodes coincide with $x=0$ (the pulse centre) and with the start of the layer
# at $x=a$. A larger quarter without layers, $[0, 2.5a]^2$, is the free-space comparison run of
# the reflection measurement: a wave reflected at its walls cannot re-enter the physical region
# before $c\,t = 2\cdot 2.5a - a - 3b = 1.11$ m, after the end of the run.

def build_case(cells, spacing, layer):
    """Quarter domain with `cells` cells per side; `layer` = (N, R0) or None (no layer)."""
    side = cells * spacing
    geometry = Rectangle(length=side, width=side)
    mesh = cartesian(shape=(cells, cells), extents=(side, side), geometry=geometry,
                     periodic=False, max_candidate_cells=10**6)
    pml = None if layer is None else AcousticPML(((0, layer[0]), (0, layer[0])),
                                                 target_reflection=layer[1])
    return geometry, mesh, pml


# ## Model
#
# Boundary conditions on every side of the computed quarter:
#
# | side | condition |
# |---|---|
# | $x=0$ and $y=0$ | rigid wall $u_n=0$ = plane of symmetry of the centred pulse |
# | $x=a+d$ and $y=a+d$ | rigid wall closing the PML (the free-space run: far rigid walls) |
#
# The pulse starts at rest: $p=p_0\exp(-\ln2\,r^2/b^2)$, $\mathbf{u}=0$.

def run_case(spacing, cells, layer, travel):
    geometry, mesh, pml = build_case(cells, spacing, layer)
    air = LinearAcoustics(domain=geometry.domain(), density=rho, sound_speed=c)
    walls = geometry.boundaries

    def initial_pressure(x, y):
        return p0 * np.exp(-beta * (np.asarray(x) ** 2 + np.asarray(y) ** 2))

    problem = Problem(
        name="Gaussian pulse in air", geometry=geometry, physics=(air,),
        boundary_conditions=(
            # symmetry planes of the centred pulse
            ReflectingWall(boundary=walls.x_min | walls.y_min, physics=air),
            # rigid backing of the absorbing layers (or the far walls of the free-space run)
            ReflectingWall(boundary=walls.x_max | walls.y_max, physics=air),
        ),
        initial_conditions=(InitialValue(air.pressure, initial_pressure),
                            InitialValue(air.velocity, (0.0, 0.0))),
    )
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=StaggeredAcousticDifference(pml=pml), time_integrator=ClassicalRungeKutta4()))
    step = courant * spacing / c
    keep = int(round(frame_travel / (c * step)))
    n_steps = int(round(travel / frame_travel)) * keep
    result = simulation.run(
        start_time=0.0, end_time=n_steps * step, time_step=step,
        execution_options=ExecutionOptions(
            checkpoint_retain_count=1,
            result_retention=ResultRetentionPolicy(mode="stride", stride=keep)))
    pressure, velocity = result.field(air.pressure), result.field(air.velocity)
    centres = np.asarray(pressure.space.dof_coordinates)
    order = np.lexsort((centres[:, 1], centres[:, 0]))       # x-major, as (ix, iy)
    shape = (cells, cells)
    p = np.array([np.asarray(v)[order].reshape(shape) for v in pressure.history["values"]])
    u = np.array([np.asarray(v)[order].reshape(shape + (2,)) for v in velocity.history["values"]])
    return dict(times=np.asarray(pressure.history["times"], float), p=p, u=u, spacing=spacing,
                x=(np.arange(cells) + 0.5) * spacing, cells=cells, layer=layer)


# ## Solve
#
# First the three accuracy runs, then the PML series with its free-space comparison run.

accuracy = {}
for refine in (5, 10, 20):
    h = b / refine
    n_layer = int(round(layer_depth / h))
    start = time.time()
    accuracy[refine] = run_case(h, int(round(a / h)) + n_layer, (n_layer, layer_R0), 0.42)
    print(f"accuracy run dx = b/{refine:2d}: {accuracy[refine]['cells']:3d} x {accuracy[refine]['cells']:3d} cells "
          f"({n_layer} layer cells), {len(accuracy[refine]['times'])} frames ({time.time() - start:.0f} s)")
main = accuracy[20]

runs = {}
cases = [("free space", int(2.5 * n_phys), None)]
cases += [(f"N={n}, R0=1e-4", n_phys + n, (n, 1e-4)) for n in (10, 20, 40, 80)]
cases += [(f"N=20, R0={r:.0e}".replace("e-0", "e-"), n_phys + 20, (20, r)) for r in (1e-2, 1e-3, 1e-6)]
for label, cells, layer in cases:
    start = time.time()
    runs[label] = run_case(dx, cells, layer, 1.02)
    print(f"PML run {label:16s}: {cells:3d} x {cells:3d} cells, {len(runs[label]['times'])} frames "
          f"({time.time() - start:.0f} s)")
free = runs["free space"]
times = free["times"]


# **Exact free-space solution.** The Hankel integrals are evaluated with 1600 Gauss-Legendre
# nodes on $0\le k\le k_\max$ with $k_\max^2/4\beta=40$ (the integrand is below $e^{-40}$
# beyond).

k_max = np.sqrt(160.0 * beta)
nodes, weights = roots_legendre(1600)
k_nodes = 0.5 * k_max * (nodes + 1.0)
amplitude = p0 / (2.0 * beta) * np.exp(-k_nodes**2 / (4.0 * beta)) * k_nodes * 0.5 * k_max * weights


def exact(r, t):
    """Exact free-space pressure and radial velocity at radii r (any shape) and time t."""
    r = np.asarray(r, float)
    kr = np.outer(r.ravel(), k_nodes)
    pressure = j0(kr) @ (amplitude * np.cos(c * k_nodes * t))
    velocity = j1(kr) @ (amplitude * np.sin(c * k_nodes * t)) / (rho * c)
    return pressure.reshape(r.shape), velocity.reshape(r.shape)


def exact_table(t, rmax):
    table_r = np.linspace(0.0, rmax, 4001)
    return (table_r,) + exact(table_r, t)


# ## Comparison with the reference
#
# **Exact solution, three grids.** For each grid: the largest deviation from the exact pressure
# over the physical quarter at $c\,t=0.18$ m and $0.27$ m (before and while the pulse enters the
# layer), relative to the exact peak at that time; and the pulse peak at three probe points
# (nearest cell centre to the nominal position; the exact value is taken at that same centre).

probe_targets = {"axis, r = 0.10 m": (0.10, 0.0), "diagonal, r = 0.17 m": (0.12, 0.12),
                 "near the layer, r = 0.28 m": (0.28, 0.0)}
grid_errors = {}
print("Largest deviation from the exact solution over the physical quarter (relative to the exact peak)")
for refine, run in accuracy.items():
    phys = int(round(a / run["spacing"]))
    X, Y = np.meshgrid(run["x"][:phys], run["x"][:phys], indexing="ij")
    row = []
    for ct in (0.18, 0.27):
        k = int(np.argmin(np.abs(c * run["times"] - ct)))
        table_r, table_p, _ = exact_table(run["times"][k], 0.5)
        p_ref = np.interp(np.hypot(X, Y), table_r, table_p)
        row.append(np.max(np.abs(run["p"][k, :phys, :phys] - p_ref)) / np.max(np.abs(p_ref)))
    grid_errors[refine] = row
    print(f"  dx = b/{refine:2d}: c t = 0.18 m: {row[0]:.3e},  c t = 0.27 m: {row[1]:.3e}")
for col, ct in enumerate((0.18, 0.27)):
    e = [grid_errors[r][col] for r in (5, 10, 20)]
    print(f"  c t = {ct:.2f} m: observed order {np.log2(e[0] / e[1]):.2f} (b/5 -> b/10), "
          f"{np.log2(e[1] / e[2]):.2f} (b/10 -> b/20)")

print("Pulse peak at the probes: OneSimulation vs exact")
probe_series = {}
for name, (px, py) in probe_targets.items():
    line = f"  {name:28s}"
    for refine, run in accuracy.items():
        i, j = (int(np.floor(v / run["spacing"])) for v in (px, py))
        r_probe = np.hypot(run["x"][i], run["x"][j])
        reference = np.array([exact(r_probe, t)[0] for t in run["times"]])
        computed = run["p"][:, i, j]
        k = int(np.argmax(reference))
        line += f"  b/{refine}: {computed[k]:.5f} vs {reference[k]:.5f} ({100 * (computed[k] / reference[k] - 1):+.2f} %)"
        if refine == 20:
            probe_series[name] = (run["times"], computed, reference)
    print(line)

# **Radial line at several times** (finest grid). Along the row of cells next to the $x$ axis,
# from the centre to the layer.

line_data = []
print("Radial line y = dx/2 on the finest grid: peak and largest deviation (relative to the exact peak)")
phys20 = int(round(a / main["spacing"]))
r_line = np.hypot(main["x"][:phys20], 0.5 * main["spacing"])
for ct in (0.06, 0.12, 0.18, 0.24):
    k = int(np.argmin(np.abs(c * main["times"] - ct)))
    reference = exact(r_line, main["times"][k])[0]
    computed = main["p"][k, :phys20, 0]
    line_data.append((main["times"][k], computed, reference))
    print(f"  c t = {ct:.2f} m: peak {computed.max():.5f} Pa at r = {r_line[np.argmax(computed)]:.4f} m vs "
          f"{reference.max():.5f} Pa at r = {r_line[np.argmax(reference)]:.4f} m "
          f"({100 * (computed.max() / reference.max() - 1):+.2f} %); largest deviation "
          f"{np.max(np.abs(computed - reference)) / reference.max():.2e}")

# **Reflected energy.** For each layer the reflected field is the difference to the free-space
# run on the physical square; its acoustic energy
# $E_r=\int\big(\delta p^2/2\rho c^2+\rho|\delta\mathbf{u}|^2/2\big)\,dA$ is normalised by the
# initial energy $E_0=\pi b^2 p_0^2/(4\ln2\,\rho c^2)$ of the whole pulse. The method-of-images
# prediction for a continuum layer is
#
# $$E_r^\mathrm{images}(t)=\sum_{\text{4 sides}}\int_{\text{square}}R_0^{\,2\cos\theta}\,
#   e_\mathrm{free}(|\mathbf{x}-\mathbf{x}_s|,t)\,dA ,$$
#
# where $\mathbf{x}_s$ is the mirror image of the centre in the rigid wall that closes the layer
# on that side (at distance $a+d$ from the centre: inside a matched layer the wave travels at the
# speed of sound and is only attenuated), $\theta$ the angle of incidence of the ray from
# $\mathbf{x}_s$, and $e_\mathrm{free}$ the exact free-space energy density. It is evaluated on
# the same cell centres. A thicker layer moves the mirror sources outwards, so the reflected
# wave arrives later and more spread out: at fixed $R_0$ the predicted peak falls with $d$
# although the continuum reflection coefficient itself does not depend on $d$. The image
# model treats each piece of the cylindrical wave as a plane wave and ignores reflection
# from the discrete layer, so agreement within a factor of about two is what can be expected.

E0 = np.pi * b**2 * p0**2 / (4.0 * np.log(2.0) * rho * c**2)
cell_area = dx * dx


def energy(dp, du):
    """Energy over the full square from quarter-domain cell values."""
    return 4.0 * cell_area * np.sum(dp**2 / (2 * rho * c**2) + 0.5 * rho * np.sum(du**2, axis=-1),
                                    axis=(-2, -1))


print(f"Initial energy: grid sum {energy(free['p'][0], free['u'][0]):.6e} J/m vs exact {E0:.6e} J/m")
reflected = {}
for label, run in runs.items():
    if run["layer"] is not None:
        dp = run["p"][:, :n_phys, :n_phys] - free["p"][:, :n_phys, :n_phys]
        du = run["u"][:, :n_phys, :n_phys] - free["u"][:, :n_phys, :n_phys]
        reflected[label] = energy(dp, du) / E0

X, Y = np.meshgrid(free["x"][:n_phys], free["x"][:n_phys], indexing="ij")
image_times = times[::2]
image_prediction = {label: np.zeros(len(image_times)) for label in reflected}
for n, t in enumerate(image_times):
    table_r, table_p, table_u = exact_table(t, 1.7)
    for label in reflected:
        cells_in_layer, r0 = runs[label]["layer"]
        wall = a + cells_in_layer * dx
        for sx, sy in ((2 * wall, 0.0), (-2 * wall, 0.0), (0.0, 2 * wall), (0.0, -2 * wall)):
            dist = np.hypot(X - sx, Y - sy)
            cos_theta = np.abs((sx - X) if sx else (sy - Y)) / dist
            e_free = (np.interp(dist, table_r, table_p) ** 2 / (2 * rho * c**2)
                      + 0.5 * rho * np.interp(dist, table_r, table_u) ** 2)
            image_prediction[label][n] += 4.0 * cell_area * np.sum(r0 ** (2 * cos_theta) * e_free) / E0

print("Peak reflected energy in the physical square, E_r / E0: computed vs continuum-layer images")
for label, curve in reflected.items():
    predicted = image_prediction[label]
    print(f"  {label:16s}: computed {curve.max():.3e} at c t = {c * times[np.argmax(curve)]:.2f} m,  "
          f"images {predicted.max():.3e} at c t = {c * image_times[np.argmax(predicted)]:.2f} m,  "
          f"ratio {curve.max() / predicted.max():.2f}")

# **Residual energy after the pulse has left.** At the end of the runs the outgoing pulse has
# left the square; in 2D a slowly decaying wake remains behind it. The energy left in the
# physical square is compared with that of the exact free-space wake.

table_r, table_p, table_u = exact_table(times[-1], 1.7)
R_cells = np.hypot(X, Y)
u_r = np.interp(R_cells, table_r, table_u)
E_wake = energy(np.interp(R_cells, table_r, table_p),
                np.stack((u_r * X / R_cells, u_r * Y / R_cells), axis=-1)) / E0
print(f"Energy left in the physical square at c t = {c * times[-1]:.2f} m; exact wake {E_wake:.4e} E0")
for label, run in runs.items():
    E_left = energy(run["p"][-1, :n_phys, :n_phys], run["u"][-1, :n_phys, :n_phys]) / E0
    print(f"  {label:16s}: {E_left:.4e} E0  ({100 * (E_left / E_wake - 1):+.2f} % vs the exact wake)")


# ## Figure
#
# Top: the pressure field on the finest grid (full square mirrored from the computed quarter;
# the shaded frame is the PML) at four times. Bottom, from left: probe signals and radial
# profiles against the exact solution; grid convergence; reflected energy over time for the
# thickness series; peak reflected energy against the continuum theory.

fig = plt.figure(figsize=(18, 10.5), constrained_layout=True)
outer = fig.add_gridspec(2, 1, height_ratios=(1.05, 0.9))
top = outer[0].subgridspec(1, 4)
bottom = outer[1].subgridspec(1, 5)
side = main["cells"] * main["spacing"]
edges = np.arange(main["cells"] + 1) * main["spacing"]
full_edges = np.concatenate((-edges[::-1], edges[1:]))
for col, ct in enumerate((0.06, 0.18, 0.30, 0.42)):
    ax = fig.add_subplot(top[col])
    k = int(np.argmin(np.abs(c * main["times"] - ct)))
    q = main["p"][k]
    full = np.block([[q[::-1, ::-1], q[::-1, :]], [q[:, ::-1], q]])
    limit = np.max(np.abs(full))
    image = ax.pcolormesh(full_edges, full_edges, full.T, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit,
                          shading="flat", rasterized=True)
    for corner, width, height in (((-side, -side), 2 * side, side - a), ((-side, a), 2 * side, side - a),
                                  ((-side, -a), side - a, 2 * a), ((a, -a), side - a, 2 * a)):
        ax.add_patch(RectanglePatch(corner, width, height, color="0.55", alpha=0.25, lw=0))
    ax.plot([-a, a, a, -a, -a], [-a, -a, a, a, -a], color="0.3", lw=0.8)
    ax.set_aspect("equal")
    ax.set(xlabel="x / m", ylabel="y / m" if col == 0 else None,
           title=f"t = {main['times'][k] * 1e3:.3f} ms  (c t = {c * main['times'][k]:.2f} m)")
    fig.colorbar(image, ax=ax, shrink=0.8, label="p / Pa")

colours = (GREEN, COPPER, "#214D43", "#CD9561")
ax = fig.add_subplot(bottom[0])
for (name, (t_run, computed, reference)), colour in zip(probe_series.items(), colours):
    ax.plot(t_run * 1e3, reference, color=colour, lw=1.4, label=f"exact, {name}")
    ax.plot(t_run * 1e3, computed, "o", ms=2.6, color=colour, mfc="none")
ax.set(xlabel="t / ms", ylabel="p / Pa", title="Probe signals (circles: OneSimulation)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(bottom[1])
for (t, computed, reference), colour in zip(line_data, colours):
    ax.plot(r_line, reference, color=colour, lw=1.4, label=f"exact, c t = {c * t:.2f} m")
    ax.plot(r_line[::4], computed[::4], "o", ms=2.4, color=colour, mfc="none")
ax.set(xlabel="r / m", ylabel="p / Pa", title="Radial profiles (circles: OneSimulation)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = fig.add_subplot(bottom[2])
spacings = np.array([b / r for r in (5, 10, 20)])
for col, (ct, colour) in enumerate(zip((0.18, 0.27), colours)):
    ax.loglog(spacings * 1e3, [grid_errors[r][col] for r in (5, 10, 20)], "o-", color=colour,
              label=f"c t = {ct:.2f} m")
ax.loglog(spacings * 1e3, grid_errors[5][0] * (spacings / spacings[0]) ** 2, ":", color=INK, lw=1.0, label="slope 2")
ax.set_xticks(spacings * 1e3, [f"{v:g}" for v in spacings * 1e3])
ax.minorticks_off()
ax.set(xlabel="grid spacing / mm", ylabel="max |p - p_exact| / peak", title="Grid convergence")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(bottom[3])
for label, colour in zip([f"N={n}, R0=1e-4" for n in (10, 20, 40, 80)], ("#A8BC91", "#5F9175", GREEN, "#214D43")):
    ax.semilogy(c * times, np.maximum(reflected[label], 1e-16), color=colour, lw=1.3, label=label)
    ax.semilogy(c * image_times, image_prediction[label], "--", color=colour, lw=1.0)
ax.plot([], [], "--", color=INK, lw=1.0, label="continuum layer (images)")
ax.set(xlabel="c t / m", ylabel="E_r / E_0", title="Reflected energy in the square, R0 = 1e-4",
       ylim=(1e-13, 1e-5))
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(bottom[4])
r0_values = (1e-2, 1e-3, 1e-4, 1e-6)
strength_labels = [f"N=20, R0={r:.0e}".replace("e-0", "e-") for r in r0_values]
ax.loglog(r0_values, [reflected[label].max() for label in strength_labels], "o-", color=GREEN,
          label="computed, N = 20")
ax.loglog(r0_values, [image_prediction[label].max() for label in strength_labels], "s--", mfc="none",
          color=INK, label="continuum layer (images)")
for n, marker in zip((10, 40, 80), ("v", "^", "D")):
    ax.loglog([1e-4], [reflected[f"N={n}, R0=1e-4"].max()], marker, color=COPPER, ms=6,
              label=f"computed, N = {n}")
ax.set(xlabel="design reflection R_0", ylabel="peak E_r / E_0", title="PML strength and thickness")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
plt.show()


# ## Going further
#
# Add a uniform mean flow to recover the original convected-pulse benchmark of Hu (1996), or
# lower the polynomial order of the damping profile and watch the discrete reflection of thin
# layers grow. Replacing the layers by plain rigid walls shows the full reflection that the
# PML removes.
