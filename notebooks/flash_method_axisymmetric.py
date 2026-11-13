"""OneSimulation example: laser-flash measurement on a coin-shaped sample with heat losses.
Requires the installed, activated OneSimulation runtime.

A uniform laser flash heats the front face of a 12.7 mm x 2 mm ceramic disc; the rear-face
temperature rise is computed on full 3D disc meshes with convective and radiative losses on
every face, graded against the exact series solution of the finite cylinder, and analysed
with the Parker, Cowan and Clark-Taylor methods to recover the thermal diffusivity.
"""


# # Laser-flash method on a coin-shaped sample
#
# In the flash method (W. J. Parker, R. J. Jenkins, C. P. Butler, G. L. Abbott, *J. Appl.
# Phys.* 32, 1679, 1961) the front face of a thin disc receives a short, spatially uniform
# laser flash and the temperature rise of the rear face is recorded. Without heat losses the
# time $t_{1/2}$ at which the rear face reaches half of its maximum rise gives the thermal
# diffusivity, $\alpha = 0.1388\,L^2/t_{1/2}$. A real sample of finite radius loses heat by
# convection and radiation from both faces and from its rim: the rise peaks and decays,
# $t_{1/2}$ shortens and the Parker formula overestimates $\alpha$. Cowan (*J. Appl. Phys.* 34,
# 926, 1963) and Clark and Taylor (*J. Appl. Phys.* 46, 714, 1975) correct for the losses
# with ratios read off the measured curve.
#
# The disc is computed as a full 3D solid. The temperature depends on the radius and the depth,
# $T(r,z,t)$, because the rim cools the outer part of the disc. Transient heat conduction
#
# $$\rho c_p\,\partial_t T = k\,\nabla^2 T$$
#
# is solved with a condition on every face:
#
# | face | condition |
# |---|---|
# | front face $z=0$ (heated by the flash) | losses $-k\,\partial_n T = h\,(T-T_\infty)$ |
# | rear face $z=L$ (measured) | losses $-k\,\partial_n T = h\,(T-T_\infty)$ |
# | rim $r=R$ | losses $-k\,\partial_n T = h\,(T-T_\infty)$ |
# | all faces, loss-free sample | insulated, $\partial_n T = 0$ |
#
# with $h = h_c + 4\varepsilon\sigma T_\infty^3$: free convection plus radiation to a large
# enclosure at the ambient temperature, linearised about $T_\infty$ (the rise is about 1 K, so
# the linearisation changes the radiative loss by less than $0.3\,\%$; this is the loss model
# of Cowan 1963).
#
# **The flash.** As in Parker's analysis the flash is instantaneous and uniform over the front
# face: an energy $Q$ per unit area is absorbed at $t=-t_0$. The computation starts at $t=0$
# from the temperature field it has produced by then in the thin heated layer,
# $T-T_\infty=\frac{Q}{\rho c_p}\frac{2}{\sqrt{\pi}\,w}\,e^{-z^2/w^2}$ with $w=2\sqrt{\alpha t_0}$
# (the exact response of a thick body, $w\ll L$); the losses during $t_0$ remove less than
# $10^{-4}$ of $Q$. Times measured from the flash are $t+t_0$.
#
# **Numerical method.** Continuous Galerkin finite elements with quadratic Lagrange (P2)
# tetrahedra on graded meshes of the full disc; time integration with **adaptive
# variable-step BDF3** (third-order backward differentiation with local error control, steps
# from a few µs at the start to about 50 ms in the decay); the linear systems are solved with the
# conjugate-gradient method (symmetric positive definite, relative residual $10^{-10}$).
# The furnace case is computed on three meshes (mesh study) and with four step tolerances
# (time-step study); the room-temperature and loss-free cases on the fine mesh.
#
# **Reference.** The exact solution of this finite-cylinder problem is a double series of
# axial and radial eigenfunctions (H. S. Carslaw and J. C. Jaeger, *Conduction of Heat in
# Solids*, 2nd ed., 1959, Secs. 3.11 and 7.6; for the flash geometry with rim losses
# J. A. Cape and G. W. Lehman, *J. Appl. Phys.* 34, 1909, 1963). With $H=h/k$ the axial
# eigenvalues solve $\tan\mu L = 2\mu H/(\mu^2-H^2)$ with orthonormal eigenfunctions
# $\varphi_m\propto\mu_m\cos\mu_m z+H\sin\mu_m z$; the radial eigenvalues solve
# $\beta J_1(\beta)=\mathrm{Bi}\,J_0(\beta)$ with $\mathrm{Bi}=hR/k$. For the initial layer
# $f(z)$ above,
#
# $$T-T_\infty=\sum_{m}\sum_{n} b_m\,\varphi_m(z)\,
#   \frac{2\,\mathrm{Bi}\,J_0(\beta_n r/R)}{(\beta_n^2+\mathrm{Bi}^2)\,J_0(\beta_n)}\,
#   e^{-\alpha(\mu_m^2+\beta_n^2/R^2)\,t},\qquad b_m=\int_0^L f\,\varphi_m\,dz .$$
#
# The computed rear-face rise is compared with this series point by point; the diffusivity is
# then recovered with the Parker, Cowan and Clark-Taylor analyses from both the computed and
# the exact curve. A three-level mesh study (observed order, Richardson estimate) and a study
# of the step-size tolerance give the discretisation errors.

from dataclasses import dataclass
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
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq
from scipy.spatial import Delaunay
from scipy.special import j0, j1

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ConjugateGradient
from onesim.numerics.time import BDF, AdaptiveTimeStep
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.results.storage.output.retention_policy import ResultRetentionPolicy
from onesim.simulation import ExecutionOptions, Problem, Simulation
from onesim.simulation.initial_conditions.values import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# A Pyroceram 9606 glass-ceramic disc, the usual reference material of flash diffusivity
# measurements, 12.7 mm in diameter and 2 mm thick (the sample size of ASTM E1461). Properties
# at 300 K from Incropera et al., *Fundamentals of Heat and Mass Transfer*, Table A.3. The flash
# energy raises a loss-free sample by 1 K. Three loss settings: no losses (the Parker limit),
# room temperature and a furnace at 800 °C, each with free convection $h_c=10$ W/(m² K) and
# emissivity 0.9 on all faces. The same constant properties are used in all three cases, so
# that only the losses change.

R = 6.35e-3              # m, sample radius
L = 2.0e-3               # m, sample thickness
k = 3.98                 # W/(m K)
rho = 2600.0             # kg/m^3
cp = 808.0               # J/(kg K)
alpha = k / (rho * cp)   # m^2/s
dT_adiabatic = 1.0       # K, final rise of a loss-free sample
Q = rho * cp * L * dT_adiabatic          # J/m^2, absorbed flash energy
w = 0.15e-3              # m, width of the heated layer at t = 0
t0 = w**2 / (4 * alpha)  # s, time since the flash at t = 0
h_conv, emissivity, sigma = 10.0, 0.9, 5.670374e-8
t_half_parker = 0.1388 * L**2 / alpha
t_end = 3.2              # s, more than ten half-rise times
ambient = {"loss-free": 293.15, "room, 20 °C": 293.15, "furnace, 800 °C": 1073.15}
h_case = {"loss-free": 0.0, **{case: h_conv + 4 * emissivity * sigma * ambient[case] ** 3
                               for case in ("room, 20 °C", "furnace, 800 °C")}}
tolerance = 3e-4         # K, local error tolerance of the adaptive time steps (0.03 % of the rise)
print(f"alpha = {alpha:.4e} m^2/s, Parker half-rise time {t_half_parker * 1e3:.1f} ms, "
      f"Q = {Q:.0f} J/m^2, heated layer w = {w * 1e3:.2f} mm (t0 = {t0 * 1e3:.2f} ms)")
for case, h in h_case.items():
    print(f"  {case:16s}: h = {h:6.2f} W/(m^2 K), face Biot hL/k = {h * L / k:.4f}, rim Biot hR/k = {h * R / k:.4f}")


# ## Geometry and mesh
#
# OneSimulation's `Cylinder` primitive has no mesher that grades towards the heated face, so
# each disc is meshed directly: concentric rings of nodes (graded towards the rim, every
# second ring rotated by half a spacing) are triangulated in the plane, the triangulation is
# extruded through the thickness in layers graded towards the heated front face (several
# quadratic elements across the heated layer $w$), and every prism is split into three
# tetrahedra by a conforming diagonal rule (Dompierre et al., Proc. 8th Int. Meshing
# Roundtable, 1999). The tetrahedra are handed to `UnstructuredSimplexMesh` together with the
# named face groups front, rear and rim. (The builder can place several discs side by side in
# one mesh; each run below uses a single disc.)

PRISM_ROTATIONS = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                            [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def split_prism(vertices):
    v = np.asarray(vertices)[PRISM_ROTATIONS[int(np.argmin(vertices))]]
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [(v[0], v[1], v[2], v[5]), (v[0], v[1], v[5], v[4]), (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]), (v[0], v[4], v[2], v[5]), (v[0], v[4], v[5], v[3])]


def graded(n, ratio):
    """n intervals on [0, 1]; last interval / first interval = ratio."""
    growth = ratio ** (1.0 / (n - 1))
    sizes = growth ** np.arange(n)
    return np.concatenate(([0.0], np.cumsum(sizes) / sizes.sum()))


def disc_mesh(rings, layers, ratio, rim_nodes):
    """Vertices and tetrahedra of one disc centred on the z axis."""
    radii = R * graded(rings, 0.4)[1:]
    plane = [[0.0, 0.0]]
    for i, r in enumerate(radii):
        n = max(6, int(round(rim_nodes * r / R)))
        phase = (np.arange(n) + 0.5 * (i % 2)) * 2 * np.pi / n
        plane += list(np.column_stack((r * np.cos(phase), r * np.sin(phase))))
    plane = np.array(plane)
    triangles = Delaunay(plane).simplices
    z = L * graded(layers, ratio)
    n_plane = len(plane)
    coordinates = np.column_stack((np.tile(plane[:, 0], len(z)), np.tile(plane[:, 1], len(z)),
                                   np.repeat(z, n_plane)))
    cells = []
    for j in range(len(z) - 1):
        lower, upper = j * n_plane, (j + 1) * n_plane
        for a, b, c in triangles:
            cells += split_prism((a + lower, b + lower, c + lower, a + upper, b + upper, c + upper))
    return coordinates, np.array(cells), z


@dataclass(frozen=True, eq=False, kw_only=True)
class SampleDiscs(Geometry):
    """Discs side by side; each has the face groups <name>_front, <name>_rear, <name>_rim."""

    samples: tuple
    dimension = 3

    def boundary_groups(self):
        names = tuple(f"{s}_{face}" for s in self.samples for face in ("front", "rear", "rim"))
        return {**{name: (name,) for name in names}, "all": names}


levels = {"coarse": (3, 6, 7.0, 16), "medium": (4, 8, 10.0, 22), "fine": (5, 10, 14.0, 28)}
spacing = 2 * R + 6e-3


def build_discs(samples):
    """samples: list of (name, mesh level). Returns geometry, mesh and per-disc data."""
    coordinates, cells, discs = [], [], {}
    offset = 0
    for i, (name, level) in enumerate(samples):
        xyz, tets, z = disc_mesh(*levels[level])
        xyz = xyz + np.array([i * spacing, 0.0, 0.0])
        discs[name] = dict(level=level, x0=i * spacing, cells=tets + offset, z=z, n_cells=len(tets))
        coordinates.append(xyz)
        cells.append(tets + offset)
        offset += len(xyz)
    coordinates, cells = np.concatenate(coordinates), np.concatenate(cells)
    p = coordinates[cells]
    signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    cells[signed < 0] = cells[signed < 0][:, [0, 2, 1, 3]]
    names = [name for name, _ in samples]
    geometry = SampleDiscs(samples=tuple(names))
    groups = {g: [] for g in geometry.boundary_groups()["all"]}
    for face in boundary_faces(cells, coordinates):
        v = np.asarray(face.vertices)
        disc = names[int(round(coordinates[v[0], 0] / spacing))]
        zf = coordinates[v, 2]
        side = "front" if np.all(zf < 1e-12) else "rear" if np.all(zf > L - 1e-12) else "rim"
        groups[f"{disc}_{side}"].append(tuple(v))
    mesh =UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells, boundary_facets=groups)
    return dict(geometry=geometry, mesh=mesh, coordinates=coordinates, cells=cells, groups=groups, discs=discs)


main_samples = [("furnace_coarse", "coarse"), ("furnace_medium", "medium"), ("furnace_fine", "fine"),
                ("room_fine", "fine"), ("lossfree_fine", "fine")]
case_of = {"furnace_coarse": "furnace, 800 °C", "furnace_medium": "furnace, 800 °C",
           "furnace_fine": "furnace, 800 °C", "room_fine": "room, 20 °C", "lossfree_fine": "loss-free"}
disc_runs = {name: build_discs([(name, level)]) for name, level in main_samples}
for level in levels:
    xyz, tets, z = disc_mesh(*levels[level])
    print(f"{level:6s} disc: {len(tets):5d} tetrahedra, {len(xyz):4d} vertices, {len(z) - 1:2d} layers, "
          f"first layer {z[1] * 1e6:.0f} um, last layer {(z[-1] - z[-2]) * 1e6:.0f} um")


# ## Model
#
# One `HeatConduction` model per run; each disc gets the boundary conditions of its loss case on
# its three face groups and the heated-layer initial field.

material = IsotropicSolid(density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
                          youngs_modulus=9.6e10, poisson_ratio=0.24, thermal_expansion_coefficient=5.7e-6)


def heated_layer(z):
    return Q / (rho * cp) * 2.0 / (np.sqrt(np.pi) * w) * np.exp(-(np.asarray(z) / w) ** 2)


def run_discs(data, step_tolerance):
    geometry = data["geometry"]
    names = list(data["discs"])
    model = HeatConduction(domain=geometry.domain(), material=material, heating=0.0)
    b = geometry.boundaries
    conditions = []
    for name in names:
        case = case_of[name]
        faces = (getattr(b, f"{name}_front") | getattr(b, f"{name}_rear") | getattr(b, f"{name}_rim"))
        if h_case[case] > 0:
            # front, rear and rim: convection + linearised radiation to the ambient
            conditions.append(ConvectiveHeatFlux(boundary=faces, physics=model, heat_transfer_coefficient=h_case[case],
                                                 ambient_temperature=ambient[case]))
        else:
            # loss-free sample: all faces insulated
            conditions.append(HeatFlux(boundary=faces, physics=model, value=0.0))
    T_inf_disc = np.array([ambient[case_of[name]] for name in names])

    def initial_temperature(x, y, z):
        disc = np.clip(np.rint(np.asarray(x) / spacing).astype(int), 0, len(names) - 1)
        return T_inf_disc[disc] + heated_layer(z)

    problem = Problem(name="Flash method", geometry=geometry, physics=(model,),
                      boundary_conditions=tuple(conditions),
                      initial_conditions=(InitialValue(model.temperature, initial_temperature),))
    simulation = Simulation(problem=problem, mesh=data["mesh"], numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=ConjugateGradient(relative_tolerance=1e-10, maximum_iterations=20000),
        time_integrator=BDF(order=3)))
    # Every accepted step is streamed to this recorder instead of being kept in the result.
    frames = {}

    def record(frame):
        if frame.get("event") == "result_frame" and frame.get("state"):
            values = frame["state"][model.model_id][model.temperature.field_id]
            frames[float(frame["time"])] = np.array(values, dtype=float, copy=True)

    start = time.time()
    result = simulation.run(
        start_time=0.0, end_time=t_end,
        time_step=AdaptiveTimeStep(initial=2e-6, minimum=1e-9, maximum=0.1,
                                   relative_tolerance=0.0, absolute_tolerance=step_tolerance),
        execution_options=ExecutionOptions(checkpoint_retain_count=1, result_retention=ResultRetentionPolicy(
            mode="external_stream", stream_callback=record)))
    space = result.field(model.temperature).space
    times = np.array(sorted(frames))
    states = np.array([frames[t] for t in times])
    dof_xyz = np.asarray(space.dof_coordinates)
    rear_dofs = [int(np.argmin(np.linalg.norm(dof_xyz - np.array([data["discs"][name]["x0"], 0.0, L]), axis=1)))
                 for name in names]
    report = result.result_set.solver_report
    return dict(space=space, times=times, states=states,
                rear={name: states[:, dof] - T_inf_disc[i] for i, (name, dof) in enumerate(zip(names, rear_dofs))},
                T_inf={name: T_inf_disc[i] for i, name in enumerate(names)}, steps=report.get("accepted_steps"),
                rejected=report.get("rejected_steps"), seconds=time.time() - start, data=data)


def temperature(run, points, t):
    """Evaluate the stored quadratic field of a run at the accepted time nearest to t."""
    k = int(np.argmin(np.abs(run["times"] - t)))
    return np.asarray(run["space"].evaluate(run["states"][k], points)).ravel(), run["times"][k]


# ## Solve
#
# Five runs with the local error tolerance of 0.3 mK; for the time-step study the coarse
# furnace disc is repeated with tolerances of 3 mK, 1 mK and 0.1 mK.

runs = {}
for name, level in main_samples:
    runs[name] = run_discs(disc_runs[name], tolerance)
    print(f"{name:15s} ({len(disc_runs[name]['cells']):5d} tetrahedra): {runs[name]['steps']} accepted / "
          f"{runs[name]['rejected']} rejected steps ({runs[name]['seconds']:.0f} s)")
step_runs = {tolerance: runs["furnace_coarse"]}
for tol in (3e-3, 1e-3, 1e-4):
    step_runs[tol] = run_discs(disc_runs["furnace_coarse"], tol)
    print(f"coarse furnace disc alone, tolerance {tol * 1e3:.1f} mK: {step_runs[tol]['steps']} accepted / "
          f"{step_runs[tol]['rejected']} rejected steps ({step_runs[tol]['seconds']:.0f} s)")


# ## Comparison with the reference
#
# **Exact series** with 250 axial and 150 radial terms; the coefficients $b_m$ of the heated
# layer are integrated numerically on a fine grid.

def axial_modes(H, count=250):
    if H == 0:
        mu = np.arange(count) * np.pi / L
        phi = lambda z: np.where(mu == 0, 1.0 / np.sqrt(L), np.sqrt(2.0 / L) * np.cos(np.outer(np.atleast_1d(z), mu)))
        return mu, phi
    F = lambda m: (m**2 - H**2) * np.sin(m * L) - 2 * m * H * np.cos(m * L)
    mu = np.array([brentq(F, max(n * np.pi / L, 1e-9 / L), (n + 1) * np.pi / L, xtol=1e-12 / L)
                   for n in range(count)])
    norm = np.sqrt(0.5 * ((mu**2 + H**2) * L + 2 * H))
    phi = lambda z: (mu * np.cos(np.outer(np.atleast_1d(z), mu)) + H * np.sin(np.outer(np.atleast_1d(z), mu))) / norm
    return mu, phi


def radial_modes(Bi, count=150):
    if Bi == 0:
        return np.array([0.0]), lambda r: np.ones((np.size(r), 1))
    G = lambda x: x * j1(x) - Bi * j0(x)
    grid = np.linspace(1e-9, (count + 2) * np.pi, 400 * (count + 2))
    values = G(grid)
    roots = np.where(np.sign(values[:-1]) != np.sign(values[1:]))[0][:count]
    beta = np.array([brentq(G, grid[i], grid[i + 1], xtol=1e-14) for i in roots])
    coefficient = 2 * Bi / ((beta**2 + Bi**2) * j0(beta))
    return beta, lambda r: coefficient * j0(np.outer(np.atleast_1d(r), beta) / R)


z_quad = np.linspace(0.0, min(12 * w, L), 24001)
series_cache = {}


def exact_rise(case, times, r=0.0, z=L):
    """Exact T - T_inf at (r, z) for the times given (arrays of r or of times)."""
    if case not in series_cache:
        h = h_case[case]
        mu, phi = axial_modes(h / k)
        beta, radial = radial_modes(h * R / k)
        b_m = np.trapezoid(heated_layer(z_quad)[:, None] * phi(z_quad), z_quad, axis=0)
        series_cache[case] = (mu, phi, beta, radial, b_m)
    mu, phi, beta, radial, b_m = series_cache[case]
    lam = alpha * (mu[:, None] ** 2 + (beta[None, :] / R) ** 2)
    axial = (b_m * phi(z)[0])[:, None]
    weight = axial * radial(r)[:, None, :]                         # (r, m, n)
    decay = np.exp(-lam[None] * np.atleast_1d(times)[:, None, None])  # (t, m, n)
    return np.einsum("rmn,tmn->tr", weight, decay).squeeze()


# The rear-face rise at the centre, computed vs exact, over the record $t\ge 5$ ms.

def half_rise(times, rise, fraction=0.5):
    peak = int(np.argmax(rise))
    target = fraction * rise[peak]
    i = int(np.argmax(rise[:peak + 1] >= target))
    return times[i - 1] + (target - rise[i - 1]) * (times[i] - times[i - 1]) / (rise[i] - rise[i - 1])


dense = np.linspace(5e-3, t_end, 3000)
exact_curves = {case: exact_rise(case, dense) for case in ambient}


def largest_deviation(run, name):
    keep = run["times"] >= 5e-3
    return np.max(np.abs(run["rear"][name][keep] - exact_rise(case_of[name], run["times"][keep])))


print("Rear-face centre: computed vs exact series")
deviation = {}
for name, run in runs.items():
    case, rise, times = case_of[name], run["rear"][name], run["times"]
    deviation[name] = largest_deviation(run, name)
    print(f"  {name:15s}: max rise {rise.max():.5f} K vs {exact_curves[case].max():.5f} K "
          f"({100 * (rise.max() / exact_curves[case].max() - 1):+.3f} %), t_1/2 {half_rise(times, rise) * 1e3:.2f} ms "
          f"vs {half_rise(dense, exact_curves[case]) * 1e3:.2f} ms, largest deviation {deviation[name] * 1e3:.2f} mK")

# **Mesh study** (furnace case). The mesh size is measured by
# $N^{-1/3}$ ($N$ tetrahedra); the observed order $p$ and the Richardson estimate follow from
# the three levels.


def richardson(values, sizes):
    f1, f2, f3 = values
    h1, h2, h3 = sizes
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.2, 12.0)
    except ValueError:
        return np.nan, np.nan
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


mesh_names = ("furnace_coarse", "furnace_medium", "furnace_fine")
sizes = [len(disc_runs[n]["cells"]) ** (-1 / 3) for n in mesh_names]
exact_peak = exact_curves["furnace, 800 °C"].max()
exact_half = half_rise(dense, exact_curves["furnace, 800 °C"])
print("Mesh study, furnace case")
for label, values, ref in (
        ("max rise / K", [runs[n]["rear"][n].max() for n in mesh_names], exact_peak),
        ("t_1/2 / ms", [half_rise(runs[n]["times"], runs[n]["rear"][n]) * 1e3 for n in mesh_names], exact_half * 1e3)):
    p, extrapolated = richardson(values, sizes)
    print(f"  {label:13s} " + "  ".join(f"{v:.5f}" for v in values)
          + f"  | order {p:.2f}, Richardson {extrapolated:.5f}, exact {ref:.5f}; change between the last two meshes "
          f"{abs(values[2] - values[1]):.2e} ({100 * abs(values[2] / values[1] - 1):.3f} %)")
print("  largest deviation from the series / mK: " + "  ".join(f"{deviation[n] * 1e3:.3f}" for n in mesh_names))

# **Time-step study** (coarse furnace disc): the step tolerance changes the time-integration
# error only (the coarse-mesh spatial error is common to all runs); the 0.1 mK run is the
# reference.

print("Time-step study, coarse furnace disc")
reference_rise = step_runs[1e-4]["rear"]["furnace_coarse"]
reference_times = step_runs[1e-4]["times"]
time_error = {}
for tol in (3e-3, 1e-3, 3e-4, 1e-4):
    run = step_runs[tol]
    rise = run["rear"]["furnace_coarse"]
    shift = np.max(np.abs(np.interp(dense, run["times"], rise) - np.interp(dense, reference_times, reference_rise)))
    time_error[tol] = shift
    print(f"  tolerance {tol * 1e3:.1f} mK: {run['steps']:4d} steps, max rise {rise.max():.6f} K, "
          f"t_1/2 {half_rise(run['times'], rise) * 1e3:.3f} ms, largest difference to the 0.1 mK run {shift * 1e3:.3f} mK")

# **Energy balance (loss-free disc).** Without losses the energy stays in the sample: the mean
# temperature rise must equal $Q/(\rho c_p L)=1$ K at every time. The quadratic field is
# integrated exactly over the disc with a degree-2 rule (4 points per tetrahedron).

coords = disc_runs["lossfree_fine"]["coordinates"]
tets = disc_runs["lossfree_fine"]["cells"]
times = runs["lossfree_fine"]["times"]
p = coords[tets]
volumes = np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6
a_q, b_q = 0.5854101966249685, 0.1381966011250105
bary = np.array([[a_q, b_q, b_q, b_q], [b_q, a_q, b_q, b_q], [b_q, b_q, a_q, b_q], [b_q, b_q, b_q, a_q]])
quad_points = np.einsum("qk,ckd->cqd", bary, p).reshape(-1, 3)
for t in (times[0], times[np.argmin(np.abs(times - 0.3))], times[-1]):
    T_q = temperature(runs["lossfree_fine"], quad_points, t)[0].reshape(-1, 4) - runs["lossfree_fine"]["T_inf"]["lossfree_fine"]
    mean_rise = float(np.sum(T_q.mean(axis=1) * volumes) / volumes.sum())
    print(f"  loss-free disc, t = {t * 1e3:7.1f} ms: mean rise {mean_rise:.6f} K vs {dT_adiabatic:.6f} K "
          f"({100 * (mean_rise - dT_adiabatic):+.4f} %)")

# **Diffusivity recovery.** Times are counted from the flash ($t+t_0$). Parker:
# $\alpha=0.1388\,L^2/t_{1/2}$. Cowan and Clark-Taylor both rest on Cowan's one-dimensional
# model of a slab with equal face losses (Biot number $Y=hL/k$): Cowan matches the ratio
# $\Delta T(10\,t_{1/2})/\Delta T(t_{1/2})$, Clark and Taylor the rise-time ratio
# $t_{0.7}/t_{0.3}$, to find $Y$, and then $\alpha=\omega_{1/2}(Y)\,L^2/t_{1/2}$ with the
# dimensionless half-rise time $\omega_{1/2}$ of that model. The model is evaluated exactly
# (its eigen-series) instead of through the published polynomial fits. Both methods ignore the
# rim losses, so a residual bias remains for a finite disc; the same analysis applied to the
# exact series separates this method bias from the discretisation error.


def cowan_model(Y):
    mu, phi = axial_modes(Y / L, 120)
    weight = phi(0.0)[0] * phi(L)[0]
    value = lambda omega: np.exp(-np.outer(np.atleast_1d(omega), mu**2) * L**2) @ weight
    grid = np.linspace(0.005, 1.5, 3000)
    curve = value(grid)
    peak = int(np.argmax(curve))
    top = curve[peak] if peak < len(grid) - 1 else value(50.0)[0]
    upper = grid[peak] if peak < len(grid) - 1 else 1.5
    at = lambda f: brentq(lambda s: value(s)[0] - f * top, 0.005, upper, xtol=1e-13)
    o3, o5, o7 = at(0.3), at(0.5), at(0.7)
    return o5, o7 / o3, value(10 * o5)[0] / value(o5)[0]


ideal = cowan_model(0.0)


def recover(times_since_start, rise):
    t = times_since_start + t0
    t3, t5, t7 = (half_rise(t, rise, f) for f in (0.3, 0.5, 0.7))
    ratio_ct, ratio_10 = t7 / t3, np.interp(10 * t5, t, rise) / np.interp(t5, t, rise)
    Y_cowan = brentq(lambda Y: cowan_model(Y)[2] - ratio_10, 0.0, 3.0) if ratio_10 < ideal[2] else 0.0
    Y_ct = brentq(lambda Y: cowan_model(Y)[1] - ratio_ct, 0.0, 3.0) if ratio_ct < ideal[1] else 0.0
    return dict(t5=t5, parker=0.1388 * L**2 / t5, cowan=cowan_model(Y_cowan)[0] * L**2 / t5,
                ct=cowan_model(Y_ct)[0] * L**2 / t5, Y_cowan=Y_cowan, Y_ct=Y_ct)


print(f"Recovered diffusivity (true alpha = {alpha:.4e} m^2/s): relative deviation in %")
print(f"  {'case':16s} {'curve':14s} {'t_1/2 / ms':>10s} {'Parker':>8s} {'Cowan':>8s} {'Clark-Taylor':>13s}   fitted Y (Cowan, CT)")
for name in ("lossfree_fine", "room_fine", "furnace_fine"):
    case = case_of[name]
    for label, (t, rise) in (("OneSimulation", (runs[name]["times"], runs[name]["rear"][name])),
                             ("exact series", (dense, exact_curves[case]))):
        out = recover(t, rise)
        print(f"  {case:16s} {label:14s} {out['t5'] * 1e3:10.2f} {100 * (out['parker'] / alpha - 1):+8.2f} "
              f"{100 * (out['cowan'] / alpha - 1):+8.2f} {100 * (out['ct'] / alpha - 1):+13.2f}   "
              f"{out['Y_cowan']:.4f}, {out['Y_ct']:.4f}  (h L/k = {h_case[case] * L / k:.4f})")


# ## Figure
#
# Top: temperature over the $r$-$z$ section through the axis of the fine furnace disc (full
# diameter, thickness exaggerated three times; the flash heated the bottom face) at three
# instants. Bottom, from left: the disc surface at the half-rise time; rear-face rise against
# the exact series for the three cases; rear-face radial profiles of the furnace case; mesh
# and time-step study.

fine_run = runs["furnace_fine"]
x_fine, coords = 0.0, disc_runs["furnace_fine"]["coordinates"]
T_inf_f = fine_run["T_inf"]["furnace_fine"]
fig = plt.figure(figsize=(18, 10.5), constrained_layout=True)
outer = fig.add_gridspec(2, 1, height_ratios=(0.8, 1.0))
top = outer[0].subgridspec(1, 3)
bottom = outer[1].subgridspec(1, 4, width_ratios=(1.1, 1.0, 1.0, 1.0))
xr = np.linspace(-R, R, 255) * (1 - 1e-9)
zr = np.linspace(0.0, L, 81)
XR, ZR = np.meshgrid(xr, zr, indexing="ij")
section_points = np.column_stack((x_fine + XR.ravel(), np.zeros(XR.size), ZR.ravel()))
for col, t_show in enumerate((0.03, 0.15, 1.2)):
    ax = fig.add_subplot(top[col])
    values, t_near = temperature(fine_run, section_points, t_show)
    values = values.reshape(XR.shape) - T_inf_f
    low = 0.0 if values.min() < 0.05 * values.max() else values.min()
    filled = ax.contourf(XR * 1e3, ZR * 3e3, np.clip(values, low, None), levels=np.linspace(low, values.max(), 41),
                         cmap=FIELD_CMAP)
    lines = values.min() + (values.max() - values.min()) * np.linspace(0.05, 0.95, 10)
    ax.contour(XR * 1e3, ZR * 3e3, values, levels=lines, colors="w", linewidths=0.4, alpha=0.7)
    ax.set_aspect("equal")
    ax.set(xlabel="x through the axis / mm", ylabel="3 z / mm" if col == 0 else None,
           title=f"Furnace case, t + t0 = {(t_near + t0) * 1e3:.0f} ms: T - T_inf on the r-z section")
    fig.colorbar(filled, ax=ax, label="K", orientation="horizontal", shrink=0.8, pad=0.02)

ax3d = fig.add_subplot(bottom[0], projection="3d")
ax3d.set_facecolor(CREAM)
faces = np.array(disc_runs["furnace_fine"]["groups"]["furnace_fine_rear"] + disc_runs["furnace_fine"]["groups"]["furnace_fine_rim"])
polys = coords[faces] - np.array([x_fine, 0.0, 0.0])
inside = polys.mean(axis=1) * np.array([1 - 1e-6, 1 - 1e-6, 1.0])
inside[:, 2] = np.clip(inside[:, 2], 1e-9, L - 1e-9)
face_T, t_near = temperature(fine_run, inside + np.array([x_fine, 0.0, 0.0]), t_half_parker)
face_T = face_T - T_inf_f
norm = colors.Normalize(vmin=face_T.min(), vmax=face_T.max())
ax3d.add_collection3d(Poly3DCollection(polys * 1e3, facecolors=FIELD_CMAP(norm(face_T)), edgecolors="k",
                                       linewidths=0.05))
ax3d.set(xlim=(-R * 1e3, R * 1e3), ylim=(-R * 1e3, R * 1e3), zlim=(0, L * 1e3), xlabel="x / mm", ylabel="y / mm")
ax3d.set_zticks([0, 1, 2])
ax3d.set_box_aspect((2 * R, 2 * R, 2.5 * L))
ax3d.view_init(elev=28, azim=-60)
ax3d.set_title(f"Rear face and rim, t + t0 = {(t_near + t0) * 1e3:.0f} ms (furnace)\n"
               f"fine mesh, {len(disc_runs['furnace_fine']['cells'])} P2 tetrahedra")
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax3d, shrink=0.5, pad=0.08,
             label="T - T_inf / K")

ax = fig.add_subplot(bottom[1])
for name, colour in zip(("lossfree_fine", "room_fine", "furnace_fine"), (GREEN, "#A8BC91", COPPER)):
    case = case_of[name]
    ax.plot((dense + t0) * 1e3, exact_curves[case], color=colour, lw=1.5, label=f"exact series, {case}")
    ax.plot((runs[name]["times"] + t0) * 1e3, runs[name]["rear"][name], "o", ms=2.8, mfc="none", color=colour)
ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
ax.set(xlabel="time since the flash / ms", ylabel="rear-face centre rise / K",
       title="Rear-face rise (circles: OneSimulation, adaptive steps)", xlim=(0, t_end * 1e3))
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(bottom[2])
r_profile = np.linspace(0.0, R, 64) * (1 - 1e-9)
for t_show, colour in zip((0.3, 1.0, 3.0), ("#A8BC91", GREEN, "#214D43")):
    computed, t_near = temperature(fine_run, np.column_stack((x_fine + r_profile, np.zeros_like(r_profile),
                                                         np.full_like(r_profile, L))), t_show)
    computed = computed - T_inf_f
    ax.plot(r_profile * 1e3, exact_rise("furnace, 800 °C", [t_near], r=r_profile), color=colour, lw=1.5,
            label=f"exact, t + t0 = {(t_near + t0) * 1e3:.0f} ms")
    ax.plot(r_profile[::3] * 1e3, computed[::3], "o", ms=3.5, mfc="none", color=colour)
ax.set(xlabel="r / mm", ylabel="rear-face rise / K", title="Rear face, furnace case: rim cooling")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

study = bottom[3].subgridspec(2, 1, hspace=0.15)
ax = fig.add_subplot(study[0])
ax.loglog([len(disc_runs[n]["cells"]) for n in mesh_names], [deviation[n] * 1e3 for n in mesh_names], "o-",
          color=GREEN, label="furnace case, tolerance 0.3 mK")
ax.set(xlabel="tetrahedra per disc", ylabel="max deviation / mK", title="Mesh study (vs exact series)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax = fig.add_subplot(study[1])
tols = (3e-3, 1e-3, 3e-4)
ax.loglog([step_runs[t]["steps"] for t in tols], [time_error[t] * 1e3 for t in tols], "s-", color=COPPER,
          label="tolerance 3, 1, 0.3 mK (coarse disc)")
ax.set(xlabel="accepted time steps", ylabel="difference / mK", title="Time-step study (vs 0.1 mK run)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
plt.show()


# ## Going further
#
# Replace the linearised radiation by the full Stefan-Boltzmann law (`RadiationCooling`) at
# 1500 °C, where the rise of a few kelvin already makes the loss nonlinear; model a finite
# laser pulse as a time-dependent front-face flux and compare with the finite-pulse
# correction; or shrink the sample radius until the rim losses dominate and the
# one-dimensional Cowan correction no longer suffices.
