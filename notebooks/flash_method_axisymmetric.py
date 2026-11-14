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

import time
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1

from onesim.cad.d2 import Circle, Plane
from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, TriangularMesher, extrude
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ConjugateGradient
from onesim.numerics.time import BDF, AdaptiveTimeStep
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.results import ResultRetentionPolicy
from onesim.results.comparisons import convergence
from onesim.results.integration import mean_value
from onesim.simulation import ExecutionOptions, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# A Pyroceram 9606 glass-ceramic disc, the usual reference material of flash diffusivity
# measurements, 12.7 mm in diameter and 2 mm thick (the sample size of ASTM E1461). Properties
# at 300 K from Incropera et al., *Fundamentals of Heat and Mass Transfer*, Table A.3. The flash
# energy raises a loss-free sample by 1 K. Three loss settings: no losses (the Parker limit),
# room temperature and a furnace at 800 °C, each with free convection $h_c=10$ W/(m² K) and
# emissivity 0.9 on all faces. The same constant properties are used in all three cases, so
# that only the losses change.

R = 6.35e-3
L = 2.0e-3
k = 3.98
rho = 2600.0
cp = 808.0
alpha = k / (rho * cp)
dT_adiabatic = 1.0
Q = rho * cp * L * dT_adiabatic
w = 0.15e-3
t0 = w**2 / (4 * alpha)
h_conv, emissivity, sigma = 10.0, 0.9, 5.670374e-8
t_half_parker = 0.1388 * L**2 / alpha
t_end = 3.2
ambient = {"loss-free": 293.15, "room, 20 °C": 293.15, "furnace, 800 °C": 1073.15}
h_case = {"loss-free": 0.0, **{case: h_conv + 4 * emissivity * sigma * ambient[case] ** 3
                               for case in ("room, 20 °C", "furnace, 800 °C")}}
tolerance = 3e-4
print(f"alpha = {alpha:.4e} m^2/s, Parker half-rise time {t_half_parker * 1e3:.1f} ms, "
      f"Q = {Q:.0f} J/m^2, heated layer w = {w * 1e3:.2f} mm (t0 = {t0 * 1e3:.2f} ms)")
for case, h in h_case.items():
    print(f"  {case:16s}: h = {h:6.2f} W/(m^2 K), face Biot hL/k = {h * L / k:.4f}, rim Biot hR/k = {h * R / k:.4f}")


# ## Geometry and mesh
#
# The disc face is triangulated by `TriangularMesher` on the exact circle, finer at the rim
# (where the radial losses bend the field) than in the centre. `extrude` sweeps the
# triangulation through the thickness in layers graded towards the heated front face
# (several quadratic elements across the heated layer $w$) and splits every prism into three
# tetrahedra by a conforming diagonal rule. The faces are named `front` ($z=0$, heated),
# `rear` ($z=L$, where the detector looks) and the rim (`circumference`).

levels = {"coarse": (3, 6, 7.0, 16), "medium": (4, 8, 10.0, 22), "fine": (5, 10, 14.0, 28)}


def disc_mesh(rings, layers, ratio, rim_nodes):
    face = TriangularMesher(target_size=1.25 * R / rings,
                            local_sizes={"circumference": 2 * np.pi * R / rim_nodes}).generate(Circle(radius=R))
    return extrude(face, L, Geometric(layers, ratio=ratio), names=("front", "rear"))


main_samples = [("furnace_coarse", "coarse"), ("furnace_medium", "medium"), ("furnace_fine", "fine"),
                ("room_fine", "fine"), ("lossfree_fine", "fine")]
case_of = {"furnace_coarse": "furnace, 800 °C", "furnace_medium": "furnace, 800 °C",
           "furnace_fine": "furnace, 800 °C", "room_fine": "room, 20 °C", "lossfree_fine": "loss-free"}
meshes = {level: disc_mesh(*spec) for level, spec in levels.items()}
for level, mesh in meshes.items():
    z = np.unique(np.round(np.asarray(mesh.coordinates)[:, 2], 12))
    print(f"{level:6s} disc: {mesh.cell_count:5d} tetrahedra, {len(mesh.coordinates):4d} vertices, {len(z) - 1:2d} layers, "
          f"first layer {z[1] * 1e6:.0f} um, last layer {(z[-1] - z[-2]) * 1e6:.0f} um")


# ## Model
#
# One `HeatConduction` model per run; the disc gets the boundary conditions of its loss case on
# its three face groups and the heated-layer initial field.

material = IsotropicSolid(density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
                          youngs_modulus=9.6e10, poisson_ratio=0.24, thermal_expansion_coefficient=5.7e-6)


def heated_layer(z):
    return Q / (rho * cp) * 2.0 / (np.sqrt(np.pi) * w) * np.exp(-(np.asarray(z) / w) ** 2)


def run_disc(name, level, step_tolerance):
    mesh = meshes[level]
    case = case_of[name]
    geometry = mesh.geometry
    b = geometry.boundaries
    model = HeatConduction(domain=geometry.domain(), material=material, heating=0.0)
    faces = b.front | b.rear | b.circumference
    if h_case[case] > 0:
        condition = ConvectiveHeatFlux(boundary=faces, physics=model, heat_transfer_coefficient=h_case[case],
                                       ambient_temperature=ambient[case])
    else:
        condition = HeatFlux(boundary=faces, physics=model, value=0.0)
    problem = Problem(name="Flash method", geometry=geometry, physics=(model,), boundary_conditions=(condition,),
                      initial_conditions=(InitialValue(model.temperature,
                                                       lambda x, y, z: ambient[case] + heated_layer(z)),))
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)),
        linear_solver=ConjugateGradient(relative_tolerance=1e-10, maximum_iterations=20000),
        time_integrator=BDF(order=3)))
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
    times = np.array(sorted(frames))
    field = result.field(model.temperature).with_states(times, [frames[t] for t in times])
    rear = np.array([float(field.evaluate([[0.0, 0.0, L]], time=t)[0]) for t in times]) - ambient[case]
    report = result.result_set.solver_report
    return dict(field=field, times=times, rear=rear, T_inf=ambient[case], mesh=mesh, b=b,
                steps=report.get("accepted_steps"), rejected=report.get("rejected_steps"), seconds=time.time() - start)


def nearest_time(run, t):
    return float(run["times"][int(np.argmin(np.abs(run["times"] - t)))])


def temperature(run, points, t):
    """The stored quadratic field of a run at the accepted time nearest to t."""
    t_near = nearest_time(run, t)
    return np.asarray(run["field"].evaluate(np.asarray(points), time=t_near)).ravel(), t_near


# ## Solve
#
# Five runs with the local error tolerance of 0.3 mK; for the time-step study the coarse
# furnace disc is repeated with tolerances of 3 mK, 1 mK and 0.1 mK.

runs = {}
for name, level in main_samples:
    runs[name] = run_disc(name, level, tolerance)
    print(f"{name:15s} ({meshes[level].cell_count:5d} tetrahedra): {runs[name]['steps']} accepted / "
          f"{runs[name]['rejected']} rejected steps ({runs[name]['seconds']:.0f} s)")
step_runs = {tolerance: runs["furnace_coarse"]}
for tol in (3e-3, 1e-3, 1e-4):
    step_runs[tol] = run_disc("furnace_coarse", "coarse", tol)
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
    weight = axial * radial(r)[:, None, :]
    decay = np.exp(-lam[None] * np.atleast_1d(times)[:, None, None])
    return np.einsum("rmn,tmn->tr", weight, decay).squeeze()


def half_rise(times, rise, fraction=0.5):
    peak = int(np.argmax(rise))
    target = fraction * rise[peak]
    i = int(np.argmax(rise[:peak + 1] >= target))
    return times[i - 1] + (target - rise[i - 1]) * (times[i] - times[i - 1]) / (rise[i] - rise[i - 1])


dense = np.linspace(5e-3, t_end, 3000)
exact_curves = {case: exact_rise(case, dense) for case in ambient}


def largest_deviation(run, name):
    keep = run["times"] >= 5e-3
    return np.max(np.abs(run["rear"][keep] - exact_rise(case_of[name], run["times"][keep])))


print("Rear-face centre: computed vs exact series")
deviation = {}
for name, run in runs.items():
    case, rise, times = case_of[name], run["rear"], run["times"]
    deviation[name] = largest_deviation(run, name)
    print(f"  {name:15s}: max rise {rise.max():.5f} K vs {exact_curves[case].max():.5f} K "
          f"({100 * (rise.max() / exact_curves[case].max() - 1):+.3f} %), t_1/2 {half_rise(times, rise) * 1e3:.2f} ms "
          f"vs {half_rise(dense, exact_curves[case]) * 1e3:.2f} ms, largest deviation {deviation[name] * 1e3:.2f} mK")

# **Mesh study** (furnace case). The mesh size is measured by
# $N^{-1/3}$ ($N$ tetrahedra); the observed order $p$ and the Richardson estimate follow from
# the three levels.


mesh_names = ("furnace_coarse", "furnace_medium", "furnace_fine")
sizes = [runs[n]["mesh"].cell_count ** (-1 / 3) for n in mesh_names]
exact_peak = exact_curves["furnace, 800 °C"].max()
exact_half = half_rise(dense, exact_curves["furnace, 800 °C"])
print("Mesh study, furnace case")
for label, values, ref in (
        ("max rise / K", [runs[n]["rear"].max() for n in mesh_names], exact_peak),
        ("t_1/2 / ms", [half_rise(runs[n]["times"], runs[n]["rear"]) * 1e3 for n in mesh_names], exact_half * 1e3)):
    estimate = convergence(values, sizes)
    order = f"{estimate.order:.2f}" if estimate.order is not None else "n/a"
    extrapolated = f"{estimate.extrapolated:.5f}" if estimate.extrapolated is not None else "n/a"
    print(f"  {label:13s} " + "  ".join(f"{v:.5f}" for v in values)
          + f"  | order {order}, Richardson {extrapolated}, exact {ref:.5f}; change between the last two meshes "
          f"{abs(values[2] - values[1]):.2e} ({100 * abs(values[2] / values[1] - 1):.3f} %)")
print("  largest deviation from the series / mK: " + "  ".join(f"{deviation[n] * 1e3:.3f}" for n in mesh_names))

# **Time-step study** (coarse furnace disc): the step tolerance changes the time-integration
# error only (the coarse-mesh spatial error is common to all runs); the 0.1 mK run is the
# reference.

print("Time-step study, coarse furnace disc")
reference_rise = step_runs[1e-4]["rear"]
reference_times = step_runs[1e-4]["times"]
time_error = {}
for tol in (3e-3, 1e-3, 3e-4, 1e-4):
    run = step_runs[tol]
    rise = run["rear"]
    shift = np.max(np.abs(np.interp(dense, run["times"], rise) - np.interp(dense, reference_times, reference_rise)))
    time_error[tol] = shift
    print(f"  tolerance {tol * 1e3:.1f} mK: {run['steps']:4d} steps, max rise {rise.max():.6f} K, "
          f"t_1/2 {half_rise(run['times'], rise) * 1e3:.3f} ms, largest difference to the 0.1 mK run {shift * 1e3:.3f} mK")

# **Energy balance (loss-free disc).** Without losses the energy stays in the sample: the mean
# temperature rise must equal $Q/(\rho c_p L)=1$ K at every time. The quadratic field is
# averaged over the disc with the library's volume integral.

lossfree = runs["lossfree_fine"]
for t in (lossfree["times"][0], nearest_time(lossfree, 0.3), lossfree["times"][-1]):
    mean_rise = mean_value(lossfree["field"], time=t) - lossfree["T_inf"]
    print(f"  loss-free disc, t = {t * 1e3:7.1f} ms: mean rise {mean_rise:.6f} K vs {dT_adiabatic:.6f} K "
          f"({100 * (mean_rise - dT_adiabatic):+.4f} %)")


# ## Figure
#
# Top: temperature over the section through the axis of the fine furnace disc (full
# diameter, thickness exaggerated three times; the flash heated the bottom face) at three
# instants. Bottom, from left: the rear face and rim at the half-rise time; rear-face rise
# against the exact series for the three cases; rear-face radial profiles of the furnace case;
# mesh and time-step study.

fine_run = runs["furnace_fine"]
field, b, T_inf_f = fine_run["field"], fine_run["b"], fine_run["T_inf"]
rise_of = lambda T, p: T - T_inf_f
r_profile = np.linspace(0.0, R, 64) * (1 - 1e-9)
rear_points = np.column_stack((r_profile, np.zeros_like(r_profile), np.full_like(r_profile, L)))
tolerances = (3e-3, 1e-3, 3e-4)
rim_computed, rim_exact = {}, {}
for t_show in (0.3, 1.0, 3.0):
    computed, t_near = temperature(fine_run, rear_points, t_show)
    label = f"t + t0 = {(t_near + t0) * 1e3:.0f} ms"
    rim_computed[label] = (r_profile[::3] * 1e3, (computed - T_inf_f)[::3])
    rim_exact[label] = (r_profile * 1e3, exact_rise("furnace, 800 °C", [t_near], r=r_profile))
fig = plots.figure(layout=[["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                           ["surface"] * 3 + ["rise"] * 3 + ["rim"] * 3 + ["mesh"] * 3,
                           ["surface"] * 3 + ["rise"] * 3 + ["rim"] * 3 + ["steps"] * 3], size=(18, 11.5))
for panel, t_show in zip("abc", (0.03, 0.15, 1.2)):
    t_near = nearest_time(fine_run, t_show)
    plots.section(field, time=t_near, plane=Plane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0)), component=rise_of,
                  contours=10, subdivisions=3, label="K", aspect=3.0, ax=fig[panel],
                  title=f"Furnace case, T - T_inf, t + t0 = {(t_near + t0) * 1e3:.0f} ms")
t_near = nearest_time(fine_run, t_half_parker)
plots.surface(field, time=t_near, boundary=b.rear | b.circumference, component=rise_of, subdivisions=2,
              camera=plots.Camera(elevation=28, azimuth=-60), edges=True, label="T - T_inf [K]", ax=fig["surface"],
              title=f"Rear face and rim, t + t0 = {(t_near + t0) * 1e3:.0f} ms (furnace), "
                    f"{fine_run['mesh'].cell_count} P2 tetrahedra")
plots.line(points={case_of[name]: ((runs[name]["times"] + t0) * 1e3, runs[name]["rear"])
                   for name in ("lossfree_fine", "room_fine", "furnace_fine")},
           reference={case_of[name]: ((dense + t0) * 1e3, exact_curves[case_of[name]])
                      for name in ("lossfree_fine", "room_fine", "furnace_fine")},
           hlines=0.5, xlimits=(0, t_end * 1e3), xlabel="time since the flash [ms]", ylabel="rear-face centre rise [K]",
           title="Rear-face rise: OneSimulation (markers), exact series (dashed)", ax=fig["rise"])
plots.line(points=rim_computed, reference=rim_exact, xlabel="r [mm]", ylabel="rear-face rise [K]",
           title="Rear face, furnace case: rim cooling", ax=fig["rim"])
plots.convergence([runs[name]["mesh"].cell_count for name in mesh_names],
                  {"furnace case, tolerance 0.3 mK": [deviation[name] * 1e3 for name in mesh_names]},
                  xlabel="tetrahedra per disc", ylabel="max deviation [mK]", title="Mesh study (vs exact series)", ax=fig["mesh"])
plots.convergence([step_runs[tol]["steps"] for tol in tolerances],
                  {"tolerance 3, 1, 0.3 mK (coarse disc)": [time_error[tol] * 1e3 for tol in tolerances]},
                  xlabel="accepted time steps", ylabel="difference [mK]", title="Time-step study (vs 0.1 mK run)", ax=fig["steps"])
plots.show()


# ## Going further
#
# Replace the linearised radiation by the full Stefan-Boltzmann law (`RadiationCooling`) at
# 1500 °C, where the rise of a few kelvin already makes the loss nonlinear; model a finite
# laser pulse as a time-dependent front-face flux and compare with the finite-pulse
# correction; or shrink the sample radius until the rim losses dominate and the
# one-dimensional Cowan correction no longer suffices.
