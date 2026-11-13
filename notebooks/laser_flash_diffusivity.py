"""OneSimulation example: laser-flash measurement of thermal diffusivity.
Requires the installed, activated OneSimulation runtime.

The OneSimulation rear-face response is checked against Parker's closed-form
series and the diffusivity is recovered from the half-rise time.
"""


# # Laser-flash measurement of thermal diffusivity
#
# The laser-flash method (Parker et al., 1961) is the standard way to measure the
# thermal diffusivity of solids. A thin, thermally insulated disc of thickness $L$
# receives a very short energy pulse $Q$ (per unit area) on its front face at
# $t=0$. The heat diffuses through the thickness and the **rear-face
# temperature** rises and saturates. The time $t_{1/2}$ at which the rear face
# has reached half of its final rise depends only on $L$ and $\alpha$, so the
# diffusivity follows from a single transient.
#
# The transient heat equation is solved with every face insulated except for the
# pulsed front face:
#
# $$\rho c_p \,\frac{\partial T}{\partial t} = k\,\frac{\partial^2 T}{\partial x^2},\qquad
#   -k\,\partial_x T\big|_{x=0} = q_p(t),\quad \partial_x T\big|_{x=L}=0,\quad T(x,0)=T_i,$$
#
# with a short pulse $q_p(t)$ of duration $\tau_p$ and energy
# $\int q_p\,dt = Q$. OneSimulation solves it spatially with finite elements
# (Lagrange order 2) on a real 3D mesh; the lateral faces and the rear face are
# insulated. The run is split into the two phases of the experiment: the
# **flash** ($0\le t\le\tau_p$, fine time steps resolving the pulse) and the
# **diffusion** ($t>\tau_p$, BDF2 time steps on the scale of $t_{1/2}$).
#
# **Reference.** For an instantaneous pulse the rear-face rise, normalised by its
# final value $\Delta T_\infty = Q/(\rho c_p L)$, is
#
# $$\theta(t)=\frac{T(L,t)-T_i}{\Delta T_\infty}
#   = 1 + 2\sum_{n=1}^{\infty}(-1)^n\exp\!\Big(-\frac{n^2\pi^2\alpha t}{L^2}\Big),$$
#
# and the half-rise time is $t_{1/2}=1.38\,L^2/(\pi^2\alpha)$ (Parker's constant
# rounds the exact root $1.370$ of $\theta=1/2$). Inverting it gives the
# measurement formula
#
# $$\alpha = \frac{1.38\,L^2}{\pi^2\,t_{1/2}}.$$
#
# We check the whole rear-face curve against the series and recover $\alpha$ from
# the simulated half-rise time.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.heat_transfer import HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions.values import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# A 2 mm stainless-steel disc, as used in a laser-flash apparatus. The sample lies
# along $x$; its cross-section is small and only serves to form a slender 3D mesh.
# The pulse energy gives a rear-face rise of about one kelvin, typical for the
# method. The pulse has a smooth (raised-cosine) shape like a real flash lamp.
#
# **Finite pulse duration.** A real flash lasts a finite time $\tau_p$. Its
# response equals the instantaneous-pulse response with the time origin shifted
# to the pulse centroid $\tau_p/2$, up to terms of order $(\tau_p/t_{1/2})^2$.
# With $\tau_p = 1\,\%$ of $t_{1/2}$ the uncorrected half-rise time is biased by
# only $0.5\,\%$ and the residual after the centroid shift is below $10^{-4}$;
# the finite-pulse correction is therefore negligible here, and we simply count
# time from the pulse centroid.

L = 2.0e-3            # m, sample thickness (x direction)
cross = 0.5e-3        # m, lateral size (insulated)
k = 15.0              # W/(m·K), thermal conductivity (stainless steel)
rho = 7900.0          # kg/m^3
cp = 500.0            # J/(kg·K)
T_i = 300.0           # K, initial temperature
Q = 1.0e4             # J/m^2, absorbed pulse energy per unit front area
alpha = k / (rho * cp)
t_half_ref = 1.38 * L**2 / (np.pi**2 * alpha)     # Parker's half-rise time
dT_inf = Q / (rho * cp * L)                        # final (adiabatic) rise
tau_p = 0.01 * t_half_ref                          # pulse duration, 1 % of t_1/2
flash_steps = 16                                   # time steps resolving the pulse
end_time = 5.0 * t_half_ref                        # the rise has saturated to within 0.3 %
diffusion_step = t_half_ref / 32.0
print(f"alpha = {alpha:.3e} m^2/s,  t_1/2 (Parker) = {1000*t_half_ref:.2f} ms,  final rise = {dT_inf:.3f} K")
print(f"pulse duration = {1e6*tau_p:.0f} us in {flash_steps} steps,  diffusion step = {1e3*diffusion_step:.2f} ms,  "
      f"{round((end_time - tau_p) / diffusion_step)} diffusion steps")

geometry = Box(length=L, width=cross, height=cross)
# Structured hexahedral mesh: many cells along x (the 1D direction), few across.
mesh = StructuredHexahedralMesher(cells=(60, 1, 1)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
    youngs_modulus=200e9, poisson_ratio=0.3, thermal_expansion_coefficient=1.6e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=0.0)
lateral = (geometry.boundaries.y_min | geometry.boundaries.y_max
           | geometry.boundaries.z_min | geometry.boundaries.z_max)


# ## Phase 1: the flash
#
# The front face receives the raised-cosine pulse
# $q_p(t) = (2Q/\tau_p)\sin^2(\pi t/\tau_p)$, whose integral is exactly $Q$. The
# pulse is integrated with backward Euler steps, which deposit the pulse energy
# exactly (the trapezoidal sum of a raised cosine is exact).

def pulse_flux(t):
    # HeatFlux is the outward flux; heating the face means a negative value.
    if 0.0 < t < tau_p:
        return -2.0 * Q / tau_p * np.sin(np.pi * t / tau_p) ** 2
    return 0.0


flash_problem = Problem(
    name="Laser flash: pulse phase",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        # front face: short heat pulse
        HeatFlux(boundary=geometry.boundaries.x_min, physics=model, value=pulse_flux),
        # rear face: insulated, its temperature is the measured signal
        HeatFlux(boundary=geometry.boundaries.x_max, physics=model, value=0.0),
        # laterally insulated (zero flux) -> 1D behaviour
        HeatFlux(boundary=lateral, physics=model, value=0.0),
    ),
    initial_conditions=(InitialValue(model.temperature, T_i),),
)
flash = Simulation(
    problem=flash_problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)),
        linear_solver=SparseDirect(),
        time_integrator=BackwardEuler(),
    ),
)
flash.validate().require_valid()
flash_result = flash.run(start_time=0.0, end_time=tau_p, time_step=tau_p / flash_steps)
flash_times = np.asarray([float(t) for t in flash_result.times])


# ## Phase 2: diffusion through the sample
#
# The temperature field at the end of the pulse becomes the initial condition of
# the diffusion phase; every face is now insulated and BDF2 steps on the scale of
# $t_{1/2}$ follow the rear-face rise up to saturation.

def flashed_profile(x, y, z):
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    values = np.asarray(flash_result.field(model.temperature).evaluate(pts, time=tau_p)).reshape(-1)
    return values.reshape(x.shape)


diffusion_problem = Problem(
    name="Laser flash: diffusion phase",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        # all six faces insulated after the pulse
        HeatFlux(boundary=(geometry.boundaries.x_min | geometry.boundaries.x_max | lateral),
                 physics=model, value=0.0),
    ),
    initial_conditions=(InitialValue(model.temperature, flashed_profile),),
)
diffusion = Simulation(
    problem=diffusion_problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)),
        linear_solver=SparseDirect(),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
    ),
)
diffusion.validate().require_valid()
result = diffusion.run(start_time=tau_p, end_time=end_time, time_step=diffusion_step)
diffusion_times = np.asarray([float(t) for t in result.times])


# ## Rear-face response and the Parker series
#
# The energy stored in the sample at the end must equal the pulse energy $Q$
# (insulated sample); the rear-face rise is then normalised by the adiabatic
# rise $\Delta T_\infty = Q/(\rho c_p L)$ and read at every stored time of both
# phases.

rear = np.array([[L, cross / 2, cross / 2]])
front = np.array([[0.0, cross / 2, cross / 2]])


def rear_temperature(res, times):
    return np.array([float(np.asarray(res.field(model.temperature).evaluate(rear, time=float(t))).reshape(-1)[0])
                     for t in times])


stored_times = np.concatenate((flash_times, diffusion_times[diffusion_times > tau_p]))
rear_T = np.concatenate((rear_temperature(flash_result, flash_times),
                         rear_temperature(result, diffusion_times[diffusion_times > tau_p])))
rise = rear_T - T_i

# Energy check: the stored energy per unit area at t_end must equal Q.
x_fine = np.linspace(0.0, L, 1201)
pts_fine = np.column_stack((x_fine, np.full_like(x_fine, cross / 2), np.full_like(x_fine, cross / 2)))
T_end = np.asarray(result.field(model.temperature).evaluate(pts_fine, time=end_time)).reshape(-1)
stored_energy = rho * cp * np.trapezoid(T_end - T_i, x_fine)
print(f"Stored energy at t_end: {stored_energy:.2f} J/m^2  (pulse energy Q = {Q:.2f} J/m^2, "
      f"deviation {100*(stored_energy/Q - 1):+.3f} %)")
assert abs(stored_energy - Q) < 0.005 * Q, "The deposited pulse energy is not conserved to 0.5 %."
theta_num = rise / dT_inf


def parker_series(t, terms=200):
    """Normalised rear-face rise for an instantaneous pulse at t = 0."""
    t = np.asarray(t, dtype=float)
    theta = np.ones_like(t)
    for n in range(1, terms + 1):
        theta += 2.0 * (-1) ** n * np.exp(-(n * np.pi) ** 2 * alpha * np.maximum(t, 0.0) / L**2)
    return np.where(t > 0.0, theta, 0.0)


# Reference time counts from the pulse centroid (finite-pulse correction).
t_eff = stored_times - tau_p / 2.0
compare = stored_times >= tau_p
theta_ref = parker_series(t_eff[compare])
curve_error = float(np.max(np.abs(theta_num[compare] - theta_ref)))
late = stored_times >= 4.5 * t_half_ref
print(f"Rear-face rise at 4.5-5 t_1/2: {np.mean(theta_num[late]):.4f} of the adiabatic rise "
      f"(series: {np.mean(parker_series(t_eff[late])):.4f})")
print(f"Maximum deviation of the rear-face curve from the Parker series: {curve_error:.4f} (of the full rise)")
assert curve_error < 0.01, "Rear-face curve deviates from the series by more than 1 % of the rise."

# Half-rise time by linear interpolation of the simulated signal, then alpha.
i = int(np.argmax(theta_num >= 0.5))
t_half_sim = stored_times[i - 1] + (0.5 - theta_num[i - 1]) * (stored_times[i] - stored_times[i - 1]) / (theta_num[i] - theta_num[i - 1])
t_half_corr = t_half_sim - tau_p / 2.0          # centroid-shifted half-rise time
alpha_parker = 1.38 * L**2 / (np.pi**2 * t_half_corr)
alpha_exact_root = 1.3698 * L**2 / (np.pi**2 * t_half_corr)
print(f"Simulated half-rise time: {1000*t_half_sim:.3f} ms (Parker reference {1000*t_half_ref:.3f} ms)")
print(f"Recovered diffusivity (Parker 1.38):  {alpha_parker:.4e} m^2/s  ({100*(alpha_parker/alpha - 1):+.2f} %)")
print(f"Recovered diffusivity (exact 1.3698): {alpha_exact_root:.4e} m^2/s  ({100*(alpha_exact_root/alpha - 1):+.2f} %)")
assert abs(alpha_parker / alpha - 1.0) < 0.02, "Recovered diffusivity deviates by more than 2 %."
assert abs(alpha_exact_root / alpha - 1.0) < 0.01, "Recovered diffusivity (exact root) deviates by more than 1 %."

# Physical sanity: the rear face heats monotonically and never overshoots; the
# front face relaxes towards the same plateau from above.
assert np.all(np.diff(rear_T) >= -1e-9), "The rear face should heat up monotonically."
assert np.max(theta_num) < 1.0 + 5e-3, "The rear-face rise must not overshoot its plateau."
front_T_end = float(np.asarray(result.field(model.temperature).evaluate(front, time=end_time)).reshape(-1)[0])
front_T_pulse = float(np.asarray(flash_result.field(model.temperature).evaluate(front, time=tau_p)).reshape(-1)[0])
print(f"Front-face rise at the end of the pulse: {front_T_pulse - T_i:.2f} K;  front-rear difference at t_end: {front_T_end - rear_T[-1]:.2e} K")
assert front_T_pulse - T_i > 5.0 * dT_inf, "The front face should be strongly heated right after the flash."
assert abs(front_T_end - rear_T[-1]) < 0.01 * dT_inf, "The sample should be nearly uniform at the end."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
dense = np.linspace(0.0, end_time, 600)
ax.plot(1000 * dense, parker_series(dense - tau_p / 2.0), color="#8c5a2b", lw=1.6, label="Parker series")
ax.plot(1000 * stored_times[compare], theta_num[compare], "o", color="#b7792f", ms=3, label="OneSimulation FEM (rear face)")
ax.axhline(0.5, color="#9a8f7d", ls="--", lw=1.0)
ax.axvline(1000 * t_half_sim, color="#9a8f7d", ls="--", lw=1.0)
ax.text(1000 * t_half_sim, 0.53, r"  $t_{1/2}$ = " + f"{1000*t_half_sim:.1f} ms", color="#6b5a46", fontsize=9)
ax.set(xlabel="Time / ms", ylabel=r"Normalised rear-face rise $\theta$",
       title="Laser flash — rear-face response of a 2 mm steel disc", ylim=(-0.02, 1.05))
ax.legend(loc="lower right")
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Lengthen the pulse to several percent of $t_{1/2}$ and apply the finite-pulse
# correction (time origin at the pulse centroid) to see how it restores the
# diffusivity. Add convective or radiative losses on the faces: the plateau then
# decays and the Cowan or Clark–Taylor corrections become necessary. A
# two-layer sample tests the multilayer laser-flash analysis.
