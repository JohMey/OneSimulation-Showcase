"""OneSimulation example: one-phase Stefan problem, melting of a slab at its melting temperature.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form Neumann similarity solution.
"""


# # Stefan problem: melting front in a semi-infinite slab
#
# A slab of ice fills $x>0$ and is initially exactly at its melting temperature
# $T_m$. From $t=0$ on, the surface $x=0$ is held at $T_s>T_m$. A liquid layer
# forms and its front $s(t)$ moves into the solid; the solid stays at $T_m$,
# so all the heat conducted through the liquid goes into latent heat
# (one-phase Stefan problem).
#
# OneSimulation solves the problem with its conservative fixed-grid enthalpy
# method: the volumetric enthalpy $H$ is advanced in every finite-volume cell,
#
# $$\frac{\partial H}{\partial t}=\nabla\cdot\big(k\,\nabla T(H)\big),\qquad
#   T(H)=T_m+\frac{H-\rho L_f\,\phi}{\rho c_p},\quad
#   \phi=\min\!\big(\max(H/(\rho L_f),0),1\big),$$
#
# with the liquid fraction $\phi$ absorbing the latent heat $L_f$ at constant
# temperature. The melting front is reconstructed as the $\phi=\tfrac12$ contour.
#
# **Reference (Neumann).** With the thermal diffusivity $\alpha=k/(\rho c_p)$
# and the Stefan number $\mathrm{St}=c_p\,(T_s-T_m)/L_f$, the front follows
#
# $$s(t)=2\lambda\sqrt{\alpha t},\qquad
#   \lambda\,e^{\lambda^2}\operatorname{erf}(\lambda)=\frac{\mathrm{St}}{\sqrt\pi},$$
#
# and the temperature in the liquid layer is
#
# $$T(x,t)=T_s-(T_s-T_m)\,\frac{\operatorname{erf}\!\big(x/(2\sqrt{\alpha t})\big)}
#   {\operatorname{erf}(\lambda)},\qquad 0\le x\le s(t),$$
#
# while $T=T_m$ ahead of the front. We check the front position at several
# times, the melted volume and the temperature profile.

from importlib.metadata import version
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from onesim.physics.heat_transfer.phase_change.numerics.backends.stefan_enthalpy import (
    StefanEnthalpyProvider,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Water/ice at atmospheric pressure. Only the liquid properties enter the
# one-phase problem, because the solid never departs from $T_m$. The domain
# is long enough that the front stays far from the insulated back face.

k = 0.60             # W/(m·K), thermal conductivity of the liquid
rho = 1000.0         # kg/m^3
cp = 4180.0          # J/(kg·K)
L_f = 334e3          # J/kg, latent heat of fusion
T_m = 273.15         # K, melting temperature
T_s = T_m + 10.0     # K, surface temperature from t=0 on
slab = 0.040         # m, modelled thickness (back face insulated)
cells = 200          # finite-volume cells across the slab
probe_times = (600.0, 1200.0, 1800.0, 2700.0, 3600.0)   # s

alpha = k / (rho * cp)
stefan = cp * (T_s - T_m) / L_f
print(f"alpha = {alpha:.3e} m^2/s,  Stefan number St = {stefan:.4f}")

# Transcendental equation for lambda, solved by Newton's method.
lam = 0.5 * sqrt(stefan)          # small-St estimate as the starting value
for _ in range(20):
    f = lam * np.exp(lam**2) * erf(lam) - stefan / sqrt(pi)
    df = np.exp(lam**2) * erf(lam) * (1.0 + 2.0 * lam**2) + 2.0 * lam / sqrt(pi)
    lam -= f / df
print(f"lambda = {lam:.6f}  (residual {lam*np.exp(lam**2)*erf(lam) - stefan/sqrt(pi):.1e})")


def front_reference(t):
    return 2.0 * lam * np.sqrt(alpha * t)


def temperature_reference(x, t):
    melted = x <= front_reference(t)
    profile = T_s - (T_s - T_m) * erf(x / (2.0 * np.sqrt(alpha * t))) / erf(lam)
    return np.where(melted, profile, T_m)


# ## Enthalpy-method solution
#
# Boundary conditions: fixed temperature $T_s$ at the heated face `x_min`,
# insulated (zero heat flux) back face `x_max`. The slab starts as solid at
# $T_m$ (liquid fraction 0). The explicit, energy-conserving time stepping
# picks its own stable step below the requested one; each stored time is
# computed in one run from $t=0$.

edges = np.linspace(0.0, slab, cells + 1)
runs = {}
for t_end in probe_times:
    solver = StefanEnthalpyProvider(
        face_axes=(edges,),
        density=rho, heat_capacity=cp, conductivity=k,
        melting_temperature=T_m, latent_heat=L_f,
        boundary_temperatures={"x_min": T_s},     # heated face held at T_s
        # x_max omitted on purpose: the provider treats it as insulated (zero flux)
        final_time=t_end, time_step=10.0,
    )
    runs[t_end] = solver.run(initial_temperature=T_m, initial_liquid_fraction=0.0)
x_cells = runs[probe_times[0]].axes[0]
print("Cells:", cells, " cell size:", f"{slab/cells*1e3:.2f} mm,", "time steps in the longest run:", runs[probe_times[-1]].step_count)


# ## Comparison with the Neumann solution

dT = T_s - T_m
print("time     front OneSim   front Neumann   rel. error   melted volume err.   max|dT|/dT   mean|dT|/dT")
front_errors, volume_errors, max_profile_errors, mean_profile_errors = [], [], [], []
for t, result in runs.items():
    assert result.front_points.shape == (1, 1), "exactly one planar front expected"
    s_num = float(result.front_points[0, 0])
    s_ref = front_reference(t)
    # The grid has unit transverse measure, so the liquid volume is the melted depth.
    rel_volume = abs(result.liquid_volume - s_ref) / s_ref
    profile_error = np.abs(result.temperature - temperature_reference(x_cells, t)) / dT
    front_errors.append(abs(s_num - s_ref) / s_ref)
    volume_errors.append(rel_volume)
    max_profile_errors.append(float(profile_error.max()))
    mean_profile_errors.append(float(profile_error.mean()))
    print(f"{t:6.0f} s   {s_num*1e3:7.3f} mm    {s_ref*1e3:7.3f} mm    {100*front_errors[-1]:6.2f} %"
          f"        {100*rel_volume:6.2f} %          {100*max_profile_errors[-1]:5.2f} %      {100*mean_profile_errors[-1]:5.2f} %")
    # Energy conservation of the enthalpy scheme: stored energy change equals the boundary heat input.
    scale = max(abs(result.initial_energy), abs(result.final_energy), abs(result.boundary_energy))
    assert abs(result.energy_balance_residual) < 1e-9 * scale, "energy budget not closed"
    assert np.all((result.liquid_fraction >= 0.0) & (result.liquid_fraction <= 1.0))
    assert result.liquid_fraction[-1] == 0.0 and abs(result.temperature[-1] - T_m) < 1e-9, \
        "the front must stay far from the insulated back face"

assert max(front_errors) < 0.03, "Front position deviates more than 3 % from 2 lambda sqrt(alpha t)."
assert max(volume_errors) < 0.03, "Melted volume deviates more than 3 % from the Neumann solution."
assert max(max_profile_errors) < 0.05, "Temperature profile deviates more than 5 % of (Ts-Tm) in some cell."
assert max(mean_profile_errors) < 0.01, "Mean temperature deviation above 1 % of (Ts-Tm)."
# Similarity check: s(t)/sqrt(t) must be constant in time.
ratios = np.array([float(r.front_points[0, 0]) / sqrt(t) for t, r in runs.items()])
assert np.ptp(ratios) / ratios.mean() < 0.03, "The front does not follow the sqrt(t) similarity law."
print(f"s(t)/sqrt(t): OneSimulation {ratios.mean():.4e} m/s^0.5 (spread {100*np.ptp(ratios)/ratios.mean():.2f} %), "
      f"Neumann {2*lam*sqrt(alpha):.4e} m/s^0.5")


# ## Figure

cream = "#fbf5eb"
fig, (ax_front, ax_profile) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax_front, ax_profile):
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)

t_fine = np.linspace(0.0, probe_times[-1], 200)
ax_front.plot(t_fine / 60, front_reference(t_fine) * 1e3, color="#214D43", lw=1.8, label="Neumann  $2\\lambda\\sqrt{\\alpha t}$")
ax_front.plot(np.array(probe_times) / 60, [float(r.front_points[0, 0]) * 1e3 for r in runs.values()],
              "o", color="#A85539", ms=6, label="OneSimulation enthalpy method")
ax_front.set(xlabel="Time / min", ylabel="Front position s / mm", title="Melting front")
ax_front.legend(loc="lower right")

colors = plt.cm.copper(np.linspace(0.25, 0.85, len(probe_times)))
x_plot = np.linspace(0.0, 0.5 * slab, 400)
for i, ((t, result), c) in enumerate(zip(runs.items(), colors)):
    ax_profile.plot(x_plot * 1e3, temperature_reference(x_plot, t) - 273.15, color=c, lw=1.6)
    shown = x_cells <= 0.5 * slab
    ax_profile.plot(x_cells[shown] * 1e3, result.temperature[shown] - 273.15, "o", color=c, ms=2.5)
    ax_profile.text(0.97, 0.93 - 0.075 * i, f"t = {t/60:.0f} min", color=c, fontsize=9,
                    ha="right", va="top", transform=ax_profile.transAxes)
ax_profile.set(xlabel="Position x / mm", ylabel="Temperature / °C",
               title="Liquid layer — lines: Neumann, dots: OneSimulation")
plt.show()


# ## Going further
#
# Halve the cell size and watch the front error shrink. Set the surface below
# $T_m$ and start from liquid (`initial_liquid_fraction=1.0`) for the
# solidification mirror image. A slab that starts *below* $T_m$ leads to the
# two-phase Stefan problem, whose $\lambda$ equation also contains the solid
# properties and the initial undercooling.
