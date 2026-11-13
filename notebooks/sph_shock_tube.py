"""OneSimulation example: the Sod shock tube solved with Lagrangian SPH particles.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the exact solution of the
Riemann problem: the plateau states behind the shock and behind the contact
discontinuity, and the full density, velocity and pressure profiles at t = 0.2.
"""


# # The Sod shock tube with smoothed-particle hydrodynamics
#
# A tube filled with an ideal gas ($\gamma=1.4$) is divided by a diaphragm. On
# the left the gas is dense and pressurised, on the right it is thin and at low
# pressure; both halves are at rest:
#
# $$(\rho,u,p)_L=(1,\,0,\,1),\qquad (\rho,u,p)_R=(0.125,\,0,\,0.1).$$
#
# When the diaphragm is removed, three waves emerge: a rarefaction fan running
# left into the dense gas, a contact discontinuity and a shock running right
# into the thin gas. Between the waves the gas settles into two plateaus that
# share the same pressure $p_*$ and velocity $u_*$ but differ in density.
#
# OneSimulation solves the problem with its **Lagrangian SPH provider**: the
# gas is represented by moving particles with a cubic-spline kernel, an ideal
# gas equation of state and Monaghan's artificial viscosity for the shock. Mass,
# momentum and total energy are conserved pairwise by construction.
#
# **Reference.** The exact solution of the Riemann problem is obtained by
# solving the pressure equation $f_L(p_*)+f_R(p_*)+u_R-u_L=0$ (shock relation on
# the right, isentropic rarefaction relation on the left) for $p_*$ and then
# assembling the self-similar wave pattern. For the Sod data this gives
# $p_*\approx0.3031$, $u_*\approx0.9275$, $\rho_{*L}\approx0.4263$ and the
# post-shock density $\rho_{*R}\approx0.2656$.

from importlib.metadata import version

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
from onesim.physics.hydrodynamics.incompressible.execution.composition.service import (
    HydrodynamicPublicApi,
)
from scipy.optimize import brentq

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The diaphragm sits at $x=0.5$ and the solution is read at $t=0.2$, when the
# shock has reached $x\approx0.85$ and the head of the rarefaction
# $x\approx0.26$. The particles live in a periodic box $[-0.5,\,1.5]$: the
# periodic seam at $x=\pm0.5$ from the ends is itself a (mirrored) Riemann
# problem, but its waves stay outside $[0,1]$ until well after $t=0.2$. Equally
# spaced particles carry the local mass $\rho\,\Delta x$; the smoothing length
# is $2.5\,\Delta x$, so each particle interacts with ten neighbours.

gamma = 1.4
rho_L, u_L, p_L = 1.0, 0.0, 1.0
rho_R, u_R, p_R = 0.125, 0.0, 0.1
x_diaphragm = 0.5
end_time = 0.2
x_lo, x_hi = -0.5, 1.5                 # periodic box
PARTICLES = 1600
dx = (x_hi - x_lo) / PARTICLES
smoothing_length = 2.5 * dx
print(f"{PARTICLES} particles, dx = {dx:.5f}, h = {smoothing_length:.5f}")


# ## Exact Riemann solution
#
# Standard exact solver: Newton-free root finding for the star pressure,
# then the self-similar sampling of the four states and the rarefaction fan.

def _sound(rho, p):
    return np.sqrt(gamma * p / rho)


def _f(p, rho_k, p_k):
    a_k = _sound(rho_k, p_k)
    if p > p_k:                                           # shock
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        return (p - p_k) * np.sqrt(A / (p + B))
    return 2.0 * a_k / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


p_star = brentq(lambda p: _f(p, rho_L, p_L) + _f(p, rho_R, p_R) + u_R - u_L, 1e-6, 10.0, xtol=1e-14)
u_star = 0.5 * (u_L + u_R) + 0.5 * (_f(p_star, rho_R, p_R) - _f(p_star, rho_L, p_L))
rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)                         # isentropic
ratio = p_star / p_R
rho_star_R = rho_R * (ratio + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * ratio + 1.0)
a_L, a_R = _sound(rho_L, p_L), _sound(rho_R, p_R)
a_star_L = _sound(rho_star_L, p_star)
shock_speed = u_R + a_R * np.sqrt((gamma + 1) / (2 * gamma) * ratio + (gamma - 1) / (2 * gamma))
head_speed, tail_speed = u_L - a_L, u_star - a_star_L
print(f"Exact star state: p* = {p_star:.4f}, u* = {u_star:.4f}, rho*L = {rho_star_L:.4f}, rho*R = {rho_star_R:.4f}")
print(f"Wave speeds: rarefaction head {head_speed:.4f}, tail {tail_speed:.4f}, contact {u_star:.4f}, shock {shock_speed:.4f}")


def exact_solution(x, t):
    xi = (x - x_diaphragm) / t
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)
    left = xi <= head_speed
    fan = (xi > head_speed) & (xi < tail_speed)
    star_l = (xi >= tail_speed) & (xi < u_star)
    star_r = (xi >= u_star) & (xi < shock_speed)
    right = xi >= shock_speed
    rho[left], u[left], p[left] = rho_L, u_L, p_L
    u[fan] = 2.0 / (gamma + 1) * (a_L + 0.5 * (gamma - 1) * u_L + xi[fan])
    a_fan = a_L - 0.5 * (gamma - 1) * (u[fan] - u_L)
    rho[fan] = rho_L * (a_fan / a_L) ** (2.0 / (gamma - 1))
    p[fan] = p_L * (a_fan / a_L) ** (2.0 * gamma / (gamma - 1))
    rho[star_l], u[star_l], p[star_l] = rho_star_L, u_star, p_star
    rho[star_r], u[star_r], p[star_r] = rho_star_R, u_star, p_star
    rho[right], u[right], p[right] = rho_R, u_R, p_R
    return rho, u, p


# ## Model and the actual computation
#
# Each particle starts from the piecewise-constant initial data: its mass is
# $\rho\,\Delta x$ and its specific internal energy $e=p/((\gamma-1)\rho)$. The
# boundary condition of the particle cloud is the periodic box. The time step
# is limited internally by the SPH signal-speed CFL condition; the requested
# step only caps it.

x_init = x_lo + (np.arange(PARTICLES) + 0.5) * dx
rho_init = np.where(x_init < x_diaphragm, rho_L, rho_R)
p_init = np.where(x_init < x_diaphragm, p_L, p_R)
gas = HydrodynamicPublicApi.create_provider(
    "sph_hydrodynamics",
    positions=x_init,
    masses=rho_init * dx,
    velocities=np.zeros(PARTICLES),
    specific_internal_energy=p_init / ((gamma - 1.0) * rho_init),
    smoothing_length=smoothing_length,
    gamma=gamma,
    viscosity_alpha=1.0, viscosity_beta=2.0,          # Monaghan artificial viscosity
    periodic_bounds=((x_lo, x_hi),),                  # boundary condition: periodic box
    final_time=end_time,
    time_step=1.0e-3,
    cfl=0.25,
)
result = gas.run()
print(f"Finished at t = {result.time:g} after {result.step_count} steps")

order = np.argsort(result.positions[:, 0])
x = result.positions[order, 0]
rho = result.density[order]
velocity = result.velocities[order, 0]
pressure = result.pressure[order]


# ## Conservation and plateau states
#
# SPH conserves mass, momentum and total energy pairwise; the balances
# reported by the provider must agree to round-off. The plateau values are
# averaged over the interior of each plateau (clear of the smeared waves) and
# compared with the exact star states.

first, last = result.initial_balance, result.final_balance
print(f"Mass {first.mass:.6f} -> {last.mass:.6f},  total energy {first.total_energy:.6f} -> {last.total_energy:.6f}")

x_head = x_diaphragm + head_speed * end_time
x_tail = x_diaphragm + tail_speed * end_time
x_contact = x_diaphragm + u_star * end_time
x_shock = x_diaphragm + shock_speed * end_time
margin = 8 * smoothing_length
star_l_window = (x > x_tail + margin) & (x < x_contact - margin)
star_r_window = (x > x_contact + margin) & (x < x_shock - margin)

plateaus = {
    "p* (left plateau)": (pressure[star_l_window].mean(), p_star),
    "p* (right plateau)": (pressure[star_r_window].mean(), p_star),
    "u* (left plateau)": (velocity[star_l_window].mean(), u_star),
    "u* (right plateau)": (velocity[star_r_window].mean(), u_star),
    "rho*L": (rho[star_l_window].mean(), rho_star_L),
    "rho*R (post-shock)": (rho[star_r_window].mean(), rho_star_R),
}
for name, (sph_value, exact_value) in plateaus.items():
    print(f"{name:20s}: SPH {sph_value:.4f}   exact {exact_value:.4f}   ({100*(sph_value/exact_value-1):+.2f} %)")


# ## Profiles at t = 0.2
#
# The whole profiles are compared on the classical window $[0,1]$ with the
# relative $L^1$ error $\int|q_\mathrm{SPH}-q_\mathrm{exact}|\,dx\,/\int|q_\mathrm{exact}|\,dx$.
# SPH spreads the shock over a few smoothing lengths and leaves the familiar
# small pressure blip at the contact, so a few percent is the honest tolerance.

window = (x >= 0.0) & (x <= 1.0)
rho_exact, u_exact, p_exact = exact_solution(x[window], end_time)


def relative_l1(sph_values, exact_values):
    return float(np.sum(np.abs(sph_values - exact_values)) / np.sum(np.abs(exact_values)))


l1 = {
    "density": relative_l1(rho[window], rho_exact),
    "velocity": relative_l1(velocity[window], u_exact),
    "pressure": relative_l1(pressure[window], p_exact),
}
for name, error in l1.items():
    print(f"Relative L1 error of the {name} profile on [0, 1]: {100*error:.2f} %")

# The shock front must sit at the exact position: locate it as the point where
# the density crosses the mid-value between the post-shock plateau and rho_R
# (searched on [0, 1] only, so the seam waves near the periodic box edges do
# not interfere).
shock_level = 0.5 * (rho_star_R + rho_R)
behind = x[(x > x_contact + margin) & (x <= 1.0) & (rho > shock_level)]
sph_shock_position = float(behind.max())
print(f"Shock position: SPH {sph_shock_position:.4f}, exact {x_shock:.4f}")
print("All checks against the exact Riemann solution passed.")


# ## Figure

x_fine = np.linspace(0.0, 1.0, 1200)
rho_fine, u_fine, p_fine = exact_solution(x_fine, end_time)
cream = "#fbf5eb"
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
fig.patch.set_facecolor(cream)
panels = (
    (axes[0], rho, rho_fine, "Density", r"$\rho$"),
    (axes[1], velocity, u_fine, "Velocity", "$u$"),
    (axes[2], pressure, p_fine, "Pressure", "$p$"),
)
for ax, sph_values, exact_values, title, label in panels:
    ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
    ax.plot(x_fine, exact_values, color="#A85539", lw=1.6, label="exact Riemann solution")
    ax.plot(x[window], sph_values[window], ".", color="#39715e", ms=3, label=f"SPH particles ({PARTICLES} total)")
    ax.set(xlabel="Position x", ylabel=label, xlim=(0, 1), title=f"{title} at t = {end_time:g}")
    ax.legend(frameon=False, fontsize=9)
plt.show()


# ## Going further
#
# Halve `dx` (double `PARTICLES`) and the smeared shock and contact sharpen
# while the plateaus stay put; reduce `viscosity_alpha` and watch post-shock
# oscillations appear, the reason artificial viscosity is needed at all. The
# same provider runs in two and three dimensions with a periodic box per
# direction, so a 2D Riemann problem is the natural next step.
