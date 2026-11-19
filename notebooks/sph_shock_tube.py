"""OneSimulation example: the Sod shock tube solved with Lagrangian SPH particles.
Requires the installed, activated OneSimulation runtime.

A diaphragm separates dense, pressurised gas from thin gas. After it is removed, a rarefaction, a
contact discontinuity and a shock emerge. The particle simulation is compared with the exact
solution of the Riemann problem.
"""

# # The Sod shock tube with smoothed-particle hydrodynamics
#
# A tube filled with an ideal gas ($\gamma=1.4$) is divided by a diaphragm. Both halves are at rest:
#
# $$(\rho,u,p)_L=(1,\,0,\,1),\qquad (\rho,u,p)_R=(0.125,\,0,\,0.1).$$
#
# When the diaphragm is removed, three waves emerge: a rarefaction fan running left into the
# dense gas, a contact discontinuity and a shock running right into the thin gas.
#
# The gas is represented by moving particles with a cubic-spline kernel and Monaghan's artificial
# viscosity for the shock. Mass, momentum and total energy are conserved by construction.
#
# **Reference.** The exact solution of the Riemann problem: the pressure $p_*$ between the waves
# solves $f_L(p_*)+f_R(p_*)=0$ (shock relation on the right, isentropic relation on the left), and
# the self-similar wave pattern follows from it (Toro, *Riemann Solvers and Numerical Methods for
# Fluid Dynamics*).

import numpy as np
from onesim.physics.hydrodynamics import HydrodynamicPublicApi
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

gamma, diaphragm, end_time = 1.4, 0.5, 0.2
left, right = (1.0, 1.0), (0.125, 0.1)


def exact_solution(x):
    """Density, velocity and pressure of the exact Riemann solution at the end time."""
    (rho_l, p_l), (rho_r, p_r) = left, right
    a_l, a_r, g = (
        np.sqrt(gamma * p_l / rho_l),
        np.sqrt(gamma * p_r / rho_r),
        (gamma - 1) / (gamma + 1),
    )
    fan = (
        lambda p: 2 * a_l / (gamma - 1) * ((p / p_l) ** ((gamma - 1) / (2 * gamma)) - 1)
    )
    shock = lambda p: (p - p_r) * np.sqrt(2 / ((gamma + 1) * rho_r) / (p + g * p_r))
    p_star = brentq(lambda p: fan(p) + shock(p), 1e-6, 10.0, xtol=1e-14)
    u_star = 0.5 * (shock(p_star) - fan(p_star))
    rho_star_l = rho_l * (p_star / p_l) ** (1 / gamma)
    rho_star_r = rho_r * (p_star / p_r + g) / (g * p_star / p_r + 1)
    shock_speed = a_r * np.sqrt(
        (gamma + 1) / (2 * gamma) * p_star / p_r + (gamma - 1) / (2 * gamma)
    )
    xi = (x - diaphragm) / end_time
    u_fan = 2 / (gamma + 1) * (a_l + xi)
    a_fan = a_l - 0.5 * (gamma - 1) * u_fan
    regions = [
        xi <= -a_l,
        xi < u_star - np.sqrt(gamma * p_star / rho_star_l),
        xi < u_star,
        xi < shock_speed,
    ]
    return (
        np.select(
            regions,
            [rho_l, rho_l * (a_fan / a_l) ** (2 / (gamma - 1)), rho_star_l, rho_star_r],
            rho_r,
        ),
        np.select(regions, [0.0, u_fan, u_star, u_star], 0.0),
        np.select(
            regions,
            [p_l, p_l * (a_fan / a_l) ** (2 * gamma / (gamma - 1)), p_star, p_star],
            p_r,
        ),
    )


# ## Simulation
#
# 1600 equally spaced particles carry the mass of their cell; the smoothing length is 2.5
# particle spacings.

count, lower, upper = 1600, -0.5, 1.5
spacing = (upper - lower) / count
start = lower + (np.arange(count) + 0.5) * spacing
density, pressure = np.where(start < diaphragm, left[0], right[0]), np.where(
    start < diaphragm, left[1], right[1]
)
gas = HydrodynamicPublicApi.create_provider(
    "sph_hydrodynamics",
    positions=start,
    masses=density * spacing,
    velocities=np.zeros(count),
    specific_internal_energy=pressure / ((gamma - 1) * density),
    smoothing_length=2.5 * spacing,
    gamma=gamma,
    viscosity_alpha=1.0,
    viscosity_beta=2.0,
    periodic_bounds=((lower, upper),),
    final_time=end_time,
    time_step=1.0e-3,
    cfl=0.25,
)
result = gas.run()


# ## Results
#
# Particle values between $x=0$ and $x=1$ against the exact solution.

order = np.argsort(result.positions[:, 0])
x = result.positions[order, 0]
inside = (x >= 0) & (x <= 1)
x = x[inside]
particles = {
    "density": result.density[order][inside],
    "velocity": result.velocities[order, 0][inside],
    "pressure": result.pressure[order][inside],
}
exact = dict(zip(particles, exact_solution(x)))
for name in particles:
    print(
        f"{name:9s} mean deviation from the exact solution: "
        f"{100 * np.abs(particles[name] - exact[name]).sum() / np.abs(exact[name]).sum():.2f} %"
    )
print(
    f"mass {result.initial_balance.mass:.6f} -> {result.final_balance.mass:.6f},  "
    f"total energy {result.initial_balance.total_energy:.6f} -> {result.final_balance.total_energy:.6f}"
)


# ## Figure

fine = np.linspace(0, 1, 1200)
fig = plots.figure(1, 3, size=(15, 4.5))
for panel, name, reference in zip(fig, particles, exact_solution(fine)):
    plots.line(
        points={"SPH particles": (x, particles[name])},
        reference={"exact Riemann solution": (fine, reference)},
        xlabel="position",
        ylabel=name,
        title=f"{name.capitalize()} at t = {end_time}",
        ax=panel,
    )
plots.show()


# ## Going further
#
# Halve the particle spacing and the waves sharpen; lower the artificial viscosity and
# oscillations appear behind the shock. A stronger pressure ratio turns the problem into a blast
# wave.
