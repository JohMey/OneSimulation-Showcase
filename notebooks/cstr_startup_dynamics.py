"""OneSimulation example: startup of a non-isothermal continuous stirred tank reactor.
Requires the installed, activated OneSimulation runtime.

A stirred tank reactor that makes propylene glycol is started from three different initial states.
The temperature overshoot and the steady state are compared with the published solution and with
the independently solved steady balance, using two adaptive time integrators.
"""

# # Startup of a continuous stirred tank reactor
#
# Propylene glycol is produced from propylene oxide (PO) and water in a perfectly mixed tank
# reactor with a cooling coil; methanol is fed as an inert co-solvent:
#
# $$\mathrm{C_3H_6O + H_2O \longrightarrow C_3H_8O_2},\qquad r = k(T)\,c_\mathrm{PO},\qquad k(T)=A\,e^{-E/(R_gT)} .$$
#
# This is the startup problem of Fogler's *Elements of Chemical Reaction Engineering* (Example
# 9-4). The reactor is one well-mixed volume $V$ with the feed $v_f$, so every state depends on
# time only:
#
# $$V\frac{dc_i}{dt} = v_f\,c_{f,i} - v\,c_i + \nu_i\,r\,V ,\qquad
#   V\Big(\sum_i c_iC_{p,i}\Big)\frac{dT}{dt}
#   = -V\,\Delta H_r\,r + Q_\mathrm{coil}(T) + v_f\sum_i c_{f,i}C_{p,i}\,(T_f-T) .$$
#
# **Reference.** The published solution reports a steady state of about 472 mol/m³ of propylene
# oxide at 337 K and, for the startup with 1400 mol/m³ at 340 K, a temperature overshoot to about
# 362 K, above the safety limit of 355 K. The steady state also follows from the algebraic
# balances, solved here independently of the time integration.

import numpy as np
from onesim.numerics.time import AdaptiveBDF, AdaptiveRungeKutta45, EventDefinition
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

volume, feed_flow, feed_temperature, coolant_temperature = 1.89, 3.47e-3, 297.0, 289.0
feed = np.array([2903.0, 36291.0, 0.0, 3629.0])
heat_capacity = np.array([146.5, 75.4, 192.6, 81.6])
stoichiometry = np.array([-1.0, -1.0, 1.0, 0.0])
contraction = stoichiometry @ (
    np.array([58.08, 18.015, 76.095, 32.042])
    * 1e-3
    / np.array([830.0, 1000.0, 1036.0, 792.0])
)
coil = 126.0 * 75.4 * (1 - np.exp(-8441.0 / (126.0 * 75.4)))
rate_constant = lambda T: 4.71e9 * np.exp(-75.358e3 / (8.314462618 * T))
reaction_enthalpy = lambda T: stoichiometry @ np.array(
    [-153.5e3, -286.1e3, -525.6e3, -238.6]
) + stoichiometry @ heat_capacity * (T - 293.0)


def reactor(time, state):
    """Rates of change of the four concentrations and the temperature."""
    concentration, T = state[:4], state[4]
    rate = rate_constant(T) * concentration[0]
    outflow = feed_flow + volume * contraction * rate
    heat = (
        -volume * reaction_enthalpy(T) * rate
        + coil * (coolant_temperature - T)
        + feed_flow * (feed @ heat_capacity) * (feed_temperature - T)
    )
    return np.r_[
        (feed_flow * feed - outflow * concentration) / volume + stoichiometry * rate,
        heat / (volume * (concentration @ heat_capacity)),
    ]


def steady_state():
    """Steady propylene-oxide concentration and temperature from the algebraic balances."""

    def oxide(T):
        roots = np.roots(
            [
                volume * contraction * rate_constant(T),
                feed_flow + rate_constant(T) * volume,
                -feed_flow * feed[0],
            ]
        )
        return roots[(roots.real >= 0) & (roots.real <= feed[0])].real[0]

    balance = lambda T: (
        -volume * reaction_enthalpy(T) * rate_constant(T) * oxide(T)
        + coil * (coolant_temperature - T)
        + feed_flow * (feed @ heat_capacity) * (feed_temperature - T)
    )
    T = brentq(balance, 300.0, 400.0, xtol=1e-12)
    return oxide(T), T


# ## Startups
#
# Each startup begins with water and a given amount of propylene oxide at a given temperature and
# runs for four hours. An event marks the temperature maximum.

startups = {
    "1400 mol/m³ PO, 340 K": (1400.0, 340.0),
    "no PO, 340 K": (0.0, 340.0),
    "no PO, 297 K": (0.0, 297.0),
}
peak = EventDefinition(
    "temperature maximum", lambda t, state: reactor(t, state)[4], direction=-1
)
integrator = AdaptiveRungeKutta45(
    relative_tolerance=1e-9, absolute_tolerance=1e-9, maximum_step=5.0
)
runs = {
    label: integrator.integrate(
        reactor,
        np.array([oxide, 55273.0, 0.0, 0.0, T]),
        0.0,
        4 * 3600.0,
        initial_step=0.1,
        events=(peak,),
    )
    for label, (oxide, T) in startups.items()
}

oxide, temperature = steady_state()
base = runs["1400 mol/m³ PO, 340 K"]
print(
    f"steady state: simulation {base.states[-1, 0]:.1f} mol/m³ PO at {base.states[-1, 4]:.2f} K, "
    f"algebraic balance {oxide:.1f} mol/m³ at {temperature:.2f} K, published about 472 mol/m³ at 337 K"
)
for label, run in runs.items():
    hottest = run.states[:, 4].max()
    print(
        f"{label:22s} highest temperature {hottest:.1f} K  ({'above' if hottest > 355 else 'below'} the 355 K limit)"
    )
print(
    f"overshoot of the first startup: {base.events[0].state[4]:.1f} K after {base.events[0].time / 60:.2f} min, published about 362 K"
)


# ## Two time integrators
#
# The first startup again with the adaptive Runge-Kutta method and with an adaptive BDF method at
# several tolerances. The error is the deviation of the peak temperature from the run above.

methods = {
    "Dormand-Prince (adaptive)": [
        AdaptiveRungeKutta45(relative_tolerance=tol, absolute_tolerance=tol)
        for tol in (1e-4, 1e-6, 1e-8)
    ],
    "BDF2 (adaptive)": [
        AdaptiveBDF(order=2, relative_tolerance=tol, absolute_tolerance=tol)
        for tol in (1e-3, 1e-5, 1e-7)
    ],
}
work = {}
for name, solvers in methods.items():
    results = [
        solver.integrate(
            reactor, base.states[0], 0.0, 4 * 3600.0, initial_step=0.1, events=(peak,)
        )
        for solver in solvers
    ]
    work[name] = (
        [result.accepted_steps for result in results],
        [
            abs(result.events[0].state[4] - base.events[0].state[4])
            for result in results
        ],
    )
    print(
        f"{name:26s} steps {work[name][0]}   error of the peak temperature [K] "
        + ", ".join(f"{e:.1e}" for e in work[name][1])
    )


# ## Figure
#
# The reactor temperature for the three startups, the path of the first startup in the plane of
# concentration and temperature, and the accuracy of the two integrators against their work.

fig = plots.figure(1, 3, size=(17, 5))
plots.line(
    y={label: (run.times / 60, run.states[:, 4]) for label, run in runs.items()},
    hlines={"safety limit": 355.0, "steady state": temperature},
    xlimits=(0, 120),
    xlabel="time [min]",
    ylabel="reactor temperature [K]",
    title="Three startups",
    ax=fig[0],
)
plots.line(
    y={label: (run.states[:, 0], run.states[:, 4]) for label, run in runs.items()},
    points={"steady state": ([oxide], [temperature])},
    xlabel="propylene oxide [mol/m³]",
    ylabel="temperature [K]",
    title="Path to the steady state",
    ax=fig[1],
)
plots.line(
    y=work,
    markers=True,
    scale="loglog",
    xlabel="time steps",
    ylabel="error of the peak temperature [K]",
    title="Accuracy against work",
    ax=fig[2],
)
plots.show()


# ## Going further
#
# Lower the coolant flow and the steady state loses its stability: the reactor then oscillates
# for ever. Raise the feed temperature to find the ignition point, or add a controller that
# throttles the feed when the temperature approaches the limit.
