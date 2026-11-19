"""OneSimulation example: hydrogen iodide decomposition in a closed batch reactor.
Requires the installed, activated OneSimulation runtime.

The reversible gas reaction 2 HI <=> H2 + I2 runs in a rigid vessel, once at fixed temperature and
once without heat exchange. The isothermal runs are compared with the closed-form solution of the
reversible second-order rate law, the adiabatic run with the energy balance and the equilibrium.
"""

# # Hydrogen iodide batch reactor, isothermal and adiabatic
#
# Hydrogen iodide decomposes reversibly into hydrogen and iodine vapour,
#
# $$2\,\mathrm{HI} \;\rightleftharpoons\; \mathrm{H_2} + \mathrm{I_2},\qquad
#   r = k_f(T)\Big(c_\mathrm{HI}^2 - \frac{c_\mathrm{H_2}c_\mathrm{I_2}}{K(T)}\Big),\qquad
#   k_f = A\,e^{-E_a/(R T)} .$$
#
# The gas fills a rigid, perfectly mixed vessel (0D). Only the forward rate constant is declared;
# the reverse rate follows from detailed balance with the equilibrium constant $K(T)$, which
# OneSimulation computes from the standard enthalpies, entropies and heat capacities of the three
# species. Because the mole number does not change, $K_c=K_p=K$:
#
# $$K(T)=\exp\!\Big(-\frac{\Delta_r G^\circ(T)}{RT}\Big),\qquad
#   \Delta_r G^\circ = \Delta_r H^\circ(T) - T\,\Delta_r S^\circ(T) .$$
#
# Two operating modes are computed:
#
# | mode | temperature | energy balance |
# |---|---|---|
# | isothermal | held at 700 ... 800 K | none (thermostat) |
# | adiabatic | free, starts at 750 K | $\sum_i c_i c_{v,i}\,\dot T = -\Delta_r U\,r$, ideal gas at constant volume |
#
# **Reference.** Arrhenius parameters of the decomposition $A = 9.17\cdot10^{10}$ dm³/(mol s),
# $E_a = 186$ kJ/mol (Atkins & de Paula, *Physical Chemistry*, table of Arrhenius parameters);
# standard thermochemical data of HI, H₂ and I₂(g) from the NIST-JANAF tables (Chase, 1998).
# At fixed temperature the extent $x=c_\mathrm{H_2}$ obeys $\dot x = a\,(x-x_1)(x-x_2)$ with
# $a=4k_f-k_f/K$ and $x_{1,2}=c_0/(2\pm K^{-1/2})$, whose solution is
#
# $$x(t)=\frac{x_1x_2\,(1-e^{a(x_1-x_2)t})}{x_2-x_1\,e^{a(x_1-x_2)t}} .$$
#
# For the adiabatic vessel, conservation of internal energy gives the temperature as a function
# of the extent, $T(x)=T_0-x\,\Delta_rH^\circ(T_0)/(c_0c_{v,\mathrm{HI}}+x\,\Delta c_p)$, and the
# final state is where $K(T(x))\,(c_0-2x)^2=x^2$. The classic measurement of the synthesis
# equilibrium by Taylor & Crist (*J. Am. Chem. Soc.* 63, 1941) gives $1/K = 54.4$ at 698.6 K.

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.chemistry import (
    ReactionIR,
    ReactionKinetics,
    ReactionNetworkIR,
    SpeciesIR,
)
from onesim.physics.multiphysics import AdiabaticReaction
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()


# ## Species and reaction
#
# Each species carries its standard enthalpy of formation, standard entropy and a constant heat
# capacity at 298.15 K. The reaction is declared reversible, so its reverse branch is built from
# this thermochemistry.

R = 8.314462618
thermo = {
    "HI": (26.50e3, 206.59, 29.16, {"H": 1, "I": 1}, 2.124e-25),
    "H2": (0.0, 130.68, 28.84, {"H": 2}, 3.347e-27),
    "I2": (62.42e3, 260.69, 36.90, {"I": 2}, 4.215e-25),
}
network = ReactionNetworkIR(
    "hydrogen_iodide",
    "1",
    species=tuple(
        SpeciesIR(
            name,
            0,
            mass,
            composition=atoms,
            thermo={
                "model": "constant-cp",
                "reference_temperature": 298.15,
                "reference_enthalpy": h,
                "reference_entropy": s,
                "heat_capacity": cp,
            },
        )
        for name, (h, s, cp, atoms, mass) in thermo.items()
    ),
    reactions=(
        ReactionIR(
            "decomposition",
            {"HI": 2},
            {"H2": 1, "I2": 1},
            {
                "kind": "arrhenius",
                "pre_exponential": 9.17e7,
                "activation_energy": 186e3,
                "boltzmann_constant": R,
            },
            reversible=True,
        ),
    ),
    strict_balance=True,
)

pressure = 10e5
dH, dS, dcp = (
    62.42e3 - 2 * 26.50e3,
    130.68 + 260.69 - 2 * 206.59,
    28.84 + 36.90 - 2 * 29.16,
)
K = lambda T: np.exp(
    -(dH + dcp * (T - 298.15) - T * (dS + dcp * np.log(T / 298.15))) / (R * T)
)
print(
    f"reaction enthalpy {dH / 1e3:.2f} kJ/mol at 298 K, heat capacity change {dcp:.2f} J/(mol K)"
)


# ## Isothermal runs
#
# Pure HI at 10 bar is charged at four temperatures. Each run covers fifteen relaxation times
# $\tau = 1/\lvert a\,(x_1-x_2)\rvert$ of the closed form in 600 classical Runge-Kutta steps.

domain = LumpedDomain()
temperatures = (700.0, 725.0, 750.0, 800.0)
models = {
    T: ReactionKinetics(domain=domain.system, network=network, temperature=T)
    for T in temperatures
}
c0 = {T: pressure / (R * T) for T in temperatures}
kf = {T: 9.17e7 * np.exp(-186e3 / (R * T)) for T in temperatures}
tau = {
    T: 1
    / (
        kf[T]
        * (1 / K(T) - 4)
        * (c0[T] / (2 + K(T) ** -0.5) - c0[T] / (2 - K(T) ** -0.5))
    )
    for T in temperatures
}
runs = {
    T: Simulation(
        problem=Problem(
            name=f"HI at {T:.0f} K",
            geometry=domain,
            physics=(model,),
            initial_conditions=(
                InitialValue(model.concentration("HI"), c0[T]),
                InitialValue(model.concentration("H2"), 0.0),
                InitialValue(model.concentration("I2"), 0.0),
            ),
        ),
        mesh=LumpedMesh(domain),
        numerics=Numerics(
            spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()
        ),
    ).run(start_time=0.0, end_time=15 * tau[T], time_step=15 * tau[T] / 600)
    for T, model in models.items()
}


# ## Comparison with the closed form
#
# The hydrogen history of every run is compared with the exact extent $x(t)$, and the final
# composition with the equilibrium constant.

x1 = {T: c0[T] / (2 + K(T) ** -0.5) for T in temperatures}
x2 = {T: c0[T] / (2 - K(T) ** -0.5) for T in temperatures}
exact = (
    lambda T, t: x1[T]
    * x2[T]
    * (1 - np.exp(kf[T] * (4 - 1 / K(T)) * (x1[T] - x2[T]) * t))
    / (x2[T] - x1[T] * np.exp(kf[T] * (4 - 1 / K(T)) * (x1[T] - x2[T]) * t))
)
hydrogen = {
    T: run.field(models[T].concentration("H2")).time_series() for T, run in runs.items()
}
for T, (t, x) in hydrogen.items():
    final_HI = runs[T].field(models[T].concentration("HI")).values[0]
    print(
        f"T = {T:.0f} K: largest deviation from x(t) {np.max(np.abs(x - exact(T, t))) / x1[T]:.1e} of the equilibrium extent, "
        f"equilibrium conversion {2 * x[-1] / c0[T]:.4f}, 1/K from the final state {final_HI**2 / x[-1]**2:.2f} (thermochemistry {1 / K(T):.2f})"
    )
print(f"1/K at 698.6 K from the species data: {1 / K(698.6):.1f}  (measured: 54.4)")


# ## Adiabatic run
#
# Without the thermostat the reactor temperature becomes a state. With
# `volumetric_heat_capacity=None` the reactor treats the gas as ideal: its heat capacity at
# constant volume and the reaction energy come from the same species data. The endothermic
# decomposition cools the gas, which slows it down and shifts its equilibrium.

T0 = 750.0
adiabatic = AdiabaticReaction(
    domain=domain.system, network=network, volumetric_heat_capacity=None
)
adiabatic_run = Simulation(
    problem=Problem(
        name="adiabatic HI",
        geometry=domain,
        physics=(adiabatic,),
        initial_conditions=(
            InitialValue(adiabatic.concentration("HI"), c0[T0]),
            InitialValue(adiabatic.concentration("H2"), 0.0),
            InitialValue(adiabatic.concentration("I2"), 0.0),
            InitialValue(adiabatic.temperature, T0),
        ),
    ),
    mesh=LumpedMesh(domain),
    numerics=Numerics(
        spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()
    ),
).run(start_time=0.0, end_time=120 * tau[T0], time_step=tau[T0] / 25)

t_ad, T_ad = adiabatic_run.field(adiabatic.temperature).time_series()
_, x_ad = adiabatic_run.field(adiabatic.concentration("H2")).time_series()
temperature_of = lambda x: T0 - x * (dH + dcp * (T0 - 298.15)) / (
    c0[T0] * (29.16 - R) + x * dcp
)
x_end = brentq(
    lambda x: K(temperature_of(x)) * (c0[T0] - 2 * x) ** 2 - x**2, 1e-9, x1[T0]
)
print(
    f"energy balance along the run: largest deviation of T from T(x) {np.max(np.abs(T_ad - temperature_of(x_ad))):.2e} K"
)
print(
    f"final state: simulation T = {T_ad[-1]:.2f} K, x = {x_ad[-1]:.4f} mol/m^3;  "
    f"energy balance + equilibrium T = {temperature_of(x_end):.2f} K, x = {x_end:.4f} mol/m^3"
)
print(
    f"equilibrium conversion: adiabatic {2 * x_ad[-1] / c0[T0]:.4f}, isothermal at 750 K {2 * x1[T0] / c0[T0]:.4f}"
)


# ## Figure
#
# Hydrogen formation at four temperatures against the exact solution, the adiabatic run against
# the isothermal one at the same start temperature, and the equilibrium constant over temperature.

t_iso, x_iso = hydrogen[T0]
T_line = np.linspace(650, 850, 200)
fig = plots.figure(1, 3, size=(18, 5))
plots.line(
    points={
        f"{T:.0f} K": (t[::12] / tau[T], x[::12] / c0[T])
        for T, (t, x) in hydrogen.items()
    },
    reference={
        f"{T:.0f} K": (t / tau[T], exact(T, t) / c0[T])
        for T, (t, x) in hydrogen.items()
    },
    xlabel="time / relaxation time",
    ylabel="H2 mole fraction",
    title="Isothermal runs, dashed: exact",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    y={
        "H2 mole fraction, adiabatic": (t_ad / 60, x_ad / c0[T0]),
        "H2 mole fraction, isothermal 750 K": (t_iso / 60, x_iso / c0[T0]),
    },
    right={"temperature, adiabatic": (t_ad / 60, T_ad)},
    hlines={"equilibrium, adiabatic": x_end / c0[T0]},
    xlimits=(0, 60 * tau[T0] / 60),
    xlabel="time [min]",
    ylabel="H2 mole fraction",
    right_label="temperature [K]",
    title="Adiabatic run from 750 K",
    legend="center right",
    ax=fig[1],
)
plots.line(
    y={"from species thermochemistry": (T_line, 1 / K(T_line))},
    points={
        "final state of the runs": (
            list(temperatures) + [T_ad[-1]],
            [
                runs[T].field(models[T].concentration("HI")).values[0] ** 2
                / hydrogen[T][1][-1] ** 2
                for T in temperatures
            ]
            + [(c0[T0] - 2 * x_ad[-1]) ** 2 / x_ad[-1] ** 2],
        ),
        "Taylor & Crist (1941)": ([698.6], [54.4]),
    },
    xlabel="temperature [K]",
    ylabel="1/K = c_HI^2 / (c_H2 c_I2)",
    title="Equilibrium of H2 + I2 = 2 HI",
    ax=fig[2],
)
plots.show()


# ## Going further
#
# Raise the pressure: the equilibrium does not move, because the mole number is unchanged, but
# the reaction gets faster with $c_0^2$. Replace the constant heat capacities by NASA polynomials
# (`"model": "NASA7"`) for a wide temperature range, or start from H₂ and I₂ to run the
# synthesis direction towards the same equilibrium.
