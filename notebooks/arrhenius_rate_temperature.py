"""OneSimulation example: temperature dependence of a first-order reaction (Arrhenius law).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form exponential decay at
each temperature and against the Arrhenius ratio of the rate coefficients.
"""

# # Arrhenius temperature dependence of a first-order reaction
#
# A species $A$ decays irreversibly into $B$ in a homogeneous, well-mixed (0D)
# isothermal batch reactor,
#
# $$A \xrightarrow{\;k(T)\;} B,\qquad \frac{dC_A}{dt}=-k(T)\,C_A ,$$
#
# but now the rate coefficient is not a fixed number: it follows the
# **Arrhenius law**
#
# $$k(T)=A\,\exp\!\left(-\frac{E_a}{R\,T}\right),$$
#
# with the pre-exponential factor $A$, the activation energy $E_a$ and the gas
# constant $R$. OneSimulation evaluates this law inside the reaction network
# (rate-law kind `arrhenius`) from the reactor temperature, so the same network
# is simply run at several temperatures. Each run integrates the species balance
# on a lumped domain with the classical fourth-order Runge-Kutta scheme.
#
# **Reference.** At a fixed temperature the decay is the exact exponential
#
# $$C_A(t)=C_{A0}\,e^{-k(T)\,t},\qquad C_B(t)=C_{A0}\bigl(1-e^{-k(T)\,t}\bigr),$$
#
# and two temperatures are linked by the Arrhenius ratio
#
# $$\frac{k(T_2)}{k(T_1)}=\exp\!\left[-\frac{E_a}{R}\left(\frac{1}{T_2}-\frac{1}{T_1}\right)\right].$$
#
# Equivalently, $\ln k$ is a straight line in $1/T$ with slope $-E_a/R$ (the
# Arrhenius plot). We check the transient at every temperature against the
# exponential, read the effective rate coefficient off each computed decay, and
# verify the ratio and the slope of the Arrhenius plot.

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
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()


# ## Parameters
#
# The values are freely adjustable. They describe a moderately activated
# liquid-phase reaction: an activation energy of 50 kJ/mol roughly doubles the
# rate every ten kelvin around room temperature. Each run spans five
# characteristic times $1/k(T)$ of its own temperature, so $A$ always decays to
# about $e^{-5}\approx 0.7\,\%$ of its initial value.

A_pre = 1.0e6
E_a = 50.0e3
R_gas = 8.314462618
C_A0 = 1.0
temperatures = (290.0, 310.0, 330.0, 350.0)


def k_arrhenius(T):
    return A_pre * np.exp(-E_a / (R_gas * T))


for T in temperatures:
    print(
        f"T = {T:5.1f} K:  k(T) = {k_arrhenius(T):.4e} 1/s,  1/k = {1.0 / k_arrhenius(T):8.2f} s"
    )


# ## Model
#
# Two species and one irreversible reaction whose rate law is the Arrhenius
# expression above. The reactor is a single lumped control volume held at a
# constant temperature; the network is reused unchanged for every temperature.

network = ReactionNetworkIR(
    "arrhenius_decay",
    "1",
    species=(
        SpeciesIR("A", 0, 1e-27, composition={"H": 1}),
        SpeciesIR("B", 0, 1e-27, composition={"H": 1}),
    ),
    reactions=(
        ReactionIR(
            "decay",
            {"A": 1},
            {"B": 1},
            {
                "kind": "arrhenius",
                "pre_exponential": A_pre,
                "activation_energy": E_a,
                "boltzmann_constant": R_gas,
            },
        ),
    ),
    strict_balance=True,
)


def run_isothermal(T):
    """Integrate the decay at temperature T over five characteristic times."""
    k = k_arrhenius(T)
    tau = 1.0 / k
    domain = LumpedDomain()
    model = ReactionKinetics(domain=domain.system, temperature=T, network=network)
    C_A = model.concentration("A")
    C_B = model.concentration("B")
    problem = Problem(
        name=f"Arrhenius decay at {T:.0f} K",
        geometry=domain,
        physics=(model,),
        initial_conditions=(InitialValue(C_A, C_A0), InitialValue(C_B, 0.0)),
    )
    simulation = Simulation(
        problem=problem,
        mesh=LumpedMesh(domain),
        numerics=Numerics(
            spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()
        ),
    )
    simulation.validate().require_valid()
    result = simulation.run(start_time=0.0, end_time=5.0 * tau, time_step=0.05 * tau)

    times, num_A = result.field(C_A).time_series()
    _, num_B = result.field(C_B).time_series()
    return times, num_A, num_B


runs = {T: run_isothermal(T) for T in temperatures}


# ## Comparison with the exponential decay at each temperature
#
# For every temperature the stored concentrations are compared, time by time,
# with $C_{A0}e^{-k(T)t}$. The effective rate coefficient is then read off the
# computed decay as the slope of $\ln C_A$ against $t$, exactly as one would from
# a measured concentration record.

k_measured = {}
for T, (times, num_A, num_B) in runs.items():
    k = k_arrhenius(T)
    exact_A = C_A0 * np.exp(-k * times)
    exact_B = C_A0 - exact_A
    err = max(
        float(np.max(np.abs(num_A - exact_A))), float(np.max(np.abs(num_B - exact_B)))
    )
    k_measured[T] = -np.polyfit(times, np.log(num_A), 1)[0]
    print(
        f"T = {T:5.1f} K:  max |C - C_exact| = {err:.3e} mol/m^3 ({100 * err / C_A0:.1e} % of C_A0),"
        f"  fitted k = {k_measured[T]:.4e} 1/s (exact {k:.4e})"
    )


# ## Arrhenius ratio and Arrhenius plot
#
# The ratio of the two extreme rate coefficients must equal
# $\exp[-E_a/R\,(1/T_2-1/T_1)]$, and the fitted $\ln k$ values must fall on a
# straight line in $1/T$ whose slope returns the activation energy.

T1, T2 = temperatures[0], temperatures[-1]
ratio_reference = np.exp(-E_a / R_gas * (1.0 / T2 - 1.0 / T1))
ratio_measured = k_measured[T2] / k_measured[T1]
print(
    f"k({T2:.0f} K)/k({T1:.0f} K): OneSimulation {ratio_measured:.4f}, Arrhenius {ratio_reference:.4f}"
)

inv_T = np.array([1.0 / T for T in temperatures])
ln_k = np.array([np.log(k_measured[T]) for T in temperatures])
slope, intercept = np.polyfit(inv_T, ln_k, 1)
E_a_fit = -slope * R_gas
A_fit = np.exp(intercept)
print(
    f"Arrhenius plot: fitted E_a = {E_a_fit / 1e3:.3f} kJ/mol (set {E_a / 1e3:.1f} kJ/mol),"
    f"  fitted A = {A_fit:.4e} 1/s (set {A_pre:.1e} 1/s)"
)


# ## Figure

inv_fine = np.linspace(inv_T.min() * 0.98, inv_T.max() * 1.02, 100)
fig = plots.figure(1, 2, size=(11, 4.6))
plots.line(
    points={
        f"T = {T:.0f} K": (runs[T][0][1::4] / 60.0, runs[T][1][1::4])
        for T in temperatures
    },
    reference={
        f"T = {T:.0f} K": (
            runs[T][0][1:] / 60.0,
            C_A0 * np.exp(-k_arrhenius(T) * runs[T][0][1:]),
        )
        for T in temperatures
    },
    scale="logx",
    xlabel="time [min]",
    ylabel="concentration C_A [mol/m^3]",
    title="Decay of A: OneSimulation (markers) and exponential (dashed)",
    ax=fig[0],
)
plots.line(
    points={"OneSimulation (fitted k)": (1000 * inv_T, ln_k)},
    reference={
        "Arrhenius law ln k = ln A - E_a / (R T)": (
            1000 * inv_fine,
            np.log(k_arrhenius(1.0 / inv_fine)),
        )
    },
    xlabel="1000 / T [1/K]",
    ylabel="ln k",
    title=f"Arrhenius plot: slope gives E_a = {E_a_fit / 1e3:.1f} kJ/mol",
    ax=fig[1],
)
plots.show()


# ## Going further
#
# Replace the constant reactor temperature by a coupled thermal balance (a
# lumped `ThermalCapacitance` cooling to its surroundings) and the rate
# coefficient follows the falling temperature in time; the decay then becomes a
# time integral of $k(T(t))$. Make the reaction exothermic to obtain thermal
# runaway behaviour, or add a reverse reaction with its own activation energy to
# see how the equilibrium constant shifts with temperature (van 't Hoff).
