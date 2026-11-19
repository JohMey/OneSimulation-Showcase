"""OneSimulation example: neutralisation of dissolved chlorine by hydroxide in a scrubber liquid.
Requires the installed, activated OneSimulation runtime.

Chlorine dissolved in the alkaline washing liquid of a scrubber reacts with hydroxide within
microseconds. The well-mixed liquid is integrated for several hydroxide excesses and compared with
the closed-form solution of the second-order rate law with unequal starting concentrations.
"""

# # Neutralisation of chlorine in a scrubber
#
# Chlorine is removed from waste gas by absorption in caustic soda. In the liquid the dissolved
# chlorine is destroyed by hydroxide,
#
# $$\mathrm{Cl_2 + 2\,OH^- \longrightarrow Cl^- + OCl^- + H_2O},\qquad r = k\,c_\mathrm{Cl_2}\,c_\mathrm{OH^-},$$
#
# a reaction of first order in each reactant although two hydroxide ions are consumed. The liquid
# volume is taken as perfectly mixed and constant (0D), and the chlorine is assumed to have
# dissolved almost to saturation, $c_\mathrm{Cl_2,0}=10^{-2}$ mol/m³, before it reacts. The rate
# constant is of the order of the diffusion limit, $k=10^7$ m³/(mol s), so the liquid must be
# followed on a microsecond scale.
#
# | species | start [mol/m³] | role |
# |---|---|---|
# | Cl₂ | 0.01 | dissolved chlorine |
# | OH⁻ | 0.025 ... 1 | hydroxide of the washing liquid |
# | Cl⁻, OCl⁻ | 0 | products |
#
# **Reference.** With the extent $x$, $c_\mathrm{Cl_2}=A_0-x$ and $c_\mathrm{OH^-}=B_0-2x$; separating
# $\dot x = k(A_0-x)(B_0-2x)$ gives the closed form (Levenspiel, *Chemical Reaction Engineering*,
# chapter 3, irreversible second-order reactions with unequal reactant ratio)
#
# $$x(t)=\frac{B_0\,(1-E)}{2-(B_0/A_0)\,E},\qquad E=e^{(B_0-2A_0)\,k\,t} .$$
#
# For a large hydroxide excess this reduces to the pseudo-first-order decay $A_0e^{-kB_0t}$.

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


# ## Model
#
# The reaction order in hydroxide is declared separately from its stoichiometric coefficient.
# Water is the solvent and is not tracked, so the element balance is switched off.

k = 1e7
network = ReactionNetworkIR(
    "chlorine_neutralisation",
    "1",
    species=(
        SpeciesIR("Cl2", 0, 1.18e-25),
        SpeciesIR("OH", -1, 2.82e-26),
        SpeciesIR("Cl", -1, 5.89e-26),
        SpeciesIR("OCl", -1, 8.55e-26),
    ),
    reactions=(
        ReactionIR(
            "neutralisation",
            {"Cl2": 1, "OH": 2},
            {"Cl": 1, "OCl": 1},
            {"kind": "constant", "value": k},
            forward_orders={"OH": 1},
        ),
    ),
    strict_balance=False,
)


# ## Runs
#
# Four washing liquids, from a slight hydroxide excess over the stoichiometric 0.02 mol/m³ to a
# strong caustic of 1 mol/m³. Each run lasts until 99.9 % of the chlorine is gone, with 800
# Runge-Kutta steps.

domain = LumpedDomain()
chlorine = 0.01
hydroxide = (0.025, 0.05, 0.2, 1.0)
models = {
    B0: ReactionKinetics(domain=domain.system, network=network) for B0 in hydroxide
}
end = {
    B0: np.log((B0 - 1.998 * chlorine) / (0.001 * B0)) / ((B0 - 2 * chlorine) * k)
    for B0 in hydroxide
}
runs = {
    B0: Simulation(
        problem=Problem(
            name=f"scrubber with {B0} mol/m3 OH",
            geometry=domain,
            physics=(model,),
            initial_conditions=(
                InitialValue(model.concentration("Cl2"), chlorine),
                InitialValue(model.concentration("OH"), B0),
                InitialValue(model.concentration("Cl"), 0.0),
                InitialValue(model.concentration("OCl"), 0.0),
            ),
        ),
        mesh=LumpedMesh(domain),
        numerics=Numerics(
            spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()
        ),
    ).run(start_time=0.0, end_time=end[B0], time_step=end[B0] / 800)
    for B0, model in models.items()
}


# ## Comparison with the closed form
#
# The chlorine history of every run against $A_0-x(t)$, and the hydroxide left at the end against
# the stoichiometry.

extent = (
    lambda B0, t: B0
    * (1 - np.exp((B0 - 2 * chlorine) * k * t))
    / (2 - B0 / chlorine * np.exp((B0 - 2 * chlorine) * k * t))
)
histories = {
    B0: run.field(models[B0].concentration("Cl2")).time_series()
    for B0, run in runs.items()
}
for B0, (t, c) in histories.items():
    hydroxide_left = runs[B0].field(models[B0].concentration("OH")).values[0]
    print(
        f"OH0 = {B0:5.3f} mol/m^3: largest deviation from the closed form {np.max(np.abs(c - (chlorine - extent(B0, t)))) / chlorine:.1e} of Cl2_0; "
        f"time to 99 % removal {t[np.argmax(c < 0.01 * chlorine)] * 1e6:6.2f} us (closed form "
        f"{np.log((B0 - 1.98 * chlorine) / (0.01 * B0)) / ((B0 - 2 * chlorine) * k) * 1e6:6.2f} us); OH left {hydroxide_left:.5f} (stoichiometry {B0 - 2 * chlorine * (1 - c[-1] / chlorine):.5f})"
    )
t_ex, c_ex = histories[1.0]
print(
    f"strong caustic: largest deviation from the pseudo-first-order decay {np.max(np.abs(c_ex - chlorine * np.exp(-k * 1.0 * t_ex))) / chlorine:.1e} of Cl2_0"
)


# ## Figure
#
# Dissolved chlorine over time for the four washing liquids against the closed form, and the
# hydroxide consumed together with the products formed in the liquid with the smallest excess.

t_fine = {B0: np.geomspace(end[B0] / 1000, end[B0], 400) for B0 in hydroxide}
t25 = runs[0.025].field(models[0.025].concentration("OH")).times
fig = plots.figure(1, 2, size=(14, 5))
plots.line(
    points={
        f"OH0 = {B0} mol/m3": (t[10::20] * 1e6, c[10::20] / chlorine)
        for B0, (t, c) in histories.items()
    },
    reference={
        f"OH0 = {B0} mol/m3": (t_fine[B0] * 1e6, 1 - extent(B0, t_fine[B0]) / chlorine)
        for B0 in hydroxide
    },
    scale="loglog",
    ylimits=(1e-3, 1.5),
    xlabel="time [us]",
    ylabel="Cl2 / Cl2 at start",
    title="Dissolved chlorine, dashed: closed form",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    y={
        name: (
            np.asarray(t25) * 1e6,
            runs[0.025].field(models[0.025].concentration(name)).time_series()[1],
        )
        for name in ("Cl2", "OH", "Cl", "OCl")
    },
    xlabel="time [us]",
    ylabel="concentration [mol/m3]",
    title="Liquid with OH0 = 0.025 mol/m3",
    legend="center right",
    ax=fig[1],
)
plots.show()


# ## Going further
#
# Couple the liquid to a gas phase with a mass-transfer coefficient to see when absorption, not
# reaction, limits the scrubber; at these rate constants the reaction happens inside the liquid
# film, which is the regime of the enhancement factor. Lower the hydroxide below the
# stoichiometric amount to see the chlorine break through.
