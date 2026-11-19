"""OneSimulation example: kinetics of the catalytic carbonylation route to ibuprofen.
Requires the installed, activated OneSimulation runtime.

The last step of the industrial ibuprofen synthesis, the palladium-catalysed carbonylation of
1-(4-isobutylphenyl)ethanol, is modelled as a reaction network in a stirred autoclave fed with
carbon monoxide through the gas-liquid interface. The simulation is compared with the closed-form
solution of the network and with the exact selectivity, and shows when gas absorption limits the
rate.
"""

# # Ibuprofen synthesis: carbonylation kinetics with gas absorption
#
# In the three-step BHC process ibuprofen is made from 1-(4-isobutylphenyl)ethanol (IBPE) and
# carbon monoxide over a homogeneous palladium catalyst in acidic solution. Under the acid the
# alcohol first dehydrates to 4-isobutylstyrene (IBS), which is then carbonylated to the branched
# acid, ibuprofen (IBU), or, as by-product, to the linear 3-(4-isobutylphenyl)propanoic acid
# (3-IPPA):
#
# | step | reaction | rate law |
# |---|---|---|
# | dehydration | IBPE → IBS + H₂O | $r_1 = k_1\,c_\mathrm{IBPE}$ |
# | branched carbonylation | IBS + CO + H₂O → IBU | $r_2 = k_2\,c_\mathrm{IBS}\,c_\mathrm{CO}$ |
# | linear carbonylation | IBS + CO + H₂O → 3-IPPA | $r_3 = k_3\,c_\mathrm{IBS}\,c_\mathrm{CO}$ |
# | absorption | CO(gas) → CO(liquid) | $N = k_La\,(c^\ast_\mathrm{CO} - c_\mathrm{CO})$ |
#
# Water is present in excess, so the rates are of zero order in water. The gas cap is held at
# constant CO pressure, which fixes the saturation concentration $c^\ast_\mathrm{CO}$ by Henry's
# law; the absorption enters the liquid as a constant supply $k_La\,c^\ast$ and a first-order
# loss $k_La\,c_\mathrm{CO}$. The liquid is perfectly mixed (0D). The rate constants are
# representative values for illustration, not a fit to a particular catalyst.
#
# **Reference.** Process: Elango et al., US patent 4,981,995 (1991); kinetic analysis of the
# carbonylation network: Seayad, Jayasree & Chaudhari, *Org. Lett.* 1 (1999) 459. Two results
# hold exactly for every absorption rate, because the dehydration does not involve CO and both
# carbonylations are of first order in CO:
#
# $$c_\mathrm{IBPE}=c_0e^{-k_1t},\qquad S=\frac{c_\mathrm{IBU}}{c_\mathrm{IBU}+c_\mathrm{IPPA}}=\frac{k_2}{k_2+k_3} .$$
#
# If the liquid stays saturated, $k' = (k_2+k_3)\,c^\ast$ and the network has the closed form of
# two consecutive first-order steps,
# $c_\mathrm{IBS} = c_0\,\frac{k_1}{k'-k_1}\big(e^{-k_1t}-e^{-k't}\big)$.

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler
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


# ## Reaction network
#
# Concentrations in mol/m³, time in seconds. Each species carries its elemental composition, so
# the three reactions are checked for element balance; the absorption term is a supply, which is
# why the strict network check is switched off.

c0, saturation = 500.0, 100.0
k1, k2, k3 = 3.0e-4, 4.8e-5, 2.0e-6
composition = {
    "IBPE": {"C": 12, "H": 18, "O": 1},
    "IBS": {"C": 12, "H": 16},
    "H2O": {"H": 2, "O": 1},
    "CO": {"C": 1, "O": 1},
    "IBU": {"C": 13, "H": 18, "O": 2},
    "IPPA": {"C": 13, "H": 18, "O": 2},
}
reactions = (
    ReactionIR(
        "dehydration",
        {"IBPE": 1},
        {"IBS": 1, "H2O": 1},
        {"kind": "constant", "value": k1},
    ),
    ReactionIR(
        "branched",
        {"IBS": 1, "CO": 1, "H2O": 1},
        {"IBU": 1},
        {"kind": "constant", "value": k2},
        forward_orders={"H2O": 0},
    ),
    ReactionIR(
        "linear",
        {"IBS": 1, "CO": 1, "H2O": 1},
        {"IPPA": 1},
        {"kind": "constant", "value": k3},
        forward_orders={"H2O": 0},
    ),
)
species = tuple(
    SpeciesIR(name, 0, 1e-25, composition=atoms) for name, atoms in composition.items()
)
print(
    "element balance of the three reactions:",
    ReactionNetworkIR(
        "check", "1", species=species, reactions=reactions, strict_balance=True
    ).strict_balance,
)


# ## Autoclave runs
#
# Four stirring intensities, from a gas-liquid transfer that keeps the liquid saturated to one that
# starves it. Each run covers four hours with the second-order BDF method and Newton's method.

domain = LumpedDomain()
transfer = (1.0, 3e-3, 1e-3, 3e-4)
networks = {
    kLa: ReactionNetworkIR(
        "ibuprofen",
        "1",
        species=species,
        reactions=reactions
        + (
            ReactionIR("desorption", {"CO": 1}, {}, {"kind": "constant", "value": kLa}),
        ),
    )
    for kLa in transfer
}
models = {
    kLa: ReactionKinetics(
        domain=domain.system, network=network, species_sources={"CO": kLa * saturation}
    )
    for kLa, network in networks.items()
}
start = {
    "IBPE": c0,
    "IBS": 0.0,
    "H2O": 2000.0,
    "CO": saturation,
    "IBU": 0.0,
    "IPPA": 0.0,
}
numerics = Numerics(
    spatial=LumpedParameters(),
    time_integrator=BDF(order=2, startup=BackwardEuler()),
    nonlinear_solver=Newton(
        linear_solver=SparseDirect(),
        jacobian="analytic",
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    ),
)
runs = {
    kLa: Simulation(
        problem=Problem(
            name=f"carbonylation kLa={kLa}",
            geometry=domain,
            physics=(model,),
            initial_conditions=tuple(
                InitialValue(model.concentration(name), value)
                for name, value in start.items()
            ),
        ),
        mesh=LumpedMesh(domain),
        numerics=numerics,
    ).run(start_time=0.0, end_time=4 * 3600.0, time_step=10.0)
    for kLa, model in models.items()
}
history = {
    kLa: {
        name: run.field(models[kLa].concentration(name)).time_series()[1]
        for name in start
    }
    for kLa, run in runs.items()
}
times = runs[transfer[0]].field(models[transfer[0]].concentration("IBPE")).times


# ## Comparison with the closed forms
#
# IBPE decay and selectivity for every run, and the intermediate IBS of the best-stirred run
# against the consecutive-reaction solution of a saturated liquid.

fast = (k2 + k3) * saturation
styrene = lambda t: c0 * k1 / (fast - k1) * (np.exp(-k1 * t) - np.exp(-fast * t))
for kLa, c in history.items():
    selectivity = c["IBU"][-1] / (c["IBU"][-1] + c["IPPA"][-1])
    print(
        f"kLa = {kLa:g} 1/s: IBPE deviation from c0 exp(-k1 t) {np.max(np.abs(c['IBPE'] - c0 * np.exp(-k1 * times))) / c0:.1e}, "
        f"selectivity {selectivity:.6f} (exact {k2 / (k2 + k3):.6f}), ibuprofen yield after 1 h {c['IBU'][times <= 3600][-1] / c0:.3f}, after 4 h {c['IBU'][-1] / c0:.3f}, "
        f"lowest dissolved CO {c['CO'].min() / saturation:.2f} c*"
    )
print(
    f"best-stirred run: IBS deviation from the saturated closed form {np.max(np.abs(history[1.0]['IBS'] - styrene(times))) / c0:.1e} of c0, "
    f"peak IBS {history[1.0]['IBS'].max():.1f} mol/m^3 at {times[np.argmax(history[1.0]['IBS'])] / 60:.0f} min "
    f"(closed form {styrene(np.log(fast / k1) / (fast - k1)):.1f} at {np.log(fast / k1) / (fast - k1) / 60:.0f} min)"
)
print(
    f"carbon monoxide balance of the starved run: absorbed {np.trapezoid(3e-4 * (saturation - history[3e-4]['CO']), times):.2f} mol/m^3, "
    f"bound in acids plus dissolved change {history[3e-4]['IBU'][-1] + history[3e-4]['IPPA'][-1] + history[3e-4]['CO'][-1] - saturation:.2f} mol/m^3"
)


# ## Figure
#
# The species of the best-stirred run against the closed forms, ibuprofen yield for the four
# stirring intensities, and the dissolved CO that limits the slow runs.

minutes = times / 60
shown = slice(None, None, 48)
fig = plots.figure(1, 3, size=(18, 5))
plots.line(
    points={
        name: (minutes[shown], history[1.0][name][shown] / c0)
        for name in ("IBPE", "IBS", "IBU")
    },
    reference={
        "IBPE": (minutes, np.exp(-k1 * times)),
        "IBS": (minutes, styrene(times) / c0),
        "IBU": (
            minutes,
            k2 / (k2 + k3) * (1 - np.exp(-k1 * times) - styrene(times) / c0),
        ),
    },
    xlabel="time [min]",
    ylabel="c / c0",
    title="Saturated liquid, dashed: closed form",
    legend="center right",
    ax=fig[0],
)
plots.line(
    y={f"kLa = {kLa} 1/s": (minutes, c["IBU"] / c0) for kLa, c in history.items()},
    xlabel="time [min]",
    ylabel="ibuprofen yield",
    title="Yield and gas absorption",
    legend="upper left",
    ax=fig[1],
)
plots.line(
    y={
        f"kLa = {kLa} 1/s": (minutes, c["CO"] / saturation)
        for kLa, c in history.items()
    },
    ylimits=(0, 1.05),
    xlabel="time [min]",
    ylabel="dissolved CO / saturation",
    title="Carbon monoxide in the liquid",
    legend="center right",
    ax=fig[2],
)
plots.show()


# ## Going further
#
# Make the dehydration reversible with the water concentration as a second-order term, or let the
# CO pressure fall in a closed autoclave by adding the gas cap as a species with a constant
# volume. With an Arrhenius rate law per step the same network gives the temperature window in
# which the carbonylation outruns the dehydration.
