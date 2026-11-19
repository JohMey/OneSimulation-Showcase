"""OneSimulation example: two stirred tank reactors in series with changing liquid volume and level control.
Requires the installed, activated OneSimulation runtime.

Two ideally mixed tank reactors with a first-order reaction fill up while the flows are fixed, then
proportional level controllers on the outlet valves take over. Volumes, mole balance, the level
response and the controlled steady state are checked against closed-form solutions.
"""

# # Stirred tank reactors in series with level control
#
# Fermenters and many liquid-phase reactors are ideally mixed tanks whose liquid volume is not fixed:
# feeds and withdrawals change the filling, and a level controller keeps it within limits. Two tanks
# with the same cross-section $S=10$ m$^2$ are connected in series. Tank 1 receives a feed with
# reactant A, tank 2 the transfer stream from tank 1 and a second, fresh feed; the product leaves tank 2.
# In both tanks A turns into B by a first-order reaction, $r=k\,c_A$.
#
# For every tank the volume, the moles of each species and the level $h=V/S$ follow from the balances
#
# $$\frac{dV}{dt}=\sum q_\text{in}-\sum q_\text{out},\qquad
#   \frac{d(Vc_i)}{dt}=\sum q_\text{in}c_{i,\text{in}}-\Big(\sum q_\text{out}\Big)c_i+\nu_i\,k\,c_A\,V .$$
#
# OneSimulation integrates them with `ProcessUnit(..., dynamic_volume=True, cross_section_area=S)`;
# a `ProcessStream` whose flow is a function of the current state can read the levels, which is how a
# controller is attached.
#
# | stream | from → to | phase 1: filling | phase 2: level control |
# |---|---|---|---|
# | feed | → tank 1, $c_A=8$ mol/m³ | 1.0 m³/min | 1.0 m³/min |
# | transfer | tank 1 → tank 2 | 0.9 m³/min | $q_{b1}+K_1(h_1-h_{s1})$ |
# | fresh feed | → tank 2, $c_A=3$ mol/m³ | 0.5 m³/min | 0.5 m³/min |
# | product | tank 2 → | 1.0 m³/min | $q_{b2}+K_2(h_2-h_{s2})$ |
#
# Phase 1 starts from 1 m³ and 1.5 m³ of pure solvent and lasts two hours; phase 2 starts from its end
# state. The rate constant is $k=0.37$ 1/min.
#
# **Reference** (O. Levenspiel, *Chemical Reaction Engineering*, 3rd ed., Wiley 1999, ch. 5 and 6;
# D. E. Seborg, T. F. Edgar, D. A. Mellichamp and F. J. Doyle, *Process Dynamics and Control*, 4th ed.,
# Wiley 2016, §2.4, §4 and §11.3). With fixed flows the volumes grow linearly, and the total
# concentration $s=c_A+c_B$ in tank 1, untouched by the reaction, has the closed form
#
# $$V_j(t)=V_j(0)+\Big(\sum q_\text{in}-\sum q_\text{out}\Big)_j\,t,\qquad
#   s_1(t)=c_{A,\text{feed}}\Big[1-\Big(\frac{V_1(0)}{V_1(t)}\Big)^{q_\text{feed}/(q_\text{feed}-q_\text{transfer})}\Big].$$
#
# Under proportional control tank 1 is a first-order system with time constant $\tau_1=S/K_1$ that
# settles with the offset $(q_\text{feed}-q_{b1})/K_1$ above its set point; tank 2 is driven by it and
# responds with two exponentials:
#
# $$h_1=h_{1\infty}+\Delta_1e^{-K_1t/S},\qquad
#   h_2=h_{2\infty}+(\Delta_2-\beta)\,e^{-K_2t/S}+\beta\,e^{-K_1t/S},\qquad \beta=\frac{K_1\Delta_1}{K_2-K_1},$$
#
# with $h_{1\infty}=h_{s1}+(q_\text{feed}-q_{b1})/K_1$, $h_{2\infty}=h_{s2}+(q_\text{feed}+q_\text{fresh}-q_{b2})/K_2$ and
# the initial distances $\Delta_j$ from them. At steady state each tank is a CSTR,
# $c_{A1}=q_\text{feed}c_{A,\text{feed}}/(q_\text{feed}+kV_1)$ and
# $c_{A2}=(q_\text{feed}c_{A1}+q_\text{fresh}c_{A,\text{fresh}})/(q_\text{feed}+q_\text{fresh}+kV_2)$.

import numpy as np
from onesim.physics.systems import ProcessModelsProvider, ProcessStream, ProcessUnit
from onesim.visualization import plots, theme
from scipy.integrate import cumulative_trapezoid

theme.set()

# ## Phase 1: filling with fixed flows
#
# Flows in m³/s, times in s. The reaction law returns the production rate of each species and the heat
# released (none here). `bind_provider_parameters` selects the classical fourth-order Runge-Kutta method
# for the balances.

minute = 60.0
area, k, feed_a, fresh_a = 10.0, 0.37 / minute, 8.0, 3.0
feed, transfer, fresh, product = 1.0 / minute, 0.9 / minute, 0.5 / minute, 1.0 / minute
rk4 = ProcessModelsProvider.bind_provider_parameters(
    {"time_integration": "classical_rk4"}
)
reaction = lambda c, temperature, time: ({"A": -k * c["A"], "B": k * c["A"]}, 0.0)
filling = ProcessModelsProvider(
    component_ids=("A", "B"),
    units=(
        ProcessUnit(
            "tank1",
            volume=1.0,
            initial_concentrations={"A": 0.0, "B": 0.0},
            reaction_law=reaction,
            dynamic_volume=True,
            cross_section_area=area,
        ),
        ProcessUnit(
            "tank2",
            volume=1.5,
            initial_concentrations={"A": 0.0, "B": 0.0},
            reaction_law=reaction,
            dynamic_volume=True,
            cross_section_area=area,
        ),
    ),
    streams=(
        ProcessStream("feed", None, "tank1", feed, {"A": feed_a}),
        ProcessStream("transfer", "tank1", "tank2", transfer),
        ProcessStream("fresh", None, "tank2", fresh, {"A": fresh_a}),
        ProcessStream("product", "tank2", None, product),
    ),
    final_time=120 * minute,
    time_step=3.0,
    **rk4,
).execute()

# The volumes against the linear closed form, the total concentration in tank 1 against $s_1(t)$, and
# the mole balance of the plant: moles in both tanks plus moles that left with the product must equal
# the moles fed.

t = filling.times
c1, c2 = filling.concentrations["tank1"], filling.concentrations["tank2"]
V1, V2 = filling.volumes["tank1"], filling.volumes["tank2"]
exact_V1, exact_V2 = 1.0 + (feed - transfer) * t, 1.5 + (fresh + transfer - product) * t
exact_s1 = feed_a * (1 - (1.0 / exact_V1) ** (feed / (feed - transfer)))
inventory = V1 * (c1["A"] + c1["B"]) + V2 * (c2["A"] + c2["B"])
left = cumulative_trapezoid(
    filling.flows["product"] * (c2["A"] + c2["B"]), t, initial=0.0
)
fed = (feed * feed_a + fresh * fresh_a) * t
print(
    f"after 2 h: V1 = {V1[-1]:.4f} m³ (closed form {exact_V1[-1]:.4f}), V2 = {V2[-1]:.4f} m³ (closed form {exact_V2[-1]:.4f}); "
    f"largest volume deviation {max(np.abs(V1 - exact_V1).max(), np.abs(V2 - exact_V2).max()):.1e} m³"
)
print(
    f"levels after 2 h: h1 = {filling.levels['tank1'][-1]:.3f} m, h2 = {filling.levels['tank2'][-1]:.3f} m"
)
print(
    f"total concentration in tank 1 against the closed form: largest deviation {np.abs(c1['A'] + c1['B'] - exact_s1).max():.1e} mol/m³ "
    f"(final value {exact_s1[-1]:.4f} mol/m³)"
)
print(
    f"mole balance: {fed[-1]:.1f} mol fed, {inventory[-1]:.1f} mol in the tanks, {left[-1]:.1f} mol left with the product; "
    f"largest relative imbalance {np.abs(inventory + left - fed)[1:].max() / fed[-1]:.1e}"
)

# ## Phase 2: proportional level control
#
# The transfer and the product valves now follow proportional laws of the current levels. Both tanks
# start from the end of phase 1.

setpoint_1, setpoint_2, bias_1, bias_2 = 1.0, 2.0, 0.9 / minute, 1.0 / minute
gain_1, gain_2 = 5.0 / minute, 2.5 / minute
controlled = ProcessModelsProvider(
    component_ids=("A", "B"),
    units=tuple(
        ProcessUnit(
            name,
            volume=filling.volumes[name][-1],
            initial_concentrations={
                s: filling.concentrations[name][s][-1] for s in "AB"
            },
            reaction_law=reaction,
            dynamic_volume=True,
            cross_section_area=area,
        )
        for name in ("tank1", "tank2")
    ),
    streams=(
        ProcessStream("feed", None, "tank1", feed, {"A": feed_a}),
        ProcessStream(
            "transfer",
            "tank1",
            "tank2",
            lambda state, time: bias_1 + gain_1 * (state.levels["tank1"] - setpoint_1),
        ),
        ProcessStream("fresh", None, "tank2", fresh, {"A": fresh_a}),
        ProcessStream(
            "product",
            "tank2",
            None,
            lambda state, time: bias_2 + gain_2 * (state.levels["tank2"] - setpoint_2),
        ),
    ),
    final_time=60 * minute,
    time_step=1.0,
    **rk4,
).execute()

tc = controlled.times
h1, h2 = controlled.levels["tank1"], controlled.levels["tank2"]
steady_1, steady_2 = (
    setpoint_1 + (feed - bias_1) / gain_1,
    setpoint_2 + (feed + fresh - bias_2) / gain_2,
)
delta_1, delta_2 = h1[0] - steady_1, h2[0] - steady_2
beta = gain_1 * delta_1 / (gain_2 - gain_1)
exact_h1 = steady_1 + delta_1 * np.exp(-gain_1 * tc / area)
exact_h2 = (
    steady_2
    + (delta_2 - beta) * np.exp(-gain_2 * tc / area)
    + beta * np.exp(-gain_1 * tc / area)
)
cstr_1 = feed * feed_a / (feed + k * area * steady_1)
cstr_2 = (feed * cstr_1 + fresh * fresh_a) / (feed + fresh + k * area * steady_2)
print(
    f"level response: largest deviation from the closed form {np.abs(h1 - exact_h1).max():.1e} m (tank 1), "
    f"{np.abs(h2 - exact_h2).max():.1e} m (tank 2); time constants {area / gain_1 / minute:.0f} and {area / gain_2 / minute:.0f} min"
)
print("                          simulation (1 h)   controller law / CSTR   deviation")
for name, value, exact in (
    ("level tank 1 [m]", h1[-1], steady_1),
    ("level tank 2 [m]", h2[-1], steady_2),
    ("c_A tank 1 [mol/m³]", controlled.concentrations["tank1"]["A"][-1], cstr_1),
    ("c_A tank 2 [mol/m³]", controlled.concentrations["tank2"]["A"][-1], cstr_2),
    (
        "product flow [m³/min]",
        minute * controlled.flows["product"][-1],
        minute * (feed + fresh),
    ),
):
    print(
        f"{name:24s} {value:16.5f} {exact:22.5f} {100 * (value / exact - 1):+11.4f} %"
    )
print(
    f"offset of proportional control: tank 1 settles {100 * (steady_1 - setpoint_1):.0f} cm, tank 2 {100 * (steady_2 - setpoint_2):.0f} cm above its set point"
)

# ## Figure
#
# Top: phase 1, the growing volumes with their closed form and the concentrations in both tanks. Bottom:
# phase 2, the levels drawn down by the controllers to their steady values above the set points, and the
# concentrations of A settling at the CSTR values.

fig = plots.figure(2, 2, size=(14, 9))
plots.line(
    t / minute,
    {"tank 1": V1, "tank 2": V2},
    reference={"tank 1": exact_V1, "tank 2": exact_V2},
    xlabel="time [min]",
    ylabel="liquid volume [m³]",
    title="Phase 1: filling (dashed: closed form)",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    t / minute,
    {
        "A, tank 1": c1["A"],
        "B, tank 1": c1["B"],
        "A, tank 2": c2["A"],
        "B, tank 2": c2["B"],
    },
    reference={"A + B in tank 1, closed form": exact_s1},
    xlabel="time [min]",
    ylabel="concentration [mol/m³]",
    title="Phase 1: concentrations",
    legend="center right",
    ax=fig[1],
)
plots.line(
    tc / minute,
    {"tank 1": h1, "tank 2": h2},
    reference={"tank 1": exact_h1, "tank 2": exact_h2},
    hlines={"set point tank 1": setpoint_1, "set point tank 2": setpoint_2},
    xlimits=(0, 30),
    xlabel="time [min]",
    ylabel="liquid level [m]",
    title="Phase 2: level control (dashed: closed form)",
    legend="upper right",
    ax=fig[2],
)
plots.line(
    tc / minute,
    {
        "A, tank 1": controlled.concentrations["tank1"]["A"],
        "A, tank 2": controlled.concentrations["tank2"]["A"],
    },
    hlines={"CSTR steady state, tank 1": cstr_1, "CSTR steady state, tank 2": cstr_2},
    xlabel="time [min]",
    ylabel="concentration of A [mol/m³]",
    title="Phase 2: approach to the steady state",
    legend="center right",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Add an integral part to the valve laws to remove the offset, or replace the proportional law by the
# blocks of `ControlSignalFlowProvider` (gain, sum, limiter) to model a valve that cannot open beyond
# its full stroke. A heat source and a temperature-dependent rate constant turn the tanks into a
# non-isothermal reactor cascade.
