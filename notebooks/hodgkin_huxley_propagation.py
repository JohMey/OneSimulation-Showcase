"""OneSimulation example: the membrane action potential of the Hodgkin-Huxley squid axon.
Requires the installed, activated OneSimulation runtime.

The space-clamped membrane of the 1952 Hodgkin-Huxley model is four ordinary differential
equations: the membrane potential and three gates. They are built as a signal-flow graph with the
published channel set of OneSimulation and integrated after a sudden depolarisation; the
all-or-nothing threshold and the shape of the action potential are compared with the original
computation of Hodgkin and Huxley.
"""

# # The Hodgkin-Huxley membrane action potential
#
# Hodgkin and Huxley (1952) described the membrane of the squid giant axon by a capacitor in
# parallel with a voltage-gated sodium channel ($m^3h$), a voltage-gated potassium channel ($n^4$)
# and a passive leak. With the whole membrane at one potential (a space-clamped "membrane action
# potential", no propagation along the axon) the model is a system of four ordinary differential
# equations:
#
# $$C_m\,\frac{dV}{dt} = -\bar g_\mathrm{Na}\,m^3h\,(V-E_\mathrm{Na})
#    - \bar g_\mathrm{K}\,n^4\,(V-E_\mathrm{K}) - g_L\,(V-E_L),\qquad
#   \frac{dy}{dt}=\alpha_y(V)\,(1-y)-\beta_y(V)\,y,\quad y\in\{m,h,n\}.$$
#
# | quantity | value |
# |---|---|
# | membrane capacitance $C_m$ | 1 µF/cm² |
# | sodium: $\bar g_\mathrm{Na}$, $E_\mathrm{Na}$ | 120 mS/cm², +50 mV |
# | potassium: $\bar g_\mathrm{K}$, $E_\mathrm{K}$ | 36 mS/cm², -77 mV |
# | leak: $g_L$, $E_L$ | 0.3 mS/cm², -54.4 mV |
# | resting potential, temperature | -65 mV, 6.3 °C |
#
# OneSimulation ships these channels with their rate functions $\alpha_y$, $\beta_y$ as
# `HodgkinHuxley1952`. The four equations are assembled as a signal-flow graph: an integrator
# block holds each state, and custom blocks evaluate the gate kinetics and the ionic current
# from the channel set. The membrane starts at rest with the gates at their resting values, and
# the potential is displaced suddenly by $\Delta V$ (a short shock that charges the membrane), as
# in the original computation. The graph is integrated with the classical fourth-order
# Runge-Kutta method.
#
# **Reference.** A. L. Hodgkin and A. F. Huxley, "A quantitative description of membrane current
# and its application to conduction and excitation in nerve", J. Physiol. 117, 500-544 (1952),
# Part V and Fig. 12: membrane action potentials computed at 6 °C for initial depolarisations of
# 6, 7, 15 and 90 mV. A displacement of 6 mV dies away, 7 mV and more give a full action
# potential: the threshold lies between 6 and 7 mV. The spike overshoots zero, lasts a few
# milliseconds at this temperature and ends with an after-hyperpolarisation below rest.

import numpy as np
from onesim.numerics.time import DEFAULT_TIME_INTEGRATOR_REGISTRY
from onesim.physics.biophysics import HodgkinHuxley1952
from onesim.physics.systems import ControlSignalFlowProvider, SignalBlock
from onesim.visualization import plots, theme

theme.set()

# ## Model
#
# The channel set gives the conductances, the reversal potentials, the gate exponents and the rate
# functions. The ionic current is summed over the channels, so the same graph runs any channel set
# of the same form. Each run starts from rest displaced by $\Delta V$ and lasts 20 ms.

membrane = HodgkinHuxley1952()
capacitance, rest = 0.01, -0.065
state = ("V",) + tuple(gate.name for gate in membrane.gates)
ionic = lambda s: sum(
    channel.conductance
    * np.prod([s[state.index(g)] ** power for g, power in channel.gate_powers.items()])
    * (s[0] - channel.reversal_potential)
    for channel in membrane.channels
)
action_potential = lambda shift: ControlSignalFlowProvider(
    blocks=(
        SignalBlock(
            "V",
            "integrator",
            inputs=("dV",),
            parameters={"initial_value": rest + shift},
        ),
        SignalBlock(
            "dV",
            "custom",
            inputs=state,
            custom_evaluator=lambda s, t, p: -ionic(s) / capacitance,
            custom_derivative=lambda *a: (),
        ),
    )
    + tuple(
        block
        for gate in membrane.gates
        for block in (
            SignalBlock(
                gate.name,
                "integrator",
                inputs=("d" + gate.name,),
                parameters={"initial_value": gate.initial_value},
            ),
            SignalBlock(
                "d" + gate.name,
                "custom",
                inputs=("V", gate.name),
                custom_evaluator=lambda s, t, p, rate=gate.rate: float(
                    np.dot(rate(np.asarray(s[0])), (1 - s[1], -s[1]))
                ),
                custom_derivative=lambda *a: (),
            ),
        )
    ),
    final_time=0.02,
    time_step=1e-5,
    time_integrator=DEFAULT_TIME_INTEGRATOR_REGISTRY.resolve_id("classical_rk4"),
).execute()

print(
    "channels:",
    ", ".join(f"{c.name} {c.conductance / 10:g} mS/cm²" for c in membrane.channels),
    "| gates at rest:",
    ", ".join(f"{g.name} = {g.initial_value:.4f}" for g in membrane.gates),
)

# ## Results
#
# The four displacements of the original figure, and a scan between 6 and 7 mV that locates the
# threshold: a run counts as an action potential when the potential overshoots 0 mV.

shifts = (0.006, 0.007, 0.015, 0.090)
runs = {shift: action_potential(shift) for shift in shifts}
scan = {
    shift: max(action_potential(shift).signals["V"])
    for shift in np.round(np.arange(0.0060, 0.00701, 0.0001), 4)
}
times = 1e3 * np.asarray(runs[shifts[0]].times)
voltage = {shift: 1e3 * np.asarray(run.signals["V"]) for shift, run in runs.items()}
threshold = min(shift for shift, peak in scan.items() if peak > 0.0)

print(
    "ΔV [mV]   peak [mV]   time of peak [ms]   after-hyperpolarisation [mV]   duration above -40 mV [ms]"
)
for shift, v in voltage.items():
    above = times[v > -40.0]
    print(
        f"{1e3 * shift:7.0f} {v.max():11.1f} {times[v.argmax()]:19.2f} {v.min():30.1f} "
        f"{(above[-1] - above[0]) if len(above) else 0.0:27.2f}"
    )
print(
    f"threshold from the scan: between {1e3 * (threshold - 0.0001):.1f} and {1e3 * threshold:.1f} mV "
    f"(Hodgkin and Huxley 1952: between 6 and 7 mV)"
)
print(
    f"potential 20 ms after a 6 mV displacement: {voltage[0.006][-1]:.2f} mV (rest {1e3 * rest:.0f} mV)"
)

# ## Figure
#
# The membrane potential after the four displacements, as in Fig. 12 of the original paper, and
# the sodium and potassium gates during the action potential after 15 mV.

fig = plots.figure(1, 2, size=(14, 5.2))
plots.line(
    times,
    {f"ΔV = {1e3 * shift:.0f} mV": v for shift, v in voltage.items()},
    hlines={"rest": 1e3 * rest},
    xlabel="time [ms]",
    ylabel="membrane potential [mV]",
    title=f"Membrane action potentials, threshold {1e3 * threshold:.1f} mV",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    times,
    {f"gate {name}": runs[0.015].signals[name] for name in state[1:]},
    xlabel="time [ms]",
    ylabel="gate opening",
    title="Gates during the action potential (ΔV = 15 mV)",
    legend="center right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Replace the initial displacement by a constant stimulus current (a `source` block added to
# the current balance): above about 6 µA/cm² the membrane fires repetitively. Raise the temperature
# with `HodgkinHuxley1952(temperature=...)` and the spike becomes shorter and smaller through the
# built-in $Q_{10}=3$ of the rates. The propagating action potential along the axon is the
# monodomain cable equation with the same channel set (`Monodomain`).
