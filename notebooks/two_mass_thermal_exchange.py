"""OneSimulation example: two lumped thermal masses exchanging heat.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form two-node solution.
"""


# # Heat exchange between two thermal masses
#
# Two compact bodies with heat capacities $C_1$ and $C_2$ are connected by a
# thermal conductance $G$ (for example a short metal bridge) and are otherwise
# perfectly insulated. Each body is small enough to stay uniform in temperature,
# so two lumped balances describe the system:
#
# $$C_1\,\frac{dT_1}{dt}=-G\,(T_1-T_2),\qquad C_2\,\frac{dT_2}{dt}=-G\,(T_2-T_1).$$
#
# In OneSimulation each body is a lumped thermal capacitance whose "ambient"
# input is fed by the other body's temperature. The two models are coupled
# monolithically and integrated with the classical fourth-order Runge-Kutta method.
#
# **Reference.** Adding the two balances shows that the total stored energy is
# conserved, so the capacity-weighted mean temperature
#
# $$\bar T=\frac{C_1 T_1+C_2 T_2}{C_1+C_2}$$
#
# stays constant, while the temperature difference decays exponentially,
#
# $$T_1(t)-T_2(t)=(T_{1,0}-T_{2,0})\,e^{-G\,(1/C_1+1/C_2)\,t}.$$
#
# Both statements are checked at every stored time.

from importlib.metadata import version
import numpy as np
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

from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.coupled_systems import MonolithicCoupling
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.field_transfer import GlobalValueTransfer
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.systems import ThermalCapacitance
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The parameters can be changed. The reference formulas below stay valid as long as
# the two heat capacities and the conductance are constant and no heat leaves the
# pair. The values describe a hot 1.3 kg copper block bolted to a cooler 1.7 kg
# aluminium block through a thin bridge.

heat_capacity_1 = 500.0     # J/K, C1 (hot body)
heat_capacity_2 = 1500.0    # J/K, C2 (cool body)
conductance = 2.0           # W/K, G of the bridge between the bodies
initial_temperature_1 = 360.0  # K
initial_temperature_2 = 300.0  # K

decay_rate = conductance * (1.0 / heat_capacity_1 + 1.0 / heat_capacity_2)
time_constant = 1.0 / decay_rate
mean_temperature = (
    heat_capacity_1 * initial_temperature_1 + heat_capacity_2 * initial_temperature_2
) / (heat_capacity_1 + heat_capacity_2)
end_time = 4.0 * time_constant
time_step = 0.02 * time_constant
print(f"time constant 1/(G(1/C1+1/C2)) = {time_constant:.1f} s,  final temperature = {mean_temperature:.2f} K")

domain = LumpedDomain()
body_1 = ThermalCapacitance(
    domain=domain.system,
    heat_capacity=heat_capacity_1,
    conductance=conductance,
    ambient_temperature=None,   # supplied by body 2 through the coupling below
    heating=0.0,
)
body_2 = ThermalCapacitance(
    domain=domain.system,
    heat_capacity=heat_capacity_2,
    conductance=conductance,
    ambient_temperature=None,   # supplied by body 1 through the coupling below
    heating=0.0,
)
problem = Problem(
    name="Two thermal masses exchanging heat",
    geometry=domain,
    physics=(body_1, body_2),
    initial_conditions=(
        InitialValue(body_1.temperature, initial_temperature_1),
        InitialValue(body_2.temperature, initial_temperature_2),
    ),
    couplings=(
        FieldCoupling(source=body_2.temperature, target=body_1.ambient, transfer=GlobalValueTransfer()),
        FieldCoupling(source=body_1.temperature, target=body_2.ambient, transfer=GlobalValueTransfer()),
    ),
)
numerics = Numerics(spatial=LumpedParameters())
simulation = Simulation(
    problem=problem,
    mesh={body_1: LumpedMesh(domain), body_2: LumpedMesh(domain)},
    numerics={body_1: numerics, body_2: numerics},
    coupling_solver=MonolithicCoupling(
        transient_systems=(body_1, body_2),
        time_integrator=ClassicalRungeKutta4(),
    ),
)
simulation.validate().require_valid()

sink = MemoryOutputSink()
result = simulation.run(
    start_time=0.0,
    end_time=end_time,
    time_step=time_step,
    output=Output(
        fields=(body_1.temperature, body_2.temperature),
        interval=time_step,
        sink=sink,
    ),
)
times = np.asarray([float(s.time) for s in sink.snapshots])


def series(field):
    return np.asarray([
        float(np.asarray(s.fields[field.field_id].values).reshape(-1)[0])
        for s in sink.snapshots
    ])


temperatures_1 = series(body_1.temperature)
temperatures_2 = series(body_2.temperature)


# ## Comparison with the closed-form solution
#
# We check the exponential decay of the temperature difference and the constancy
# of the capacity-weighted mean at every stored time. A passing check shows that
# this numerical example agrees with the two-node balance; it is not a measurement
# of real blocks.

difference = temperatures_1 - temperatures_2
reference_difference = (initial_temperature_1 - initial_temperature_2) * np.exp(-decay_rate * times)
max_difference_error = float(np.max(np.abs(difference - reference_difference)))
weighted_mean = (heat_capacity_1 * temperatures_1 + heat_capacity_2 * temperatures_2) / (
    heat_capacity_1 + heat_capacity_2
)
max_mean_drift = float(np.max(np.abs(weighted_mean - mean_temperature)))
print(f"Maximum error of the temperature difference: {max_difference_error:.3e} K")
print(f"Maximum drift of the capacity-weighted mean:  {max_mean_drift:.3e} K")

# Each temperature individually follows from the two invariants.
# Recover the decay rate from the log-slope of the difference.
fitted_rate = -np.polyfit(times, np.log(difference), 1)[0]
print(f"Decay rate: fitted {fitted_rate:.6f} 1/s, exact G(1/C1+1/C2) = {decay_rate:.6f} 1/s")


# ## Figure

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
for ax in axes:
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
dense_t = np.linspace(0.0, end_time, 400)
dense_diff = (initial_temperature_1 - initial_temperature_2) * np.exp(-decay_rate * dense_t)
axes[0].plot(dense_t, mean_temperature + heat_capacity_2 / (heat_capacity_1 + heat_capacity_2) * dense_diff,
             color="#6b442f", lw=1.6, label="Closed-form $T_1$")
axes[0].plot(dense_t, mean_temperature - heat_capacity_1 / (heat_capacity_1 + heat_capacity_2) * dense_diff,
             color="#39715E", lw=1.6, label="Closed-form $T_2$")
axes[0].plot(times[::4], temperatures_1[::4], "o", color="#c56838", ms=4, label="OneSimulation body 1")
axes[0].plot(times[::4], temperatures_2[::4], "s", color="#83B9AE", ms=4, label="OneSimulation body 2")
axes[0].axhline(mean_temperature, color="#9a8f7d", ls="--", lw=1.0, label="Capacity-weighted mean")
axes[0].set(ylabel="Temperature / K", title="Two thermal masses exchanging heat")
axes[0].legend(loc="center right", fontsize=9)
axes[1].semilogy(dense_t, dense_diff, color="#6b442f", lw=1.6, label=r"$\Delta T_0\,e^{-G(1/C_1+1/C_2)t}$")
axes[1].semilogy(times[::4], difference[::4], "o", color="#c56838", ms=4, label="OneSimulation $T_1-T_2$")
axes[1].set(xlabel="Time / s", ylabel="Temperature difference / K")
axes[1].legend(loc="upper right")
plt.show()


# ## Going further
#
# Make the two heat capacities equal and confirm that the final temperature is
# the arithmetic mean. Give one body a constant `heating` power and the pair
# then warms without bound at the rate $P/(C_1+C_2)$ on top of the decaying
# difference; or add a third body to build a small thermal network. In each case
# write down the new reference before changing the checks.
