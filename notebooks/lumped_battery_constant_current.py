"""OneSimulation example: constant-current discharge of a lumped battery cell.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form coulomb-counting,
ohmic-drop and Joule-heating solution of the equivalent-circuit cell.
"""


# # Constant-current discharge of a lumped battery cell
#
# The simplest useful battery model treats the cell as a single lumped state of
# charge $s$ behind an open-circuit voltage $U_\mathrm{oc}(s)$ and an internal
# resistance $R$. A constant discharge current $I>0$ drains the charge capacity $Q$
# (coulomb counting), the terminal voltage sits below the open-circuit value by the
# ohmic drop, and the dissipated power heats the cell, which in turn loses heat to the
# surroundings through a thermal conductance $h$:
#
# $$Q\,\frac{ds}{dt}=-I,\qquad U=U_\mathrm{oc}(s)-I\,R,\qquad
# C_\mathrm{th}\,\frac{dT}{dt}=I^2R-h\,(T-T_\infty).$$
#
# OneSimulation treats this as a zero-dimensional (lumped) model with two
# differential states, the state of charge and the temperature, and integrates them
# in time with the classical fourth-order Runge-Kutta method. The open-circuit
# voltage and the resistance are supplied as explicit functions of state of charge
# and temperature, so any cell chemistry can be described.
#
# **Reference.** With constant $I$, $R$ and a linear open-circuit curve
# $U_\mathrm{oc}(s)=U_0+k\,s$ every quantity has a closed form:
#
# $$s(t)=s_0-\frac{I\,t}{Q},\qquad U(t)=U_0+k\,s(t)-I\,R,\qquad
# T(t)=T_\infty+\frac{I^2R}{h}\left(1-e^{-h\,t/C_\mathrm{th}}\right)$$
#
# for a cell that starts at the ambient temperature. The state of charge falls
# linearly, the terminal voltage follows it with a constant offset, and the
# temperature settles exponentially with the thermal time constant
# $\tau=C_\mathrm{th}/h$ at the excess $\Delta T_\infty=I^2R/h$.

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
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.chemistry import BatteryChemistry, LumpedBattery
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The parameters can be changed. The reference formulas below stay valid as long as
# the current, the resistance and the slope of the open-circuit curve are constant
# and the cell starts at the ambient temperature. The values describe a small
# 2 Ah lithium-ion cell discharged at 1C (2 A) from 90 % to 10 % state of charge.

capacity_ampere_hours = 2.0      # Ah, nominal charge capacity
discharge_current = 2.0          # A, constant load current I (positive = discharge)
ocv_offset = 3.2                 # V, U_0: open-circuit voltage at zero state of charge
ocv_slope = 0.8                  # V, k: open-circuit voltage rise over the full range
internal_resistance = 0.05       # Ohm, R
thermal_capacity = 60.0          # J/K, C_th of the cell
heat_transfer = 0.24             # W/K, thermal conductance h to the surroundings
ambient_temperature = 298.15     # K, T_inf
initial_soc = 0.9                # state of charge at t = 0
final_soc = 0.1                  # state of charge at which the discharge is stopped

charge_capacity = capacity_ampere_hours * 3600.0        # C, Q
end_time = (initial_soc - final_soc) * charge_capacity / discharge_current
time_step = end_time / 288                              # 10 s for the default values
thermal_time_constant = thermal_capacity / heat_transfer
steady_excess = discharge_current**2 * internal_resistance / heat_transfer
print(f"capacity Q = {charge_capacity:.0f} C,  discharge time = {end_time:.0f} s")
print(f"thermal time constant = {thermal_time_constant:.0f} s,  steady temperature rise = {steady_excess:.3f} K")

chemistry = BatteryChemistry(
    "linear-ocv-cell",
    open_circuit_voltage=lambda soc, temperature: ocv_offset + ocv_slope * soc,
    internal_resistance=lambda soc, temperature: internal_resistance,
)
domain = LumpedDomain()
model = LumpedBattery(
    domain=domain.system,
    chemistry=chemistry,
    charge_capacity=charge_capacity,
    current=discharge_current,
    thermal_capacity=thermal_capacity,
    heat_transfer=heat_transfer,
    ambient_temperature=ambient_temperature,
)
simulation = Simulation(
    problem=Problem(
        name="Constant-current discharge of a lumped battery cell",
        geometry=domain,
        physics=(model,),
        initial_conditions=(
            InitialValue(model.state_of_charge, initial_soc),
            InitialValue(model.temperature, ambient_temperature),
        ),
    ),
    mesh=LumpedMesh(domain),
    numerics=Numerics(
        spatial=LumpedParameters(),
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
        fields=(model.state_of_charge, model.temperature, model.terminal_voltage, model.ohmic_heat),
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


soc = series(model.state_of_charge)
temperatures = series(model.temperature)
terminal_voltages = series(model.terminal_voltage)
ohmic_heats = series(model.ohmic_heat)


# ## Comparison with the closed-form solution
#
# We evaluate the coulomb-counting line, the ohmic voltage drop and the exponential
# temperature rise at every stored time. In addition we read the thermal time
# constant off the simulated temperature trace. A passing check shows that this
# numerical example agrees with the equivalent-circuit equations; it is not a
# validation against a measured cell.

reference_soc = initial_soc - discharge_current * times / charge_capacity
reference_voltage = ocv_offset + ocv_slope * reference_soc - discharge_current * internal_resistance
reference_temperature = ambient_temperature + steady_excess * (1.0 - np.exp(-times / thermal_time_constant))

max_soc_error = float(np.max(np.abs(soc - reference_soc)))
max_voltage_error = float(np.max(np.abs(terminal_voltages - reference_voltage)))
max_temperature_error = float(np.max(np.abs(temperatures - reference_temperature)))
print(f"Maximum state-of-charge error: {max_soc_error:.3e}")
print(f"Maximum terminal-voltage error: {max_voltage_error:.3e} V")
print(f"Maximum temperature error:     {max_temperature_error:.3e} K  (rise {steady_excess:.3f} K)")

# The ohmic drop is constant and equals I R; the ohmic heat equals I^2 R.
open_circuit = ocv_offset + ocv_slope * soc
ohmic_drop = open_circuit - terminal_voltages
print(f"Ohmic drop I R = {np.mean(ohmic_drop) * 1e3:.2f} mV,  ohmic heat I^2 R = {np.mean(ohmic_heats):.3f} W")

# Discharge slope read off the trace: ds/dt = -I/Q.

# Thermal time constant from the approach to the steady excess, log(1 - dT/dT_inf) = -t/tau,
# using the first part of the trace where the remaining gap is well resolved.
window = times < 2.0 * thermal_time_constant
gap = 1.0 - (temperatures[window][1:] - ambient_temperature) / steady_excess
measured_tau = -1.0 / np.polyfit(times[window][1:], np.log(gap), 1)[0]
print(f"Thermal time constant: measured {measured_tau:.2f} s, exact {thermal_time_constant:.2f} s")


# ## Figure

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
for ax in axes:
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
dense_t = np.linspace(0.0, end_time, 400)
dense_soc = initial_soc - discharge_current * dense_t / charge_capacity
axes[0].plot(dense_t / 60, ocv_offset + ocv_slope * dense_soc, "--", color="#9a8f7d", lw=1.0,
             label=r"Open-circuit voltage $U_\mathrm{oc}(s)$")
axes[0].plot(dense_t / 60, ocv_offset + ocv_slope * dense_soc - discharge_current * internal_resistance,
             color="#6b442f", lw=1.8, label="Closed-form terminal voltage")
axes[0].plot(times[::8] / 60, terminal_voltages[::8], "o", color="#c56838", ms=4, label="OneSimulation RK4")
axes[0].set(ylabel="Voltage / V", title="Constant-current discharge of a lumped battery cell")
axes[0].legend(loc="upper right")
axes[1].plot(dense_t / 60, ambient_temperature + steady_excess * (1 - np.exp(-dense_t / thermal_time_constant)),
             color="#6b442f", lw=1.8, label="Joule-heating exponential")
axes[1].plot(times[::8] / 60, temperatures[::8], "o", color="#c56838", ms=4, label="OneSimulation RK4")
axes[1].axhline(ambient_temperature + steady_excess, color="#9a8f7d", ls="--", lw=1.0,
                label=r"$T_\infty+I^2R/h$")
axes[1].set(xlabel="Time / min", ylabel="Cell temperature / K")
axes[1].legend(loc="lower right")
plt.show()


# ## Going further
#
# Replace the linear open-circuit curve by a tabulated or polynomial one; the
# coulomb-counting line is unchanged, and the terminal voltage still equals
# $U_\mathrm{oc}(s)-IR$ at every stored time. A temperature-dependent resistance
# couples the two states and the thermal reference must then be integrated
# numerically. A time-dependent `current` function turns the constant discharge
# into a drive cycle; passing `current=None` lets a circuit model, for example a
# series RLC load, supply the current through a field coupling.
