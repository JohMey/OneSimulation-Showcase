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

import numpy as np

from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.chemistry import BatteryChemistry, LumpedBattery
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()


# ## Parameters and model
#
# The parameters can be changed. The reference formulas below stay valid as long as
# the current, the resistance and the slope of the open-circuit curve are constant
# and the cell starts at the ambient temperature. The values describe a small
# 2 Ah lithium-ion cell discharged at 1C (2 A) from 90 % to 10 % state of charge.

capacity_ampere_hours = 2.0
discharge_current = 2.0
ocv_offset = 3.2
ocv_slope = 0.8
internal_resistance = 0.05
thermal_capacity = 60.0
heat_transfer = 0.24
ambient_temperature = 298.15
initial_soc = 0.9
final_soc = 0.1

charge_capacity = capacity_ampere_hours * 3600.0
end_time = (initial_soc - final_soc) * charge_capacity / discharge_current
time_step = end_time / 288
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
times = sink.times


soc = sink.time_series(model.state_of_charge)[1]
temperatures = sink.time_series(model.temperature)[1]
terminal_voltages = sink.time_series(model.terminal_voltage)[1]
ohmic_heats = sink.time_series(model.ohmic_heat)[1]


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

open_circuit = ocv_offset + ocv_slope * soc
ohmic_drop = open_circuit - terminal_voltages
print(f"Ohmic drop I R = {np.mean(ohmic_drop) * 1e3:.2f} mV,  ohmic heat I^2 R = {np.mean(ohmic_heats):.3f} W")


# Thermal time constant from the approach to the steady excess, log(1 - dT/dT_inf) = -t/tau,
# using the first part of the trace where the remaining gap is well resolved.
window = times < 2.0 * thermal_time_constant
gap = 1.0 - (temperatures[window][1:] - ambient_temperature) / steady_excess
measured_tau = -1.0 / np.polyfit(times[window][1:], np.log(gap), 1)[0]
print(f"Thermal time constant: measured {measured_tau:.2f} s, exact {thermal_time_constant:.2f} s")


# ## Figure

dense_t = np.linspace(0.0, end_time, 400)
dense_soc = initial_soc - discharge_current * dense_t / charge_capacity
fig = plots.figure(2, 1, size=(8, 7))
plots.line(dense_t / 60, {"open-circuit voltage": ocv_offset + ocv_slope * dense_soc},
           points={"OneSimulation RK4": (times[::8] / 60, terminal_voltages[::8])},
           reference={"closed-form terminal voltage": ocv_offset + ocv_slope * dense_soc - discharge_current * internal_resistance},
           ylabel="voltage [V]", title="Constant-current discharge of a lumped battery cell", ax=fig[0])
plots.line(points={"OneSimulation RK4": (times[::8] / 60, temperatures[::8])},
           reference={"Joule-heating exponential": (dense_t / 60, ambient_temperature + steady_excess
                                                    * (1 - np.exp(-dense_t / thermal_time_constant)))},
           hlines={"steady temperature": ambient_temperature + steady_excess},
           xlabel="time [min]", ylabel="cell temperature [K]", legend="lower right", ax=fig[1])
plots.show()


# ## Going further
#
# Replace the linear open-circuit curve by a tabulated or polynomial one; the
# coulomb-counting line is unchanged, and the terminal voltage still equals
# $U_\mathrm{oc}(s)-IR$ at every stored time. A temperature-dependent resistance
# couples the two states and the thermal reference must then be integrated
# numerically. A time-dependent `current` function turns the constant discharge
# into a drive cycle; passing `current=None` lets a circuit model, for example a
# series RLC load, supply the current through a field coupling.
