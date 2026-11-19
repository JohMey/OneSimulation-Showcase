"""OneSimulation example: equivalent-circuit model of a nickel-metal hydride battery cell.
Requires the installed, activated OneSimulation runtime.

A NiMH cell is described by a state-of-charge dependent voltage source, a series resistance, one
resistor-capacitor branch and an Arrhenius self-discharge. A pulse discharge, the impedance spectrum
and a storage test at three temperatures are compared with the closed-form responses of the
circuit.
"""

# # Equivalent-circuit model of a nickel-metal hydride cell
#
# Equivalent-circuit models describe a cell by what is measured at its terminals. Here a voltage
# source $U_\mathrm{oc}(s)$ that depends on the state of charge $s$, a series resistance $R_0$ and
# one RC branch for the charge-transfer and diffusion polarisation $u_1$:
#
# $$Q\frac{ds}{dt}=-I-I_\mathrm{sd}(T),\qquad C_1\frac{du_1}{dt}=I-\frac{u_1}{R_1},\qquad V=U_\mathrm{oc}(s)-R_0I-u_1 .$$
#
# NiMH cells lose charge on the shelf through internal side reactions; the self-discharge current
# follows an Arrhenius law $I_\mathrm{sd}=k_\mathrm{ref}\,Q\,\exp\!\big[-\tfrac{E_a}{R}\big(\tfrac1T-\tfrac1{T_\mathrm{ref}}\big)\big]$.
# The open-circuit curve is a Nernst-type curve around 1.28 V.
#
# In OneSimulation the cell is a lumped battery (state of charge, voltage source, series resistance,
# self-discharge as internal charge loss) and the RC branch is an electrical circuit driven by the
# same current; the branch voltage enters the cell as an additional voltage drop. The small-signal
# impedance of the same parameters is a battery impedance model.
#
# **Reference.** The closed-form solution of the linear circuit: for a piecewise constant current
# with steps $\Delta I_k$ at $t_k$, $u_1(t)=\sum_k\Delta I_kR_1\big(1-e^{-(t-t_k)/\tau_1}\big)$ with
# $\tau_1=R_1C_1$; the impedance $Z(\omega)=R_0+R_1/(1+i\omega\tau_1)$; and the linear loss of state of
# charge $I_\mathrm{sd}t/Q$ in storage. Model structure and parameter ranges as in G. L. Plett,
# *Battery Management Systems, Vol. I: Battery Modeling* (Artech House, 2015).

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.chemistry import (
    BatteryChemistry,
    BatteryImpedance,
    BatteryRelaxationBranch,
    LumpedBattery,
)
from onesim.physics.systems import (
    Circuit,
    CircuitCapacitor,
    CircuitCurrentSource,
    CircuitResistor,
)
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Cell parameters
#
# A 2.5 Ah cell; self-discharge of 20 % per month at 25 °C with an activation energy of 35 kJ/mol.

capacity, r0, r1, tau1 = 2.5 * 3600, 0.020, 0.015, 40.0
k_ref, activation, gas = 0.20 / (30 * 86400), 35e3, 8.314462618
open_circuit = lambda s: 1.28 + 0.08 * (s - 0.5) + 0.012 * np.log(s / (1 - s))
self_discharge = (
    lambda T: k_ref * capacity * np.exp(-activation / gas * (1 / T - 1 / 298.15))
)
chemistry = BatteryChemistry(
    "nimh",
    open_circuit_voltage=lambda s, T: open_circuit(s),
    internal_resistance=lambda s, T: r0,
)
impedance = BatteryImpedance(
    series_resistance=r0,
    branches=(BatteryRelaxationBranch(resistance=r1, relaxation_time=tau1),),
)

# ## Pulse discharge
#
# From 90 % state of charge: 10 min at 1C, 10 min rest, 5 min at 2C, 15 min rest, 2 min of 1C charge
# and 8 min rest, at 25 °C. The circuit is integrated with the trapezoidal rule at 0.1 s steps.

steps = np.array(
    [(0, 2.5), (600, 0.0), (1200, 5.0), (1500, 0.0), (2400, -2.5), (2520, 0.0)]
)
load = lambda t: steps[np.searchsorted(steps[:, 0], t, side="right") - 1, 1]
times = np.arange(0.0, 3001.0, 1.0)
branch = Circuit(
    components=(
        CircuitCurrentSource(name="load", positive="0", negative="rc", current=load),
        CircuitResistor(name="r1", positive="rc", negative="0", resistance=r1),
        CircuitCapacitor(
            name="c1",
            positive="rc",
            negative="0",
            capacitance=tau1 / r1,
            initial_voltage=0.0,
        ),
    )
).solve(times=times, max_step=0.1)
polarisation = branch.voltages["rc"]

cell = LumpedDomain()
model = LumpedBattery(
    domain=cell.system,
    chemistry=chemistry,
    charge_capacity=capacity,
    current=load,
    thermal_capacity=40.0,
    heat_transfer=0.05,
    ambient_temperature=298.15,
    effects={
        "internal_charge_loss": self_discharge(298.15),
        "additional_voltage_drop": lambda t: np.interp(t, times, polarisation),
    },
)
problem = Problem(
    name="NiMH pulse test",
    geometry=cell,
    physics=(model,),
    initial_conditions=(
        InitialValue(model.state_of_charge, 0.9),
        InitialValue(model.temperature, 298.15),
    ),
)
numerics = Numerics(spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4())
record = MemoryOutputSink()
Simulation(problem=problem, mesh=LumpedMesh(cell), numerics=numerics).run(
    start_time=0.0,
    end_time=3000.0,
    time_step=1.0,
    output=Output(
        fields=(model.state_of_charge, model.terminal_voltage, model.temperature),
        interval=1.0,
        sink=record,
    ),
)
soc, voltage, temperature = (
    record.time_series(field)[1]
    for field in (model.state_of_charge, model.terminal_voltage, model.temperature)
)

# ## Impedance spectrum and storage
#
# The circuit is driven by a sinusoidal current of 0.1 A at five frequencies, starting on the
# periodic solution; amplitude and phase of the response give the impedance. The storage test keeps
# the cell at rest for 30 days at 0, 25 and 45 °C.

frequencies = np.array([1e-3, 4e-3, 1e-2, 4e-2, 1e-1])
sine = {
    f: Circuit(
        components=(
            CircuitCurrentSource(
                name="load",
                positive="0",
                negative="rc",
                current=lambda t, f=f: 0.1 * np.cos(2 * np.pi * f * t),
            ),
            CircuitResistor(name="r1", positive="rc", negative="0", resistance=r1),
            CircuitCapacitor(
                name="c1",
                positive="rc",
                negative="0",
                capacitance=tau1 / r1,
                initial_voltage=(0.1 * r1 / (1 + 2j * np.pi * f * tau1)).real,
            ),
        )
    ).solve(times=np.linspace(0, 1 / f, 401), max_step=1 / (400 * f))
    for f in frequencies
}
measured = {
    f: r0
    + 2
    * np.mean(run.voltages["rc"][:-1] * np.exp(-2j * np.pi * f * run.times[:-1]))
    / 0.1
    for f, run in sine.items()
}
storage = {}
for celsius in (0, 25, 45):
    shelf = LumpedBattery(
        domain=cell.system,
        chemistry=chemistry,
        charge_capacity=capacity,
        current=0.0,
        thermal_capacity=40.0,
        heat_transfer=0.05,
        ambient_temperature=273.15 + celsius,
        effects={"internal_charge_loss": self_discharge(273.15 + celsius)},
    )
    shelf_record = MemoryOutputSink()
    Simulation(
        problem=Problem(
            name="NiMH storage",
            geometry=cell,
            physics=(shelf,),
            initial_conditions=(
                InitialValue(shelf.state_of_charge, 0.9),
                InitialValue(shelf.temperature, 273.15 + celsius),
            ),
        ),
        mesh=LumpedMesh(cell),
        numerics=numerics,
    ).run(
        start_time=0.0,
        end_time=30 * 86400.0,
        time_step=3600.0,
        output=Output(
            fields=(shelf.state_of_charge, shelf.terminal_voltage),
            interval=86400.0,
            sink=shelf_record,
        ),
    )
    storage[celsius] = (
        shelf_record.times,
        shelf_record.time_series(shelf.state_of_charge)[1],
        shelf_record.time_series(shelf.terminal_voltage)[1],
    )

# ## Results

exact_rc = sum(
    np.where(times >= t_k, dI * r1 * (1 - np.exp(-(times - t_k) / tau1)), 0.0)
    for t_k, dI in zip(steps[:, 0], np.diff(np.r_[0.0, steps[:, 1]]))
)
charge = np.r_[0.0, np.cumsum([load(t) for t in times[:-1]])]
exact_soc = 0.9 - (charge + self_discharge(298.15) * times) / capacity
exact_voltage = (
    open_circuit(exact_soc) - r0 * np.array([load(t) for t in times]) - exact_rc
)
print(
    f"branch voltage against the closed form: largest deviation {1e3 * np.abs(polarisation - exact_rc).max():.3f} mV "
    f"(peak {1e3 * polarisation.max():.1f} mV); KCL residual {np.abs(branch.kcl_residual).max():.1e} A"
)
print(
    f"state of charge against coulomb counting: {np.abs(soc - exact_soc[record.times.astype(int)]).max():.1e}; "
    f"terminal voltage against the closed form: {1e3 * np.abs(voltage - exact_voltage[record.times.astype(int)]).max():.3f} mV"
)
print(f"temperature rise in the 2C pulse {temperature.max() - 298.15:.3f} K")
print(
    "frequency [Hz]   Z from the circuit [mΩ]      Z of the impedance model [mΩ]   |difference| [mΩ]"
)
for f, z in measured.items():
    exact = impedance.spectrum(f)
    print(
        f"{f:14.3g}   {1e3 * z.real:8.3f} {1e3 * z.imag:+8.3f}i       {1e3 * exact.real:8.3f} {1e3 * exact.imag:+8.3f}i   {1e3 * abs(z - exact):12.2e}"
    )
for celsius, (t, s, v) in storage.items():
    print(
        f"storage at {celsius:2d} °C: state of charge after 30 days {s[-1]:.4f}, closed form {0.9 - self_discharge(273.15 + celsius) * t[-1] / capacity:.4f}, "
        f"open-circuit voltage {v[0]:.4f} -> {v[-1]:.4f} V"
    )

# ## Figure

fig = plots.figure(2, 2, size=(14, 10))
plots.line(
    times / 60,
    {"simulation": voltage},
    reference={"closed form": exact_voltage},
    right={"current [A]": np.array([load(t) for t in times])},
    xlabel="time [min]",
    ylabel="terminal voltage [V]",
    right_label="current [A]",
    title="Pulse test at 25 °C",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    times / 60,
    {"circuit": 1e3 * polarisation},
    reference={"closed form": 1e3 * exact_rc},
    xlabel="time [min]",
    ylabel="RC-branch voltage u₁ [mV]",
    title="Polarisation of the RC branch",
    legend="upper right",
    ax=fig[1],
)
spectrum = impedance.spectrum(np.geomspace(1e-4, 10, 300))
plots.line(
    1e3 * spectrum.real,
    {"impedance model": -1e3 * spectrum.imag},
    points={
        "time-domain circuit": (
            1e3 * np.array([z.real for z in measured.values()]),
            -1e3 * np.array([z.imag for z in measured.values()]),
        )
    },
    xlabel="Re Z [mΩ]",
    ylabel="-Im Z [mΩ]",
    title="Impedance (Nyquist)",
    legend="upper right",
    ax=fig[2],
)
plots.line(
    points={
        f"{celsius} °C": (t[::2] / 86400, s[::2])
        for celsius, (t, s, v) in storage.items()
    },
    reference={
        f"{celsius} °C": (
            t / 86400,
            0.9 - self_discharge(273.15 + celsius) * t / capacity,
        )
        for celsius, (t, s, v) in storage.items()
    },
    xlabel="storage time [days]",
    ylabel="state of charge",
    title="Self-discharge on the shelf (dashed: closed form)",
    legend="lower left",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Add a second RC branch for the slow diffusion, let the self-discharge depend on the state of
# charge, or fit $R_0$, $R_1$ and $\tau_1$ to a measured spectrum with the impedance model.
