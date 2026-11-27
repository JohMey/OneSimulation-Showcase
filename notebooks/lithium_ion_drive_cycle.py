"""OneSimulation example: monitoring a lithium-ion cell over a hybrid-vehicle drive cycle in 1D.
Requires the installed, activated OneSimulation runtime.

A graphite-SiOx | NMC811 cell is driven by a charge-sustaining hybrid-vehicle current profile with
the Doyle-Fuller-Newman model. State of charge, voltage, salt depletion and the negative-electrode
potential that governs lithium plating are monitored and checked against coulomb counting and the
relaxed open-circuit voltage.
"""

# # Monitoring a lithium-ion cell over a hybrid-vehicle drive cycle
#
# In a hybrid vehicle the battery supplies short bursts of power for acceleration and takes back
# energy during regenerative braking, while the engine keeps the state of charge in a narrow window.
# The cell sees pulses of several C in both directions. Besides voltage and state of charge, a
# battery management system cares about quantities that are only visible inside a physical model:
# the salt concentration in the electrolyte, and the potential of the negative electrode against
# lithium, $\phi_s-\phi_e$, at the separator. During fast charging pulses this potential drops; when
# it falls below 0 V, metallic lithium can plate on the graphite.
#
# The cell is the Doyle-Fuller-Newman model of the discharge-rest-charge example. The current profile
# is a synthetic charge-sustaining microcycle of 400 s: idle, acceleration bursts of up to 2.5C,
# cruising and regenerative braking pulses of up to 2C, then 65 s of engine charging whose current
# (about 2.6C) is set so that the net charge of the microcycle is zero. The microcycle is repeated
# three times and followed by 30 min of rest.
#
# **Reference.** Cell parameters of Chen et al., *J. Electrochem. Soc.* 167 (2020) 080534 (LG M50,
# 5 Ah); model of Doyle, Fuller and Newman, *J. Electrochem. Soc.* 140 (1993) 1526. Checks: (i) the
# state of charge from coulomb counting, $s(t)=s_0-\int I\,dt/Q$, equals the mean stoichiometry of
# both electrodes; (ii) after a cycle with zero net charge both electrodes return to their initial
# mean stoichiometry; (iii) during the final rest the voltage relaxes to the open-circuit voltage of
# the initial state.

import numpy as np
from onesim.physics.chemistry import (
    BatteryElectrodeLayer,
    BatteryElectrolyteLayer,
    PorousBatteryBDF,
    PorousBatteryFiniteVolume,
    PorousBatteryMesh,
    PorousElectrodeBattery,
)
from onesim.visualization import plots, theme

theme.set()

# ## Cell at 60 % state of charge

faraday, area, soc0 = 96485.33212, 0.1027, 0.6
negative_ocp = lambda x, T: (
    1.9793 * np.exp(-39.3631 * x)
    + 0.2482
    - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
    - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
    - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
)
positive_ocp = lambda y, T: (
    4.4875
    - 0.8090 * y
    - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
    - 17.7326 * np.tanh(15.7890 * (y - 0.3117))
    + 17.5842 * np.tanh(15.9308 * (y - 0.3120))
)
negative = BatteryElectrodeLayer(
    thickness=85.2e-6,
    porosity=0.25,
    electrolyte_diffusivity=1.7694e-10 * 0.25**1.5,
    electrolyte_conductivity=0.9487 * 0.25**1.5,
    particle_radius=5.86e-6,
    particle_diffusivity=3.3e-14,
    active_fraction=0.75,
    maximum_concentration=33133,
    initial_stoichiometry=0.0279 + soc0 * (0.9014 - 0.0279),
    reference_exchange_current=6.48e-7 * np.sqrt(1000) * 33133 / 2,
    electronic_conductivity=215,
    open_circuit_potential=negative_ocp,
)
positive = BatteryElectrodeLayer(
    thickness=75.6e-6,
    porosity=0.335,
    electrolyte_diffusivity=1.7694e-10 * 0.335**1.5,
    electrolyte_conductivity=0.9487 * 0.335**1.5,
    particle_radius=5.22e-6,
    particle_diffusivity=4e-15,
    active_fraction=0.665,
    maximum_concentration=63104,
    initial_stoichiometry=0.9077 + soc0 * (0.2661 - 0.9077),
    reference_exchange_current=3.42e-6 * np.sqrt(1000) * 63104 / 2,
    electronic_conductivity=0.18,
    open_circuit_potential=positive_ocp,
)
separator = BatteryElectrolyteLayer(
    thickness=12e-6,
    porosity=0.47,
    electrolyte_diffusivity=1.7694e-10 * 0.47**1.5,
    electrolyte_conductivity=0.9487 * 0.47**1.5,
)
cell = PorousElectrodeBattery(
    negative=negative,
    separator=separator,
    positive=positive,
    electrolyte_concentration=1000,
    transference_number=0.2594,
)
capacity = (
    faraday
    * negative.active_fraction
    * negative.thickness
    * negative.maximum_concentration
    * (0.9014 - 0.0279)
    * area
)

# ## Drive cycle
#
# Segments of the microcycle as (start time in s, C-rate), positive on discharge.
#
# The switch times are passed as breakpoints, so the integrator restarts exactly at every change of
# the current; the pulse voltage window 2.5-4.3 V of the cell is set as cut-off and would stop the run if
# the cycle left it. Particles with 32 shells, layers with 20, 10 and 20 control volumes.

segments = np.array(
    [
        (0, 0.0),
        (20, 2.5),
        (35, 1.0),
        (75, -2.0),
        (85, 0.0),
        (100, 2.0),
        (112, 0.8),
        (172, -1.5),
        (186, 0.0),
        (200, 1.5),
        (230, 0.5),
        (290, -1.0),
        (305, -0.5),
        (370, 0.0),
    ]
)
period, cycles = 400.0, 3
segments[-2, 1] -= (np.diff(np.r_[segments[:, 0], period]) @ segments[:, 1]) / (
    370 - 305
)
profile = (
    lambda t: 5.0
    / area
    * segments[np.searchsorted(segments[:, 0], t % period, side="right") - 1, 1]
    * (t < cycles * period)
)
end = cycles * period + 1800.0
times = np.r_[
    np.arange(0.0, cycles * period, 1.0), np.linspace(cycles * period, end, 181)
]
switches = np.r_[
    (np.arange(cycles)[:, None] * period + segments[:, 0]).ravel(), cycles * period, end
]
run = cell.solve(
    mode="DFN",
    current_density=profile,
    final_time=end,
    mesh=PorousBatteryMesh.from_model(cell, 32, (20, 10, 20)),
    spatial=PorousBatteryFiniteVolume(tolerance=1e-12, residual_limit=2e-8),
    time_integrator=PorousBatteryBDF(relative_tolerance=1e-8, absolute_tolerance=1e-9),
    output_times=times,
    breakpoints=switches,
    cutoff_voltage=(2.5, 4.3),
)

# ## Monitoring and checks

charge = np.r_[0.0, np.cumsum(np.diff(times) * run.current_density[:-1])] * area
coulomb = soc0 - charge / capacity
mean = [
    np.mean(c @ np.diff(np.linspace(0, 1, 33) ** 3), axis=1)
    / layer.maximum_concentration
    for c, layer in zip(run.particle_concentration, (negative, positive))
]
soc_negative = (
    soc0
    - (negative.initial_stoichiometry - mean[0])
    * faraday
    * area
    * 0.75
    * 85.2e-6
    * 33133
    / capacity
)
soc_positive = (
    soc0
    - (mean[1] - positive.initial_stoichiometry)
    * faraday
    * area
    * 0.665
    * 75.6e-6
    * 63104
    / capacity
)
plating = run.solid_potential[:, 19] - run.electrolyte_potential[:, 19]
relaxed = positive_ocp(positive.initial_stoichiometry, 0) - negative_ocp(
    negative.initial_stoichiometry, 0
)
driving = times < cycles * period
print(f"run ended by: {run.termination_reason}")
print(
    f"net charge of one microcycle: {np.diff(np.r_[segments[:, 0], period]) @ segments[:, 1] * 5.0:.2e} A s; "
    f"rms current {area * np.sqrt(np.mean(run.current_density[driving] ** 2)):.2f} A, peaks {area * run.current_density.max():.1f} A "
    f"and {area * run.current_density.min():.1f} A"
)
print(
    f"state of charge from the particles against coulomb counting: largest difference {np.abs(soc_negative - coulomb).max():.1e} (negative), "
    f"{np.abs(soc_positive - coulomb).max():.1e} (positive)"
)
print(
    f"after {cycles} cycles: mean stoichiometries {mean[0][-1]:.7f} and {mean[1][-1]:.7f}, start {negative.initial_stoichiometry:.7f} and "
    f"{positive.initial_stoichiometry:.7f}"
)
print(
    f"state-of-charge window while driving {coulomb[driving].min():.4f} - {coulomb[driving].max():.4f}; voltage window "
    f"{run.terminal_voltage[driving].min():.3f} - {run.terminal_voltage[driving].max():.3f} V"
)
print(
    f"lowest salt concentration {run.electrolyte_concentration.min():.0f} mol/m³; lowest negative-electrode potential at the separator "
    f"{1e3 * plating.min():.1f} mV vs Li (plating below 0 mV)"
)
for minutes in (1, 5, 30):
    k = np.searchsorted(times, cycles * period + 60 * minutes)
    print(
        f"{minutes:2d} min into the rest: V - U_oc(initial) = {1e3 * (run.terminal_voltage[k] - relaxed):+.2f} mV"
    )
print(
    f"energy delivered {np.trapezoid(np.maximum(run.current_density, 0) * run.terminal_voltage, times) * area / 3600:.3f} Wh, "
    f"taken back {-np.trapezoid(np.minimum(run.current_density, 0) * run.terminal_voltage, times) * area / 3600:.3f} Wh"
)

# ## Figure

fig = plots.figure(2, 2, size=(15, 9.5))
plots.line(
    times / 60,
    {"terminal voltage": run.terminal_voltage},
    right={"current [A]": area * run.current_density},
    hlines={"open-circuit voltage at start": relaxed},
    xlabel="time [min]",
    ylabel="voltage [V]",
    right_label="current [A]",
    title="Voltage and current",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    times / 60,
    {"negative particles": soc_negative},
    reference={"coulomb counting": coulomb},
    points={"positive particles": (times[::40] / 60, soc_positive[::40])},
    xlabel="time [min]",
    ylabel="state of charge",
    title="State of charge",
    legend="upper right",
    ax=fig[1],
)
plots.line(
    times / 60,
    {"lowest salt concentration": run.electrolyte_concentration.min(axis=1)},
    right={"highest salt concentration": run.electrolyte_concentration.max(axis=1)},
    xlabel="time [min]",
    ylabel="lowest salt [mol/m³]",
    right_label="highest salt [mol/m³]",
    title="Electrolyte",
    legend="center right",
    ax=fig[2],
)
plots.line(
    times[:400],
    {"negative potential at the separator": 1e3 * plating[:400]},
    right={"current [A]": area * run.current_density[:400]},
    hlines={"plating threshold": 0.0},
    xlabel="time [s]",
    ylabel="φs - φe [mV vs Li]",
    right_label="current [A]",
    title="Plating margin during the first microcycle",
    legend="upper right",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# The engine-charging phase uses up the plating margin and nearly empties the electrolyte near the
# positive collector. Spread the same charge over a longer, gentler phase, or lower the temperature
# through the exchange current and the diffusivities, and watch both margins change.
