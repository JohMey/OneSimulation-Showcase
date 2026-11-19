"""OneSimulation example: power losses of a lithium-ion cell in a hybrid pulse power test.
Requires the installed, activated OneSimulation runtime.

A graphite-SiOx | NMC811 cell runs through a hybrid pulse power characterisation test at three
states of charge with the Doyle-Fuller-Newman model. The dissipated power is split into electrolyte
conduction, salt concentration, electrode conduction, reaction and particle diffusion losses, and
the parts are checked against the terminal energy loss of the cell.
"""

# # Power losses of a lithium-ion cell in a hybrid pulse power test
#
# The hybrid pulse power characterisation (HPPC) test probes a cell with a 10 s discharge pulse,
# 40 s rest and a 10 s regenerative charge pulse at 75 % of the discharge current, repeated at
# several states of charge. Every watt the cell does not deliver is turned into heat somewhere
# inside it. With the Doyle-Fuller-Newman model (equations in the discharge-rest-charge example) the
# loss of terminal power against the reversible power of the mean electrode stoichiometries splits
# exactly into five parts:
#
# $$I\big(U_+(\bar\theta_+)-U_-(\bar\theta_-)\big)-IV=
#   \underbrace{\int\frac{i_e^2}{\kappa}dx}_{\text{electrolyte conduction}}
#   \underbrace{-\int i_e\frac{2(1-t_+)RT}{F}\frac{\partial\ln c_e}{\partial x}dx}_{\text{salt concentration}}
#   +\underbrace{\int\frac{i_s^2}{\sigma}dx}_{\text{electrode conduction}}
#   +\underbrace{\int a\,j\,\eta\,dx}_{\text{reaction}}
#   +\underbrace{\int a\,j\,\big(U(\theta^\mathrm{surf})-U(\bar\theta)\big)dx}_{\text{particle diffusion}},$$
#
# with the overpotential $\eta=\phi_s-\phi_e-U(\theta^\mathrm{surf})$. The identity follows from
# charge conservation, $\partial i_e/\partial x=a\,j$ and $i_s+i_e=I$, by integration by parts; on
# the finite-volume mesh the same summation by parts holds face by face.
#
# **Reference.** Test protocol of the *FreedomCAR Battery Test Manual for Power-Assist Hybrid
# Electric Vehicles* (INL, 2003); model of Doyle, Fuller and Newman, *J. Electrochem. Soc.* 140
# (1993) 1526; cell parameters of Chen et al., *J. Electrochem. Soc.* 167 (2020) 080534 (LG M50,
# 5 Ah). Check: the five loss parts add up to the terminal power loss at every output time, and
# their time integrals to the terminal energy loss of every pulse.

import numpy as np
from onesim.physics.chemistry import (
    BatteryElectrodeLayer,
    BatteryElectrolyteLayer,
    PorousElectrodeBattery,
)
from onesim.visualization import plots, theme
from scipy.integrate import solve_ivp

theme.set()

# ## Cell at 80 % state of charge

faraday, area, soc0 = 96485.33212, 0.1027, 0.8
thermal = 8.314462618 * 298.15 / faraday
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

# ## HPPC test
#
# Pulses of 1.5C (7.5 A) discharge and 1.125C charge at 80, 50 and 20 % state of charge; between the
# blocks the cell is discharged by 30 % at 1C and rests for 15 minutes.
#
# The current switches abruptly. The stiff BDF integrator (the default of the model) gets a maximum
# step of 2 s through the `time_solver` port, so that no pulse can fall between two steps.

one_c, step = 5.0 / area, 0.3 * 5.0904 / 5.0 * 3600
starts = np.array([0.0, 100 + step + 900, 2 * (100 + step + 900)])
table = np.array(
    sorted(
        [
            (s + offset, rate)
            for s in starts
            for offset, rate in ((0, 1.5), (10, 0), (50, -1.125), (60, 0))
        ]
        + [
            (s + offset, rate)
            for s in starts[:2]
            for offset, rate in ((100, 1.0), (100 + step, 0))
        ]
    )
)
program = lambda t: one_c * table[np.searchsorted(table[:, 0], t, side="right") - 1, 1]
end = starts[2] + 100
times = np.unique(
    np.r_[
        np.linspace(0, end, 1500),
        np.concatenate([s + np.linspace(0, 100, 501) for s in starts]),
    ]
)
counts = (10, 5, 10)
stepping = lambda rhs, span, start, output_times, jacobian_sparsity: solve_ivp(
    rhs,
    span,
    start,
    method="BDF",
    t_eval=output_times,
    jac_sparsity=jacobian_sparsity,
    rtol=1e-8,
    atol=1e-9,
    max_step=2.0,
)
run = cell.solve(
    mode="DFN",
    current_density=program,
    final_time=end,
    radial_cells=16,
    through_cells=counts,
    output_times=times,
    time_solver=stepping,
)

# ## Loss parts
#
# All parts are evaluated on the finite-volume faces and cells of the solution, in W per m² of
# electrode.

nn, ns, nq = counts
current, ionic, phi_e, salt = (
    run.current_density,
    run.electrolyte_current_density,
    run.electrolyte_potential,
    run.electrolyte_concentration,
)
concentration_loss = -(
    ionic[:, 1:-1]
    * 2
    * (1 - cell.transference_number)
    * thermal
    * np.diff(np.log(salt), axis=1)
).sum(axis=1)
conduction_loss = (
    -(ionic[:, 1:-1] * np.diff(phi_e, axis=1)).sum(axis=1) - concentration_loss
)
electrode_loss = (
    -(
        (current[:, None] - ionic[:, 1:nn])
        * np.diff(run.solid_potential[:, :nn], axis=1)
    ).sum(axis=1)
    - (
        (current[:, None] - ionic[:, nn + ns + 1 : -1])
        * np.diff(run.solid_potential[:, nn:], axis=1)
    ).sum(axis=1)
    + current**2
    * (
        negative.thickness / nn / negative.electronic_conductivity
        + positive.thickness / nq / positive.electronic_conductivity
    )
    / 2
)
width = np.repeat([negative.thickness / nn, positive.thickness / nq], (nn, nq))
surface_area = np.repeat(
    [
        3 * negative.active_fraction / negative.particle_radius,
        3 * positive.active_fraction / positive.particle_radius,
    ],
    (nn, nq),
)
surface_potential = np.c_[
    negative_ocp(
        run.particle_surface_concentration[0] / negative.maximum_concentration, 0
    ),
    positive_ocp(
        run.particle_surface_concentration[1] / positive.maximum_concentration, 0
    ),
]
mean = [
    np.mean(c @ np.diff(np.linspace(0, 1, 17) ** 3), axis=1)
    / layer.maximum_concentration
    for c, layer in zip(run.particle_concentration, (negative, positive))
]
mean_potential = np.repeat(
    np.c_[negative_ocp(mean[0], 0), positive_ocp(mean[1], 0)], (nn, nq), axis=1
)
reaction = width * surface_area * run.interfacial_current_density
reaction_loss = (
    reaction
    * (
        run.solid_potential
        - np.c_[phi_e[:, :nn], phi_e[:, nn + ns :]]
        - surface_potential
    )
).sum(axis=1)
particle_loss = (reaction * (surface_potential - mean_potential)).sum(axis=1)
parts = {
    "electrolyte conduction": conduction_loss,
    "salt concentration": concentration_loss,
    "electrode conduction": electrode_loss,
    "reaction": reaction_loss,
    "particle diffusion": particle_loss,
}
terminal_loss = current * (
    positive_ocp(mean[1], 0) - negative_ocp(mean[0], 0) - run.terminal_voltage
)

# ## Results

print(
    f"largest |terminal power loss - sum of the five parts| over the test: {np.abs(terminal_loss - sum(parts.values())).max():.1e} W/m² "
    f"(largest loss {terminal_loss.max():.1f} W/m²)"
)
pulses = {
    f"{label}\n{soc:.0%}": (s + offset, s + offset + 10)
    for s, soc in zip(starts, (0.8, 0.5, 0.2))
    for label, offset in (("discharge", 0.0), ("charge", 50.0))
}
energy = {
    name: {
        part: area
        * np.trapezoid(
            values[(times >= a) & (times <= b - 1e-9)],
            times[(times >= a) & (times <= b - 1e-9)],
        )
        for part, values in {**parts, "terminal": terminal_loss}.items()
    }
    for name, (a, b) in pulses.items()
}
print(
    "pulse                 terminal loss [J]   sum of parts [J]   "
    + "   ".join(f"{part:>11.11s}" for part in parts)
)
for name, e in energy.items():
    print(
        f"{name.replace('\n', ' at '):20s} {e['terminal']:15.4f} {sum(e[p] for p in parts):18.4f}   "
        + "   ".join(f"{100 * e[p] / e['terminal']:9.1f} %" for p in parts)
    )

# ## Figure

block = (times >= starts[1] - 5) & (times <= starts[1] + 95)
fig = plots.figure(1, 3, size=(19, 5.4))
plots.line(
    times / 60,
    {"terminal voltage": run.terminal_voltage},
    right={"current [A]": area * current},
    xlabel="time [min]",
    ylabel="voltage [V]",
    right_label="current [A]",
    title="HPPC test at 80, 50 and 20 % SOC",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    times[block] - starts[1],
    {part: area * values[block] for part, values in parts.items()},
    reference={
        "terminal power loss": (times[block] - starts[1], area * terminal_loss[block])
    },
    xlabel="time in the 50 % block [s]",
    ylabel="power loss [W]",
    title="Where the power goes",
    legend="upper right",
    ax=fig[1],
)
plots.bars(
    list(pulses),
    {part: [energy[name][part] for name in pulses] for part in parts},
    reference=[energy[name]["terminal"] for name in pulses],
    ylabel="energy loss in the 10 s pulse [J]",
    title="Loss parts per 10 s pulse",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# The loss parts are the local heat sources of a thermal model: add them per layer to find where a
# cell heats up during fast charging, or repeat the test at lower temperature with smaller exchange
# currents and diffusivities.
