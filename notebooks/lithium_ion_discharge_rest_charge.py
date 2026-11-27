"""OneSimulation example: discharge, rest and recharge of an isothermal lithium-ion cell in 1D.
Requires the installed, activated OneSimulation runtime.

A graphite-SiOx | NMC811 cell with published parameters is discharged at 1C, left at rest and
recharged with the Doyle-Fuller-Newman porous-electrode model through the thickness of the cell.
Lithium conservation, coulomb counting, the relaxed open-circuit voltage and a mesh study are
checked against their exact values.
"""

# # Discharge, rest and recharge of a lithium-ion cell
#
# The cell is a sandwich of a porous negative electrode, a separator and a porous positive electrode
# between two current collectors. The Doyle-Fuller-Newman (DFN) model resolves the thickness
# coordinate $x$ and, at every $x$ in the electrodes, a spherical active particle with radius
# coordinate $r$:
#
# $$\frac{\partial c_s}{\partial t}=\frac{D_s}{r^2}\frac{\partial}{\partial r}\Big(r^2\frac{\partial c_s}{\partial r}\Big),\qquad
#   \varepsilon\frac{\partial c_e}{\partial t}=\frac{\partial}{\partial x}\Big(D_e\frac{\partial c_e}{\partial x}\Big)+\frac{(1-t_+)\,a\,j}{F},$$
#
# $$\frac{\partial i_e}{\partial x}=a\,j,\qquad i_e=-\kappa\frac{\partial\phi_e}{\partial x}+\frac{2(1-t_+)RT\kappa}{F}\frac{\partial\ln c_e}{\partial x},\qquad
#   i_s=-\sigma\frac{\partial\phi_s}{\partial x},\qquad j=2j_0\sinh\frac{F(\phi_s-\phi_e-U(c_s^\mathrm{surf}))}{2RT}.$$
#
# | boundary | condition |
# |---|---|
# | current collectors | $i_s=I$, $i_e=0$, no salt flux |
# | separator | $i_e=I$, no electrode phase |
# | particle centre / surface | symmetry / $-D_s\,\partial_r c_s = j/F$ |
#
# The cell starts at 90 % state of charge. The current program is 45 min at 1C (5 A), one hour at
# rest, 45 min of 1C charge and half an hour of rest. The cell is isothermal at 25 °C.
#
# **Reference.** Parameters, open-circuit curves and stoichiometry limits of the LG M50 cell are
# those of Chen et al., *J. Electrochem. Soc.* 167 (2020) 080534; the model is that of Doyle,
# Fuller and Newman, *J. Electrochem. Soc.* 140 (1993) 1526. Independent checks: (i) lithium in
# particles and electrolyte is conserved; (ii) coulomb counting fixes the mean stoichiometry of each
# electrode, $\Delta\bar\theta=-\int I\,dt/(F\varepsilon_sLc_{s,\max}A)$; (iii) at rest the terminal
# voltage relaxes to the open-circuit voltage of the mean stoichiometries,
# $U_+(\bar\theta_+)-U_-(\bar\theta_-)$; (iv) the voltage converges under mesh refinement.

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

# ## Cell
#
# Effective electrolyte transport follows Bruggeman's relation with exponent 1.5; the electrode area
# is 0.1027 m², so 1C is 48.7 A/m².

faraday, area = 96485.33212, 0.1027
one_c = 5.0 / area
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
    initial_stoichiometry=0.0279 + 0.9 * (0.9014 - 0.0279),
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
    initial_stoichiometry=0.9077 + 0.9 * (0.2661 - 0.9077),
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

# ## Simulation
#
# The current density is a function of time (positive on discharge); its switch times are passed as
# breakpoints, so the integrator restarts exactly at every switch. The particles are divided into 32
# spherical shells, the layers into 20, 10 and 20 control volumes. A mesh study repeats the 45 min
# discharge with 8 to 64 particle shells and with 5 to 40 control volumes in each electrode.

program = lambda t: one_c * (float(t < 2700) - float(6300 <= t < 9000))
times = np.linspace(0.0, 10800.0, 1081)
run = cell.solve(
    mode="DFN",
    current_density=program,
    final_time=10800.0,
    mesh=PorousBatteryMesh.from_model(cell, 32, (20, 10, 20)),
    spatial=PorousBatteryFiniteVolume(tolerance=1e-12, residual_limit=2e-8),
    time_integrator=PorousBatteryBDF(relative_tolerance=1e-8, absolute_tolerance=1e-9),
    output_times=times,
    breakpoints=(0.0, 2700.0, 6300.0, 9000.0, 10800.0),
)
radial = {
    n: cell.solve(
        mode="DFN",
        current_density=one_c,
        final_time=2700.0,
        mesh=PorousBatteryMesh.from_model(cell, n, (20, 10, 20)),
        spatial=PorousBatteryFiniteVolume(tolerance=1e-12, residual_limit=2e-8),
        time_integrator=PorousBatteryBDF(relative_tolerance=1e-8, absolute_tolerance=1e-9),
        output_times=(0.0, 2700.0),
    ).terminal_voltage[-1]
    for n in (8, 16, 32, 64)
}
layered = {
    n: cell.solve(
        mode="DFN",
        current_density=one_c,
        final_time=2700.0,
        mesh=PorousBatteryMesh.from_model(cell, 32, (n, n // 2, n)),
        spatial=PorousBatteryFiniteVolume(tolerance=1e-12, residual_limit=2e-8),
        time_integrator=PorousBatteryBDF(relative_tolerance=1e-8, absolute_tolerance=1e-9),
        output_times=(0.0, 2700.0),
    ).terminal_voltage[-1]
    for n in (5, 10, 20, 40)
}

# ## Results

mean = [
    np.mean(c @ np.diff(np.linspace(0, 1, 33) ** 3), axis=1)
    / layer.maximum_concentration
    for c, layer in zip(run.particle_concentration, (negative, positive))
]
relaxed = positive_ocp(mean[1], 0) - negative_ocp(mean[0], 0)
counted = (
    one_c
    * 2700
    / (
        faraday
        * np.array(
            [
                layer.active_fraction * layer.thickness * layer.maximum_concentration
                for layer in (negative, positive)
            ]
        )
    )
)
ends = {
    "end of discharge": 269,
    "end of first rest": 629,
    "end of charge": 899,
    "end of program": 1080,
}
print(
    f"lithium inventory: largest relative drift {np.ptp(run.lithium_inventory) / run.lithium_inventory[0]:.1e}"
)
print(
    f"coulomb counting after 45 min at 1C: mean stoichiometry change negative {mean[0][0] - mean[0][270]:.6f} "
    f"(exact {counted[0]:.6f}), positive {mean[1][270] - mean[1][0]:.6f} (exact {counted[1]:.6f})"
)
print(
    f"after the full program: mean stoichiometries {mean[0][-1]:.6f} and {mean[1][-1]:.6f} (start {negative.initial_stoichiometry:.6f} and {positive.initial_stoichiometry:.6f})"
)
for name, k in ends.items():
    print(
        f"{name:18s} t = {times[k] / 60:5.0f} min   V = {run.terminal_voltage[k]:.4f} V   "
        f"open-circuit voltage of the mean stoichiometries {relaxed[k]:.4f} V   difference {1e3 * (run.terminal_voltage[k] - relaxed[k]):+.2f} mV"
    )
errors = {
    "particle shells": {
        1 / n: abs(v - radial[64]) for n, v in radial.items() if n < 64
    },
    "electrode cells": {
        1 / n: abs(v - layered[40]) for n, v in layered.items() if n < 40
    },
}
print(
    "voltage at the end of discharge: "
    + ", ".join(f"{v:.5f} V ({n} shells)" for n, v in radial.items())
    + "; "
    + ", ".join(f"{v:.5f} V ({n} cells)" for n, v in layered.items())
)
for name, e in errors.items():
    h = sorted(e)
    print(
        f"observed order in the {name}: {np.log2(e[h[2]] / e[h[1]]):.2f}, {np.log2(e[h[1]] / e[h[0]]):.2f}"
    )

# ## Figure
#
# Terminal voltage against the relaxed open-circuit voltage; salt concentration and reaction
# current through the cell at five moments; the radial lithium profile in the positive particle next
# to the separator; and the mesh study.

moments = {
    "5 min": 30,
    "25 min": 150,
    "45 min (end of discharge)": 269,
    "after 1 h rest": 629,
    "end of charge": 899,
}
x = 1e6 * run.electrolyte_positions
fig = plots.figure(2, 3, size=(17, 9.5))
plots.line(
    times / 60,
    {"terminal voltage": run.terminal_voltage},
    reference={"open-circuit voltage of mean stoichiometries": relaxed},
    right={"current [A]": area * run.current_density},
    xlabel="time [min]",
    ylabel="voltage [V]",
    right_label="current [A]",
    title="Discharge, rest and charge at 1C",
    legend="upper center",
    ax=fig[0],
)
plots.line(
    x,
    {name: run.electrolyte_concentration[k] for name, k in moments.items()},
    vlines=[1e6 * negative.thickness, 1e6 * (negative.thickness + separator.thickness)],
    xlabel="x [µm]",
    ylabel="salt concentration [mol/m³]",
    title="Electrolyte through the cell",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    1e6 * run.interfacial_positions,
    {name: run.interfacial_current_density[k] for name, k in moments.items()},
    vlines=[1e6 * negative.thickness, 1e6 * (negative.thickness + separator.thickness)],
    xlabel="x [µm]",
    ylabel="reaction current density j [A/m²]",
    title="Where the reaction takes place",
    legend="upper right",
    ax=fig[2],
)
plots.line(
    1e6 * run.radii[1],
    {
        name: run.particle_concentration[1][k, 0] / positive.maximum_concentration
        for name, k in moments.items()
    },
    xlabel="radius [µm]",
    ylabel="stoichiometry θ₊",
    title="Positive particle next to the separator",
    legend="upper left",
    ax=fig[3],
)
plots.line(
    times / 60,
    {"negative": mean[0], "positive": mean[1]},
    xlabel="time [min]",
    ylabel="mean stoichiometry",
    title="Lithium between the electrodes",
    legend="center right",
    ax=fig[4],
)
plots.convergence(
    None,
    {name: (list(e), list(e.values())) for name, e in errors.items()},
    order=(1, 2),
    xlabel="relative cell size (1 / number of cells)",
    ylabel="voltage error at end of discharge [V]",
    title="Mesh study",
    ax=fig[5],
)
plots.show()

# ## Going further
#
# Change the program to a constant-current constant-voltage charge, give the electrolyte
# concentration-dependent transport, or compare the full model with the single-particle model at the
# same current.
