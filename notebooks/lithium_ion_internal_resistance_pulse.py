"""OneSimulation example: internal resistance of a lithium-ion cell from current pulses.
Requires the installed, activated OneSimulation runtime.

Current pulses are applied to a graphite-SiOx | NMC811 cell at rest with the Doyle-Fuller-Newman
model. The instantaneous resistance is compared with the closed-form porous-electrode resistance of
Newman and Tobias in a mesh study, and the growth of the resistance during a 10 s pulse is followed
over amplitude and state of charge.
"""

# # Internal resistance of a lithium-ion cell from current pulses
#
# When a current step $I$ is applied to a cell at rest, the voltage first drops at once and then
# keeps falling while concentration gradients build up in the particles and the electrolyte. The
# pulse resistance $R(t)=\big(U_\mathrm{oc}-V(t)\big)/I$ is the usual way to characterise the
# internal resistance of a cell. In the Doyle-Fuller-Newman model (equations in the
# discharge-rest-charge example) the instantaneous part comes from electronic and ionic conduction
# and from the charge-transfer reaction distributed through the porous electrodes.
#
# For a linearised Butler-Volmer law, $a\,j = (a/r_\mathrm{ct})\,(\phi_s-\phi_e-U)$ with
# $r_\mathrm{ct}=RT/(F\,j_0)$, the porous electrode of thickness $L$ with conductivities $\sigma$
# (solid) and $\kappa$ (electrolyte) has the area resistance
#
# $$R_\mathrm{el}=\frac{L}{\kappa+\sigma}\left[1+\frac{2+\big(\frac{\sigma}{\kappa}+\frac{\kappa}{\sigma}\big)\cosh\nu}{\nu\,\sinh\nu}\right],\qquad
#   \nu=L\sqrt{\frac{a}{r_\mathrm{ct}}\Big(\frac{1}{\kappa}+\frac{1}{\sigma}\Big)} ,$$
#
# and the cell adds the separator, $R_0=R_{\mathrm{el},-}+L_\mathrm{sep}/\kappa_\mathrm{sep}+R_{\mathrm{el},+}$.
#
# **Reference.** J. Newman and C. W. Tobias, *J. Electrochem. Soc.* 109 (1962) 1183, for the
# porous-electrode resistance; cell parameters of Chen et al., *J. Electrochem. Soc.* 167 (2020)
# 080534 (LG M50, 5 Ah). The comparison uses a pulse of C/20, small enough for linear kinetics; 1C
# and 3C pulses show the nonlinear reaction resistance.

from dataclasses import replace

import numpy as np
from onesim.physics.chemistry import (
    BatteryElectrodeLayer,
    BatteryElectrolyteLayer,
    PorousElectrodeBattery,
)
from onesim.visualization import plots, theme

theme.set()

# ## Cell at 50 % state of charge

faraday, area, thermal = 96485.33212, 0.1027, 8.314462618 * 298.15 / 96485.33212
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
    initial_stoichiometry=0.0279 + 0.5 * (0.9014 - 0.0279),
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
    initial_stoichiometry=0.9077 + 0.5 * (0.2661 - 0.9077),
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

# The same cell at other states of charge differs only in the initial stoichiometries; the rest
# voltage is the open-circuit voltage of those.

socs = np.round(np.linspace(0.05, 0.95, 19), 2)
cells = {
    soc: replace(
        cell,
        negative=replace(
            negative, initial_stoichiometry=0.0279 + soc * (0.9014 - 0.0279)
        ),
        positive=replace(
            positive, initial_stoichiometry=0.9077 + soc * (0.2661 - 0.9077)
        ),
    )
    for soc in socs
}
rest = {
    soc: positive_ocp(c.positive.initial_stoichiometry, 0)
    - negative_ocp(c.negative.initial_stoichiometry, 0)
    for soc, c in cells.items()
}

# ## Newman-Tobias resistance
#
# The closed form with the exchange current $j_0=2\,j_{0,\mathrm{ref}}\sqrt{\theta(1-\theta)}$ of
# each electrode at its initial stoichiometry.

newman_tobias = (
    lambda cell: cell.separator.thickness / cell.separator.electrolyte_conductivity
    + sum(
        layer.thickness
        / (k + s)
        * (1 + (2 + (s / k + k / s) * np.cosh(nu)) / (nu * np.sinh(nu)))
        for layer, k, s in (
            (layer, layer.electrolyte_conductivity, layer.electronic_conductivity)
            for layer in (cell.negative, cell.positive)
        )
        for nu in [
            layer.thickness
            * np.sqrt(
                3
                * layer.active_fraction
                / layer.particle_radius
                * 2
                * layer.reference_exchange_current
                * np.sqrt(
                    layer.initial_stoichiometry * (1 - layer.initial_stoichiometry)
                )
                / thermal
                * (1 / k + 1 / s)
            )
        ]
    )
)

# ## Pulses
#
# The instantaneous response is the voltage at the switch-on of a C/20 step; it is computed on
# particle meshes from 10 to 320 shells. Then 10 s pulses of C/20, 1C and 3C at 50 % state of charge
# followed by 30 s of rest, and 10 s pulses of C/20 and 1C from 10 to 90 % state of charge.

small = 0.05 * 5.0 / area
shells = (10, 20, 40, 80, 160, 320)
instant = {
    n: (
        rest[0.5]
        - cell.solve(
            mode="DFN",
            current_density=small,
            final_time=1e-6,
            radial_cells=n,
            through_cells=(20, 10, 20),
            output_times=(0.0,),
        ).terminal_voltage[0]
    )
    / small
    for n in shells
}
times = np.r_[0.0, np.geomspace(1e-3, 9.99, 80), np.linspace(10.01, 40.0, 60)]
pulses = {
    rate: cell.solve(
        mode="DFN",
        current_density=lambda t, rate=rate: rate * 5.0 / area * float(t < 10.0),
        final_time=40.0,
        radial_cells=80,
        through_cells=(12, 6, 12),
        output_times=times,
    )
    for rate in (0.05, 1.0, 3.0)
}
sweep = {
    (rate, soc): cells[soc]
    .solve(
        mode="DFN",
        current_density=rate * 5.0 / area,
        final_time=10.0,
        radial_cells=80,
        through_cells=(12, 6, 12),
        output_times=(0.0, 10.0),
    )
    .terminal_voltage
    for rate in (0.05, 1.0)
    for soc in socs[1:-1:2]
}

# ## Results

reference = newman_tobias(cell)
extrapolated = 2 * instant[320] - instant[160]
print(
    f"Newman-Tobias resistance at 50 % SOC: {1e4 * reference:.4f} Ω·cm² = {1e3 * reference / area:.3f} mΩ for the 5 Ah cell"
)
for n, value in instant.items():
    print(
        f"  {n:4d} particle shells: {1e4 * value:.4f} Ω·cm², deviation {100 * (value / reference - 1):+.3f} %"
    )
print(
    f"  Richardson extrapolation of 160 and 320 shells: deviation {100 * (extrapolated / reference - 1):+.3f} %"
)
for rate, run in pulses.items():
    resistance = (rest[0.5] - run.terminal_voltage) / (rate * 5.0 / area)
    print(
        f"{rate:4.2f}C pulse: R(0) = {1e3 * resistance[0] / area:.2f} mΩ, R(1 s) = {1e3 * np.interp(1.0, times, resistance) / area:.2f} mΩ, "
        f"R(10 s) = {1e3 * resistance[times < 10][-1] / area:.2f} mΩ; voltage 30 s after the pulse "
        f"{1e3 * (run.terminal_voltage[-1] - rest[0.5]):+.2f} mV from rest"
    )
print(
    "SOC     R(0) at C/20   Newman-Tobias   deviation   R(10 s) at C/20   R(10 s) at 1C   [mΩ]"
)
for soc in socs[1:-1:2]:
    r0, nt = (rest[soc] - sweep[0.05, soc][0]) / (0.05 * 5.0), newman_tobias(
        cells[soc]
    ) / area
    print(
        f"{soc:4.1f}   {1e3 * r0:12.2f}   {1e3 * nt:13.2f}   {100 * (r0 / nt - 1):+8.2f} %   {1e3 * (rest[soc] - sweep[0.05, soc][1]) / 0.25:15.2f}"
        f"   {1e3 * (rest[soc] - sweep[1.0, soc][1]) / 5.0:13.2f}"
    )

# ## Figure
#
# The pulse responses, the pulse resistance over time, the particle-mesh study of the
# instantaneous resistance and the resistance over the state of charge.

fig = plots.figure(2, 2, size=(14, 10))
plots.line(
    times,
    {
        f"{rate:g}C": 1e3 * (run.terminal_voltage - rest[0.5])
        for rate, run in pulses.items()
    },
    xlabel="time [s]",
    ylabel="V - U_oc [mV]",
    title="10 s discharge pulses at 50 % SOC",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    times[1:81],
    {
        f"{rate:g}C": 1e3 * (rest[0.5] - run.terminal_voltage[1:81]) / (rate * 5.0)
        for rate, run in pulses.items()
    },
    hlines={"Newman-Tobias R₀": 1e3 * reference / area},
    scale="logx",
    xlabel="time after switch-on [s]",
    ylabel="pulse resistance [mΩ]",
    title="Instantaneous and growing resistance",
    legend="upper left",
    ax=fig[1],
)
plots.convergence(
    [1 / n for n in shells],
    {"|R₀ - Newman-Tobias| / R₀": [abs(v / reference - 1) for v in instant.values()]},
    order=1,
    xlabel="particle shell width / radius",
    ylabel="relative deviation",
    title="Particle-mesh study at C/20",
    ax=fig[2],
)
plots.line(
    100 * socs[1:-1:2],
    {
        f"{label}, {rate:g}C": [
            1e3 * (rest[soc] - sweep[rate, soc][k]) / (5.0 * rate)
            for soc in socs[1:-1:2]
        ]
        for rate in (0.05, 1.0)
        for k, label in ((0, "R(0)"), (1, "R(10 s)"))
    },
    markers=True,
    styles={"R(0), 1C": "dotted", "R(10 s), 1C": "dotted"},
    colors={"R(0), 0.05C": 0, "R(10 s), 0.05C": 1, "R(0), 1C": 0, "R(10 s), 1C": 1},
    reference={
        "Newman-Tobias R₀": (
            100 * socs,
            [1e3 * newman_tobias(c) / area for c in cells.values()],
        )
    },
    xlabel="state of charge [%]",
    ylabel="resistance [mΩ]",
    title="Resistance over the state of charge",
    legend="upper center",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Split the instantaneous resistance into electrode and separator parts by setting one electrode's
# exchange current very high, or fit a two-branch equivalent circuit to the pulse resistance curve.
