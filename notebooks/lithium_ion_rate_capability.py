"""OneSimulation example: rate capability of a lithium-ion cell with the Doyle-Fuller-Newman model.
Requires the installed, activated OneSimulation runtime.

A graphite-SiOx | NMC811 cell is discharged at six rates from C/10 to 1.5C. Delivered capacity,
energy and the salt depletion of the electrolyte are compared with the stoichiometric capacity and
with the steady salt balance, which predicts the limiting current of the cell.
"""

# # Rate capability of a lithium-ion cell
#
# The faster a cell is discharged, the larger its polarisation, the earlier it reaches the cut-off
# voltage and the less capacity and energy it delivers. In a cell with thick, dense electrodes the
# electrolyte is often the bottleneck: lithium salt is produced in the negative electrode and
# consumed in the positive one, and the salt concentration near the positive collector falls.
#
# For a uniform reaction rate the steady salt balance $-D_e\,c_e'=(1-t_+)\,i_e/F$ with an ionic
# current rising linearly from zero to $I$ in the negative electrode, constant in the separator and
# falling back to zero in the positive electrode gives
#
# $$c_e(x)=c_0+\frac{(1-t_+)\,I}{F}\big(\bar g-g(x)\big),\qquad g(x)=\int_0^x\frac{i_e/I}{D_e}\,dx,$$
#
# where $\bar g$ is the porosity-weighted mean of $g$ that conserves the salt. The concentration
# reaches zero at the positive collector at the limiting current
# $I_\mathrm{lim}=F c_0/\big((1-t_+)(g(L)-\bar g)\big)$, so the minimum concentration falls as
# $c_0(1-I/I_\mathrm{lim})$.
#
# **Reference.** Cell parameters of Chen et al., *J. Electrochem. Soc.* 167 (2020) 080534 (LG M50,
# 5 Ah); model of Doyle, Fuller and Newman, *J. Electrochem. Soc.* 140 (1993) 1526. Checks: (i) at
# low rate the delivered capacity approaches the stoichiometric capacity of the limiting electrode;
# (ii) the electrolyte follows the steady salt balance and the limiting current above; (iii) the
# capacity-rate data are summarised by Peukert's law $Q\propto I^{1-k}$.

import numpy as np
from onesim.physics.chemistry import (
    BatteryElectrodeLayer,
    BatteryElectrolyteLayer,
    PorousElectrodeBattery,
)
from onesim.visualization import plots, theme

theme.set()

# ## Cell

faraday, area, transference = 96485.33212, 0.1027, 0.2594
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
    initial_stoichiometry=0.9014,
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
    initial_stoichiometry=0.2661,
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
    transference_number=transference,
)
stoichiometric = faraday * 0.75 * 85.2e-6 * 33133 * (0.9014 - 0.0279) * area / 3600

# ## Discharges
#
# Each discharge starts from the full cell and runs a little past the 2.5 V cut-off (the end time is
# given as a fraction of the nominal hour count); the capacity is read at the cut-off.

rates = {0.1: 1.02, 0.2: 1.02, 0.5: 1.01, 1.0: 0.995, 1.25: 0.985, 1.5: 0.972}
runs = {
    rate: cell.solve(
        mode="DFN",
        current_density=5.0 * rate / area,
        final_time=end * 3600 / rate,
        radial_cells=12,
        through_cells=(8, 4, 8),
        output_times=np.linspace(0, end * 3600 / rate, 800),
    )
    for rate, end in rates.items()
}

# ## Results

layers = (
    (negative, 0.0),
    (separator, negative.thickness),
    (positive, negative.thickness + separator.thickness),
)
x = np.linspace(0, negative.thickness + separator.thickness + positive.thickness, 4001)
g = lambda x: np.where(
    x < negative.thickness,
    x**2 / (2 * negative.thickness * negative.electrolyte_diffusivity),
    np.where(
        x < layers[2][1],
        negative.thickness / (2 * negative.electrolyte_diffusivity)
        + (x - negative.thickness) / separator.electrolyte_diffusivity,
        g_end
        - (x[-1] - x) ** 2
        / (2 * positive.thickness * positive.electrolyte_diffusivity),
    ),
)
g_end = (
    negative.thickness / (2 * negative.electrolyte_diffusivity)
    + separator.thickness / separator.electrolyte_diffusivity
    + positive.thickness / (2 * positive.electrolyte_diffusivity)
)
porosity = np.select(
    [x < negative.thickness, x < layers[2][1]],
    [negative.porosity, separator.porosity],
    positive.porosity,
)
g_mean = np.trapezoid(porosity * g(x), x) / np.trapezoid(porosity, x)
limiting = faraday * 1000 / ((1 - transference) * (g_end - g_mean))
results = {}
for rate, run in runs.items():
    capacity = 5.0 * rate * run.times / 3600
    cut = np.argmax(run.terminal_voltage < 2.5)
    q = np.interp(
        2.5, run.terminal_voltage[cut : cut - 2 : -1], capacity[cut : cut - 2 : -1]
    )
    keep = capacity <= q
    results[rate] = dict(
        q=q,
        energy=np.trapezoid(run.terminal_voltage[keep], capacity[keep]),
        salt=run.electrolyte_concentration[keep].min(),
        profile=run.electrolyte_concentration[np.flatnonzero(keep)[-1]],
    )
peukert = (
    1
    - np.polyfit(
        np.log(list(rates)[2:]), np.log([results[r]["q"] for r in list(rates)[2:]]), 1
    )[0]
)
print(
    f"stoichiometric capacity of the negative electrode {stoichiometric:.4f} Ah; limiting current of the steady salt balance "
    f"{limiting:.1f} A/m² = {limiting * area:.2f} A = {limiting * area / 5:.2f}C"
)
print(
    "C-rate   capacity [Ah]  of stoichiometric   energy [Wh]   mean voltage [V]   min salt [mol/m³]   steady balance c0(1-I/I_lim)"
)
for rate, r in results.items():
    print(
        f"{rate:6.2f}   {r['q']:12.4f}   {100 * r['q'] / stoichiometric:13.1f} %   {r['energy']:11.3f}   {r['energy'] / r['q']:15.4f}"
        f"   {r['salt']:17.1f}   {1000 * (1 - 5 * rate / area / limiting):17.1f}"
    )
print(f"Peukert exponent from C/2 to 1.5C: k = {peukert:.4f}")

# ## Figure
#
# Discharge curves, capacity and energy over the rate, the salt profile at the cut-off against the
# steady balance, and the minimum salt concentration on its way to the limiting current.

fig = plots.figure(2, 2, size=(14, 10))
plots.line(
    y={
        f"{rate:g}C": (5.0 * rate * run.times / 3600, run.terminal_voltage)
        for rate, run in runs.items()
    },
    hlines={"cut-off": 2.5},
    xlimits=(0, 5.3),
    ylimits=(2.4, 4.25),
    xlabel="discharged capacity [Ah]",
    ylabel="terminal voltage [V]",
    title="Discharge curves from C/10 to 1.5C",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    list(rates),
    {"capacity [Ah]": [r["q"] for r in results.values()]},
    right={"energy [Wh]": [r["energy"] for r in results.values()]},
    hlines={"stoichiometric capacity": stoichiometric},
    ylimits=(4.82, 5.12),
    markers=True,
    scale="logx",
    xlabel="C-rate",
    ylabel="capacity to 2.5 V [Ah]",
    right_label="energy to 2.5 V [Wh]",
    title="Rate capability",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    1e6 * runs[0.1].electrolyte_positions,
    {f"{rate:g}C": r["profile"] for rate, r in results.items() if rate >= 0.5},
    reference={
        f"{rate:g}C": (
            1e6 * x,
            1000 + (1 - transference) * 5 * rate / area / faraday * (g_mean - g(x)),
        )
        for rate in results
        if rate >= 0.5
    },
    vlines=[1e6 * negative.thickness, 1e6 * layers[2][1]],
    xlabel="x [µm]",
    ylabel="salt concentration [mol/m³]",
    title="Salt at the cut-off, dashed: steady balance",
    legend="lower left",
    ax=fig[2],
)
plots.line(
    points={
        "simulation": (
            5 * np.array(list(rates)) / area,
            [r["salt"] for r in results.values()],
        )
    },
    reference={
        "steady balance c₀(1 - I/I_lim)": (
            np.linspace(0, limiting, 50),
            1000 * (1 - np.linspace(0, limiting, 50) / limiting),
        )
    },
    vlines={"limiting current": limiting},
    xlabel="current density [A/m²]",
    ylabel="minimum salt concentration [mol/m³]",
    title="Towards the limiting current",
    legend="upper right",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Thinner or more porous electrodes raise the limiting current: change the thickness and porosity
# and see the rate capability improve at the expense of the stored energy per area.
