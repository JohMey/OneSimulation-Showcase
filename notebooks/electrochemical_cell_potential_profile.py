"""OneSimulation example: the potential profile across an electrochemical cell.
Requires the installed, activated OneSimulation runtime.

A Daniell-type cell with planar zinc and copper electrodes is computed at open circuit,
during discharge and during charge, with and without electrode kinetics. The potential
steps at the electrodes and the ohmic drop in the electrolyte are checked against the
closed-form voltage budget of the cell.
"""

# # Potential profile across an electrochemical cell
#
# A battery or galvanic cell delivers its open-circuit voltage only when no current
# flows. Under load the cell voltage is lower, during charging it is higher, and the
# difference is spent in three places: the activation overpotentials at the two
# electrodes and the ohmic drop in the electrolyte between them. This example makes
# that budget visible as a potential profile across a Daniell-type cell,
#
# $$\mathrm{Zn}\,|\,\mathrm{Zn^{2+}}\,\|\,\mathrm{Cu^{2+}}\,|\,\mathrm{Cu},\qquad
#   E^0_\mathrm{Zn}=-0.76\ \mathrm{V},\quad E^0_\mathrm{Cu}=+0.34\ \mathrm{V},\quad
#   U_\mathrm{OCV}=1.10\ \mathrm{V}.$$
#
# In the electrolyte (uniform composition, conductivity $\kappa$) the current density is
# $\mathbf i=-\kappa\nabla\phi_l$ with $\nabla\cdot\mathbf i=0$. At each electrode the
# metal potential $\phi_s$, the electrolyte potential $\phi_l$ and the equilibrium
# potential $E_\mathrm{eq}$ define the overpotential $\eta=\phi_s-\phi_l-E_\mathrm{eq}$,
# and the current follows Butler-Volmer kinetics with $\alpha_a=\alpha_c=1/2$:
#
# $$i=2\,i_0\sinh\!\Big(\frac{nF\eta}{2RT}\Big)\quad\Longleftrightarrow\quad
#   \eta=\frac{2RT}{nF}\,\operatorname{arsinh}\frac{i}{2i_0}.$$
#
# Without kinetics ($i_0\to\infty$, primary current distribution) $\eta=0$. OneSimulation
# solves the current balance with `ElectrolyteCurrentDistribution` on a line between the
# two electrodes.
#
# | Boundary | Condition |
# |---|---|
# | $x=0$, zinc | $\phi_s=0$ (ground), $E_\mathrm{eq}=-0.76$ V, $i_0=20$ A/m² |
# | $x=L$, copper | $\phi_s=U$ (cell voltage), $E_\mathrm{eq}=+0.34$ V, $i_0=5$ A/m² |
#
# **Reference.** J. Newman and N. P. Balsara, *Electrochemical Systems*, 4th ed.,
# Wiley 2021, ch. 8 and 18; A. J. Bard and L. R. Faulkner, *Electrochemical Methods*,
# 2nd ed., Wiley 2001, ch. 3. For a planar cell the current density $i$ (positive on
# discharge) satisfies the voltage budget
#
# $$U_\mathrm{OCV}-U=\frac{iL}{\kappa}+\frac{2RT}{nF}\Big[\operatorname{arsinh}\frac{i}{2i_{0,\mathrm{Zn}}}
#   +\operatorname{arsinh}\frac{i}{2i_{0,\mathrm{Cu}}}\Big],$$
#
# which reduces to Ohm's law $i=\kappa\,(U_\mathrm{OCV}-U)/L$ for the primary distribution.
# The electrolyte potential is linear between the electrodes, and at open circuit it sits
# $0.76$ V above the zinc metal.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import cartesian
from onesim.physics.chemistry import CurrentElectrode, ElectrolyteCurrentDistribution
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()


# ## Cell
#
# A 5 mm electrolyte gap with a conductivity of 5 S/m (about that of a molar zinc or
# copper sulphate solution), two-electron reactions at 25 °C. The exchange current
# densities are illustrative values for the two metals.

gap, conductivity, temperature, electrons = 5e-3, 5.0, 298.15, 2
zinc, copper, open_circuit = -0.76, 0.34, 1.10
exchange = {"zinc": 20.0, "copper": 5.0}
thermal = 8.314462618 * temperature / 96485.33212
line = Line(start=(0.0, 0.0, 0.0), end=(gap, 0.0, 0.0))
mesh = cartesian(shape=(40,), extents=(gap,), geometry=line, periodic=False).build()
print(
    f"electrolyte resistance L/κ = {gap / conductivity * 1e4:.1f} Ω cm², RT/F = {1e3 * thermal:.2f} mV"
)

# ## Three operating points and a polarisation curve
#
# Open circuit (1.10 V), discharge at 0.80 V and charge at 1.40 V, each with kinetics
# (secondary distribution) and without (primary distribution); then a sweep of the cell
# voltage from 0.2 V to 1.6 V.

cells = {
    (voltage, kinetics): ElectrolyteCurrentDistribution(
        conductivity=conductivity,
        temperature=temperature,
        electrodes={
            "start": CurrentElectrode(
                potential=0.0,
                equilibrium_potential=zinc,
                electron_number=electrons,
                exchange_current_density=exchange["zinc"] if kinetics else None,
            ),
            "end": CurrentElectrode(
                potential=voltage,
                equilibrium_potential=copper,
                electron_number=electrons,
                exchange_current_density=exchange["copper"] if kinetics else None,
            ),
        },
    ).solve(mesh)
    for voltage in (*np.round(np.linspace(0.2, 1.6, 15), 2), 1.4)
    for kinetics in (True, False)
}
centres = (np.arange(40) + 0.5) * gap / 40

# ## Comparison with the voltage budget
#
# The current from the closed-form budget, the overpotentials and the ohmic drop at the
# three operating points; the electrolyte potentials at the electrode surfaces come from
# the solution.

budget = lambda u: brentq(
    lambda i: i * gap / conductivity
    + 2
    * thermal
    / electrons
    * (
        np.arcsinh(i / (2 * exchange["zinc"]))
        + np.arcsinh(i / (2 * exchange["copper"]))
    )
    - (open_circuit - u),
    -1e4,
    1e4,
)
print(
    " U [V]   kinetics   current [A/m²]   budget [A/m²]   η_Zn [mV]   η_Cu [mV]   ohmic drop [mV]   φ_l at Zn [V]"
)
for voltage in (1.1, 0.8, 1.4):
    for kinetics in (True, False):
        result = cells[(voltage, kinetics)]
        current = result.electrode_currents["end"]
        surface = result.electrode_potentials
        print(
            f"{voltage:5.2f} {'BV' if kinetics else 'none':>9s} {current:15.4f} {budget(voltage) if kinetics else conductivity * (open_circuit - voltage) / gap:15.4f} "
            f"{1e3 * (0.0 - surface['start'] - zinc):11.2f} {1e3 * (voltage - surface['end'] - copper):11.2f} "
            f"{1e3 * (surface['start'] - surface['end']):17.2f} {float(surface['start']):15.4f}"
        )
sweep = np.round(np.linspace(0.2, 1.6, 15), 2)
computed = np.array([cells[(u, True)].electrode_currents["end"] for u in sweep])
print(
    f"polarisation curve: largest deviation from the budget {np.abs(computed - [budget(u) for u in sweep]).max():.1e} A/m² "
    f"(currents up to {np.abs(computed).max():.0f} A/m²); largest linearity defect of φ_l "
    f"{max(np.abs(np.diff(r.potential, 2)).max() for r in cells.values()):.1e} V"
)

# ## Figure
#
# The potential across the cell (metal, electrolyte, metal): the steps at the electrodes
# are the equilibrium potentials plus the overpotentials, the slope in the electrolyte is
# the ohmic drop. On the right the cell voltage against the current, with and without
# electrode kinetics. Dotted verticals: the electrode surfaces.

states = {"open circuit, 1.10 V": 1.1, "discharge, 0.80 V": 0.8, "charge, 1.40 V": 1.4}
fig = plots.figure(1, 2, size=(15, 5.5))
plots.line(
    y={
        name: (
            1e3 * np.r_[-1.5e-3, 0.0, 0.0, centres, gap, gap, gap + 1.5e-3],
            np.r_[
                0.0,
                0.0,
                cells[(u, True)].electrode_potentials["start"],
                cells[(u, True)].potential,
                cells[(u, True)].electrode_potentials["end"],
                u,
                u,
            ],
        )
        for name, u in states.items()
    },
    vlines=[0.0, 1e3 * gap],
    xlabel="position x [mm] (zinc metal | electrolyte | copper metal)",
    ylabel="potential [V] (zinc metal = 0)",
    title="Potential profile with Butler-Volmer kinetics",
    colors={"open circuit, 1.10 V": 6, "discharge, 0.80 V": 1, "charge, 1.40 V": 0},
    legend="lower right",
    ax=fig[0],
)
plots.line(
    points={
        "secondary (Butler-Volmer)": (computed, sweep),
        "primary (no kinetics)": (
            np.array([cells[(u, False)].electrode_currents["end"] for u in sweep]),
            sweep,
        ),
    },
    reference={
        "secondary (Butler-Volmer)": (
            np.array([budget(u) for u in np.linspace(0.2, 1.6, 141)]),
            np.linspace(0.2, 1.6, 141),
        ),
        "primary (no kinetics)": (
            conductivity * (open_circuit - np.linspace(0.2, 1.6, 141)) / gap,
            np.linspace(0.2, 1.6, 141),
        ),
    },
    hlines={"open-circuit voltage": open_circuit},
    vlines=[0.0],
    xlabel="current density i [A/m²] (discharge > 0)",
    ylabel="cell voltage U [V]",
    title="Polarisation curve (dashed: voltage budget)",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Raise the exchange current densities and the secondary curve approaches the primary
# straight line; lower them and the activation losses dominate at small currents (the
# Tafel regime, $\eta\approx\frac{RT}{\alpha nF}\ln(i/i_0)$). In two or three dimensions
# the same model gives the primary and secondary current distributions over shaped
# electrodes, where the Wagner number $\kappa\,(d\eta/di)/L$ decides how uniform the
# current is. Concentration overpotentials need the tertiary model with species
# transport (`PoissonNernstPlanck`).
