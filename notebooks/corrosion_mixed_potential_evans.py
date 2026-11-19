"""OneSimulation example: corrosion of iron in aerated water, the Evans diagram and the mixed potential.
Requires the installed, activated OneSimulation runtime.

A zero-dimensional electrode model combines the anodic dissolution of iron with the cathodic
reduction of dissolved oxygen, whose current is limited by oxygen transport. The corrosion
potential and the corrosion current are computed as the mixed potential of the two reactions
and checked against the closed-form intersections of the Tafel lines and the limiting current,
and against an independent root of the current balance; the oxygen supply is then varied.
"""

# # Corrosion of iron in aerated water: Evans diagram and mixed potential
#
# A piece of iron in neutral, aerated water corrodes because two electrode reactions run on
# the same metal surface at the same potential $E$: the anodic dissolution of the metal and
# the cathodic reduction of dissolved oxygen,
#
# $$\mathrm{Fe}\rightarrow\mathrm{Fe^{2+}}+2e^-,\qquad \mathrm{O_2}+2\mathrm{H_2O}+4e^-\rightarrow4\,\mathrm{OH^-}.$$
#
# Far from their equilibrium potentials both follow Tafel laws, and the oxygen reduction is
# limited by the diffusion of oxygen to the surface (limiting current density $i_L$):
#
# $$i_a=i_{0,a}\,10^{(E-E_a)/b_a},\qquad
#   |i_c|=\frac{i_k}{1+i_k/i_L},\quad i_k=i_{0,c}\,10^{-(E-E_c)/b_c}.$$
#
# A freely corroding electrode carries no net current, $i_a=|i_c|$: this fixes the
# corrosion (mixed) potential $E_\mathrm{corr}$ and the corrosion current density
# $i_\mathrm{corr}$. OneSimulation's `ElectrodePolarization` holds the two reactions
# (`TafelReaction`, one with a transport limit); `solve_stationary()` returns the
# polarisation curve over a potential sweep, and `mixed_potential()` solves the current
# balance with a selectable nonlinear solver.
#
# **Reference.** U. R. Evans, *The Corrosion and Oxidation of Metals*, Arnold 1960;
# D. A. Jones, *Principles and Prevention of Corrosion*, 2nd ed., Prentice Hall 1996,
# ch. 3-4. Without transport limit the intersection of the two Tafel lines is closed form,
#
# $$E_\mathrm{corr}=\frac{\log_{10}(i_{0,c}/i_{0,a})+E_a/b_a+E_c/b_c}{1/b_a+1/b_c},\qquad
#   i_\mathrm{corr}=i_{0,a}\,10^{(E_\mathrm{corr}-E_a)/b_a};$$
#
# when oxygen transport controls, $i_\mathrm{corr}\to i_L$ and
# $E_\mathrm{corr}\to E_a+b_a\log_{10}(i_L/i_{0,a})$. In between, the balance is solved
# independently with a bracketing root finder. Faraday's law turns the corrosion current
# into a penetration rate $\dot d=i_\mathrm{corr}M/(nF\rho)$.

import numpy as np
from onesim.materials import TafelReaction
from onesim.numerics.nonlinear import LeastSquares
from onesim.physics.chemistry import ElectrodePolarization
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()


# ## Electrode kinetics
#
# Potentials against the standard hydrogen electrode. Iron: $E_a=-0.44$ V, $i_{0,a}=10^{-3}$
# A/m², $b_a=60$ mV/decade. Oxygen at pH 7: $E_c=+0.80$ V, $i_{0,c}=10^{-6}$ A/m²,
# $b_c=120$ mV/decade, and a limiting current density of 0.1 A/m² for still, air-saturated
# water at room temperature (values of the order given by Jones, ch. 3).

iron = {"potential": -0.44, "exchange": 1e-3, "slope": 0.06}
oxygen = {"potential": 0.80, "exchange": 1e-6, "slope": 0.12}
limiting, molar_mass, density, faraday = 0.1, 55.845e-3, 7874.0, 96485.33212
sweep = np.linspace(-0.75, 0.30, 421)
electrode = lambda limit: ElectrodePolarization(
    reactions=(
        TafelReaction(
            reaction_id="iron dissolution",
            exchange_current_density=iron["exchange"],
            equilibrium_potential=iron["potential"],
            tafel_slope=iron["slope"],
            direction="anodic",
        ),
        TafelReaction(
            reaction_id="oxygen reduction",
            exchange_current_density=oxygen["exchange"],
            equilibrium_potential=oxygen["potential"],
            tafel_slope=oxygen["slope"],
            direction="cathodic",
            limiting_current_density=limit,
        ),
    ),
    potential_sweep=sweep,
)

# ## Mixed potential
#
# The current balance is solved with Levenberg-Marquardt between -0.8 V and +0.3 V; the
# independent reference is a bracketing root of the same balance written out by hand.

solver = LeastSquares(method="levenberg_marquardt")
corrosion = electrode(limiting).mixed_potential(
    bounds=(-0.8, 0.3), solver=solver, current_tolerance=1e-12
)
balance = lambda E, limit: (
    iron["exchange"] * 10 ** ((E - iron["potential"]) / iron["slope"])
    - 1
    / (
        1 / (oxygen["exchange"] * 10 ** (-(E - oxygen["potential"]) / oxygen["slope"]))
        + 1 / limit
    )
)
reference = brentq(balance, -0.8, 0.3, args=(limiting,), xtol=1e-14)
tafel_potential = (
    np.log10(oxygen["exchange"] / iron["exchange"])
    + iron["potential"] / iron["slope"]
    + oxygen["potential"] / oxygen["slope"]
) / (1 / iron["slope"] + 1 / oxygen["slope"])
current = corrosion.reaction_current_densities["iron dissolution"]
print(
    f"corrosion potential {corrosion.potential:.6f} V (independent root {reference:.6f} V, "
    f"transport-limit estimate {iron['potential'] + iron['slope'] * np.log10(limiting / iron['exchange']):.6f} V)"
)
print(
    f"corrosion current density {current:.6f} A/m² (limiting current {limiting} A/m²), balance residual {corrosion.balance_residual:.1e} A/m², "
    f"{corrosion.evaluations} evaluations"
)
print(
    f"penetration rate of iron: {current * molar_mass / (2 * faraday * density) * 1e3 * 3.15576e7:.3f} mm/year"
)
print(
    f"without transport limit the Tafel lines would cross at {tafel_potential:.4f} V and "
    f"{iron['exchange'] * 10 ** ((tafel_potential - iron['potential']) / iron['slope']):.1f} A/m²"
)

# ## Oxygen supply
#
# The limiting current density grows with stirring, flow and oxygen content. Over seven
# decades of $i_L$ the corrosion current moves from transport control ($i_\mathrm{corr}=i_L$)
# to activation control (the Tafel intersection).

limits = np.logspace(-3, 4, 29)
points = [
    electrode(limit).mixed_potential(
        bounds=(-0.8, 0.3), solver=solver, current_tolerance=1e-12
    )
    for limit in limits
]
potentials = np.array([p.potential for p in points])
currents = np.array([p.reaction_current_densities["iron dissolution"] for p in points])
roots = np.array(
    [brentq(balance, -0.8, 0.3, args=(limit,), xtol=1e-14) for limit in limits]
)
print(
    f"largest deviation of the corrosion potential from the independent root over the sweep: {np.abs(potentials - roots).max():.1e} V"
)
print(
    f"i_corr / i_L at i_L = 1e-3 A/m²: {currents[0] / limits[0]:.4f};  i_corr at i_L = 1e4 A/m²: {currents[-1]:.2f} A/m² "
    f"(Tafel intersection {iron['exchange'] * 10 ** ((tafel_potential - iron['potential']) / iron['slope']):.2f} A/m²)"
)

# ## Figure
#
# The Evans diagram (potential against the logarithm of the current density of each branch,
# with the net polarisation curve), and the corrosion current and potential against the oxygen
# limiting current with their two closed-form limits.

curve = electrode(limiting).solve_stationary()
tafel_current = iron["exchange"] * 10 ** (
    (tafel_potential - iron["potential"]) / iron["slope"]
)
fig = plots.figure(1, 3, size=(18, 5.5))
plots.line(
    y={
        "iron dissolution": (
            np.abs(curve.reaction_current_densities["iron dissolution"]),
            curve.potential,
        ),
        "oxygen reduction": (
            np.abs(curve.reaction_current_densities["oxygen reduction"]),
            curve.potential,
        ),
        "net current |i|": (np.abs(curve.current_density), curve.potential),
    },
    points={"corrosion point": (np.array([current]), np.array([corrosion.potential]))},
    vlines={"limiting current": limiting},
    colors={"net current |i|": 6},
    styles={"net current |i|": "dashed"},
    scale="logx",
    xlimits=(1e-5, 1e3),
    ylimits=(-0.75, 0.3),
    xlabel="current density |i| [A/m²]",
    ylabel="potential E [V vs SHE]",
    title="Evans diagram",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    points={"OneSimulation": (limits, currents)},
    reference={
        "transport control i = i_L": (limits, limits),
        "Tafel intersection": (limits, np.full_like(limits, tafel_current)),
    },
    scale="loglog",
    ylimits=(5e-4, 3e2),
    xlabel="oxygen limiting current i_L [A/m²]",
    ylabel="corrosion current i_corr [A/m²]",
    title="Corrosion current against oxygen supply",
    legend="lower right",
    ax=fig[1],
)
plots.line(
    points={"OneSimulation": (limits, potentials)},
    reference={
        "independent root": (limits, roots),
        "E_a + b_a log(i_L / i_0a)": (
            limits,
            iron["potential"] + iron["slope"] * np.log10(limits / iron["exchange"]),
        ),
    },
    hlines={"Tafel intersection": tafel_potential},
    scale="logx",
    ylimits=(-0.65, -0.05),
    xlabel="oxygen limiting current i_L [A/m²]",
    ylabel="corrosion potential E_corr [V vs SHE]",
    title="Corrosion potential",
    legend="lower right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Add a second cathodic reaction, hydrogen evolution, and lower the pH: below about pH 4 it
# overtakes oxygen reduction and the corrosion current no longer follows the oxygen supply.
# Couple the iron to zinc in a galvanic pair by adding the zinc dissolution branch: the mixed
# potential drops and the iron current falls (cathodic protection). Measured polarisation
# data can be fitted for the Tafel parameters with the least-squares `fit` of a
# `SimulationStudyService` built on the same model.
