"""OneSimulation example: a reversible reaction relaxing to chemical equilibrium.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the known closed-form relaxation.
"""


# # Reversible reaction A <-> B
#
# A species $A$ converts into $B$ and back in a homogeneous, well-mixed (0D)
# batch reactor through two opposing first-order steps:
#
# $$A \xrightarrow{\;k_f\;} B,\qquad B \xrightarrow{\;k_r\;} A .$$
#
# With mass-action rate laws the homogeneous balances are
#
# $$\frac{dC_A}{dt}=-k_f C_A + k_r C_B,\qquad
#   \frac{dC_B}{dt}=+k_f C_A - k_r C_B,\qquad C_A(0)=C_{A0},\;C_B(0)=0 .$$
#
# OneSimulation integrates this network on a lumped (0D) domain with an explicit
# classical Runge-Kutta scheme. The two opposing steps are declared as two
# irreversible constant-rate reactions sharing the same two species.
#
# **Reference.** Because $C_A+C_B=C_{A0}$ is conserved, $C_A$ obeys a single
# linear relaxation equation. With the relaxation rate $\lambda=k_f+k_r$ and the
# equilibrium value $C_A^{\mathrm{eq}}=\dfrac{k_r}{k_f+k_r}C_{A0}$,
#
# $$C_A(t)=C_A^{\mathrm{eq}}+\bigl(C_{A0}-C_A^{\mathrm{eq}}\bigr)e^{-\lambda t},$$
#
# and at equilibrium detailed balance fixes the ratio
# $C_B^{\mathrm{eq}}/C_A^{\mathrm{eq}}=k_f/k_r$. We compare both the transient and
# the equilibrium ratio.

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
from onesim.physics.chemistry import (
    ReactionIR,
    ReactionKinetics,
    ReactionNetworkIR,
    SpeciesIR,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The values are freely adjustable. The forward and reverse coefficients set the
# relaxation rate $\lambda=k_f+k_r$ and the equilibrium split. The window spans
# twelve relaxation times, so the system is essentially at equilibrium at the end.

k_f = 1.2             # 1/s, forward rate coefficient A -> B
k_r = 0.4             # 1/s, reverse rate coefficient B -> A
C_A0 = 1.0            # mol/m^3, initial concentration of A
C_B0 = 0.0            # mol/m^3, initial concentration of B

relaxation_rate = k_f + k_r
C_total = C_A0 + C_B0
C_A_eq = k_r / relaxation_rate * C_total
C_B_eq = k_f / relaxation_rate * C_total
tau = 1.0 / relaxation_rate
end_time = 12.0 * tau
time_step = 0.02 * tau
print(f"relaxation rate k_f+k_r = {relaxation_rate:.3f} 1/s,  relaxation time = {tau:.3f} s")
print(f"equilibrium: C_A_eq = {C_A_eq:.4f}, C_B_eq = {C_B_eq:.4f} mol/m^3,  ratio k_f/k_r = {k_f / k_r:.3f}")


# ## Model
#
# Two species ($A$, $B$) and two irreversible reactions with constant rate
# coefficients: $A\to B$ at $k_f$ and $B\to A$ at $k_r$. Together they form the
# reversible pair in a single lumped control volume.

network = ReactionNetworkIR(
    "reversible_pair",
    "1",
    species=(
        SpeciesIR("A", 0, 1e-27, composition={"H": 1}),
        SpeciesIR("B", 0, 1e-27, composition={"H": 1}),
    ),
    reactions=(
        ReactionIR("forward", {"A": 1}, {"B": 1}, {"kind": "constant", "value": k_f}),
        ReactionIR("reverse", {"B": 1}, {"A": 1}, {"kind": "constant", "value": k_r}),
    ),
    strict_balance=True,
)

domain = LumpedDomain()
model = ReactionKinetics(domain=domain.system, temperature=300.0, network=network)
C_A = model.concentration("A")
C_B = model.concentration("B")

problem = Problem(
    name="Reversible reaction equilibrium",
    geometry=domain,
    physics=(model,),
    initial_conditions=(
        InitialValue(C_A, C_A0),
        InitialValue(C_B, C_B0),
    ),
)
simulation = Simulation(
    problem=problem,
    mesh=LumpedMesh(domain),
    numerics=Numerics(
        spatial=LumpedParameters(),
        time_integrator=ClassicalRungeKutta4(),
    ),
)
simulation.validate().require_valid()
result = simulation.run(start_time=0.0, end_time=end_time, time_step=time_step)


# ## Comparison with the relaxation solution
#
# The stored concentration histories are compared, time by time, with the
# closed-form exponential relaxation, and the final state is compared with the
# equilibrium split and the detailed-balance ratio.

def series(field):
    history = result.field(field).history
    times = np.asarray(history["times"], dtype=float)
    values = np.asarray([value[0] for value in history["values"]], dtype=float)
    return times, values

times, num_A = series(C_A)
_, num_B = series(C_B)

exact_A = C_A_eq + (C_A0 - C_A_eq) * np.exp(-relaxation_rate * times)
exact_B = C_B_eq + (C_B0 - C_B_eq) * np.exp(-relaxation_rate * times)

err = max(float(np.max(np.abs(num_A - exact_A))), float(np.max(np.abs(num_B - exact_B))))
rel_error = err / C_total
print(f"Maximum absolute error against the relaxation: {err:.3e} mol/m^3"
      f"  ({100 * rel_error:.4f} % of total)")

# Equilibrium: final concentrations and the detailed-balance ratio.
final_A, final_B = num_A[-1], num_B[-1]
print(f"final C_A = {final_A:.5f} (eq {C_A_eq:.5f}),  final C_B = {final_B:.5f} (eq {C_B_eq:.5f})")
ratio = final_B / final_A
print(f"equilibrium ratio C_B/C_A = {ratio:.4f},  k_f/k_r = {k_f / k_r:.4f}")

# Physical sanity: total mass conserved throughout.
mass_error = float(np.max(np.abs(num_A + num_B - C_total)))
print(f"Maximum mass-balance error C_A+C_B-C_total: {mass_error:.3e} mol/m^3")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
t_fine = np.linspace(0.0, end_time, 400)
ax.plot(t_fine, C_A_eq + (C_A0 - C_A_eq) * np.exp(-relaxation_rate * t_fine),
        color="#b5651d", lw=1.6, label="C_A exact")
ax.plot(t_fine, C_B_eq + (C_B0 - C_B_eq) * np.exp(-relaxation_rate * t_fine),
        color="#1f6f78", lw=1.6, label="C_B exact")
ax.plot(times, num_A, "o", color="#b5651d", ms=4, label="C_A OneSimulation")
ax.plot(times, num_B, "s", color="#1f6f78", ms=4, label="C_B OneSimulation")
ax.axhline(C_A_eq, color="#b5651d", ls=":", lw=1.0)
ax.axhline(C_B_eq, color="#1f6f78", ls=":", lw=1.0)
ax.set(xlabel="Time / s", ylabel="Concentration / (mol/m$^3$)",
       title="Reversible reaction — lines: relaxation, markers: OneSimulation 0D kinetics")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.show()


# ## Going further
#
# Change the ratio $k_f/k_r$ and watch the equilibrium split move while the
# relaxation rate stays $k_f+k_r$. Start from the product side ($C_A=0$,
# $C_B=C_{B0}$) and confirm the same equilibrium is reached from above. Replace
# the two constant laws by Arrhenius laws and couple a temperature to study how
# the equilibrium shifts with $T$.
