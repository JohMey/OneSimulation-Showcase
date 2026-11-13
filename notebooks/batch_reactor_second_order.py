"""OneSimulation example: second-order reaction kinetics in a well-mixed reactor.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the known closed-form hyperbola of
a second-order batch reaction.
"""


# # Second-order reaction 2A -> P
#
# A single species $A$ is consumed by an irreversible second-order reaction in a
# homogeneous, well-mixed (0D) batch reactor, two molecules of $A$ forming one
# product $P$:
#
# $$2A \xrightarrow{\;k\;} P .$$
#
# With a mass-action rate law the reaction rate is $r=k\,C_A^2$. Because each
# reaction event removes two molecules of $A$, the stoichiometry gives
#
# $$\frac{dC_A}{dt}=-2\,k\,C_A^2,\qquad \frac{dC_P}{dt}=+k\,C_A^2,\qquad
#   C_A(0)=C_{A0},\;\; C_P(0)=0 .$$
#
# OneSimulation integrates the homogeneous species balance on a lumped (0D)
# domain with an explicit classical Runge-Kutta scheme; the rate coefficient is
# declared as a constant mass-action law in a reaction network, and the factor of
# two in $dC_A/dt$ comes from the reactant stoichiometry, not from the rate law.
#
# **Reference.** Separating variables in $dC_A/dt=-2kC_A^2$ gives the hyperbola
#
# $$\frac{1}{C_A(t)}=\frac{1}{C_{A0}}+2\,k\,t,\qquad\text{so}\qquad
#   C_A(t)=\frac{C_{A0}}{1+2\,k\,C_{A0}\,t}.$$
#
# Mass is conserved as $C_A(t)+2\,C_P(t)=C_{A0}$ (two $A$ per $P$). We compare the
# computed transient with this closed form.

from importlib.metadata import version

import numpy as np
import matplotlib.pyplot as plt

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
# The values are freely adjustable. The product $k\,C_{A0}$ sets the
# characteristic time $1/(2kC_{A0})$; the window spans several of those times, so
# $A$ falls well below its initial value.

k = 0.5               # m^3/(mol s), second-order rate coefficient
C_A0 = 2.0            # mol/m^3, initial concentration of A
C_P0 = 0.0            # mol/m^3, initial concentration of product P
half_time = 1.0 / (2.0 * k * C_A0)   # s, time for A to halve
end_time = 10.0 * half_time
time_step = 0.02 * half_time
print(f"k = {k:.3f} m^3/(mol s),  half-time 1/(2 k C_A0) = {half_time:.2f} s,  end time = {end_time:.2f} s")


# ## Model
#
# Two species ($A$, $P$) and one irreversible reaction $2A\to P$ with a constant
# (mass-action) rate coefficient. The element balance closes with $A$ carrying one
# unit of a fictitious element and $P$ carrying two. The reactor is a single
# lumped control volume.

network = ReactionNetworkIR(
    "second_order_dimerization",
    "1",
    species=(
        SpeciesIR("A", 0, 1e-27, composition={"X": 1}),
        SpeciesIR("P", 0, 2e-27, composition={"X": 2}),
    ),
    reactions=(
        ReactionIR("dimerize", {"A": 2}, {"P": 1}, {"kind": "constant", "value": k}),
    ),
    strict_balance=True,
)

domain = LumpedDomain()
model = ReactionKinetics(domain=domain.system, temperature=300.0, network=network)
C_A = model.concentration("A")
C_P = model.concentration("P")

problem = Problem(
    name="Second-order batch reactor",
    geometry=domain,
    physics=(model,),
    initial_conditions=(
        InitialValue(C_A, C_A0),
        InitialValue(C_P, C_P0),
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


# ## Comparison with the hyperbola
#
# The stored history of each concentration is read back and compared, time by
# time, with the closed-form second-order solution.

def series(field):
    history = result.field(field).history
    times = np.asarray(history["times"], dtype=float)
    values = np.asarray([value[0] for value in history["values"]], dtype=float)
    return times, values

times, num_A = series(C_A)
_, num_P = series(C_P)

exact_A = C_A0 / (1.0 + 2.0 * k * C_A0 * times)
exact_P = 0.5 * (C_A0 - exact_A)

err_A = float(np.max(np.abs(num_A - exact_A)))
err_P = float(np.max(np.abs(num_P - exact_P)))
rel_error = max(err_A, err_P) / C_A0
print(f"Maximum absolute error against the hyperbola: {max(err_A, err_P):.3e} mol/m^3"
      f"  ({100 * rel_error:.4f} % of C_A0)")
assert rel_error < 1e-3, "Deviation larger than the example tolerance (0.1 % of C_A0)."

# Physical sanity: mass conserved as C_A + 2 C_P, A monotonically decreasing, P
# monotonically increasing.
mass_error = float(np.max(np.abs(num_A + 2.0 * num_P - C_A0)))
print(f"Maximum mass-balance error C_A+2 C_P-C_A0: {mass_error:.3e} mol/m^3")
assert mass_error < 1e-9, "The total C_A + 2 C_P should stay constant."
assert np.all(np.diff(num_A) <= 1e-12), "C_A should decrease monotonically."
assert np.all(np.diff(num_P) >= -1e-12), "C_P should increase monotonically."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
t_fine = np.linspace(0.0, end_time, 400)
a_fine = C_A0 / (1.0 + 2.0 * k * C_A0 * t_fine)
ax.plot(t_fine, a_fine, color="#b5651d", lw=1.6, label="C_A exact")
ax.plot(t_fine, 0.5 * (C_A0 - a_fine), color="#1f6f78", lw=1.6, label="C_P exact")
ax.plot(times, num_A, "o", color="#b5651d", ms=4, label="C_A OneSimulation")
ax.plot(times, num_P, "s", color="#1f6f78", ms=4, label="C_P OneSimulation")
ax.set(xlabel="Time / s", ylabel="Concentration / (mol/m$^3$)",
       title="Second-order batch 2A -> P — lines: hyperbola, markers: OneSimulation 0D kinetics")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.show()


# ## Going further
#
# Compare with the first-order case: a second-order reaction tails off much more
# slowly because the rate $k C_A^2$ collapses as $A$ is depleted, so $C_A$ follows
# a $1/t$ hyperbola rather than an exponential. Raise $C_{A0}$ and the early decay
# sharpens, since the second-order rate scales with the square of concentration.
