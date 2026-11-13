"""OneSimulation example: first-order reaction kinetics in a well-mixed reactor.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the known closed-form exponential decay.
"""


# # First-order reaction A -> B
#
# A single species $A$ decays irreversibly into a product $B$ in a homogeneous,
# well-mixed (0D) batch reactor:
#
# $$A \xrightarrow{\;k\;} B .$$
#
# With a first-order (mass-action) rate law the reaction rate is $r=k\,C_A$, so
# the concentrations obey
#
# $$\frac{dC_A}{dt}=-k\,C_A,\qquad \frac{dC_B}{dt}=+k\,C_A,\qquad
#   C_A(0)=C_{A0},\;\; C_B(0)=0 .$$
#
# OneSimulation integrates the homogeneous species balance on a lumped (0D)
# domain with an explicit classical Runge-Kutta scheme; the rate coefficient is
# declared as a constant mass-action law in a reaction network.
#
# **Reference.** The linear system has the exact solution
#
# $$C_A(t)=C_{A0}\,e^{-k t},\qquad C_B(t)=C_{A0}\,\bigl(1-e^{-k t}\bigr),$$
#
# and total mass is conserved, $C_A(t)+C_B(t)=C_{A0}$. We compare the computed
# transient with this exponential.

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
# The values are freely adjustable. The rate coefficient $k$ sets the
# characteristic time $1/k$; the window spans five of those times, so $A$ decays
# to about $e^{-5}\approx0.7\,\%$ of its initial value.

k = 0.8               # 1/s, first-order rate coefficient
C_A0 = 1.0            # mol/m^3, initial concentration of A
C_B0 = 0.0            # mol/m^3, initial concentration of B
tau = 1.0 / k         # s, characteristic time
end_time = 5.0 * tau
time_step = 0.05 * tau
print(f"k = {k:.3f} 1/s,  characteristic time 1/k = {tau:.2f} s,  end time = {end_time:.2f} s")


# ## Model
#
# Two species ($A$, $B$) and one irreversible reaction $A\to B$ with a constant
# (mass-action) rate coefficient. The reactor is a single lumped control volume.

network = ReactionNetworkIR(
    "first_order_decay",
    "1",
    species=(
        SpeciesIR("A", 0, 1e-27, composition={"H": 1}),
        SpeciesIR("B", 0, 1e-27, composition={"H": 1}),
    ),
    reactions=(
        ReactionIR("decay", {"A": 1}, {"B": 1}, {"kind": "constant", "value": k}),
    ),
    strict_balance=True,
)

domain = LumpedDomain()
model = ReactionKinetics(domain=domain.system, temperature=300.0, network=network)
C_A = model.concentration("A")
C_B = model.concentration("B")

problem = Problem(
    name="First-order reaction decay",
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


# ## Comparison with the exponential solution
#
# The stored history of each concentration is read back and compared, time by
# time, with the closed-form exponential.

def series(field):
    history = result.field(field).history
    times = np.asarray(history["times"], dtype=float)
    values = np.asarray([value[0] for value in history["values"]], dtype=float)
    return times, values

times, num_A = series(C_A)
_, num_B = series(C_B)

exact_A = C_A0 * np.exp(-k * times)
exact_B = C_A0 * (1.0 - np.exp(-k * times))

err_A = float(np.max(np.abs(num_A - exact_A)))
err_B = float(np.max(np.abs(num_B - exact_B)))
rel_error = max(err_A, err_B) / C_A0
print(f"Maximum absolute error against the exponential: {max(err_A, err_B):.3e} mol/m^3"
      f"  ({100 * rel_error:.4f} % of C_A0)")
assert rel_error < 1e-3, "Deviation larger than the example tolerance (0.1 % of C_A0)."

# Physical sanity: total mass conserved, A monotonically decreasing, B increasing.
mass_error = float(np.max(np.abs(num_A + num_B - C_A0)))
print(f"Maximum mass-balance error C_A+C_B-C_A0: {mass_error:.3e} mol/m^3")
assert mass_error < 1e-9, "The total concentration C_A+C_B should stay constant."
assert np.all(np.diff(num_A) <= 1e-12), "C_A should decrease monotonically."
assert np.all(np.diff(num_B) >= -1e-12), "C_B should increase monotonically."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
t_fine = np.linspace(0.0, end_time, 400)
ax.plot(t_fine, C_A0 * np.exp(-k * t_fine), color="#b5651d", lw=1.6, label="C_A exact")
ax.plot(t_fine, C_A0 * (1.0 - np.exp(-k * t_fine)), color="#1f6f78", lw=1.6, label="C_B exact")
ax.plot(times, num_A, "o", color="#b5651d", ms=4, label="C_A OneSimulation")
ax.plot(times, num_B, "s", color="#1f6f78", ms=4, label="C_B OneSimulation")
ax.set(xlabel="Time / s", ylabel="Concentration / (mol/m$^3$)",
       title="First-order decay — lines: exponential, markers: OneSimulation 0D kinetics")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.show()


# ## Going further
#
# Lower the time step and watch the Runge-Kutta error against the exponential
# shrink at fourth order. Replace the constant rate law by an Arrhenius law
# (`{"kind": "arrhenius", ...}`) and couple a temperature so the rate coefficient
# follows $k(T)$, or add a reverse reaction $B\to A$ to obtain a relaxation to
# chemical equilibrium instead of a complete conversion.
