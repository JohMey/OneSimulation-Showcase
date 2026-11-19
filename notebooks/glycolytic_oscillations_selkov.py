"""OneSimulation example: sustained oscillations in a metabolic reaction network.
Requires the installed, activated OneSimulation runtime.

The Sel'kov model of glycolysis is written as a mass-action reaction network with an autocatalytic
step. Growth rate and frequency of small disturbances are measured from the simulation and compared
with the eigenvalues of the steady state; the Hopf bifurcation where oscillations set in is located.
"""

# # Oscillations in a metabolic reaction network
#
# In glycolysis the enzyme phosphofructokinase is activated by its own product ADP. This positive
# feedback can make the concentrations of the intermediates oscillate with a period of minutes,
# as observed in yeast extracts. Sel'kov reduced the mechanism to two species, the substrate $Y$
# (fructose-6-phosphate) and the product $X$ (ADP), in a reactor with a steady supply of $Y$:
#
# | step | reaction | rate (dimensionless) |
# |---|---|---|
# | supply | $\varnothing \to Y$ | $b$ |
# | uncatalysed conversion | $Y \to X$ | $a\,y$ |
# | autocatalytic conversion | $Y + 2X \to 3X$ | $x^2y$ |
# | removal | $X \to \varnothing$ | $x$ |
#
# $$\dot x = -x + a\,y + x^2y,\qquad \dot y = b - a\,y - x^2y .$$
#
# OneSimulation integrates the network on a well-mixed (0D) volume; the cubic step is an ordinary
# mass-action law of total order three.
#
# **Reference.** E. E. Sel'kov, *Self-oscillations in glycolysis*, Eur. J. Biochem. 4 (1968) 79;
# the analysis follows S. H. Strogatz, *Nonlinear Dynamics and Chaos*, section 7.3. The single
# steady state is $x^\ast=b$, $y^\ast=b/(a+b^2)$. Its Jacobian has the trace and determinant
#
# $$\tau = -\frac{b^4+(2a-1)\,b^2+a+a^2}{a+b^2},\qquad \Delta = a+b^2 ,$$
#
# so small disturbances grow or decay like $e^{\lambda t}$ with $\lambda=\tau/2\pm i\sqrt{\Delta-\tau^2/4}$.
# The steady state loses stability in a Hopf bifurcation where $\tau=0$,
# $b^2=\tfrac12\big(1-2a\pm\sqrt{1-8a}\big)$, and the oscillation is born with the angular
# frequency $\sqrt{\Delta}$.

import numpy as np
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
from onesim.results import oscillation_fit
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()


# ## Reaction network
#
# The supply enters as a constant species source of $Y$; the removal of $X$ is a reaction without
# products, so the element balance is not enforced.

a = 0.08
network = ReactionNetworkIR(
    "selkov_glycolysis",
    "1",
    species=(SpeciesIR("X", 0, 1e-25), SpeciesIR("Y", 0, 1e-25)),
    reactions=(
        ReactionIR("uncatalysed", {"Y": 1}, {"X": 1}, {"kind": "constant", "value": a}),
        ReactionIR(
            "autocatalytic",
            {"Y": 1, "X": 2},
            {"X": 3},
            {"kind": "constant", "value": 1.0},
        ),
        ReactionIR("removal", {"X": 1}, {}, {"kind": "constant", "value": 1.0}),
    ),
    strict_balance=False,
)
hopf = np.sqrt(0.5 * (1 - 2 * a + np.array([-1, 1]) * np.sqrt(1 - 8 * a)))
print(f"a = {a}: oscillations expected for {hopf[0]:.4f} < b < {hopf[1]:.4f}")


# ## Linear stability from the simulation
#
# For a sweep over the supply rate $b$ the network starts a small distance $10^{-4}$ away from the
# steady state. While the disturbance is small it is a single complex mode; projecting it onto the
# left eigenvector of the Jacobian gives a signal $\propto e^{\lambda t}$, whose growth rate and
# frequency `oscillation_fit` measures.

domain = LumpedDomain()
supplies = np.linspace(0.2, 1.1, 19)
models = {
    b: ReactionKinetics(domain=domain.system, network=network, species_sources={"Y": b})
    for b in supplies
}
runs = {
    b: Simulation(
        problem=Problem(
            name=f"Selkov b={b:.2f}",
            geometry=domain,
            physics=(model,),
            initial_conditions=(
                InitialValue(model.concentration("X"), b + 1e-4),
                InitialValue(model.concentration("Y"), b / (a + b**2)),
            ),
        ),
        mesh=LumpedMesh(domain),
        numerics=Numerics(
            spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()
        ),
    ).run(start_time=0.0, end_time=30.0, time_step=0.05)
    for b, model in models.items()
}

jacobian = {
    b: np.array(
        [[-1 + 2 * b**2 / (a + b**2), a + b**2], [-2 * b**2 / (a + b**2), -(a + b**2)]]
    )
    for b in supplies
}
eigen = {b: np.linalg.eig(J.T) for b, J in jacobian.items()}
left = {
    b: values_vectors[1][:, np.argmax(values_vectors[0].imag)]
    for b, values_vectors in eigen.items()
}
fits = {
    b: oscillation_fit(
        run.field(models[b].concentration("X")).times,
        left[b][0] * (run.field(models[b].concentration("X")).time_series()[1] - b)
        + left[b][1]
        * (run.field(models[b].concentration("Y")).time_series()[1] - b / (a + b**2)),
    )
    for b, run in runs.items()
}
exact = {
    b: values_vectors[0][np.argmax(values_vectors[0].imag)]
    for b, values_vectors in eigen.items()
}
for b in supplies:
    print(
        f"b = {b:.3f}: growth rate {fits[b].growth_rate:+.5f} (eigenvalue {exact[b].real:+.5f}),  "
        f"frequency {abs(fits[b].frequency):.5f} (eigenvalue {abs(exact[b].imag):.5f})"
    )
growth = np.array([fits[b].growth_rate for b in supplies])
crossings = [
    supplies[i]
    - growth[i] * (supplies[i + 1] - supplies[i]) / (growth[i + 1] - growth[i])
    for i in np.flatnonzero(np.diff(np.sign(growth)))
]
print(
    "Hopf points from the measured growth rates: "
    + ", ".join(f"{value:.4f}" for value in crossings)
    + f"   (analytic {hopf[0]:.4f}, {hopf[1]:.4f})"
)


# ## Limit cycles
#
# Inside the unstable range the disturbance grows until the nonlinearity limits it: the network
# settles on a closed orbit. Three supply rates are followed from the same small disturbance over
# 400 time units. The period of the cycle is the mean spacing of the maxima of $x$.

cycles = {
    b: ReactionKinetics(domain=domain.system, network=network, species_sources={"Y": b})
    for b in (0.40, 0.60, 0.80)
}
cycle_runs = {
    b: Simulation(
        problem=Problem(
            name=f"Selkov cycle b={b:.2f}",
            geometry=domain,
            physics=(model,),
            initial_conditions=(
                InitialValue(model.concentration("X"), b + 0.01),
                InitialValue(model.concentration("Y"), b / (a + b**2)),
            ),
        ),
        mesh=LumpedMesh(domain),
        numerics=Numerics(
            spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()
        ),
    ).run(start_time=0.0, end_time=400.0, time_step=0.05)
    for b, model in cycles.items()
}
orbits = {
    b: (
        run.field(cycles[b].concentration("X")).time_series(),
        run.field(cycles[b].concentration("Y")).time_series()[1],
    )
    for b, run in cycle_runs.items()
}
for b, ((t, x), y) in orbits.items():
    late = t > 330
    peaks = t[1:-1][(x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & late[1:-1]]
    print(
        f"b = {b:.2f}: x oscillates between {x[late].min():.3f} and {x[late].max():.3f}, period {np.diff(peaks).mean():.3f}, "
        f"linear estimate at onset 2 pi / sqrt(a + b^2) = {2 * np.pi / np.sqrt(a + b**2):.3f}"
    )


# ## Figure
#
# Growth rate and frequency of the steady state over the supply rate, the limit cycles in the
# phase plane, and the oscillation of the substrate and product at $b = 0.6$.

b_line = np.linspace(0.2, 1.1, 300)
trace = -(b_line**4 + (2 * a - 1) * b_line**2 + a + a**2) / (a + b_line**2)
(t6, x6), y6 = orbits[0.60]
fig = plots.figure(1, 3, size=(18, 5))
plots.line(
    points={
        "growth rate, OneSimulation": (supplies, growth),
        "frequency, OneSimulation": (
            supplies,
            [abs(fits[b].frequency) for b in supplies],
        ),
    },
    reference={
        "growth rate, eigenvalue": (b_line, trace / 2),
        "frequency, eigenvalue": (
            b_line,
            np.sqrt(np.maximum(a + b_line**2 - trace**2 / 4, 0)),
        ),
    },
    hlines={"neutral": 0.0},
    vlines={"Hopf point": hopf[0], "Hopf point ": hopf[1]},
    xlabel="supply rate b",
    ylabel="Re and Im of lambda",
    title="Stability of the steady state",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    y={f"b = {b:.2f}": (x[t > 330], y[t > 330]) for b, ((t, x), y) in orbits.items()},
    points={"steady states": (list(orbits), [b / (a + b**2) for b in orbits])},
    xlabel="product x (ADP)",
    ylabel="substrate y (F6P)",
    title="Limit cycles",
    ax=fig[1],
)
plots.line(
    y={"product x (ADP)": (t6, x6), "substrate y (F6P)": (t6, y6)},
    xlimits=(0, 120),
    xlabel="time",
    ylabel="concentration",
    title="Oscillation at b = 0.6",
    legend="upper left",
    ax=fig[2],
)
plots.show()


# ## Going further
#
# Approach the upper Hopf point from inside and watch the cycle shrink to zero amplitude while its
# period tends to $2\pi/\sqrt{\Delta}$. Couple two such reactors through a slow exchange of $X$
# to study synchronisation, or replace the constant supply by a periodic one to drive the
# network into resonance or chaos.
