"""OneSimulation example: tubular reactor with axial dispersion and Danckwerts boundary conditions.
Requires the installed, activated OneSimulation runtime.

A first-order reaction runs in a tube in which the fluid is partly back-mixed. The 1D
advection-dispersion-reaction balance with Danckwerts inlet and outlet conditions is solved with
quadratic finite elements for Peclet numbers from almost mixed to almost plug flow and compared
with the closed-form solution of Danckwerts and Wehner & Wilhelm.
"""

# # Danckwerts benchmark: tubular reactor with axial dispersion
#
# In a real tube the fluid is neither perfectly mixed nor in ideal plug flow. The dispersion model
# superposes an axial dispersion coefficient $D$ on the mean velocity $u$. For a first-order
# reaction $A\to B$ the steady concentration along the tube of length $L$ obeys
#
# $$u\,\frac{dc}{dx} = D\,\frac{d^2c}{dx^2} - k\,c,\qquad 0<x<L .$$
#
# Danckwerts (1953) argued that the dispersion stops at the ends of the vessel. All reactant that
# enters must cross the inlet plane, and nothing flows back through the outlet:
#
# | boundary | condition | as total outward flux $\;-(u\,c - D\,c')\cdot n$ |
# |---|---|---|
# | inlet $x=0$ | $u\,c_0 = u\,c - D\,c'$ | prescribed: $-u\,c_0$ |
# | outlet $x=L$ | $c' = 0$ | advective only: $u\,c$ |
#
# The inlet condition makes the concentration jump at the entrance; that jump is the signature of
# the benchmark. Two numbers govern the result, the Peclet number $\mathrm{Pe}=uL/D$ and the
# Damköhler number $\mathrm{Da}=kL/u$.
#
# **Reference.** P. V. Danckwerts, *Continuous flow systems*, Chem. Eng. Sci. 2 (1953) 1;
# J. F. Wehner & R. H. Wilhelm, Chem. Eng. Sci. 6 (1956) 89. With $a=\sqrt{1+4\,\mathrm{Da}/\mathrm{Pe}}$
# and $z=x/L$ the exact profile is
#
# $$\frac{c(z)}{c_0}=\frac{2\,e^{\mathrm{Pe}\,z/2}\Big[(1+a)\,e^{a\mathrm{Pe}(1-z)/2}-(1-a)\,e^{-a\mathrm{Pe}(1-z)/2}\Big]}
#   {(1+a)^2e^{a\mathrm{Pe}/2}-(1-a)^2e^{-a\mathrm{Pe}/2}},$$
#
# which tends to the stirred tank, $1/(1+\mathrm{Da})$, for $\mathrm{Pe}\to0$ and to plug flow,
# $e^{-\mathrm{Da}}$, for $\mathrm{Pe}\to\infty$.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.chemistry import SpeciesTransport, TransportSpecies
from onesim.results.comparisons import convergence
from onesim.simulation import OutwardFluxBoundary, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()


# ## Tube, mesh and numerics
#
# A 1 m tube with $u = 1$ m/s and $\mathrm{Da}=2$. The tube starts empty and is run for fifteen
# residence times with the second-order BDF method until the profile is steady; space is
# discretised with quadratic Lagrange elements.

length, velocity, rate, inlet = 1.0, 1.0, 2.0, 1.0
tube = Line(start=(0.0, 0.0, 0.0), end=(length, 0.0, 0.0))
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(2), quadrature_order=5),
    linear_solver=SparseDirect(),
    nonlinear_solver=Newton(
        linear_solver=SparseDirect(),
        jacobian="analytic",
        absolute_tolerance=1e-12,
        relative_tolerance=1e-11,
    ),
    time_integrator=BDF(order=2, startup=BackwardEuler()),
)
a = lambda Pe: np.sqrt(1 + 4 * rate * length / velocity / Pe)
exact = lambda Pe, z: (
    2
    * np.exp(Pe * z / 2)
    * (
        (1 + a(Pe)) * np.exp(a(Pe) * Pe * (1 - z) / 2)
        - (1 - a(Pe)) * np.exp(-a(Pe) * Pe * (1 - z) / 2)
    )
    / (
        (1 + a(Pe)) ** 2 * np.exp(a(Pe) * Pe / 2)
        - (1 - a(Pe)) ** 2 * np.exp(-a(Pe) * Pe / 2)
    )
)


# ## Model and Peclet sweep
#
# The same reactor for eight Peclet numbers on 80 elements. The Danckwerts conditions enter as
# total outward fluxes: a fixed feed flux at the inlet, the advective flux of the interior state at
# the outlet.

peclets = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
models = {
    Pe: SpeciesTransport(
        domain=tube.line,
        velocity=velocity,
        species=(
            TransportSpecies("A", diffusivity=velocity * length / Pe, decay_rate=rate),
        ),
    )
    for Pe in peclets
}
problems = {
    Pe: Problem(
        name="Danckwerts tubular reactor",
        geometry=tube,
        physics=(model,),
        boundary_conditions=(
            OutwardFluxBoundary(
                physics=model,
                boundary=tube.boundaries.start,
                flux=lambda **kw: np.array([-velocity * inlet]),
                jacobian=lambda **kw: np.zeros((1, 1)),
            ),
            OutwardFluxBoundary(
                physics=model,
                boundary=tube.boundaries.end,
                flux=lambda interior, **kw: velocity * interior,
                jacobian=lambda **kw: np.array([[velocity]]),
                wave_speed=velocity,
            ),
        ),
        initial_conditions=(InitialValue(model.concentration("A"), 0.0),),
    )
    for Pe, model in models.items()
}
fields = {
    Pe: Simulation(
        problem=problems[Pe],
        mesh=StructuredLineMesher(cells=80).generate(tube),
        numerics=numerics,
    )
    .run(start_time=0.0, end_time=15 * length / velocity, time_step=0.05)
    .field(model.concentration("A"))
    for Pe, model in models.items()
}


# ## Results
#
# Outlet and inlet concentration of every run against the closed form, and the largest deviation
# along the tube.

z = np.linspace(0.0, 1.0, 401)
line_points = np.column_stack([z * length, 0 * z, 0 * z])
profiles = {Pe: np.ravel(field.evaluate(line_points)) for Pe, field in fields.items()}
for Pe, c in profiles.items():
    print(
        f"Pe = {Pe:5.1f}: outlet c/c0 {c[-1]:.6f} (exact {exact(Pe, 1.0):.6f}), inlet c/c0 {c[0]:.6f} (exact {exact(Pe, 0.0):.6f}), "
        f"largest deviation along the tube {np.max(np.abs(c - exact(Pe, z))):.1e}"
    )
print(
    f"limits: stirred tank 1/(1+Da) = {1 / (1 + rate * length / velocity):.4f}, plug flow exp(-Da) = {np.exp(-rate * length / velocity):.4f}"
)


# ## Mesh study
#
# The outlet concentration at $\mathrm{Pe}=20$ on four meshes. At the element nodes quadratic
# elements are superconvergent, so the error falls with the fourth power of the element size.

cells = (5, 10, 20, 40)
outlet = [
    float(
        np.ravel(
            Simulation(
                problem=problems[20.0],
                mesh=StructuredLineMesher(cells=n).generate(tube),
                numerics=numerics,
            )
            .run(start_time=0.0, end_time=15 * length / velocity, time_step=0.05)
            .field(models[20.0].concentration("A"))
            .evaluate(np.array([[length, 0.0, 0.0]]))
        )[0]
    )
    for n in cells
]
study = convergence(outlet, [length / n for n in cells])
for n, value in zip(cells, outlet):
    print(
        f"{n:3d} elements: outlet c/c0 {value:.9f}, error {abs(value - exact(20.0, 1.0)):.2e}"
    )
print(
    f"observed order of convergence {study.order:.2f}, Richardson extrapolation {study.extrapolated:.9f} (exact {exact(20.0, 1.0):.9f})"
)


# ## Figure
#
# Concentration profiles along the tube with the jump at the inlet, the outlet conversion between
# the stirred-tank and the plug-flow limits, and the mesh study.

Pe_line = np.geomspace(0.05, 100, 200)
fig = plots.figure(1, 3, size=(18, 5))
plots.line(
    points={
        f"Pe = {Pe:g}": (z[::20], profiles[Pe][::20]) for Pe in (0.1, 1.0, 5.0, 50.0)
    },
    reference={f"Pe = {Pe:g}": (z, exact(Pe, z)) for Pe in (0.1, 1.0, 5.0, 50.0)},
    hlines={"feed": inlet},
    ylimits=(0, 1.05),
    xlabel="position x / L",
    ylabel="c / c0",
    title="Profiles, dashed: exact",
    ax=fig[0],
)
plots.line(
    points={"OneSimulation": (peclets, [1 - profiles[Pe][-1] for Pe in peclets])},
    reference={"Danckwerts": (Pe_line, 1 - exact(Pe_line, 1.0))},
    hlines={
        "stirred tank": 1 - 1 / (1 + rate * length / velocity),
        "plug flow": 1 - np.exp(-rate * length / velocity),
    },
    scale="logx",
    ylimits=(0.65, 0.88),
    xlabel="Peclet number uL/D",
    ylabel="conversion at the outlet",
    title="Between mixed and plug flow (Da = 2)",
    legend="lower right",
    ax=fig[1],
)
plots.line(
    y={
        "outlet error": (
            [length / n for n in cells],
            [abs(value - exact(20.0, 1.0)) for value in outlet],
        )
    },
    markers=True,
    scale="loglog",
    slope=4,
    xlabel="element size [m]",
    ylabel="error of c/c0 at the outlet",
    title="Mesh study at Pe = 20",
    ax=fig[2],
)
plots.show()


# ## Going further
#
# Raise the Damköhler number and watch the inlet jump grow. Swap the quadratic elements for a
# finite-volume scheme with upwinding to reach Peclet numbers of several thousand, or replace the
# first-order decay by a reaction network to model a non-linear rate law in the same tube.
