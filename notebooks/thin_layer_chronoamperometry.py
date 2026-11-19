"""OneSimulation example: chronoamperometry in a thin-layer electrochemical cell.
Requires the installed, activated OneSimulation runtime.

A potential step drives the reduction of all electroactive species in a thin solution
layer to the diffusion limit. The computed current and charge are checked against the
Cottrell equation at short times and against the exact thin-layer series solution at
all times, including the exhaustive electrolysis of the layer.
"""

# # Chronoamperometry in a thin-layer cell
#
# A thin layer of solution of thickness $L$ is trapped between a working electrode at
# $x=0$ and an inert wall at $x=L$. It contains an electroactive species O at the bulk
# concentration $c_0$. At $t=0$ the electrode potential is stepped far beyond the
# formal potential, so every O that reaches the electrode is reduced at once
# ($\mathrm{O}+n\,e^-\to\mathrm{R}$) and its surface concentration drops to zero. O
# then diffuses towards the electrode until the layer is exhausted:
#
# $$\frac{\partial c}{\partial t}=D\,\frac{\partial^2c}{\partial x^2},\qquad
#   c(x,0)=c_0,\quad c(0,t)=0,\quad \left.\frac{\partial c}{\partial x}\right|_{x=L}=0.$$
#
# The current density is $i=nFD\,\partial c/\partial x|_{x=0}$, and the charge passed is
# $q(t)=nF\left(c_0L-\int_0^Lc\,dx\right)$. OneSimulation solves the diffusion equation
# with quadratic finite elements on a mesh graded towards the electrode and BDF2 steps
# under adaptive step control. The current is the gradient of the computed concentration
# at the electrode, the charge the amount of O that has left the layer.
#
# **Reference.** A. J. Bard and L. R. Faulkner, *Electrochemical Methods*, 2nd ed.,
# Wiley 2001, sections 5.2 and 11.7; A. T. Hubbard, *Study of the kinetics of
# electrochemical reactions by thin-layer voltammetry*, J. Electroanal. Chem. 22,
# 165-174 (1969). As long as the depletion layer $\sqrt{Dt}$ is much thinner than $L$,
# the electrode does not feel the wall (Cottrell, Anson):
#
# $$i=nFc_0\sqrt{\frac{D}{\pi t}},\qquad q=2nFc_0\sqrt{\frac{Dt}{\pi}}.$$
#
# The exact solution of the finite layer is the series
#
# $$i=\frac{2nFDc_0}{L}\sum_{k\ge0}e^{-\lambda_k^2Dt},\qquad
#   q=nFc_0L\Big(1-\sum_{k\ge0}\frac{2}{\lambda_k^2L^2}e^{-\lambda_k^2Dt}\Big),\qquad
#   \lambda_k=\frac{(2k+1)\pi}{2L},$$
#
# so the current decays exponentially once the layer is depleted and the charge
# approaches $nFc_0L$: the cell works as a coulometer.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import NonuniformLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, Nitsche
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, AdaptiveTimeStep, BackwardEuler
from onesim.physics.chemistry import SpeciesTransport, TransportSpecies
from onesim.simulation import (
    OutwardFluxBoundary,
    PrescribedValueBoundary,
    Problem,
    Simulation,
)
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.special import erf

theme.set()


# ## Cell
#
# A 50 µm layer of a 1 mM solution of a one-electron redox species with the diffusion
# coefficient $10^{-9}$ m²/s. The diffusion time $L^2/D$ is 2.5 s; the experiment runs
# for twice that.
# The electrode holds the concentration at zero; the inert wall is a zero outward flux.

thickness, diffusivity, bulk, electrons, faraday = 50e-6, 1.0e-9, 1.0, 1, 96485.33212
end_time, start = 2.0 * thickness**2 / diffusivity, 1e-5
print(
    f"diffusion time L²/D = {thickness**2 / diffusivity:.2f} s, exhaustive charge n F c0 L = {electrons * faraday * bulk * thickness:.4f} C/m²"
)

line = Line(start=(0.0, 0.0, 0.0), end=(thickness, 0.0, 0.0))
model = SpeciesTransport(
    domain=line.line,
    velocity=0.0,
    nonnegative=False,
    species=(TransportSpecies("O", diffusivity=diffusivity),),
)
problem = Problem(
    name="Thin-layer chronoamperometry",
    geometry=line,
    physics=(model,),
    boundary_conditions=(
        PrescribedValueBoundary(
            physics=model,
            boundary=line.boundaries.start,
            value=lambda time, coordinate: {"O": 0.0},
        ),
        OutwardFluxBoundary(
            physics=model,
            boundary=line.boundaries.end,
            flux=lambda **kw: np.zeros(1),
            jacobian=lambda **kw: np.zeros((1, 1)),
        ),
    ),
    initial_conditions=(
        InitialValue(
            model.concentration("O"),
            lambda x, y, z: bulk * erf(x / (2 * np.sqrt(diffusivity * start))),
        ),
    ),
)

# ## Mesh and simulation
#
# Sixty quadratic elements, geometrically graded so that the first one is 50 times thinner than the
# last: the element next to the electrode is 0.05 µm thick and resolves the depletion layer
# already after about $10^{-5}$ s. The simulation starts at $t_0=10^{-5}$ s from the exact early profile
# $c=c_0\,\mathrm{erf}\big(x/2\sqrt{Dt_0}\big)$, which the wall 50 µm away cannot yet feel; this avoids the
# jump between the initial and the electrode concentration. The step control starts at $10^{-7}$ s and
# lets the steps grow as the transient slows down.

faces = thickness * (np.geomspace(1.0, 51.0, 61) - 1.0) / 50.0
mesh = NonuniformLineMesher(faces=tuple(faces)).generate(line)
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(
            element=Lagrange(order=2), boundary_enforcement=Nitsche()
        ),
        nonlinear_solver=Newton(linear_solver=SparseDirect(), jacobian="analytic"),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
    ),
).run(
    start_time=start,
    end_time=end_time,
    time_step=AdaptiveTimeStep(
        initial=1e-7,
        minimum=1e-9,
        maximum=0.05,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-4,
    ),
)
concentration = result.field(model.concentration("O"))
amount = concentration.integrate()
times = np.asarray(amount.times)
charge = electrons * faraday * (bulk * thickness - np.asarray(amount.values))
current = np.array(
    [
        electrons
        * faraday
        * diffusivity
        * concentration.gradient(np.zeros((1, 3)), time=t)[0, 0]
        for t in times
    ]
)
print(
    f"{len(times) - 1} accepted steps from {np.diff(times)[0]:.1e} s to {np.diff(times).max():.3f} s"
)

# ## Comparison
#
# The thin-layer series with 400 terms, and the Cottrell and Anson expressions, at the
# accepted times from $10^{-4}$ s on.


def thin_layer(t):
    rates = (
        (((2 * np.arange(400) + 1) * np.pi / (2 * thickness)) ** 2)[:, None]
        * diffusivity
        * t
    )
    return (
        2
        * electrons
        * faraday
        * diffusivity
        * bulk
        / thickness
        * np.exp(-rates).sum(axis=0),
        electrons
        * faraday
        * bulk
        * thickness
        * (
            1
            - (2 * diffusivity * t / (rates * thickness**2) * np.exp(-rates)).sum(
                axis=0
            )
        ),
    )


exact_current, exact_charge = thin_layer(times[1:])
cottrell = electrons * faraday * bulk * np.sqrt(diffusivity / (np.pi * times[1:]))
compared = times[1:] >= 1e-4
print(
    "    t [s]   current [A/m²]  thin-layer   Cottrell     charge [C/m²]  thin-layer    Anson"
)
for target in (1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.5, 4.0):
    k = np.argmin(np.abs(times[1:] - target))
    print(
        f"{times[1 + k]:9.2e} {current[1 + k]:13.4f} {exact_current[k]:11.4f} {cottrell[k]:10.4f} "
        f"{charge[1 + k]:16.5f} {exact_charge[k]:11.5f} {2 * electrons * faraday * bulk * np.sqrt(diffusivity * times[1 + k] / np.pi):9.5f}"
    )
print(
    f"largest charge deviation from the series from 1e-4 s on: {np.abs(charge[1:] / exact_charge - 1)[compared].max():.2%}"
)
print(
    f"largest current deviation from the series from 1e-4 s on: {np.abs(current[1:] / exact_current - 1)[compared].max():.2%}"
)
departure = times[1:][compared][np.argmax((cottrell / exact_current)[compared] > 1.01)]
print(
    f"Cottrell overestimates the current by more than 1 % from t = {departure:.3f} s on (D t / L² = {diffusivity * departure / thickness**2:.3f})"
)
print(
    f"charge after {end_time:.0f} s: {charge[-1] / (electrons * faraday * bulk * thickness):.4f} of n F c0 L"
)

# ## Figure
#
# The depletion profiles, the current on logarithmic axes (Cottrell's $t^{-1/2}$ law and
# the exponential decay of the exhausted layer), and the charge approaching the
# coulometric limit.

positions = 0.5 * (faces[1:] + faces[:-1])
profiles = {
    f"t = {t:g} s": np.r_[
        0.0,
        np.asarray(
            result.field(model.concentration("O")).evaluate(
                np.column_stack((positions, 0 * positions, 0 * positions)),
                time=times[np.argmin(np.abs(times - t))],
            )
        ).reshape(-1),
    ]
    for t in (0.01, 0.1, 0.5, 1.5, 5.0)
}
fig = plots.figure(1, 3, size=(17, 5))
plots.line(
    np.r_[0.0, positions] * 1e6,
    profiles,
    xlabel="distance from the electrode x [µm]",
    ylabel="concentration of O [mol/m³]",
    title="Depletion of the layer",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    points={"OneSimulation": (times[1:][compared][::3], current[1:][compared][::3])},
    reference={
        "thin-layer series": (times[1:][compared], exact_current[compared]),
        "Cottrell": (times[1:][compared], cottrell[compared]),
    },
    scale="loglog",
    xlabel="time t [s]",
    ylabel="current density i [A/m²]",
    title="Current transient",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    points={"OneSimulation": (times[1:][compared][::3], charge[1:][compared][::3])},
    reference={
        "thin-layer series": (times[1:][compared], exact_charge[compared]),
        "Anson (semi-infinite)": (
            times[1:][compared],
            2
            * electrons
            * faraday
            * bulk
            * np.sqrt(diffusivity * times[1:][compared] / np.pi),
        ),
    },
    hlines={"n F c0 L": electrons * faraday * bulk * thickness},
    ylimits=(0.0, 6.0),
    scale="logx",
    legend="upper left",
    xlabel="time t [s]",
    ylabel="charge q [C/m²]",
    title="Charge: exhaustive electrolysis",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Make the layer ten times thicker and the Cottrell regime extends a hundred times
# longer: the classical semi-infinite chronoamperometry. Replace the zero surface
# concentration by a Butler-Volmer flux in an `OutwardFluxBoundary` to see the
# kinetically limited current at small overpotentials. With a second species R and a
# reverse potential step the same model gives double-step chronoamperometry.
