"""OneSimulation example: parameter estimation for a non-ideal stirred reactor (dead-zone model).
Requires the installed, activated OneSimulation runtime.

A real stirred tank is poorly mixed in a part of its volume. It is modelled as two ideal tanks,
a well-mixed zone and a dead zone that exchange liquid. The active fraction and the exchange flow
are recovered from a tracer step experiment by a Levenberg-Marquardt least-squares fit in which
every residual evaluation is a simulation of the two tanks; the time integration is checked
against the exact matrix-exponential solution, the fitted values against the values that
generated the synthetic data.
"""

# # Parameter estimation for a non-ideal stirred reactor
#
# A stirred tank of volume $V=1$ m³ is fed with $q=1$ l/s of liquid, so its mean residence time is
# $\tau=V/q=1000$ s. An ideal stirred tank would answer a tracer step at the inlet with
# $F(t)=1-e^{-t/\tau}$. Real tanks deviate: corners, baffles and the region behind the impeller are
# poorly mixed. The classical **dead-zone model** splits the tank into a well-mixed zone of volume
# $\alpha V$ and a dead zone of volume $(1-\alpha)V$ that exchange the flow $\beta q$:
#
# $$\alpha V\frac{dc_1}{dt}=q\,(c_\mathrm{in}-c_1)+\beta q\,(c_2-c_1),\qquad
#   (1-\alpha)V\frac{dc_2}{dt}=\beta q\,(c_1-c_2).$$
#
# The outlet concentration is $c_1$. The tank responds faster than an ideal tank at first (only
# the active zone is flushed) and slower later (the dead zone empties slowly).
#
# OneSimulation builds the two balances as a signal-flow graph (two integrators fed by weighted
# sums) and integrates it with the classical Runge-Kutta method. The `SimulationStudyService`
# turns the model into a least-squares problem: `fit` runs the graph for trial values of
# $(\alpha,\beta)$, forms the residuals of the outlet response divided by the noise level and
# minimises their sum of squares with the Levenberg-Marquardt method; it returns the estimates
# with their linearised covariance. The best ideal tank is fitted the same way, with one
# first-order block.
#
# | quantity | value |
# |---|---|
# | tank volume $V$ | 1 m³ |
# | feed $q$ | 1 l/s |
# | tracer step | $c_\mathrm{in}=1$ for $t>0$ |
# | samples | every 60 s over 4 h, 1 % measurement noise |
#
# The measurements in this notebook are **synthetic**: they are generated with $\alpha=0.65$ and
# $\beta=0.10$ and disturbed with random noise, so the fit can be judged against known values.
#
# **Reference.** Levenspiel, *Chemical Reaction Engineering*, 3rd ed., Ch. 12 (models for non-ideal
# flow, the dead-zone and exchange models); Fogler, *Elements of Chemical Reaction Engineering*,
# Ch. 18. The model is linear, $\dot{\mathbf c}=A\,\mathbf c+\mathbf b$, and its step response is
# exactly $\mathbf c(t)=\bigl(I-e^{At}\bigr)\,\mathbf 1$.

import numpy as np
from onesim.numerics.time import DEFAULT_TIME_INTEGRATOR_REGISTRY
from onesim.physics.systems import ControlSignalFlowProvider, SignalBlock
from onesim.simulation.studies import SimulationStudyService, StudyParameter
from onesim.visualization import plots, theme
from scipy.linalg import expm

theme.set()

# ## Model
#
# `tank(values)` builds the signal-flow graph for one pair $(\alpha,\beta)$; the study runs it for
# four hours with steps of 5 s and reads the outlet at the sampling times.

volume, feed = 1.0, 1e-3
residence_time = volume / feed
samples = np.arange(60.0, 4 * 3600.0 + 1, 60.0)
rk4 = DEFAULT_TIME_INTEGRATOR_REGISTRY.resolve_id("classical_rk4")
tank = lambda values: ControlSignalFlowProvider(
    blocks=(
        SignalBlock("inlet", "constant", parameters={"value": 1.0}),
        SignalBlock("active", "integrator", inputs=("d_active",)),
        SignalBlock("dead", "integrator", inputs=("d_dead",)),
        SignalBlock(
            "d_active",
            "sum",
            inputs=("inlet", "active", "dead"),
            parameters={
                "weights": (
                    feed / (values["alpha"] * volume),
                    -(1 + values["beta"]) * feed / (values["alpha"] * volume),
                    values["beta"] * feed / (values["alpha"] * volume),
                )
            },
        ),
        SignalBlock(
            "d_dead",
            "sum",
            inputs=("active", "dead"),
            parameters={
                "weights": (
                    values["beta"] * feed / ((1 - values["alpha"]) * volume),
                    -values["beta"] * feed / ((1 - values["alpha"]) * volume),
                )
            },
        ),
    ),
    final_time=samples[-1],
    time_step=5.0,
    time_integrator=rk4,
)
outlet = lambda run, block: {
    f"c_{i}": c for i, c in enumerate(np.interp(samples, run.times, run.signals[block]))
}
study = SimulationStudyService(
    (StudyParameter("alpha", 0.05, 0.99), StudyParameter("beta", 0.01, 5.0)),
    tank,
    lambda run: outlet(run, "active"),
    solve=lambda graph: graph.execute(),
)

# ## Synthetic experiment
#
# The exact response for the true parameters, disturbed with normally distributed noise of 1 % of
# the step height.

true = {"alpha": 0.65, "beta": 0.10}
exact = lambda t, a, b: np.array(
    [
        1
        - (
            expm(
                np.array([[-(1 + b), b], [b * a / (1 - a), -b * a / (1 - a)]])
                * feed
                / (a * volume)
                * s
            )
            @ np.ones(2)
        )[0]
        for s in np.atleast_1d(t)
    ]
)
measured = exact(samples, *true.values()) + np.random.default_rng(7).normal(
    0.0, 0.01, samples.size
)

# ## Results
#
# First the signal-flow integration against the exact solution, then the least-squares fits of
# the dead-zone model (start $\alpha=0.5$, $\beta=0.5$) and of the ideal tank.

simulated = study.sweep(combinations=[true]).runs[0].measurements
print(
    f"time integration against the matrix exponential: largest deviation "
    f"{np.abs(np.array([simulated[f'c_{i}'] for i in range(samples.size)]) - exact(samples, *true.values())).max():.1e}"
)
fit = study.fit(
    residuals=lambda m: np.array(
        [(m[f"c_{i}"] - measured[i]) / 0.01 for i in range(samples.size)]
    ),
    start={"alpha": 0.5, "beta": 0.5},
    method="levenberg_marquardt",
    maximum_runs=100,
)
ideal = SimulationStudyService(
    (StudyParameter("tau", 100.0, 5000.0),),
    lambda values: ControlSignalFlowProvider(
        blocks=(
            SignalBlock("inlet", "constant", parameters={"value": 1.0}),
            SignalBlock(
                "outlet",
                "first_order",
                inputs=("inlet",),
                parameters={"time_constant": values["tau"]},
            ),
        ),
        final_time=samples[-1],
        time_step=5.0,
        time_integrator=rk4,
    ),
    lambda run: outlet(run, "outlet"),
    solve=lambda graph: graph.execute(),
).fit(
    residuals=lambda m: np.array(
        [(m[f"c_{i}"] - measured[i]) / 0.01 for i in range(samples.size)]
    ),
    start={"tau": residence_time},
    method="levenberg_marquardt",
    maximum_runs=50,
)
print(
    f"dead-zone fit: {fit.run_count} simulations, converged {fit.converged}, rank {fit.rank}, "
    f"χ² = {fit.residuals @ fit.residuals:.1f} for {fit.degrees_of_freedom} degrees of freedom"
)
print("parameter         true    estimate ± standard error   deviation")
for name, label in (("alpha", "active fraction"), ("beta", "exchange ratio")):
    print(
        f"{label:16s} {true[name]:6.3f} {fit.parameters[name]:10.4f} ± {fit.standard_errors[name]:.4f}       "
        f"{(fit.parameters[name] - true[name]) / fit.standard_errors[name]:+5.2f} σ"
    )
print(
    f"correlation of the two estimates {fit.covariance[0, 1] / np.sqrt(fit.covariance[0, 0] * fit.covariance[1, 1]):+.2f}"
)
for name, result in (("dead-zone model", fit), ("ideal stirred tank", ideal)):
    print(
        f"{name:20s} rms residual {0.01 * np.sqrt(np.mean(result.residuals ** 2)):.4f}  (noise level 0.0100)"
    )
print(
    f"best ideal tank: residence time {ideal.parameters['tau']:.0f} ± {ideal.standard_errors['tau']:.0f} s instead of V/q = {residence_time:.0f} s"
)
print(
    f"well-mixed volume {fit.parameters['alpha'] * volume:.3f} m³, exchange flow {fit.parameters['beta'] * feed * 1e3:.3f} l/s"
)

# ## Figure
#
# The measured response with the fitted dead-zone model and the best ideal tank, the residuals of
# both models, and the sum of squared residuals over a grid of the two parameters (a sweep of the
# same study), whose minimum is the estimate.

grid = SimulationStudyService(
    (
        StudyParameter("alpha", 0.45, 0.85, values=tuple(np.linspace(0.45, 0.85, 13))),
        StudyParameter("beta", 0.02, 0.30, values=tuple(np.linspace(0.02, 0.30, 13))),
    ),
    tank,
    lambda run: outlet(run, "active"),
    solve=lambda graph: graph.execute(),
).sweep()
misfit = np.log10(
    np.array(
        [
            np.sum(
                (
                    np.array([run.measurements[f"c_{i}"] for i in range(samples.size)])
                    - measured
                )
                ** 2
            )
            for run in grid.runs
        ]
    ).reshape(13, 13)
)
fig = plots.figure(1, 3, size=(18, 5))
plots.line(
    samples / 60,
    {
        "dead-zone model, fitted": measured + 0.01 * fit.residuals,
        "ideal stirred tank, fitted": measured + 0.01 * ideal.residuals,
    },
    points={"synthetic measurements": (samples[::3] / 60, measured[::3])},
    xlabel="time [min]",
    ylabel="outlet concentration / step height",
    title="Tracer step response",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    points={
        "dead-zone model": (samples / 60, 0.01 * fit.residuals),
        "ideal stirred tank": (samples / 60, 0.01 * ideal.residuals),
    },
    hlines=0.0,
    xlabel="time [min]",
    ylabel="model - measured",
    title="Residuals",
    ax=fig[1],
)
plots.heatmap(
    np.linspace(0.45, 0.85, 13),
    np.linspace(0.02, 0.30, 13),
    misfit,
    contours=12,
    label="log10(sum of squared residuals)",
    aspect="auto",
    xlabel="active fraction α",
    ylabel="exchange ratio β",
    title="Least-squares objective",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Replace the step by a short tracer pulse and fit the residence-time distribution $E(t)$
# instead; add a bypass stream to the model (the Cholette-Cloutier model) and see whether the data
# can tell bypass and dead zone apart; or lower the noise and watch the standard errors shrink
# with its square root.
