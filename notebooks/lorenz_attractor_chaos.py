"""OneSimulation example: the Lorenz system, from a stable focus to the strange attractor.
Requires the installed, activated OneSimulation runtime.

The three Lorenz equations are built from signal blocks and integrated in time. The
result is checked against the closed-form fixed points, the largest Lyapunov exponent
of the classical attractor, and the sensitivity of two almost identical starts.
"""

# # The Lorenz system: from a stable focus to the strange attractor
#
# In 1963 E. N. Lorenz reduced a model of thermal convection between two plates to
# three ordinary differential equations,
#
# $$\dot x=\sigma\,(y-x),\qquad \dot y=x\,(\rho-z)-y,\qquad \dot z=x\,y-\beta\,z,$$
#
# with the Prandtl number $\sigma$, the scaled Rayleigh number $\rho$ and the geometry
# factor $\beta$. For $\sigma=10$, $\beta=8/3$ and $\rho=28$ the solution never settles
# and never repeats: it winds around two unstable foci on the butterfly-shaped
# strange attractor, and two starts that differ in the ninth digit soon follow
# completely different paths.
#
# OneSimulation assembles the system as a signal-flow graph: three integrator blocks
# hold $x$, $y$, $z$, a weighted sum forms $\dot x$, and two blocks evaluate the
# nonlinear right-hand sides. The same graph also carries the linearised (tangent)
# equations for a small perturbation $(\xi,\eta,\zeta)$,
#
# $$\dot\xi=\sigma(\eta-\xi),\qquad \dot\eta=(\rho-z)\,\xi-\eta-x\,\zeta,\qquad
#   \dot\zeta=y\,\xi+x\,\eta-\beta\,\zeta,$$
#
# whose length grows on average as $e^{\lambda_1 t}$. The graph is integrated with the
# classical fourth-order Runge-Kutta method.
#
# **Reference.** E. N. Lorenz, *Deterministic nonperiodic flow*, J. Atmos. Sci. 20,
# 130-141 (1963); J. C. Sprott, *Chaos and Time-Series Analysis*, Oxford University
# Press 2003. Besides the origin, the system has the two fixed points
#
# $$C_\pm=\left(\pm\sqrt{\beta(\rho-1)},\ \pm\sqrt{\beta(\rho-1)},\ \rho-1\right),$$
#
# which are stable foci below the Hopf value $\rho_H=\sigma(\sigma+\beta+3)/(\sigma-\beta-1)\approx24.74$
# and unstable above it. For $\rho=28$ the largest Lyapunov exponent is
# $\lambda_1\approx0.9056$, and the phase volume contracts at the constant rate
# $\sigma+1+\beta$.

import numpy as np
from onesim.numerics.time import DEFAULT_TIME_INTEGRATOR_REGISTRY
from onesim.physics.systems import ControlSignalFlowProvider, SignalBlock
from onesim.visualization import plots, theme

theme.set()


# ## Parameters
#
# The classical values. Three runs: the chaotic case from $(1,1,1)$, a twin started
# $10^{-9}$ away in $x$, and a subcritical case $\rho=15$ that must spiral into a fixed
# point. The tangent vector starts as a unit vector along $x$.

sigma, beta, lyapunov_published = 10.0, 8.0 / 3.0, 0.9056
hopf = sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)
cases = {"chaotic": (28.0, 1.0), "twin": (28.0, 1.0 + 1e-9), "steady": (15.0, 1.0)}
print(
    f"Hopf value ρ_H = {hopf:.3f};  fixed point for ρ = 15: x = ±{np.sqrt(beta * 14.0):.5f}, z = 14"
)

# ## Signal-flow graph and simulation
#
# 150 time units with a step of 0.002. The blocks `dy` and `dz` evaluate the
# nonlinear rates from their inputs; the tangent blocks use the current state.

runs = {
    name: ControlSignalFlowProvider(
        blocks=(
            SignalBlock(
                "x", "integrator", inputs=("dx",), parameters={"initial_value": x0}
            ),
            SignalBlock(
                "y", "integrator", inputs=("dy",), parameters={"initial_value": 1.0}
            ),
            SignalBlock(
                "z", "integrator", inputs=("dz",), parameters={"initial_value": 1.0}
            ),
            SignalBlock(
                "dx", "sum", inputs=("y", "x"), parameters={"weights": (sigma, -sigma)}
            ),
            SignalBlock(
                "dy",
                "custom",
                inputs=("x", "y", "z"),
                parameters={"rho": rho},
                custom_evaluator=lambda s, t, p: s[0] * (p["rho"] - s[2]) - s[1],
                custom_derivative=lambda *a: (),
            ),
            SignalBlock(
                "dz",
                "custom",
                inputs=("x", "y", "z"),
                custom_evaluator=lambda s, t, p: s[0] * s[1] - beta * s[2],
                custom_derivative=lambda *a: (),
            ),
            SignalBlock(
                "xi", "integrator", inputs=("dxi",), parameters={"initial_value": 1.0}
            ),
            SignalBlock("eta", "integrator", inputs=("deta",)),
            SignalBlock("zeta", "integrator", inputs=("dzeta",)),
            SignalBlock(
                "dxi",
                "sum",
                inputs=("eta", "xi"),
                parameters={"weights": (sigma, -sigma)},
            ),
            SignalBlock(
                "deta",
                "custom",
                inputs=("x", "z", "xi", "eta", "zeta"),
                parameters={"rho": rho},
                custom_evaluator=lambda s, t, p: (p["rho"] - s[1]) * s[2]
                - s[3]
                - s[0] * s[4],
                custom_derivative=lambda *a: (),
            ),
            SignalBlock(
                "dzeta",
                "custom",
                inputs=("x", "y", "xi", "eta", "zeta"),
                custom_evaluator=lambda s, t, p: s[1] * s[2]
                + s[0] * s[3]
                - beta * s[4],
                custom_derivative=lambda *a: (),
            ),
        ),
        final_time=150.0,
        time_step=0.002,
        time_integrator=DEFAULT_TIME_INTEGRATOR_REGISTRY.resolve_id("classical_rk4"),
    ).execute()
    for name, (rho, x0) in cases.items()
}
times = np.asarray(runs["chaotic"].times)
state = {
    name: np.array([run.signals[k] for k in ("x", "y", "z")])
    for name, run in runs.items()
}
print("evaluation order:", " -> ".join(runs["chaotic"].evaluation_order))

# ## Comparison
#
# The subcritical run must end on $C_\pm$. The largest Lyapunov exponent is the slope of
# $\ln|(\xi,\eta,\zeta)|$ over time from $t=15$ on: the start $(1,1,1)$ lies close to the
# unstable focus $C_-$, and the trajectory first spirals slowly away from it before it
# reaches the attractor. The twin trajectory measures the same growth directly, until
# the separation reaches the size of the attractor.

steady = state["steady"][:, -1]
print(
    f"ρ = 15 after 150 time units: |x| = {abs(steady[0]):.5f}, |y| = {abs(steady[1]):.5f}, z = {steady[2]:.5f};  "
    f"C±: {np.sqrt(beta * 14.0):.5f}, {np.sqrt(beta * 14.0):.5f}, 14.00000"
)
tangent = np.log(
    np.sqrt(
        sum(np.asarray(runs["chaotic"].signals[k]) ** 2 for k in ("xi", "eta", "zeta"))
    )
)
late = times >= 15.0
lyapunov = np.polyfit(times[late], tangent[late], 1)[0]
print(
    f"largest Lyapunov exponent (ρ = 28): {lyapunov:.4f}, published {lyapunov_published:.4f}  "
    f"({100 * (lyapunov / lyapunov_published - 1):+.1f} %)"
)
separation = np.linalg.norm(state["twin"] - state["chaotic"], axis=0)
diverged = times[np.argmax(separation > 1.0)]
print(
    f"twin trajectories 1e-9 apart: separation exceeds 1 after t = {diverged:.1f}, largest separation {separation.max():.1f}"
)
print(
    f"mean of z on the attractor (t > 15): {state['chaotic'][2][late].mean():.2f};  "
    f"volume contraction rate σ + 1 + β = {sigma + 1 + beta:.3f}"
)

# ## Figure
#
# The attractor in the $x$-$z$ plane with the two unstable foci; the two twin
# trajectories; the logarithmic growth of the tangent vector and of the twin
# separation against the published exponent; and the subcritical run spiralling into
# its fixed point.

foci = np.sqrt(beta * 27.0)
fig = plots.figure(2, 2, size=(14, 10))
plots.line(
    y={
        "trajectory, ρ = 28": (
            state["chaotic"][0][late & (times <= 60.0)],
            state["chaotic"][2][late & (times <= 60.0)],
        )
    },
    points={"foci C±": (np.array([-foci, foci]), np.array([27.0, 27.0]))},
    xlabel="x",
    ylabel="z",
    title="Strange attractor (x-z projection)",
    legend="upper center",
    ax=fig[0],
)
plots.line(
    times,
    {"start x0 = 1": state["chaotic"][0], "start x0 = 1 + 1e-9": state["twin"][0]},
    vlines={"separation > 1": diverged},
    xlimits=(0.0, 40.0),
    xlabel="time t",
    ylabel="x(t)",
    title="Sensitivity to the initial condition",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    times,
    {
        "tangent vector ln|ξ|": tangent,
        "twin separation ln|Δ| + ln 1e9": np.log(separation) + np.log(1e9),
    },
    reference={
        "slope λ1 = 0.9056": lyapunov_published * times
        + (tangent[late][0] - lyapunov_published * 15.0)
    },
    xlabel="time t",
    ylabel="logarithmic growth",
    title=f"Largest Lyapunov exponent: {lyapunov:.3f}",
    legend="upper left",
    ax=fig[2],
)
plots.line(
    times,
    {"x": state["steady"][0], "z": state["steady"][2]},
    hlines={
        "fixed point x": np.sign(steady[0]) * np.sqrt(beta * 14.0),
        "fixed point z = ρ - 1": 14.0,
    },
    xlimits=(0.0, 30.0),
    xlabel="time t",
    ylabel="state",
    title="ρ = 15 < ρ_H: spiral into the fixed point",
    legend="center right",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Scan `rho` from 20 to 30 and watch the stable foci lose stability at $\rho_H$; just
# below it, long chaotic transients precede the final spiral (transient chaos). At
# $\rho\approx 99.6$ and $\rho\approx 160$ the system leaves chaos for periodic
# windows. Replace the classical Runge-Kutta integrator by an adaptive method from
# the same registry and compare when the twin trajectories separate: for a chaotic
# system the step size only shifts that time, it never removes the divergence.
