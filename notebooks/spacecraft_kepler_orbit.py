"""OneSimulation example: a spacecraft on an elliptical transfer orbit around the Earth.
Requires the installed, activated OneSimulation runtime.

The equations of motion of a spacecraft in the gravity field of the Earth are built
from signal blocks and integrated over three revolutions. The result is checked
against Kepler's laws: the ellipse, the orbital period, and the conservation of
energy and angular momentum.
"""

# # Spacecraft on an elliptical transfer orbit
#
# A spacecraft leaves a low Earth parking orbit on a geostationary transfer orbit: at
# the perigee, 250 km above the surface, it has just the speed that carries it out to
# the geostationary radius of 42 164 km. Only the Earth's central gravity acts, so the
# motion stays in the orbital plane and obeys
#
# $$\ddot{\mathbf r}=-\frac{\mu}{r^3}\,\mathbf r,\qquad \mathbf r=(x,y),\quad r=|\mathbf r|,$$
#
# with the gravitational parameter of the Earth $\mu=GM=3.986\,004\times10^{14}$ m³/s².
# OneSimulation assembles the equations as a signal-flow graph: four integrator blocks
# hold the position and the velocity, two blocks evaluate the gravitational
# acceleration, and the graph is integrated with the classical fourth-order
# Runge-Kutta method at a fixed step.
#
# **Reference.** Kepler's laws for the two-body problem (H. D. Curtis, *Orbital
# Mechanics for Engineering Students*, 3rd ed., Butterworth-Heinemann 2014, ch. 2).
# With the perigee and apogee radii $r_p$, $r_a$, the semi-major axis and the
# eccentricity are $a=(r_p+r_a)/2$, $e=(r_a-r_p)/(r_a+r_p)$, and
#
# $$r(\theta)=\frac{a(1-e^2)}{1+e\cos\theta},\qquad T=2\pi\sqrt{\frac{a^3}{\mu}},\qquad
#   \varepsilon=\frac{v^2}{2}-\frac{\mu}{r}=-\frac{\mu}{2a},\qquad h=|\mathbf r\times\mathbf v|=\sqrt{\mu a(1-e^2)}.$$
#
# The vis-viva equation gives the perigee speed $v_p=\sqrt{\mu\,(2/r_p-1/a)}$.

import numpy as np
from onesim.numerics.time import DEFAULT_TIME_INTEGRATOR_REGISTRY
from onesim.physics.systems import ControlSignalFlowProvider, SignalBlock
from onesim.results.sampling import crossing_times
from onesim.visualization import plots, theme

theme.set()


# ## Orbit
#
# The spacecraft starts at the perigee on the $x$ axis with the vis-viva speed in the
# $y$ direction and is followed for three revolutions.

mu, earth_radius = 3.986004418e14, 6.378137e6
perigee, apogee = earth_radius + 250e3, 42.164e6
semi_major, eccentricity = 0.5 * (perigee + apogee), (apogee - perigee) / (
    apogee + perigee
)
period = 2 * np.pi * np.sqrt(semi_major**3 / mu)
perigee_speed = np.sqrt(mu * (2 / perigee - 1 / semi_major))
energy_exact, momentum_exact = -mu / (2 * semi_major), np.sqrt(
    mu * semi_major * (1 - eccentricity**2)
)
print(
    f"a = {semi_major / 1e3:.1f} km, e = {eccentricity:.5f}, period T = {period:.1f} s ({period / 3600:.3f} h), "
    f"perigee speed {perigee_speed:.2f} m/s"
)

# ## Signal-flow graph and step study
#
# The same graph is integrated with steps of 120, 60, 30 and 15 s; the 15 s run is the
# result shown below. Near the perigee the spacecraft covers about 150 km per step.

steps = (120.0, 60.0, 30.0, 15.0)
runs = {
    dt: ControlSignalFlowProvider(
        blocks=(
            SignalBlock(
                "x", "integrator", inputs=("vx",), parameters={"initial_value": perigee}
            ),
            SignalBlock("y", "integrator", inputs=("vy",)),
            SignalBlock("vx", "integrator", inputs=("ax",)),
            SignalBlock(
                "vy",
                "integrator",
                inputs=("ay",),
                parameters={"initial_value": perigee_speed},
            ),
            SignalBlock(
                "ax",
                "custom",
                inputs=("x", "y"),
                custom_evaluator=lambda s, t, p: -mu * s[0] / np.hypot(*s) ** 3,
                custom_derivative=lambda *a: (),
            ),
            SignalBlock(
                "ay",
                "custom",
                inputs=("x", "y"),
                custom_evaluator=lambda s, t, p: -mu * s[1] / np.hypot(*s) ** 3,
                custom_derivative=lambda *a: (),
            ),
        ),
        final_time=3 * period,
        time_step=dt,
        time_integrator=DEFAULT_TIME_INTEGRATOR_REGISTRY.resolve_id("classical_rk4"),
    ).execute()
    for dt in steps
}
orbit = {
    dt: {k: np.asarray(run.signals[k]) for k in ("x", "y", "vx", "vy")}
    for dt, run in runs.items()
}
energy = {
    dt: 0.5 * (s["vx"] ** 2 + s["vy"] ** 2) - mu / np.hypot(s["x"], s["y"])
    for dt, s in orbit.items()
}
momentum = {dt: s["x"] * s["vy"] - s["y"] * s["vx"] for dt, s in orbit.items()}
times, s = np.asarray(runs[15.0].times), orbit[15.0]
radius, anomaly = np.hypot(s["x"], s["y"]), np.arctan2(s["y"], s["x"])

# ## Comparison with Kepler's laws
#
# The period is the time between successive perigee passages, where $y$ crosses zero
# upwards; the orbit radius is compared with the conic section at the same true
# anomaly $\theta$; energy and angular momentum must stay at their initial values.

ellipse = (
    lambda theta: semi_major
    * (1 - eccentricity**2)
    / (1 + eccentricity * np.cos(theta))
)
passages = np.asarray(
    crossing_times(times, s["y"], 0.0, direction="rising", after=0.5 * period)
)
print(
    f"perigee passages after {', '.join(f'{t:.1f}' for t in passages)} s;  "
    f"period {np.mean(np.diff(np.r_[0.0, passages])):.2f} s, Kepler {period:.2f} s"
)
print(
    f"apogee radius {radius.max() / 1e3:.2f} km (exact {apogee / 1e3:.2f} km),  "
    f"largest deviation from the ellipse {np.abs(radius / ellipse(anomaly) - 1).max():.1e}"
)
print(
    f"specific energy {energy[15.0][0] / 1e6:.4f} MJ/kg (exact {energy_exact / 1e6:.4f}),  "
    f"angular momentum {momentum[15.0][0] / 1e6:.2f} km²/s (exact {momentum_exact / 1e6:.2f})"
)
print(
    "  step [s]   energy drift after 3 revolutions   largest energy error   angular-momentum drift"
)
for dt in steps:
    print(
        f"{dt:9.0f} {abs(energy[dt][-1] / energy_exact - 1):24.2e} {np.abs(energy[dt] / energy_exact - 1).max():22.2e} "
        f"{abs(momentum[dt][-1] / momentum_exact - 1):24.2e}"
    )
drift = [abs(energy[dt][-1] / energy_exact - 1) for dt in steps]
largest = [np.abs(energy[dt] / energy_exact - 1).max() for dt in steps]
print(
    f"observed order between the two finest steps: {np.log2(largest[-2] / largest[-1]):.2f} for the largest energy error, "
    f"{np.log2(drift[-2] / drift[-1]):.2f} for the drift after whole revolutions"
)

# The classical Runge-Kutta method has global order four, and so does the largest energy
# error along the orbit. Nearly all of the error is made during the fast perigee
# passages, and right at the perigee part of it cancels, so the drift evaluated after
# whole revolutions falls even faster.

# ## Figure
#
# The computed orbit over the Kepler ellipse with the Earth to scale; the distance from
# the Earth's centre over time (dotted verticals: the perigee passages); the relative drift of energy and angular momentum; and
# the energy drift against the step size.

theta = np.linspace(0.0, 2 * np.pi, 721)
fig = plots.figure(2, 2, size=(14, 10.5))
plots.line(
    y={
        "computed orbit": (s["x"][::4] / 1e6, s["y"][::4] / 1e6),
        "Earth": (
            earth_radius * np.cos(theta) / 1e6,
            earth_radius * np.sin(theta) / 1e6,
        ),
    },
    reference={
        "Kepler ellipse": (
            ellipse(theta) * np.cos(theta) / 1e6,
            ellipse(theta) * np.sin(theta) / 1e6,
        )
    },
    xlabel="x [1000 km]",
    ylabel="y [1000 km]",
    title="Geostationary transfer orbit",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    times / 3600,
    {"computed r(t)": radius / 1e6},
    hlines={"apogee r_a": apogee / 1e6, "perigee r_p": perigee / 1e6},
    vlines=passages / 3600,
    xlabel="time [h]",
    ylabel="distance from the Earth's centre [1000 km]",
    title="Radius over three revolutions",
    legend="center right",
    ax=fig[1],
)
plots.line(
    times / 3600,
    {
        "energy ε/ε0 - 1": energy[15.0] / energy_exact - 1,
        "angular momentum h/h0 - 1": momentum[15.0] / momentum_exact - 1,
    },
    xlabel="time [h]",
    ylabel="relative drift",
    title="Conservation laws, step 15 s",
    legend="center",
    ax=fig[2],
)
plots.convergence(
    np.array(steps),
    {
        "largest energy error": np.array(largest),
        "drift after 3 revolutions": np.array(drift),
    },
    order=(4, 5),
    xlabel="time step [s]",
    ylabel="relative energy drift",
    title="Energy drift against the step size",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Add a third block pair for the $J_2$ oblateness acceleration and watch the line of
# apsides rotate; or a thrust block that fires at apogee to circularise the orbit at
# the geostationary radius (the classical Hohmann manoeuvre, $\Delta v\approx1.47$ km/s).
# With a symplectic integrator instead of Runge-Kutta the energy error stays bounded
# instead of drifting, a property that matters for very long propagations.
