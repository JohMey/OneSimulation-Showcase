"""OneSimulation example: cyclotron motion and E x B drift of an ion in uniform fields.
Requires the installed, activated OneSimulation runtime.

A proton gyrates on a helix in a uniform magnetic field and drifts sideways when an electric field
is added. The trajectories integrated by OneSimulation with three time integrators are checked
against the closed-form Larmor radius, cyclotron frequency, drift velocity and orbit.
"""

# # Ion cyclotron motion
#
# A charged particle of mass $m$ and charge $q$ in uniform fields obeys the Lorentz force law
#
# $$m\,\frac{d\mathbf v}{dt}=q\,(\mathbf E+\mathbf v\times\mathbf B),\qquad \frac{d\mathbf x}{dt}=\mathbf v .$$
#
# With $\mathbf B=B\,\hat{\mathbf z}$ alone, the velocity across the field rotates at the cyclotron
# frequency while the velocity along the field stays constant: the particle moves on a helix. An
# additional electric field across $\mathbf B$ makes the centre of the circle drift perpendicular
# to both fields, independent of charge and mass.
#
# | quantity | value |
# |---|---|
# | particle | proton, $m=1.6726\times10^{-27}$ kg, $q=e$ |
# | magnetic field | $B=0.1$ T along $z$ |
# | initial velocity | $v_\perp=100$ km/s along $x$, $v_\parallel=20$ km/s along $z$ |
# | electric field (second run) | $E=2$ kV/m along $x$ |
#
# `ChargedParticle` declares the six equations of motion for position and velocity as a lumped
# (0D) model in the uniform fields; OneSimulation integrates them in time like any other transient
# model, so the integrator is a free choice.
#
# **Reference** (F. F. Chen, *Introduction to Plasma Physics and Controlled Fusion*, 3rd ed.,
# Springer 2016, §2.2; J. D. Jackson, *Classical Electrodynamics*, 3rd ed., §12.3).
#
# $$\omega_c=\frac{|q|B}{m},\qquad r_L=\frac{m v_\perp}{|q|B},\qquad
#   \mathbf v_E=\frac{\mathbf E\times\mathbf B}{B^2}.$$
#
# In complex notation $w=x+\mathrm iy$ a positive ion starting at $w_0$ with transverse velocity
# $u_0$ follows $w(t)=w_0+v_Et+\mathrm i\,(u_0-v_E)\,(e^{-\mathrm i\omega_ct}-1)/\omega_c$, where
# $v_E$ is the drift velocity written as a complex number.

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler, ClassicalRungeKutta4
from onesim.physics.plasma import ChargedParticle
from onesim.results import oscillation_fit
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.constants import e, m_p

theme.set()

# ## Parameters
#
# The proton circles 1.52 million times per second on a radius of about 10.4 mm.

B, v_perp, v_par, E = 0.1, 1.0e5, 2.0e4, 2.0e3
omega = e * B / m_p
period, larmor, drift = 2 * np.pi / omega, m_p * v_perp / (e * B), E / B
print(
    f"cyclotron frequency {omega / (2 * np.pi) / 1e6:.4f} MHz, Larmor radius {1e3 * larmor:.4f} mm, E x B drift {drift / 1e3:.1f} km/s"
)

# ## Model
#
# One problem per field configuration. The particle starts at $(0, r_L, 0)$ so that the circle of
# the first run is centred on the axis. Ten gyration periods are integrated with 40 steps per
# period by the classical Runge-Kutta method, the backward Euler method and the second-order BDF
# method; the implicit methods use Newton's method with the analytic Jacobian.

domain = LumpedDomain()
fields = {"B only": (0.0, 0.0, 0.0), "E x B": (E, 0.0, 0.0)}
particles = {
    name: ChargedParticle(
        domain=domain.system,
        mass=m_p,
        charge=e,
        electric=electric,
        magnetic=(0.0, 0.0, B),
    )
    for name, electric in fields.items()
}
start = (0.0, larmor, 0.0, v_perp, 0.0, v_par)
problems = {
    name: Problem(
        name=f"Proton, {name}",
        geometry=domain,
        physics=(model,),
        initial_conditions=tuple(
            InitialValue(ref, value)
            for ref, value in zip(
                (
                    model.x,
                    model.y,
                    model.z,
                    model.velocity_x,
                    model.velocity_y,
                    model.velocity_z,
                ),
                start,
            )
        ),
    )
    for name, model in particles.items()
}
newton = Newton(linear_solver=SparseDirect(), jacobian="analytic")
integrators = {
    "classical Runge-Kutta": Numerics(
        spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4()
    ),
    "backward Euler": Numerics(
        spatial=LumpedParameters(),
        time_integrator=BackwardEuler(),
        nonlinear_solver=newton,
    ),
    "BDF2": Numerics(
        spatial=LumpedParameters(),
        time_integrator=BDF(order=2),
        nonlinear_solver=newton,
    ),
}
runs = {
    (name, method): Simulation(
        problem=problems[name], mesh=LumpedMesh(domain), numerics=numerics
    ).run(start_time=0.0, end_time=10 * period, time_step=period / 40)
    for name in problems
    for method, numerics in integrators.items()
}
orbits = {
    key: np.array(
        [
            result.field(ref).time_series()[1]
            for ref in (
                particles[key[0]].x,
                particles[key[0]].y,
                particles[key[0]].z,
                particles[key[0]].velocity_x,
                particles[key[0]].velocity_y,
            )
        ]
    )
    for key, result in runs.items()
}
times = (
    runs["B only", "classical Runge-Kutta"]
    .field(particles["B only"].x)
    .time_series()[0]
)

# ## Comparison with the closed forms
#
# The frequency is fitted to the rotating transverse velocity; the guiding centre is the mean
# position over the first and over the last period; the radius is the mean distance from it.

exact = (
    lambda t, drift_velocity: 1j * larmor
    + drift_velocity * t
    + 1j * (v_perp - drift_velocity) * (np.exp(-1j * omega * t) - 1) / omega
)
print(
    "B only: integrator       frequency [MHz]   Larmor radius [mm]   |v_perp| after 10 periods   largest orbit error [mm]"
)
for method in integrators:
    x, y, z, vx, vy = orbits["B only", method]
    centre = np.mean(x[:40] + 1j * y[:40])
    print(
        f"  {method:22s} {abs(oscillation_fit(times, vx + 1j * vy).frequency) / (2 * np.pi) / 1e6:15.5f} "
        f"{1e3 * np.mean(np.abs(x[:40] + 1j * y[:40] - centre)):20.4f} {np.hypot(vx[-1], vy[-1]) / v_perp:27.5f} "
        f"{1e3 * np.abs(x + 1j * y - exact(times, 0)).max():25.3f}"
    )
print(
    f"  {'closed form':22s} {omega / (2 * np.pi) / 1e6:15.5f} {1e3 * larmor:20.4f} {1.0:27.5f}"
)
print(
    "E x B:  integrator       drift of the guiding centre [km/s]   largest orbit error [mm]"
)
for method in integrators:
    x, y = orbits["E x B", method][:2]
    first, last = (np.mean(x[sl] + 1j * y[sl]) for sl in (slice(0, 40), slice(-41, -1)))
    print(
        f"  {method:22s} {abs(last - first) / (9 * period) / 1e3:28.3f} {1e3 * np.abs(x + 1j * y - exact(times, -1j * drift)).max():26.3f}"
    )
print(f"  {'closed form E/B':22s} {drift / 1e3:28.3f}")
x, y, z = orbits["B only", "classical Runge-Kutta"][:3]
print(
    f"pitch of the helix: {1e3 * (z[40] - z[0]):.4f} mm per turn, closed form v_par 2 pi / omega_c = {1e3 * v_par * period:.4f} mm"
)

# ## Figure

fine = np.linspace(0, 10 * period, 4000)
circle, cycloid = exact(fine, 0), exact(fine, -1j * drift)
fig = plots.figure(1, 3, size=(18, 5.2))
plots.line(
    points={
        method: (
            1e3 * orbits["B only", method][0][::3],
            1e3 * orbits["B only", method][1][::3],
        )
        for method in integrators
    },
    reference={"exact circle": (1e3 * circle.real, 1e3 * circle.imag)},
    xlabel="x [mm]",
    ylabel="y [mm]",
    title="Gyration across B, 10 periods",
    legend="center",
    ax=fig[0],
)
plots.line(
    1e6 * times,
    {
        method: np.hypot(*orbits["B only", method][3:5]) / v_perp
        for method in integrators
    },
    hlines={"exact": 1.0},
    ylimits=(0, 1.1),
    xlabel="time [µs]",
    ylabel="|v_perp| / v_perp(0)",
    title="Kinetic energy across B",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    points={
        "classical Runge-Kutta": (
            1e3 * orbits["E x B", "classical Runge-Kutta"][0][:121],
            1e3 * orbits["E x B", "classical Runge-Kutta"][1][:121],
        )
    },
    reference={"exact orbit": (1e3 * cycloid.real[:1201], 1e3 * cycloid.imag[:1201])},
    xlabel="x [mm]",
    ylabel="y [mm]",
    title="E x B drift with E along x, 3 periods",
    legend="lower left",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Halve the time step and watch the backward Euler damping shrink only linearly while the
# Runge-Kutta orbit error drops sixteen-fold. Give the particle the charge $-e$ and the mass of an
# electron: it gyrates the other way and 1836 times faster, but drifts with the same $\mathbf v_E$.
# A field that varies in space calls for the guiding-centre models `UniformGuidingCenter` and
# `NonuniformGuidingCenter`.
