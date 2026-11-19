"""OneSimulation example: the Dzhanibekov effect of a spinning T-handle.
Requires the installed, activated OneSimulation runtime.

A T-handle is read from a STEP file; its principal moments of inertia come from the CAD solid.
Euler's equations of the torque-free rigid body are integrated for spins about the three
principal axes: the spin about the intermediate axis flips periodically, the other two are
stable. Energy and angular momentum are conserved, and the flip period is checked against the
exact solution in Jacobi elliptic functions.
"""

# # Dzhanibekov effect: the flipping T-handle
#
# In 1985 the cosmonaut Vladimir Dzhanibekov noticed that a wing nut spinning off its bolt in
# weightlessness turns over every few seconds, keeps spinning and turns back. The effect is the
# **intermediate axis theorem**: a rigid body with principal moments of inertia $I_1<I_2<I_3$
# spins stably about the axes of the smallest and the largest moment, but not about the
# intermediate one. Without torques the angular velocity $\boldsymbol\omega$ in the body frame
# obeys Euler's equations
#
# $$I_1\dot\omega_1=(I_2-I_3)\,\omega_2\omega_3,\qquad I_2\dot\omega_2=(I_3-I_1)\,\omega_3\omega_1,\qquad
#   I_3\dot\omega_3=(I_1-I_2)\,\omega_1\omega_2 ,$$
#
# which conserve the kinetic energy $E=\tfrac12\sum I_i\omega_i^2$ and the magnitude of the
# angular momentum $M=|\,(I_1\omega_1,I_2\omega_2,I_3\omega_3)\,|$.
#
# The body is a T-handle of aluminium bar from the CAD file `dzhanibekov_t_handle.step`, written
# by `dzhanibekov_t_handle_geometry.py` with the CAD objects of OneSimulation. Its principal axes
# are the coordinate axes: $x$ along the crossbar (smallest moment), $y$ along the stem
# (intermediate) and $z$ normal to the plane of the T (largest).
#
# **Reference.** L. D. Landau and E. M. Lifshitz, *Mechanics*, 3rd ed., Pergamon (1976), §37: for
# $M^2>2EI_2$ the components are Jacobi elliptic functions of
# $\tau=t\sqrt{(I_3-I_2)(M^2-2EI_1)/(I_1I_2I_3)}$ with the modulus
# $k^2=(I_2-I_1)(2EI_3-M^2)/\big((I_3-I_2)(M^2-2EI_1)\big)$; $\omega_2\propto\operatorname{sn}\tau$
# changes sign twice in the period
#
# $$T=4K(k)\sqrt{\frac{I_1I_2I_3}{(I_3-I_2)(M^2-2EI_1)}} .$$

import numpy as np
from onesim.cad.exchange import load
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.multibody import EulerTop
from onesim.results.sampling import crossing_times
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.special import ellipk

theme.set()

# ## Geometry and inertia
#
# `load` returns the solid; its `inertia` is the inertia tensor about the centre of mass per unit
# density. Multiplied by the density of aluminium it gives the principal moments.

handle = load("dzhanibekov_t_handle.step")
density = 2700.0
inertia = tuple(np.diag(handle.inertia) * density)
print(
    f"mass {1e3 * density * handle.volumes[0].measure:.1f} g, centre of mass ({', '.join(f'{1e3 * c:.2f}' for c in handle.center_of_mass)}) mm"
)
print(
    "principal moments of inertia [kg mm²]: "
    + ", ".join(f"I_{axis} = {1e6 * value:.3f}" for axis, value in zip("xyz", inertia))
)

view = plots.Camera(elevation=30, azimuth=-55)
fig = plots.figure(1, 2, size=(14, 5.5))
plots.geometry(
    handle,
    selection={"crossbar": handle.boundaries.crossbar, "stem": handle.boundaries.stem},
    labels=False,
    camera=view,
    title="T-handle with its two bars",
    ax=fig[0],
)
plots.render(handle, camera=view, title="Rendered part", ax=fig[1])
plots.show()

# ## Model
#
# `EulerTop` holds Euler's equations for the three body-frame components of the angular velocity.
# The handle is spun at 5 revolutions per second about each principal axis in turn, with a small
# disturbance of 1 % of the spin on the component about the axis of the largest moment (about the
# smallest axis for the spin about $z$). Classical Runge-Kutta with a step of 1 ms integrates 3 s.

spin, disturbance = 2 * np.pi * 5.0, 0.01
domain = LumpedDomain()
top = EulerTop(domain=domain.system, inertia=inertia, torque=(0.0, 0.0, 0.0))
starts = {
    "x (smallest)": (spin, 0.0, disturbance * spin),
    "y (intermediate)": (0.0, spin, disturbance * spin),
    "z (largest)": (disturbance * spin, 0.0, spin),
}
numerics = Numerics(spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4())
runs = {
    axis: Simulation(
        problem=Problem(
            name=f"Spin about {axis}",
            geometry=domain,
            physics=(top,),
            initial_conditions=tuple(
                InitialValue(top.angular_velocity(c), w) for c, w in zip("xyz", omega)
            ),
        ),
        mesh=LumpedMesh(domain),
        numerics=numerics,
    ).run(start_time=0.0, end_time=3.0, time_step=1e-3)
    for axis, omega in starts.items()
}

# ## Results
#
# Energy and angular momentum over the whole run, the largest excursion of the two disturbed
# components, and the flip period of the intermediate-axis spin from the sign changes of
# $\omega_y$ against the elliptic-function period.

omega = {
    axis: np.array([run.field(top.angular_velocity(c)).time_series()[1] for c in "xyz"])
    for axis, run in runs.items()
}
times = runs["y (intermediate)"].field(top.angular_velocity("y")).time_series()[0]
energy, momentum = (
    0.5 * np.dot(inertia, np.square(starts["y (intermediate)"])),
    np.linalg.norm(np.multiply(inertia, starts["y (intermediate)"])),
)
i1, i2, i3 = inertia
landau = (
    4
    * ellipk(
        (i2 - i1)
        * (2 * energy * i3 - momentum**2)
        / ((i3 - i2) * (momentum**2 - 2 * energy * i1))
    )
    * np.sqrt(i1 * i2 * i3 / ((i3 - i2) * (momentum**2 - 2 * energy * i1)))
)
flips = crossing_times(times, omega["y (intermediate)"][1], 0.0, direction="both")

for axis, run in runs.items():
    e, m = (
        run.field(top.energy).time_series()[1],
        run.field(top.momentum_norm).time_series()[1],
    )
    k = "xyz".index(axis[0])
    print(
        f"spin about {axis:17s} relative drift: energy {np.ptp(e) / e[0]:.1e}, angular momentum {np.ptp(m) / m[0]:.1e}; "
        f"spin component from {omega[axis][k].min() / spin:+.3f} to {omega[axis][k].max() / spin:+.3f} of the initial spin"
    )
print(
    f"intermediate axis: {len(flips)} flips in 3 s, flip period {2 * np.diff(flips).mean():.5f} s, "
    f"elliptic-function period {landau:.5f} s, deviation {100 * (2 * np.diff(flips).mean() / landau - 1):+.3f} %"
)

# ## Figure
#
# The three components of the angular velocity for the spin about the intermediate axis, the spin
# component for all three axes, and the path of the angular-momentum direction in the body frame:
# it circles the stable axes and swings from pole to pole around the intermediate one.

fig = plots.figure(1, 3, size=(18, 5))
plots.line(
    times,
    {f"ω_{c}": omega["y (intermediate)"][k] / spin for k, c in enumerate("xyz")},
    xlabel="time [s]",
    ylabel="angular velocity / initial spin",
    title="Spin about the intermediate axis y",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    times,
    {f"spin about {axis}": omega[axis]["xyz".index(axis[0])] / spin for axis in runs},
    ylimits=(-1.2, 1.8),
    xlabel="time [s]",
    ylabel="spin component / initial spin",
    title="Stable and unstable axes",
    legend="upper right",
    ax=fig[1],
)
plots.line(
    points={
        f"spin about {axis}": (
            np.multiply(inertia, omega[axis].T)[::5, 0]
            / np.linalg.norm(np.multiply(inertia, starts[axis])),
            np.multiply(inertia, omega[axis].T)[::5, 1]
            / np.linalg.norm(np.multiply(inertia, starts[axis])),
        )
        for axis in runs
    },
    xlimits=(-1.1, 1.1),
    ylimits=(-1.1, 1.1),
    xlabel="M_x / |M|",
    ylabel="M_y / |M|",
    title="Angular momentum seen from the body",
    legend="lower left",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the stem in `dzhanibekov_t_handle_geometry.py` until its moment about $y$ is no longer
# intermediate, write the STEP file again and watch the flips disappear; or make the disturbance
# smaller: the flip period grows only with its logarithm.
