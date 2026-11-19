"""OneSimulation example: oblique whistler waves in three-dimensional Hall MHD.
Requires the installed, activated OneSimulation runtime.

Six small plasma waves travel through a periodic box at different angles to the magnetic field.
Their frequencies are measured from the simulation and compared with the exact dispersion
relation, once with spectral and once with finite-difference derivatives.
"""

# # Oblique whistler waves in Hall MHD
#
# A uniform plasma is threaded by a magnetic field $\mathbf B_0$ along $z$. On scales below the ion
# inertial length $d_i$ the Hall term splits the Alfvén wave into two branches: the fast
# **whistler** wave and the slower **ion-cyclotron** wave. Their frequency depends on the angle
# $\theta$ between the wave vector $\mathbf k$ and $\mathbf B_0$.
#
# This notebook launches six such waves together in a periodic box, measures their frequencies in
# a three-dimensional Hall-MHD simulation and compares them with theory.

import numpy as np
from onesim.meshing import TensorCollocationMesh
from onesim.numerics.discretization import CenteredDifferences, FourierCollocation
from onesim.numerics.time import ClassicalRungeKutta4TimeIntegrator
from onesim.physics.magnetohydrodynamics import LinearMHDWaves, ResistiveMHD3DSolver
from onesim.results import oscillation_fit
from onesim.visualization import plots, theme

theme.set()


# ## Plasma and waves
#
# Units with $\mu_0=1$: density 1, field strength 1 (Alfvén speed $v_A=1$), sound speed
# $c_s=0.5$ and ion inertial length $d_i=0.5$. `LinearMHDWaves` describes the small-amplitude
# waves of this plasma. The box has the side $2\pi$, so wave vectors are integer triples; all six
# have $|\mathbf k|=3$ and differ in their angle to the field.

gamma, hall_coefficient = 5 / 3, 0.5
waves = LinearMHDWaves(
    density=1.0,
    pressure=0.15,
    magnetic_field=(0, 0, 1),
    gamma=gamma,
    hall_coefficient=hall_coefficient,
)
modes = [
    ((0, 0, 3), "whistler"),
    ((1, 2, 2), "whistler"),
    ((2, 2, 1), "whistler"),
    ((0, 3, 0), "whistler"),
    ((2, 1, 2), "ion_cyclotron"),
    ((2, -2, 1), "ion_cyclotron"),
]
angles = np.array([np.degrees(np.arccos(abs(k[2]) / 3)) for k, _ in modes])
exact = np.array([waves.frequency(k, branch) for k, branch in modes])


# ## Reference from the literature
#
# Independently of the library, the frequencies follow from the dispersion relation of Hameiri,
# Ishizawa and Ishida (*Phys. Plasmas* 12, 072109, 2005), a cubic equation for $\omega^2$:
#
# $$\left(\omega^2-k_\parallel^2v_A^2\right)\left(\omega^4-k^2(v_A^2+c_s^2)\,\omega^2+k^2k_\parallel^2v_A^2c_s^2\right)
#   =\omega^2\,k_\parallel^2k^2\,v_A^2\,d_i^2\left(\omega^2-k^2c_s^2\right),\qquad k_\parallel=k\cos\theta .$$


def literature_frequencies(k, angle):
    """Slow, ion-cyclotron and whistler frequency from the dispersion relation of Hameiri et al."""
    va2, cs2, di2 = (
        waves.alfven_speed**2,
        waves.sound_speed**2,
        waves.ion_inertial_length**2,
    )
    k2, kp2 = k**2, (k * np.cos(np.radians(angle))) ** 2
    cubic = [
        1.0,
        -(k2 * (va2 + cs2) + kp2 * va2 * (1 + k2 * di2)),
        k2 * kp2 * va2 * (va2 + 2 * cs2 + k2 * di2 * cs2),
        -k2 * kp2**2 * va2**2 * cs2,
    ]
    return np.sqrt(np.sort(np.abs(np.roots(cubic).real)))


# ## Simulation
#
# The mesh is a periodic box with `points` samples per direction. The solver receives the mesh, the
# spatial discretisation and the time integrator from outside. It starts from the uniform plasma
# plus the six waves, each with a magnetic amplitude of $10^{-7}$, and advances it for two Alfvén
# crossing times of the box. While it runs, `observe` records the complex amplitude of every
# wave; `oscillation_fit` turns each recorded signal into a frequency.


def measure_frequencies(spatial, points):
    """Simulate the six waves on points^3 mesh samples and return their measured frequencies."""
    axis = np.arange(points) * (2 * np.pi / points)
    mesh = TensorCollocationMesh(
        axes=(axis, axis, axis),
        coordinate_system="cartesian",
        periodic=(True, True, True),
    )
    solver = ResistiveMHD3DSolver(
        mesh=mesh,
        spatial=spatial,
        time_integrator=ClassicalRungeKutta4TimeIntegrator(),
        gamma=gamma,
        hall_coefficient=hall_coefficient,
        resistivity=0.0,
        final_time=4 * np.pi,
        time_step=0.008,
    )
    grid = np.meshgrid(*mesh.axes, indexing="ij")
    start = sum(
        waves.perturbation(k, branch, grid, amplitude=1e-7) for k, branch in modes
    )
    result = solver.run(
        solver.conservative_state(*waves.primitive_state(start)),
        observe_interval=0.08,
        observe=lambda state: [
            waves.amplitude(k, branch, state.magnetic_field, axes=mesh.axes)
            for k, branch in modes
        ],
    )
    signals = result.history.values.T
    return (
        np.array(
            [
                oscillation_fit(result.history.times, signal).frequency
                for signal in signals
            ]
        ),
        result,
    )


methods = {
    "spectral": FourierCollocation(),
    "central differences": CenteredDifferences(),
}
grid_points = [16, 24, 32]
runs = {
    (name, n): measure_frequencies(spatial, n)
    for name, spatial in methods.items()
    for n in grid_points
}
measured, result = runs["spectral", 32]


# ## Results
#
# Frequencies of the spectral run on $32^3$ points against the library's linear theory and the
# literature formula.

print(
    "wave vector   angle  branch          simulation   linear theory   literature   deviation"
)
for (k, branch), angle, simulation, theory in zip(modes, angles, measured, exact):
    literature = literature_frequencies(3.0, angle)[2 if branch == "whistler" else 1]
    print(
        f"{str(k):12s} {angle:5.1f}°  {branch:14s} {simulation:10.6f}   {theory:10.6f}   {literature:10.6f}"
        f"   {simulation / theory - 1:+.1e}"
    )

# The spectral derivative is exact for every resolved wave, so its error does not depend on the
# grid. Central differences approach the exact frequencies with second order.

errors = {
    name: [np.max(np.abs(runs[name, n][0] / exact - 1)) for n in grid_points]
    for name in ("central differences", "spectral")
}
for name, values in errors.items():
    print(
        f"{name:20s} largest frequency error on {grid_points} points: "
        + ", ".join(f"{e:.1e}" for e in values)
    )


# ## Figure
#
# The magnetic field in a plane across $\mathbf B_0$, one recorded wave signal against
# $\cos\omega t$, the measured frequencies on the dispersion relation, and the grid study.

fig = plots.figure(1, 4, size=(17, 4.4))

plots.heatmap(
    result.axes[0],
    result.axes[1],
    result.magnetic_field[0][:, :, 0] / 1e-7,
    colormap="diverging",
    label="b_x / 10^-7",
    xlabel="x",
    ylabel="y",
    title="Magnetic field b_x at z = 0",
    ax=fig[0],
)

times, signal = result.history.times, result.history.values[:, 2]
smooth = np.linspace(0, times[-1], 600)
plots.line(
    points={"simulation": (times, (signal / signal[0]).real)},
    reference={"cos(omega t)": (smooth, np.cos(exact[2] * smooth))},
    xlabel="time",
    ylabel="wave amplitude",
    title="Whistler wave k = (2, 2, 1)",
    legend="lower left",
    ax=fig[1],
)

theta = np.linspace(0, 90, 91)
theory = np.array([literature_frequencies(3.0, angle) for angle in theta])
plots.line(
    theta,
    {"ion-cyclotron": theory[:, 1], "whistler": theory[:, 2]},
    points={"simulation": (angles, measured)},
    xlabel="angle between k and B_0 [deg]",
    ylabel="frequency",
    title="Dispersion relation, |k| = 3",
    ax=fig[2],
)

plots.convergence(
    2 * np.pi / np.array(grid_points),
    errors,
    order=2,
    xlabel="grid spacing",
    ylabel="largest frequency error",
    title="Grid study",
    ax=fig[3],
)
plots.show()


# ## Going further
#
# Set `hall_coefficient = 0` to recover the ideal Alfvén and magnetosonic waves, raise the
# amplitude to see nonlinear coupling between the waves, or give the solver a `resistivity`:
# every wave is then damped at the rate $\eta k^2/2$, which `oscillation_fit` returns as
# `growth_rate`.
