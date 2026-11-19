"""OneSimulation example: absorption of a Gaussian pressure pulse by perfectly matched layers.
Requires the installed, activated OneSimulation runtime.

A pressure pulse released in air spreads as a cylindrical wave and leaves the domain through
perfectly matched layers. The air volume is read from a STEP file; the simulation is compared with
the exact free-space solution, and the reflection from layers of different thickness is measured.
"""

# # Gaussian pulse and perfectly matched layers
#
# A Gaussian pressure pulse of half-width 30 mm is released at rest in air. It spreads as a
# cylindrical wave. To imitate open space, the computational square ends in **perfectly matched
# layers** (PML): strips in which the wave is damped without being reflected at their entrance.
#
# The acoustic equations for pressure and velocity are solved with staggered finite differences
# and the classical Runge-Kutta method. By symmetry one quarter of the square is computed, with
# reflecting walls on the two symmetry lines.
#
# The quarter of the air volume comes from the CAD file `gaussian_pulse_pml.step`, written by
# `gaussian_pulse_pml_geometry.py` with the CAD objects of OneSimulation. The model lives on its
# cross-section, a square of 0.36 m whose sides keep the names `x_min`, `x_max`, `y_min`, `y_max`.
#
# **Reference.** In free space the pulse $p_0e^{-\beta r^2}$ evolves as the Hankel integral
#
# $$p(r,t)=\frac{p_0}{2\beta}\int_0^\infty e^{-k^2/4\beta}\,\cos(ckt)\,J_0(kr)\,k\,\mathrm dk .$$

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import cartesian
from onesim.numerics import Numerics
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.acoustics import (
    AcousticPML,
    LinearAcoustics,
    StaggeredAcousticDifference,
)
from onesim.simulation import Problem, ReflectingWall, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.special import j0, roots_legendre

theme.set()

sound_speed, half_width, inner, spacing = 343.0, 0.030, 0.30, 0.003
beta = np.log(2) / half_width**2

# ## Geometry and mesh
#
# `section` cuts the loaded solid at half thickness. The square is 0.36 m wide; the inner 0.30 m
# are evaluated, and the outer cells hold the layer.

air_volume = load("gaussian_pulse_pml.step")
square = air_volume.section((0, 0, 1), origin=(0, 0, 0.005))
side = square.size[0]
cells = round(side / spacing)
mesh = cartesian(
    shape=(cells, cells),
    extents=(side, side),
    geometry=square,
    periodic=False,
    max_candidate_cells=10**6,
)

fig = plots.figure(1, 2, size=(13, 4.5))
plots.geometry(air_volume, labels=False, title="Quarter of the air volume", ax=fig[0])
plots.geometry(square, title="Cross-section with its named sides", ax=fig[1])
plots.show()


def exact_pressure(r, t):
    """Free-space pressure of the pulse at the radii r and the time t."""
    nodes, weights = roots_legendre(1600)
    k = 0.5 * np.sqrt(160 * beta) * (nodes + 1)
    amplitude = (
        np.exp(-(k**2) / (4 * beta))
        * k
        * 0.5
        * np.sqrt(160 * beta)
        * weights
        / (2 * beta)
    )
    return j0(np.outer(np.ravel(r), k)) @ (amplitude * np.cos(sound_speed * k * t))


# ## Simulation
#
# The outermost `layer` cells on the two outer sides are PML; without a layer the outer sides are
# plain reflecting walls.


def simulate(layer):
    """Pressure history of the pulse with a PML of `layer` cells (0: reflecting walls)."""
    air = LinearAcoustics(
        domain=square.domain(), density=1.204, sound_speed=sound_speed
    )
    problem = Problem(
        name="Gaussian pulse",
        geometry=square,
        physics=(air,),
        boundary_conditions=(
            ReflectingWall(boundary=square.boundaries.all, physics=air),
        ),
        initial_conditions=(
            InitialValue(air.pressure, lambda x, y: np.exp(-beta * (x**2 + y**2))),
            InitialValue(air.velocity, (0.0, 0.0)),
        ),
    )
    pml = (
        AcousticPML(((0, layer), (0, layer)), target_reflection=1e-4) if layer else None
    )
    numerics = Numerics(
        spatial=StaggeredAcousticDifference(pml=pml),
        time_integrator=ClassicalRungeKutta4(),
    )
    step = 0.25 * spacing / sound_speed
    return (
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .run(start_time=0.0, end_time=1200 * step, time_step=step)
        .field(air.pressure)
    )


runs = {layer: simulate(layer) for layer in (0, 5, 10, 20)}
pressure = runs[20]

# ## Results
#
# While the pulse crosses the square, the field with the 20-cell layer is compared with the
# exact solution. After 2 ms the pulse has left; any difference to the free-space solution in the
# inner 0.30 m was reflected by the outer sides.

axis = (np.arange(round(inner / spacing)) + 0.5) * spacing
grid = np.array([[x, y] for x in axis for y in axis])
radius = np.hypot(*grid.T)
times = pressure.times
nearest = lambda field, t: float(field.times[np.argmin(np.abs(field.times - t))])
snapshot = lambda field, t: field.evaluate(grid, time=nearest(field, t))
for t in (0.5e-3, 0.8e-3):
    deviation = np.abs(
        snapshot(pressure, t) - exact_pressure(radius, nearest(pressure, t))
    ).max()
    print(
        f"t = {1e3 * t:.1f} ms: largest deviation from the exact solution {deviation:.4f} of the initial peak"
    )
reflected = {
    layer: np.abs(
        snapshot(field, 2.0e-3) - exact_pressure(radius, nearest(field, 2.0e-3))
    ).max()
    for layer, field in runs.items()
}
for layer, value in reflected.items():
    print(
        f"{'reflecting walls' if layer == 0 else f'PML of {layer:2d} cells':17s} largest deviation from free space after 2 ms: {value:.1e}"
    )

# ## Figure
#
# The pulse inside the square and while it leaves through the layer, the pressure at a point near
# the layer over time, and the reflected pressure against the layer thickness.

probe = [[0.27, 0.0015]]
signals = {
    ("reflecting walls" if layer == 0 else f"PML, {layer} cells"): (
        1e3 * times,
        field.time_series(probe)[1].ravel(),
    )
    for layer, field in runs.items()
    if layer in (0, 20)
}
free_space = np.array([exact_pressure(np.hypot(*probe[0]), t)[0] for t in times[::10]])

fig = plots.figure(2, 2, size=(13, 11))
for panel, t in zip((fig[0], fig[1]), (0.5e-3, 0.8e-3)):
    plots.heatmap(
        axis,
        axis,
        snapshot(pressure, t).reshape(len(axis), len(axis)).T,
        colormap="diverging",
        centre=0.0,
        label="pressure / initial peak",
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"Pressure after {1e3 * t:.1f} ms",
        ax=panel,
    )
plots.line(
    y=signals,
    reference={"exact, free space": (1e3 * times[::10], free_space)},
    xlabel="time [ms]",
    ylabel="pressure at 0.27 m / initial peak",
    title="Near the outer side: the wall reflects, the layer does not",
    ax=fig[2],
)
plots.bars(
    [("walls" if layer == 0 else f"{layer} cells") for layer in reflected],
    list(reflected.values()),
    scale="log",
    ylabel="reflected pressure after 2 ms / initial peak",
    title="Reflection against the layer thickness",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Lower the target reflection of the layer or make it thinner than five cells to see where it
# stops working; release the pulse off centre, so that the wave meets the layer at a shallow
# angle.
