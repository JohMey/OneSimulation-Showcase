"""OneSimulation example: laser-flash measurement on a coin-shaped sample with heat losses.
Requires the installed, activated OneSimulation runtime.

A laser flash heats the front face of a ceramic disc and the temperature rise of the rear face is
recorded. The three-dimensional simulation is compared with the exact series solution, and the
thermal diffusivity is read off the computed curve as in the real measurement.
"""


# # Laser-flash method on a coin-shaped sample
#
# In the flash method (Parker et al., *J. Appl. Phys.* 32, 1679, 1961) the front face of a thin
# disc receives a short laser flash and the temperature of the rear face is recorded. Without heat
# losses, the time $t_{1/2}$ at which the rear face reaches half of its maximum rise gives the
# thermal diffusivity:
#
# $$\alpha = 0.1388\,\frac{L^2}{t_{1/2}} .$$
#
# A real sample loses heat from both faces and from its rim. The rise then peaks and decays,
# $t_{1/2}$ shortens and Parker's formula overestimates $\alpha$. This notebook computes the
# measurement at room temperature and in a furnace at 800 °C, where radiation makes the losses
# large.
#
# The disc (12.7 mm diameter, 2 mm thick) is a full 3D solid with
#
# $$\rho c_p\,\partial_t T = k\,\nabla^2 T,\qquad -k\,\partial_n T = h\,(T-T_\infty)\ \text{on every face},$$
#
# and $h = h_c + 4\varepsilon\sigma T_\infty^3$ (free convection plus linearised radiation).
#
# **The flash.** It deposits its energy in a thin layer below the front face. The simulation starts
# from that layer, $T-T_\infty\propto e^{-z^2/w^2}$ with $w=0.15$ mm, scaled so that the sample
# would warm by 1 K without losses.

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1

from onesim.cad.d2 import Circle, Plane
from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, TriangularMesher, extrude
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ConjugateGradient
from onesim.numerics.time import BDF, AdaptiveTimeStep
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()

radius, thickness, layer = 6.35e-3, 2.0e-3, 0.15e-3
ceramic = IsotropicSolid(density=2600.0, specific_heat_capacity=808.0, thermal_conductivity=3.98,
                         youngs_modulus=9.6e10, poisson_ratio=0.24, thermal_expansion_coefficient=5.7e-6)
diffusivity = ceramic.thermal_conductivity / (ceramic.density * ceramic.specific_heat_capacity)
ambient = {"room, 20 °C": 293.15, "furnace, 800 °C": 1073.15}
loss = {case: 10.0 + 4 * 0.9 * 5.670374e-8 * T**3 for case, T in ambient.items()}
flash = lambda z: thickness * 2 / (np.sqrt(np.pi) * layer) * np.exp(-(z / layer) ** 2)


# ## Reference: the exact series
#
# The cooling cylinder has an exact solution: a product of axial modes $\cos$/$\sin(\mu_m z)$ and
# radial modes $J_0(\beta_n r/R)$, each decaying at the rate $\alpha(\mu_m^2+\beta_n^2/R^2)$
# (Carslaw and Jaeger, *Conduction of Heat in Solids*).

def exact_rise(h, times, modes=(250, 150)):
    """Temperature rise at the centre of the rear face from the Fourier-Bessel series."""
    H, biot = h / ceramic.thermal_conductivity, h * radius / ceramic.thermal_conductivity
    axial_equation = lambda m: (m**2 - H**2) * np.sin(m * thickness) - 2 * m * H * np.cos(m * thickness)
    mu = np.array([brentq(axial_equation, max(n * np.pi, 1e-9) / thickness, (n + 1) * np.pi / thickness, xtol=1e-9)
                   for n in range(modes[0])])
    phi = lambda z: ((mu * np.cos(np.outer(z, mu)) + H * np.sin(np.outer(z, mu)))
                     / np.sqrt(0.5 * ((mu**2 + H**2) * thickness + 2 * H)))
    radial_equation = lambda x: x * j1(x) - biot * j0(x)
    grid = np.linspace(1e-9, (modes[1] + 2) * np.pi, 400 * (modes[1] + 2))
    brackets = np.flatnonzero(np.diff(np.sign(radial_equation(grid))))[:modes[1]]
    beta = np.array([brentq(radial_equation, grid[i], grid[i + 1], xtol=1e-14) for i in brackets])
    z = np.linspace(0.0, 12 * layer, 24001)
    axial = np.trapezoid(flash(z)[:, None] * phi(z), z, axis=0) * phi(thickness)[0]
    radial = 2 * biot / ((beta**2 + biot**2) * j0(beta))
    rate = diffusivity * (mu[:, None] ** 2 + (beta / radius) ** 2)
    return np.einsum("m,n,tmn->t", axial, radial, np.exp(-rate * np.asarray(times)[:, None, None]))


# ## Simulation
#
# The disc is a triangle mesh of the circle, extruded through the thickness with thin layers at the
# heated face. Quadratic elements, a third-order BDF integrator and adaptive time steps resolve
# the flash layer at the start and the slow cooling at the end.

face = TriangularMesher(target_size=radius / 4, local_sizes={"circumference": 2 * np.pi * radius / 28}).generate(
    Circle(radius=radius))
mesh = extrude(face, thickness, Geometric(10, ratio=14.0), names=("front", "rear"))
faces = mesh.geometry.boundaries


def simulate(case):
    """Temperature history of the disc after the flash for one ambient condition."""
    model = HeatConduction(domain=mesh.geometry.domain(), material=ceramic, heating=0.0)
    problem = Problem(
        name="Flash method", geometry=mesh.geometry, physics=(model,),
        boundary_conditions=(ConvectiveHeatFlux(boundary=faces.front | faces.rear | faces.circumference, physics=model,
                                                heat_transfer_coefficient=loss[case], ambient_temperature=ambient[case]),),
        initial_conditions=(InitialValue(model.temperature, lambda x, y, z: ambient[case] + flash(z)),))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), time_integrator=BDF(order=3),
                        linear_solver=ConjugateGradient(relative_tolerance=1e-10, maximum_iterations=20000))
    steps = AdaptiveTimeStep(initial=2e-6, minimum=1e-9, maximum=0.1, relative_tolerance=0.0, absolute_tolerance=3e-4)
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).run(start_time=0.0, end_time=3.2, time_step=steps)
    return result.field(model.temperature)


fields = {case: simulate(case) for case in ambient}


# ## Results
#
# The rear-face signal of each case against the series, and the diffusivity that Parker's formula
# gives when the losses are ignored.

signals, exact = {}, {}
print(f"{'case':16s} {'max. rise [K]':>14s} {'exact':>8s} {'t_1/2 [ms]':>11s} {'exact':>8s} {'Parker alpha / true alpha':>27s}")
for case, field in fields.items():
    times, rear = field.time_series([[0.0, 0.0, thickness]])
    rise = rear.ravel() - ambient[case]
    series = exact_rise(loss[case], times[1:])
    half_time = [np.interp(0.5 * curve.max(), curve[:curve.argmax() + 1], instants[:curve.argmax() + 1])
                 for instants, curve in ((times, rise), (times[1:], series))]
    signals[case], exact[case] = (1e3 * times, rise), (1e3 * times[1:], series)
    print(f"{case:16s} {rise.max():14.4f} {series.max():8.4f} {1e3 * half_time[0]:11.1f} {1e3 * half_time[1]:8.1f}"
          f" {0.1388 * thickness**2 / half_time[0] / diffusivity:27.3f}")


# ## Figure
#
# The temperature in a section through the axis shortly after the flash and near the half-rise
# time (furnace case, thickness drawn three times enlarged), the rear face and rim, and the
# rear-face signals.

furnace = fields["furnace, 800 °C"]
rise = lambda T, position: T - ambient["furnace, 800 °C"]
instants = furnace.time_series()[0]
early, half = instants[np.searchsorted(instants, 0.03)], instants[np.searchsorted(instants, 0.26)]

fig = plots.figure(2, 2, size=(15, 10))
for panel, instant in zip((fig[0], fig[1]), (early, half)):
    plots.section(furnace, time=instant, plane=Plane(origin=(0, 0, 0), normal=(0, 1, 0)), component=rise, contours=10,
                  subdivisions=3, aspect=3.0, label="T - T_inf [K]", title=f"Section through the axis, {1e3 * instant:.0f} ms",
                  ax=panel)
plots.surface(furnace, time=half, boundary=faces.rear | faces.circumference, component=rise, subdivisions=2, edges=True,
              camera=plots.Camera(elevation=28, azimuth=-60), label="T - T_inf [K]",
              title=f"Rear face and rim, {1e3 * half:.0f} ms", ax=fig[2])
plots.line(points=signals, reference=exact, hlines={"half of the loss-free rise": 0.5}, xlabel="time [ms]",
           ylabel="rear-face rise [K]", title="Rear-face signal: simulation and exact series", ax=fig[3])
plots.show()


# ## Going further
#
# Replace the linearised radiation by the full Stefan-Boltzmann law (`RadiationCooling`), model a
# finite laser pulse as a time-dependent heat flux on the front face, or shrink the radius until
# the rim losses dominate.
