"""OneSimulation example: wicking of water into a strip of filter paper.
Requires the installed, activated OneSimulation runtime.

The lower edge of a dry paper strip touches water; capillary suction draws the water up until
gravity balances it. The strip is read from a STEP file and cut to its mid-plane; the unsaturated
flow follows the mixed-form Richards equation and is compared with Philip's sorptivity, the
square-root law of Lucas and Washburn, and the hydrostatic equilibrium.
"""

# # Wicking in a paper strip
#
# A strip of filter paper, 20 mm wide and 40 mm high, hangs with its lower edge in water. The
# pores of the dry paper suck the water up: fast at first, the wetted height growing with the
# square root of time, then ever slower as the weight of the lifted water balances the capillary
# suction. The water flow in the paper is unsaturated porous-media flow, the **Richards equation**
# for the pressure head $h$,
#
# $$\frac{\partial\theta(h)}{\partial t}=\nabla\cdot\big(K(h)\,(\nabla h+\mathbf e_y)\big),$$
#
# with the van Genuchten water retention $\theta(h)$ and the Mualem conductivity $K(h)$ of the
# paper: porosity 0.7, $\alpha=12$ 1/m, $n=2.5$, saturated conductivity $10^{-6}$ m/s.
#
# | edge | condition |
# |---|---|
# | `y_min`, in the water | $h_0=0$ (saturated) |
# | `x_min`, `x_max`, `y_max` | no flow |
# | initial state | dry paper, $h_i=-1$ m |
#
# The strip comes from the CAD file `paper_strip_wicking.step`, written by
# `paper_strip_wicking_geometry.py` with the CAD objects of OneSimulation.
#
# **Reference.** While gravity is still negligible the uptake per unit width grows as
# $I=S\sqrt t$ (Lucas-Washburn behaviour, J. R. Philip, Soil Sci. 84 (1957) 257-264), with the
# sorptivity from Parlange's integral (J.-Y. Parlange, Soil Sci. 119 (1975) 158-161),
#
# $$S^2=\int_{h_i}^{0}\big(\theta_s+\theta(h)-2\theta_i\big)\,K(h)\,\mathrm dh .$$
#
# Later, gravity slows the rise. Treating the wetted zone as a sharp front driven by a constant
# suction $h_f=S^2/(2K_s\Delta\theta)$ gives Washburn's rise with gravity in the Green-Ampt form
# (W. H. Green, G. A. Ampt, J. Agric. Sci. 4 (1911) 1-24),
#
# $$t(L)=\frac{\Delta\theta\,h_f}{K_s}\Big(-\ln\big(1-L/h_f\big)-L/h_f\Big),\qquad \Delta\theta=\theta_s-\theta_i ,$$
#
# an approximation for the wetted height $L$ that is accurate when the front is sharp; the uptake
# is then $I=\Delta\theta\,L$.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import DampedNewton
from onesim.numerics.time import BackwardEuler
from onesim.physics.hydrodynamics import PressureHead, Richards, VanGenuchtenRetention
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.integrate import quad

theme.set()

# ## Geometry and mesh
#
# `section` cuts the loaded strip at half thickness; $x$ runs across the strip, $y$ upwards from
# the water surface. Linear triangles of 1 mm.

sheet = load("paper_strip_wicking.step")
strip = sheet.section((0, 0, 1), origin=(0, 0, 1e-4))
width, height = strip.size[:2]
mesh = TriangularMesher(target_size=1e-3).generate(strip)

fig = plots.figure(1, 2, size=(12, 6))
plots.geometry(
    sheet,
    selection={"edge in the water": sheet.boundaries.y_min},
    labels=False,
    camera=plots.Camera(elevation=20, azimuth=-60),
    title="Paper strip from the STEP file",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Mid-plane: {mesh.cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `Richards` with the van Genuchten retention; gravity acts along $-y$ of the section by default.
# Backward Euler keeps the storage change exact in every step (mixed form); 1200 steps of 5 s
# follow the first 100 minutes. The damped Newton method handles the sharp front in the dry paper.

paper = Richards(
    domain=strip.domain(),
    retention=VanGenuchtenRetention(alpha=12.0, n=2.5),
    conductivity=1e-6,
    saturated_water_content=0.7,
    residual_water_content=0.0,
)
problem = Problem(
    name="Paper strip wicking",
    geometry=strip,
    physics=(paper,),
    initial_conditions=(InitialValue(paper.pressure_head, -1.0),),
    boundary_conditions=(
        PressureHead(physics=paper, boundary=strip.boundaries.y_min, value=0.0),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=1), quadrature_order=2),
    nonlinear_solver=DampedNewton(
        linear_solver=SparseDirect(),
        relative_tolerance=1e-8,
        absolute_tolerance=1e-10,
        maximum_iterations=60,
    ),
    time_integrator=BackwardEuler(),
)
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).run(
    start_time=0.0, end_time=6000.0, time_step=5.0
)
water = result.field(paper.water_content)

# ## Results
#
# The uptake per unit width is the integral of the water content gained, divided by the strip
# width; it is compared with $S\sqrt t$. The wetted height is where the water content crosses half
# of the porosity $\theta_s$ above $\theta_i$ on the centre line. The uptake is compared
# with $S\sqrt t$ and with the Green-Ampt rise.

theta_i, theta_0, k_0 = (
    float(paper.volumetric_water_content(-1.0)),
    paper.saturated_water_content,
    paper.conductivity,
)
sorptivity = np.sqrt(
    quad(
        lambda h: (theta_0 + paper.volumetric_water_content(h) - 2 * theta_i)
        * paper.hydraulic_conductivity(h),
        -1.0,
        0.0,
    )[0]
)
uptake = water.integrate()
times, gained = (
    np.asarray(uptake.times),
    (np.asarray(uptake.values) - uptake.values[0]) / width,
)
y = np.linspace(0.0, height, 481)
centre = np.column_stack((np.full_like(y, width / 2), y))
wetted = lambda t: y[
    np.argmax(water.evaluate(centre, time=t) < (theta_0 + theta_i) / 2)
]
suction = sorptivity**2 / (2 * k_0 * (theta_0 - theta_i))
rise = np.linspace(0.0, 0.999 * suction, 4000)
green_ampt = lambda t: np.interp(
    t,
    (theta_0 - theta_i)
    * suction
    / k_0
    * (-np.log(1 - rise / suction) - rise / suction),
    rise,
)
print(
    f"sorptivity (Parlange): {1e3 * sorptivity:.3f} mm/s^0.5, Green-Ampt suction head {1e3 * suction:.1f} mm"
)
print(
    "   t [s]   uptake [mm]   S sqrt(t) [mm]   ratio   Green-Ampt [mm]   ratio   wetted height [mm]"
)
for t in (60.0, 120.0, 300.0, 600.0, 1200.0, 3000.0, 6000.0):
    i = int(np.argmin(np.abs(times - t)))
    ga = (theta_0 - theta_i) * green_ampt(times[i])
    print(
        f"{times[i]:8.0f} {1e3 * gained[i]:12.3f} {1e3 * sorptivity * np.sqrt(times[i]):15.3f} {gained[i] / (sorptivity * np.sqrt(times[i])):8.3f}"
        f" {1e3 * ga:16.3f} {gained[i] / ga:8.3f} {1e3 * wetted(times[i]):17.1f}"
    )

# ## Figure
#
# The water content in the strip after 2, 15 and 100 minutes; the uptake against $S\sqrt t$ and the
# Green-Ampt rise; and the water content on the centre line at the same times.

nearest = lambda t: float(times[np.argmin(np.abs(times - t))])
fig = plots.figure(1, 5, size=(22, 7))
for panel, t in zip((fig[0], fig[1], fig[2]), (120.0, 900.0, 6000.0)):
    plots.surface(
        water,
        time=nearest(t),
        limits=(0.0, 0.7),
        xlabel="x [m]",
        ylabel="y [m]",
        label="water content",
        title=f"after {nearest(t) / 60:.0f} min",
        ax=panel,
    )
plots.line(
    points={"simulation": (times[1:], 1e3 * gained[1:])},
    reference={
        "S sqrt(t)": (times[1:], 1e3 * sorptivity * np.sqrt(times[1:])),
        "Green-Ampt rise": (
            times[1:],
            1e3 * (theta_0 - theta_i) * green_ampt(times[1:]),
        ),
    },
    scale="loglog",
    xlabel="time [s]",
    ylabel="uptake per width [mm]",
    title="Square-root law, then gravity",
    legend="upper left",
    ax=fig[3],
)
plots.line(
    y={
        f"{t / 60:.0f} min": (water.evaluate(centre, time=nearest(t)), 1e3 * y)
        for t in (120.0, 900.0, 6000.0)
    },
    xlabel="water content",
    ylabel="height y [mm]",
    title="Water content on the centre line",
    legend="upper right",
    ax=fig[4],
)
plots.show()

# ## Going further
#
# Lay the strip flat (`gravity=0.0` in `Richards`): the uptake then follows $S\sqrt t$ for ever.
# Or change the pore-size parameter $\alpha$ to see how far finer or coarser papers lift the water.
