"""OneSimulation example: horizontal infiltration into an unsaturated soil column.
Requires the installed, activated OneSimulation runtime.

The OneSimulation Richards solution is checked against the Boltzmann similarity
solution of horizontal absorption: the self-similar water-content profile and the
square-root-of-time law of the cumulative infiltration.
"""

# # Horizontal infiltration into an unsaturated soil column
#
# A long horizontal column of soil is initially at a uniform (fairly dry) water
# content $\theta_i$. From $t=0$ on, the inlet face $x=0$ is held close to
# saturation, so water is drawn into the column by capillarity alone; gravity
# plays no role in a horizontal column. The far end stays dry during the whole
# experiment.
#
# OneSimulation solves the mixed form of the Richards equation with a
# conservative finite-volume scheme,
#
# $$\frac{\partial\theta(h)}{\partial t}
#   =\frac{\partial}{\partial x}\Big(K(h)\,\frac{\partial h}{\partial x}\Big),\qquad
#   h(x,0)=h_i,\quad h(0,t)=h_0,\quad q(L,t)=0,$$
#
# with the van Genuchten-Mualem closure for the water content $\theta(h)$ and
# the hydraulic conductivity $K(h)$. Written in terms of $\theta$ this is a
# nonlinear diffusion equation with the soil-water diffusivity
# $D(\theta)=K/C$, $C=d\theta/dh$.
#
# **Reference.** Without gravity and without a length scale the profile depends on
# Boltzmann's similarity variable $\varphi=x/\sqrt{t}$ only,
# $\theta(x,t)=\Theta(\varphi)$, where $\Theta$ solves
#
# $$\frac{d}{d\varphi}\Big(D(\Theta)\frac{d\Theta}{d\varphi}\Big)
#   +\frac{\varphi}{2}\frac{d\Theta}{d\varphi}=0,\qquad
#   \Theta(0)=\theta_0,\quad\Theta(\infty)=\theta_i,$$
#
# and the cumulative infiltration grows as the square root of time,
#
# $$I(t)=S\sqrt{t},\qquad S=-2\,D(\theta_0)\,\Theta'(0),$$
#
# with $S$ the sorptivity of the soil. $D$ varies by a factor of forty across the
# wetting front, so $\Theta(\varphi)$ has no closed form; it is obtained as a
# boundary-value problem of this ordinary differential equation, independently of
# the finite-volume solver.

import numpy as np
from onesim.meshing import TensorCollocationMesh
from onesim.numerics.time import ForwardEulerTimeIntegrator
from onesim.physics.hydrodynamics import Richards1DProvider
from onesim.physics.hydrodynamics.porous.numerics.equations.multiphysics.richards import (
    RichardsColumnFiniteVolume,
)
from onesim.visualization import plots, theme
from scipy.integrate import solve_bvp

theme.set()

# ## Soil column
#
# A loam-type soil in a column of 0.5 m, 400 cells. The inlet is held at a small suction
# of 0.1 m (95 % of saturation), the far end is a no-flow face and gravity is switched
# off. The solver advances the stored water with explicit steps of 0.02 s and the
# harmonic mean of the cell conductivities.

length, cells, inlet_head, initial_head = 0.5, 400, -0.10, -1.00
x = (np.arange(cells) + 0.5) * length / cells
column = TensorCollocationMesh(
    axes=(x,), coordinate_system="cartesian", periodic=(False,)
)
soil = Richards1DProvider(
    mesh=column,
    spatial=RichardsColumnFiniteVolume(),
    time_integrator=ForwardEulerTimeIntegrator(),
    residual_water_content=0.08,
    saturated_water_content=0.43,
    van_genuchten_alpha=3.6,
    van_genuchten_n=1.56,
    saturated_hydraulic_conductivity=1.0e-5,
    gravity=0.0,
    initial_pressure_head=initial_head,
    bottom_pressure_head=inlet_head,
    top_pressure_head=None,
    final_time=60.0,
    time_step=0.02,
)
inlet, initial = float(soil.water_content(inlet_head)), float(
    soil.water_content(initial_head)
)
print(f"water content: {inlet:.4f} at the inlet, {initial:.4f} initially")

# ## Simulation
#
# One hour in segments of one minute, each started from the pressure head of the one
# before; the water that entered through the inlet in each segment adds up to the
# cumulative infiltration.

times, infiltration, head, profiles, residual = [0.0], [0.0], None, {}, 0.0
for minute in range(1, 61):
    segment = soil.run(head)
    head, residual = segment.pressure_head, max(residual, segment.mass_balance_residual)
    times.append(60.0 * minute)
    infiltration.append(infiltration[-1] + segment.boundary_exchange)
    if minute in (10, 20, 40, 60):
        profiles[60.0 * minute] = segment.water_content
times, infiltration = np.array(times), np.array(infiltration)
print(
    f"cumulative infiltration after one hour: {1e3 * infiltration[-1]:.2f} mm, largest mass-balance residual {residual:.1e} m"
)

# ## Similarity solution
#
# With $q=D(\Theta)\,\Theta'$ the similarity equation is the first-order system
# $\Theta'=q/D$, $q'=-\tfrac{\varphi}{2}\,q/D$ between $\Theta(0)=\theta_0$ and
# $\Theta=\theta_i$ far ahead of the front. The diffusivity comes from the conductivity
# and the water capacity of the same soil law.

far = 6.5e-3
diffusivity = lambda theta: (
    soil.hydraulic_conductivity(soil.pressure_head(np.clip(theta, initial, inlet)))
    / soil.water_capacity(soil.pressure_head(np.clip(theta, initial, inlet)))
)
phi = np.linspace(0.0, far, 600)
similarity = solve_bvp(
    lambda p, y: np.vstack(
        (y[1] / diffusivity(y[0]), -0.5 * p * y[1] / diffusivity(y[0]))
    ),
    lambda left, right: np.array([left[0] - inlet, right[0] - initial]),
    phi,
    np.vstack(
        (
            initial + (inlet - initial) * np.exp(-((phi / 2e-3) ** 2)),
            np.full_like(phi, -1e-4),
        )
    ),
    tol=1e-9,
    max_nodes=200000,
)
profile = lambda p: similarity.sol(np.minimum(p, far))[0]
sorptivity = -2.0 * similarity.sol(0.0)[1]
print(f"sorptivity from the similarity solution: {sorptivity:.4e} m/√s")

# ## Comparison
#
# The profiles at four times against $\Theta(x/\sqrt{t})$, their collapse onto one curve
# in $\varphi$, and the cumulative infiltration against $S\sqrt{t}$.

print("time [min]   largest profile deviation   rms deviation   (fractions of θ0 - θi)")
for t, theta in profiles.items():
    deviation = np.abs(theta - profile(x / np.sqrt(t))) / (inlet - initial)
    print(
        f"{t / 60:10.0f} {100 * deviation.max():24.2f} % {100 * np.sqrt(np.mean(deviation**2)):13.2f} %"
    )
common = np.linspace(0.0, 3.9e-3, 200)
collapsed = np.array(
    [np.interp(common, x / np.sqrt(t), theta) for t, theta in profiles.items()]
)
late = times >= 600.0
print(
    f"spread of the four profiles at equal φ: {100 * np.ptp(collapsed, axis=0).max() / (inlet - initial):.2f} % of θ0 - θi"
)
print(
    f"cumulative infiltration from 10 min on: I / (S √t) - 1 within "
    f"{100 * np.abs(infiltration[late] / (sorptivity * np.sqrt(times[late])) - 1).max():.2f} %, "
    f"fitted exponent {np.polyfit(np.log(times[late]), np.log(infiltration[late]), 1)[0]:.4f} (similarity: 0.5)"
)

# ## Figure
#
# The wetting profiles in the column, the same profiles against the similarity variable,
# and the cumulative infiltration against $\sqrt{t}$, a straight line of slope $S$.

fine = np.linspace(0.0, 0.32, 800)
fig = plots.figure(1, 3, size=(16, 4.8))
plots.line(
    points={
        f"{t / 60:.0f} min": (1e2 * x[::5], theta[::5]) for t, theta in profiles.items()
    },
    reference={
        f"{t / 60:.0f} min": (1e2 * fine, profile(fine / np.sqrt(t))) for t in profiles
    },
    xlimits=(0.0, 32.0),
    xlabel="position x [cm]",
    ylabel="water content θ",
    title="Wetting profiles: simulation (markers), similarity (dashed)",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    points={
        f"{t / 60:.0f} min": (1e3 * x[::5] / np.sqrt(t), theta[::5])
        for t, theta in profiles.items()
    },
    reference={"Θ(φ), similarity solution": (1e3 * phi, profile(phi))},
    xlimits=(0.0, 1e3 * far),
    xlabel="φ = x / √t [mm/√s]",
    ylabel="water content θ",
    title="All profiles on one curve",
    legend="upper right",
    ax=fig[1],
)
plots.line(
    points={"simulation": (np.sqrt(times[::3]), 1e3 * infiltration[::3])},
    reference={"S √t": (np.sqrt(times), 1e3 * sorptivity * np.sqrt(times))},
    xlabel="√t [√s]",
    ylabel="cumulative infiltration I [mm]",
    title="Sorptivity law I = S √t",
    legend="upper left",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Lower `initial_head` to start from a drier soil: the diffusivity contrast across the
# front grows and the harmonic-mean conductivity needs finer cells to keep the early
# infiltration right. Set `gravity=1.0` to turn the column vertical and watch the
# late-time infiltration rate approach the saturated conductivity instead of decaying as
# $1/\sqrt{t}$ (Philip's two-term law $I=S\sqrt{t}+At$).
