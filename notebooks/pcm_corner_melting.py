"""OneSimulation example: melting of a phase change material in a heated corner.
Requires the installed, activated OneSimulation runtime.

A block of solid paraffin is heated on two adjacent walls and melts from them. The block is read
from a STEP file and cut to its cross-section; the latent heat sits in a smoothed melting window
and every time step balances the enthalpy. Away from the corner the melt front is compared with the two-phase Neumann solution, at
the corner it shows the two-dimensional bulge.
"""

# # Melting of a phase change material in a heated corner
#
# Latent-heat stores use paraffins that melt a few degrees above room temperature. A square block
# of n-octadecane (60 mm × 60 mm), solid at 20 °C, is suddenly heated to 40 °C on two adjacent walls;
# the other two walls are insulated. The paraffin melts at $T_m=28.2$ °C and the melt front moves
# into the block, fastest near the heated corner where heat arrives from two sides. Heat moves by
# conduction only (the melt is thin and natural convection is neglected):
#
# $$\rho\,c(T)\,\frac{\partial T}{\partial t}=\nabla\cdot\big(k(T)\,\nabla T\big).$$
#
# | property | solid | liquid |
# |---|---|---|
# | conductivity $k$ [W/(m K)] | 0.358 | 0.152 |
# | specific heat capacity $c$ [J/(kg K)] | 1934 | 2196 |
# | density $\rho$ [kg/m³] | 800 | 800 |
# | latent heat $L$ [kJ/kg] | 244 | |
#
# | wall | condition |
# |---|---|
# | `heated`: $x=0$ and $y=0$ | $T=40$ °C |
# | `insulated`: $x=60$ mm and $y=60$ mm | no heat flux |
#
# **Melting window and enthalpy.** The melting is spread over 2 K around $T_m$: the liquid
# fraction rises smoothly across this window, $c(T)$ carries a bump whose area is exactly the latent
# heat, and $k(T)$ follows the liquid fraction. One heat equation then covers both phases. Each
# implicit time step balances the enthalpy $H(T)=\int\rho\,c\,\mathrm{d}T$ rather than
# $\rho\,c(T)\,\Delta T$, so no latent heat is lost when the front crosses a node within a step.
#
# The block comes from the CAD file `pcm_corner_melting.step`; the script
# `pcm_corner_melting_geometry.py` builds it with the CAD objects of OneSimulation. The
# two-dimensional model is its mid-plane section.
#
# **Reference.** Near the insulated wall $y=60$ mm the heat comes from the wall $x=0$ alone, and the
# front follows the Neumann solution for melting of a semi-infinite solid (Carslaw and Jaeger,
# *Conduction of Heat in Solids*, 1959, §11.2): $X(t)=2\lambda\sqrt{\alpha_l t}$ with
#
# $$\frac{e^{-\lambda^2}}{\operatorname{erf}\lambda}-\frac{k_s\nu\,(T_m-T_i)}{k_l\,(T_w-T_m)}\,
#   \frac{e^{-\nu^2\lambda^2}}{\operatorname{erfc}(\nu\lambda)}=\frac{\sqrt\pi\,\lambda\,\rho L}{\rho c_l\,(T_w-T_m)},
#   \qquad \nu=\sqrt{\alpha_l/\alpha_s}.$$
#
# The Neumann solution assumes a solid that extends without end. The block is 60 mm wide and is
# heated from a second wall, so its solid warms during the two hours and the computed front runs a
# few tenths of a millimetre ahead of the Neumann front towards the end. Along the diagonal no
# closed form exists; there the simulation shows how much further the front has travelled.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import PhaseChangeMaterial
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import AdaptiveTimeStep, BackwardEuler
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.optimize import brentq
from scipy.special import erf, erfc

theme.set()

# ## Geometry and mesh
#
# `load` returns the block with its named faces; `section` returns the mid-plane square, whose
# sides keep the names of the faces they were cut from. The triangles are 1.5 mm at the heated
# walls, where the front travels, and grow to 4 mm in the solid.

block = load("pcm_corner_melting.step")
faces = block.boundaries
side = block.size[0]
region = block.section((0, 0, 1), origin=(0, 0, block.size[2] / 2))
sides = region.boundaries
mesh = TriangularMesher(
    target_size=0.004,
    size_field=lambda p: np.minimum(
        0.004, 0.0015 + 0.1 * np.maximum(np.minimum(p[:, 0], p[:, 1]) - 0.018, 0.0)
    ),
).generate(region)

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    block,
    selection={"heated, 40 °C": faces.heated, "insulated": faces.insulated},
    labels=False,
    camera=plots.Camera(elevation=40, azimuth=-60),
    title="Paraffin block with its named faces",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Mid-plane section: {mesh.cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Material
#
# `PhaseChangeMaterial` holds both phases, the melting temperature, the latent heat and the width of
# the melting window; its enthalpy across the window confirms the latent heat it carries.

density, latent, melting = 800.0, 244e3, 301.35
solid, liquid = {"k": 0.358, "c": 1934.0}, {"k": 0.152, "c": 2196.0}
paraffin = PhaseChangeMaterial(
    density=density,
    melting_temperature=melting,
    latent_heat=latent,
    transition_width=2.0,
    solid_heat_capacity=solid["c"],
    liquid_heat_capacity=liquid["c"],
    solid_conductivity=solid["k"],
    liquid_conductivity=liquid["k"],
)
print(
    f"latent heat in the melting window: {(paraffin.enthalpy(melting + 1.0) - paraffin.enthalpy(melting - 1.0) - 2.0 * (solid['c'] + liquid['c']) / 2) / 1e3:.1f} kJ/kg "
    f"(n-octadecane {latent / 1e3:.0f} kJ/kg)"
)

# ## Model
#
# Newton's method in every implicit Euler step of the enthalpy balance. The step size adapts: it
# starts at 5 s, where the walls jump to 40 °C, keeps the estimated local error below 0.3 K and grows
# up to 4 minutes; two hours are computed.

wall, initial, hour = 313.15, 293.15, 3600.0
model = HeatConduction(domain=region.domain(), material=paraffin, heating=0.0)
problem = Problem(
    name="Paraffin melting in a corner",
    geometry=region,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=sides.x_min | sides.y_min, physics=model, value=wall),
        HeatFlux(boundary=sides.x_max | sides.y_max, physics=model, value=0.0),
    ),
    initial_conditions=(InitialValue(model.temperature, initial),),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=SparseDirect(),
    nonlinear_solver=Newton(
        linear_solver=SparseDirect(), relative_tolerance=1e-6, maximum_iterations=25
    ),
    time_integrator=BackwardEuler(),
)
steps = AdaptiveTimeStep(
    initial=5.0,
    minimum=0.05,
    maximum=240.0,
    relative_tolerance=0.0,
    absolute_tolerance=0.3,
)
temperature = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .run(start_time=0.0, end_time=2 * hour, time_step=steps)
    .field(model.temperature)
)

# ## Results
#
# The melt front along the line $y=57$ mm next to the insulated wall against the Neumann front, and
# along the diagonal from the heated corner, every half hour.

alpha_l, alpha_s = liquid["k"] / (density * liquid["c"]), solid["k"] / (
    density * solid["c"]
)
nu = np.sqrt(alpha_l / alpha_s)
lam = brentq(
    lambda l: np.exp(-(l**2)) / erf(l)
    - solid["k"]
    * nu
    * (melting - initial)
    / (liquid["k"] * (wall - melting))
    * np.exp(-((nu * l) ** 2))
    / erfc(nu * l)
    - np.sqrt(np.pi) * l * latent / (liquid["c"] * (wall - melting)),
    1e-6,
    3.0,
)
s = np.linspace(0, 0.04, 801)
times, edge = temperature.time_series(np.column_stack((s, np.full_like(s, 0.057))))
diagonal = temperature.time_series(np.column_stack((s, s)))[1]
front = np.array([np.interp(-melting, -profile, s) for profile in edge])
corner = np.array(
    [np.sqrt(2) * np.interp(-melting, -profile, s) for profile in diagonal]
)
print(f"Neumann constant lambda = {lam:.4f}")
print(
    "time [h]   front at y = 57 mm [mm]   Neumann [mm]   deviation [mm]   front along the diagonal [mm]"
)
for index in [
    int(np.argmin(np.abs(times - hours * hour))) for hours in (0.5, 1.0, 1.5, 2.0)
]:
    exact = 2 * lam * np.sqrt(alpha_l * times[index])
    print(
        f"{times[index] / hour:8.1f} {1e3 * front[index]:25.2f} {1e3 * exact:14.2f} {1e3 * (front[index] - exact):+16.2f} "
        f"{1e3 * corner[index]:29.2f}"
    )

# ## Figure
#
# The temperature after two hours with isotherms, and the front position over the square root of time near the insulated wall and along the diagonal.

fig = plots.figure(1, 2, size=(15, 6.5))
plots.surface(
    temperature,
    time=2 * hour,
    component=lambda T, position: T - 273.15,
    limits=(20, 40),
    contours=10,
    subdivisions=3,
    xlabel="x [m]",
    ylabel="y [m]",
    label="T [°C]",
    title="Temperature after 2 h",
    ax=fig[0],
)
plots.line(
    points={
        "near the insulated wall, y = 57 mm": (
            np.sqrt(times[1:] / hour),
            1e3 * front[1:],
        ),
        "along the diagonal, distance from the corner": (
            np.sqrt(times[1:] / hour),
            1e3 * corner[1:],
        ),
    },
    reference={
        "Neumann, X = 2 λ (α_l t)^½": (
            np.sqrt(times[1:] / hour),
            2e3 * lam * np.sqrt(alpha_l * times[1:]),
        )
    },
    xlabel="square root of time [h^½]",
    ylabel="melt front distance [mm]",
    title="Melt front",
    legend="upper left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Add natural convection in the melt (`ConvectiveHeatTransfer` with a buoyant flow), which bends the
# front upward, or put a metal fin into the block in `pcm_corner_melting_geometry.py` and compare
# the melting times.
