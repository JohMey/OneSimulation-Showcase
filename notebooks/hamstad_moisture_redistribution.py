"""OneSimulation example: HAMSTAD benchmark 2, moisture redistribution in a homogeneous wall.
Requires the installed, activated OneSimulation runtime.

A 200 mm thick layer of a hygroscopic building material has reached equilibrium with very humid
air. Then the air on both sides becomes drier, 45 % relative humidity on one side and 65 % on the
other, and the layer dries out over weeks. This isothermal benchmark of the European HAMSTAD
project has an analytical solution. OneSimulation computes it with its moisture transport model,
driven by the relative humidity at the surfaces through the sorption isotherm and the vapour
transfer coefficients, and compares the moisture content profiles after 100, 300 and 1000 hours.
"""

# # HAMSTAD benchmark 2: moisture redistribution in a homogeneous wall
#
# The moisture content $w$ (kg/m³) of a porous material is in equilibrium with the relative
# humidity $\varphi$ of its pore air through the sorption isotherm $w(\varphi)$. Moisture moves as
# liquid (adsorbed films and capillary water) and as vapour,
#
# $$\frac{\partial w}{\partial t}+\nabla\cdot\mathbf g=0,\qquad
#   \mathbf g=-D_w\,\nabla w-\delta_p\,p_{sat}\,\nabla\varphi ,$$
#
# with the moisture diffusivity $D_w$ and the vapour permeability $\delta_p$. At a surface the
# vapour exchange with the ambient air is
# $\mathbf g\cdot\mathbf n=\beta\,p_{sat}\,(\varphi_s-\varphi_a)$. OneSimulation's
# `MoistureTransport` solves this balance with the moisture content as conserved state and the
# relative humidity from the isotherm; `VapourSurfaceTransfer` states the surface law.
#
# | Quantity | Value |
# |---|---|
# | thickness | 200 mm |
# | temperature | 20 °C, isothermal |
# | sorption isotherm | $w=116\,\big(1-\ln\varphi/0.118\big)^{-0.869}$ kg/m³ |
# | moisture diffusivity $D_w$ | $6\cdot10^{-10}$ m²/s |
# | vapour permeability $\delta_p$ | $10^{-15}$ s |
# | initial state | $\varphi=95\,\%$ |
# | ambient air | $\varphi_a=45\,\%$ at $x=0$, $65\,\%$ at $x=0.2$ m |
# | vapour transfer coefficient $\beta$ | $10^{-3}$ s/m on both sides |
#
# **Reference.** C.-E. Hagentoft et al., Assessment method of numerical prediction models for
# combined heat, air and moisture transfer in building components: benchmarks for one-dimensional
# cases, *J. Thermal Envelope and Building Science* 27 (2004) 327-352. With the constant
# diffusivity the moisture content obeys the linear diffusion equation, the surface resistance is
# negligible ($\beta\,p_{sat}\,L/(D_w\,dw/d\varphi)\sim10^5$) and the vapour part is four to five
# orders smaller than the liquid part. The analytical solution is then the Fourier series of the
# slab between the surface equilibria $w_0=w(0.45)$ and $w_L=w(0.65)$, starting from $w_i=w(0.95)$:
#
# $$w(x,t)=w_0+(w_L-w_0)\frac{x}{L}+\sum_{n\ge1}b_n\sin\frac{n\pi x}{L}\,e^{-D_w(n\pi/L)^2t},\qquad
#   b_n=\frac{2}{n\pi}\Big[(w_i-w_0)\big(1-(-1)^n\big)+(w_L-w_0)(-1)^n\Big].$$

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fvm import CentralDifference, FiniteVolume, LeastSquaresGradient
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF
from onesim.physics.hydrodynamics import (
    HansenIsotherm,
    MoistureTransport,
    VapourSurfaceTransfer,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()


# ## Model
#
# The isotherm of the benchmark is a Hansen isotherm
# $w=w_h\,(1-\ln\varphi/a)^{-1/n}$ with $w_h=116$ kg/m³, $a=0.118$ and $1/n=0.869$. The wall is a
# line through its thickness; the initial moisture content is the equilibrium with 95 %.

thickness, diffusivity, hour = 0.2, 6e-10, 3600.0
isotherm = HansenIsotherm(saturation_content=116.0, a=0.118, n=1 / 0.869)
wall = Line(start=(0.0, 0.0, 0.0), end=(thickness, 0.0, 0.0))
moisture = MoistureTransport(
    domain=wall.line,
    sorption_isotherm=isotherm,
    moisture_diffusivity=diffusivity,
    vapour_permeability=1e-15,
    temperature=293.15,
)
problem = Problem(
    name="HAMSTAD benchmark 2",
    geometry=wall,
    physics=(moisture,),
    initial_conditions=(
        InitialValue(moisture.moisture_content, float(isotherm.moisture_content(0.95))),
    ),
    boundary_conditions=(
        VapourSurfaceTransfer(
            physics=moisture,
            boundary=wall.boundaries.start,
            ambient_relative_humidity=0.45,
            transfer_coefficient=1e-3,
        ),
        VapourSurfaceTransfer(
            physics=moisture,
            boundary=wall.boundaries.end,
            ambient_relative_humidity=0.65,
            transfer_coefficient=1e-3,
        ),
    ),
)

# ## Simulation
#
# Finite volumes on 50 and 200 cells with the second-order BDF method, 4 h and 1 h steps over
# 1000 hours; Newton's method solves each step.

runs = {
    (cells, step): Simulation(
        problem=problem,
        mesh=StructuredLineMesher(cells=cells).generate(wall),
        numerics=Numerics(
            spatial=FiniteVolume(
                gradient=LeastSquaresGradient(), diffusion=CentralDifference()
            ),
            time_integrator=BDF(order=2),
            nonlinear_solver=Newton(
                linear_solver=SparseDirect(), absolute_tolerance=1e-9
            ),
        ),
    )
    .run(start_time=0.0, end_time=1000 * hour, time_step=step * hour)
    .field(moisture.moisture_content)
    for cells, step in ((50, 4.0), (200, 1.0))
}

# ## Comparison with the analytical solution
#
# The moisture content in every cell after 100, 300 and 1000 hours against the Fourier series.

w0, wl, wi = (float(isotherm.moisture_content(phi)) for phi in (0.45, 0.65, 0.95))
modes = np.arange(1, 4001)
wavenumbers = modes * np.pi / thickness
amplitudes = (
    2
    / (modes * np.pi)
    * ((wi - w0) * (1 - (-1.0) ** modes) + (wl - w0) * (-1.0) ** modes)
)
print(
    f"surface equilibria w(45 %) = {w0:.3f}, w(65 %) = {wl:.3f}, initial w(95 %) = {wi:.3f} kg/m³"
)
print(
    " cells   step [h]   time [h]   largest deviation [kg/m³]   relative to the drop w_i - w_0"
)
for (cells, step), field in runs.items():
    x = np.asarray(field.space.dof_coordinates)[:, 0]
    for hours in (100, 300, 1000):
        exact = (
            w0
            + (wl - w0) * x / thickness
            + np.sin(np.outer(x, wavenumbers))
            @ (amplitudes * np.exp(-diffusivity * wavenumbers**2 * hours * hour))
        )
        deviation = np.max(np.abs(np.asarray(field.at(hours * hour).values) - exact))
        print(
            f"{cells:6d} {step:9.0f} {hours:10d} {deviation:22.4f} {deviation / (wi - w0):27.1e}"
        )
fine = runs[(200, 1.0)]
times = np.arange(0, 1001, 20)
stored = np.array([fine.integrate(time=t * hour).value for t in times])
analytical_store = thickness * (
    (w0 + wl) / 2
    + np.exp(-diffusivity * np.outer(times * hour, wavenumbers**2))
    @ (amplitudes * (1 - (-1.0) ** modes) / (modes * np.pi))
)
print(
    f"stored moisture after 1000 h: {stored[-1]:.4f} kg/m², analytical {analytical_store[-1]:.4f} kg/m², "
    f"largest deviation over the run {np.max(np.abs(stored - analytical_store)):.4f} kg/m²"
)

# ## Figure
#
# Left: the moisture content profiles on 200 cells (markers, every fourth cell) and the analytical
# solution (dashed). Middle: the relative humidity in the pores from the isotherm, which carries
# the strong nonlinearity of the problem. Right: the moisture stored in the wall over time.

x = np.asarray(fine.space.dof_coordinates)[:, 0]
profiles = {
    f"{hours} h": np.asarray(fine.at(hours * hour).values) for hours in (100, 300, 1000)
}
fig = plots.figure(1, 3, size=(18, 5))
plots.line(
    points={label: (1e3 * x[::4], values[::4]) for label, values in profiles.items()},
    reference={
        f"{hours} h": (
            1e3 * x,
            w0
            + (wl - w0) * x / thickness
            + np.sin(np.outer(x, wavenumbers))
            @ (amplitudes * np.exp(-diffusivity * wavenumbers**2 * hours * hour)),
        )
        for hours in (100, 300, 1000)
    },
    hlines={"initial": wi},
    xlabel="position x [mm]",
    ylabel="moisture content w [kg/m³]",
    title="Moisture content (dashed: analytical)",
    legend="lower center",
    ax=fig[0],
)
plots.line(
    y={
        label: (1e3 * x, 100 * isotherm.relative_humidity(values))
        for label, values in profiles.items()
    },
    xlabel="position x [mm]",
    ylabel="relative humidity [%]",
    title="Relative humidity in the pores",
    legend="lower center",
    ax=fig[1],
)
plots.line(
    y={"OneSimulation": (times, stored)},
    reference={"analytical": (times, analytical_store)},
    xlabel="time [h]",
    ylabel="stored moisture [kg/m²]",
    title="Moisture in the wall",
    legend="upper right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Give the material a humidity-dependent diffusivity, `moisture_diffusivity=` any function of the
# relative humidity, or a `SorptionIsotherm` from measured data: the model stays the same and only
# the analytical comparison is lost. The same declaration runs on planar and revolved sections and
# in three dimensions, for example a corner of the wall or a drying timber column, and
# `FixedRelativeHumidity` and `MoistureFlux` state other surface conditions.
