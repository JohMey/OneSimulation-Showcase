"""OneSimulation example: split-and-recombine micromixer at five flow rates (3D).
Requires the installed, activated OneSimulation runtime.

The same micromixer is run as a batch of five flow rates over more than a decade. The mixer is read
from a STEP file; for every flow rate the 3D flow is solved with Taylor-Hood elements and the solute
with quadratic elements. The pressure drop is checked for its proportionality to the flow rate
(creeping flow) and the outlet mixing against the lamination law of diffusion across the layers.
"""

# # Split-and-recombine micromixer at five flow rates
#
# A micromixer is designed for a range of flow rates, not for one, and the design study runs the
# same model as a batch over that range. In a split-and-recombine mixer the layers that leave the
# last element have a thickness $s$ set by the geometry; the faster the flow, the less time diffusion
# has to bridge them. With the residence time $t\propto 1/\bar U$ the unmixed fraction decays as
# $e^{-\pi^2Dt/s^2}$, so its logarithm falls linearly with $1/\bar U$, that is with the inverse
# Péclet number $1/Pe=D/(\bar Uw)$ (lamination law; V. Hessel, H. Löwe, F. Schönfeld, *Chem. Eng.
# Sci.* 60, 2479-2501, 2005). In creeping flow the pressure drop grows in proportion to the flow
# rate.
#
# The mixer comes from the CAD file `split_recombine_micromixer_flow_rates.step`, written by
# `split_recombine_micromixer_flow_rates_geometry.py`: square channels of side $w=1$ mm with two
# split-and-recombine elements and a straight outlet section of 4 mm. Water with a solute of
# diffusivity $D=10^{-8}$ m²/s enters at mean speeds from 0.15 to 2 mm/s ($Re$ from 0.15 to 2,
# $Pe$ from 15 to 200).
#
# | face | flow | solute |
# |---|---|---|
# | `inlet` | fully developed duct profile, mean speed $\bar U$ | $c=\tfrac12\left(1+\tanh((w/2-y)/0.05\,\text{mm})\right)$ |
# | `outlet` | traction free | no diffusive flux |
# | all walls | no slip | no flux |
#
# **Reference.** Creeping flow: $\Delta p\propto\bar U$. Lamination law: in the outlet section
# $\ln(1-M)$, with the mixing index $M$, falls linearly with $1/Pe$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid, NewtonianFluid
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.heat_transfer import (
    ConvectiveHeatTransfer,
    FixedTemperature,
    HeatFlux,
)
from onesim.physics.hydrodynamics import (
    FixedVelocity,
    FlowTraction,
    IncompressibleNavierStokes,
    NoSlipWall,
    fully_developed_inflow,
)
from onesim.results.sampling import field_function
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# Tetrahedra of 0.25 mm; `split_boundary_cells` divides the cells whose corners all lie on walls.

mm = 1e-3
mixer = load("split_recombine_micromixer_flow_rates.step")
faces = mixer.boundaries
walls = faces.all - faces.inlet - faces.outlet
side = mixer.size[1]
mesh = TetrahedralMesher(target_size=0.25 * mm, split_boundary_cells=True).generate(
    mixer
)

view = plots.Camera(elevation=30, azimuth=-55)
fig = plots.figure(1, 2, size=(16, 5.5))
plots.geometry(
    mixer,
    selection={"inlet": faces.inlet, "outlet": faces.outlet},
    labels=False,
    camera=view,
    title="Split-and-recombine micromixer",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## The batch of flow rates
#
# For each mean speed one flow problem and one solute problem in its flow.

water, diffusivity = NewtonianFluid(density=1000.0, dynamic_viscosity=1.0e-3), 1e-8
solute = IsotropicSolid(
    density=1.0,
    specific_heat_capacity=1.0,
    thermal_conductivity=diffusivity,
    youngs_modulus=1.0,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=0.0,
)
speeds = (0.15e-3, 0.3e-3, 0.6e-3, 1.2e-3, 2e-3)
runs = {}
for speed in speeds:
    flow = IncompressibleNavierStokes(domain=mixer.volume, material=water)
    problem = Problem(
        name="SAR mixer batch: flow",
        geometry=mixer,
        physics=(flow,),
        boundary_conditions=(
            FixedVelocity(
                boundary=faces.inlet,
                physics=flow,
                value=fully_developed_inflow(
                    mesh, faces.inlet, mean_velocity=speed, no_slip=walls
                ),
            ),
            FlowTraction(boundary=faces.outlet, physics=flow, value=(0.0, 0.0, 0.0)),
            NoSlipWall(boundary=walls, physics=flow),
        ),
    )
    numerics = Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        initial_guess={flow.velocity: (0.0, 0.0, 0.0)},
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(),
            absolute_tolerance=1e-20,
            relative_tolerance=1e-9,
        ),
    )
    solved = Simulation(
        problem=problem, mesh=mesh, numerics=numerics
    ).solve_stationary()
    velocity = solved.field(flow.velocity)
    transport = ConvectiveHeatTransfer(
        domain=mixer.volume, material=solute, velocity=field_function(velocity)
    )
    problem = Problem(
        name="SAR mixer batch: solute",
        geometry=mixer,
        physics=(transport,),
        boundary_conditions=(
            FixedTemperature(
                physics=transport,
                boundary=faces.inlet,
                value=lambda y: 0.5 * (1 + np.tanh((side / 2 - y) / (0.05 * mm))),
            ),
            HeatFlux(physics=transport, boundary=faces.all - faces.inlet, value=0.0),
        ),
    )
    numerics = Numerics(
        spatial=FiniteElement(
            element=Lagrange(order=2), convection=GalerkinAdvection()
        ),
        linear_solver=SparseDirect(),
    )
    runs[speed] = (
        velocity,
        solved.field(flow.pressure),
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .solve_stationary()
        .field(transport.temperature),
    )

# ## Results
#
# Pressures and the flow-weighted mixing index $M=1-\sigma/\sigma_0$ on cross-sections of the
# straight inlet and outlet sections.

grid_y, grid_z = np.meshgrid(
    np.linspace(0.04, 0.96, 24) * side, np.linspace(0.04, 0.96, 24) * side
)
section = lambda x: np.column_stack(
    (np.full(grid_y.size, x), grid_y.ravel(), grid_z.ravel())
)
sample = lambda run, x: (
    np.ravel(run[2].evaluate(section(x))),
    run[0].evaluate(section(x))[:, 0],
)
deviation = lambda values, weights: np.sqrt(
    np.average((values - np.average(values, weights=weights)) ** 2, weights=weights)
)
stations = (9.2 * mm, 11 * mm, 12.8 * mm)
mixing = {
    speed: np.array(
        [
            1 - deviation(*sample(run, x)) / deviation(*sample(run, 0.5 * mm))
            for x in stations
        ]
    )
    for speed, run in runs.items()
}
drops = {
    speed: np.mean(run[1].evaluate(section(0.5 * mm)))
    - np.mean(run[1].evaluate(section(12.8 * mm)))
    for speed, run in runs.items()
}
peclet = {speed: speed * side / diffusivity for speed in speeds}
slope, offset = np.polyfit(
    [1 / peclet[s] for s in speeds], [np.log(1 - mixing[s][-1]) for s in speeds], 1
)
fitted = {s: 1 - np.exp(offset + slope / peclet[s]) for s in speeds}

print(
    " U [mm/s]   Re     Pe    Δp [Pa]   (Δp/U) / (Δp/U at the lowest speed)   mixing index at x = 9.2, 11, 12.8 mm   lamination law"
)
for speed in speeds:
    print(
        f"{1e3 * speed:8.2f} {water.density * speed * side / water.dynamic_viscosity:5.2f} {peclet[speed]:6.0f} {drops[speed]:9.4f}"
        f" {drops[speed] / speed / (drops[speeds[0]] / speeds[0]):22.4f}            "
        + "  ".join(f"{m:.4f}" for m in mixing[speed])
        + f" {fitted[speed]:14.4f}"
    )
print(
    f"\nlamination law fitted to all five runs: ln(1 - M) = {offset:.3f} {slope:+.2f} / Pe"
)

# ## Figure
#
# Outlet mixing and pressure drop over the Péclet number, and the lamination-law plot.

inverse = np.linspace(0.0, 1.1 / peclet[speeds[0]], 50)
fig = plots.figure(1, 2, size=(15, 5.5))
plots.line(
    points={
        f"x = {x / mm:.1f} mm": (
            [peclet[s] for s in speeds],
            [mixing[s][index] for s in speeds],
        )
        for index, x in enumerate(stations)
    },
    right={"pressure drop": ([peclet[s] for s in speeds], [drops[s] for s in speeds])},
    right_label="pressure drop [Pa]",
    scale="logx",
    markers=True,
    ylimits=(0.0, 1.02),
    xlabel="Péclet number Pe = U w / D",
    ylabel="mixing index M",
    title="The batch of flow rates",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    points={
        "simulation, outlet": (
            [1 / peclet[s] for s in speeds],
            [np.log(1 - mixing[s][-1]) for s in speeds],
        )
    },
    reference={"lamination law, fitted": (inverse, offset + slope * inverse)},
    xlabel="1 / Pe",
    ylabel="ln(1 - M)",
    title="Lamination law: straight line in 1/Pe",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Add a third element in `split_recombine_micromixer_flow_rates_geometry.py`: the layer thickness
# halves and the slope of the lamination law grows about fourfold, so the same mixing is reached at
# four times the flow rate.
