"""OneSimulation example: split-and-recombine micromixer (3D).
Requires the installed, activated OneSimulation runtime.

Two liquids enter a square microchannel side by side; two split-and-recombine elements cut the
stream, reshape the halves and join them again, doubling the number of layers each time. The mixer
is read from a STEP file; the 3D creeping flow is solved with Taylor-Hood elements and the solute
with quadratic elements. The results are checked against the exact square-duct flow, for mass
conservation, against the layer doubling and with a mesh study.
"""

# # Split-and-recombine micromixer
#
# Without turbulence, two liquids in a microchannel mix only by diffusion across their interface.
# Split-and-recombine (SAR) mixers speed this up geometrically: each element cuts the stream into
# an upper and a lower half, squeezes the upper half into the left and the lower half into the right
# part of the channel, stretches both back to full height and joins them side by side. The two
# layers become four after the first element and eight after the second, and the diffusion distance
# halves each time (multilamination).
#
# The mixer comes from the CAD file `split_recombine_micromixer.step`, written by
# `split_recombine_micromixer_geometry.py`: square channels of side $w=1$ mm, two elements 4 mm long
# whose branches are separated by walls 0.2 mm thick, an inlet section of 1 mm and an outlet section
# of 4 mm. Water enters at the mean speed $\bar U=1$ mm/s ($Re=1$); the left half carries the solute.
# The diffusivity gives the Péclet number $Pe=\bar Uw/D=100$.
#
# $$-\nabla p+\mu\nabla^2\mathbf u=0,\quad\nabla\cdot\mathbf u=0,\qquad\mathbf u\cdot\nabla c=D\nabla^2c$$
#
# | face | flow | solute |
# |---|---|---|
# | `inlet` | fully developed duct profile, mean speed $\bar U$ | $c=\tfrac12\left(1+\tanh((w/2-y)/0.05\,\text{mm})\right)$ |
# | `outlet` | traction free | no diffusive flux |
# | all walls | no slip | no flux |
#
# **Reference.** In the straight outlet section the flow is the fully developed flow in a square
# duct, whose pressure gradient is exact: $dp/dx=-12\,\mu Q/(0.4217\,w^4)$ (F. M. White, *Viscous
# Fluid Flow*, section 3-3). Water and solute that enter must leave. The layer count must double in
# every element (V. Hessel, H. Löwe, F. Schönfeld, "Micromixers - a review on passive and active
# mixing principles", *Chem. Eng. Sci.* 60, 2479-2501, 2005).

import numpy as np
from onesim.cad.d2 import Plane
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
from onesim.results.derived import derived_field
from onesim.results.sampling import field_function
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# Tetrahedra of 0.25 mm and, for the mesh study, 0.2 mm. `split_boundary_cells` divides the few cells
# whose four corners all lie on walls, where the quadratic velocity would have no free node.

mm = 1e-3
mixer = load("split_recombine_micromixer.step")
faces = mixer.boundaries
walls = faces.all - faces.inlet - faces.outlet
side = mixer.size[1]
meshes = {
    size: TetrahedralMesher(target_size=size * mm, split_boundary_cells=True).generate(
        mixer
    )
    for size in (0.25, 0.2)
}

view = plots.Camera(elevation=30, azimuth=-55)
fig = plots.figure(1, 2, size=(16, 5.5))
plots.geometry(
    mixer,
    selection={"inlet": faces.inlet, "outlet": faces.outlet},
    labels=False,
    camera=view,
    title="Two split-and-recombine elements",
    ax=fig[0],
)
plots.mesh(
    meshes[0.2],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{meshes[0.2].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Flow and solute
#
# The creeping flow on each mesh, then the solute in it.

water, speed = NewtonianFluid(density=1000.0, dynamic_viscosity=1.0e-3), 1e-3
solute = IsotropicSolid(
    density=1.0,
    specific_heat_capacity=1.0,
    thermal_conductivity=speed * side / 100,
    youngs_modulus=1.0,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=0.0,
)
flows, concentrations = {}, {}
for size, mesh in meshes.items():
    flow = IncompressibleNavierStokes(domain=mixer.volume, material=water)
    problem = Problem(
        name="SAR mixer: flow",
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
    flows[size] = solved.field(flow.velocity), solved.field(flow.pressure)
    transport = ConvectiveHeatTransfer(
        domain=mixer.volume, material=solute, velocity=field_function(flows[size][0])
    )
    problem = Problem(
        name="SAR mixer: solute",
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
    concentrations[size] = (
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .solve_stationary()
        .field(transport.temperature)
    )

# ## Results
#
# Cross-sections every half millimetre, sampled at the points in the liquid at least 0.08 mm from a
# wall; the mixing index is $1-\sigma/\sigma_0$ with the flow-weighted standard deviation $\sigma$ and
# its value $\sigma_0$ at the inlet. The layers are counted as the sign changes of $c-\bar c$ across
# the channel at mid-height.

grid_y, grid_z = np.meshgrid(
    np.linspace(0.02, 0.98, 25) * side, np.linspace(0.02, 0.98, 25) * side
)
section = lambda x: np.column_stack(
    (np.full(grid_y.size, x), grid_y.ravel(), grid_z.ravel())
)
margin = 0.08 * mm * np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
stations = np.arange(0.25, 13.0, 0.5) * mm
inside = [
    np.all([mixer.contains(section(x) + shift) for shift in margin], axis=0)
    for x in stations
]
mixing = {}
for size, c in concentrations.items():
    w = [
        flows[size][0].evaluate(section(x)[keep])[:, 0]
        for x, keep in zip(stations, inside)
    ]
    v = [np.ravel(c.evaluate(section(x)[keep])) for x, keep in zip(stations, inside)]
    spread = np.array(
        [
            np.sqrt(np.average((a - np.average(a, weights=b)) ** 2, weights=b))
            for a, b in zip(v, w)
        ]
    )
    mixing[size] = 1 - spread / spread[0]
across = np.column_stack(
    (np.zeros(201), np.linspace(0.02, 0.98, 201) * side, np.full(201, side / 2))
)
layers = {
    x: np.ravel(concentrations[0.2].evaluate(across + (x, 0, 0)))
    for x in (0.5 * mm, 4.5 * mm, 8.5 * mm)
}
count = {
    x: int(np.count_nonzero(np.diff(np.sign(c - c.mean())))) + 1
    for x, c in layers.items()
}
duct = 12 * water.dynamic_viscosity * speed * side**2 / (0.4217 * side**4)

for size in meshes:
    velocity, pressure = flows[size]
    gradient = (
        np.mean(pressure.evaluate(section(10 * mm)))
        - np.mean(pressure.evaluate(section(12 * mm)))
    ) / (2 * mm)
    water_in, water_out = (
        velocity.integrate(boundary=faces.inlet, normal_component=True).value,
        velocity.integrate(boundary=faces.outlet, normal_component=True).value,
    )
    carried = (
        lambda face: derived_field(
            velocity,
            lambda u, points, c=concentrations[size]: np.ravel(c.evaluate(points))
            * u[:, 0],
            name="solute_flux",
            unit="1",
        )
        .integrate(boundary=face)
        .value
    )
    print(
        f"mesh {size} mm, {meshes[size].cell_count} tetrahedra: outlet pressure gradient / square duct {gradient / duct:.4f}, "
        f"water balance {(water_in + water_out) / water_out:+.1e}, solute balance {carried(faces.outlet) / carried(faces.inlet) - 1:+.1e}, "
        f"mixing index at the outlet {mixing[size][-1]:.3f}"
    )
print(
    "layers across the channel at mid-height: "
    + ", ".join(f"x = {x / mm:.1f} mm: {n}" for x, n in count.items())
)

# ## Figure
#
# The solute on cross-sections before, between and behind the elements, and the mixing index along
# the mixer.

fig = plots.figure(
    layout=[["a", "b", "c", "d"], ["mixing", "mixing", "mixing", "mixing"]],
    size=(18, 10),
)
for panel, x, title in (
    ("a", 0.5, "inlet"),
    ("b", 4.5, "after one element"),
    ("c", 8.5, "after two elements"),
    ("d", 12.5, "outlet"),
):
    plots.section(
        concentrations[0.2],
        plane=Plane(origin=(x * mm, 0, 0), normal=(1, 0, 0)),
        aspect=1.0,
        limits=(0.0, 1.0),
        subdivisions=2,
        label="solute c",
        title=f"x = {x} mm, {title}",
        ax=fig[panel],
    )
plots.line(
    stations / mm,
    {"mesh 0.2 mm": mixing[0.2]},
    points={"mesh 0.25 mm": (stations[::2] / mm, mixing[0.25][::2])},
    vlines={"element 1": 1.0, "element 2": 5.0, "outlet section": 9.0},
    ylimits=(0.0, 1.02),
    xlabel="distance x [mm]",
    ylabel="mixing index 1 - σ/σ0",
    title="Mixing along the mixer, Pe = 100",
    legend="lower right",
    ax=fig["mixing"],
)
plots.show()

# ## Going further
#
# Add elements in `split_recombine_micromixer_geometry.py` and write the STEP file again: every
# element doubles the layer count, so the mixing length falls exponentially with the number of
# elements. Raising the flow rate raises the Péclet number in proportion.
