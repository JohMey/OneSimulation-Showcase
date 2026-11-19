"""OneSimulation example: Rayleigh-Benard convection and the critical Rayleigh number (2D).
Requires the installed, activated OneSimulation runtime.

A horizontal fluid layer heated from below stays at rest until the Rayleigh number exceeds a
critical value; then convection rolls form and carry heat upwards. The layer is read from a STEP
file and cut to its cross-section; the Navier-Stokes equations with the Boussinesq buoyancy force
and the heat transport are solved for a sweep of Rayleigh numbers. The onset found from the sweep
is compared with the critical Rayleigh number 1707.8 of linear stability theory, and the Nusselt
number with the published value for steady rolls.
"""

# # Rayleigh-Bénard convection: onset of the convection rolls
#
# A fluid layer of depth $d$ lies between two rigid horizontal plates; the lower plate is warmer
# than the upper one by $\Delta T$. Warm, light fluid sits under cold, heavy fluid, but viscosity
# and heat conduction damp every disturbance as long as the Rayleigh number
#
# $$Ra=\frac{g\beta\Delta T\,d^3}{\nu\alpha}$$
#
# stays below the critical value $Ra_c=1707.76$. Above it the conduction state is unstable and
# steady two-dimensional rolls of wavelength $2\pi d/3.117$ appear. The Nusselt number, the heat
# carried across the layer divided by the conducted heat, grows from 1 linearly in $Ra-Ra_c$ near
# the onset.
#
# In units of $d$, $\Delta T$ and $\alpha$ (Prandtl number $Pr=1$) the steady Boussinesq equations
# are solved on a layer one critical wavelength wide. Its side walls are planes of symmetry of the
# roll pattern (free slip, no heat flux), so two counter-rotating rolls fit exactly; the initial
# temperature carries a small disturbance of that shape. The layer comes from the CAD file
# `rayleigh_benard_convection_onset.step`, written by `rayleigh_benard_convection_onset_geometry.py`
# with the CAD objects of OneSimulation.
#
# | face | flow | heat |
# |---|---|---|
# | `heated_bottom` | no slip | $T=1$ |
# | `cooled_top` | no slip | $T=0$ |
# | `sides` | free slip (symmetry) | no flux |
#
# **Reference.** Linear stability of the layer between rigid plates: $Ra_c=1707.76$ at wave number
# $a_c=3.117$ (W. H. Reid and D. L. Harris, *Phys. Fluids* 1, 102 (1958); S. Chandrasekhar,
# *Hydrodynamic and Hydromagnetic Stability*, 1961, Ch. II). Steady rolls of the critical wavelength
# at $Ra=2500$: $Nu=1.47$ (R. M. Clever and F. H. Busse, *J. Fluid Mech.* 65, 625 (1974)).

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import ThermalFluid
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.coupled_systems import StationaryPartitionedCoupling
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange, TaylorHood
from onesim.numerics.field_transfer import FieldInterpolation
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import FixedPoint, Newton
from onesim.physics.heat_transfer import (
    BoussinesqBuoyancy,
    ConvectiveHeatTransfer,
    FixedTemperature,
    HeatFlux,
)
from onesim.physics.hydrodynamics import (
    FreeSlipLaw,
    IncompressibleNavierStokes,
    NoSlipWall,
    WallShear,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The cross-section at half thickness keeps the face names. Two meshes: triangles of 1/8 and 1/12
# of the layer depth.

part = load("rayleigh_benard_convection_onset.step")
layer = part.section((0, 0, 1), origin=(0, 0, part.size[2] / 2))
faces = layer.boundaries
width = layer.size[0]
meshes = {
    size: TriangularMesher(target_size=size).generate(layer) for size in (1 / 8, 1 / 12)
}

fig = plots.figure(1, 2, size=(14, 5))
plots.geometry(
    part,
    selection={
        "heated bottom": part.boundaries.heated_bottom,
        "cooled top": part.boundaries.cooled_top,
        "symmetry sides": part.boundaries.sides,
    },
    labels=False,
    camera=plots.Camera(elevation=62, azimuth=-90),
    title="Fluid layer, one critical wavelength wide",
    ax=fig[0],
)
plots.mesh(
    meshes[1 / 12],
    groups=("all",),
    legend=False,
    title=f"Finer mesh: {meshes[1 / 12].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Sweep over the Rayleigh number
#
# Flow (Taylor-Hood) and heat (quadratic elements) are coupled both ways: `BoussinesqBuoyancy` turns
# the temperature into the body force, the solved velocity carries the heat.
# `StationaryPartitionedCoupling` iterates the two stationary solvers to their common fixed point.
# One Rayleigh number lies below the onset, four above it, on the coarser mesh; $Ra=2500$ is also
# solved on the finer mesh.

prandtl = 1.0
fluid = ThermalFluid(
    density=1.0,
    dynamic_viscosity=prandtl,
    thermal_conductivity=1.0,
    specific_heat_capacity=1.0,
    thermal_expansion_coefficient=1.0,
)
flow = IncompressibleNavierStokes(domain=layer.domain(), material=fluid)
heat = ConvectiveHeatTransfer(domain=layer.domain(), material=fluid, velocity=None)
numerics = {
    flow: Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        initial_guess={flow.velocity: (0.0, 0.0)},
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(),
            absolute_tolerance=1e-10,
            relative_tolerance=1e-9,
        ),
    ),
    heat: Numerics(
        spatial=FiniteElement(
            element=Lagrange(order=2), convection=GalerkinAdvection()
        ),
        linear_solver=SparseDirect(),
    ),
}
runs = [(rayleigh, 1 / 8) for rayleigh in (1500.0, 1800.0, 2000.0, 2250.0, 2500.0)] + [
    (2500.0, 1 / 12)
]
results = {}
for rayleigh, size in runs:
    problem = Problem(
        name=f"Rayleigh-Benard layer, Ra = {rayleigh:.0f}",
        geometry=layer,
        physics=(flow, heat),
        boundary_conditions=(
            NoSlipWall(physics=flow, boundary=faces.heated_bottom | faces.cooled_top),
            WallShear(physics=flow, boundary=faces.sides, law=FreeSlipLaw()),
            FixedTemperature(physics=heat, boundary=faces.heated_bottom, value=1.0),
            FixedTemperature(physics=heat, boundary=faces.cooled_top, value=0.0),
            HeatFlux(physics=heat, boundary=faces.sides, value=0.0),
        ),
        constraints=(MeanValue(flow.pressure, 0.0),),
        initial_conditions=(
            InitialValue(
                heat.temperature,
                lambda x, y: 1
                - y
                + 0.05 * np.sin(np.pi * y) * np.cos(2 * np.pi * x / width),
            ),
        ),
        couplings=(
            FieldCoupling(
                source=heat.temperature,
                target=flow.field("body_force"),
                transfer=BoussinesqBuoyancy(
                    gravity=(0, -rayleigh * prandtl), reference_temperature=0.5
                ),
            ),
            FieldCoupling(
                source=flow.velocity,
                target=heat.advecting_velocity,
                transfer=FieldInterpolation(),
            ),
        ),
    )
    results[rayleigh, size] = Simulation(
        problem=problem,
        mesh={flow: meshes[size], heat: meshes[size]},
        numerics=numerics,
        coupling_solver=StationaryPartitionedCoupling(
            stationary_systems=(flow, heat),
            iteration=FixedPoint(
                relative_tolerance=1e-6,
                absolute_tolerance=1e-9,
                maximum_iterations=400,
                relaxation=0.8,
            ),
        ),
    ).solve_stationary()

# ## Onset and heat transfer
#
# The Nusselt number is the heat entering through the bottom per unit width. Near the onset
# $Nu-1$ and the square of the roll velocity both grow linearly with $Ra-Ra_c$; a fit through the
# convecting states, extrapolated to zero, gives two estimates of the critical Rayleigh number.

probe = np.array(
    [[x, y] for x in np.linspace(0, width, 41) for y in np.linspace(0.05, 0.95, 19)]
)
table = {
    key: (
        -result.field(heat.temperature).boundary_flux(faces.heated_bottom) / width,
        result.field(heat.temperature).boundary_flux(faces.cooled_top) / width,
        np.abs(result.field(flow.velocity).evaluate(probe)).max(),
        len(result.result_set.solver_report["coupling"]["scaled_changes"]),
    )
    for key, result in results.items()
}
print(
    "    Ra   mesh h   Nu (bottom)   Nu (top)   largest velocity [alpha/d]   coupling sweeps"
)
for (rayleigh, size), (bottom, top, speed, sweeps) in table.items():
    print(
        f"{rayleigh:6.0f} {size:8.4f} {bottom:13.5f} {top:10.5f} {speed:28.4f} {sweeps:17d}"
    )
convecting = [rayleigh for rayleigh, size in runs[:-1] if rayleigh > 1707.76]
nusselt_fit = np.polyfit(
    convecting, [table[rayleigh, 1 / 8][0] - 1 for rayleigh in convecting], 2
)
speed_fit = np.polyfit(
    convecting, [table[rayleigh, 1 / 8][2] ** 2 for rayleigh in convecting], 2
)
onsets = {
    name: min(np.roots(fit).real, key=lambda root: abs(root - 1700))
    for name, fit in (("Nu - 1", nusselt_fit), ("velocity²", speed_fit))
}
print(
    f"\nbelow the onset (Ra = 1500): Nu - 1 = {table[1500.0, 1 / 8][0] - 1:.1e}, largest velocity {table[1500.0, 1 / 8][2]:.1e}: "
    f"the disturbance has decayed"
)
for name, onset in onsets.items():
    print(
        f"critical Rayleigh number from {name:9s}: {onset:8.1f}, linear stability 1707.76 ({100 * (onset / 1707.76 - 1):+.2f} %)"
    )
print(
    f"Nusselt number at Ra = 2500: {table[2500.0, 1 / 8][0]:.4f}, Clever & Busse 1.47 ({100 * (table[2500.0, 1 / 8][0] / 1.47 - 1):+.2f} %); "
    f"finer mesh {table[2500.0, 1 / 12][0]:.4f} ({100 * (table[2500.0, 1 / 12][0] / table[2500.0, 1 / 8][0] - 1):+.3f} %)"
)

# ## Figure
#
# Temperature and roll streamlines at $Ra=2500$, and the Nusselt number over the Rayleigh number
# with the extrapolated onset.

grid_x, grid_y = np.linspace(0.0, width, 81), np.linspace(0.0, 1.0, 41)
flow_samples = (
    results[2500.0, 1 / 12]
    .field(flow.velocity)
    .evaluate(np.array([[a, b] for a in grid_x for b in grid_y]))
    .reshape(81, 41, 2)
)
curve = np.linspace(onsets["Nu - 1"], 2600.0, 200)
fig = plots.figure(
    layout=[["temperature", "nusselt"], ["streamlines", "nusselt"]], size=(16, 8.5)
)
plots.surface(
    results[2500.0, 1 / 12].field(heat.temperature),
    limits=(0.0, 1.0),
    contours=10,
    subdivisions=2,
    label="T",
    xlabel="x / d",
    ylabel="y / d",
    title="Temperature, Ra = 2500",
    ax=fig["temperature"],
)
plots.streamlines(
    grid_x,
    grid_y,
    flow_samples[..., 0],
    flow_samples[..., 1],
    density=1.2,
    label="speed [alpha/d]",
    xlabel="x / d",
    ylabel="y / d",
    title="Two counter-rotating rolls",
    ax=fig["streamlines"],
)
plots.line(
    np.r_[1200.0, curve],
    {
        "fit through the convecting states": np.r_[
            1.0, 1 + np.polyval(nusselt_fit, curve)
        ]
    },
    points={
        "OneSimulation": (
            [rayleigh for rayleigh, size in runs[:-1]],
            [table[key][0] for key in runs[:-1]],
        ),
        "Clever & Busse (1974)": ([2500.0], [1.47]),
    },
    vlines={"Ra_c = 1707.76": 1707.76},
    ylimits=(0.9, 1.6),
    xlabel="Rayleigh number",
    ylabel="Nusselt number",
    title="Onset of convection",
    legend="upper left",
    ax=fig["nusselt"],
)
plots.show()

# ## Going further
#
# Change the width of the layer in `rayleigh_benard_convection_onset_geometry.py` and the onset
# moves up along the neutral curve $Ra(a)$; raise the Rayleigh number further to follow
# $Nu(Ra)$ until the rolls become time dependent; or replace the upper plate by a free surface
# (free slip), for which the critical value drops to 1100.65.
