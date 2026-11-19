"""OneSimulation example: supersonic expansion fan at a convex corner (Prandtl-Meyer).
Requires the installed, activated OneSimulation runtime.

Gas at Mach 2 flows along a wall that turns away from the stream by 10 degrees. The gas region is
read from a STEP file and cut to its cross-section; the compressible Euler equations are solved
with a second-order finite-volume scheme (HLLC flux, MUSCL reconstruction, Barth-Jespersen
limiter) driven to steady state by local pseudo-time stepping. The fan is compared with the exact
Prandtl-Meyer solution on three meshes.
"""

# # Supersonic expansion fan at a convex corner
#
# When a supersonic stream meets a wall that turns away from it, the flow cannot follow the corner
# at once. It turns gradually through a centred fan of Mach waves that spread out from the corner:
# the Mach number rises and the pressure falls isentropically across the fan. The flow is described
# by the Prandtl-Meyer function
#
# $$\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}\arctan\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}-\arctan\sqrt{M^2-1},$$
#
# with $\nu(M_2)=\nu(M_1)+\theta$ for a turn by the angle $\theta$. The fan starts on the Mach line
# $\mu_1=\arcsin(1/M_1)$ of the oncoming flow and ends on the line $\mu_2-\theta$; inside it, the
# Mach number on each ray from the corner follows from $\mu(M)=\phi+\nu(M)-\nu(M_1)$.
#
# The gas region comes from the CAD file `prandtl_meyer_expansion_fan.step`, written by
# `prandtl_meyer_expansion_fan_geometry.py`: a wall that turns by $\theta=10°$ at $x=0.5$ m. The model is
# its cross-section. The gas ($\gamma=1.4$) arrives at $M_1=2$; density, pressure and velocity are
# given relative to the free stream, which has the density 1.4, the pressure 1 and the sound speed 1.
#
# $$\frac{\partial}{\partial t}\begin{pmatrix}\rho\\\rho\mathbf u\\E\end{pmatrix}+\nabla\cdot
#   \begin{pmatrix}\rho\mathbf u\\\rho\mathbf u\otimes\mathbf u+p\mathbf I\\(E+p)\mathbf u\end{pmatrix}=0,\qquad
#   p=(\gamma-1)\Big(E-\tfrac12\rho|\mathbf u|^2\Big)$$
#
# | side | condition |
# |---|---|
# | `inlet` | supersonic inflow: $\rho$, $\mathbf u$, $p$ of the free stream |
# | `outlet` | supersonic outflow (all characteristics leave) |
# | `wall`, `ramp`, `upper` | slip wall |
#
# **Reference.** The exact Prandtl-Meyer solution (J. D. Anderson, *Modern Compressible Flow*, 3rd
# ed., McGraw-Hill 2003, ch. 4): for $M_1=2$ and $\theta=10°$, $M_2=2.385$ and
# $p_2/p_1=0.5480$.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fluxes import HLLC
from onesim.numerics.fvm import (
    MUSCL,
    BarthJespersenLimiter,
    FiniteVolume,
    LocalTimeStepping,
)
from onesim.numerics.time import ForwardEuler, TimeStepLimits
from onesim.physics.hydrodynamics import (
    CompressibleEuler,
    IdealGasEquationOfState,
    SlipWall,
    SupersonicInflow,
    SupersonicOutflow,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## Geometry and mesh
#
# `section` cuts the slab at half thickness. Triangles of 5, 4 and 3 cm for the mesh study.

part = load("prandtl_meyer_expansion_fan.step")
region = part.section((0, 0, 1), origin=(0, 0, part.size[2] / 2))
faces = region.boundaries
meshes = {
    size: TriangularMesher(target_size=size).generate(region)
    for size in (0.05, 0.04, 0.03)
}

fig = plots.figure(1, 2, size=(15, 6))
plots.geometry(
    part,
    selection={"inlet": part.boundaries.inlet, "turned wall": part.boundaries.ramp},
    labels=False,
    camera=plots.Camera(elevation=60, azimuth=-70),
    title="Gas region above the convex corner",
    ax=fig[0],
)
plots.mesh(
    meshes[0.03],
    groups=("all",),
    legend=False,
    title=f"{meshes[0.03].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# An ideal gas; the free stream also fills the region at the start. Local pseudo-time
# stepping (each cell at its own stable step, CFL 0.9) drives the solution to steady state in 1500
# iterations.

gamma, mach, pressure_in, density_in = 1.4, 2.0, 1.0, 1.4
speed = mach * np.sqrt(gamma * pressure_in / density_in)
energy = pressure_in / (gamma - 1) + 0.5 * density_in * speed**2
model = CompressibleEuler(
    domain=region.domain(), equation_of_state=IdealGasEquationOfState(gamma=gamma)
)
problem = Problem(
    name="Prandtl-Meyer expansion",
    geometry=region,
    physics=(model,),
    boundary_conditions=(
        SupersonicInflow(
            boundary=faces.inlet,
            physics=model,
            density=density_in,
            velocity=(speed, 0.0),
            pressure=pressure_in,
        ),
        SupersonicOutflow(boundary=faces.outlet, physics=model),
        SlipWall(boundary=faces.wall | faces.ramp | faces.upper, physics=model),
    ),
    initial_conditions=(
        InitialValue(model.density, density_in),
        InitialValue(model.momentum, (density_in * speed, 0.0)),
        InitialValue(model.total_energy, energy),
    ),
)
numerics = Numerics(
    spatial=FiniteVolume(
        convection=HLLC(),
        reconstruction=MUSCL(),
        limiter=BarthJespersenLimiter(),
        stability=LocalTimeStepping(cfl=0.9),
    ),
    time_integrator=ForwardEuler(),
    step_limits=TimeStepLimits(stability_safety_factor=1.0),
)
results = {
    size: Simulation(problem=problem, mesh=mesh, numerics=numerics).run(
        start_time=0, end_time=1500, time_step=1.0
    )
    for size, mesh in meshes.items()
}

# ## Results
#
# The exact solution on rays from the corner, and the state behind the fan sampled along the turned
# wall.

turn = np.radians(10)
nu = lambda m: np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(
    np.sqrt((gamma - 1) / (gamma + 1) * (m**2 - 1))
) - np.arctan(np.sqrt(m**2 - 1))
behind = brentq(lambda m: nu(m) - nu(mach) - turn, mach, 5.0)
isentropic = lambda m: (
    (1 + 0.5 * (gamma - 1) * mach**2) / (1 + 0.5 * (gamma - 1) * m**2)
) ** (gamma / (gamma - 1))
head, tail = np.arcsin(1 / mach), np.arcsin(1 / behind) - turn
rays = np.linspace(tail - 0.08, head + 0.08, 121)
exact_mach = np.array(
    [
        (
            mach
            if phi >= head
            else (
                behind
                if phi <= tail
                else brentq(
                    lambda m: np.arcsin(1 / m) - (nu(m) - nu(mach)) - phi, mach, behind
                )
            )
        )
        for phi in rays
    ]
)
arc = np.column_stack((0.5 + 0.7 * np.cos(rays), 0.7 * np.sin(rays)))
along = (
    np.array([0.5, 0.0])
    + np.outer(np.linspace(0.45, 0.85, 20), (np.cos(turn), -np.sin(turn)))
    + 0.04 * np.array([np.sin(turn), np.cos(turn)])
)
fields = {
    size: (result.field(model.mach_number), result.field(model.pressure))
    for size, result in results.items()
}

print(
    f"exact: M2 = {behind:.4f}, p2/p1 = {isentropic(behind):.4f}, fan from {np.degrees(head):.2f}° to {np.degrees(tail):.2f}°"
)
print(
    " mesh [cm]  triangles   M behind the fan   deviation   p2/p1    deviation   mean |M - exact| on the arc r = 0.7 m"
)
for size, (mach_field, pressure_field) in fields.items():
    m2, p2 = (
        np.mean(mach_field.evaluate(along)),
        np.mean(pressure_field.evaluate(along)) / pressure_in,
    )
    error = np.mean(np.abs(np.ravel(mach_field.evaluate(arc)) - exact_mach))
    print(
        f"{100 * size:8.0f} {meshes[size].cell_count:10d} {m2:14.4f} {100 * (m2 / behind - 1):+11.2f} % {p2:8.4f} {100 * (p2 / isentropic(behind) - 1):+8.2f} %"
        f" {error:20.4f}"
    )

# ## Figure
#
# The Mach number in the gas region and on the arc of radius 0.7 m around the corner, against the
# exact centred fan.

fig = plots.figure(1, 2, size=(16, 6))
plots.surface(
    fields[0.03][0],
    limits=(1.95, 2.45),
    contours=10,
    subdivisions=1,
    label="Mach number",
    title="Mach number: the centred fan spreads from the corner",
    ax=fig[0],
)
plots.line(
    np.degrees(rays),
    {
        f"simulation, {100 * size:.0f} cm": np.ravel(fields[size][0].evaluate(arc))
        for size in meshes
    },
    reference={"exact Prandtl-Meyer fan": exact_mach},
    xlabel="ray angle from the incoming flow direction [°]",
    ylabel="Mach number",
    title="Mach number on the arc r = 0.7 m around the corner",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Turn the wall the other way in `prandtl_meyer_expansion_fan_geometry.py` (a compression corner) and
# write the STEP file again: the flow then turns through an oblique shock, which the
# theta-beta-Mach relation predicts. Raising the turn angle until the pressure behind the fan
# vanishes gives the largest possible turn, about 104° at Mach 2.
