"""OneSimulation example: bulk temperature in a heated square duct (3D laminar flow and heat transfer).
Requires the installed, activated OneSimulation runtime.

Water flows through a square duct whose walls take up a uniform heat flux. The bulk (mixing-cup)
temperature, the quantity every heat transfer coefficient refers to, is evaluated along the duct
with the integration operators of OneSimulation and compared with the energy balance; the area
mean and the wall temperature show why the distinction matters, and the fully developed Nusselt
number is compared with the tabulated value for the square duct.
"""

# # Bulk temperature in a heated square duct
#
# Water (density 1000 kg/m³, viscosity $10^{-3}$ Pa s, conductivity 0.6 W/(m K), heat capacity
# 4180 J/(kg K), $Pr=7$) flows through a square duct of side $a=2$ mm at $Re=20$ and enters at
# 293.15 K with a fully developed velocity profile. All four walls take up the uniform heat flux
# $q=2000$ W/m². The flow is solved with the steady Navier-Stokes equations and carries the heat,
#
# $$\rho c_p\,\mathbf u\cdot\nabla T=\nabla\cdot(k\nabla T).$$
#
# A heat transfer coefficient is always defined with the **bulk temperature**, the temperature the
# fluid would reach if the cross-section were collected and mixed,
#
# $$T_b(z)=\frac{\int_A u_z\,T\,dA}{\int_A u_z\,dA},$$
#
# not with the plain area mean: the fast core carries more enthalpy than the slow fluid near the
# hot walls.
#
# One quarter of the duct comes from the CAD file `bulk_temperature_square_duct.step` (faces
# `z_min` inlet, `z_max` outlet, `x_max` and `y_max` walls, `x_min` and `y_min` symmetry planes);
# the script `bulk_temperature_square_duct_geometry.py` builds it with the CAD objects of
# OneSimulation.
#
# | face | flow | heat |
# |---|---|---|
# | `z_min` | fully developed profile, $Re=20$ | $T=293.15$ K |
# | `z_max` | traction free | no conduction |
# | `x_max`, `y_max` | no slip | heat flux $q$ |
# | `x_min`, `y_min` | free slip | no flux |
#
# **Reference.** The energy balance fixes the bulk temperature exactly at every station,
# $T_b(z)=T_\mathrm{in}+q\,P\,z/(\dot m c_p)$ with the perimeter $P=4a$. Far from the inlet the
# temperature profile no longer changes shape; for a square duct with peripherally uniform wall
# flux the Nusselt number based on the mean wall temperature is $Nu_{H2}=3.091$ (Shah and London,
# *Laminar Flow Forced Convection in Ducts*, 1978, Table 42).

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid, NewtonianFluid
from onesim.meshing import Geometric, Piecewise, StructuredHexahedralMesher, Uniform
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
    FreeSlipLaw,
    IncompressibleNavierStokes,
    NoSlipWall,
    WallShear,
    fully_developed_inflow,
)
from onesim.results.derived import derived_field
from onesim.results.integration import integrate_plane, mean_value
from onesim.results.sampling import field_function
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# A hexahedral mesh, finer towards the heated walls and towards the inlet.

duct = load("bulk_temperature_square_duct.step")
faces = duct.boundaries
half, _, length = duct.size
side = 2 * half
mesh = StructuredHexahedralMesher(
    cells=(8, 8, 40),
    grading=(
        Geometric(8, ratio=3.0, toward="end"),
        Geometric(8, ratio=3.0, toward="end"),
        Piecewise(
            (
                (0.2 * length, Geometric(12, ratio=4.0, toward="start")),
                (0.8 * length, Uniform(28)),
            )
        ),
    ),
).generate(duct)

view = plots.Camera(elevation=25, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 5.5))
plots.geometry(
    duct,
    selection={
        "inlet": faces.z_min,
        "heated walls": faces.x_max | faces.y_max,
        "symmetry": faces.x_min | faces.y_min,
    },
    labels=False,
    camera=view,
    title="Quarter of the duct",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} hexahedra",
    ax=fig[1],
)
plots.show()

# ## Flow and heat transfer
#
# The solved velocity drives the heat transport one way (constant properties).

water = NewtonianFluid(density=1000.0, dynamic_viscosity=1.0e-3)
conductivity, capacity, flux, inlet = 0.6, 4180.0, 2000.0, 293.15
mean_speed = 20.0 * water.dynamic_viscosity / (water.density * side)
walls, symmetry = faces.x_max | faces.y_max, faces.x_min | faces.y_min
flow = IncompressibleNavierStokes(domain=duct.volume, material=water)
velocity = (
    Simulation(
        problem=Problem(
            name="Duct flow",
            geometry=duct,
            physics=(flow,),
            boundary_conditions=(
                FixedVelocity(
                    boundary=faces.z_min,
                    physics=flow,
                    value=fully_developed_inflow(
                        mesh, faces.z_min, mean_velocity=mean_speed, no_slip=walls
                    ),
                ),
                FlowTraction(
                    boundary=faces.z_max,
                    physics=flow,
                    value=lambda x, y, z: np.zeros((len(x), 3)),
                ),
                NoSlipWall(boundary=walls, physics=flow),
                WallShear(boundary=symmetry, physics=flow, law=FreeSlipLaw()),
            ),
            initial_conditions=(InitialValue(flow.velocity, (0.0, 0.0, 0.0)),),
        ),
        mesh=mesh,
        numerics=Numerics(
            spatial=FiniteElement(element=TaylorHood()),
            initial_guess={flow.velocity: (0.0, 0.0, 0.0)},
            nonlinear_solver=Newton(
                linear_solver=SparseDirect(),
                absolute_tolerance=1e-14,
                relative_tolerance=1e-10,
            ),
        ),
    )
    .solve_stationary()
    .field(flow.velocity)
)
heat = ConvectiveHeatTransfer(
    domain=duct.volume,
    velocity=field_function(velocity),
    material=IsotropicSolid(
        density=water.density,
        specific_heat_capacity=capacity,
        thermal_conductivity=conductivity,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=0.0,
    ),
)
temperature = (
    Simulation(
        problem=Problem(
            name="Duct heat transfer",
            geometry=duct,
            physics=(heat,),
            boundary_conditions=(
                FixedTemperature(boundary=faces.z_min, physics=heat, value=inlet),
                HeatFlux(boundary=walls, physics=heat, value=-flux),
                HeatFlux(boundary=faces.z_max | symmetry, physics=heat, value=0.0),
            ),
        ),
        mesh=mesh,
        numerics=Numerics(
            spatial=FiniteElement(
                element=Lagrange(order=2), convection=GalerkinAdvection()
            ),
            linear_solver=SparseDirect(),
        ),
    )
    .solve_stationary()
    .field(heat.temperature)
)

# ## Bulk temperature
#
# `derived_field` forms $u_zT$, `integrate_plane` integrates it and the axial velocity over a cut
# across the duct, and their ratio is the bulk temperature. `mean_value` on the same cut gives the
# area mean, and on the wall line the mean wall temperature.

stations = np.linspace(1e-3, length - 1e-3, 29)
cut = lambda z: Plane(origin=(0, 0, z), normal=(0, 0, 1))
carried = derived_field(
    velocity,
    lambda u, points: u[:, 2] * np.ravel(temperature.evaluate(points)),
    name="enthalpy_flux",
    unit="K*m/s",
)
bulk = np.array(
    [
        integrate_plane(carried, cut(z)).value
        / integrate_plane(velocity, cut(z), normal_component=True).value
        for z in stations
    ]
)
area_mean = np.array([mean_value(temperature, plane=cut(z)) for z in stations])
s = np.linspace(0.0, half, 21)
wall = np.array(
    [
        np.mean(
            np.r_[
                temperature.evaluate(
                    np.column_stack((np.full_like(s, half), s, np.full_like(s, z)))
                ),
                temperature.evaluate(
                    np.column_stack((s, np.full_like(s, half), np.full_like(s, z)))
                ),
            ]
        )
        for z in stations
    ]
)
balance = inlet + flux * 4 * side * stations / (
    water.density * mean_speed * side**2 * capacity
)
nusselt = flux * side / (conductivity * (wall - bulk))
developed = stations > 0.6 * length
print(
    f"Re = 20, Pr = {water.dynamic_viscosity * capacity / conductivity:.2f}; flow rate through the quarter "
    f"{integrate_plane(velocity, cut(length / 2), normal_component=True).value / (mean_speed * half**2):.5f} x the prescribed"
)
print(
    "z [mm]   bulk T [K]   energy balance [K]   deviation [mK]   area mean [K]   mean wall T [K]"
)
for z, tb, te, ta, tw in list(zip(stations, bulk, balance, area_mean, wall))[::4]:
    print(
        f"{1e3 * z:6.1f} {tb:12.4f} {te:18.4f} {1e3 * (tb - te):14.2f} {ta:15.4f} {tw:15.4f}"
    )
print(
    f"largest deviation of the bulk temperature from the energy balance: {1e3 * np.abs(bulk - balance).max():.2f} mK "
    f"of {balance[-1] - inlet:.3f} K rise"
)
print(
    f"fully developed Nusselt number (last 40 %): {nusselt[developed].mean():.4f}, tabulated Nu_H2 = 3.091 "
    f"({100 * (nusselt[developed].mean() / 3.091 - 1):+.2f} %)"
)
print(
    f"using the area mean instead of the bulk temperature would give Nu = "
    f"{(flux * side / (conductivity * (wall - area_mean)))[developed].mean():.3f}"
)

# ## Figure
#
# The temperature on the outlet cross-section, the three temperatures along the duct and the local
# Nusselt number.

fig = plots.figure(
    layout=[["outlet", "temperatures"], ["outlet", "nusselt"]], size=(15, 9)
)
plots.section(
    temperature,
    plane=cut(length - 1e-6),
    contours=10,
    subdivisions=3,
    symmetry={"planes": [((1, 0, 0), (0, 0, 0)), ((0, 1, 0), (0, 0, 0))]},
    label="T [K]",
    xlabel="x [m]",
    ylabel="y [m]",
    title="Temperature at the outlet",
    ax=fig["outlet"],
)
plots.line(
    1e3 * stations,
    {"mean wall temperature": wall, "area mean": area_mean},
    points={"bulk temperature": (1e3 * stations, bulk)},
    reference={"energy balance": (1e3 * stations, balance)},
    xlabel="z along the duct [mm]",
    ylabel="temperature [K]",
    title="Bulk, area mean and wall temperature",
    legend="upper left",
    ax=fig["temperatures"],
)
plots.line(
    1e3 * stations,
    {"simulation": nusselt},
    hlines={"Nu_H2 = 3.091": 3.091},
    ylimits=(2.5, 8),
    xlabel="z along the duct [mm]",
    ylabel="local Nusselt number",
    title="Thermal development",
    ax=fig["nusselt"],
)
plots.show()

# ## Going further
#
# Hold the walls at a fixed temperature instead and compare with $Nu_T=2.976$; make the duct
# rectangular in `bulk_temperature_square_duct_geometry.py` and follow the Nusselt number with the
# aspect ratio; or heat only one wall, as in a cold plate.
