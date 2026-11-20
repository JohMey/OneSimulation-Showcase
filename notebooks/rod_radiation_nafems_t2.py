"""OneSimulation example: steady conduction in a rod with a radiating end (NAFEMS T2).
Requires the installed, activated OneSimulation runtime.

One end of a steel rod is held at 1000 K, the other end radiates to surroundings at 300 K. The rod
is read from a STEP file; Newton's method solves the nonlinear radiation condition, and the end
temperature is compared with the NAFEMS target and with the exact root of the energy balance.
"""

# # Conduction in a rod with a radiating end (NAFEMS T2)
#
# The NAFEMS thermal benchmark T2 tests a nonlinear boundary condition in the simplest possible
# setting: one-dimensional steady conduction along a rod of length $L=0.1$ m whose far end loses
# heat by grey-body radiation,
#
# $$k\,\frac{d^2T}{dx^2}=0,\qquad T(0)=T_0,\qquad -k\,\frac{dT}{dx}\Big|_{x=L}=\varepsilon\sigma\,\big(T_L^4-T_\infty^4\big).$$
#
# | boundary | condition |
# |---|---|
# | `hot_end`, $x=0$ | fixed temperature $T_0=1000$ K |
# | `radiating_end`, $x=L$ | radiation, $\varepsilon=0.98$, surroundings $T_\infty=300$ K |
# | `sides` | insulated |
#
# with $k=55.6$ W/(m K). The radiation law makes the problem nonlinear in the end temperature.
#
# The rod comes from the CAD file `rod_radiation_nafems_t2.step`; the script
# `rod_radiation_nafems_t2_geometry.py` builds it with the CAD objects of OneSimulation. The sides
# are insulated, so the field is one-dimensional along the rod.
#
# **Reference.** NAFEMS, *The Standard NAFEMS Benchmarks* (TNSB, Rev. 3, 1990), test T2: the
# temperature of the radiating end is **927 K**. The profile is linear, so the end temperature is
# also the root of the energy balance $k\,(T_0-T_L)/L=\varepsilon\sigma(T_L^4-T_\infty^4)$, which
# the notebook evaluates for the benchmark and for two smaller emissivities.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.heat_transfer import (
    FixedTemperature,
    HeatConduction,
    HeatFlux,
    RadiativeHeatFlux,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## Geometry and mesh
#
# `load` returns the rod with its named faces. Ten quadratic hexahedra along the axis are more than
# enough for a linear profile.

rod = load("rod_radiation_nafems_t2.step")
faces = rod.boundaries
length, width, _ = rod.size
mesh = StructuredHexahedralMesher(cells=(10, 1, 1)).generate(rod)

fig = plots.figure(1, 2, size=(14, 4))
plots.geometry(
    rod,
    selection={
        "hot end, 1000 K": faces.hot_end,
        "radiating end": faces.radiating_end,
        "sides": faces.sides,
    },
    labels=False,
    camera=plots.Camera(elevation=25, azimuth=-60),
    title="Rod with its named faces",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=plots.Camera(elevation=25, azimuth=-60),
    title=f"{mesh.cell_count} quadratic hexahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `RadiativeHeatFlux` imposes the full Stefan-Boltzmann law $\varepsilon\sigma(T^4-T_\infty^4)$;
# Newton's method solves the nonlinear system from a uniform first guess of 600 K. The benchmark
# emissivity is solved together with two smaller ones.

hot, surroundings, sigma = 1000.0, 300.0, 5.670374419e-8
steel = IsotropicSolid(
    density=7850.0,
    specific_heat_capacity=460.0,
    thermal_conductivity=55.6,
    youngs_modulus=2.1e11,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=1.2e-5,
)
model = HeatConduction(domain=rod.volume, material=steel, heating=0.0)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=SparseDirect(),
    nonlinear_solver=Newton(relative_tolerance=1e-10),
    initial_guess={model.temperature: 600.0},
)
fields = {
    emissivity: Simulation(
        problem=Problem(
            name="NAFEMS T2",
            geometry=rod,
            physics=(model,),
            boundary_conditions=(
                FixedTemperature(boundary=faces.hot_end, physics=model, value=hot),
                RadiativeHeatFlux(
                    boundary=faces.radiating_end,
                    physics=model,
                    emissivity=emissivity,
                    ambient_temperature=surroundings,
                ),
                HeatFlux(boundary=faces.sides, physics=model, value=0.0),
            ),
        ),
        mesh=mesh,
        numerics=numerics,
    )
    .solve_stationary()
    .field(model.temperature)
    for emissivity in (0.98, 0.5, 0.2)
}

# ## Results
#
# The end temperature against the root of the energy balance, and the radiated heat flux against
# the conducted one.

exact_end = lambda e: brentq(
    lambda T: steel.thermal_conductivity * (hot - T) / length
    - e * sigma * (T**4 - surroundings**4),
    surroundings,
    hot,
)
centre = lambda x: np.column_stack(
    (x, np.full_like(x, width / 2), np.full_like(x, width / 2))
)
print(
    "emissivity   T(L) simulation [K]   T(L) exact root [K]   deviation [K]   radiated flux [W/m²]   conducted flux [W/m²]"
)
for emissivity, field in fields.items():
    end = float(field.evaluate(centre(np.array([length])))[0])
    print(
        f"{emissivity:10.2f} {end:21.4f} {exact_end(emissivity):21.4f} {end - exact_end(emissivity):+15.2e} "
        f"{field.boundary_flux(faces.radiating_end) / width**2:22.1f} {-field.boundary_flux(faces.hot_end) / width**2:22.1f}"
    )
print(
    f"NAFEMS T2 target 927 K, simulation {float(fields[0.98].evaluate(centre(np.array([length])))[0]):.2f} K"
)

# ## Figure
#
# The temperature along the rod for the three emissivities with the exact linear profiles, and the
# energy balance at the radiating end: conduction and radiation cross at the end temperature.

x = np.linspace(0, length, 101)
T = np.linspace(800.0, 1000.0, 201)
fig = plots.figure(1, 2, size=(15, 5.5))
plots.line(
    y={
        f"ε = {e}": (1e3 * x, field.evaluate(centre(x)).ravel())
        for e, field in fields.items()
    },
    reference={
        f"ε = {e}": (1e3 * x, hot + (exact_end(e) - hot) * x / length) for e in fields
    },
    points={"NAFEMS target 927 K": ([1e3 * length], [927.0])},
    xlabel="position along the rod [mm]",
    ylabel="T [K]",
    title="Temperature along the rod (dashed: exact)",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    y={
        "conducted, k (T0 - T) / L": (
            T,
            1e-3 * steel.thermal_conductivity * (hot - T) / length,
        ),
        "radiated, ε σ (T⁴ - T∞⁴), ε = 0.98": (
            T,
            1e-3 * 0.98 * sigma * (T**4 - surroundings**4),
        ),
    },
    vlines={
        "simulated end temperature": float(
            fields[0.98].evaluate(centre(np.array([length])))[0]
        )
    },
    xlabel="end temperature T(L) [K]",
    ylabel="heat flux [kW/m²]",
    title="Energy balance at the radiating end",
    legend="upper left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Make the conductivity temperature dependent (a `TabulatedLaw` in `IsotropicSolid`): the profile
# bends and only the simulation still gives the end temperature. Or lengthen the rod in
# `rod_radiation_nafems_t2_geometry.py` and write the STEP file again, and let the sides radiate as
# well, which turns the rod into a radiating fin.
