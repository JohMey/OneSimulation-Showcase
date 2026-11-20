"""OneSimulation example: reflection, transmission and absorption of a flat and a rough metal coating.
Requires the installed, activated OneSimulation runtime.

A plane wave of red light falls on a glass substrate with a thin gold coating. One period of the
structure, read from STEP files, is solved on its 2D cross-section with Floquet ports for both
polarizations. The flat coating is compared with the exact thin-film solution; the rough coating
scatters part of the light into many diffraction orders and absorbs more.
"""

# # Light scattered by a rough metal coating
#
# A glass substrate ($n=1.5$) carries a gold coating 50 nm thick ($\varepsilon=-11.7+1.26\,\mathrm i$ at
# $\lambda_0=633$ nm). Light falls on it from the air at normal incidence. On an optically flat coating
# the light is reflected, transmitted and absorbed as in a thin film; a rough top surface scatters part
# of it into oblique directions and raises the absorption.
#
# The rough surface is periodic over $W=4\,\mu$m, so the field obeys Bloch conditions and leaves
# through a finite number of diffraction orders $\sin\theta_m=m\lambda_0/(nW)$. OneSimulation solves
# one period on its cross-section, for $E_z$ (s polarization, electric field along the grooves) and
# $H_z$ (p polarization); `FloquetPort` feeds the specular order and absorbs all orders, and
# `diffraction_efficiencies` returns the power of every order:
#
# | boundary | condition |
# |---|---|
# | `y_max` (air) | Floquet port, plane wave fed at normal incidence |
# | `y_min` (glass) | Floquet port |
# | `x_min` ↔ `x_max` | Bloch periodic, $k_x=0$ |
#
# The cell, the glass and the two coatings come from `rough_surface_scattering_floquet.step` and the
# files `..._glass.step`, `..._flat.step` and `..._rough.step`, drawn in micrometres by
# `rough_surface_scattering_floquet_geometry.py` with the CAD objects of OneSimulation; the rough
# top surface is a periodic sum of eight cosines with random phases, 15 nm rms.
#
# **Reference.** The flat coating is a single absorbing film between air and glass; its reflectance
# and transmittance follow from the Airy formula of the film, computed here by the coherent thin-film
# solution `CoherentOpticalStack` (Born and Wolf, *Principles of Optics*, 7th ed., §13.4). For the
# rough coating there is no closed form: the computed orders and the power absorbed in the gold must
# conserve energy, $R+T+A=1$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import (
    CoherentOpticalStack,
    IsotropicElectromagneticMaterial,
    OpticalLayer,
    RegionwiseElectromagneticMaterial,
)
from onesim.meshing import PeriodicPair, TriangularMesher, scaled
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    BlochPeriodicBoundary,
    FloquetPort,
    FrequencyDomainMaxwell,
    diffraction_efficiencies,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

light = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)

# ## Geometry and mesh
#
# `section` cuts the plates at mid-depth. Quadratic triangles of 40 nm in air and glass and 6 nm in
# the coating, which is thinner than the skin depth of gold (about 30 nm) only by a factor of two;
# `scaled` turns the micrometres of the drawing into metres. The periodic sides get matching nodes.

plates = {
    name: load(f"rough_surface_scattering_floquet{suffix}.step")
    for name, suffix in (
        ("cell", ""),
        ("glass", "_glass"),
        ("flat", "_flat"),
        ("rough", "_rough"),
    )
}
cut = lambda part: part.section((0, 0, 1), origin=(0, 0, 0.025))
cell = cut(plates["cell"])
sides = cell.boundaries
meshes = {
    name: scaled(
        TriangularMesher(
            target_size=0.04,
            embedded_regions=(
                cut(plates["glass"]).named("glass"),
                cut(plates[name]).named("gold"),
            ),
            region_sizes={"gold": 0.006},
            hard_sizes={"x_min": 0.006, "x_max": 0.006},
            grading=0.3,
        ).generate(cell),
        "um",
    )
    for name in ("flat", "rough")
}

fig = plots.figure(1, 2, size=(16, 5))
plots.geometry(
    [plates["glass"].named("glass"), plates["rough"].named("rough gold coating")],
    camera=plots.Camera(elevation=60, azimuth=-80),
    title="One period: glass and rough coating (STEP)",
    length_unit="um",
    drawing_unit="um",
    ax=fig[0],
)
plots.mesh(
    meshes["rough"],
    groups=("all",),
    legend=False,
    length_unit="um",
    title=f"Rough coating: {meshes['rough'].cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# One solve per coating and polarization at $\lambda_0=633$ nm; the orders on the air side give the
# reflectance, those on the glass side the transmittance.

wavelength, gold, n_glass = 633e-9, -11.7 + 1.26j, 1.5
material = RegionwiseElectromagneticMaterial(
    {
        "gold": IsotropicElectromagneticMaterial(relative_permittivity=gold),
        "glass": IsotropicElectromagneticMaterial(relative_permittivity=n_glass**2),
    },
    default=IsotropicElectromagneticMaterial(),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=ComplexSparseDirect(),
)
solve = lambda coating, polarization: (
    lambda m: (
        Simulation(
            problem=Problem(
                name="Coated glass",
                geometry=cell,
                physics=(m,),
                boundary_conditions=(
                    BlochPeriodicBoundary(
                        physics=m,
                        pair=PeriodicPair(sides.x_min, sides.x_max),
                        wavevector=(0.0, 0.0, 0.0),
                    ),
                    FloquetPort(
                        physics=m, boundary=sides.y_max, incident_amplitude=1.0
                    ),
                    FloquetPort(physics=m, boundary=sides.y_min),
                ),
            ),
            mesh=meshes[coating],
            numerics=numerics,
        ).solve_stationary(),
        m,
    )
)(
    FrequencyDomainMaxwell(
        domain=cell.domain(),
        frequency=light / wavelength,
        material=material,
        polarization=polarization,
        phasor_convention="exp(-i*omega*t)",
        amplitude_convention="peak",
    )
)
solved = {
    (coating, polarization): solve(coating, polarization)
    for coating in ("flat", "rough")
    for polarization in ("Ez", "Hz")
}
runs = {key: diffraction_efficiencies(*value) for key, value in solved.items()}
impedance = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)
incident = {"Ez": 4e-6 / (2 * impedance), "Hz": 4e-6 * impedance / 2}
absorbed = {
    key: result.field(m.absorbed_power_density).integrate().value / incident[key[1]]
    for key, (result, m) in solved.items()
}

# ## Results
#
# Reflectance $R$ (all orders in the air), transmittance $T$ (all orders in the glass), absorptance $A$
# from the power dissipated in the gold (per period, relative to the incident power $W|E_0|^2/2\eta_0$
# or $W\eta_0|H_0|^2/2$), and the energy balance $R+T+A-1$; the diffuse part is the power outside the
# specular orders.

film = CoherentOpticalStack(
    layers=(OpticalLayer(thickness=50e-9, index=np.sqrt(gold)),)
)
exact = {
    "Ez": film(wavelength, 1.0, 1.0, n_glass, polarization="s"),
    "Hz": film(wavelength, 1.0, 1.0, n_glass, polarization="p"),
}
print(
    "coating  polarization      R         T         A      R+T+A-1   diffuse R   diffuse T   orders   thin-film solution R / T / A"
)
for (coating, polarization), orders in runs.items():
    R = sum(v for (port, order), v in orders.items() if port == 0)
    T = sum(v for (port, order), v in orders.items() if port == 1)
    ref = exact[polarization]
    print(
        f"{coating:7s}  {polarization:12s} {R:9.5f} {T:9.5f} {absorbed[(coating, polarization)]:9.5f} {R + T + absorbed[(coating, polarization)] - 1:+9.1e} {R - orders.get((0, 0), 0.0):11.2e} {T - orders.get((1, 0), 0.0):11.2e} {len(orders):8d}   "
        + (
            f"{ref.reflected:.5f} / {ref.transmitted:.5f} / {1 - ref.reflected - ref.transmitted:.5f}"
            if coating == "flat"
            else ""
        )
    )
for polarization in ("Ez", "Hz"):
    flat = runs[("flat", polarization)]
    print(
        f"flat coating, {polarization}: |R - R_film| = {abs(sum(v for (p, _), v in flat.items() if p == 0) - exact[polarization].reflected):.1e}, "
        f"|T - T_film| = {abs(sum(v for (p, _), v in flat.items() if p == 1) - exact[polarization].transmitted):.1e}"
    )

# ## Figure
#
# The power reflected into each diffraction order by the rough coating, against the angle of the
# order in the air, for both polarizations (the specular order is off the scale).

angle = lambda order: np.degrees(np.arcsin(np.clip(order * wavelength / 4e-6, -1, 1)))
fig = plots.figure(1, 2, size=(16, 5.5))
for panel, polarization in enumerate(("Ez", "Hz")):
    orders = {
        o: v for (p, o), v in runs[("rough", polarization)].items() if p == 0 and o != 0
    }
    plots.line(
        points={
            "rough coating, reflected orders": (
                angle(np.array(list(orders))),
                np.array(list(orders.values())),
            )
        },
        scale="logy",
        ylimits=(1e-5, 1e-1),
        xlabel="angle of the order in air [°]",
        ylabel="power in the order",
        title=f"Diffuse reflection, {'s' if polarization == 'Ez' else 'p'} polarization",
        legend="upper right",
        ax=fig[panel],
    )
plots.show()

# ## Going further
#
# Raise the roughness in `rough_surface_scattering_floquet_geometry.py` or shorten its correlation
# length: the diffuse part grows and, for p polarization, rough gold couples light into surface
# plasmons, which shows as a dip in the reflectance. A nonzero Bloch wavevector gives oblique
# incidence.
