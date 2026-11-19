"""OneSimulation example: reflection, transmission and diffraction of a plasmonic wire grating.
Requires the installed, activated OneSimulation runtime.

A grating of silver nanowires above a glass substrate is lit by green light at every angle of
incidence, in both polarizations. One period, read from STEP files, is solved as a two-dimensional
section with Floquet ports and silver as a Drude metal; the specular and first-order reflection and
transmission and the power absorbed in the wires are reported, and their sum is checked against the
incident power.
"""

# # Plasmonic wire grating
#
# Silver wires of 40 nm radius, 400 nm apart and 20 nm above a glass substrate ($n=1.5$), form a
# grating for light of 550 nm. Besides the specular reflection and the refraction into the glass,
# the period lets the first diffraction order $m=-1$ propagate in the glass at every angle and in
# the air beyond $\sin\theta>\lambda/p-1$ (22 degrees). In s polarization (electric field along the
# wires) the silver acts as a wire medium that reflects and diffracts strongly; in p polarization
# (magnetic field along the wires) the wires are polarized across and, far below their plasmon
# resonance in the ultraviolet, let most of the light through.
#
# Silver is a Drude metal, $\varepsilon_m=\varepsilon_\infty-\omega_p^2/(\omega^2+\mathrm i\gamma\omega)$
# with $\varepsilon_\infty=3.7$, $\omega_p=1.37\times10^{16}$ rad/s and $\gamma=3.2\times10^{13}$ rad/s.
# The section solves $E_z$ (s) or $H_z$ (p); Bloch-periodic sides carry
# $k_x=k_0\sin\theta$, Floquet ports above and below feed the plane wave and take every order out.
#
# | boundary | condition |
# |---|---|
# | `x_min`, `x_max` | Bloch periodic, $k_x=k_0\sin\theta$ |
# | `y_max` (air) | Floquet port, incident wave |
# | `y_min` (glass) | Floquet port |
#
# **Reference** (M. Born and E. Wolf, *Principles of Optics*, 7th ed., §8.6; Poynting's theorem). The
# grating equation $k_{x,m}=k_0\sin\theta+2\pi m/p$ fixes the directions of the orders; their
# efficiencies have no closed form, so their accuracy is checked by a mesh study (the mesh halved
# everywhere, the wire mesh halved again). What the wires do not reflect or transmit they absorb, so
#
# $$\sum_m R_m+\sum_m T_m+A=1,\qquad A=\frac1{P_{inc}}\int_{\rm wire}\tfrac12\omega\,\varepsilon_0\,\mathrm{Im}\,\varepsilon_m|E|^2\,\mathrm dA,$$
#
# with $R_m$, $T_m$ from the port amplitudes and $A$ from the absorbed power density of the Drude
# metal. The Galerkin solution keeps this balance to rounding on any mesh, so it confirms that ports
# and dissipation are consistent rather than measuring the discretization error.
#
# The STEP parts are drawn in micrometres, the unit that suits wires of a few tens of nanometres;
# `scaled` brings the meshes to metres, so the model is solved in SI units and the plots show
# micrometres again.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import (
    DrudeElectromagneticMaterial,
    IsotropicElectromagneticMaterial,
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

wavelength, period = 550e-9, 0.4e-6
light = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
impedance = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)
frequency, k0 = light / wavelength, 2 * np.pi / wavelength
silver = DrudeElectromagneticMaterial(
    plasma_frequency=1.37e16, collision_rate=3.2e13, high_frequency_permittivity=3.7
)

# ## Geometry and mesh
#
# `load` reads the cell, the glass and the wire; `section` cuts them at half thickness. The
# periodic sides are meshed alike; quadratic triangles of 20 nm, and 4 nm in the wire, where the
# field decays within 25 nm of the surface; `scaled(..., "um")` turns the micrometres of the drawing
# into metres.

parts = {
    name: load(f"plasmonic_wire_grating{name}.step").section(
        (0, 0, 1), origin=(0, 0, 0.05)
    )
    for name in ("", "_glass", "_wire")
}
cell = parts[""]
sides = cell.boundaries
mesh = scaled(
    TriangularMesher(
        target_size=0.02,
        embedded_regions=(parts["_glass"].named("glass"), parts["_wire"].named("wire")),
        region_sizes={"wire": 0.004},
        periodic_pairs=(PeriodicPair(sides.x_min, sides.x_max),),
    ).generate(cell),
    "um",
)

fig = plots.figure(1, 2, size=(10, 8))
plots.geometry(
    cell,
    selection={
        "periodic sides": sides.x_min | sides.x_max,
        "port in air": sides.y_max,
        "port in glass": sides.y_min,
    },
    labels=False,
    title="One period (STEP)",
    length_unit="um",
    drawing_unit="um",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"{mesh.cell_count} triangles",
    length_unit="um",
    ax=fig[1],
)
plots.show()

# ## Model
#
# For every angle the Bloch wavevector is set and `diffraction_efficiencies` returns the power of
# every propagating order at both ports relative to the incident power: `(0, m)` reflected, `(1, m)`
# transmitted. The absorbed power is integrated over the cell (only the silver dissipates).

material = RegionwiseElectromagneticMaterial(
    {
        "glass": IsotropicElectromagneticMaterial(relative_permittivity=2.25),
        "wire": silver,
    },
    default=IsotropicElectromagneticMaterial(),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=ComplexSparseDirect(),
)
models = {
    polarization: FrequencyDomainMaxwell(
        domain=cell.domain(),
        frequency=frequency,
        material=material,
        polarization=polarization,
        phasor_convention="exp(-i*omega*t)",
        amplitude_convention="peak",
    )
    for polarization in ("Ez", "Hz")
}
solve = lambda polarization, angle, grid=mesh: Simulation(
    problem=Problem(
        name="wire grating",
        geometry=cell,
        physics=(models[polarization],),
        boundary_conditions=(
            BlochPeriodicBoundary(
                physics=models[polarization],
                pair=PeriodicPair(sides.x_min, sides.x_max),
                wavevector=(k0 * np.sin(np.radians(angle)), 0.0, 0.0),
            ),
            FloquetPort(
                physics=models[polarization],
                boundary=sides.y_max,
                incident_amplitude=1.0,
            ),
            FloquetPort(physics=models[polarization], boundary=sides.y_min),
        ),
    ),
    mesh=grid,
    numerics=numerics,
).solve_stationary()

angles = np.arange(0.0, 80.1, 2.5)
results = {
    (polarization, angle): solve(polarization, angle)
    for polarization in ("Ez", "Hz")
    for angle in angles
}
efficiencies = {
    key: diffraction_efficiencies(result, models[key[0]])
    for key, result in results.items()
}
incident = {"Ez": 0.5 / impedance * period, "Hz": 0.5 * impedance * period}
absorbed = {
    key: result.field(models[key[0]].absorbed_power_density).integrate().value
    / (incident[key[0]] * np.cos(np.radians(key[1])))
    for key, result in results.items()
}

# ## Results
#
# The orders present, their efficiencies at a few angles, and the power balance over all angles.

names = {"Ez": "s (E_z)", "Hz": "p (H_z)"}
print(
    f"silver at {1e9 * wavelength:.0f} nm: eps = {silver.relative_permittivity_at(frequency):.2f}"
)
for polarization in ("Ez", "Hz"):
    print(
        f"\n{names[polarization]}   angle   R_0      R_-1     T_0      T_-1     T_+1     A        sum"
    )
    for angle in (0.0, 20.0, 40.0, 60.0, 80.0):
        e = efficiencies[(polarization, angle)]
        print(
            f"           {angle:5.1f}   {e.get((0, 0), 0):.4f}   {e.get((0, -1), 0):.4f}   {e.get((1, 0), 0):.4f}   {e.get((1, -1), 0):.4f}   "
            f"{e.get((1, 1), 0):.4f}   {absorbed[(polarization, angle)]:.4f}   {sum(e.values()) + absorbed[(polarization, angle)]:.6f}"
        )
balance = {
    polarization: max(
        abs(
            sum(efficiencies[(polarization, a)].values())
            + absorbed[(polarization, a)]
            - 1
        )
        for a in angles
    )
    for polarization in ("Ez", "Hz")
}
symmetry = {
    polarization: abs(
        efficiencies[(polarization, 0.0)].get((1, -1), 0)
        - efficiencies[(polarization, 0.0)].get((1, 1), 0)
    )
    for polarization in ("Ez", "Hz")
}
print(
    f"\nlargest |sum R + sum T + A - 1| over all angles: s {balance['Ez']:.1e}, p {balance['Hz']:.1e}"
)
print(
    f"normal incidence |T_-1 - T_+1| (mirror symmetry): s {symmetry['Ez']:.1e}, p {symmetry['Hz']:.1e}"
)
print(
    f"first reflected order appears beyond arcsin(lambda/p - 1) = {np.degrees(np.arcsin(wavelength / period - 1)):.1f} deg"
)

# The mesh study: triangles of 10 nm and 2 nm in the wire, at three angles.

fine = scaled(
    TriangularMesher(
        target_size=0.01,
        embedded_regions=(parts["_glass"].named("glass"), parts["_wire"].named("wire")),
        region_sizes={"wire": 0.002},
        periodic_pairs=(PeriodicPair(sides.x_min, sides.x_max),),
    ).generate(cell),
    "um",
)
refined = {
    (polarization, angle): diffraction_efficiencies(
        solve(polarization, angle, fine), models[polarization]
    )
    for polarization in ("Ez", "Hz")
    for angle in (0.0, 40.0, 70.0)
}
print(
    f"\nmesh study ({mesh.cell_count} -> {fine.cell_count} triangles): largest change of any efficiency"
)
for (polarization, angle), values in refined.items():
    print(
        f"  {names[polarization]} at {angle:4.1f} deg: {max(abs(values[k] - efficiencies[(polarization, angle)].get(k, 0.0)) for k in values):.1e}"
    )

# ## Figure
#
# Specular and first-order reflection, transmission and absorption against the angle of incidence,
# and the magnetic field around the wires in p polarization at 40 degrees.

x, y = np.asarray(mesh.coordinates).T
field = np.abs(
    results[("Hz", 40.0)]
    .field(models["Hz"].magnetic_flux_density)
    .evaluate(np.column_stack((x, y)))[:, 2]
) / (VACUUM_PERMEABILITY * 1.0)
fig = plots.figure(1, 3, size=(24, 6.5))
for panel, polarization in zip(fig, ("Ez", "Hz")):
    plots.line(
        angles,
        {
            "R_0": [efficiencies[(polarization, a)].get((0, 0), 0) for a in angles],
            "R_-1": [efficiencies[(polarization, a)].get((0, -1), 0) for a in angles],
            "T_0": [efficiencies[(polarization, a)].get((1, 0), 0) for a in angles],
            "T_-1": [efficiencies[(polarization, a)].get((1, -1), 0) for a in angles],
            "A": [absorbed[(polarization, a)] for a in angles],
        },
        vlines={"R_-1 opens": np.degrees(np.arcsin(wavelength / period - 1))},
        xlimits=(0, 80),
        ylimits=(0, 1),
        xlabel="angle of incidence [deg]",
        ylabel="fraction of the incident power",
        title=f"{names[polarization]} polarization",
        legend="upper right",
        ax=panel,
    )
plots.heatmap(
    x,
    y,
    field,
    triangles=np.asarray(mesh.cells),
    limits=(0, np.percentile(field, 99.5)),
    label="|H_z| / |H_inc|",
    ylimits=(-0.3e-6, 0.4e-6),
    title="p polarization at 40 deg",
    length_unit="um",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the wire radius or the period in `plasmonic_wire_grating_geometry.py`, or sweep the
# wavelength instead of the angle: the absorption peak of the p polarization follows the localized
# plasmon of the wire, near $\varepsilon_m=-\varepsilon_{\rm surrounding}$.
