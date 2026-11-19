"""OneSimulation example: a conical dielectric probe on skin at 35 GHz.
Requires the installed, activated OneSimulation runtime.

A PTFE-filled circular waveguide ends in a PTFE cone whose tip touches the skin. The TE11 mode of
the guide is fed at 30 to 40 GHz and the reflection is computed for healthy skin and for skin with a
lesion of higher water content (a skin tumour) under the tip. The body of revolution is solved on its
meridian for the azimuthal mode m = 1; the port mode is compared with the Bessel cutoff of the
guide and the power balance with the Poynting flux into the absorbing layers.
"""

# # Conical dielectric probe on skin
#
# Millimetre waves near 35 GHz are absorbed strongly by water, and skin tumours hold more water than
# healthy skin. A small probe pressed onto the skin therefore reflects differently over a tumour:
# microwave reflectometry. The probe here is a circular waveguide of 2.5 mm radius filled with PTFE
# ($\varepsilon_r=2.08$) that tapers into a PTFE cone of 6 mm length with a tip of 1 mm radius. The
# skin is 1.5 mm thick on fat; the lesion under the tip is 5 mm across and 1 mm deep.
#
# The TE11 mode of a circular guide varies as $\cos\varphi$, $\sin\varphi$: it is the azimuthal mode
# $m=1$ of a body of revolution. `FrequencyDomainMaxwell` with polarization `Erphiz` and
# `azimuthal_mode=1` solves all three field components $(E_r,E_\varphi,E_z)\,e^{\mathrm i\varphi}$ on
# the meridian plane with Nédélec elements; a `SectionWaveguidePort` on the guide cross-section
# solves the modes of its line, feeds the TE11 mode and absorbs what comes back. Absorbing layers
# (`CartesianPML`) 2 mm thick close the air and the tissue.
#
# | boundary | condition |
# |---|---|
# | `port` (cross-section of the guide) | waveguide port, TE11 ($m=1$), 1 W |
# | `metal` (guide wall) | perfect electric conductor |
# | `outer` (behind the absorbing layers) | perfect electric conductor |
# | axis | regular for $m=1$: $E_\varphi$ and $E_z$ vanish |
#
# Permittivities at 35 GHz (written $\varepsilon'+\mathrm i\varepsilon''$): skin $13.9+14.5\mathrm i$
# and fat $3.2+0.9\mathrm i$ from the Cole-Cole fits of Gabriel et al.; for the tumour tissue of the
# lesion the model assumes $21+22\mathrm i$, the skin value with about 50 % more water. They are kept constant over
# the band.
#
# **Reference** (D. M. Pozar, *Microwave Engineering*, 4th ed., §3.4; S. Gabriel, R. W. Lau and
# C. Gabriel, *Phys. Med. Biol.* 41, 2271 (1996); Poynting's theorem). The port mode must have the
# propagation constant of the TE11 mode of the PTFE-filled guide,
# $\beta=\sqrt{\varepsilon_rk_0^2-(x'_{11}/a)^2}$ with $x'_{11}=1.8412$ the first zero of $J_1'$. The
# reflection is checked by the power balance $1-|S_{11}|^2=P_{abs}+P_{rad}$: $P_{abs}$ from the
# absorbed power density in skin, lesion and fat, $P_{rad}$ from the Poynting flux
# $\tfrac12\mathrm{Re}(\mathbf E\times\mathbf H^*)\cdot\mathbf n\,2\pi r$ through the surfaces just inside
# the absorbing layers - three parts of the solution that are computed independently. A mesh study
# gives the discretization error of $S_{11}$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import (
    CartesianPML,
    IsotropicElectromagneticMaterial,
    RegionwiseElectromagneticMaterial,
)
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec, RevolvedH1SpaceFactory
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell,
    PerfectElectricConductor,
    SectionWaveguidePort,
    port_modes,
    scattering_parameters,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.special import jnp_zeros

theme.set()

mm, radius, ptfe = 1e-3, 2.5e-3, 2.08
light = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
tissue = {"skin": 13.9 + 14.5j, "fat": 3.2 + 0.9j, "tumour": 21.0 + 22.0j}

# ## Geometry and mesh
#
# `load` reads the half solids of revolution; `section` with the plane $y=0$ returns their meridians:
# the air, the PTFE of guide and cone, the skin, the fat and the lesion. Quadratic triangles of
# 0.4 mm, 0.16 mm in skin and lesion (the wavelength in skin is about 1.9 mm).

plane = {
    name: load(f"conical_dielectric_probe{name}.step").section((0, 1, 0))
    for name in ("", "_ptfe", "_skin", "_fat", "_lesion")
}
section = plane[""]
sides = section.boundaries
regions = tuple(
    plane[f"_{name}"].named(name) for name in ("ptfe", "skin", "fat", "lesion")
)
mesher = lambda size: TriangularMesher(
    target_size=size,
    embedded_regions=regions,
    region_sizes={"skin": 0.4 * size, "lesion": 0.4 * size},
)
mesh = mesher(0.4 * mm).generate(section)

fig = plots.figure(1, 2, size=(12, 9))
plots.geometry(
    section,
    selection={"port": sides.port, "metal": sides.metal, "outer": sides.outer},
    labels=False,
    title="Meridian of probe, skin and fat (STEP)",
    length_unit="mm",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"{mesh.cell_count} quadratic triangles",
    length_unit="mm",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The absorbing layers fill the outer 2 mm: beyond $r=6$ mm, below $z=-4$ mm and above $z=8$ mm
# outside the guide wall. One model per frequency and tissue.

medium = lambda permittivity: IsotropicElectromagneticMaterial(
    relative_permittivity=permittivity
)
material = lambda frequency, tumour: CartesianPML(
    material=RegionwiseElectromagneticMaterial(
        {
            "ptfe": medium(ptfe),
            "skin": medium(tissue["skin"]),
            "fat": medium(tissue["fat"]),
            "lesion": medium(tissue["tumour" if tumour else "skin"]),
        },
        default=medium(1.0),
    ),
    damping=lambda x, y, z: 4
    * np.pi
    * frequency
    * np.column_stack(
        (
            np.clip((x - 6 * mm) / (2 * mm), 0, None) ** 2,
            0 * x,
            np.clip((-4 * mm - z) / (2 * mm), 0, None) ** 2
            + np.clip((z - 8 * mm) / (2 * mm), 0, None) ** 2 * (x > 2.8 * mm),
        )
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=2), provider=RevolvedH1SpaceFactory()),
    linear_solver=ComplexSparseDirect(),
)
model = lambda frequency, tumour: FrequencyDomainMaxwell(
    domain=section.domain(),
    frequency=frequency,
    material=material(frequency, tumour),
    polarization="Erphiz",
    azimuthal_mode=1,
    phasor_convention="exp(-i*omega*t)",
    amplitude_convention="peak",
)
solve = lambda physics, grid=mesh: Simulation(
    problem=Problem(
        name="conical probe",
        geometry=section,
        physics=(physics,),
        boundary_conditions=(
            SectionWaveguidePort(
                physics=physics, boundary=sides.port, modes=1, incident_power=1.0
            ),
            PerfectElectricConductor(
                physics=physics, boundary=sides.metal | sides.outer
            ),
        ),
    ),
    mesh=grid,
    numerics=numerics,
).solve_stationary()

frequencies = np.linspace(30e9, 40e9, 11)
cases = {
    (frequency, tumour): model(frequency, tumour)
    for frequency in frequencies
    for tumour in (False, True)
}
results = {key: solve(physics) for key, physics in cases.items()}
reflection = {
    key: scattering_parameters(result, cases[key])[(0, 0)]
    for key, result in results.items()
}

# ## Results
#
# The port mode against the Bessel cutoff, the reflection with and without tumour, and the power
# balance. The Poynting flux leaves through $r=6$ mm, $z=-4$ mm and $z=8$ mm ($r>2.8$ mm);
# $\mathbf H=\mathbf B/\mu_0$.

surfaces = [
    ((6 * mm, -4 * mm), (6 * mm, 8 * mm), (1.0, 0.0)),
    ((0.0, -4 * mm), (6 * mm, -4 * mm), (0.0, -1.0)),
    ((2.8 * mm, 8 * mm), (6 * mm, 8 * mm), (0.0, 1.0)),
]
steps = np.linspace(0.0, 1.0, 2001)
lines = [
    (
        np.asarray(a) + np.outer(steps, np.subtract(b, a)),
        np.hypot(*np.subtract(b, a)),
        np.asarray(normal),
    )
    for a, b, normal in surfaces
]
flux = {
    key: sum(
        np.trapezoid(
            (
                0.5
                * np.cross(
                    result.field(cases[key].electric_field).evaluate(points),
                    np.conj(
                        result.field(cases[key].magnetic_flux_density).evaluate(points)
                    )
                    / VACUUM_PERMEABILITY,
                )[:, [0, 2]]
                @ normal
            ).real
            * 2
            * np.pi
            * points[:, 0],
            steps * length,
        )
        for points, length, normal in lines
    )
    for key, result in results.items()
}
absorbed = {
    key: result.field(cases[key].absorbed_power_density).integrate().value
    for key, result in results.items()
}
print(
    "frequency   beta TE11 [rad/m]          |S11| [dB]            1-|S11|^2   P_abs    P_rad    P_abs+P_rad"
)
print("[GHz]       computed      exact        healthy   tumour      (tumour)")
for frequency in frequencies:
    k0 = 2 * np.pi * frequency / light
    beta = port_modes(results[(frequency, True)], cases[(frequency, True)])[0][
        "propagation_constant"
    ][0].real
    exact = np.sqrt(ptfe * k0**2 - (jnp_zeros(1, 1)[0] / radius) ** 2)
    key = (frequency, True)
    print(
        f"{frequency / 1e9:6.1f}     {beta:9.3f}   {exact:9.3f}     {20 * np.log10(abs(reflection[(frequency, False)])):7.2f}  {20 * np.log10(abs(reflection[key])):7.2f}"
        f"      {1 - abs(reflection[key]) ** 2:.4f}     {absorbed[key]:.4f}   {flux[key]:.4f}   {absorbed[key] + flux[key]:.4f}"
    )
contrast = max(
    abs(20 * np.log10(abs(reflection[(f, True)]) / abs(reflection[(f, False)])))
    for f in frequencies
)
balance = max(
    abs(1 - abs(reflection[key]) ** 2 - absorbed[key] - flux[key]) for key in results
)
print(
    f"\nlargest change of |S11| by the tumour: {contrast:.2f} dB; largest power-balance mismatch: {balance:.1e}"
)

# The mesh study at 35 GHz: triangles of 0.25 mm (0.1 mm in skin and lesion).

fine = mesher(0.25 * mm).generate(section)
refined = {
    tumour: scattering_parameters(
        solve(cases[(35e9, tumour)], fine), cases[(35e9, tumour)]
    )[(0, 0)]
    for tumour in (False, True)
}
for tumour, value in refined.items():
    print(
        f"{'tumour ' if tumour else 'healthy'}  S11 = {reflection[(35e9, tumour)]:.4f} ({mesh.cell_count} triangles), {value:.4f} ({fine.cell_count} triangles): "
        f"change {abs(value - reflection[(35e9, tumour)]):.1e}"
    )

# ## Figure
#
# The reflection over the band with and without tumour, and the electric field in the probe and the
# tissue at 35 GHz with the tumour.

r, z = np.asarray(mesh.coordinates).T
values = np.linalg.norm(
    results[(35e9, True)]
    .field(cases[(35e9, True)].electric_field)
    .evaluate(np.column_stack((r, z))),
    axis=1,
)
fig = plots.figure(1, 2, size=(14, 7.5))
plots.line(
    frequencies / 1e9,
    {
        "healthy skin": [
            20 * np.log10(abs(reflection[(f, False)])) for f in frequencies
        ],
        "tumour lesion under the tip": [
            20 * np.log10(abs(reflection[(f, True)])) for f in frequencies
        ],
    },
    xlabel="frequency [GHz]",
    ylabel="|S11| [dB]",
    title="Reflection of the probe",
    legend="upper right",
    ax=fig[0],
)
plots.heatmap(
    r,
    z,
    values,
    triangles=np.asarray(mesh.cells),
    limits=(0, np.percentile(values, 99)),
    label="|E| [V/m]",
    length_unit="mm",
    xlabel="r [m]",
    ylabel="z [m]",
    ylimits=(-4 * mm, 12 * mm),
    title="|E| at 35 GHz with tumour",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Change the size or depth of the lesion in `conical_dielectric_probe_geometry.py`, or give the probe a longer
# cone: the field at the tip, and with it the sensitivity, depends on how much of the guided power
# the cone delivers to its tip rather than radiating it from its flanks. The absorbed power density
# is the heat source of a bioheat model that gives the temperature rise of the skin.
