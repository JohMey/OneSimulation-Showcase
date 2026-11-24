"""OneSimulation example: input impedance and radiation of a biconical antenna (2D-axisymmetric).
Requires the installed, activated OneSimulation runtime.

Two metal cones tip to tip, read from a STEP file, are fed by a 100 ohm generator across the gap
between their tips and solved on the meridian plane from 1 to 5 GHz. The input impedance approaches
the characteristic impedance of the biconical line, and the radiated power from the far field
balances the power the antenna accepts.
"""

# # Biconical antenna
#
# Two cones meeting tip to tip form a biconical transmission line: it guides a spherical TEM wave
# outward from the feed with the characteristic impedance $Z_c=120\ln\cot(\theta_0/2)$ for the half
# angle $\theta_0$ of the cones. Once the cones are longer than the wavelength, little of the wave is
# reflected at their ends, and the antenna is matched over a wide band with a pattern like that of a
# dipole: omnidirectional about the axis, strongest in the plane of the feed.
#
# Here each cone has a half angle of 47° ($Z_c\approx100\ \Omega$), a slant length of 100 mm and a flat
# top. The generator drives the gap of 2 mm between the tips across a thin feed wire. Everything is
# symmetric about the axis, and the TEM wave has only $E_r$, $E_z$ and $H_\varphi$: OneSimulation
# solves the meridian plane with `polarization="Erphiz"` and `azimuthal_mode=0`, and `LumpedPort`
# feeds the gap with 1 W from a 100 Ω generator.
#
# | boundary | condition |
# |---|---|
# | `port` (feed wire surface) | lumped port, 100 Ω, 1 W available |
# | `metal` (cones) | perfect conductor |
# | `outer` (box walls) | perfect conductor behind a spherical PML from 350 to 450 mm |
#
# The air comes from `biconical_antenna_impedance.step`, written by
# `biconical_antenna_impedance_geometry.py` with the CAD objects of OneSimulation.
#
# **Reference.** (i) The characteristic impedance of the infinite biconical line,
# $Z_c=120\ln\cot(\theta_0/2)=100.0\ \Omega$ (Schelkunoff, *Electromagnetic Waves*, 1943; Balanis,
# *Antenna Theory*, 4th ed., §9.3); the impedance of the finite antenna oscillates about it, less and
# less as the cones grow longer than the wavelength. (ii) Energy balance: the power radiated through
# the far field, $P_\mathrm{rad}=\frac{1}{2\eta_0}\oint|\mathbf F|^2\,\mathrm d\Omega$, equals the
# power the antenna accepts from the generator, $P_\mathrm{acc}=(1-|S_{11}|^2)\cdot1$ W.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import (
    IsotropicElectromagneticMaterial,
    spherical_pml,
)
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec, RevolvedH1SpaceFactory
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell,
    LumpedPort,
    PerfectElectricConductor,
    axisymmetric_far_field,
    input_impedance,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

light = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
eta = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)

# ## Geometry and mesh
#
# `section` with the plane $y=0$ gives the meridian plane ($x=r$). Quadratic triangles of 7 mm
# (eight per wavelength at 5 GHz), refined to 0.7 mm at the feed gap and 2 mm on the cones.

air = load("biconical_antenna_impedance.step")
meridian = air.section((0, 1, 0))
faces = meridian.boundaries
mesh = TriangularMesher(
    target_size=0.007, local_sizes={"port": 0.0007, "metal": 0.002}, grading=0.25
).generate(meridian)

fig = plots.figure(1, 2, size=(14, 6.5))
plots.geometry(
    air,
    selection={"feed": air.boundaries.port, "cones": air.boundaries.metal},
    labels=False,
    camera=plots.Camera(elevation=20, azimuth=-60),
    title="Air around the antenna, half (STEP)",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Meridian plane: {mesh.cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# One solve per frequency from 1 to 5 GHz in steps of 0.25 GHz.

model = lambda frequency: FrequencyDomainMaxwell(
    domain=meridian.domain(),
    frequency=frequency,
    material=spherical_pml(IsotropicElectromagneticMaterial(), 0.35, 0.45),
    polarization="Erphiz",
    azimuthal_mode=0,
    phasor_convention="exp(-i*omega*t)",
    amplitude_convention="peak",
)
numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=2), provider=RevolvedH1SpaceFactory()),
    linear_solver=ComplexSparseDirect(),
)
solve = lambda m: Simulation(
    problem=Problem(
        name="Bicone",
        geometry=meridian,
        physics=(m,),
        boundary_conditions=(
            LumpedPort(
                physics=m, boundary=faces.port, impedance=100.0, incident_power=1.0
            ),
            PerfectElectricConductor(physics=m, boundary=faces.metal | faces.outer),
        ),
    ),
    mesh=mesh,
    numerics=numerics,
).solve_stationary()
frequencies = np.arange(1.0e9, 5.01e9, 0.25e9)
runs = {f: (solve(m := model(f)), m) for f in frequencies}

# ## Results
#
# Input impedance (in the engineering convention $e^{+j\omega t}$, the conjugate of the solver's
# phasors), reflection coefficient against 100 Ω, and the energy balance from the far field.

impedance = np.array([np.conj(input_impedance(*runs[f])) for f in frequencies])
reflection = np.abs((impedance - 100.0) / (impedance + 100.0))
theta = np.linspace(0, np.pi, 181)
radiated = np.array(
    [
        np.pi
        / eta
        * np.trapezoid(
            np.sum(
                np.abs(axisymmetric_far_field(*runs[f], theta, 0.0, radius=0.3)) ** 2,
                axis=1,
            )
            * np.sin(theta),
            theta,
        )
        for f in frequencies
    ]
)
characteristic = 120 * np.log(1 / np.tan(np.radians(47.0) / 2))
print(f"characteristic impedance of the biconical line: {characteristic:.2f} ohm")
print(
    "f [GHz]   slant / wavelength   Z_in [ohm]            |S11|     P_rad [W]   P_acc [W]   difference"
)
for f, z, s, p in zip(frequencies, impedance, reflection, radiated):
    print(
        f"{f / 1e9:7.2f} {0.1 * f / light:18.2f}   {z.real:7.2f} {z.imag:+8.2f}j {s:10.4f} {p:11.4f} {1 - s**2:11.4f} {p / (1 - s**2) - 1:+11.1e}"
    )
long = frequencies >= 3e9
print(
    f"above 3 GHz (cones longer than one wavelength): mean resistance {impedance[long].real.mean():.1f} ohm, mean reactance {impedance[long].imag.mean():+.1f} ohm, "
    f"largest |S11| {reflection[long].max():.3f}"
)

# ## Figure
#
# Left: the input impedance against frequency with the characteristic impedance of the biconical
# line. Right: the radiation pattern $|\mathbf F|^2$ in the elevation plane at 1.5, 3 and 5 GHz,
# normalized to its maximum; it is symmetric about the plane of the feed, and at high frequencies the
# finite cones split the main lobe.

fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    frequencies / 1e9,
    {"resistance": impedance.real, "reactance": impedance.imag},
    markers=True,
    hlines={"Z_c of the biconical line": characteristic},
    xlabel="frequency [GHz]",
    ylabel="input impedance [ohm]",
    title="Input impedance",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    90 - np.degrees(theta),
    {
        f"{f / 1e9:.1f} GHz": (lambda p: p / p.max())(
            np.sum(
                np.abs(axisymmetric_far_field(*runs[f], theta, 0.0, radius=0.3)) ** 2,
                axis=1,
            )
        )
        for f in (1.5e9, 3.0e9, 5.0e9)
    },
    xlabel="elevation above the plane of the feed [°]",
    ylabel="|F|^2 / max",
    title="Radiation pattern",
    legend="upper left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Change the half angle in `biconical_antenna_impedance_geometry.py`: 30° raises the characteristic
# impedance to about 158 Ω. Spherical caps instead of flat tops smooth the oscillation of the
# impedance at low frequencies.
