"""OneSimulation example: a monopole antenna on a ground plane fed by a magnetic frill.
Requires the installed, activated OneSimulation runtime.

A quarter-wave monopole stands on an infinite ground plane and is fed through it by a coaxial line,
whose aperture field is the magnetic frill of antenna theory. The axisymmetric field is solved on
the meridian plane read from a STEP file, with the coaxial TEM mode as waveguide port. Input
impedance, reflection, radiated power and radiation pattern are compared with the sinusoidal-current
(induced EMF) model of the monopole and with the power balance of the port.
"""

# # Monopole antenna fed by a magnetic frill
#
# Voltages are not defined in an electromagnetic wave model, so an antenna needs a feed that is a
# field. The classic one is the **magnetic frill**: the radial electric field
# $E_r=V/(r\ln(b/a))$ in the annular aperture of a coaxial line between its inner radius $a$ and
# outer radius $b$, equivalent to a ring of magnetic current $M_\varphi=-2E_r$ on the ground plane.
# Here the frill is not imposed by hand: 10 mm of the 50 ohm coaxial line are part of the model, and
# a `SectionWaveguidePort` at its end feeds the TEM mode and absorbs what is reflected. The field
# in the aperture then is the frill field of the actual antenna current.
#
# The monopole (height $h=70$ mm, radius $a=1.5$ mm) and its feed are bodies of revolution, and the
# field has only $H_\varphi$, $E_r$ and $E_z$: `FrequencyDomainMaxwell` with polarization `Hphi` on
# the meridian plane ($r$, $z$). A first-order absorbing `ImpedanceBoundary` on the hemisphere of
# 0.3 m (one wavelength at 1 GHz) lets the radiation leave.
#
# | boundary | condition |
# |---|---|
# | `feed` (end of the coaxial line) | waveguide port, TEM mode, 1 W |
# | `antenna`, `ground` | perfect electric conductor |
# | `outer` (hemisphere) | impedance boundary $Y_0=\sqrt{\varepsilon_0/\mu_0}$ |
# | `axis` | regular: $H_\varphi=0$ |
#
# The impedance is read inside the coaxial line, 5 mm below the ground plane, where the field is a
# pure TEM wave: $V=\int_a^bE_r\,\mathrm dr$ and $I=2\pi rH_\varphi$ (Ampère) give $Z=V/I$, whose
# reflection $|Z-Z_c|/|Z+Z_c|$ ($Z_c=\eta_0\ln(b/a)/2\pi$) must match the reflection of the port and
# $\tfrac12\mathrm{Re}(VI^*)$ the power accepted at the port. Moving the reflection coefficient
# $\Gamma$ along the 5 mm of line, $\Gamma_L=\Gamma e^{-2\mathrm ik\ell}$, gives the input impedance at
# the ground plane, $Z_{in}=Z_c(1+\Gamma_L)/(1-\Gamma_L)$. `far_field` with `ground_plane=0`
# mirrors the hemisphere in the ground plane and returns the pattern of the full space.
#
# **Reference** (C. A. Balanis, *Antenna Theory*, 4th ed., §4.6 and §4.7.4; S. A. Schelkunoff and
# H. T. Friis, *Antennas: Theory and Practice*, 1952). With the sinusoidal current
# $I(z)=I_0\sin k(h-z)$ the monopole radiates
#
# $$E_\theta=\frac{\mathrm i\eta_0I_0}{2\pi}\frac{e^{\mathrm ikr}}{r}
#   \frac{\cos(kh\cos\theta)-\cos kh}{\sin\theta},\qquad
#   R_{in}=\frac{\eta_0}{2\pi\sin^2kh}\int_0^{\pi/2}\frac{(\cos(kh\cos\theta)-\cos kh)^2}{\sin\theta}\mathrm d\theta,$$
#
# the input resistance referred to the base current (36.5 ohm for $h=\lambda/4$). The thickness of
# a real wire shortens the resonance by a few percent but leaves the resistance there close to the
# thin-wire value.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import IsotropicElectromagneticMaterial
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, RevolvedH1SpaceFactory
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell,
    ImpedanceBoundary,
    PerfectElectricConductor,
    SectionWaveguidePort,
    far_field,
    scattering_parameters,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

height, wire, shield = 0.07, 0.0015, 0.00345
light = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
impedance = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)
characteristic = impedance * np.log(shield / wire) / (2 * np.pi)

# ## Geometry and mesh
#
# `load` reads the half solid of revolution; `section` with the plane $y=0$ returns its meridian
# plane with the named sides. Quadratic triangles of 12 mm (a 25th of the wavelength), 0.7 mm along
# the antenna and 0.4 mm in the coaxial line.

plane = load("monopole_magnetic_frill_feed.step").section((0, 1, 0))
sides = plane.boundaries
mesh = TriangularMesher(
    target_size=0.012, local_sizes={"antenna": 7e-4, "feed": 4e-4}
).generate(plane)

fig = plots.figure(1, 2, size=(14, 7))
plots.geometry(
    plane,
    selection={
        "feed (coaxial port)": sides.feed,
        "antenna": sides.antenna,
        "ground": sides.ground,
        "outer": sides.outer,
    },
    labels=False,
    title="Meridian plane of monopole and feed (STEP)",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"{mesh.cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# One model per frequency from 0.8 to 1.3 GHz; the port feeds 1 W in the TEM mode of the coaxial
# line.

numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2), provider=RevolvedH1SpaceFactory()),
    linear_solver=ComplexSparseDirect(),
)
model = lambda frequency: FrequencyDomainMaxwell(
    domain=plane.domain(),
    frequency=frequency,
    material=IsotropicElectromagneticMaterial(),
    polarization="Hphi",
    phasor_convention="exp(-i*omega*t)",
    amplitude_convention="peak",
)
solve = lambda physics: Simulation(
    problem=Problem(
        name="monopole",
        geometry=plane,
        physics=(physics,),
        boundary_conditions=(
            SectionWaveguidePort(
                physics=physics, boundary=sides.feed, modes=1, incident_power=1.0
            ),
            PerfectElectricConductor(
                physics=physics, boundary=sides.antenna | sides.ground
            ),
            ImpedanceBoundary(
                physics=physics, boundary=sides.outer, admittance=1 / impedance
            ),
        ),
    ),
    mesh=mesh,
    numerics=numerics,
).solve_stationary()

frequencies = np.linspace(0.8e9, 1.3e9, 11)
models = [model(frequency) for frequency in frequencies]
results = [solve(physics) for physics in models]

# ## Results
#
# Voltage and current in the coaxial line, the reflection and power they imply against those of the
# port, and the input impedance at the ground plane.

across, below = np.linspace(wire, shield, 201), 0.005
aperture = np.column_stack((across, -below + 0 * across))
voltage = np.array(
    [
        np.trapezoid(
            result.field(physics.electric_field).evaluate(aperture)[:, 0], across
        )
        for result, physics in zip(results, models)
    ]
)
middle = np.array([[(wire + shield) / 2, -below]])
current = np.array(
    [
        2
        * np.pi
        * middle[0, 0]
        * result.field(physics.magnetic_flux_density).evaluate(middle)[0, 1]
        / VACUUM_PERMEABILITY
        for result, physics in zip(results, models)
    ]
)
line_impedance = voltage / current
reflection = np.array(
    [
        abs(scattering_parameters(result, physics)[(0, 0)])
        for result, physics in zip(results, models)
    ]
)
from_impedance = np.abs(
    (line_impedance - characteristic) / (line_impedance + characteristic)
)
load_reflection = (
    (line_impedance - characteristic)
    / (line_impedance + characteristic)
    * np.exp(-2j * 2 * np.pi * frequencies / light * below)
)
input_impedance = characteristic * (1 + load_reflection) / (1 - load_reflection)
print(f"coaxial line: Z_c = {characteristic:.2f} ohm")
print(
    "f [GHz]   Z_in at the ground [ohm]   |S11| port   |Z-Zc|/|Z+Zc|   accepted power [W]: 1-|S11|^2   Re(V I*)/2"
)
for f, z, s, r, v, i in zip(
    frequencies, input_impedance, reflection, from_impedance, voltage, current
):
    print(
        f"{f / 1e9:6.2f}   {z.real:7.2f} {z.imag:+8.2f}i      {s:.4f}       {r:.4f}            {1 - s**2:.4f}          {0.5 * (v * np.conj(i)).real:.4f}"
    )

# The resonance lies where the reactance changes sign; the induced EMF resistance of the
# sinusoidal current is evaluated at the same electrical height.

resonance = np.interp(0.0, input_impedance.imag[::-1], frequencies[::-1])
kh = 2 * np.pi * resonance / light * height
theta = np.linspace(1e-4, np.pi / 2, 4001)
emf = (
    impedance
    / (2 * np.pi * np.sin(kh) ** 2)
    * np.trapezoid(
        (np.cos(kh * np.cos(theta)) - np.cos(kh)) ** 2 / np.sin(theta), theta
    )
)
print(
    f"\nresonance at {resonance / 1e9:.3f} GHz (h = {height / (light / resonance):.3f} wavelengths); "
    f"R_in there {np.interp(resonance, frequencies, input_impedance.real):.1f} ohm, induced EMF (thin wire) {emf:.1f} ohm"
)

# The radiated power from the far field over the upper half space against the power accepted at
# the port, and the pattern against the sinusoidal-current pattern, at 1 GHz.

at = int(np.argmin(np.abs(frequencies - 1.0e9)))
angles = np.linspace(0.01, np.pi / 2 - 0.01, 90)
pattern = far_field(
    results[at], models[at], angles, radius=0.2, center=(0.0, 0.0), ground_plane=0.0
)
radiated = np.trapezoid(
    np.sum(np.abs(pattern) ** 2, axis=1) / (2 * impedance) * 2 * np.pi * np.sin(angles),
    angles,
)
kh = 2 * np.pi * frequencies[at] / light * height
ideal = np.abs((np.cos(kh * np.cos(angles)) - np.cos(kh)) / np.sin(angles))
shape = np.linalg.norm(pattern, axis=1)
print(
    f"at {frequencies[at] / 1e9:.1f} GHz: radiated {radiated:.4f} W, accepted 1 - |S11|^2 = {1 - reflection[at]**2:.4f} W; "
    f"pattern against the sinusoidal current (both normalized at the horizon): max deviation {np.max(np.abs(shape / shape[-1] - ideal / ideal[-1])):.1e}"
)

# ## Figure
#
# Input impedance and reflection over frequency, the pattern in the vertical plane, and the
# magnitude of $H_\varphi$ near the antenna at 1 GHz.

r, z = np.asarray(mesh.coordinates).T
field = (
    np.abs(
        results[at]
        .field(models[at].magnetic_flux_density)
        .evaluate(np.column_stack((r, z)))[:, 1]
    )
    / VACUUM_PERMEABILITY
)
fig = plots.figure(1, 3, size=(24, 6.5))
plots.line(
    frequencies / 1e9,
    {"R_in": input_impedance.real, "X_in": input_impedance.imag},
    hlines={"Z_c": characteristic, "induced EMF R": emf},
    xlabel="frequency [GHz]",
    ylabel="impedance [ohm]",
    title="Input impedance at the ground plane",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    np.degrees(angles),
    {"computed |F|": shape / shape[-1]},
    reference={"computed |F|": ideal / ideal[-1]},
    xlimits=(0, 90),
    ylimits=(0, 1.05),
    xlabel="angle from the zenith [deg]",
    ylabel="|E| / |E(horizon)|",
    title="Pattern at 1 GHz (dashed: sinusoidal current)",
    legend="upper left",
    ax=fig[1],
)
plots.heatmap(
    1e3 * r,
    1e3 * z,
    field,
    triangles=np.asarray(mesh.cells),
    limits=(0, np.percentile(field[z > 0.005], 95)),
    label="|H_phi| [A/m]",
    xlabel="r [mm]",
    ylabel="z [mm]",
    xlimits=(0, 150),
    ylimits=(-10, 150),
    title="Field near the antenna at 1 GHz",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Make the wire thinner in `monopole_magnetic_frill_feed_geometry.py`: the resonance moves towards
# a quarter wavelength and the impedance curve towards the thin-wire values. A top hat (a disc on
# the antenna) or a dielectric sleeve shortens the antenna for the same resonance.
