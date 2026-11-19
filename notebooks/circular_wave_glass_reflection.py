"""OneSimulation example: reflection of a circularly polarized plane wave at an air-glass interface.
Requires the installed, activated OneSimulation runtime.

A circularly polarized plane wave travels through a column of air, read from a STEP file, and hits
glass. The 3D frequency-domain field is split into incident and reflected waves and checked against
the Fresnel coefficients, the reversed sense of rotation of the reflected wave, the power balance
and the superposition of two linearly polarized solutions.
"""

# # Reflection of a circularly polarized plane wave at glass
#
# A plane wave in air meets the flat surface of glass with refractive index $n$ at normal
# incidence. Part of it is reflected, the rest is transmitted. For a circularly polarized wave
# both field components are reflected with the same coefficient, but because the reflected wave
# travels the other way, its sense of rotation with respect to the propagation direction is
# reversed: right-handed light comes back left-handed.
#
# The 3D model is a column of air, periodic in $x$ and $y$, so that it represents an unbounded
# plane wave. It comes from the CAD file `circular_wave_glass_reflection.step` (faces `x_min` ...
# `z_max`), written by `circular_wave_glass_reflection_geometry.py` with the CAD objects of
# OneSimulation. Two impedance boundaries close it:
#
# | boundary | condition |
# |---|---|
# | $x$ and $y$ sides | periodic pairs |
# | $z=0$ | impedance boundary of air, $Y_0=\sqrt{\varepsilon_0/\mu_0}$, with the incident field $\mathbf E_0=(\hat{\mathbf x}+\mathrm i\hat{\mathbf y})/\sqrt2$ |
# | $z=\ell$ | impedance boundary of glass, $Y=nY_0$, $n=1.5$: the glass half-space behind the interface |
#
# The impedance condition $\mathbf n\times\mathbf H=Y\,\mathbf n\times(\mathbf n\times\mathbf E)$
# is exact for a plane wave entering a half-space of admittance $Y$, so the glass need not be
# meshed. Nédélec elements of order 2 solve
# $\nabla\times(\mu_r^{-1}\nabla\times\mathbf E)-k_0^2\varepsilon_r\mathbf E=0$ at 3 GHz.
#
# **Reference** (M. Born and E. Wolf, *Principles of Optics*, 7th ed., §1.5; E. Hecht, *Optics*,
# 5th ed., §4.6 and §8.1). The Fresnel coefficients at normal incidence and the field in front of
# the interface (with $z'=z-\ell$) are
#
# $$r=\frac{1-n}{1+n},\qquad t=\frac{2}{1+n},\qquad R=r^2,\qquad T=n\,t^2=1-R,\qquad
#   \mathbf E(z)=\mathbf E_0\big(e^{\mathrm ik_0z}+r\,e^{\mathrm ik_0\ell}e^{-\mathrm ik_0z'}\big).$$
#
# For $n=1.5$: $r=-0.2$, $R=4\,\%$, $T=96\,\%$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import IsotropicElectromagneticMaterial
from onesim.meshing import PeriodicPair, StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import FrequencyDomainMaxwell, ImpedanceBoundary
from onesim.results.comparisons import convergence
from onesim.simulation import PeriodicBoundary, Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# Hexahedra with one cell across the periodic column; three meshes along the 200 mm, two
# wavelengths at 3 GHz.

column = load("circular_wave_glass_reflection.step")
faces = column.boundaries
width, depth, length = column.size
meshes = {
    cells: StructuredHexahedralMesher(cells=(1, 1, cells)).generate(column)
    for cells in (10, 20, 40)
}

fig = plots.figure(1, 2, size=(13, 5.5))
plots.geometry(
    column,
    selection={"incidence plane": faces.z_min, "air-glass interface": faces.z_max},
    labels=False,
    camera=plots.Camera(elevation=20, azimuth=-60),
    title="Air column",
    ax=fig[0],
)
plots.mesh(
    meshes[40],
    groups=("all",),
    legend=False,
    camera=plots.Camera(elevation=20, azimuth=-60),
    title=f"Finest mesh: {meshes[40].cell_count} hexahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The same problem is solved with the circular incident field on every mesh, and on the finest
# mesh once more with each linear polarization alone.

frequency, n = 3.0e9, 1.5
admittance = np.sqrt(VACUUM_PERMITTIVITY / VACUUM_PERMEABILITY)
k0 = 2 * np.pi * frequency * np.sqrt(VACUUM_PERMITTIVITY * VACUUM_PERMEABILITY)
model = FrequencyDomainMaxwell(
    domain=column.volume,
    frequency=frequency,
    material=IsotropicElectromagneticMaterial(),
    phasor_convention="exp(-i*omega*t)",
    amplitude_convention="peak",
)
numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=ComplexSparseDirect()
)
incident = {
    "circular": (1 / np.sqrt(2), 1j / np.sqrt(2), 0.0),
    "x": (1 / np.sqrt(2), 0.0, 0.0),
    "y": (0.0, 1 / np.sqrt(2), 0.0),
}
solve = lambda polarization, mesh: Simulation(
    problem=Problem(
        name="Plane wave at glass",
        geometry=column,
        physics=(model,),
        boundary_conditions=(
            *(
                PeriodicBoundary(
                    physics=model,
                    pair=PeriodicPair(faces[side + "_min"], faces[side + "_max"]),
                )
                for side in "xy"
            ),
            ImpedanceBoundary(
                physics=model,
                boundary=faces.z_min,
                admittance=admittance,
                incident_electric_field=incident[polarization],
            ),
            ImpedanceBoundary(
                physics=model, boundary=faces.z_max, admittance=n * admittance
            ),
        ),
    ),
    mesh=mesh,
    numerics=numerics,
).solve_stationary()
circular = {cells: solve("circular", mesh) for cells, mesh in meshes.items()}
linear = {polarization: solve(polarization, meshes[40]) for polarization in ("x", "y")}

# ## Comparison with Fresnel
#
# On a line along the column the field is fitted by a forward and a backward wave,
# $\mathbf E=\mathbf A\,e^{\mathrm ik_0z}+\mathbf B\,e^{-\mathrm ik_0z'}$; the reflection coefficient is
# $B/(A\,e^{\mathrm ik_0\ell})$ for each component. The transmitted power follows from the field on the
# interface, $T=n\,|\mathbf E(\ell)|^2/|\mathbf E_0|^2$.

z = np.linspace(0, length, 81)
line = np.column_stack((np.full_like(z, 0.4 * width), np.full_like(z, 0.6 * depth), z))
waves = np.column_stack((np.exp(1j * k0 * z), np.exp(-1j * k0 * (z - length))))
r, t = (1 - n) / (1 + n), 2 / (1 + n)
fields = {
    cells: result.field(model.electric_field).evaluate(line)
    for cells, result in circular.items()
}
split = {
    cells: np.linalg.lstsq(waves, field[:, :2], rcond=None)[0]
    for cells, field in fields.items()
}
reflection = {
    cells: B / (A * np.exp(1j * k0 * length)) for cells, (A, B) in split.items()
}
print(
    "hexahedra   r (E_x)              r (E_y)              R          T          R + T"
)
for cells, coefficient in reflection.items():
    transmitted = n * np.sum(np.abs(fields[cells][-1, :2]) ** 2)
    print(
        f"{meshes[cells].cell_count:9d}   {coefficient[0].real:+.5f}{coefficient[0].imag:+.5f}i   {coefficient[1].real:+.5f}{coefficient[1].imag:+.5f}i "
        f"{np.mean(np.abs(coefficient) ** 2):9.5f} {transmitted:10.5f} {np.mean(np.abs(coefficient) ** 2) + transmitted:10.5f}"
    )
print(
    f"Fresnel     r = {r:+.5f}             r = {r:+.5f}            {r**2:9.5f} {n * t**2:10.5f} {1.0:10.5f}"
)
study = convergence(
    [abs(coefficient[0] - r) for coefficient in reflection.values()],
    [length / cells for cells in reflection],
)
print(f"error of r falls with the cell size at order {study.order:.2f}")

A, B = split[40]
print(
    f"rotation in time: incident {'x -> y' if (A[0].conj() * A[1]).imag > 0 else 'y -> x'}, "
    f"reflected {'x -> y' if (B[0].conj() * B[1]).imag > 0 else 'y -> x'}; "
    f"relative to its direction of travel the reflected wave turns the other way"
)
superposed = linear["x"].field(model.electric_field).evaluate(line) + 1j * linear[
    "y"
].field(model.electric_field).evaluate(line)
print(
    f"circular solution against x + i y superposition of the linear solutions: largest difference "
    f"{np.abs(superposed - fields[40]).max():.1e} V/m"
)

# ## Figure
#
# The standing-wave pattern of the field strength in front of the glass, and the direction of the
# incident and of the reflected field vector along the column at one instant: both turn by 360°
# per wavelength, but in opposite senses, so the reflected wave has the opposite handedness.

fine = np.linspace(0, length, 600)
exact = np.abs(
    np.exp(1j * k0 * fine)
    + r * np.exp(1j * k0 * length) * np.exp(-1j * k0 * (fine - length))
) / np.sqrt(2)
screw = lambda amplitude, sign: np.degrees(
    np.unwrap(
        np.arctan2(
            (amplitude[1] * np.exp(sign * 1j * k0 * (z - length))).real,
            (amplitude[0] * np.exp(sign * 1j * k0 * (z - length))).real,
        )
    )
)
fig = plots.figure(1, 2, size=(13, 5))
plots.line(
    points={
        "E_x": (1e3 * z, np.abs(fields[40][:, 0])),
        "E_y": (1e3 * z, np.abs(fields[40][:, 1])),
    },
    reference={"Fresnel": (1e3 * fine, exact)},
    ylimits=(0.5, 0.9),
    xlabel="position z [mm]",
    ylabel="|E| [V/m]",
    title="Standing wave in front of the glass (interface at 200 mm)",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    1e3 * z,
    {
        "incident wave": screw(A * np.exp(1j * k0 * length), 1),
        "reflected wave": screw(B, -1),
    },
    xlabel="position z [mm]",
    ylabel="direction of E at t = 0 [°]",
    title="Opposite screw sense of the two waves",
    legend="upper center",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Give the glass a complex index through the admittance, $Y=(n+\mathrm i\kappa)Y_0$, to model an
# absorbing substrate, or replace the glass boundary by a perfect conductor: the wave is then
# reflected completely with $r=-1$, and the column holds a pure standing wave.
