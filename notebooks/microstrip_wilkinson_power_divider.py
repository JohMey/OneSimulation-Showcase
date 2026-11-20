"""OneSimulation example: a microstrip Wilkinson power divider in 3D.
Requires the installed, activated OneSimulation runtime.

Two quarter-wave arms split a 3 GHz signal equally; a 100-ohm resistor between their ends isolates
the outputs from each other. The divider is read from STEP files and solved in 3D with Nédélec
elements: the copper is a conducting sheet, the resistor a lumped element on the substrate and each
line ends in a TEM port. The S-matrix is compared with the circuit theory of the divider.
"""

# # Microstrip Wilkinson power divider
#
# A Wilkinson divider splits the power entering port 1 equally and in phase between ports 2 and 3
# while all three ports stay matched and the two outputs isolated from each other. Two arms of
# $\sqrt2\,Z_0=70.71$ Ω, a quarter wavelength long, join the input to the outputs; a resistor of
# $2Z_0=100$ Ω between the ends of the arms carries no current when port 1 drives both outputs alike,
# but absorbs the power that one output sends towards the other. Here the arms form a ring of radius
# 5.03 mm on a 0.813 mm substrate of relative permittivity 3.55, for 3 GHz; the resistor bridges a
# 1 mm gap between the arm ends, and 50 Ω lines lead to the sides of a box of air.
#
# OneSimulation solves the time-harmonic Maxwell equation
#
# $$\nabla\times\big(\mu_r^{-1}\nabla\times\mathbf E\big)-k_0^2\,\varepsilon_r\,\mathbf E=0$$
#
# with second-order Nédélec elements. The copper is a sheet without thickness on the substrate
# (`region_interface` of a thin layer of air shaped like it); the resistor is a `LumpedElement` on the
# face of the substrate between the arm ends (the top of a thin piece of substrate under the gap): a sheet of uniform current across the gap whose voltage
# and current obey $V=RI$. The lines end in `TEMPort`s that compute the quasi-TEM mode and the
# characteristic impedance of their cross-sections; `scattering_matrix` feeds the three ports in turn
# from one factorization and renormalizes the waves to 50 Ω.
#
# | faces | condition |
# |---|---|
# | `port_1`, `port_2`, `port_3` | TEM port, signal conductor: the copper |
# | resistor face | lumped element, $R=100$ Ω across the gap |
# | `ground`, `wall`, copper | perfect electric conductor |
# | `lid` | impedance boundary $Y_0=\sqrt{\varepsilon_0/\mu_0}$ (open top) |
#
# **Reference** (D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley 2012, §7.3; E. J. Wilkinson,
# *An N-way hybrid power divider*, IRE Trans. MTT 8 (1960) 116-118). At the centre frequency the
# ideal divider has
#
# $$[S]=\frac{-\mathrm j}{\sqrt2}\begin{pmatrix}0&1&1\\1&0&0\\1&0&0\end{pmatrix}\quad(e^{\mathrm j\omega t}):$$
#
# $|S_{21}|=|S_{31}|=-3.01$ dB, all ports matched and $S_{23}=0$; power fed into port 2 goes half to
# port 1 and half into the resistor. At other frequencies the ideal circuit (two lines of electrical
# length $\theta=\frac\pi2f/f_0$ and the resistor) is solved by its nodal admittance matrix,
# $S=(I-Z_0Y)(I+Z_0Y)^{-1}$.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import (
    IsotropicElectromagneticMaterial,
    RegionwiseElectromagneticMaterial,
)
from onesim.meshing import LocalRefinement, TetrahedralMesher, region_interface
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell,
    ImpedanceBoundary,
    LumpedElement,
    PerfectElectricConductor,
    TEMPort,
    port_values,
    scattering_matrix,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

eta0 = np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)
f0, height = 3e9, 0.813e-3

# ## Geometry and mesh
#
# `load` reads the box of air with its named faces, the substrate, the thin layer of air shaped like
# the copper and a thin piece of substrate under the gap whose top face is the resistor; all become
# regions of one mesh. Second-order tetrahedra of
# 1 mm follow the copper, 0.5 mm the resistor, 1.6 mm the substrate and 5 mm the air.

box = load("microstrip_wilkinson_power_divider.step")
substrate = load("microstrip_wilkinson_power_divider_substrate.step").named("substrate")
strips = load("microstrip_wilkinson_power_divider_strips.step").named("strips")
resistor = load("microstrip_wilkinson_power_divider_resistor.step").named("resistor")
faces = box.boundaries
mesh = TetrahedralMesher(
    target_size=5e-3,
    embedded_regions=(substrate, strips, resistor),
    refinements=(
        LocalRefinement(region=strips, target_size=1e-3),
        LocalRefinement(region=resistor, target_size=0.5e-3),
        LocalRefinement(region=substrate, target_size=1.6e-3),
    ),
).generate(box)
copper, film = region_interface(mesh, strips, substrate), region_interface(
    mesh, resistor
)

view = plots.Camera(elevation=45, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [substrate, strips, resistor],
    labels=False,
    camera=view,
    title="Substrate, copper and resistor (STEP)",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} quadratic tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `simulation` builds the model at one frequency with the three TEM ports and the resistor; the sweep
# runs from 2 to 4 GHz.

material = RegionwiseElectromagneticMaterial(
    {
        name: IsotropicElectromagneticMaterial(relative_permittivity=3.55)
        for name in ("substrate", "resistor")
    },
    default=IsotropicElectromagneticMaterial(),
)
simulation = lambda f, fed=None: (
    Simulation(
        problem=Problem(
            name="Wilkinson divider",
            geometry=box,
            physics=(
                m := FrequencyDomainMaxwell(
                    domain=box.volume,
                    frequency=f,
                    material=material,
                    phasor_convention="exp(-i*omega*t)",
                    amplitude_convention="peak",
                ),
            ),
            boundary_conditions=tuple(
                TEMPort(
                    physics=m,
                    boundary=getattr(faces, f"port_{k}"),
                    signal=copper,
                    incident_power=float(k == fed),
                )
                for k in (1, 2, 3)
            )
            + (
                LumpedElement(physics=m, boundary=film, resistance=100.0),
                PerfectElectricConductor(
                    physics=m, boundary=faces.ground | faces.wall | copper
                ),
                ImpedanceBoundary(physics=m, boundary=faces.lid, admittance=1 / eta0),
            ),
        ),
        mesh=mesh,
        numerics=Numerics(
            spatial=FiniteElement(element=Nedelec(order=2)),
            linear_solver=ComplexSparseDirect(),
        ),
    ),
    m,
)
frequencies = np.round(np.linspace(2e9, 4e9, 5), -6)
sweep = {
    f: scattering_matrix(*simulation(f), reference_impedance=50.0) for f in frequencies
}


def ideal(f):
    theta, e, c = np.pi / 2 * f / f0, np.eye(3), 1 / (50 * np.sqrt(2))
    y = (
        sum(
            c
            * (
                1j / np.tan(theta) * (np.outer(e[i], e[i]) + np.outer(e[j], e[j]))
                - 1j / np.sin(theta) * (np.outer(e[i], e[j]) + np.outer(e[j], e[i]))
            )
            for i, j in ((0, 1), (0, 2))
        )
        + np.outer(e[1] - e[2], e[1] - e[2]) / 100
    )
    return (np.eye(3) - 50 * y) @ np.linalg.inv(np.eye(3) + 50 * y)


# ## Results
#
# The characteristic impedance of the lines, the S-matrix at 3 GHz against the ideal divider, the
# power that the resistor takes when an output is fed, and the S-parameters over the band.

centre = sweep[f0]
result, m = (lambda run: (run[0].solve_stationary(), run[1]))(simulation(f0, fed=1))
dB = lambda x: 20 * np.log10(max(abs(x), 1e-3))
print(
    "lines (TEM ports): Z0 = "
    + ", ".join(
        f"{v['characteristic_impedance'].real:.2f}" for v in port_values(result, m)
    )
    + " ohm (Hammerstad-Jensen 50.00 ohm)"
)
print(f"at {f0 / 1e9:.1f} GHz, |S| (dB):   computed" + " " * 30 + "ideal")
for i in range(3):
    print(
        f"  row {i + 1}:  "
        + "  ".join(f"{dB(centre[i, j]):7.2f}" for j in range(3))
        + "        "
        + "  ".join(
            f"{dB(ideal(f0)[i, j]) if abs(ideal(f0)[i, j]) > 1e-12 else -np.inf:7.2f}"
            for j in range(3)
        )
    )
print(
    f"  arg S21 - arg S31 = {np.degrees(np.angle(centre[1, 0] / centre[2, 0])):.2f} deg (in phase), |S21|^2 + |S31|^2 = "
    f"{abs(centre[1, 0]) ** 2 + abs(centre[2, 0]) ** 2:.3f} (1)"
)
print(
    f"port 2 fed: to port 1 {abs(centre[0, 1]) ** 2:.3f}, reflected {abs(centre[1, 1]) ** 2:.3f}, to port 3 {abs(centre[2, 1]) ** 2:.4f}, "
    f"into the resistor and the open top {1 - np.sum(np.abs(centre.matrix[:, 1]) ** 2):.3f} of the power (ideal 0.5, 0, 0, 0.5)"
)
print(
    f"port 1 fed: lost {1 - np.sum(np.abs(centre.matrix[:, 0]) ** 2):.4f} of the power (resistor and open top; ideal 0)"
)
print(
    f"reciprocity max |S - S^T| over the band: {max(np.abs(S.matrix - S.matrix.T).max() for S in sweep.values()):.1e}"
)
print(
    "f [GHz]  |S11| dB  |S21| dB  |S22| dB  |S32| dB    ideal: |S11| dB  |S21| dB  |S22| dB  |S32| dB"
)
for f, S in sweep.items():
    print(
        f"{f / 1e9:6.2f} "
        + "".join(f"{dB(S[i, j]):9.2f} " for i, j in ((0, 0), (1, 0), (1, 1), (2, 1)))
        + "  "
        + "".join(
            f"{dB(ideal(f)[i, j]):9.2f} " for i, j in ((0, 0), (1, 0), (1, 1), (2, 1))
        )
    )

# ## Figure
#
# Left: input match, transmission, output match and isolation over the band against the ideal
# divider. Right: the electric field in the substrate at 3 GHz with port 1 fed: equal waves in both
# arms, no voltage across the resistor.

fine = np.linspace(2e9, 4e9, 121)
labels = {(0, 0): "|S11|", (1, 0): "|S21|", (1, 1): "|S22|", (2, 1): "|S32|"}
fig = plots.figure(1, 2, size=(18, 7))
plots.line(
    points={
        name: (frequencies / 1e9, [dB(S[i, j]) for S in sweep.values()])
        for (i, j), name in labels.items()
    },
    reference={
        f"ideal {name}": (fine / 1e9, [dB(ideal(f)[i, j]) for f in fine])
        for (i, j), name in labels.items()
    },
    xlabel="frequency [GHz]",
    ylabel="[dB]",
    ylimits=(-45, 2),
    title="S-parameters of the divider",
    legend="lower right",
    ax=fig[0],
)
plots.section(
    result.field(m.electric_field),
    plane=Plane(origin=(0.0, 0.0, 0.6 * height), normal=(0.0, 0.0, 1.0)),
    component=lambda E, p: np.linalg.norm(np.abs(E), axis=1),
    limits="robust",
    subdivisions=1,
    length_unit="mm",
    label="|E| [V/m]",
    title="Electric field in the substrate, 3 GHz, port 1 fed",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Change the resistor in the model (`resistance=`) to see how the isolation depends on it, or add
# a series inductance (`inductance=`) for the parasitics of a real chip resistor. In
# `microstrip_wilkinson_power_divider_geometry.py` unequal arm impedances give an unequal split.
