"""OneSimulation example: a frequency sweep in octave bands with a new mesh for every band.
Requires the installed, activated OneSimulation runtime.

A mesh fine enough for the highest frequency of a sweep wastes unknowns at the low end. Here the
transmission loss of an expansion-chamber muffler, read from a STEP file, is swept from 88 Hz to
2.8 kHz in six octave bands, and each band gets its own mesh of curved quadratic tetrahedra, sized
to its upper band edge. The result is compared with one fine mesh for the whole sweep and with the
plane-wave formula.
"""

# # Remeshing per octave band
#
# The element size a frequency-domain acoustic model needs scales with the wavelength: about six
# quadratic elements per wavelength keep the error small. A sweep over several octaves therefore
# needs a mesh 32 times finer at the top than at the bottom; using that mesh everywhere costs far
# more unknowns than needed. Remeshing per band follows the wavelength instead: band $b$ with upper
# edge $f_b$ gets elements of size $h_b=c/(6f_b)$, limited by what the geometry itself requires.
#
# The test object is a concentric expansion chamber (radius 60 mm, length 250 mm) between pipes of
# radius 20 mm. A quarter of the air is computed, cut in the two symmetry planes.
#
# | boundary | condition |
# |---|---|
# | inlet | incoming plane wave $p_\mathrm{in}$, reflected wave absorbed: $v_n=(p-2p_\mathrm{in})/(\rho_0c)$ |
# | outlet | anechoic, $v_n=p/(\rho_0c)$ |
# | walls and symmetry planes | rigid, $\partial p/\partial n=0$ |
#
# **References.** Concentric pipes excite only axisymmetric modes, and the first of them cuts on at
# $3.8317\,c/(2\pi R)=3486$ Hz, above the sweep. At low frequency the plane-wave formula holds
# (L. E. Kinsler et al., *Fundamentals of Acoustics*, 4th ed., §10.10; M. L. Munjal, *Acoustics of
# Ducts and Mufflers*, 2nd ed., §3.5):
#
# $$TL=10\log_{10}\Big[1+\tfrac14\big(m-\tfrac1m\big)^2\sin^2kL\Big],\qquad m=(R/r)^2 .$$
#
# With rising frequency the evanescent higher modes at the two area steps lengthen the chamber
# acoustically and shift the domes of the transmission loss (A. Selamet and P. M. Radavich,
# *J. Sound Vib.* 201, 407, 1997), so the formula is the check of the low bands. The accuracy of the
# band meshes in every band is measured against one fine mesh, half the size of the top band's.
#
# The octave bands follow IEC 61260: centre frequencies 125 Hz to 4 kHz halved, here 125 Hz to
# 2 kHz, band edges at $f_c/\sqrt2$ and $\sqrt2 f_c$.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TetrahedralMesher
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.physics.acoustics import (
    AcousticHarmonicPressure,
    ComplexPressurePencil,
    ConstantPressureForms,
    PressureAdmittanceBoundary,
    PressureTraceSamples,
    SampledPressureAdmittanceForms,
    SampledPressureVolumeForms,
    pressure_space_volume_samples,
)
from onesim.visualization import plots, theme
from scipy.sparse.linalg import spsolve

theme.set()

c0, rho0, incident = 343.0, 1.204, 1.0
impedance = rho0 * c0
chamber_radius, chamber_length, pipe_radius = 0.06, 0.25, 0.02
centres = 125.0 * 2.0 ** np.arange(5)
bands = {fc: np.geomspace(fc / np.sqrt(2), fc * np.sqrt(2), 13) for fc in centres}

# ## Geometry and meshes
#
# `load` returns the quarter muffler with its named faces. Each band gets a mesh of curved quadratic
# tetrahedra with $h_b=c/(6f_b)$, at most 100 mm; a curved element may turn by 60° on the round walls.
# The fine reference mesh uses half the size of the top band throughout.

muffler = load("octave_band_remeshing.step")
faces = muffler.boundaries
size = {fc: min(0.100, c0 / (6 * np.sqrt(2) * fc)) for fc in centres}
fine = size[centres[-1]] / 2
meshes = {
    h: TetrahedralMesher(
        target_size=h, geometry_order=2, curvature_angle=np.pi / 3
    ).generate(muffler)
    for h in sorted(set(size.values()) | {fine})
}

view = plots.Camera(elevation=25, azimuth=-60)
fig = plots.figure(1, 3, size=(21, 7))
plots.geometry(
    muffler,
    selection={
        "inlet": faces.inlet,
        "outlet": faces.outlet,
        "walls": faces.walls,
        "symmetry planes": faces.symmetry,
    },
    labels=False,
    camera=view,
    title="Quarter of the muffler (STEP)",
    ax=fig[0],
)
for panel, fc in zip((fig[1], fig[2]), (centres[0], centres[-1])):
    plots.mesh(
        meshes[size[fc]],
        groups=("all",),
        legend=False,
        camera=view,
        title=f"Mesh of the {fc:.0f} Hz band: {meshes[size[fc]].cell_count} tetrahedra",
        ax=panel,
    )
plots.show()

# ## Model
#
# Quadratic Lagrange elements; `SampledPressureVolumeForms` integrates the volume forms and the
# port traces come from the space's `trace_data`. One function sets up the model on a mesh and
# sweeps the given frequencies with `AcousticHarmonicPressure`; it is called once per band with the
# band's mesh, and once per band with the fine mesh.


def solve_band(h, frequencies):
    """Transmission loss at the frequencies on the mesh of size h, and the number of unknowns."""
    mesh = meshes[h]
    space = FiniteElement(element=Lagrange(order=2)).bind_space(mesh)
    volume = ConstantPressureForms(
        *SampledPressureVolumeForms(
            size=space.dof_count,
            samples=lambda: pressure_space_volume_samples(space),
            inverse_density=lambda omega, sample: 1 / rho0,
            compressibility=lambda omega, sample: 1 / impedance / c0,
        )(0.0)
    )
    ports = {
        face: SampledPressureAdmittanceForms(
            size=space.dof_count,
            admittance=lambda omega, sample: 1.0,
            samples=lambda face=face: iter(
                [
                    PressureTraceSamples(space.cell_dofs[owner], points, weights, basis)
                    for owner, points, weights, _, basis, _ in map(
                        space.trace_data, mesh.facets(getattr(faces, face))
                    )
                ]
            ),
        )(0.0)
        for face in ("inlet", "outlet")
    }
    drive = ports["inlet"] @ np.ones(space.dof_count)
    pressure = (
        AcousticHarmonicPressure(
            operator=ComplexPressurePencil(
                volume,
                boundary=PressureAdmittanceBoundary(
                    lambda omega: (ports["inlet"] + ports["outlet"]) / impedance
                ),
            ),
            angular_frequencies=2 * np.pi * frequencies,
            load=lambda omega: 2j * omega * incident / impedance * drive,
            linear_solver=spsolve,
        )
        .run()
        .pressure
    )
    transmitted = np.real(
        np.sum(pressure.conj() * (ports["outlet"] @ pressure.T).T, axis=1)
    )
    return dict(
        tl=10 * np.log10(incident**2 * drive.sum() / transmitted), dofs=space.dof_count
    )


remeshed = {fc: solve_band(size[fc], bands[fc]) for fc in centres}
reference = {fc: solve_band(fine, bands[fc]) for fc in centres}

# ## Results
#
# Per band: element size and unknowns of the band mesh, its deviation from the fine mesh and from
# the plane-wave formula, and the work in unknowns times frequencies.

m = (chamber_radius / pipe_radius) ** 2
formula = lambda f: 10 * np.log10(
    1 + 0.25 * (m - 1 / m) ** 2 * np.sin(2 * np.pi * f / c0 * chamber_length) ** 2
)
print(f"fine mesh: h = {1e3 * fine:.1f} mm, {reference[centres[0]]['dofs']} unknowns")
print(
    " band [Hz]   h [mm]   unknowns   max |TL - TL fine| [dB]   max |TL - formula| [dB]   fine mesh: max |TL - formula| [dB]"
)
for fc in centres:
    print(
        f"{fc:8.0f}   {1e3 * size[fc]:6.1f}   {remeshed[fc]['dofs']:8d}   {np.max(np.abs(remeshed[fc]['tl'] - reference[fc]['tl'])):23.3f}"
        f"   {np.max(np.abs(remeshed[fc]['tl'] - formula(bands[fc]))):23.3f}   {np.max(np.abs(reference[fc]['tl'] - formula(bands[fc]))):34.3f}"
    )
work = sum(remeshed[fc]["dofs"] * len(bands[fc]) for fc in centres), sum(
    reference[fc]["dofs"] * len(bands[fc]) for fc in centres
)
print(
    f"work (unknowns x frequencies): remeshed {work[0]}, one fine mesh {work[1]}, ratio {work[1] / work[0]:.1f}"
)
print(
    f"largest difference between remeshed and fine sweep: "
    f"{max(np.max(np.abs(remeshed[fc]['tl'] - reference[fc]['tl'])) for fc in centres):.3f} dB"
)

# ## Figure
#
# Left: the transmission loss over the whole sweep, band by band, with the fine-mesh sweep and the
# formula. Right: the unknowns used in each band.

fig = plots.figure(1, 2, size=(17, 6.5))
plots.line(
    y={f"{fc:.0f} Hz band mesh": (bands[fc], remeshed[fc]["tl"]) for fc in centres},
    points={
        "one fine mesh": (
            np.concatenate([bands[fc][::3] for fc in centres]),
            np.concatenate([reference[fc]["tl"][::3] for fc in centres]),
        )
    },
    reference={
        "plane-wave formula": (
            np.geomspace(80, 2900, 600),
            formula(np.geomspace(80, 2900, 600)),
        )
    },
    scale="logx",
    xlabel="frequency [Hz]",
    ylabel="transmission loss [dB]",
    title="Transmission loss, remeshed per octave band",
    legend="upper left",
    ax=fig[0],
)
plots.bars(
    [f"{fc:.0f} Hz" for fc in centres],
    [remeshed[fc]["dofs"] for fc in centres],
    scale="log",
    ylabel="pressure unknowns",
    title=f"Unknowns per band (fine mesh: {reference[centres[0]]['dofs']})",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Switch to third-octave bands (`2.0 ** (np.arange(16) / 3)`) for a finer staircase of meshes, or
# lower the elements per wavelength to see where the band meshes start to lose accuracy at their
# upper edges. The same pattern applies to any geometry in a STEP file, for example a room or a
# car cabin.
