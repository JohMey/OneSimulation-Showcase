"""OneSimulation example: lumping a side-branch resonator into an impedance boundary.
Requires the installed, activated OneSimulation runtime.

A Helmholtz resonator on the side of a duct removes one frequency from the sound that passes. The
resonator alone is computed once and condensed into its input impedance; that impedance then
replaces it as a boundary condition on the duct. The three-dimensional models are read from STEP
files and solved with curved quadratic tetrahedra; full and lumped transmission loss are compared
with each other and with the classical side-branch formula.
"""

# # Lumping a side-branch resonator into an impedance
#
# Detailed sub-components - a resonator, a perforate, a liner - make acoustic models large. If a
# component touches the rest of the system only through one small opening, it can be computed
# once and replaced by its impedance $Z(\omega)=\bar p/U$ on that opening. Here a Helmholtz
# resonator (neck 10 mm in radius and 30 mm long, cavity 30 mm in radius and 50 mm long) sits on
# the side of a duct of radius 25 mm.
#
# The workflow has three models, all in half of the air (cut in the plane of symmetry $y=0$):
#
# 1. **resonator alone**, driven by a uniform velocity $U$ on the *port*, a section of the neck
#    5 mm above the duct wall: gives $Z(\omega)$;
# 2. **lumped model**, the duct with a 5 mm stub of the neck whose end carries the admittance
#    $1/Z(\omega)$;
# 3. **full model**, duct and resonator together, as the check.
#
# | boundary | condition |
# |---|---|
# | duct inlet | incoming plane wave $p_\mathrm{in}$, reflected wave absorbed: $v_n=(p-2p_\mathrm{in})/(\rho_0c)$ |
# | duct outlet | anechoic, $v_n=p/(\rho_0c)$ |
# | port (resonator alone) | uniform normal velocity $U$ into the resonator |
# | port (lumped model) | impedance, $v_n=p/Z(\omega)$ |
# | all walls, symmetry plane | rigid, $\partial p/\partial n=0$ |
#
# **Reference.** For plane waves a side branch of acoustic impedance $Z_b$ on a duct of area $S$
# transmits (L. E. Kinsler et al., *Fundamentals of Acoustics*, 4th ed., §10.9)
#
# $$TL=20\log_{10}\Big|1+\frac{\rho_0c}{2SZ_b}\Big|,\qquad
#   Z_b=\frac{i\omega\rho_0L'}{S_n}+\frac{\rho_0c^2}{i\omega V},$$
#
# for a lumped resonator with neck area $S_n$, cavity volume $V$ and effective neck length $L'$
# (the neck plus end corrections of $0.48\sqrt{S_n}(1-1.25a/R_\mathrm{cavity})$ inside, after
# U. Ingard, *J. Acoust. Soc. Am.* 25, 1953, and about $0.82a$ towards the duct). The transmission
# loss peaks at $f_0=\tfrac c{2\pi}\sqrt{S_n/(L'V)}$; the lumped estimate is good to a few per cent.

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
duct_radius, neck_radius, neck_length, cavity_radius, cavity_length = (
    0.025,
    0.010,
    0.030,
    0.030,
    0.050,
)
frequencies = np.arange(100.0, 1001.0, 10.0)
omegas = 2 * np.pi * frequencies

# ## Geometry and mesh
#
# `load` reads the three half models with their named faces. Curved quadratic tetrahedra of 7 mm
# (a fiftieth of the shortest wavelength) fill them.

parts = {
    name: load(f"resonator_impedance_lumping{suffix}.step")
    for name, suffix in (("full", ""), ("resonator", "_resonator"), ("lumped", "_duct"))
}
meshes = {
    name: TetrahedralMesher(target_size=0.007, geometry_order=2).generate(part)
    for name, part in parts.items()
}

full = parts["full"]
view = plots.Camera(elevation=30, azimuth=-50)
fig = plots.figure(1, 3, size=(21, 6.5))
plots.geometry(
    full,
    selection={
        "inlet": full.boundaries.inlet,
        "outlet": full.boundaries.outlet,
        "walls": full.boundaries.walls,
        "symmetry plane": full.boundaries.symmetry,
    },
    labels=False,
    camera=view,
    title="Full model: duct with resonator (STEP)",
    ax=fig[0],
)
plots.mesh(
    meshes["resonator"],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Resonator alone: {meshes['resonator'].cell_count} tetrahedra",
    ax=fig[1],
)
plots.mesh(
    meshes["lumped"],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Lumped duct: {meshes['lumped'].cell_count} tetrahedra",
    ax=fig[2],
)
plots.show()

# ## Model
#
# Quadratic Lagrange elements; `SampledPressureVolumeForms` integrates the volume forms, and the
# traces of inlet, outlet and port come from the space's `trace_data`. `AcousticHarmonicPressure`
# sweeps 100 Hz to 1 kHz in each model.


def solve_model(name, admittances, load):
    """Pressure over the sweep; admittances maps face names to Y(omega), load(traces, omega) is the drive."""
    mesh, faces = meshes[name], parts[name].boundaries
    space = FiniteElement(element=Lagrange(order=2)).bind_space(mesh)
    volume = ConstantPressureForms(
        *SampledPressureVolumeForms(
            size=space.dof_count,
            samples=lambda: pressure_space_volume_samples(space),
            inverse_density=lambda omega, sample: 1 / rho0,
            compressibility=lambda omega, sample: 1 / impedance / c0,
        )(0.0)
    )
    traces = {
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
        for face in set(admittances) | {"inlet", "port"}
        if face in dir(faces)
    }
    boundary = PressureAdmittanceBoundary(
        lambda omega: sum(
            admittance(omega) * traces[face] for face, admittance in admittances.items()
        )
    )
    result = AcousticHarmonicPressure(
        operator=ComplexPressurePencil(volume, boundary=boundary),
        angular_frequencies=omegas,
        load=lambda omega: load(traces, omega, space.dof_count),
        linear_solver=spsolve,
    ).run()
    return dict(
        space=space,
        mesh=mesh,
        pressure=result.pressure,
        traces=traces,
        dofs=space.dof_count,
    )


transmission_loss = lambda run: 10 * np.log10(
    incident**2
    * run["traces"]["inlet"].sum()
    / np.real(
        np.sum(
            run["pressure"].conj() * (run["traces"]["outlet"] @ run["pressure"].T).T,
            axis=1,
        )
    )
)
duct_ports = {
    "inlet": lambda omega: 1 / impedance,
    "outlet": lambda omega: 1 / impedance,
}
plane_wave = (
    lambda traces, omega, size: 2j
    * omega
    * incident
    / impedance
    * (traces["inlet"] @ np.ones(size))
)

velocity = 1e-3
resonator = solve_model(
    "resonator",
    {"port": lambda omega: 0.0},
    lambda traces, omega, size: 1j
    * omega
    * velocity
    * (traces["port"] @ np.ones(size)),
)
port_impedance = (
    resonator["pressure"]
    @ (resonator["traces"]["port"] @ np.ones(resonator["dofs"]))
    / resonator["traces"]["port"].sum()
    / velocity
)
lumped = solve_model(
    "lumped",
    {
        **duct_ports,
        "port": lambda omega: 1 / np.interp(omega.real, omegas, port_impedance),
    },
    plane_wave,
)
complete = solve_model("full", duct_ports, plane_wave)

# ## Results
#
# The resonance of the resonator alone (where its input reactance crosses zero), the transmission
# loss of the lumped and of the full model, and the side-branch formula.

S, S_n, V = (
    np.pi * duct_radius**2,
    np.pi * neck_radius**2,
    np.pi * cavity_radius**2 * cavity_length,
)
effective = (
    neck_length
    + 0.48 * np.sqrt(S_n) * (1 - 1.25 * neck_radius / cavity_radius)
    + 0.82 * neck_radius
)
branch = 1j * omegas * rho0 * effective / S_n + rho0 * c0**2 / (1j * omegas * V)
formula = 20 * np.log10(np.abs(1 + impedance / (2 * S * branch)))
tl = {
    "lumped model": transmission_loss(lumped),
    "full model": transmission_loss(complete),
}
crossing = np.flatnonzero(np.diff(np.sign(port_impedance.imag)))[0]
f_port = np.interp(
    0.0,
    port_impedance.imag[crossing : crossing + 2],
    frequencies[crossing : crossing + 2],
)
print(
    f"unknowns: resonator alone {resonator['dofs']}, lumped duct {lumped['dofs']}, full model {complete['dofs']}"
)
print(
    f"resonator alone: input reactance crosses zero at {f_port:.1f} Hz (5 mm of neck are left in the duct model)"
)
print(
    f"lumped formula: effective neck length {1e3 * effective:.1f} mm, f0 = {c0 / (2 * np.pi) * np.sqrt(S_n / (effective * V)):.1f} Hz"
)
for name, values in tl.items():
    peak = np.argmax(values)
    print(
        f"{name}: peak transmission loss {values[peak]:.1f} dB at {frequencies[peak]:.0f} Hz (sweep step 10 Hz)"
    )
print(
    f"formula: peak {formula.max():.1f} dB at {frequencies[np.argmax(formula)]:.0f} Hz"
)
print(
    f"lumped against full model: largest TL difference {np.max(np.abs(tl['lumped model'] - tl['full model'])):.3f} dB, "
    f"away from the peak (TL < 20 dB) {np.max(np.abs(tl['lumped model'] - tl['full model'])[tl['full model'] < 20]):.3f} dB"
)
print(
    f"full model against the formula, TL < 20 dB: largest difference {np.max(np.abs(tl['full model'] - formula)[tl['full model'] < 20]):.2f} dB"
)

# ## Figure
#
# Left: the normalised input impedance of the resonator. Right: the transmission loss of the full
# and the lumped model and the side-branch formula.

fig = plots.figure(1, 2, size=(17, 6.5))
plots.line(
    frequencies,
    {
        "Re Z / rho c": port_impedance.real / impedance,
        "Im Z / rho c": port_impedance.imag / impedance,
    },
    hlines=[0.0],
    ylimits=(-1.5, 1.0),
    xlabel="frequency [Hz]",
    ylabel="normalised impedance",
    title="Input impedance of the resonator at the port",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    frequencies,
    tl,
    reference={"side-branch formula": formula},
    xlabel="frequency [Hz]",
    ylabel="transmission loss [dB]",
    title="Transmission loss of the duct",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# The same route condenses any sub-component that meets the system through one opening: a
# perforated plate, an absorbing liner, a loudspeaker. Several openings lead to an impedance
# matrix instead of one number per frequency. Change the resonator in
# `resonator_impedance_lumping_geometry.py` and write the STEP files again: only the resonator-alone
# model has to be solved again.
