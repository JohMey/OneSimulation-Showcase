"""OneSimulation example: radiation impedance and end correction of a flanged pipe.
Requires the installed, activated OneSimulation runtime.

A piston drives a pipe that opens through an infinite flange into free air. The axisymmetric
pressure field is computed with quadratic elements in the r-z plane; the radiation impedance of the
open end and its end correction are compared with Rayleigh's flanged piston and with Norris and
Sheng's value for the pipe. The measured impedance then replaces the outside air in a model of the
pipe alone.
"""

# # Radiation impedance of a flanged pipe
#
# A rigid piston at the closed end of a pipe (radius $a=20$ mm, length $L=250$ mm) oscillates with
# velocity $U e^{i\omega t}$. The pipe opens through an infinite rigid flange into air. For small
# amplitudes the pressure obeys the Helmholtz equation, and since nothing depends on the angle
# around the pipe axis it is solved in the meridian plane $(r,z)$:
#
# $$\frac{1}{r}\frac{\partial}{\partial r}\Big(\frac{r}{\rho_0}\frac{\partial p}{\partial r}\Big)
#   +\frac{\partial}{\partial z}\Big(\frac{1}{\rho_0}\frac{\partial p}{\partial z}\Big)
#   +\frac{\omega^2}{\rho_0c^2}\,p=0 .$$
#
# | boundary | condition |
# |---|---|
# | piston, $z=0$ | normal velocity $U$ into the air, $\partial_n p=i\omega\rho_0U$ |
# | pipe wall and flange | rigid, $\partial_n p=0$ |
# | hemisphere of radius 0.3 m | outgoing spherical wave, $v_n=p\,(1+1/(ikr))/(\rho_0c)$ |
# | axis $r=0$ | regular (natural in the axisymmetric form) |
#
# The open end acts on the pipe like a load impedance $Z_L=p/v$. At low frequency it is a small
# resistance plus the inertia of an extra slug of air of length $\delta$, the **end correction**:
# $Z_L/\rho_0c\approx\tfrac12(ka)^2+ik\delta$.
#
# **Reference.** Lord Rayleigh (*The Theory of Sound*, Vol. II, §302) gives the impedance of a
# flat piston of radius $a$ in an infinite baffle,
#
# $$\frac{Z_P}{\rho_0c}=1-\frac{2J_1(2ka)}{2ka}+i\,\frac{2\mathbf H_1(2ka)}{2ka},\qquad
#   \delta_P=\frac{8a}{3\pi}=0.849\,a ,$$
#
# with the Struve function $\mathbf H_1$ (also L. E. Kinsler et al., *Fundamentals of Acoustics*,
# 4th ed., §7.5). The open end of a pipe is not a flat piston: its velocity peaks at the edge, and
# A. N. Norris and I. C. Sheng (*J. Sound Vib.* 135, 85-93, 1989) found $\delta=0.8216\,a$ for
# the flanged pipe at low frequency. The pipe itself is a plane-wave transmission line,
#
# $$Z_\mathrm{in}=\rho_0c\,\frac{Z_L+i\rho_0c\tan kL}{\rho_0c+iZ_L\tan kL},$$
#
# which turns the computed input impedance at the piston into the load impedance of the open end.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.numerics.fem import FiniteElement, Lagrange, RevolvedH1SpaceFactory
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
from onesim.results.comparisons import convergence
from onesim.visualization import plots, theme
from scipy.sparse.linalg import spsolve
from scipy.special import j1, struve

theme.set()

c0, rho0, velocity = 343.0, 1.204, 1e-3
impedance = rho0 * c0
frequencies = np.linspace(40.0, 2000.0, 197)
omegas = 2 * np.pi * frequencies

# ## Geometry and mesh
#
# The two STEP files hold half solids of revolution: the pipe with the hemisphere of outside air,
# and the pipe alone. `section` with the plane $y=0$ returns their meridian half-planes $r\ge0$.
# The triangles are smallest at the rim of the open end, where the velocity is singular, and grow
# towards the hemisphere. Three meshes are made, from 2.5 mm at the rim and 25 mm (a seventh of the
# shortest wavelength) far away down to a quarter of those sizes.

parts = {
    "pipe and outside air": load("flanged_pipe_radiation_impedance.step"),
    "pipe alone": load("flanged_pipe_radiation_impedance_pipe.step"),
}
planes = {name: part.section((0, 1, 0)) for name, part in parts.items()}
radius, length = planes["pipe alone"].size[:2]
rim = np.array([radius, length])
levels = (1.0, 0.5, 0.25)
meshes = {
    (name, s): TriangularMesher(
        target_size=0.025 * s,
        size_field=lambda p, s=s: np.minimum(
            0.025 * s, 0.0025 * s + 0.2 * np.linalg.norm(p[:, :2] - rim, axis=1)
        ),
    ).generate(plane)
    for name, plane in planes.items()
    for s in levels
}
finest = levels[-1]

full = parts["pipe and outside air"]
view = plots.Camera(elevation=18, azimuth=-150)
fig = plots.figure(1, 2, size=(15, 6.5))
plots.geometry(
    full,
    selection={
        "piston": full.boundaries.piston,
        "pipe wall": full.boundaries.wall,
        "flange": full.boundaries.flange,
        "outer hemisphere": full.boundaries.outer,
    },
    labels=False,
    camera=view,
    title="Half of the air volume (STEP)",
    ax=fig[0],
)
plots.mesh(
    meshes["pipe and outside air", 0.5],
    groups=("all",),
    legend=False,
    title=f"Meridian plane, middle mesh: {meshes['pipe and outside air', 0.5].cell_count} triangles "
    f"(pipe alone: {meshes['pipe alone', 0.5].cell_count})",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `RevolvedH1SpaceFactory` turns the quadratic Lagrange space of the meridian plane into an
# axisymmetric one: every volume and boundary weight carries the factor $2\pi r$. The volume forms
# come from `SampledPressureVolumeForms`, the boundary traces of a face from the space's
# `trace_data`. The outer hemisphere gets the spherical-wave admittance, and the piston load is
# $i\omega U\int\phi_i\,\mathrm dS$; `AcousticHarmonicPressure` solves the sweep.
#
# The second model is the pipe alone. Its opening carries a uniform admittance $1/Z_L$; the
# impedance is either the one measured in the first model or Rayleigh's piston.


def solve_pipe(name, level, end_face, open_end):
    """Input impedance at the piston over the sweep; open_end(omega, points) is the admittance of end_face."""
    plane, mesh = planes[name], meshes[name, level]
    space = FiniteElement(
        element=Lagrange(order=2), provider=RevolvedH1SpaceFactory()
    ).bind_space(mesh)
    volume = ConstantPressureForms(
        *SampledPressureVolumeForms(
            size=space.dof_count,
            samples=lambda: pressure_space_volume_samples(space),
            inverse_density=lambda omega, sample: 1 / rho0,
            compressibility=lambda omega, sample: 1 / impedance / c0,
        )(0.0)
    )
    traces = {
        face: [
            PressureTraceSamples(space.cell_dofs[owner], points, weights, basis)
            for owner, points, weights, _, basis, _ in map(
                space.trace_data, mesh.facets(getattr(plane.boundaries, face))
            )
        ]
        for face in ("piston", end_face)
    }
    piston = SampledPressureAdmittanceForms(
        size=space.dof_count,
        samples=lambda: iter(traces["piston"]),
        admittance=lambda omega, sample: 1.0,
    )(0.0) @ np.ones(space.dof_count)
    radiation = SampledPressureAdmittanceForms(
        size=space.dof_count,
        samples=lambda: iter(traces[end_face]),
        admittance=lambda omega, sample: open_end(omega, sample.points),
    )
    result = AcousticHarmonicPressure(
        operator=ComplexPressurePencil(
            volume, boundary=PressureAdmittanceBoundary(radiation)
        ),
        load=lambda omega: 1j * omega * velocity * piston,
        angular_frequencies=omegas,
        linear_solver=spsolve,
    ).run()
    return dict(
        space=space,
        mesh=mesh,
        pressure=result.pressure,
        dofs=space.dof_count,
        z_in=result.pressure @ piston / (np.pi * radius**2) / velocity,
    )


spherical_wave = (
    lambda omega, points: (
        1 + c0 / (1j * omega.real * np.linalg.norm(points - [0.0, length], axis=1))
    )
    / impedance
)
study = {
    s: solve_pipe("pipe and outside air", s, "outer", spherical_wave) for s in levels
}
outside = study[finest]
k = omegas / c0
line_load = (
    lambda z_in: impedance
    * (z_in - 1j * impedance * np.tan(k * length))
    / (impedance - 1j * z_in * np.tan(k * length))
)
load_impedance = line_load(outside["z_in"])
rayleigh = impedance * (
    1
    - j1(2 * k * radius) / (k * radius)
    + 1j * struve(1, 2 * k * radius) / (k * radius)
)
lumped = {
    "measured impedance": solve_pipe(
        "pipe alone",
        finest,
        "opening",
        lambda omega, points: 1 / np.interp(omega.real, omegas, load_impedance),
    ),
    "Rayleigh piston": solve_pipe(
        "pipe alone",
        finest,
        "opening",
        lambda omega, points: 1 / np.interp(omega.real, omegas, rayleigh),
    ),
}

# ## Results
#
# The load impedance of the open end, divided by $\rho_0c$, against Rayleigh's piston; the end
# correction $\delta=\mathrm{Im}\,Z_L/(\rho_0ck)$ against both references, with the mesh study; and
# the input impedance of the two pipe-alone models against the full model. The measured impedance
# reproduces the full model; Rayleigh's flat piston has a 3 % longer end correction, which shifts
# the sharp resonance peaks of $|Z_\mathrm{in}|$ by a few hertz and so changes the curve by tens of
# percent right at the peaks.

delta = load_impedance.imag / (impedance * k) / radius
print(
    f"unknowns: pipe and outside air {outside['dofs']}, pipe alone {lumped['Rayleigh piston']['dofs']}"
)
print(
    "  f [Hz]    ka      Re Z_L/rho c   Rayleigh    Im Z_L/rho c   Rayleigh    delta/a   Rayleigh delta/a"
)
for f in (100, 200, 400, 800, 1200, 1600, 2000):
    i = np.argmin(np.abs(frequencies - f))
    print(
        f"{frequencies[i]:7.0f}  {k[i] * radius:6.3f}   {load_impedance[i].real / impedance:11.5f}  {rayleigh[i].real / impedance:9.5f}"
        f"   {load_impedance[i].imag / impedance:11.5f}  {rayleigh[i].imag / impedance:9.5f}   {delta[i]:7.4f}"
        f"   {rayleigh[i].imag / (impedance * k[i] * radius):7.4f}"
    )
low = frequencies <= 100
corrections = [
    np.mean(line_load(study[s]["z_in"]).imag[low] / (impedance * k[low] * radius))
    for s in levels
]
trend = convergence(corrections, list(levels))
print(
    "end correction below 100 Hz on the three meshes: "
    + ", ".join(
        f"{value:.4f} a ({study[s]['dofs']} unknowns)"
        for s, value in zip(levels, corrections)
    )
)
print(
    f"finest mesh {corrections[-1]:.4f} a, extrapolated {trend.extrapolated:.4f} a (observed order {trend.order:.1f}); "
    f"Norris and Sheng 0.8216 a: {100 * (corrections[-1] / 0.8216 - 1):+.2f} % (finest), "
    f"{100 * (trend.extrapolated / 0.8216 - 1):+.2f} % (extrapolated); flat piston 8/(3 pi) = {8 / (3 * np.pi):.4f} a"
)
for name, run in lumped.items():
    print(
        f"pipe alone with the {name}: largest deviation of |Z_in| from the full model "
        f"{np.max(np.abs(run['z_in'] - outside['z_in']) / np.abs(outside['z_in'])):.2e} (relative, over the sweep)"
    )
peaks = [
    frequencies[i]
    for i in range(1, len(frequencies) - 1)
    if abs(outside["z_in"][i])
    > max(abs(outside["z_in"][i - 1]), abs(outside["z_in"][i + 1]))
]
quarter_wave = [
    (2 * n - 1) * c0 / (4 * (length + 0.8216 * radius))
    for n in range(1, len(peaks) + 1)
]
print(
    "resonances of |Z_in| (sweep step 10 Hz) [Hz]:",
    ", ".join(f"{p:.0f}" for p in peaks),
    "| quarter-wave with delta = 0.8216 a:",
    ", ".join(f"{q:.0f}" for q in quarter_wave),
)

# ## Figure
#
# Top: the sound pressure level in the meridian plane at the first resonance, and the load
# impedance of the open end. Bottom: the end correction over frequency and the input impedance at
# the piston for the full model and the two pipe-alone models.

first = np.argmin(np.abs(frequencies - peaks[0]))
x, y = np.asarray(outside["mesh"].coordinates).T
level = 20 * np.log10(
    np.abs(
        np.asarray(
            outside["space"].evaluate(
                outside["pressure"][first], outside["mesh"].coordinates
            )
        )
    )
    / 2e-5
)
fig = plots.figure(2, 2, size=(16, 12))
plots.heatmap(
    x,
    y,
    level,
    triangles=np.asarray(outside["mesh"].cells),
    limits=(level.max() - 50, level.max()),
    contours=10,
    label="sound pressure level [dB]",
    xlabel="r [m]",
    ylabel="z [m]",
    title=f"First resonance, {frequencies[first]:.0f} Hz, piston velocity 1 mm/s",
    ax=fig[0],
)
plots.line(
    k * radius,
    {
        "Re Z_L / rho c, simulation": load_impedance.real / impedance,
        "Im Z_L / rho c, simulation": load_impedance.imag / impedance,
    },
    reference={
        "Re Z_L / rho c, simulation": rayleigh.real / impedance,
        "Im Z_L / rho c, simulation": rayleigh.imag / impedance,
    },
    xlabel="ka",
    ylabel="normalised impedance",
    title="Open end (dashed: Rayleigh's flat piston)",
    legend="upper left",
    ax=fig[1],
)
plots.line(
    k * radius,
    {"simulation": delta},
    reference={"flat piston (Rayleigh)": rayleigh.imag / (impedance * k * radius)},
    hlines={"Norris and Sheng, 0.8216 a": 0.8216},
    xlabel="ka",
    ylabel="end correction delta / a",
    title="End correction of the flanged pipe",
    legend="lower left",
    ax=fig[2],
)
plots.line(
    frequencies,
    {
        "pipe and outside air": np.abs(outside["z_in"]) / impedance,
        "pipe alone, measured impedance": np.abs(lumped["measured impedance"]["z_in"])
        / impedance,
    },
    reference={
        "pipe alone, Rayleigh piston": np.abs(lumped["Rayleigh piston"]["z_in"])
        / impedance
    },
    scale="logy",
    xlabel="frequency [Hz]",
    ylabel="|Z_in| / rho c",
    title="Input impedance at the piston",
    legend="lower right",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Remove the flange in `flanged_pipe_radiation_impedance_geometry.py` (a thin pipe wall standing
# in the air) and write the STEP file again: the end correction drops towards Levine and
# Schwinger's 0.6133 a. Or move the hemisphere closer to the opening to see where the
# spherical-wave admittance, exact only for the monopole part of the far field, starts to cost
# accuracy.
