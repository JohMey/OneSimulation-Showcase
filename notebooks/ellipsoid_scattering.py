"""OneSimulation example: a sound wave scattered by a rigid triaxial ellipsoid.
Requires the installed, activated OneSimulation runtime.

A plane wave meets a rigid ellipsoid with three different semi-axes, obliquely, so that nothing in
the problem is symmetric. The scattered field is computed in three dimensions with curved quadratic
tetrahedra at ka = 0.25, 0.5 and 1, each frequency on its own STEP model and mesh, and compared
with the low-frequency scattering law of a rigid body, built from the ellipsoid's volume and its
added-mass tensor.
"""

# # Sound scattered by a rigid ellipsoid
#
# A plane wave $p_i=p_0e^{-ik\,\hat{\mathbf d}\cdot\mathbf x}$ with direction
# $\hat{\mathbf d}=(1,1,1)/\sqrt3$ falls on a rigid ellipsoid with semi-axes $a_1=100$ mm,
# $a_2=60$ mm and $a_3=40$ mm. The scattered wave $p_s$ solves the Helmholtz equation outside the
# body, with the condition that the total normal velocity on the body vanishes. It is computed out to
# an outer sphere of radius $R$ with $kR=8$ for each frequency.
#
# | boundary | condition |
# |---|---|
# | ellipsoid | rigid: $\partial_np_s=-\partial_np_i$ |
# | outer sphere $r=R$ | outgoing spherical wave, $v_n=p_s\,(1+1/(ikR))/(\rho_0c)$ |
#
# The outer condition is exact for the monopole part of the scattered wave and reflects the dipole
# part by about $1/(kR)^2\approx1.5$ %; a frequency-domain perfectly matched layer would remove that
# but is not used here.
#
# **Reference.** For a body small compared with the wavelength the scattered wave is a monopole,
# because the body displaces fluid, plus a dipole, because the fluid has to flow around it
# (Lord Rayleigh, *Phil. Mag.* 44, 28, 1897; G. Dassios and R. Kleinman, *Low Frequency Scattering*,
# Oxford 2000, ch. 3):
#
# $$p_s(\mathbf x)\simeq p_0k^2\,\frac{e^{-ikr}}{4\pi r}\Big[-V+\Big(1-\frac{i}{kr}\Big)\,
#   \hat{\mathbf r}\cdot(V\mathbf I+\mathbf A)\cdot\hat{\mathbf d}\Big],$$
#
# with the volume $V$ and the added-mass tensor $\mathbf A$ of the body (per unit fluid density).
# For an ellipsoid $\mathbf A$ is diagonal in the axes with $A_j=V\alpha_j/(2-\alpha_j)$ and
# $\alpha_j=a_1a_2a_3\int_0^\infty\mathrm d\lambda/\big((a_j^2+\lambda)\sqrt{(a_1^2+\lambda)(a_2^2+\lambda)(a_3^2+\lambda)}\big)$
# (H. Lamb, *Hydrodynamics*, 6th ed., §114). The law is the first term of an expansion in $ka$; the
# simulation must approach it as the frequency falls, with a deviation that shrinks like $(ka)^2$.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import LocalRefinement, TetrahedralMesher, with_order
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
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve

theme.set()

c0, rho0, amplitude = 343.0, 1.204, 1.0
impedance = rho0 * c0
axes = np.array([0.10, 0.06, 0.04])
direction = np.ones(3) / np.sqrt(3)
cases = {
    0.25: "ellipsoid_scattering_ka025.step",
    0.5: "ellipsoid_scattering_ka050.step",
    1.0: "ellipsoid_scattering.step",
}

# ## Geometry and meshes
#
# Each STEP file holds the air between the ellipsoid and an outer sphere of radius $8/k$. The
# tetrahedra are a sixth of the wavelength far away and 12 mm on the ellipsoid; `with_order` lifts
# them to quadratic cells whose edge nodes lie on the curved faces.

parts = {ka: load(name) for ka, name in cases.items()}
meshes = {
    ka: with_order(
        TetrahedralMesher(
            target_size=2 * np.pi * axes[0] / ka / 6,
            refinements=(
                LocalRefinement(region=part.boundaries.body, target_size=0.012),
            ),
        ).generate(part),
        2,
    )
    for ka, part in parts.items()
}

part = parts[1.0]
view = plots.Camera(elevation=25, azimuth=-50)
fig = plots.figure(1, 2, size=(15, 7))
plots.geometry(
    part,
    selection={"ellipsoid": part.boundaries.body, "outer sphere": part.boundaries.air},
    labels=False,
    camera=view,
    title="Air around the ellipsoid, ka = 1 (STEP)",
    ax=fig[0],
)
plots.mesh(
    meshes[1.0],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"ka = 1: {meshes[1.0].cell_count} curved quadratic tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Quadratic Lagrange elements; `SampledPressureVolumeForms` integrates the volume forms and the
# traces of ellipsoid and outer sphere come from the space's `trace_data`. The outer sphere carries
# the admittance, the ellipsoid the load $-\int\rho_0^{-1}\partial_np_i\,\phi_i\,\mathrm dS$ with
# the normal of the ellipsoid. `AcousticHarmonicPressure` solves each frequency on its own mesh.

incident = lambda k, points: amplitude * np.exp(-1j * k * points @ direction)
body_normal = (
    lambda points: points / axes**2 / np.linalg.norm(points / axes**2, axis=1)[:, None]
)
incident_flux = (
    lambda k, points: (body_normal(points) @ direction) * 1j * k * incident(k, points)
)


def solve_ellipsoid(ka):
    """Scattered pressure at one frequency on its mesh."""
    mesh, faces, k = meshes[ka], parts[ka].boundaries, ka / axes[0]
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
        for face in ("body", "air")
    }
    outer = parts[ka].size[0] / 2
    pressure = (
        AcousticHarmonicPressure(
            operator=ComplexPressurePencil(
                volume,
                boundary=PressureAdmittanceBoundary(
                    lambda omega: (1 + 1 / (1j * k * outer)) / impedance * traces["air"]
                ),
            ),
            angular_frequencies=[c0 * k],
            load=lambda omega: -traces["body"]
            @ incident_flux(k, space.dof_coordinates)
            / rho0,
            linear_solver=spsolve,
        )
        .run()
        .pressure[0]
    )
    return dict(
        space=space, mesh=mesh, pressure=pressure, dofs=space.dof_count, outer=outer
    )


runs = {ka: solve_ellipsoid(ka) for ka in cases}

# ## Reference
#
# Volume, Lamb's coefficients and the low-frequency law.

volume = 4 * np.pi / 3 * np.prod(axes)
alpha = np.array(
    [
        np.prod(axes)
        * quad(lambda s: 1 / ((a**2 + s) * np.sqrt(np.prod(axes**2 + s))), 0, np.inf)[0]
        for a in axes
    ]
)
added = volume * alpha / (2 - alpha)
rayleigh = (
    lambda k, points: amplitude
    * k**2
    * np.exp(-1j * k * np.linalg.norm(points, axis=1))
    / (4 * np.pi * np.linalg.norm(points, axis=1))
    * (
        -volume
        + (1 - 1j / (k * np.linalg.norm(points, axis=1)))
        * (
            (points / np.linalg.norm(points, axis=1)[:, None])
            @ ((volume + added) * direction)
        )
    )
)
print(
    f"V = {1e6 * volume:.1f} cm3; Lamb coefficients alpha = {np.round(alpha, 4)} (sum {alpha.sum():.4f}); "
    f"added mass / V = {np.round(added / volume, 4)}"
)

# ## Results
#
# The scattered pressure on 400 points spread over a sphere of radius $R/2$ around the body, against
# the low-frequency law.

golden = np.pi * (3 - np.sqrt(5)) * np.arange(400)
heights = 1 - (np.arange(400) + 0.5) / 200
directions = np.column_stack(
    (
        np.sqrt(1 - heights**2) * np.cos(golden),
        np.sqrt(1 - heights**2) * np.sin(golden),
        heights,
    )
)
deviation = {}
for ka, run in runs.items():
    k = ka / axes[0]
    points = run["outer"] / 2 * directions
    computed, law = np.asarray(
        run["space"].evaluate(run["pressure"], points)
    ), rayleigh(k, points)
    deviation[ka] = np.max(np.abs(computed - law)) / np.max(np.abs(law))
    print(
        f"ka = {ka:.2f} ({run['dofs']} unknowns, R = {run['outer']:.1f} m): largest deviation from the low-frequency law "
        f"{deviation[ka]:.3f} of its maximum; RMS |p_s| {np.sqrt(np.mean(np.abs(computed) ** 2)):.3e} Pa "
        f"(law {np.sqrt(np.mean(np.abs(law) ** 2)):.3e} Pa)"
    )
orders = [
    np.log(deviation[b] / deviation[a]) / np.log(b / a)
    for a, b in ((0.25, 0.5), (0.5, 1.0))
]
print(
    f"observed order of the deviation in ka: {orders[0]:.2f}, {orders[1]:.2f} (about 2 expected; the reflection of the outer condition adds a floor)"
)

# ## Figure
#
# Left: the total pressure on the ellipsoid at $ka=1$ - largest where the wave arrives, smallest in
# the shadow. Middle: the scattered amplitude at the 400 receivers at $ka=0.25$, simulation against
# law; on the diagonal they agree. Right: the deviation against $ka$.

fine = runs[1.0]
surface = np.asarray(fine["mesh"].facets(parts[1.0].boundaries.body))
nodes = np.unique(surface)
points = np.asarray(fine["mesh"].coordinates)[nodes]
total = np.abs(
    np.asarray(fine["space"].evaluate(fine["pressure"], points))
    + incident(1.0 / axes[0], points)
)
low = runs[0.25]
receivers = low["outer"] / 2 * directions
simulated, law = np.abs(
    np.asarray(low["space"].evaluate(low["pressure"], receivers))
), np.abs(rayleigh(0.25 / axes[0], receivers))
fig = plots.figure(1, 3, size=(22, 6.5))
plots.trisurface(
    points,
    np.searchsorted(nodes, surface),
    total,
    camera=view,
    label="|p| / p0",
    title="Total pressure on the ellipsoid, ka = 1",
    ax=fig[0],
)
plots.line(
    points={"receivers on r = R/2": (1e3 * law, 1e3 * simulated)},
    reference={"equality": ([0, 1e3 * law.max()], [0, 1e3 * law.max()])},
    xlabel="|p_s|, low-frequency law [mPa]",
    ylabel="|p_s|, simulation [mPa]",
    title="Scattered amplitude, ka = 0.25",
    legend="upper left",
    ax=fig[1],
)
plots.line(
    np.array(list(deviation)),
    {"largest deviation": np.array(list(deviation.values()))},
    markers=True,
    slope=2.0,
    scale="loglog",
    xlabel="ka",
    ylabel="deviation from the low-frequency law",
    title="Approach to the low-frequency law",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Raise $ka$ beyond 1 in `ellipsoid_scattering_geometry.py` (outer sphere $8/k$) to enter the
# region where the ellipsoid casts a real shadow and the low-frequency law no longer holds; or turn
# the incidence onto a principal axis, where the field becomes axisymmetric for a spheroid.
