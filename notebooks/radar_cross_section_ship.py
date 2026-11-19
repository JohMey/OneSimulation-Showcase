"""OneSimulation example: the radar cross section of a ship.
Requires the installed, activated OneSimulation runtime.

A radar wave of 150 MHz meets the side silhouette of a 12 m boat, modelled as a perfect conductor in
a two-dimensional section read from a STEP file. The scattered field is solved with the incident
plane wave as background field inside an absorbing layer; the far field gives the bistatic and the
monostatic radar cross section for both polarizations. A conducting cylinder in the same setup is
compared with its exact Bessel series, and the ship with the optical theorem.
"""

# # Radar cross section of a ship
#
# A distant radar illuminates a boat with a plane wave. The boat scatters part of the wave in all
# directions; the radar cross section $\sigma$ measures how strongly it does so in a direction, as
# the area that would intercept and isotropically re-radiate the observed power. In a
# two-dimensional section (a body long across the section) it is a length:
#
# $$\sigma(\varphi)=\lim_{r\to\infty}2\pi r\frac{|E_s|^2}{|E_0|^2}=\frac4k|F(\varphi)|^2,
#   \qquad E_s\simeq\sqrt{\frac2{\pi kr}}\,e^{\mathrm i(kr-\pi/4)}F(\varphi).$$
#
# The hull, the deck house and the mast are perfect conductors. The model solves the **scattered
# field** only: the incident wave $E_0$ is given analytically as `background_field`, and on the
# conductor the total tangential field vanishes, $E_s=-E_0$ - a `PerfectElectricConductor` on the
# inner boundary (the silhouette) carries the total field, the one behind the absorbing layer the
# scattered field. Both polarizations are solved: $E_z$ along the boat (TM, `Ez`) and $H_z$ along
# the boat (TE, `Hz`). `far_field` transforms the field on a circle around the boat into $F$.
#
# | boundary | condition |
# |---|---|
# | silhouette of the boat (inner boundary) | perfect conductor, total field $E=0$ tangentially |
# | outer box behind the 1.5 m absorbing layer | perfect conductor, scattered field |
#
# **Reference** (C. A. Balanis, *Advanced Engineering Electromagnetics*, 2nd ed., §11.5 and
# §11.8; J. A. Stratton, *Electromagnetic Theory*, §9.20). A conducting cylinder of radius $a$
# scatters the series $F=\sum_n s_n e^{\mathrm in\varphi}$ with $s_n=-J_n(ka)/H^{(1)}_n(ka)$ (TM) and
# $s_n=-J_n'(ka)/H^{(1)\prime}_n(ka)$ (TE). For any lossless scatterer the optical theorem ties the
# total scattered power to the forward amplitude:
#
# $$\sigma_s=\frac1{2\pi}\oint\frac4k|F|^2\,\mathrm d\varphi=\sigma_{ext}=-\frac4k\,\mathrm{Re}\,F(\varphi_{\rm forward}).$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import (
    CartesianPML,
    IsotropicElectromagneticMaterial,
)
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell,
    PerfectElectricConductor,
    far_field,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.special import h1vp, hankel1, jv, jvp

theme.set()

frequency = 150e6
light = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
wave, omega, admittance = (
    2 * np.pi * frequency / light,
    2 * np.pi * frequency,
    np.sqrt(VACUUM_PERMITTIVITY / VACUUM_PERMEABILITY),
)

# ## Geometry and mesh
#
# `load` reads the air around the boat and, for the validation, around the cylinder; `section`
# cuts them at half thickness. Quadratic triangles of 0.2 m, a tenth of the wavelength, and 5 cm
# along the conductors.

bodies = {
    name: load(f"radar_cross_section_ship{suffix}.step").section(
        (0, 0, 1), origin=(0, 0, 0.1)
    )
    for name, suffix in (("ship", ""), ("cylinder", "_cylinder"))
}
meshes = {
    name: TriangularMesher(target_size=0.2, local_sizes={"all": 0.05}).generate(body)
    for name, body in bodies.items()
}

fig = plots.figure(1, 2, size=(16, 7))
plots.geometry(
    bodies["ship"],
    selection={"conductor and outer wall": bodies["ship"].boundaries.all},
    labels=False,
    title="Air around the boat (STEP)",
    ax=fig[0],
)
plots.mesh(
    meshes["ship"],
    groups=("all",),
    legend=False,
    title=f"{meshes['ship'].cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The absorbing layer is a `CartesianPML` whose damping grows quadratically over the outer 1.5 m;
# the background material is the same layer around air, so that the incident wave passes through
# it unchanged. The radar sits on the right, its wave arrives at the elevation angle $\beta$ above
# the horizon: $\hat k=(-\cos\beta,-\sin\beta)$.

damping = lambda x, y, z: np.column_stack(
    (
        3 * omega * np.clip((np.abs(x) - 9.5) / 1.5, 0, None) ** 2,
        3 * omega * np.clip((np.abs(y) - 8.5) / 1.5, 0, None) ** 2,
        0 * x,
    )
)
medium = CartesianPML(material=IsotropicElectromagneticMaterial(), damping=damping)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=ComplexSparseDirect(),
)
incident = lambda polarization, direction: lambda x, y, z: (
    np.exp(1j * wave * (direction[0] * x + direction[1] * y))[:, None]
    * (
        np.array([0.0, 0.0, 1.0])
        if polarization == "Ez"
        else np.array([-direction[1], direction[0], 0.0])
    )
)
models = lambda name, polarization, direction: FrequencyDomainMaxwell(
    domain=bodies[name].domain(),
    frequency=frequency,
    material=medium,
    polarization=polarization,
    background_field=incident(polarization, direction),
    background_material=medium,
    phasor_convention="exp(-i*omega*t)",
    amplitude_convention="peak",
)
direction = lambda elevation: (
    -np.cos(np.radians(elevation)),
    -np.sin(np.radians(elevation)),
)
simulate = lambda name, physics: Simulation(
    problem=Problem(
        name=name,
        geometry=bodies[name],
        physics=(physics,),
        boundary_conditions=(
            PerfectElectricConductor(
                physics=physics, boundary=bodies[name].boundaries.all
            ),
        ),
    ),
    mesh=meshes[name],
    numerics=numerics,
).solve_stationary()
circles = {
    "ship": {"radius": 6.8, "center": (0.0, 1.0)},
    "cylinder": {"radius": 3.0, "center": (0.0, 0.0)},
}

angles = np.radians(np.arange(0.0, 360.0, 1.0))
lit = {"ship": 10.0, "cylinder": 180.0}
cases = {
    (name, polarization): models(name, polarization, direction(lit[name]))
    for name in ("ship", "cylinder")
    for polarization in ("Ez", "Hz")
}
results = {key: simulate(key[0], physics) for key, physics in cases.items()}
bistatic = {
    key: far_field(results[key], physics, angles, **circles[key[0]])
    / (1.0 if key[1] == "Ez" else admittance)
    for key, physics in cases.items()
}

# ## Results
#
# The cylinder, lit from the left ($\hat k=+x$), against the Bessel series; then the optical theorem
# for both bodies: the scattered width from the integral of $|F|^2$ over all directions against the
# extinction width from the forward amplitude.

orders, ka = np.arange(-30, 31), wave * 1.0
series = {
    "Ez": -jv(orders, ka) / hankel1(orders, ka),
    "Hz": -jvp(orders, ka) / h1vp(orders, ka),
}
print(
    "cylinder ka = pi      max |sigma - sigma_series| / max sigma    sigma back [m]: computed  series"
)
for polarization in ("Ez", "Hz"):
    exact = (
        4
        / wave
        * np.abs(np.exp(1j * np.outer(angles, orders)) @ series[polarization]) ** 2
    )
    computed = 4 / wave * np.abs(bistatic[("cylinder", polarization)]) ** 2
    print(
        f"{'TM (E_z)' if polarization == 'Ez' else 'TE (H_z)':20s}   {np.max(np.abs(computed - exact)) / exact.max():30.1e}"
        f"                    {computed[180]:8.3f}  {exact[180]:8.3f}"
    )
print(
    "\nbody      polarization   scattered width [m]   extinction width [m]   difference"
)
for (name, polarization), pattern in bistatic.items():
    forward = int(
        np.argmin(
            np.abs(
                np.angle(
                    np.exp(1j * (angles - np.radians(190.0 if name == "ship" else 0.0)))
                )
            )
        )
    )
    scattered = 4 / wave * np.mean(np.abs(pattern) ** 2)
    extinction = -4 / wave * pattern[forward].real
    print(
        f"{name:9s} {'TM (E_z)' if polarization == 'Ez' else 'TE (H_z)':12s}   {scattered:19.3f}   {extinction:20.3f}   {abs(scattered / extinction - 1):10.1e}"
    )

# The monostatic cross section is what the radar itself receives: the boat is lit at elevations
# from 0 to 60 degrees and the backscattered amplitude is read in the direction of the radar.

elevations = np.arange(0.0, 61.0, 5.0)
monostatic = {
    polarization: np.array(
        [
            4
            / wave
            * abs(
                far_field(
                    simulate("ship", physics),
                    physics,
                    np.radians([b]),
                    **circles["ship"],
                )[0]
                / (1.0 if polarization == "Ez" else admittance)
            )
            ** 2
            for b, physics in (
                (b, models("ship", polarization, direction(b))) for b in elevations
            )
        ]
    )
    for polarization in ("Ez", "Hz")
}
print("\nelevation [deg]   monostatic sigma TM [m]   TE [m]")
for b, tm, te in zip(elevations, monostatic["Ez"], monostatic["Hz"]):
    print(f"{b:10.0f}        {tm:12.2f}          {te:8.2f}")

# ## Figure
#
# The total field $|E_z|$ around the boat lit at 10 degrees, the bistatic cross section of the boat
# in dB above 1 m (the radar lies at 10 degrees), and the cylinder against its series.

result, physics = results[("ship", "Ez")], cases[("ship", "Ez")]
x, y = np.asarray(meshes["ship"].coordinates).T
total = np.abs(
    result.field(physics.electric_field).evaluate(np.column_stack((x, y)))[:, 2]
)
fig = plots.figure(1, 3, size=(26, 7))
plots.heatmap(
    x,
    y,
    total,
    triangles=np.asarray(meshes["ship"].cells),
    limits=(0, 2.2),
    label="|E_z| / |E_0|",
    xlabel="x [m]",
    ylabel="y [m]",
    xlimits=(-9.5, 9.5),
    ylimits=(-8.5, 8.5),
    title="Total field, TM wave at 10 deg elevation",
    ax=fig[0],
)
plots.line(
    np.degrees(angles),
    {
        "TM (E_z)": 10 * np.log10(4 / wave * np.abs(bistatic[("ship", "Ez")]) ** 2),
        "TE (H_z)": 10 * np.log10(4 / wave * np.abs(bistatic[("ship", "Hz")]) ** 2),
    },
    vlines={"radar": 10.0, "forward": 190.0},
    xlimits=(0, 360),
    xlabel="observation angle [deg]",
    ylabel="sigma [dB m]",
    title="Bistatic cross section of the boat",
    legend="lower right",
    ax=fig[1],
)
plots.line(
    np.degrees(angles),
    {
        f"{'TM' if p == 'Ez' else 'TE'} computed": 4
        / wave
        * np.abs(bistatic[("cylinder", p)]) ** 2
        for p in ("Ez", "Hz")
    },
    reference={
        f"{'TM' if p == 'Ez' else 'TE'} computed": 4
        / wave
        * np.abs(np.exp(1j * np.outer(angles, orders)) @ series[p]) ** 2
        for p in ("Ez", "Hz")
    },
    xlimits=(0, 360),
    xlabel="observation angle [deg]",
    ylabel="sigma [m]",
    title="Cylinder ka = pi (dashed: Bessel series)",
    legend="upper center",
    ax=fig[2],
)
plots.show()

fig = plots.figure(1, 1, size=(9, 5.5))
plots.line(
    elevations,
    {
        "TM (E_z)": 10 * np.log10(monostatic["Ez"]),
        "TE (H_z)": 10 * np.log10(monostatic["Hz"]),
    },
    markers=True,
    xlabel="elevation of the radar [deg]",
    ylabel="monostatic sigma [dB m]",
    title="What the radar receives",
    legend="upper right",
    ax=fig[0],
)
plots.show()

# ## Going further
#
# Change the silhouette in `radar_cross_section_ship_geometry.py`: tilting the flat walls of the deck
# house away from the radar removes their specular flashes from the monostatic cross section - the
# idea of stealth shaping. A lossy coating on the hull is a thin region with a complex permittivity.
