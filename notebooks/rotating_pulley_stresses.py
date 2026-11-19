"""OneSimulation example: stresses in a belt-driven pulley as a function of speed.
Requires the installed, activated OneSimulation runtime.

A spoked aluminium pulley transmits a constant shaft power to a flat belt. Its mid-plane section,
cut from a STEP file, is solved in plane stress with the capstan belt load and the centrifugal body
force at two speeds; global equilibrium, the exact rotating-disc solution on the pulley blank and a
three-level mesh study check the results.
"""

# # Stresses in a rotating pulley
#
# A driving pulley of outer radius $R=125$ mm and thickness $t=10$ mm sits on a shaft (bore radius
# 25 mm) and drives a flat belt that wraps it over the angle $\beta=170^\circ$. Eight rounded,
# trapezoidal cut-outs leave a hub, eight tapered spokes and a rim. The shaft power $P$ is constant,
# so the torque $M=P/\omega$ falls as the speed rises, while the centrifugal load grows with
# $\omega^2$: at low speed the belt dominates, at high speed the inertia does.
#
# The plate is thin and loaded in its plane, so it is a **plane-stress** problem on the mid-plane
# section. The pulley comes from the CAD file `rotating_pulley_stresses.step`, written by
# `rotating_pulley_stresses_geometry.py` with the CAD objects of OneSimulation; `section` cuts it at
# half thickness and keeps the face groups `bore`, `rim` and `cutouts` as named edges.
#
# **Belt load.** With friction coefficient $\mu$ and impending slip, the strand tensions obey the
# capstan relation $F_1/F_2=e^{\mu\beta}$ and $F_1-F_2=M/R$. Along the contact, measured by the
# angle $\theta$ from the point where the slack strand leaves, the belt presses on the rim with
#
# $$p(\theta)=\frac{F_2}{R\,t}\,e^{\mu\theta}\ \text{(inward)},\qquad
#   \tau(\theta)=\mu\,p(\theta)\ \text{(against the rotation)}.$$
#
# **Inertia.** In the rotating frame every material point carries the centrifugal body force
# $\rho\,\omega^2\,\mathbf r$, given by `BodyForce.centrifugal`.
#
# | edge | condition |
# |---|---|
# | `bore` | held, $u_x=u_y=0$ |
# | `rim` | belt traction over the wrap angle, free elsewhere |
# | `cutouts` | traction free |
#
# **Reference.** There is no closed-form solution for the spoked pulley. The notebook uses
# (i) global equilibrium: the shaft reaction must balance the two strand forces, (ii) the exact
# plane-stress solution of a rotating annular disc held at the bore (S. Timoshenko and J. N. Goodier,
# *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970, §32) on the pulley blank, and (iii) a three-level
# mesh study of the peak stress.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    BodyForce,
    FixedDisplacementComponents,
    LinearElasticity,
    Traction,
    TractionFree,
)
from onesim.results.derived import von_mises
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# Three meshes of quadratic triangles differ in the size along the walls of the cut-outs, where the
# rounded corners carry the peak stress.

pulley = load("rotating_pulley_stresses.step")
thickness = pulley.size[2]
plate = pulley.section((0, 0, 1), origin=(0, 0, thickness / 2))
edges = plate.boundaries
outer_radius = plate.size[0] / 2
sizes = [0.0040, 0.0028, 0.0020]
meshes = [
    TriangularMesher(target_size=0.008, local_sizes={"cutouts": size}).generate(plate)
    for size in sizes
]

fig = plots.figure(1, 2, size=(15, 6.5))
plots.geometry(
    pulley,
    selection={
        "bore": pulley.boundaries.bore,
        "rim": pulley.boundaries.rim,
        "cutouts": pulley.boundaries.cutouts,
    },
    labels=False,
    camera=plots.Camera(elevation=55, azimuth=-90),
    title="Pulley with its named faces",
    ax=fig[0],
)
plots.mesh(
    meshes[-1],
    groups=("all",),
    legend=False,
    title=f"Mid-plane section, finest mesh: {meshes[-1].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Aluminium. Both loads depend on the speed: the centrifugal body force grows with $\omega^2$, the
# belt traction falls with the torque $P/\omega$. The problem is solved at 1000 and 9000 rpm on the
# three meshes.

power, friction, wrap, speeds, bore_radius = (
    30e3,
    0.3,
    np.radians(170.0),
    (1000.0, 9000.0),
    0.025,
)
aluminium = IsotropicSolid(
    density=2700.0,
    specific_heat_capacity=900.0,
    thermal_conductivity=200.0,
    youngs_modulus=70e9,
    poisson_ratio=0.33,
    thermal_expansion_coefficient=23e-6,
)
angular = lambda rpm: 2 * np.pi * rpm / 60
slack_force = (
    lambda rpm: power / angular(rpm) / outer_radius / (np.exp(friction * wrap) - 1)
)
pressure = lambda rpm, angle: np.where(
    np.abs(angle) < wrap / 2,
    slack_force(rpm)
    / (outer_radius * thickness)
    * np.exp(friction * (angle + wrap / 2)),
    0.0,
)
belt = lambda rpm: lambda x, y: -pressure(rpm, np.arctan2(y, x))[..., None] * np.stack(
    (
        np.cos(np.arctan2(y, x)) + friction * np.sin(np.arctan2(y, x)),
        np.sin(np.arctan2(y, x)) - friction * np.cos(np.arctan2(y, x)),
    ),
    axis=-1,
)
models = {
    rpm: LinearElasticity(
        domain=plate.domain(),
        material=aluminium,
        kinematics="plane_stress",
        loading=BodyForce.centrifugal(
            density=aluminium.density, angular_velocity=angular(rpm)
        ),
    )
    for rpm in speeds
}
problems = {
    rpm: Problem(
        name="Pulley in plane stress",
        geometry=plate,
        physics=(model,),
        boundary_conditions=(
            FixedDisplacementComponents(
                boundary=edges.bore, physics=model, components=("x", "y")
            ),
            Traction(boundary=edges.rim, physics=model, value=belt(rpm)),
            TractionFree(boundary=edges.all - edges.bore - edges.rim, physics=model),
        ),
    )
    for rpm, model in models.items()
}
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
results = {
    (rpm, size): Simulation(
        problem=problems[rpm], mesh=mesh, numerics=numerics
    ).solve_stationary()
    for rpm in speeds
    for size, mesh in zip(sizes, meshes)
}
stress = {rpm: results[rpm, sizes[-1]].field(models[rpm].stress) for rpm in speeds}
mises = {
    rpm: results[rpm, sizes[-1]].field(models[rpm].von_mises_stress) for rpm in speeds
}

# ## Global equilibrium and mesh study
#
# The reaction of the shaft on the bore must balance the two strand forces
# $F_1\mathbf t_1+F_2\mathbf t_2$; the plane-stress model carries the load per unit thickness. The
# peak von Mises stress sits on the walls of the cut-outs.

for rpm in speeds:
    tight, slack = slack_force(rpm) * np.exp(friction * wrap), slack_force(rpm)
    strands = tight * np.array(
        [-np.sin(wrap / 2), np.cos(wrap / 2)]
    ) + slack * np.array([-np.sin(wrap / 2), -np.cos(wrap / 2)])
    reaction = (
        thickness
        * np.asarray(
            results[rpm, sizes[-1]].field(models[rpm].displacement).reaction(edges.bore)
        )[:2]
    )
    print(
        f"{rpm:5.0f} rpm: torque {power / angular(rpm):6.1f} N m, strand forces {tight:6.1f} and {slack:5.1f} N; "
        f"shaft reaction ({reaction[0]:8.2f}, {reaction[1]:8.2f}) N against ({-strands[0]:8.2f}, {-strands[1]:8.2f}) N"
    )

peaks = {
    rpm: [
        results[rpm, size]
        .field(models[rpm].von_mises_stress)
        .sample(along=edges.cutouts, subdivisions=3)
        .values.max()
        for size in sizes
    ]
    for rpm in speeds
}
print(
    "size on the cut-outs [mm]   triangles   peak von Mises stress [MPa] at 1000 / 9000 rpm"
)
for index, (size, mesh) in enumerate(zip(sizes, meshes)):
    print(
        f"{1e3 * size:25.1f} {mesh.cell_count:11d} {peaks[speeds[0]][index] / 1e6:21.2f} / {peaks[speeds[1]][index] / 1e6:.2f}"
    )
for rpm in speeds:
    print(
        f"{rpm:5.0f} rpm: change between the two finest meshes {100 * abs(peaks[rpm][-1] / peaks[rpm][-2] - 1):.2f} %"
    )

# ## Speed sweep
#
# The problem is linear, so the stress at any speed is a combination of the two solved states: one
# part grows with $\omega^2$, the other with $P/\omega$. `von_mises` evaluates the combined tensors on
# the walls of the cut-outs and at a point on a spoke.

sweep = np.arange(1000.0, 9001.0, 250.0)
weights = np.linalg.inv([[angular(rpm) ** 2, power / angular(rpm)] for rpm in speeds])
probe = [[0.019, 0.054]]
on_walls = np.stack(
    [stress[rpm].sample(along=edges.cutouts, subdivisions=3).values for rpm in speeds]
)
at_probe = np.stack([stress[rpm].evaluate(probe) for rpm in speeds])
combined = lambda states, rpm: np.tensordot(
    np.array([angular(rpm) ** 2, power / angular(rpm)]) @ weights, states, axes=1
)
sweep_peak = np.array([von_mises(combined(on_walls, rpm)).max() for rpm in sweep])
sweep_probe = np.array([von_mises(combined(at_probe, rpm))[0] for rpm in sweep])
print(
    f"lowest peak stress {sweep_peak.min() / 1e6:.1f} MPa at {sweep[sweep_peak.argmin()]:.0f} rpm"
)

# ## Rotating disc
#
# The section of the pulley blank at 9000 rpm without belt is compared with the exact plane-stress
# solution of the rotating annular disc held at the bore, $u(a)=0$, free at $r=b$:
#
# $$\sigma_r=c\Big(-k(3+\nu)r^2+C_1(1+\nu)-\frac{C_2(1-\nu)}{r^2}\Big),\qquad
#   \sigma_\theta=c\Big(-k(1+3\nu)r^2+C_1(1+\nu)+\frac{C_2(1-\nu)}{r^2}\Big),$$
#
# with $c=E/(1-\nu^2)$, $k=(1-\nu^2)\rho\omega^2/(8E)$ and $C_1$, $C_2$ from the two edge conditions.
# Radial and hoop stress follow from the Cartesian components along a ray.

blank = load("rotating_pulley_stresses_blank.step").section(
    (0, 0, 1), origin=(0, 0, thickness / 2)
)
disc = LinearElasticity(
    domain=blank.domain(),
    material=aluminium,
    kinematics="plane_stress",
    loading=BodyForce.centrifugal(
        density=aluminium.density, angular_velocity=angular(speeds[1])
    ),
)
disc_mesh = TriangularMesher(target_size=0.006, local_sizes={"bore": 0.002}).generate(
    blank
)
disc_stress = (
    Simulation(
        problem=Problem(
            name="Pulley blank",
            geometry=blank,
            physics=(disc,),
            boundary_conditions=(
                FixedDisplacementComponents(
                    boundary=blank.boundaries.bore, physics=disc, components=("x", "y")
                ),
                TractionFree(boundary=blank.boundaries.rim, physics=disc),
            ),
        ),
        mesh=disc_mesh,
        numerics=numerics,
    )
    .solve_stationary()
    .field(disc.stress)
)
E, nu, w = aluminium.youngs_modulus, aluminium.poisson_ratio, angular(speeds[1])
k, c = (1 - nu**2) * aluminium.density * w**2 / (8 * E), E / (1 - nu**2)
C1, C2 = np.linalg.solve(
    [[bore_radius, 1 / bore_radius], [c * (1 + nu), -c * (1 - nu) / outer_radius**2]],
    [k * bore_radius**3, c * k * (3 + nu) * outer_radius**2],
)
r = np.linspace(bore_radius + 5e-4, outer_radius - 5e-4, 60)
direction = np.array([np.cos(0.35), np.sin(0.35)])
components = disc_stress.evaluate(r[:, None] * direction)
radial = (
    components[:, 0] * direction[0] ** 2
    + components[:, 1] * direction[1] ** 2
    + 2 * components[:, 3] * direction[0] * direction[1]
)
hoop = (
    components[:, 0] * direction[1] ** 2
    + components[:, 1] * direction[0] ** 2
    - 2 * components[:, 3] * direction[0] * direction[1]
)
exact_radial = c * (-k * (3 + nu) * r**2 + C1 * (1 + nu) - C2 * (1 - nu) / r**2)
exact_hoop = c * (-k * (1 + 3 * nu) * r**2 + C1 * (1 + nu) + C2 * (1 - nu) / r**2)
print(
    f"rotating disc at 9000 rpm, {disc_mesh.cell_count} triangles: hoop stress near the bore {hoop[0] / 1e6:.3f} MPa "
    f"(exact {exact_hoop[0] / 1e6:.3f}), near the rim {hoop[-1] / 1e6:.3f} MPa (exact {exact_hoop[-1] / 1e6:.3f}); "
    f"largest deviation along the radius {100 * max(np.abs(hoop - exact_hoop).max(), np.abs(radial - exact_radial).max()) / exact_hoop.max():.2f} % of the peak"
)

# ## Figure
#
# The von Mises stress in the plate at both speeds, the speed sweep, the rotating disc against its
# exact solution and the mesh study.

fig = plots.figure(2, 2, size=(15, 12))
for index, rpm in enumerate(speeds):
    plots.surface(
        mises[rpm],
        component=lambda s, position: s / 1e6,
        limits=(0, 10 * round(0.075 * peaks[rpm][-1] / 1e6)),
        subdivisions=3,
        label="von Mises stress [MPa]",
        title=f"{rpm:.0f} rpm: torque {power / angular(rpm):.0f} N m, peak {peaks[rpm][-1] / 1e6:.1f} MPa",
        ax=fig[index],
    )
plots.line(
    sweep,
    {"peak on the cut-outs": sweep_peak / 1e6, "point on a spoke": sweep_probe / 1e6},
    xlabel="rotational speed [rpm]",
    ylabel="von Mises stress [MPa]",
    title="Constant power: belt against inertia",
    ax=fig[2],
)
plots.line(
    points={
        "hoop": (1e3 * r[::3], hoop[::3] / 1e6),
        "radial": (1e3 * r[::3], radial[::3] / 1e6),
    },
    reference={
        "hoop": (1e3 * r, exact_hoop / 1e6),
        "radial": (1e3 * r, exact_radial / 1e6),
    },
    xlabel="radius [mm]",
    ylabel="stress [MPa]",
    title="Blank at 9000 rpm: simulation (markers), exact (dashed)",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Change the cut-outs or their corner radii in `rotating_pulley_stresses_geometry.py`, write the STEP
# files again and follow the peak stress; raise the friction coefficient or the wrap angle to see the
# belt load move.
