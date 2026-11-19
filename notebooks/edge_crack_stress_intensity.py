"""OneSimulation example: stress intensity factor of a single-edge crack in a plate under tension.
Requires the installed, activated OneSimulation runtime.

A steel plate with an edge crack is pulled at its ends. The plate is read from a STEP file for
four crack lengths and cut to its cross-section; the stress intensity factor follows from the
J-integral, evaluated as a domain integral on three rings around the crack tip, and is compared
with the handbook solution of Tada, Paris and Irwin and with Irwin's crack-opening profile.
"""

# # Stress intensity factor of a single-edge crack
#
# Whether an existing crack grows is decided in linear elastic fracture mechanics by the
# **stress intensity factor** $K_I$: near the tip the stresses grow like $K_I/\sqrt{2\pi r}$, and
# the crack becomes unstable when $K_I$ reaches the fracture toughness $K_{Ic}$ of the material.
# $K_I$ depends on the load, the crack length and the shape of the part, so it has to be computed.
#
# The most robust way is the **J-integral** of Rice, the energy released per unit crack
# advance. As a domain integral over a ring around the tip it reads
#
# $$J=\int_A\Big(\sigma_{ij}\,\frac{\partial u_i}{\partial x_1}-W\,\delta_{1j}\Big)\frac{\partial q}{\partial x_j}\,\mathrm dA ,
#   \qquad K_I=\sqrt{E\,J}\quad\text{(plane stress)},$$
#
# with the strain energy density $W$ and a weight $q$ that is 1 inside the ring and 0 outside.
# $J$ is the same for every ring; checking that is part of the result.
#
# **Setup.** A steel plate ($E=200$ GPa, $\nu=0.3$), width $W=100$ mm, height 400 mm, 10 mm
# thick, with an edge crack of length $a$ at half height, $a/W=0.2$, 0.3, 0.4 and 0.5, in plane
# stress. The plates come from the CAD files `edge_crack_stress_intensity_<a/W in %>.step`,
# written by `edge_crack_stress_intensity_geometry.py` with the CAD objects of OneSimulation; the
# crack is a sharp wedge 0.2 mm wide at its mouth.
#
# | side | condition |
# |---|---|
# | `y_min` (bottom) | held, $u_x=u_y=0$ |
# | `y_max` (top) | tension $\sigma=100$ MPa |
# | `x_min`, `x_max`, `crack.side` | traction free |
#
# **Reference.** H. Tada, P.C. Paris and G.R. Irwin, *The Stress Analysis of Cracks Handbook*,
# 3rd ed., ASME Press 2000: $K_I=\sigma\sqrt{\pi a}\,F(a/W)$ with
# $F=1.12-0.231\,\alpha+10.55\,\alpha^2-21.72\,\alpha^3+30.39\,\alpha^4$, $\alpha=a/W$, accurate to
# 0.5 % for $\alpha\le0.6$. Near the tip the crack opens like Irwin's profile
# $\delta(r)=(8K_I/E)\sqrt{r/2\pi}$.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.physics.solid_mechanics import SmallStrainSolid, small_strain_j_integral
from onesim.results.comparisons import convergence
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns each plate, `section` its cross-section at half thickness with the names of the
# cut faces on its sides. The triangles are 0.5 mm at the crack tip and grow to 10 mm away from
# it; for the mesh study the plate with $a/W=0.3$ is also meshed with 2 mm and 1 mm at the tip.

ratios = (0.2, 0.3, 0.4, 0.5)
plates = {
    ratio: load(f"edge_crack_stress_intensity_{round(100 * ratio)}.step")
    for ratio in ratios
}
width, height, thickness = plates[0.3].size
sections = {
    ratio: plate.section((0, 0, 1), origin=(0, 0, thickness / 2))
    for ratio, plate in plates.items()
}
tips = {ratio: np.array([ratio * width, height / 2]) for ratio in ratios}
graded = lambda tip, size: TriangularMesher(
    target_size=0.01,
    size_field=lambda p: np.minimum(
        size + 0.3 * np.hypot(p[:, 0] - tip[0], p[:, 1] - tip[1]), 0.01
    ),
)
meshes = {
    ratio: graded(tips[ratio], 0.0005).generate(section)
    for ratio, section in sections.items()
}
tip_sizes = [0.002, 0.001, 0.0005]
study_meshes = [
    graded(tips[0.3], size).generate(sections[0.3]) for size in tip_sizes[:2]
] + [meshes[0.3]]

faces = sections[0.3].boundaries
fig = plots.figure(1, 2, size=(12, 7))
plots.geometry(
    plates[0.3],
    selection={
        "held": plates[0.3].boundaries.y_min,
        "pulled": plates[0.3].boundaries.y_max,
        "crack": plates[0.3].boundaries["crack.side"],
    },
    labels=False,
    camera=plots.Camera(elevation=30, azimuth=-60),
    title="Plate with a/W = 0.3",
    ax=fig[0],
)
plots.mesh(
    meshes[0.3],
    groups=("all",),
    legend=False,
    title=f"a/W = 0.3: {meshes[0.3].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Plane stress with quadratic triangles. The virtual crack advance $q$ is a smooth step from 1
# inside the radius $r_1$ to 0 outside $r_2$; `small_strain_j_integral` takes its gradient and
# returns $J$ per unit thickness. Three rings: 2-6 mm, 4-10 mm and 6-15 mm.

stress, youngs_modulus = 100e6, 200e9
steel = SmallStrainSolid(
    youngs_modulus=youngs_modulus,
    poisson_ratio=0.3,
    kinematics="plane_stress",
    thickness=thickness,
)
solve = lambda mesh, faces: steel.solve(
    mesh=mesh,
    times=(0.0,),
    order=2,
    displacements=((faces.y_min, "x", 0.0), (faces.y_min, "y", 0.0)),
    tractions=((faces.y_max, (0.0, stress)),),
)
results = {
    ratio: solve(meshes[ratio], section.boundaries)
    for ratio, section in sections.items()
}
study = [solve(mesh, faces) for mesh in study_meshes]
advance = lambda tip, inner, outer: lambda p: np.einsum(
    "i,q,qj->qij",
    (1.0, 0.0),
    -6
    * (s := np.clip((np.hypot(*(p - tip).T) - inner) / (outer - inner), 0, 1))
    * (1 - s)
    / ((outer - inner) * np.hypot(*(p - tip).T)),
    p - tip,
)
rings = [(0.002, 0.006), (0.004, 0.010), (0.006, 0.015)]
intensity = {
    ratio: [
        np.sqrt(
            youngs_modulus
            * small_strain_j_integral(result, advance(tips[ratio], *ring))
        )
        for ring in rings
    ]
    for ratio, result in results.items()
}
study_intensity = [
    [
        np.sqrt(
            youngs_modulus * small_strain_j_integral(result, advance(tips[0.3], *ring))
        )
        for ring in rings
    ]
    for result in study
]
tada = (
    lambda a: stress
    * np.sqrt(np.pi * a * width)
    * (1.12 - 0.231 * a + 10.55 * a**2 - 21.72 * a**3 + 30.39 * a**4)
)

# ## Comparison with the reference
#
# $K_I$ on the three rings for every crack length against the handbook, and the mesh study for
# $a/W=0.3$.

print(
    f"{'a/W':>5s} {'triangles':>10s} {'K_I, ring 1':>12s} {'ring 2':>9s} {'ring 3':>9s} {'Tada':>9s} {'deviation':>10s}  [MPa m^0.5]"
)
for ratio in ratios:
    values = intensity[ratio]
    print(
        f"{ratio:5.1f} {meshes[ratio].cell_count:10d} {values[0] / 1e6:12.3f} {values[1] / 1e6:9.3f} {values[2] / 1e6:9.3f}"
        f" {tada(ratio) / 1e6:9.3f} {100 * (np.mean(values) / tada(ratio) - 1):+9.2f} %"
    )
print(
    "mesh study, a/W = 0.3: size at the tip [mm], mean K_I [MPa m^0.5], deviation from Tada"
)
for size, values in zip(tip_sizes, study_intensity):
    print(
        f"  {1e3 * size:4.1f} mm   {np.mean(values) / 1e6:8.3f}   {100 * (np.mean(values) / tada(0.3) - 1):+6.2f} %"
    )
print(
    f"change between the two finest meshes: {100 * convergence([np.mean(v) for v in study_intensity], tip_sizes).change:.2f} %"
)

# ## Figure
#
# The vertical displacement around the crack of $a/W=0.3$, the normalised stress intensity
# $F=K_I/\sigma\sqrt{\pi a}$ against the handbook curve, the mesh study on the three rings, and the
# crack opening behind the tip against Irwin's profile.

result, tip = results[0.3], tips[0.3]
nodes = len(meshes[0.3].coordinates)
behind = np.linspace(2e-4, 0.25 * tip[0], 60)
opening = (
    result.evaluate_displacement(
        np.column_stack((tip[0] - behind, np.full_like(behind, tip[1] + 6e-4)))
    )[:, 1]
    - result.evaluate_displacement(
        np.column_stack((tip[0] - behind, np.full_like(behind, tip[1] - 6e-4)))
    )[:, 1]
)
alpha = np.linspace(0.15, 0.55, 81)
fig = plots.figure(2, 2, size=(14, 11))
plots.heatmap(
    1e3 * result.coordinates[:nodes, 0],
    1e3 * result.coordinates[:nodes, 1],
    1e6 * result.displacement[0, :nodes, 1],
    triangles=np.asarray(meshes[0.3].cells),
    limits=(60, 240),
    contours=12,
    label="u_y [µm]",
    xlabel="x [mm]",
    ylabel="y [mm]",
    xlimits=(0, 60),
    ylimits=(180, 220),
    title="Vertical displacement, a/W = 0.3",
    ax=fig[0],
)
plots.line(
    points={
        f"ring {k + 1}": (
            np.array(ratios),
            [
                intensity[ratio][k] / (stress * np.sqrt(np.pi * ratio * width))
                for ratio in ratios
            ],
        )
        for k in range(3)
    },
    reference={
        "Tada, Paris and Irwin": (
            alpha,
            tada(alpha) / (stress * np.sqrt(np.pi * alpha * width)),
        )
    },
    xlabel="a / W",
    ylabel="F = K_I / (σ √(πa))",
    title="Geometry factor of the edge crack",
    legend="upper left",
    ax=fig[1],
)
plots.line(
    points={
        f"ring {k + 1}": (
            1e3 * np.array(tip_sizes),
            [values[k] / 1e6 for values in study_intensity],
        )
        for k in range(3)
    },
    hlines={"Tada, Paris and Irwin": tada(0.3) / 1e6},
    ylimits=(0.99 * tada(0.3) / 1e6, 1.005 * tada(0.3) / 1e6),
    xlabel="element size at the tip [mm]",
    ylabel="K_I [MPa m^0.5]",
    title="Mesh study, a/W = 0.3",
    legend="lower left",
    ax=fig[2],
)
plots.line(
    y={"simulation": (1e3 * behind, 1e6 * opening)},
    reference={
        "Irwin's profile with the handbook K_I": (
            1e3 * behind,
            1e6 * 8 * tada(0.3) / youngs_modulus * np.sqrt(behind / (2 * np.pi)),
        )
    },
    xlabel="distance behind the tip r [mm]",
    ylabel="crack opening δ [µm]",
    title="Crack opening near the tip, a/W = 0.3",
    legend="upper left",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Change the crack length or the plate height in `edge_crack_stress_intensity_geometry.py` and
# write the STEP files again; at small heights the bending constraint of the held end lowers
# $K_I$. Compare with the fracture toughness of a steel (about 50 MPa m$^{0.5}$) to find the
# critical stress for each crack length.
