"""OneSimulation example: stress around a small circular hole in a large plate (Kirsch problem).
Requires the installed, activated OneSimulation runtime.

A large plate with a small hole is pulled in one direction. The plate is read from a STEP file and
cut to its mid-plane; the plane-stress solution on quadratic triangles is compared with the Kirsch
solution for the infinite plate: the stress concentration factor of 3 and the decay of the stress
away from the hole, with a mesh study and a plate-size study.
"""

# # Stress around a small hole in a large plate (Kirsch)
#
# A thin plate with a circular hole of radius $a$ carries a uniform tension $\sigma_0$ in the
# $x$ direction far from the hole. For an infinite plate Kirsch (1898) found the stress field in
# closed form. In polar coordinates about the hole centre, with the angle $\theta$ measured from
# the load direction, the hoop stress is
#
# $$\sigma_{\theta\theta}(r,\theta)=\frac{\sigma_0}{2}\Big(1+\frac{a^2}{r^2}\Big)
#   -\frac{\sigma_0}{2}\Big(1+\frac{3a^4}{r^4}\Big)\cos 2\theta .$$
#
# At the edge of the hole it reaches $3\sigma_0$ at $\theta=90°$ (stress concentration factor
# $K_t=3$) and $-\sigma_0$ at $\theta=0$, and it decays to the remote value within a few hole
# radii.
#
# The plate is modelled as a quarter, a square of half-width $L=20a$, from the CAD file
# `kirsch_plate_hole_stress.step`; the script `kirsch_plate_hole_stress_geometry.py` builds it with
# the CAD objects of OneSimulation and also writes plates with $L=5a$, $10a$ and $40a$ for the
# plate-size study. The plate is thin and loaded in its plane, so the model is plane stress on its
# mid-plane.
#
# | boundary | condition |
# |---|---|
# | $x=0$ (`symmetry_x`) | $u_x=0$ |
# | $y=0$ (`symmetry_y`) | $u_y=0$ |
# | $x=L$ (`loaded`) | traction $(\sigma_0,0)$ |
# | hole and $y=L$ | traction free |
#
# Steel, $E=200$ GPa, $\nu=0.3$, $a=10$ mm, $\sigma_0=100$ MPa.
#
# **Reference.** G. Kirsch, *Die Theorie der Elastizität und die Bedürfnisse der
# Festigkeitslehre*, Zeitschrift des VDI 42 (1898); S. Timoshenko and J. N. Goodier, *Theory of
# Elasticity*, 3rd ed., McGraw-Hill (1970), Sec. 35. The hoop stress is the closed
# Kirsch formula, written out below.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.results.comparisons import convergence
from onesim.visualization import plots, theme

theme.set()


def kirsch_hole_hoop_stress(distance, radius, remote_stress, angle):
    # Hoop stress at a distance from the edge of a circular hole in an infinite plate under
    # uniaxial remote stress; the angle is measured from the load direction.
    ratio_squared = (radius / (radius + np.asarray(distance, dtype=float))) ** 2
    return (
        remote_stress
        / 2
        * (1 + ratio_squared - (1 + 3 * ratio_squared**2) * np.cos(2 * angle))
    )


# ## Geometry and mesh
#
# `load` returns the quarter plate with its named faces; `section` cuts it at half thickness and
# returns the planar region, whose sides keep the names of the faces they were cut from. The
# triangles are graded from the hole, where three meshes use cells of $a/10$, $a/20$ and $a/40$.

plates = {
    size: load(f"kirsch_plate_hole_stress{suffix}.step")
    for size, suffix in ((5, "_5a"), (10, "_10a"), (20, ""), (40, "_40a"))
}
plate = plates[20]
faces = plate.boundaries
radius, thickness = 0.010, plate.size[2]
sections = {
    size: part.section((0, 0, 1), origin=(0, 0, thickness / 2))
    for size, part in plates.items()
}
mesher = lambda cell: TriangularMesher(
    target_size=2 * radius, grading=0.15, local_sizes={"hole.outer_wall": cell}
)
cells = [radius / 10, radius / 20, radius / 40]
meshes = [mesher(cell).generate(sections[20]) for cell in cells]

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    plate,
    selection={
        "loaded": faces.loaded,
        "hole": faces.hole,
        "symmetry": faces.symmetry_x | faces.symmetry_y,
    },
    labels=False,
    camera=plots.Camera(elevation=55, azimuth=-70),
    title="Quarter plate, L = 20a",
    ax=fig[0],
)
plots.mesh(
    meshes[1],
    groups=("all",),
    legend=False,
    title=f"Mid-plane mesh: {meshes[1].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Plane stress with quadratic triangles. The same model is solved on the three meshes of the
# reference plate and, with the finest hole cells, on the four plate sizes.

youngs_modulus, poisson_ratio, remote = 200e9, 0.3, 100e6
steel = SmallStrainSolid(
    youngs_modulus=youngs_modulus,
    poisson_ratio=poisson_ratio,
    kinematics="plane_stress",
    thickness=thickness,
)
solve = lambda mesh, sides: steel.solve(
    mesh=mesh,
    times=(0.0,),
    order=2,
    displacements=((sides.x_min, "x", 0.0), (sides.y_min, "y", 0.0)),
    tractions=((sides.x_max, (remote, 0.0)),),
)
results = [solve(mesh, sections[20].boundaries) for mesh in meshes]
sized = {
    size: solve(mesher(cells[-1]).generate(region), region.boundaries)
    for size, region in sections.items()
}

# ## Results
#
# The stress follows from the gradient of the quadratic displacement field, evaluated with
# Hooke's law for plane stress at any point of the plate: at the hole edge, along the two symmetry
# lines and on the mesh nodes for the map.

hooke = (
    lambda result, points: youngs_modulus
    / (1 - poisson_ratio**2)
    * (
        lambda g: np.stack(
            (
                g[:, 0, 0] + poisson_ratio * g[:, 1, 1],
                g[:, 1, 1] + poisson_ratio * g[:, 0, 0],
                (1 - poisson_ratio) / 2 * (g[:, 0, 1] + g[:, 1, 0]),
            ),
            axis=-1,
        )
    )(result.space.gradient(result.displacement[0], np.atleast_2d(points)))
)
peaks = [hooke(result, (0.0, radius))[0, 0] / remote for result in results]
size_peaks = {
    size: hooke(result, (0.0, radius))[0, 0] / remote for size, result in sized.items()
}
study = convergence(peaks, cells)
distance = np.linspace(0.0, 4 * radius, 81)
above = hooke(
    results[-1], np.column_stack((np.zeros_like(distance), radius + distance))
)[:, 0]
beside = hooke(
    results[-1], np.column_stack((radius + distance, np.zeros_like(distance)))
)[:, 1]

print(
    f"stress concentration at the hole edge, θ = 90°: simulation {peaks[-1]:.4f}, Kirsch 3.0000, deviation {100 * (peaks[-1] / 3 - 1):+.2f} %"
)
print(
    f"hoop stress at the hole edge, θ = 0: simulation {beside[0] / remote:.4f} σ0, Kirsch -1.0000 σ0"
)
print(
    f"largest deviation from Kirsch along the two symmetry lines up to 5a: "
    f"{100 * max(np.abs(above - kirsch_hole_hoop_stress(distance, radius, remote, np.pi / 2)).max(), np.abs(beside - kirsch_hole_hoop_stress(distance, radius, remote, 0.0)).max()) / remote:.2f} % of σ0"
)
print("hole cell size   triangles   K_t")
for cell, mesh, peak in zip(cells, meshes, peaks):
    print(f"a/{radius / cell:<14.0f} {mesh.cell_count:9d} {peak:9.4f}")
print(f"observed order {study.order:.1f}, extrapolated K_t {study.extrapolated:.4f}")
print("plate half-width   K_t      deviation from 3")
for size, peak in size_peaks.items():
    print(f"{size:>14d}a {peak:8.4f} {100 * (peak / 3 - 1):+10.2f} %")

# ## Figure
#
# The stress $\sigma_{xx}$ near the hole, the hoop stress along the two symmetry lines against
# Kirsch, and the stress concentration factor over the plate size: the finite plate approaches
# the infinite one as $a/L$ goes to zero.

mesh = meshes[-1]
nodal = hooke(results[-1], mesh.coordinates)[:, 0] / 1e6
fine = np.linspace(0.0, 4 * radius, 300)
fig = plots.figure(1, 3, size=(18, 5.5))
plots.heatmap(
    1e3 * mesh.coordinates[:, 0],
    1e3 * mesh.coordinates[:, 1],
    nodal,
    triangles=np.asarray(mesh.cells),
    limits=(-100, 300),
    contours=0,
    label="σ_xx [MPa]",
    xlimits=(0, 30),
    ylimits=(0, 30),
    xlabel="x [mm]",
    ylabel="y [mm]",
    title="σ_xx around the hole",
    ax=fig[0],
)
plots.line(
    points={
        "θ = 90°, along x = 0": (distance[::3] / radius, above[::3] / remote),
        "θ = 0, along y = 0": (distance[::3] / radius, beside[::3] / remote),
    },
    reference={
        "Kirsch, θ = 90°": (
            fine / radius,
            kirsch_hole_hoop_stress(fine, radius, remote, np.pi / 2) / remote,
        ),
        "Kirsch, θ = 0": (
            fine / radius,
            kirsch_hole_hoop_stress(fine, radius, remote, 0.0) / remote,
        ),
    },
    hlines=0.0,
    xlabel="distance from the hole edge (r - a)/a",
    ylabel="hoop stress / σ0",
    title="Hoop stress along the symmetry lines",
    legend="lower right",
    ax=fig[1],
)
plots.line(
    [1 / size for size in size_peaks],
    {"simulation, L = 5a, 10a, 20a, 40a": list(size_peaks.values())},
    markers=True,
    hlines={"Kirsch, infinite plate": 3.0},
    xlimits=(0, 0.21),
    ylimits=(2.9, 3.4),
    xlabel="hole radius / plate half-width a/L",
    ylabel="stress concentration factor K_t",
    title="From the finite to the infinite plate",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Replace the circle in `kirsch_plate_hole_stress_geometry.py` by an ellipse and compare with
# Inglis' formula $K_t=1+2b/c$, or load the plate in both directions: biaxial tension lowers
# the concentration to 2, pure shear raises it to 4.
