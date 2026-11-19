"""OneSimulation example: a random geometry of spherical inclusions in a cube.
Requires the installed, activated OneSimulation runtime.

Spheres of random size and position are placed in a cube one after another, each kept only if it
keeps a gap to the cube walls and to every sphere placed before (random sequential addition). The
matrix and the inclusions become one assembly of the CAD objects of OneSimulation and one mesh
with a volume for every part.

The volume fraction of the CAD model and of the mesh is compared with the sum of the sphere
volumes, and the placement rule is checked on the finished geometry.
"""

# # Creating a random geometry: inclusions in a matrix
#
# Particle-filled composites, porous media and foams are modelled as random arrangements of
# simple bodies. A common way to build one is random sequential addition: draw a centre and a
# radius, keep the sphere if it does not come too close to anything placed before, repeat. With a
# seed for the random numbers the geometry is random but reproducible.
#
# Here up to 300 attempts place spheres with radii between 3 and 6 mm in a cube of 50 mm. Every
# sphere keeps at least $g$ = 1 mm from the walls and from every other sphere, so the matrix
# between them never becomes thinner than that and the mesh stays well formed. The matrix is the
# cube minus the spheres; matrix and spheres are the parts of an `Assembly`, which the mesher
# fills as one mesh with a volume per part that meet node by node at every sphere surface.
#
# **Reference.** The inclusion volume is the sum of the sphere volumes,
# $V_i=\sum_k \tfrac43\pi r_k^3$, and the volume fraction $\varphi = V_i/a^3$; the matrix must have
# $a^3 - V_i$. The placement rule requires $|c_j-c_k| \ge r_j+r_k+g$ for every pair.

from functools import reduce

import numpy as np
from onesim.cad.d3 import Assembly, Box, Sphere
from onesim.meshing import TetrahedralMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## Random sequential addition
#
# The candidates are drawn at once from a seeded generator; the loop keeps those that respect the
# gap.

side, smallest, largest, gap = 0.05, 0.003, 0.006, 0.001
random = np.random.default_rng(2026)
radii = random.uniform(smallest, largest, 300)
candidates = [(random.uniform(r + gap, side - r - gap, 3), r) for r in radii]
placed = []
for centre, r in candidates:
    if all(np.linalg.norm(centre - other) >= r + rk + gap for other, rk in placed):
        placed.append((centre, r))
print(f"{len(placed)} of {len(candidates)} candidate spheres placed")

spheres = [
    Sphere(radius=r, origin=tuple(centre)).named(f"inclusion_{k + 1}")
    for k, (centre, r) in enumerate(placed)
]
matrix = reduce(
    lambda body, ball: body - ball, spheres, Box(length=side, width=side, height=side)
).named("matrix")
composite = Assembly(parts=[matrix, *spheres])
mesh = TetrahedralMesher(target_size=0.004, curvature_angle=0.35).generate(composite)

view = plots.Camera(elevation=25, azimuth=-55)
fig = plots.figure(1, 2, size=(16, 7))
plots.geometry(
    matrix,
    selection={"inclusions": matrix.select(faces=lambda face: face.kind == "sphere")},
    labels=False,
    camera=view,
    title=f"{len(spheres)} random inclusions placed in the cube",
    ax=fig[0],
)
plots.mesh(
    mesh,
    cell_values=quality(mesh)["shape"],
    limits=(0, 1),
    plane=((side / 2, side / 2, side / 2), (0.0, 0.0, 1.0)),
    label="shape measure",
    camera=plots.Camera(elevation=60, azimuth=-60),
    title=f"One mesh of both, cut at mid-height: {mesh.cell_count} cells",
    ax=fig[1],
)
plots.show()

# ## Results
#
# Volume of the inclusions and of the matrix in the CAD model and in the mesh against the sum of
# the sphere volumes, and the smallest clearance between two spheres and between a sphere and a
# wall. The flat faces of linear tetrahedra lie inside the round spheres, so the meshed inclusions
# come out smaller and the matrix larger by the same amount; with spheres of 3 to 6 mm radius the
# difference is a few per cent.

exact = sum(4 / 3 * np.pi * r**3 for _, r in placed)
cells = quality(mesh)["size"]
meshed = {
    part.id: cells[list(mesh.region_cells(composite.volume_of(part.id)))].sum()
    for part in composite.volumes
}
in_mesh = sum(value for name, value in meshed.items() if name != "matrix")
print(
    f"inclusions: CAD {sum(ball.measure for ball in spheres) * 1e6:.4f} cm3, mesh {in_mesh * 1e6:.4f} cm3, "
    f"sum of 4/3 pi r^3 {exact * 1e6:.4f} cm3"
)
print(
    f"matrix: CAD {matrix.measure * 1e6:.4f} cm3, mesh {meshed['matrix'] * 1e6:.4f} cm3, a^3 - V_i {(side**3 - exact) * 1e6:.4f} cm3"
)
print(
    f"volume fraction: CAD {1 - matrix.measure / side**3:.4f}, mesh {in_mesh / side**3:.4f}, closed form {exact / side**3:.4f}"
)
clearance = min(
    np.linalg.norm(cj - ck) - rj - rk
    for j, (cj, rj) in enumerate(placed)
    for ck, rk in placed[j + 1 :]
)
walls = min(min(c.min() - r, side - c.max() - r) for c, r in placed)
print(
    f"smallest clearance between spheres {1e3 * clearance:.3f} mm, to the walls {1e3 * walls:.3f} mm (required {1e3 * gap:.1f} mm)"
)
print(f"contact faces between parts: {len(composite.interfaces)} (one per inclusion)")

# ## Figure
#
# The cross-section of the matrix at mid-height with the circles cut out by the inclusions, and the
# radii of the placed spheres.

fig = plots.figure(1, 2, size=(16, 7))
cut = matrix.section((0, 0, 1), (side / 2, side / 2, side / 2))
plots.geometry(
    cut,
    selection={"matrix section": cut.boundaries.all},
    labels=False,
    title="Matrix cut at mid-height",
    ax=fig[0],
)
counts, edges = np.histogram(
    [1e3 * r for _, r in placed], bins=6, range=(1e3 * smallest, 1e3 * largest)
)
plots.bars(
    [f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])],
    counts,
    ylabel="number of spheres",
    title="Radii of the placed spheres [mm]",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Change the seed for another realisation, replace the spheres by randomly turned cylinders
# (`Cylinder(...).rotated(axis, angle)`) for a fibre composite, or give matrix and inclusions their
# own conductivity with `regionwise` and compute the effective conductivity of the composite.
