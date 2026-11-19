"""OneSimulation example: a Menger sponge as a recursively defined CAD object.
Requires the installed, activated OneSimulation runtime.

The Menger sponge is the three-dimensional cousin of the Cantor set: every cube is divided into 27
and the centre cube and the six face centres are taken out, again and again. Two levels of the
recursion are built from boxes with the CAD objects of OneSimulation, then meshed with tetrahedra.

Volume, surface area and centre of mass of each level are compared with the closed-form
geometric series of the sponge.
"""

# # Recursively defined geometry: the Menger sponge
#
# A geometry that repeats itself at smaller and smaller scales is easiest to describe by a rule and
# a loop. The Menger sponge of level $n$ is a cube of side $a$ from which, at every level
# $k=1,\dots,n$, all points are removed whose $k$-th ternary digit is 1 in at least two of the three
# coordinates. Removing exactly those points is the same as cutting square bars of width
# $a/3^k$ through the whole cube along each axis, at the positions where both cross coordinates
# have the digit 1 at that level: $3\cdot 9^{k-1}$ bars at level $k$.
#
# Each level keeps 20 of the 27 sub-cubes, so volume and surface follow geometric series:
#
# $$V_n = a^3\Big(\frac{20}{27}\Big)^n,\qquad S_n = 2a^2\Big(\frac{20}{9}\Big)^n + 4a^2\Big(\frac{8}{9}\Big)^n ,$$
#
# and the centre of mass stays in the centre of the cube. The volume tends to zero and the surface
# to infinity: the limit has the fractal dimension $\ln 20/\ln 3 \approx 2.73$.
#
# **Reference.** K. Menger, *Allgemeine Räume und Cartesische Räume*, Proc. Royal Academy
# Amsterdam 29 (1926) 476-482; volume and surface series as in B. Mandelbrot, *The Fractal Geometry
# of Nature*, Freeman 1982, ch. 14.

from functools import reduce

import numpy as np
from onesim.cad.d3 import Box
from onesim.meshing import TetrahedralMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## The recursion
#
# For every level the bars are a list comprehension over the grid of $3^k\times3^k$ positions; the
# sponge of a level is the cube with all bars of all levels up to it subtracted one after another
# (`reduce` over the list with `body - bar`).

side = 0.09
cube = Box(length=side, width=side, height=side)
bars = {
    level: [
        bar
        for i in range(3**level)
        for j in range(3**level)
        if i % 3 == 1 and j % 3 == 1
        for w in [side / 3**level]
        for bar in (
            Box(length=w, width=w, height=side, origin=(i * w, j * w, 0.0)),
            Box(length=side, width=w, height=w, origin=(0.0, i * w, j * w)),
            Box(length=w, width=side, height=w, origin=(i * w, 0.0, j * w)),
        )
    ]
    for level in (1, 2)
}
sponges = [cube]
for level in (1, 2):
    sponges.append(reduce(lambda body, bar: body - bar, bars[level], sponges[-1]))
print(", ".join(f"level {level}: {len(items)} bars" for level, items in bars.items()))

meshes = [
    TetrahedralMesher(target_size=side / 9).generate(sponges[0]),
    TetrahedralMesher(target_size=side / 15).generate(sponges[1]),
    TetrahedralMesher(target_size=side / 27).generate(sponges[2]),
]

view = plots.Camera(elevation=28, azimuth=-52)
fig = plots.figure(1, 2, size=(16, 7))
plots.render(
    sponges[2],
    camera=view,
    material="ceramic",
    title="Menger sponge of level 2",
    ax=fig[0],
)
plots.mesh(
    meshes[2],
    cell_values=quality(meshes[2])["shape"],
    limits=(0, 1),
    plane=((side / 2, side / 2, side / 2), (0.0, 1.0, 0.0)),
    label="shape measure",
    camera=plots.Camera(elevation=28, azimuth=128),
    title=f"Level 2: {meshes[2].cell_count} tetrahedra, cut at the mid-plane",
    ax=fig[1],
)
plots.show()

# ## Results
#
# Volume and surface area of the CAD body and of the mesh, and the centre of mass, against the
# closed-form series.

volume = lambda n: side**3 * (20 / 27) ** n
surface = lambda n: 2 * side**2 * (20 / 9) ** n + 4 * side**2 * (8 / 9) ** n
print(
    "level  faces   volume CAD [cm3]  mesh     closed form   surface CAD [cm2]  closed form   cells"
)
for level, (sponge, mesh) in enumerate(zip(sponges, meshes)):
    print(
        f"{level:5d} {len(sponge.faces):6d} {sponge.measure * 1e6:14.4f} {quality(mesh)['size'].sum() * 1e6:10.4f} {volume(level) * 1e6:12.4f}"
        f" {sponge.surface_area * 1e4:17.4f} {surface(level) * 1e4:12.4f} {mesh.cell_count:9d}"
    )
centre = sponges[2].center_of_mass
print(
    f"centre of mass of level 2: ({1e3 * centre[0]:.6f}, {1e3 * centre[1]:.6f}, {1e3 * centre[2]:.6f}) mm, "
    f"centre of the cube ({1e3 * side / 2:.6f}, {1e3 * side / 2:.6f}, {1e3 * side / 2:.6f}) mm"
)
print(
    f"volume fraction left at level 2: {sponges[2].measure / side**3:.6f} (20/27)^2 = {(20 / 27)**2:.6f}"
)

# ## Figure
#
# The three levels side by side, rendered as shaded solids, with their volume and surface.

fig = plots.figure(1, 3, size=(19, 6.5))
for level in range(3):
    plots.render(
        sponges[level],
        camera=view,
        material="ceramic",
        title=f"level {level}: V = {sponges[level].measure / side**3:.3f} a^3, S = {sponges[level].surface_area / side**2:.3f} a^2",
        ax=fig[level],
    )
plots.show()

# ## Going further
#
# Extend the dictionary of bars to level 3 (273 bars; the CAD body takes a few minutes), build the
# Sierpinski tetrahedron from `Cone(..., top_radius=0)` pyramids the same way, or solve a heat
# conduction problem on the sponge to see how the effective conductivity falls with the level.
