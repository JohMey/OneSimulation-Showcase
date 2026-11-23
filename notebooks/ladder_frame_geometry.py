"""OneSimulation example: the geometry of a ladder frame and its mid-surface shell mesh.
Requires the installed, activated OneSimulation runtime.

A vehicle ladder frame with two channel rails and four cross members is imported from STEP. The mid-surface of all walls is extracted in one
step, with the corners of the channels and the T-joints of the cross members, and meshed with
quadrilateral shell elements.

Volume, mid-surface area and wall thickness are compared with closed forms of the plate layout.
"""

# # Ladder frame geometry
#
# Truck and trailer chassis are ladder frames: two long side rails of channel section and a row of
# cross members between them. The walls are thin against every other dimension, so the frame is
# analysed with shell elements on the mid-surface of its walls rather than with solid elements.
#
# `part.midsurfaces()` finds the pairs of opposite wall faces of a solid and returns one sheet that
# lies halfway between them, joined at corners and T-joints and carrying the wall thickness.
# `SheetMesher` lays quadrilaterals on it.
#
# | member | section | count |
# |---|---|---|
# | side rail | channel $h\times f\times t$ = 160 x 60 x 6 mm, flanges pointing inwards, 2.4 m long | 2 |
# | cross member | flat web 100 x 6 mm between the rail webs, frame width 0.8 m | 4 |
#
# **Reference.** A wall of thickness $t$ meeting another wall at a corner or a T-joint has its
# mid-surface run to the mid-plane of the other wall. With rail length $L$, frame width $W$ and
# cross member height $h_c$:
#
# $$A_\text{mid} = 2\big[L(h-t) + 2L(f-\tfrac t2)\big] + 4(W-t)\,h_c,\qquad
#   V = 2\big[Lht + 2L(f-t)t\big] + 4(W-2t)\,h_c t .$$

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import SheetMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## Inspect the imported solid
#
# Read the demonstration chassis from its STEP file. Two channels and four flat
# cross members define the envelope. The downloadable CAD script sets their
# nominal section sizes; the imported solid determines the envelope dimensions.

frame = load("ladder_frame_geometry.step")
rail, width, height = frame.size
flange, wall, cross = 0.06, 0.006, 0.10
volume = (
    2 * (rail * height * wall + 2 * rail * (flange - wall) * wall)
    + 4 * (width - 2 * wall) * cross * wall
)
print(f"Imported frame: {rail:.2f} × {width:.2f} × {height:.2f} m, {len(frame.faces)} CAD faces.")
print(f"Solid volume {frame.measure * 1e6:.3f} cm³; plate-layout reference {volume * 1e6:.3f} cm³.")

# ## Mid-surface and shell mesh
#
# The mid-surface has one face per wall; the cross members end on the mid-planes of the rail webs.

middle = frame.midsurfaces()
area = 2 * (rail * (height - wall) + 2 * rail * (flange - wall / 2)) + 4 * (width - wall) * cross
coarse = SheetMesher(target_size=0.020).generate(middle)
mesh = SheetMesher(target_size=0.010).generate(middle)
for spacing, level in ((20, coarse), (10, mesh)):
    print(f"{spacing:g} mm target: {len(level.coordinates)} shell nodes, {len(level.quadrilaterals)} quads, area {level.area:.6f} m²")
print(
    f"mid-surface: {len(middle.faces)} faces for 10 walls, thickness {middle.thickness * 1e3:.3f} mm"
)
print(
    f"mid-surface area {middle.area:.6f} m2, closed form {area:.6f} m2, deviation {middle.area / area - 1:+.1e}"
)
print(
    f"shell mesh: {len(mesh.quadrilaterals)} quadrilaterals, area {mesh.area:.6f} m2; "
    f"shell volume area x thickness {mesh.area * middle.thickness * 1e6:.3f} cm3"
)
report = quality(mesh)
print(
    f"quadrilaterals: scaled Jacobian min {report['scaled_jacobian'].min():.3f}, warpage max {report['warpage'].max():.2e}"
)

view = plots.Camera(elevation=30, azimuth=-55)
fig = plots.figure(1, 2, size=(17, 7))
plots.geometry(
    frame,
    selection={"walls": frame.boundaries.all},
    labels=False,
    camera=view,
    title="Ladder frame: two channel rails, four cross members",
    ax=fig[0],
)
plots.mesh(
    mesh,
    camera=view,
    title=f"Shell mesh on the mid-surface: {len(mesh.quadrilaterals)} quadrilaterals",
    ax=fig[1],
)
plots.show()

# ## Figure
#
# The mid-surface of all walls, and the shell mesh seen from the rear end, where the cross member
# meets the rail webs in T-joints.

joint = plots.Camera(elevation=35, azimuth=-130)
fig = plots.figure(1, 2, size=(17, 7))
plots.geometry(middle, labels=False, camera=view, title="Mid-surface of all walls", ax=fig[0])
plots.mesh(mesh, camera=joint, title="Shell mesh at the rear cross member", ax=fig[1])
plots.show()

# ## Going further
#
# Load the frame in torsion with `ElasticShell` on `ShellSurface.from_mesh(mesh)`
# (`onesim.physics.solid_mechanics`), replace the flat cross members by tubes, or add gusset plates
# at the joints with `frame.gusset(...)`.
