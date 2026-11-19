"""OneSimulation example: wrapping flat profiles onto cylinders, cones and spheres.
Requires the installed, activated OneSimulation runtime.

A flat outline, a label or lettering, is wrapped onto the curved faces of CAD objects of
OneSimulation without stretching it: around a shaft, onto the side of a cone and onto a sphere. The
wrapped outline is raised or sunk into the body, compared with a plain projection and meshed.

Every added or removed volume and every covered area is compared with its closed form on the
unrolled face.
"""

# # Wrapping a CAD geometry
#
# Lettering on a bottle, a grip pattern on a handle, a heating track on a pipe, a label strip on a
# shaft: a flat outline is laid onto a curved face so that its lengths are kept. On a cylinder or a
# cone the face can be unrolled into the plane; the outline is drawn there and rolled up with the
# face. On a sphere, which cannot be unrolled, distances from the anchor point are measured along
# the surface (geodesic polar coordinates).
#
# `part.wrap(profile, face, height, at=point)` wraps the profile onto one face around the point
# `at` and raises it by `height` (sinks it for a negative height). The faces of the wrapped
# feature form the group `wrap`. For comparison, `part.emboss(...)` projects the profile straight
# onto the face, which stretches it where the face turns away.
#
# **Reference.** A strip of width $w$ (around) and height $h$ (along) on a cylinder of radius $R$
# covers the angle $w/R$. Raised by $t$ it adds $V=\tfrac12\,\tfrac{w}{R}\,\big((R+t)^2-R^2\big)\,h$,
# sunk by $t$ it removes $\tfrac12\,\tfrac{w}{R}\,\big(R^2-(R-t)^2\big)\,h$. A cone unrolls without
# stretching, so a patch covers exactly its own area on the side face. A circle of radius $\rho$ on a
# sphere of radius $R_s$ covers the cap $2\pi R_s^2\,(1-\cos(\rho/R_s))$, not the projected
# $\pi\rho^2$.

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Cone, Cylinder, Sphere
from onesim.meshing import TetrahedralMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## A label strip and lettering on a shaft
#
# A shaft of 20 mm radius and 60 mm length. A strip of 30 x 10 mm is wrapped around it across the
# seam of the wall and raised by 2 mm; the same strip sunk by 2 mm; and the letters "ONE", 12 mm
# high, raised by 1 mm on the other side.

radius, length, width, height, rise = 0.02, 0.06, 0.03, 0.01, 0.002
shaft = Cylinder(radius=radius, length=length)
strip = Profile.rectangle(width, height, origin=(-width / 2, -height / 2))
raised = shaft.wrap(
    strip, shaft.boundaries.outer_wall, rise, at=(radius, 0.0, length / 2)
)
sunk = shaft.wrap(
    strip, shaft.boundaries.outer_wall, -rise, at=(radius, 0.0, length / 2)
)
projected = shaft.emboss(
    strip, shaft.boundaries.outer_wall, rise, at=(radius, 0.0, length / 2)
)
letters = Profile.text("ONE", 0.012)
lettered = raised.wrap(
    letters, raised.boundaries.outer_wall, 0.001, at=(-radius, 0.0, length / 2)
)

added = 0.5 * (width / radius) * ((radius + rise) ** 2 - radius**2) * height
removed = 0.5 * (width / radius) * (radius**2 - (radius - rise) ** 2) * height
lettering = (
    sum(item.area for item in letters)
    * ((radius + 0.001) ** 2 - radius**2)
    / (2 * radius)
)
print(
    f"raised strip:  {(raised.measure - shaft.measure) * 1e9:9.3f} mm3 added,   closed form {added * 1e9:9.3f} mm3"
)
print(
    f"sunk strip:    {(shaft.measure - sunk.measure) * 1e9:9.3f} mm3 removed, closed form {removed * 1e9:9.3f} mm3"
)
print(
    f"projected:     {(projected.measure - shaft.measure) * 1e9:9.3f} mm3 added ({(projected.measure - shaft.measure) / added - 1:+.1%}: "
    f"the projection stretches the strip)"
)
print(
    f"lettering:     {(lettered.measure - raised.measure) * 1e9:9.3f} mm3 added,   closed form {lettering * 1e9:9.3f} mm3"
)
print(
    f"faces of the wrapped strip: {len(raised.boundaries.wrap.members)}; lettered shaft: {len(lettered.faces)} faces"
)

mesh = TetrahedralMesher(target_size=0.004).generate(lettered)
view = plots.Camera(elevation=15, azimuth=150)
fig = plots.figure(1, 2, size=(16, 7))
plots.geometry(
    lettered,
    selection={"wrapped strip and letters": lettered.boundaries.wrap},
    labels=False,
    camera=view,
    title="Shaft with a wrapped strip and lettering",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Mesh: {mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## On a cone and on a sphere
#
# A patch of 20 x 12 mm is sunk 1 mm into the side of a cone (30 mm to 15 mm radius, 50 mm high);
# a disc of 12 mm radius is raised 1 mm on a ball of 30 mm radius.

cone = Cone(bottom_radius=0.03, top_radius=0.015, height=0.05)
grooved = cone.wrap(
    Profile.rectangle(0.02, 0.012, origin=(-0.01, -0.006)), cone.boundaries.side, -0.001
)
ball = Sphere(radius=0.03)
skin = ball.faces[0].id
reach = 0.012
capped = ball.wrap(
    Profile.circle(reach), ball.boundaries[skin], 0.001, at=(0.03, 0.0, 0.0)
)
covered = ball.topology.face(skin).area - capped.topology.face(skin).area
print(
    f"cone: side face area covered {(cone.topology.face('side').area - grooved.topology.face('side').area) * 1e6:.4f} mm2, "
    f"patch area {0.02 * 0.012 * 1e6:.4f} mm2"
)
print(
    f"sphere: cap covered {covered * 1e6:.4f} mm2, geodesic circle {2 * np.pi * 0.03**2 * (1 - np.cos(reach / 0.03)) * 1e6:.4f} mm2, "
    f"flat circle {np.pi * reach**2 * 1e6:.4f} mm2"
)

cone_mesh = TetrahedralMesher(target_size=0.004).generate(grooved)
ball_mesh = TetrahedralMesher(target_size=0.004).generate(capped)
for name, body, mesh in (("cone", grooved, cone_mesh), ("sphere", capped, ball_mesh)):
    print(
        f"{name}: {mesh.cell_count} cells, mesh volume {quality(mesh)['size'].sum() * 1e6:.4f} cm3, CAD {body.measure * 1e6:.4f} cm3"
    )

# ## Figure
#
# The raised and the sunk strip against the projected one, and the wrapped cone and sphere.

fig = plots.figure(2, 2, size=(16, 12))
plots.render(
    raised,
    camera=plots.Camera(elevation=15, azimuth=0),
    material="aluminium",
    title="Strip wrapped and raised by 2 mm",
    ax=fig[0],
)
plots.render(
    sunk,
    camera=plots.Camera(elevation=15, azimuth=0),
    material="aluminium",
    title="Strip wrapped and sunk by 2 mm",
    ax=fig[1],
)
plots.geometry(
    grooved,
    selection={"wrapped patch": grooved.boundaries.wrap},
    labels=False,
    camera=plots.Camera(elevation=20, azimuth=-30),
    title="Patch sunk into the side of a cone",
    ax=fig[2],
)
plots.render(
    capped,
    camera=plots.Camera(elevation=10, azimuth=20),
    material="ceramic",
    title="Disc raised by 1 mm on a sphere",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Wrap a grip pattern of many small profiles (`Profile.slot`) around a handle, wrap along the axis
# with `direction=(0, 0, 1)`, or use the wrapped faces as a heated surface in a thermal model.
