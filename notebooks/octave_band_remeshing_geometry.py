"""Geometry of the OneSimulation example "Remeshing per octave band".

Builds a quarter of the air in a concentric expansion-chamber muffler with the CAD objects of
OneSimulation: a chamber of radius 60 mm and length 250 mm between an inlet and an outlet pipe of
radius 20 mm and length 100 mm, cut in the two symmetry planes x = 0 and y = 0. It writes
octave_band_remeshing.step with the faces inlet, outlet, symmetry and walls.
"""

from onesim.cad.d3 import Box, Cylinder

chamber_radius, chamber_length, pipe_radius, pipe_length = 0.06, 0.25, 0.02, 0.10
muffler = (
    Cylinder(radius=chamber_radius, length=chamber_length).named("chamber")
    | Cylinder(
        radius=pipe_radius, length=pipe_length, origin=(0.0, 0.0, -pipe_length)
    ).named("inlet_pipe")
    | Cylinder(
        radius=pipe_radius, length=pipe_length, origin=(0.0, 0.0, chamber_length)
    ).named("outlet_pipe")
)
quarter = muffler & Box(
    length=0.2, width=0.2, height=0.8, origin=(0.0, 0.0, -0.2)
).named("cut")
faces = quarter.boundaries
inlet, outlet, symmetry = (
    faces["inlet_pipe.bottom"],
    faces["outlet_pipe.top"],
    faces["cut.x_min"] | faces["cut.y_min"],
)
named = (inlet | outlet | symmetry).members
walls = quarter.select(faces=lambda face: face.id not in named)
quarter.grouped(inlet=inlet, outlet=outlet, symmetry=symmetry, walls=walls).save(
    "octave_band_remeshing.step"
)
