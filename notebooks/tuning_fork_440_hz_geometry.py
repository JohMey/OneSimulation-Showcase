"""Geometry of the OneSimulation example "Tuning fork: the 440 Hz prong mode".

Builds a steel tuning fork with the CAD objects of OneSimulation: two prongs 5 mm x 6 mm in
cross-section and 96 mm long (the length beam theory gives for 440 Hz), 10 mm apart, joined by a
bridge 12 mm high with a U-shaped inner fillet of 4 mm radius, and a round stem 6 mm in diameter and 35 mm long. Writes
tuning_fork_440_hz.step with the face groups tips (the free ends of the prongs) and stem_end (the
lower end of the stem, where the fork is held).
"""

from onesim.cad.d3 import Box, Cylinder

prong, width, length, gap, bridge, fillet = 0.005, 0.006, 0.096, 0.010, 0.012, 0.004
stem_radius, stem_length = 0.003, 0.035
outer = gap / 2 + prong
fork = Box(
    length=2 * outer,
    width=width,
    height=bridge + length,
    origin=(-outer, -width / 2, -bridge),
) - Box(length=gap, width=width, height=length, origin=(-gap / 2, -width / 2, 0.0))
fork = fork.fillet(
    fillet,
    edges=fork.select(
        edges=lambda edge: abs(edge.start[2]) < 1e-9
        and abs(edge.end[2]) < 1e-9
        and abs(abs(edge.start[0]) - gap / 2) < 1e-9
        and abs(edge.start[0] - edge.end[0]) < 1e-9
    ),
)
fork = fork | Cylinder(
    radius=stem_radius,
    length=stem_length + 0.001,
    origin=(0.0, 0.0, -bridge - stem_length),
    axis=(0.0, 0.0, 1.0),
)
fork = fork.grouped(
    tips=fork.select(faces=lambda face: abs(face.center[2] - length) < 1e-9),
    stem_end=fork.select(
        faces=lambda face: abs(face.center[2] + bridge + stem_length) < 1e-9
    ),
)
fork.save("tuning_fork_440_hz.step")
