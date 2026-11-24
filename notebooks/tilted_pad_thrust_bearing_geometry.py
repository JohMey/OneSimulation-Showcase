"""Geometry of the OneSimulation example "Tilted pad thrust bearing".

Builds with the CAD objects of OneSimulation the lubricated surface of a thrust-bearing pad as a thin
plate 40 mm long (x, the direction of sliding) with its corner at the origin. File
tilted_pad_thrust_bearing.step holds the square pad, 40 mm wide; the files
tilted_pad_thrust_bearing_w05.step, _w2.step, _w4.step and _w8.step hold pads 0.5, 2, 4 and 8 times
as wide as long. The faces are named inlet (x = 0, where the oil enters), outlet (x = L, the thin end of the
film) and sides (the two edges along the direction of sliding).
"""

from onesim.cad.d3 import Box

length, thickness = 0.04, 0.002
for name, ratio in (("", 1.0), ("_w05", 0.5), ("_w2", 2.0), ("_w4", 4.0), ("_w8", 8.0)):
    pad = Box(length=length, width=ratio * length, height=thickness)
    named = lambda condition: pad.select(faces=condition)
    pad = pad.grouped(
        inlet=named(lambda face: abs(face.center[0]) < 1e-9),
        outlet=named(lambda face: abs(face.center[0] - length) < 1e-9),
        sides=named(
            lambda face: abs(face.center[1]) < 1e-9
            or abs(face.center[1] - ratio * length) < 1e-9
        ),
    )
    pad.save(f"tilted_pad_thrust_bearing{name}.step")
