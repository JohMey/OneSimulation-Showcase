"""Geometry of the OneSimulation example "Turbulent boundary layer on a flat plate".

Builds the air above a flat plate as a slab 0.05 m thick with the CAD objects of OneSimulation and
writes turbulent_flat_plate_zero_pressure_gradient.step. The plate is 2 m long and starts at x = 0;
the stream enters 0.33 m upstream of its leading edge, the far field lies 1 m above it. The faces
are named for the example: inflow, outflow, top, lead (the symmetry plane ahead of the plate),
plate, and the two span faces sides. The example computes on the cross-section z = 0.025 m.
"""

from onesim.cad.d3 import Box

lead = Box(length=0.33, width=1.0, height=0.05, origin=(-0.33, 0.0, 0.0))
plate = Box(length=2.0, width=1.0, height=0.05, origin=(0.0, 0.0, 0.0))
air = lead | plate
bottom = lambda face: face.kind == "plane" and abs(face.center[1]) < 1e-9
air = air.grouped(
    inflow=air.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0] + 0.33) < 1e-9
    ),
    outflow=air.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[0] - 2.0) < 1e-9
    ),
    top=air.select(
        faces=lambda face: face.kind == "plane" and abs(face.center[1] - 1.0) < 1e-9
    ),
    lead=air.select(faces=lambda face: bottom(face) and face.center[0] < 0),
    plate=air.select(faces=lambda face: bottom(face) and face.center[0] > 0),
    sides=air.select(
        faces=lambda face: face.kind == "plane"
        and (abs(face.center[2]) < 1e-9 or abs(face.center[2] - 0.05) < 1e-9)
    ),
)
air.save("turbulent_flat_plate_zero_pressure_gradient.step")
