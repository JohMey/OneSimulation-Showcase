"""Geometry of the OneSimulation example "Low-cycle fatigue of a shaft with a transverse hole".

Builds a steel shaft of 30 mm diameter and 120 mm length along x with a transverse through-hole of
6 mm diameter at mid-length (axis along z) with the CAD objects of OneSimulation and writes
cylinder_hole_low_cycle_fatigue_swt.step. The faces are named clamped (x = 0), loaded (x = 120 mm),
hole and mantle.
"""

from onesim.cad.d3 import Cylinder

radius, length, hole = 0.015, 0.120, 0.003
shaft = Cylinder(radius=radius, length=length, axis=(1.0, 0.0, 0.0)) \
    - Cylinder(radius=hole, length=2 * radius + 0.01, origin=(length / 2, 0.0, -radius - 0.005), axis=(0.0, 0.0, 1.0))
shaft = shaft.grouped(clamped=shaft.select(faces=lambda face: face.kind == "plane" and abs(face.center[0]) < 1e-9),
                      loaded=shaft.select(faces=lambda face: face.kind == "plane" and abs(face.center[0] - length) < 1e-9),
                      hole=shaft.select(faces=lambda face: face.kind == "cylinder" and face.radius < 0.01),
                      mantle=shaft.select(faces=lambda face: face.kind == "cylinder" and face.radius > 0.01))
shaft.save("cylinder_hole_low_cycle_fatigue_swt.step")
