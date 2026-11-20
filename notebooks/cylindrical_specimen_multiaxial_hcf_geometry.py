"""Geometry of the OneSimulation example "Multiaxial high-cycle fatigue of a cylindrical specimen".

Builds a cylindrical test specimen, 10 mm in diameter and 60 mm long, with the CAD objects of
OneSimulation, names its faces (clamped bottom, loaded top, mantle) and writes
cylindrical_specimen_multiaxial_hcf.step.
"""

from onesim.cad.d3 import Cylinder

specimen = Cylinder(radius=0.005, length=0.060)
specimen = specimen.grouped(
    clamped=specimen.select(
        faces=lambda face: face.kind == "plane" and face.center[2] < 1e-9
    ),
    loaded=specimen.select(
        faces=lambda face: face.kind == "plane" and face.center[2] > 0.060 - 1e-9
    ),
    mantle=specimen.select(faces=lambda face: face.kind == "cylinder"),
)
specimen.save("cylindrical_specimen_multiaxial_hcf.step")
