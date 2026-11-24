"""Geometry of the OneSimulation example "Luneburg lens with a mapped permittivity".

Builds, as thin plates with the CAD objects of OneSimulation, the air around a cylindrical lens out to
a circle of radius 165 mm (luneburg_lens_mapped_permittivity.step, with the outer face named outer)
and the lens, a disc of radius 90 mm (luneburg_lens_mapped_permittivity_lens.step).
"""

from onesim.cad.d3 import Cylinder

lens, outer, thickness = 0.09, 0.165, 0.005
air = Cylinder(radius=outer, length=thickness)
air = air.grouped(outer=air.select(faces=lambda face: face.kind == "cylinder"))
air.save("luneburg_lens_mapped_permittivity.step")
Cylinder(radius=lens, length=thickness).save(
    "luneburg_lens_mapped_permittivity_lens.step"
)
