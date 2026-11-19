"""Geometry of the OneSimulation example "Dzhanibekov effect: the flipping T-handle".

Builds a T-handle of aluminium bar with the CAD objects of OneSimulation: a crossbar 80 mm long
and 10 mm in diameter along x and a stem 50 mm long and 8 mm in diameter along y, standing on the
middle of the crossbar. Writes dzhanibekov_t_handle.step with the face groups crossbar and stem.
"""

from onesim.cad.d3 import Cylinder

crossbar = Cylinder(
    radius=0.005, length=0.080, origin=(-0.040, 0.0, 0.0), axis=(1.0, 0.0, 0.0)
)
stem = Cylinder(
    radius=0.004, length=0.050, origin=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0)
)
handle = crossbar | stem
handle = handle.grouped(
    crossbar=handle.select(faces=lambda face: face.center[1] < 0.004),
    stem=handle.select(faces=lambda face: face.center[1] >= 0.004),
)
handle.save("dzhanibekov_t_handle.step")
