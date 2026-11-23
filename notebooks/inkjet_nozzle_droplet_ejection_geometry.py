"""Geometry of the OneSimulation example "Droplet ejection from an inkjet nozzle".

Builds the fluid space of a piezo inkjet nozzle as half a solid of revolution about the z axis with
the CAD objects of OneSimulation, drawn in micrometres: the ink chamber (radius 60 um), the conical
nozzle narrowing to the orifice of radius 20 um at z = 0, and the air above the nozzle plate
(radius 100 um, 700 um high) into which the drop is fired. Writes
inkjet_nozzle_droplet_ejection.step with the face groups inlet (bottom of the chamber, where the
actuator pushes the ink), nozzle (chamber, cone and orifice walls), plate (outer face of the nozzle
plate), open (side and top of the air space) and axis (the cut through the axis).
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

outline = Path((0.0, -100.0), (Segment((60.0, -100.0), name="inlet"), Segment((60.0, -60.0), name="nozzle"),
                               Segment((20.0, -20.0), name="nozzle"), Segment((20.0, 0.0), name="nozzle"),
                               Segment((100.0, 0.0), name="plate"), Segment((100.0, 700.0), name="open"),
                               Segment((0.0, 700.0), name="open"), Segment((0.0, -100.0), name="axis")))
nozzle = Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2)
nozzle = nozzle.grouped(axis=nozzle.boundaries.start | nozzle.boundaries.end)
nozzle.save("inkjet_nozzle_droplet_ejection.step")
