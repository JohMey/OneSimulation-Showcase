"""Geometry of the OneSimulation example "Moisture uptake and swelling stresses in a paperboard roll".

Builds a 90-degree sector of one half of a paperboard roll (core radius 50 mm, outer radius 400 mm,
half width 500 mm, axis along z, mid-plane at z = 0) with the CAD objects of OneSimulation and writes
paperboard_roll_moisture_swelling.step. The example computes on its meridian section y = 0. The
faces are named core, mantle, end (z = 500 mm) and middle (z = 0).
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder

roll = AnnularCylinder(inner_radius=0.05, outer_radius=0.40, length=0.50)
sector = roll.sector(np.pi / 2, axis=(0.0, 0.0, 1.0), start=(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0))
sector = sector.grouped(core=sector.select(faces=lambda face: face.kind == "cylinder" and face.radius < 0.2),
                        mantle=sector.select(faces=lambda face: face.kind == "cylinder" and face.radius > 0.2),
                        middle=sector.select(faces=lambda face: abs(face.center[2]) < 1e-9),
                        end=sector.select(faces=lambda face: abs(face.center[2] - 0.50) < 1e-9))
sector.save("paperboard_roll_moisture_swelling.step")
