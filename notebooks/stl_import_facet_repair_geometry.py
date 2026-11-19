"""Input file of the OneSimulation example "STL import and facet repair".

Builds a pillow block (base plate, boss, bore, two bolt holes) with the CAD objects of OneSimulation,
writes its surface as triangles and damages it the way exported or downloaded STL files often are:
a third of the triangles face inwards and six triangles are missing. Writes
stl_import_facet_repair.stl.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder, FacetBody, UnionFromObjects

block = UnionFromObjects(
    objects=[
        Box(length=0.12, width=0.05, height=0.015),
        Cylinder(radius=0.022, length=0.03, origin=(0.06, 0.025, 0.015)),
    ]
)
block = block.hole(0.01, at=(0.015, 0.025, 0.015)).hole(0.01, at=(0.105, 0.025, 0.015))
block = block - Cylinder(radius=0.012, length=0.045, origin=(0.06, 0.025, 0.0)).named(
    "bore"
)
skin = block.faceted()
random = np.random.default_rng(3)
triangles = skin.triangles.copy()
inwards = random.random(len(triangles)) < 0.3
triangles[inwards] = triangles[inwards][:, [0, 2, 1]]
missing = np.zeros(len(triangles), dtype=bool)
missing[random.choice(len(triangles), 6, replace=False)] = True
FacetBody(points=skin.points, triangles=triangles[~missing]).save(
    "stl_import_facet_repair.stl"
)
