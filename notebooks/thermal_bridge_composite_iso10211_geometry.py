"""Geometry of the OneSimulation example "Thermal bridge: composite structure (EN ISO 10211 case 2)".

Builds the wall detail, 500 mm wide (x), 47.5 mm thick (y) and 10 mm deep, with the CAD objects of
OneSimulation and writes thermal_bridge_composite_iso10211.step with the faces inside (y = 0, warm
side), outside (y = 47.5 mm, cold side) and the adiabatic cut faces. Three more files hold the
regions of the materials that differ from the insulation: the L-shaped aluminium profile
(..._aluminium.step), the wooden batten (..._wood.step) and the concrete layer (..._concrete.step).
The example computes on the cross-section z = 5 mm of all four.
"""

from onesim.cad.d3 import Box, UnionFromObjects

mm, depth = 1e-3, 0.01
wall = Box(length=500 * mm, width=47.5 * mm, height=depth)
faces = wall.boundaries
wall = wall.grouped(
    inside=faces.y_min,
    outside=faces.y_max,
    adiabatic=faces.x_min | faces.x_max | faces.z_min | faces.z_max,
)
wall.save("thermal_bridge_composite_iso10211.step")

regions = {
    "aluminium": UnionFromObjects(
        objects=[
            Box(length=500 * mm, width=1.5 * mm, height=depth),
            Box(length=1.5 * mm, width=36.5 * mm, height=depth),
            Box(length=15 * mm, width=1.5 * mm, height=depth, origin=(0, 35 * mm, 0)),
        ]
    ),
    "wood": Box(length=15 * mm, width=5 * mm, height=depth, origin=(0, 36.5 * mm, 0)),
    "concrete": Box(
        length=500 * mm, width=6 * mm, height=depth, origin=(0, 41.5 * mm, 0)
    ),
}
for name, region in regions.items():
    region.save(f"thermal_bridge_composite_iso10211_{name}.step")
