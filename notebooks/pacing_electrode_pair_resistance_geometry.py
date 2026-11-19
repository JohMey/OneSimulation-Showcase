"""Geometry of the OneSimulation example "Pacemaker electrodes: current distribution and pair resistance".

Builds one quarter of the tissue below the surface on which two hemispherical electrodes (radius
1 mm) sit, with the CAD objects of OneSimulation: a quarter of a half ball of 100 mm radius, cut by
the mid-plane between the electrodes (face group midplane) and by the plane through both electrode
centres (symmetry); the tissue surface is surface, the far boundary outer and the electrode
electrode. One file per centre distance: pacing_electrode_pair_resistance.step (8 mm) and
pacing_electrode_pair_resistance_4mm.step, ..._16mm.step.
"""

from onesim.cad.d3 import Box, Sphere

electrode, tissue = 1.0e-3, 0.1
quarter = Box(
    length=tissue, width=tissue, height=tissue, origin=(0.0, 0.0, -tissue)
).named("cut")
for distance in (0.004, 0.008, 0.016):
    region = (Sphere(radius=tissue).named("far") & quarter) - Sphere(
        radius=electrode, origin=(distance / 2, 0.0, 0.0)
    ).named("tip")
    faces = region.boundaries
    region = region.grouped(
        electrode=region.select(faces=lambda face: face.id.startswith("tip")),
        midplane=faces["cut.x_min"],
        symmetry=faces["cut.y_min"],
        surface=faces["cut.z_max"],
        outer=region.select(faces=lambda face: face.id.startswith("far")),
    )
    region.save(
        "pacing_electrode_pair_resistance.step"
        if distance == 0.008
        else f"pacing_electrode_pair_resistance_{round(1e3 * distance)}mm.step"
    )
