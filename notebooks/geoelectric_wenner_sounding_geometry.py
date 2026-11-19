"""Geometry of the OneSimulation example "Geoelectrical sounding over a layered earth".

Builds one quarter of the ground below a Wenner electrode array with the CAD objects of
OneSimulation: a quarter of a half ball of 250 m radius, cut by the mid-plane of the array (face
group midplane) and by the vertical plane along the array (symmetry); the ground surface is
surface, the far boundary outer, and the hemispherical current electrode A (radius 25 cm) at
x = 1.5 a is electrode. One file per electrode spacing a: geoelectric_wenner_sounding.step
(a = 8 m) and geoelectric_wenner_sounding_2m.step, ..._4m.step and ..._16m.step. The
basement below the 5 m thick top layer is written to geoelectric_wenner_sounding_basement.step.
"""

from onesim.cad.d3 import Box, Sphere

ground, electrode, layer = 250.0, 0.25, 5.0
quarter = Box(
    length=ground, width=ground, height=ground, origin=(0.0, 0.0, -ground)
).named("cut")
for spacing in (2.0, 4.0, 8.0, 16.0):
    region = (Sphere(radius=ground).named("far") & quarter) - Sphere(
        radius=electrode, origin=(1.5 * spacing, 0.0, 0.0)
    ).named("a")
    faces = region.boundaries
    region = region.grouped(
        electrode=region.select(faces=lambda face: face.id.startswith("a.")),
        midplane=faces["cut.x_min"],
        symmetry=faces["cut.y_min"],
        surface=faces["cut.z_max"],
        outer=region.select(faces=lambda face: face.id.startswith("far")),
    )
    region.save(
        "geoelectric_wenner_sounding.step"
        if spacing == 8.0
        else f"geoelectric_wenner_sounding_{round(spacing)}m.step"
    )
(
    Sphere(radius=ground)
    & Box(
        length=ground, width=ground, height=ground - layer, origin=(0.0, 0.0, -ground)
    )
).save("geoelectric_wenner_sounding_basement.step")
