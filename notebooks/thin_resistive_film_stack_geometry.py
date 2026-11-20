"""Geometry of the OneSimulation example "Current through a thin resistive film in a layered device".

Builds a layered device with the CAD objects of OneSimulation: a base plate of 1 m x 1 m x 0.1 m, a
resistive film of 20 mm and a cover plate of 0.1 m that carries a round contact pad of 0.3 m radius
and 0.1 m height off the centre. The bottom face of the base is the group inlet, the top of the pad
the group contact. thin_resistive_film_stack_resolved.step holds the three parts base, film and cover;
thin_resistive_film_stack_thin.step holds base and cover only, which meet in the middle plane of the
film.
"""

from onesim.cad.d3 import Assembly, Box, Cylinder

side, plate, film, pad = 1.0, 0.1, 0.02, 0.3
top = lambda face: face.kind == "plane" and abs(face.center[2] - 2 * plate) < 1e-9


def parts(bottom_of_cover, top_of_base):
    base = Box(
        length=side,
        width=side,
        height=top_of_base + film + plate,
        origin=(0.0, 0.0, -film - plate),
    )
    cover = Box(
        length=side,
        width=side,
        height=plate - bottom_of_cover,
        origin=(0.0, 0.0, bottom_of_cover),
    )
    cover = cover | Cylinder(radius=pad, length=plate, origin=(0.4, 0.6, plate))
    return base.grouped(inlet=base.boundaries.z_min).named("base"), cover.grouped(
        contact=cover.select(faces=top)
    ).named("cover")


base, cover = parts(0.0, -film)
film_layer = Box(length=side, width=side, height=film, origin=(0.0, 0.0, -film)).named(
    "film"
)
Assembly(parts=(base, film_layer, cover)).save(
    "thin_resistive_film_stack_resolved.step"
)
Assembly(parts=parts(-film / 2, -film / 2)).save("thin_resistive_film_stack_thin.step")
