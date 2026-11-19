"""Geometry of the OneSimulation example "Floor slab and balcony thermal bridge" (EN ISO 10211 case 3).

Builds the external corner of a building with the CAD objects of OneSimulation: the two wall legs,
the floor slab that runs through the wall into a balcony, and the floor finish. The construction is
written to thermal_bridges_iso10211_cases.step with the face groups alpha (lower room), beta (upper
room), gamma (outside) and delta (cut planes). Four more files hold the regions of the materials
that differ from the interior leaf: thermal_bridges_iso10211_cases_slab.step, ..._finish.step,
..._brick.step and ..._insulation.step.
"""

from onesim.cad.d3 import Box, UnionFromObjects

end, balcony, top, soffit = 1.3, -0.6, 2.15, 1.0


def wall_legs(inner, outer):
    """The two legs of the corner between the distances inner and outer from the outside faces."""
    return UnionFromObjects(
        objects=[
            Box(
                length=outer - inner,
                width=end - inner,
                height=top,
                origin=(inner, inner, 0.0),
            ),
            Box(
                length=end - outer,
                width=outer - inner,
                height=top,
                origin=(outer, inner, 0.0),
            ),
        ]
    )


def environment(face):
    """Environment of a face: alpha lower room, beta upper room, gamma outside, delta cut plane."""
    x, y, z = face.center
    if min(abs(z), abs(z - top), abs(x - end), abs(y - end)) < 1e-9:
        return "delta"
    if abs(x) < 1e-9 or abs(y) < 1e-9 or y < 0:
        return "gamma"
    return "alpha" if z <= soffit + 1e-9 else "beta"


slab = Box(
    length=end - 0.15, width=end - balcony, height=0.15, origin=(0.15, balcony, soffit)
)
finish = Box(length=end - 0.3, width=end - 0.3, height=0.05, origin=(0.3, 0.3, 1.15))
construction = UnionFromObjects(objects=[wall_legs(0.0, 0.3), slab, finish])
construction = construction.grouped(
    **{
        name: construction.select(faces=lambda face: environment(face) == name)
        for name in ("alpha", "beta", "gamma", "delta")
    }
)
construction.save("thermal_bridges_iso10211_cases.step")

regions = {
    "slab": slab,
    "finish": finish,
    "brick": wall_legs(0.0, 0.10),
    "insulation": wall_legs(0.10, 0.15),
}
for name, region in regions.items():
    region.save(f"thermal_bridges_iso10211_cases_{name}.step")
