"""Build a uniform-wall demonstration chassis and export its solid geometry."""

from onesim.cad.d3 import Box, UnionFromObjects

rail, width, height, flange, wall, cross = 2.4, 0.8, 0.16, 0.06, 0.006, 0.1
stations = (0.2, 0.9, 1.5, 2.2)
plates = [
    plate
    for y0, inward in ((0.0, 1), (width - wall, -1))
    for plate in (
        Box(length=rail, width=wall, height=height, origin=(0.0, y0, 0.0)),
        Box(
            length=rail,
            width=flange - wall,
            height=wall,
            origin=(0.0, y0 + wall if inward > 0 else y0 - flange + wall, 0.0),
        ),
        Box(
            length=rail,
            width=flange - wall,
            height=wall,
            origin=(0.0, y0 + wall if inward > 0 else y0 - flange + wall, height - wall),
        ),
    )
]
plates += [
    Box(
        length=wall,
        width=width - 2 * wall,
        height=cross,
        origin=(x - wall / 2, wall, (height - cross) / 2),
    )
    for x in stations
]
frame = UnionFromObjects(objects=plates)
frame.save("ladder_frame_geometry.step")
