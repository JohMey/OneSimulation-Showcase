"""Geometry of the OneSimulation example "Young's modulus and Poisson's ratio from a tensile test".

Builds one eighth of a flat dog-bone tensile specimen with the CAD objects of OneSimulation: a
reduced section 12.5 mm wide, 3 mm thick and 60 mm long, fillets of 12.5 mm radius and grip ends
20 mm wide and 30 mm long. The x axis runs across the width, y along the specimen (the pulling
direction) and z through the thickness; the part keeps x >= 0, y >= 0 and z >= 0. It is written to
tensile_test_parameter_estimation.step with the face groups symmetry_x, symmetry_y (mid-length),
symmetry_z (mid-thickness), grip (the pulled end) and gauge_side (the side face of the reduced
section, where the extensometer measures the contraction of the width).
"""

from onesim.cad.d3 import Box, Cylinder

gauge_width, thickness, gauge_length, radius, grip_width, grip_length = (
    0.0125,
    0.003,
    0.060,
    0.0125,
    0.020,
    0.030,
)
shoulder = 0.012
reduced = Box(length=gauge_width / 2, width=gauge_length / 2, height=thickness / 2)
fillet = Box(
    length=grip_width / 2,
    width=shoulder,
    height=thickness / 2,
    origin=(0.0, gauge_length / 2, 0.0),
) - Cylinder(
    radius=radius,
    length=2 * thickness,
    origin=(gauge_width / 2 + radius, gauge_length / 2, -thickness / 2),
)
grip = Box(
    length=grip_width / 2,
    width=grip_length,
    height=thickness / 2,
    origin=(0.0, gauge_length / 2 + shoulder, 0.0),
)
specimen = reduced | fillet | grip
end = gauge_length / 2 + shoulder + grip_length
plane = lambda axis, value: specimen.select(
    faces=lambda face: face.kind == "plane" and abs(face.center[axis] - value) < 1e-9
)
specimen = specimen.grouped(
    symmetry_x=plane(0, 0.0),
    symmetry_y=plane(1, 0.0),
    symmetry_z=plane(2, 0.0),
    grip=plane(1, end),
    gauge_side=specimen.select(
        faces=lambda face: face.kind == "plane"
        and abs(face.center[0] - gauge_width / 2) < 1e-9
        and face.center[1] < gauge_length / 2
    ),
)
specimen.save("tensile_test_parameter_estimation.step")
