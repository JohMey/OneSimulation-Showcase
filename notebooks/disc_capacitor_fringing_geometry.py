"""Geometry of the OneSimulation example "Capacitance of a disc capacitor with fringing fields".

Builds one quarter of the capacitor with the CAD objects of OneSimulation: two metal discs (radius
10 cm, 5 mm thick) with round leads on the axis, 3 cm apart, in an air cylinder. The space between
the discs holds a quartz-glass disc. The modelled part is everything that carries a field, the air
together with the glass; the metal is cut out. It is written to disc_capacitor_fringing.step (air
cylinder of 20 cm radius) and disc_capacitor_fringing_wide.step (40 cm) with the face groups
terminal, ground, symmetry and outer. The glass disc, which marks the region of the dielectric, is
written to disc_capacitor_fringing_glass.step.
"""

from onesim.cad.d3 import Box, Cylinder

disc_radius, plate, gap, lead_radius, height = 0.10, 0.005, 0.03, 0.0075, 0.20
lower = 0.08
upper = lower + plate + gap
ground = Cylinder(radius=disc_radius, length=plate, origin=(0, 0, lower)) | Cylinder(
    radius=lead_radius, length=lower
)
terminal = Cylinder(radius=disc_radius, length=plate, origin=(0, 0, upper)) | Cylinder(
    radius=lead_radius, length=height - upper - plate, origin=(0, 0, upper + plate)
)
for radius, name in (
    (0.20, "disc_capacitor_fringing.step"),
    (0.40, "disc_capacitor_fringing_wide.step"),
):
    quadrant = Box(length=radius, width=radius, height=height).named("cut")
    field = (
        (Cylinder(radius=radius, length=height) & quadrant)
        - ground.named("ground")
        - terminal.named("terminal")
    )
    faces = field.boundaries
    field = field.grouped(
        terminal=field.select(faces=lambda face: face.id.startswith("terminal.")),
        ground=field.select(faces=lambda face: face.id.startswith("ground.")),
        symmetry=faces["cut.x_min"] | faces["cut.y_min"],
        outer=faces.bottom | faces.top | faces.outer_wall,
    )
    field.save(name)
glass = Cylinder(radius=disc_radius, length=gap, origin=(0, 0, lower + plate)) & Box(
    length=disc_radius, width=disc_radius, height=height
)
glass.save("disc_capacitor_fringing_glass.step")
