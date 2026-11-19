"""Geometry of the OneSimulation example "Characteristic impedance of a parallel-wire line".

Builds a 1 mm long piece of the air around a parallel-wire line with the CAD objects of
OneSimulation: two wires of 0.5 mm radius, 5 mm apart (centre to centre), inside an air cylinder
that stands in for the unbounded surroundings. The air cylinder of 125 mm radius is written to
parallel_wire_line_impedance.step, smaller ones of 50 mm and 25 mm radius to
parallel_wire_line_impedance_r50.step and parallel_wire_line_impedance_r25.step. The faces keep the
names left.outer_wall and right.outer_wall (the wire surfaces) and outer_wall (the far boundary).
"""

from onesim.cad.d3 import Cylinder

wire, spacing, length = 0.5e-3, 5.0e-3, 1.0e-3
for far, suffix in ((125e-3, ""), (50e-3, "_r50"), (25e-3, "_r25")):
    air = Cylinder(radius=far, length=length)
    for name, x in (("left", -spacing / 2), ("right", spacing / 2)):
        air = air - Cylinder(radius=wire, length=length, origin=(x, 0.0, 0.0)).named(
            name
        )
    air.save(f"parallel_wire_line_impedance{suffix}.step")
