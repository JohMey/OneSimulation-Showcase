"""Geometry of the OneSimulation example "Thermal stresses in a coated plate".

Builds a thin slab of one half of the plate (30 mm long, 10 mm substrate with a 2 mm coating,
0.5 mm deep) with the CAD objects of OneSimulation and writes layered_plate_thermal_stress.step;
its faces keep the names x_min, x_max, y_min, y_max, z_min and z_max. The coating, which marks the
region of the second material, is written to layered_plate_thermal_stress_coating.step. The
example computes on the cross-section z = 0.25 mm in plane strain.
"""

from onesim.cad.d3 import Box

half_length, substrate, coating, slab = 0.030, 0.010, 0.002, 0.0005
plate = Box(length=half_length, width=substrate + coating, height=slab)
coat = Box(length=half_length, width=coating, height=slab, origin=(0.0, substrate, 0.0))
plate.save("layered_plate_thermal_stress.step")
coat.save("layered_plate_thermal_stress_coating.step")
