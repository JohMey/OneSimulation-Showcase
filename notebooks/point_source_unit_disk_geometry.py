"""Build the disk carrier and name its lateral wall for the zero-potential rim."""

from onesim.cad.d3 import Cylinder

disk = Cylinder(radius=1.0, length=0.1)
disk.grouped(rim=disk.boundaries.outer_wall).save("point_source_unit_disk.step")
