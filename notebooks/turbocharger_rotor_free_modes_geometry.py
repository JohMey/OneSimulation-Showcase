"""Geometry of the OneSimulation example "Free vibration modes of a turbocharger rotor".

Builds a simplified turbocharger rotor with the CAD objects of OneSimulation: a 10 mm shaft, 150 mm
long, with the turbine wheel (44 mm disk) at one end, the compressor wheel (50 mm disk) at the other
and a thrust collar between the bearing seats. The faces of the two wheels and of the shaft are named
and the part is written to turbocharger_rotor_free_modes.step.
"""

from onesim.cad.d3 import Cylinder, UnionFromObjects

shaft = Cylinder(radius=0.005, length=0.150)
turbine = Cylinder(radius=0.022, length=0.016)
collar = Cylinder(radius=0.009, length=0.006, origin=(0.0, 0.0, 0.110))
compressor = Cylinder(radius=0.025, length=0.012, origin=(0.0, 0.0, 0.138))
rotor = UnionFromObjects(objects=[shaft, turbine, collar, compressor])
rotor = rotor.grouped(
    turbine=rotor.select(faces=lambda face: face.center[2] < 0.016 + 1e-9),
    compressor=rotor.select(faces=lambda face: face.center[2] > 0.138 - 1e-9),
    shaft=rotor.select(
        faces=lambda face: face.kind == "cylinder" and face.radius < 0.006
    ),
)
rotor.save("turbocharger_rotor_free_modes.step")
