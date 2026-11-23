"""Build a 78 mm round-pronged steel tuning fork and label its two end faces."""
from math import radians, tan
from pathlib import Path
from tempfile import TemporaryDirectory

from onesim.cad import CadGeometryBuilder
from onesim.cad.d3 import Box, Cone, Cylinder, Sphere, Torus
from onesim.cad.exchange import load

mm = 1e-3
bend, radius, length = 7.5*mm, 2.5*mm, 78*mm
ring = Torus(major_radius=bend, minor_radius=radius, axis=(0, 1, 0))
lower_half = Box(length=0.04, width=0.04, height=0.03, origin=(-0.02, -0.02, -0.03))
stem = Cone(bottom_radius=radius, top_radius=radius-20*mm*tan(radians(2)),
            height=20*mm, origin=(0, 0, -bend), axis=(0, 0, -1))
handle = Sphere(radius=4*mm, origin=(0, 0, -bend-22.5*mm))
prongs = [Cylinder(radius=radius, length=length, origin=(x, 0, 0)) for x in (-bend, bend)]
fork = (ring & lower_half) | stem | handle | prongs[0] | prongs[1]

with TemporaryDirectory() as directory:
    neutral = Path(directory) / "fork.step"
    shape = CadGeometryBuilder().build(fork)
    shape.kernel.export_step(shape.reference, neutral)
    part = load(neutral)
    tips = part.select(faces=lambda face: abs(face.center[2]-length) < 1e-8)
    part.grouped(tips=tips).save("tuning_fork_prong_length_optimization.step")
