"""Mid-plane carrier of an 80 × 50 mm water-filled fracture."""
from onesim.cad.d3 import Box

Box(length=0.080, width=0.050, height=0.00030, origin=(0.010, 0.020, 0)).save("rock_fracture_reynolds_flow.step")
