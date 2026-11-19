"""OneSimulation example: a thin metal plate held hot at one edge and cooled by a liquid on both faces.
Requires the installed, activated OneSimulation runtime.

The plate is 100 times thinner than it is long. It is read from a STEP file and solved as a full 3D
solid; the temperature along the plate and the heat it gives off are compared with the classical
fin solution, which treats the out-of-plane heat loss as a sink in a one-dimensional equation.
"""

# # Out-of-plane heat transfer in a thin plate
#
# An aluminium plate, 100 mm × 100 mm and only 1 mm thick, is held at 100 °C along one edge and
# immersed in a liquid at 20 °C that cools both large faces with a film coefficient
# $h=100$ W/(m² K). The other three edges are insulated. Heat flows along the plate and leaves it
# through its faces, out of the plane.
#
# The plate is so thin that its temperature hardly changes through the thickness: the Biot number
# $hd/k=5\times10^{-4}$ is tiny. Averaged over the thickness, the heat equation becomes the fin
# equation with the out-of-plane loss as a sink,
#
# $$k\,d\,\frac{d^2T}{dx^2}=2h\,(T-T_\infty),$$
#
# which is how thin plates are often reduced to two (here even one) dimensions. This notebook
# solves the full three-dimensional plate instead and checks the reduction.
#
# | face | condition |
# |---|---|
# | `base`, $x=0$ | $T=100$ °C |
# | `faces`, both large faces | $-k\,\partial_nT=h\,(T-T_\infty)$, $h=100$ W/(m² K), $T_\infty=20$ °C |
# | `edges`: the far edge and the two sides | insulated |
#
# The plate comes from the CAD file `thin_plate_fin_cooling.step`; the script
# `thin_plate_fin_cooling_geometry.py` builds it with the CAD objects of OneSimulation.
#
# **Reference.** The fin with an insulated tip (F. P. Incropera, D. P. DeWitt, *Fundamentals of Heat
# and Mass Transfer*, ch. 3.6):
#
# $$\frac{T(x)-T_\infty}{T_b-T_\infty}=\frac{\cosh m(L-x)}{\cosh mL},\qquad
#   Q=\sqrt{2hk\,d}\;W\,(T_b-T_\infty)\tanh mL,\qquad m=\sqrt{\frac{2h}{k\,d}} .$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, Piecewise, StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux,
    FixedTemperature,
    HeatConduction,
    HeatFlux,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the plate with its named faces. Quadratic hexahedra, one through the thickness and
# graded along the plate towards the held edge, where the temperature falls fastest.

plate = load("thin_plate_fin_cooling.step")
faces = plate.boundaries
length, width, thickness = plate.size
grading = (
    Piecewise(((length, Geometric(24, ratio=4.0)),)),
    Piecewise(((width, Geometric(4, ratio=1.0)),)),
    Piecewise(((thickness, Geometric(1, ratio=1.0)),)),
)
mesh = StructuredHexahedralMesher(
    cells=tuple(g.cells for g in grading), grading=grading
).generate(plate)

view = plots.Camera(elevation=30, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 5.5))
plots.geometry(
    plate,
    selection={
        "base, 100 °C": faces.base,
        "cooled faces": faces.faces,
        "insulated edges": faces.edges,
    },
    labels=False,
    camera=view,
    title="Plate with its named faces",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} quadratic hexahedra",
    ax=fig[1],
)
plots.show()

# ## Model and solution

base, liquid, film = 373.15, 293.15, 100.0
aluminium = IsotropicSolid(
    density=2700.0,
    specific_heat_capacity=900.0,
    thermal_conductivity=200.0,
    youngs_modulus=7e10,
    poisson_ratio=0.33,
    thermal_expansion_coefficient=2.3e-5,
)
model = HeatConduction(domain=plate.volume, material=aluminium, heating=0.0)
problem = Problem(
    name="Thin plate cooled on both faces",
    geometry=plate,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=faces.base, physics=model, value=base),
        ConvectiveHeatFlux(
            boundary=faces.faces,
            physics=model,
            heat_transfer_coefficient=film,
            ambient_temperature=liquid,
        ),
        HeatFlux(boundary=faces.edges, physics=model, value=0.0),
    ),
)
temperature = (
    Simulation(
        problem=problem,
        mesh=mesh,
        numerics=Numerics(
            spatial=FiniteElement(element=Lagrange(order=2)),
            linear_solver=SparseDirect(),
        ),
    )
    .solve_stationary()
    .field(model.temperature)
)

# ## Results
#
# The temperature on the mid-plane along the centre line, the difference between the two faces, and
# the heat given off, against the fin solution.

k = aluminium.thermal_conductivity
m = np.sqrt(2 * film / (k * thickness))
fin = lambda x: liquid + (base - liquid) * np.cosh(m * (length - x)) / np.cosh(
    m * length
)
x = np.linspace(0, length, 201)
along = lambda z: np.column_stack((x, np.full_like(x, width / 2), np.full_like(x, z)))
middle, top = (
    temperature.evaluate(along(thickness / 2)).ravel(),
    temperature.evaluate(along(thickness)).ravel(),
)
heat = temperature.boundary_flux(faces.faces)
exact_heat = (
    np.sqrt(2 * film * k * thickness) * width * (base - liquid) * np.tanh(m * length)
)
print(
    f"fin parameter m L = {m * length:.3f}, Biot number h d / k = {film * thickness / k:.1e}"
)
print(
    f"largest deviation of the mid-plane temperature from the fin solution: {np.abs(middle - fin(x)).max():.4f} K"
)
print(
    f"largest difference between mid-plane and face: {np.abs(middle - top).max():.4f} K"
)
print(
    f"tip temperature: simulation {middle[-1] - 273.15:.3f} °C, fin solution {fin(length) - 273.15:.3f} °C"
)
print(
    f"heat given off: simulation {heat:.3f} W, fin solution {exact_heat:.3f} W ({100 * (heat / exact_heat - 1):+.3f} %), "
    f"heat entering at the base {-temperature.boundary_flux(faces.base):.3f} W"
)

# ## Figure
#
# The surface temperature of the plate, and the temperature along the plate against the fin
# solution.

fig = plots.figure(1, 2, size=(15, 6))
plots.surface(
    temperature,
    component=lambda T, position: T - 273.15,
    limits=(20, 100),
    contours=9,
    subdivisions=2,
    camera=view,
    label="T [°C]",
    title="Surface temperature",
    ax=fig[0],
)
plots.line(
    points={"simulation, mid-plane": (1e3 * x[::5], middle[::5] - 273.15)},
    reference={"fin solution": (1e3 * x, fin(x) - 273.15)},
    hlines={"liquid, 20 °C": 20.0},
    xlabel="distance from the held edge x [mm]",
    ylabel="T [°C]",
    title="Temperature along the plate",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Make the plate thicker in `thin_plate_fin_cooling_geometry.py` and write the STEP file again, or
# raise the film coefficient to boiling values: once $hd/k$ approaches 0.1 the temperature varies
# through the thickness and the fin reduction loses accuracy, while the 3D model stays valid.
