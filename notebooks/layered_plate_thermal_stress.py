"""OneSimulation example: thermal stresses in a coated plate (plane strain).
Requires the installed, activated OneSimulation runtime.

A coated plate is cooled from the deposition temperature. The two layers contract differently, so
the plate bends and the coating ends up in compression. The plate is read from a STEP file and cut
to its cross-section; the 2D plane-strain stress field is compared with the closed-form bilayer
solution, and the stress concentration at the free edge is shown.
"""

# # Thermal stresses in a coated plate
#
# A substrate plate (10 mm thick) carries a coating (2 mm thick). The coating is deposited at
# 800 °C, where both layers are stress free, and the plate is then cooled to 150 °C. The substrate
# contracts more than the coating, so the coating ends up in compression and the plate bends. Near
# the free ends the stresses must turn around, which concentrates peel and shear stress at the
# interface.
#
# | layer | $E$ | $\nu$ | $\alpha$ |
# |---|---|---|---|
# | substrate | 130 GPa | 0.28 | $3\cdot10^{-6}$ 1/K |
# | coating | 70 GPa | 0.17 | $5\cdot10^{-7}$ 1/K |
#
# The plate is long in the third direction and is in plane strain, $u_z=0$.
#
# A thin slab of one half of the 60 mm long plate comes from the CAD file
# `layered_plate_thermal_stress.step`; `layered_plate_thermal_stress_coating.step` holds the
# coating, which marks the region of the second material. The script
# `layered_plate_thermal_stress_geometry.py` builds both with the CAD objects of OneSimulation. The
# model is two-dimensional: it lives on the cross-section of the slab, with one region and one
# material per layer.
#
# **Reference.** Away from the ends the plate behaves like a bilayer beam (Timoshenko, 1925). With
# $E_i'=E_i/(1-\nu_i^2)$ and the thermal strain $e_i=(1+\nu_i)\alpha_i\Delta T$,
#
# $$\sigma_{xx}(y)=E_i'\,\bigl(\varepsilon_0+\kappa y-e_i\bigr),\qquad
#   \sum_i\int_i\sigma_{xx}\,dy=0,\qquad \sum_i\int_i\sigma_{xx}\,y\,dy=0 ,$$
#
# where the two balances fix the strain $\varepsilon_0$ and the curvature $\kappa$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid, regionwise
from onesim.meshing import Geometric, cartesian
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.field_transfer import FiniteElementEvaluation
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction
from onesim.physics.solid_mechanics import (
    FixedDisplacementComponents,
    IsotropicThermalExpansion,
    LinearElasticity,
    TractionFree,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the slab and the coating, `section` cuts both at half depth; the regions keep the
# side names `x_min` (mid-span), `x_max` (free end), `y_min` and `y_max`. The quadrilateral mesh has
# a node line on the interface and is graded towards it and towards the free end.

plate, coat = load("layered_plate_thermal_stress.step"), load(
    "layered_plate_thermal_stress_coating.step"
)
half_length, thickness, slab = plate.size
coating = coat.size[1]
substrate = thickness - coating
section = plate.section((0, 0, 1), origin=(0, 0, slab / 2))
coating_region = coat.section((0, 0, 1), origin=(0, 0, slab / 2))
sides = section.boundaries
along = half_length * Geometric(25, ratio=50.0, toward="end").reference()
across = np.r_[
    substrate * Geometric(13, ratio=40.0, toward="end").reference(),
    substrate + coating * Geometric(7, ratio=8.0).reference()[1:],
]
mesh = cartesian(
    geometry=section,
    shape=(along.size - 1, across.size - 1),
    extents=(half_length, thickness),
    node_axes=(tuple(along), tuple(across)),
).build()

fig = plots.figure(1, 2, size=(15, 5))
plots.geometry(
    [plate.named("substrate"), coat.named("coating")],
    camera=plots.Camera(elevation=35, azimuth=-60),
    title="Slab of the half plate",
    ax=fig[0],
)
plots.geometry(
    section,
    title=f"Cross-section with its named sides, meshed with {mesh.cell_count} quadrilaterals",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Plane strain on the cross-section. The mid-span $x=0$ is a symmetry line, held vertically at its
# lower end. The material properties are given region by region. The uniform final temperature
# comes from a heat-conduction model on the same section and enters the mechanics as thermal
# strain.

layers = {"substrate": (130e9, 0.28, 3.0e-6), "coating": (70e9, 0.17, 5.0e-7)}
deposition, final = 1073.15, 423.15
layered = lambda index, unit: regionwise(
    ((coating_region, layers["coating"][index]),),
    default=layers["substrate"][index],
    unit=unit,
)
material = IsotropicSolid(
    density=3000.0,
    specific_heat_capacity=700.0,
    thermal_conductivity=30.0,
    youngs_modulus=layered(0, "Pa"),
    poisson_ratio=layered(1, "1"),
    thermal_expansion_coefficient=layered(2, "1/K"),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)

heat = HeatConduction(domain=section.domain(), material=material)
cooled = Simulation(
    problem=Problem(
        name="Final temperature",
        geometry=section,
        physics=(heat,),
        boundary_conditions=(
            FixedTemperature(boundary=sides.all, physics=heat, value=final),
        ),
    ),
    mesh=mesh,
    numerics=numerics,
).solve_stationary()

solid = LinearElasticity(
    domain=section.domain(),
    material=material,
    kinematics="plane_strain",
    thermal_strain=IsotropicThermalExpansion(reference_temperature=deposition),
)
problem = Problem(
    name="Coated plate",
    geometry=section,
    physics=(solid,),
    boundary_conditions=(
        TractionFree(boundary=sides.x_max | sides.y_min | sides.y_max, physics=solid),
        FixedDisplacementComponents(
            boundary=sides.x_min,
            physics=solid,
            components=("x", "y"),
            where={"y": lambda x, y: y < 1e-9},
        ),
    ),
    couplings=(
        FieldCoupling(
            source=cooled.field(heat.temperature),
            target=solid.temperature,
            transfer=FiniteElementEvaluation(),
        ),
    ),
)
stress = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(solid.stress)
)

# ## Results
#
# The axial stress at mid-span against the bilayer solution of Timoshenko.


def bilayer_stress(y):
    """Axial stress of the bilayer beam at the height y."""
    stiffness, load_vector = np.zeros((2, 2)), np.zeros(2)
    for (E, nu, alpha), (lower, upper) in zip(
        layers.values(), ((0, substrate), (substrate, substrate + coating))
    ):
        moments = [(upper**k - lower**k) / k for k in (1, 2, 3)]
        stiffness += E / (1 - nu**2) * np.array([moments[:2], moments[1:]])
        load_vector += (
            E
            / (1 - nu**2)
            * (1 + nu)
            * alpha
            * (final - deposition)
            * np.array(moments[:2])
        )
    strain, curvature = np.linalg.solve(stiffness, load_vector)
    E, nu, alpha = (
        np.where(np.asarray(y) > substrate, c, s) for s, c in zip(*layers.values())
    )
    return (
        E
        / (1 - nu**2)
        * (strain + curvature * np.asarray(y) - (1 + nu) * alpha * (final - deposition))
    )


heights = {
    "lower face": 2e-6,
    "substrate at the interface": substrate - 2e-6,
    "coating at the interface": substrate + 2e-6,
    "upper face": substrate + coating - 2e-6,
}
at_mid_span = stress.evaluate([[1e-6, y] for y in heights.values()])[:, 0]
print(f"{'position':28s} simulation [MPa]   bilayer [MPa]")
for (label, y), value in zip(heights.items(), at_mid_span):
    print(f"{label:28s} {value / 1e6:16.2f} {bilayer_stress(y) / 1e6:15.2f}")

# ## Figure
#
# The axial stress near one end of the plate, the stress through the thickness at mid-span, and
# the peel and shear stress along the interface towards the free end.

megapascal = lambda s, position: s[:, 0] / 1e6
y = np.linspace(1e-6, substrate + coating - 1e-6, 121)
through = stress.evaluate(np.column_stack((np.full_like(y, 1e-6), y)))[:, 0]
distance = np.geomspace(1e-4, 0.01, 80)
interface = stress.evaluate(
    np.column_stack((half_length - distance, np.full_like(distance, substrate - 2e-6)))
)

fig = plots.figure(1, 3, size=(17, 5))
plots.surface(
    stress,
    component=megapascal,
    colormap="diverging",
    limits=(-110, 110),
    subdivisions=3,
    label="σ_xx [MPa]",
    title="Axial stress near the free end",
    xlimits=(0.01, 0.03),
    xlabel="x from mid-span [m]",
    ylabel="y [m]",
    ax=fig[0],
)
plots.line(
    points={"simulation": (through[::4] / 1e6, 1e3 * y[::4])},
    reference={"bilayer solution": (bilayer_stress(y) / 1e6, 1e3 * y)},
    hlines={"interface": 1e3 * substrate},
    xlabel="σ_xx at mid-span [MPa]",
    ylabel="height [mm]",
    title="Stress through the thickness",
    legend="lower right",
    ax=fig[1],
)
plots.line(
    1e3 * distance,
    {
        "peel stress σ_yy": interface[:, 1] / 1e6,
        "shear stress σ_xy": interface[:, 3] / 1e6,
    },
    scale="logx",
    xlabel="distance from the free end [mm]",
    ylabel="stress at the interface [MPa]",
    title="Stress concentration at the free edge",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Make the coating thinner in `layered_plate_thermal_stress_geometry.py`, write the files again,
# and the stress in it approaches the film limit
# $E_c'(1+\nu)\,\Delta\alpha\,\Delta T$; add a compliant interlayer and watch the edge stresses
# drop.
