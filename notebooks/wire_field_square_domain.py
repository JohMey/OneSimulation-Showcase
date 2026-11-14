"""OneSimulation example: magnetic field of a long wire in a square, magnetically insulated air domain.
Requires the installed, activated OneSimulation runtime.

A straight round conductor runs through the centre of a square box whose walls carry the return
current. The field is computed with edge elements and compared with the exact conformal-mapping
solution and the exact inductance per unit length.
"""


# # Magnetic field of a long wire in a square air domain
#
# A round conductor of radius $a=1$ cm carries $I=1$ A uniformly over its cross-section. It runs
# through the centre of a square air region of side $s=20$ cm with magnetic insulation,
# $\mathbf n\times\mathbf A=0$, on its walls. Near the wire the field is the familiar circular
# field; towards the walls it is pushed into the shape of the square, stronger towards the side
# midpoints and weaker towards the corners.
#
# The model solves $\nabla\times(\mu_0^{-1}\nabla\times\mathbf A)=\mathbf J$ with second-order
# Nédélec edge elements and a Coulomb gauge, on a slab of the cross-section that is one layer of
# tetrahedra thick.
#
# **Reference.** The Schwarz-Christoffel map $z(w)=R_c\,w\,{}_2F_1(\tfrac14,\tfrac12;\tfrac54;-w^4)$
# takes the unit disc onto the square; $R_c=0.53935\,s$ is the conformal radius. Outside the wire
# the vector potential is $A_z=-\frac{\mu_0I}{2\pi}\ln|w|$, and the inductance per unit length is
# exactly
#
# $$L'=\frac{\mu_0}{2\pi}\left[\frac14+\ln\frac{R_c}{a}\right].$$

import numpy as np
from scipy.special import hyp2f1

from onesim.cad.d2 import Plane
from onesim.cad.d3 import Cylinder
from onesim.materials import VacuumPermeability
from onesim.meshing import Geometric, disc_blocks, extrude, merge_blocks, o_grid
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import CoulombGauge, MagneticInsulation, MagneticVectorPotential, Magnetostatics, RegionCurrent
from onesim.results.integration import mean_value, measure
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()

wire_radius, side, current = 0.01, 0.20, 1.0
mu0 = float(VacuumPermeability().evaluate())
conformal_radius = 0.53935 * side
exact_inductance = mu0 / (2 * np.pi) * (0.25 + np.log(conformal_radius / wire_radius))


def exact_field(x, y):
    """|B| from the conformal map of the disc onto the square (linear law inside the wire)."""
    z = (x + 1j * y) / conformal_radius
    w = z.copy()
    for _ in range(60):
        w = w - (w * hyp2f1(0.25, 0.5, 1.25, -w**4) - z) * np.sqrt(1 + w**4)
    outside = mu0 * current / (2 * np.pi * conformal_radius) * np.abs(np.sqrt(1 + w**4) / w)
    return np.where(np.hypot(x, y) < wire_radius, mu0 * current * np.hypot(x, y) / (2 * np.pi * wire_radius**2), outside)


# ## Mesh and model
#
# `disc_blocks` meshes the wire, `o_grid` the air between the wire and the square walls with cells
# that grow with the distance from the wire. The quadrilaterals are split into triangles and
# extruded into one layer of tetrahedra. The wire carries a uniform axial current density.

def solve(n):
    """Magnetic flux density on the mesh with n cells per quarter of the wire circumference."""
    thickness = 2.5 * np.pi * wire_radius / (2 * n)
    blocks = disc_blocks((0, 0), (wire_radius,), n, n // 2) + o_grid(
        (0, 0), wire_radius, side / 2, n, 3 * n // 2, radial_grading=Geometric(3 * n // 2, ratio=10.0),
        inner_name=None, outer_names=("walls",) * 4)
    mesh = extrude(merge_blocks(blocks).triangulated(), thickness, 1, names=("front", "back"))
    faces = mesh.geometry.boundaries
    wire_area = 2 * n * wire_radius**2 * np.sin(np.pi / (2 * n))
    wire = RegionCurrent.axial(Cylinder(radius=wire_radius, length=thickness), current, area=wire_area,
                               within=mesh.geometry.domain())
    model = Magnetostatics(domain=mesh.geometry.domain(), permeability=VacuumPermeability(),
                           formulation=MagneticVectorPotential(gauge=CoulombGauge()), current_sources=(wire,))
    problem = Problem(name="Wire in an insulated square", geometry=mesh.geometry, physics=(model,), boundary_conditions=(
        MagneticInsulation(boundary=faces.walls | faces.front | faces.back, physics=model),))
    numerics = Numerics(spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=SparseDirect())
    field = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.magnetic_flux_density)
    return field, mesh, thickness


runs = {4 * n: solve(n) for n in (8, 12, 16)}
field, mesh, thickness = runs[64]


# ## Results
#
# The inductance per unit length follows from the field energy, $L'=2W'/I^2$ with
# $W'=\int B^2/(2\mu_0)\,\mathrm dA$.

squared = lambda B, position: (B**2).sum(axis=1)
inductance = {cells: mean_value(B, component=squared) * measure(B) / (mu0 * current**2 * t) for cells, (B, _, t) in runs.items()}
print("cells around the wire   tetrahedra   L' [nH/m]   deviation from the exact value")
for cells, value in inductance.items():
    print(f"{cells:21d} {runs[cells][1].cell_count:12d} {1e9 * value:11.3f}   {100 * (value / exact_inductance - 1):+.4f} %")
print(f"{'exact':>21s} {'':12s} {1e9 * exact_inductance:11.3f}")


# ## Figure
#
# The field strength over the cross-section, the field along a line towards a side midpoint and
# towards a corner against the exact solution, and the mesh study of the inductance.

strength = lambda B, position: 1e6 * np.hypot(B[:, 0], B[:, 1])
lines = {"towards a side": (1.0, 0.0, 0.099), "towards a corner": (2**-0.5, 2**-0.5, 0.139)}
computed, exact = {}, {}
for label, (dx, dy, reach) in lines.items():
    coarse, fine = np.linspace(0.003, reach, 24), np.linspace(0.0005, reach, 300)
    points = np.column_stack((dx * coarse, dy * coarse, np.full_like(coarse, thickness / 2)))
    computed[label] = (100 * coarse, strength(field.evaluate(points), points))
    exact[label] = (100 * fine, 1e6 * exact_field(dx * fine, dy * fine))

fig = plots.figure(1, 3, size=(17, 5))
plots.section(field, plane=Plane(origin=(0, 0, thickness / 2), normal=(0, 0, 1)), component=strength, limits=(0, 20),
              contours=10, subdivisions=3, label="|B| [µT]", title="Field strength", ax=fig[0])
plots.line(points=computed, reference=exact, vlines={"wire surface": 1.0}, scale="logy", xlabel="distance from the axis [cm]",
           ylabel="|B| [µT]", title="Simulation (markers) and exact field", ax=fig[1])
plots.convergence(list(inductance), {"L' from the field energy": [abs(value / exact_inductance - 1) for value in inductance.values()]},
                  order=-2, xlabel="cells around the wire", ylabel="relative deviation from the exact L'",
                  title="Mesh study", ax=fig[2])
plots.show()


# ## Going further
#
# Shrink the square to see how tight magnetic insulation deforms the field and lowers the
# inductance, move the wire off centre, or give it a permeability $\mu_r\neq1$.
