"""OneSimulation example: capacitance matrix of two conducting spheres.
Requires the installed, activated OneSimulation runtime.

Two spheres of different size sit close together in air. The 3D electrostatic field gives their
Maxwell capacitance matrix, which is compared with the exact series solution in a mesh study.
"""


# # Capacitance matrix of two spheres
#
# Two conducting spheres with the radii $a=0.1$ m and $b=0.3$ m have their centres $c=0.5$ m
# apart, a gap of only 0.1 m. Their charges depend linearly on both potentials,
#
# $$\begin{pmatrix}Q_1\\Q_2\end{pmatrix}=
#   \begin{pmatrix}C_{11}&C_{12}\\C_{12}&C_{22}\end{pmatrix}\begin{pmatrix}V_1\\V_2\end{pmatrix}.$$
#
# Column $j$ of this **Maxwell capacitance matrix** follows from holding sphere $j$ at 1 V and the
# other at 0 V, solving Laplace's equation $\nabla\cdot(\varepsilon_0\nabla V)=0$ in the air and
# reading the charge on both spheres.
#
# **Reference.** In bispherical coordinates the problem has an exact series solution (W. R. Smythe,
# *Static and Dynamic Electricity*, 1950; J. Lekner, *J. Electrostatics* 69, 2011), with
# $\cosh u=(c^2-a^2-b^2)/(2ab)$:
#
# $$C_{11}=4\pi\varepsilon_0\,ab\sinh u\sum_{n=0}^{\infty}\frac{1}{a\sinh nu+b\sinh(n+1)u},\qquad
#   C_{12}=-4\pi\varepsilon_0\,\frac{ab}{c}\sinh u\sum_{n=1}^{\infty}\frac{1}{\sinh nu},$$
#
# and $C_{22}$ like $C_{11}$ with $a$ and $b$ exchanged.

import numpy as np

from onesim.cad.d3 import Box, Regrouped, Sphere
from onesim.materials import VacuumPermittivity
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()


# ## Parameters and reference

a, b_radius, c = 0.1, 0.3, 0.5
centres = np.array([-c / 2, c / 2])
outer_radius = 4.0
scale = 4 * np.pi * float(VacuumPermittivity().evaluate())


def series_capacitance(a, b, c, terms=200):
    """Maxwell capacitance matrix of two spheres in free space from the bispherical series."""
    u = np.arccosh((c * c - a * a - b * b) / (2 * a * b))
    n, m = np.arange(terms), np.arange(1, terms)
    with np.errstate(over="ignore"):
        c11 = a * b * np.sinh(u) * np.sum(1 / (a * np.sinh(n * u) + b * np.sinh((n + 1) * u)))
        c22 = a * b * np.sinh(u) * np.sum(1 / (b * np.sinh(n * u) + a * np.sinh((n + 1) * u)))
        c12 = -a * b / c * np.sinh(u) * np.sum(1 / np.sinh(m * u))
    return scale * np.array([[c11, c12], [c12, c22]])


C_exact = series_capacitance(a, b_radius, c)


# ## Geometry and mesh
#
# The air is a sphere of 4 m radius, grounded on its surface, with the two conductors removed.
# The two symmetry planes through the common axis reduce it to a quarter. Curved quadratic
# tetrahedra follow the spherical surfaces; local refinement resolves the spheres and the gap.

air = Sphere(radius=outer_radius) & Box(length=2 * outer_radius, width=outer_radius, height=outer_radius,
                                        origin=(-outer_radius, 0.0, 0.0)).named("plane")
for index, (centre, radius) in enumerate(zip(centres, (a, b_radius)), start=1):
    air = air - Sphere(radius=radius, origin=(centre, 0.0, 0.0)).named(f"sphere_{index}")
air = Regrouped(source=air, groups={"outer": ("surface",), "section": ("plane.z_min",),
                                    "symmetry": ("plane.y_min", "plane.z_min")})
faces = air.boundaries
spheres = (faces.sphere_1, faces.sphere_2)


# ## Model and solution
#
# `potential.reaction(boundary)` returns the charge on a conductor of the quarter model. The
# grounded outer sphere replaces infinity; its effect on the potential coefficients $P=C^{-1}$
# is the image term $1/(4\pi\varepsilon_0R)+x_ix_j/(4\pi\varepsilon_0R^3)$, which is added back.

def capacitance_matrix(size):
    """Free-space Maxwell matrix on a mesh with the given cell size at the small sphere."""
    gap_centre = 0.5 * ((centres[0] + a) + (centres[1] - b_radius))
    mesh = TetrahedralMesher(target_size=0.15 * outer_radius, geometry_order=2, max_candidate_cells=1_000_000, refinements=(
        LocalRefinement(region=faces.sphere_1, target_size=size),
        LocalRefinement(region=faces.sphere_2, target_size=2 * size),
        LocalRefinement(region=Sphere(radius=0.12, origin=(gap_centre, 0.0, 0.0)), target_size=size),
        LocalRefinement(region=Sphere(radius=0.6), target_size=6 * size))).generate(air)
    model = PoissonEquation(domain=air.volume, permittivity=VacuumPermittivity())
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2), quadrature_order=5), linear_solver=SparseDirect())
    potentials = []
    for driven in range(2):
        problem = Problem(name="Two-sphere capacitance", geometry=air, physics=(model,), boundary_conditions=(
            FixedPotential(boundary=spheres[driven], physics=model, value=1.0),
            FixedPotential(boundary=spheres[1 - driven] | faces.outer, physics=model, value=0.0),
            ZeroNormalElectricDisplacement(boundary=faces.symmetry, physics=model)))
        potentials.append(Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.potential))
    grounded = np.array([[4.0 * potentials[j].reaction(spheres[i]) for j in range(2)] for i in range(2)])
    image = 1.0 / (scale * outer_radius) + np.outer(centres, centres) / (scale * outer_radius**3)
    return np.linalg.inv(np.linalg.inv(grounded) + image), mesh, potentials[0]


sizes = (0.034, 0.024, 0.017)
matrices, meshes, fields = zip(*(capacitance_matrix(size) for size in sizes))


# ## Results

entries = {"C11": (0, 0), "C22": (1, 1), "C12": (0, 1)}
deviation = {name: [100 * (C[ij] / C_exact[ij] - 1) for C in matrices] for name, ij in entries.items()}
print("tetrahedra   " + "   ".join(f"{name} [pF]  dev. [%]" for name in entries))
for k, mesh in enumerate(meshes):
    print(f"{mesh.cell_count:10d}   " + "   ".join(f"{matrices[k][ij] * 1e12:8.4f}  {deviation[name][k]:+8.3f}"
                                                  for name, ij in entries.items()))
print("exact series " + "   ".join(f"{C_exact[ij] * 1e12:8.4f}          " for ij in entries.values()))
print(f"reciprocity on the finest mesh: C12 / C21 - 1 = {matrices[-1][0, 1] / matrices[-1][1, 0] - 1:+.1e}")


# ## Figure
#
# The potential in the plane through both centres with sphere 1 at 1 V, and the mesh study.

fig = plots.figure(1, 2, size=(14, 6))
plots.surface(fields[-1], boundary=faces.section, symmetry={"planes": [((0, 1, 0), (0, 0, 0))]}, limits=(0.0, 1.0),
              contours=19, subdivisions=3, label="potential [V]", title="Sphere 1 at 1 V, sphere 2 grounded",
              xlimits=(-0.55, 0.65), ylimits=(-0.6, 0.6), ax=fig[0])
plots.convergence([mesh.cell_count for mesh in meshes], {name: np.abs(values) for name, values in deviation.items()},
                  xlabel="curved quadratic tetrahedra (quarter domain)", ylabel="|deviation from the exact series| [%]",
                  title="Mesh study", ax=fig[1])
plots.show()


# ## Going further
#
# Move the spheres closer (`c = 0.42`) and watch $C_{12}$ grow as the gap closes; the mesh then
# needs a smaller cell size in the gap. The circuit-style capacitances follow from the matrix:
# each sphere to ground $C_{11}+C_{12}$ and $C_{22}+C_{12}$, sphere to sphere $-C_{12}$.
