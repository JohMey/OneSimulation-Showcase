"""OneSimulation example: thick elliptic plate under pressure (NAFEMS benchmark LE10).
Requires the installed, activated OneSimulation runtime.

A thick plate shaped like an elliptic ring bends under a uniform pressure. The stress at the
benchmark point is compared with the published NAFEMS target value on three meshes.
"""


# # Thick elliptic plate under pressure (NAFEMS LE10)
#
# The plate is an elliptic ring: the hole is the ellipse $(x/2)^2+(y/1)^2=1$, the outer edge the
# ellipse $(x/3.25)^2+(y/2.75)^2=1$ (metres), and the plate is 0.6 m thick. A pressure of 1 MPa
# acts on the upper face. Because of symmetry only the quarter $x\ge0,\ y\ge0$ is modelled.
#
# | face | condition |
# |---|---|
# | symmetry faces $x=0$ and $y=0$ | no displacement normal to the face |
# | outer curved face | $u_x=u_y=0$ |
# | mid-height line of the outer face | $u_z=0$ |
# | upper face | pressure 1 MPa |
# | lower face and hole | free |
#
# Steel with $E=210$ GPa and $\nu=0.3$.
#
# **Reference.** NAFEMS, *The Standard NAFEMS Benchmarks*, test LE10: the stress at point
# $D=(2,0,0.3)$, on the loaded face at the end of the hole's long axis, is
# $\sigma_{yy}=-5.38$ MPa.

import numpy as np

from onesim.materials import IsotropicSolid
from onesim.meshing import ExplicitGeometry, Geometric, TwoSided, VolumeBlock, merge_volume_blocks
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacementComponents, LinearElasticity, Pressure, TractionFree
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()

hole, outer_edge, thickness = (2.0, 1.0), (3.25, 2.75), 0.6
steel = IsotropicSolid(density=7850.0, specific_heat_capacity=470.0, thermal_conductivity=50.0,
                       youngs_modulus=210e9, poisson_ratio=0.30, thermal_expansion_coefficient=1.2e-5)


# ## Geometry and mesh
#
# The quarter ring is one block: `plate_map` takes the unit cube onto it, blending the two
# ellipses. The cells are graded towards the hole and towards point D, with a node layer at
# mid-height for the line support.

def plate_map(xi, eta, zeta):
    """Point of the plate for the block coordinates: xi across the ring, eta around, zeta through."""
    a, b = (1 - xi) * hole[0] + xi * outer_edge[0], (1 - xi) * hole[1] + xi * outer_edge[1]
    return a * np.cos(np.pi / 2 * eta), b * np.sin(np.pi / 2 * eta), thickness * (zeta - 0.5)


def face_name(point, normal):
    """Name of a boundary face from its centre."""
    x, y, z = point
    if abs(abs(z) - thickness / 2) < 1e-9:
        return "top" if z > 0 else "bottom"
    if min(abs(x), abs(y)) < 1e-9:
        return "symmetry_y" if abs(y) < 1e-9 else "symmetry_x"
    if abs(np.hypot(x / hole[0], y / hole[1]) - 1) < abs(np.hypot(x / outer_edge[0], y / outer_edge[1]) - 1):
        return "hole"
    return "outer"


plate = ExplicitGeometry(dimension=3, boundary_names=("top", "bottom", "symmetry_x", "symmetry_y", "hole", "outer"),
                         classifier=face_name)
faces = plate.boundaries


def solve(n):
    """Stress field on the mesh with n cells across the ring and around it."""
    through = np.r_[0.5 * Geometric(n // 2, ratio=4.0, toward="end").reference(), 0.5 + 0.5 * TwoSided(n // 2 + 1, 3.0).reference()[1:]]
    block = VolumeBlock(plate_map, Geometric(n, ratio=4.0).reference(), Geometric(n, ratio=3.0).reference(), through)
    mesh = merge_volume_blocks([block], plate, order=2)
    model = LinearElasticity(domain=plate.domain(), material=steel)
    problem = Problem(name="NAFEMS LE10", geometry=plate, physics=(model,), boundary_conditions=(
        FixedDisplacementComponents(boundary=faces.outer, physics=model, components=("x", "y", "z"),
                                    where={"z": lambda x, y, z: abs(z) < 1e-9}),
        FixedDisplacementComponents(boundary=faces.symmetry_x, physics=model, components=("x",)),
        FixedDisplacementComponents(boundary=faces.symmetry_y, physics=model, components=("y",)),
        Pressure(boundary=faces.top, physics=model, value=1.0e6),
        TractionFree(boundary=faces.bottom | faces.hole, physics=model)))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    return Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.stress), mesh


runs = [solve(n) for n in (8, 12, 16)]
stress, mesh = runs[-1]


# ## Results
#
# The stress $\sigma_{yy}$ at point D on the three meshes against the NAFEMS target.

target = -5.38
at_d = [stress_field.evaluate([[hole[0], 0.0, thickness / 2]])[0, 1] / 1e6 for stress_field, _ in runs]
print("hexahedra   σ_yy at D [MPa]   deviation from the NAFEMS target -5.38 MPa")
for (_, level), value in zip(runs, at_d):
    print(f"{level.cell_count:9d} {value:17.4f} {100 * (value / target - 1):+12.2f} %")


# ## Figure
#
# The stress $\sigma_{yy}$ on the loaded face and on the symmetry plane $y=0$, and the mesh study.

megapascal = lambda s, position: s[:, 1] / 1e6
fig = plots.figure(1, 3, size=(16, 5))
plots.surface(stress, boundary=faces.top, component=megapascal, colormap="diverging", limits=(-16, 16), subdivisions=3,
              label="σ_yy [MPa]", title="Loaded face", ax=fig[0])
plots.surface(stress, boundary=faces.symmetry_y, component=megapascal, colormap="diverging", limits=(-8, 8), subdivisions=3,
              label="σ_yy [MPa]", title="Symmetry plane y = 0", ax=fig[1])
plots.line([level.cell_count for _, level in runs], {"simulation": at_d}, hlines={"NAFEMS target": target}, markers=True,
           xlabel="hexahedra", ylabel="σ_yy at D [MPa]", title="Mesh study at point D", legend="center right", ax=fig[2])
plots.show()


# ## Going further
#
# Replace the pressure by a line load on the hole edge, or compare with the thin-plate variant of
# the benchmark (LE1), where the plate is in plane stress.
