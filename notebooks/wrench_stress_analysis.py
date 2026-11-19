"""OneSimulation example: stresses and strains in a combination wrench.
Requires the installed, activated OneSimulation runtime.

A 13 mm wrench grips the hexagon head of a fixed M8 bolt and is loaded by a hand force on the box
end. The part is read from a STEP file, meshed with curved quadratic tetrahedra and checked against
global equilibrium, beam theory in the handle and a mesh study of the peak stress in the throat.
"""

# # Stresses and strains in a wrench
#
# A combination wrench has an open end and a box end. The open end grips the hexagon head of an
# M8 bolt whose shank is held fixed at its far end, and the hand pushes on the box end with
# $F=150$ N perpendicular to the handle, in the plane of the wrench. The torque on the bolt is
# carried through the jaws, the rounded throat of the open end and the neck between head and
# handle.
#
# Wrench and bolt come as one solid from the CAD file `wrench_stress_analysis.step` (bonded jaws,
# as in the original problem statement). The file names three face groups: `fixed`, `hand` and
# `throat`. The script `wrench_stress_analysis_geometry.py` builds the part with the CAD objects of
# OneSimulation and writes the file.
#
# | face | condition |
# |---|---|
# | `fixed`: end face of the bolt shank | $\mathbf u=0$ |
# | `hand`: outer rim of the box end | traction $-p\,\mathbf e_y$ on the upper half, total force $F$ |
# | every other face | traction free |
#
# **Reference.** Global equilibrium (reaction at the fixed face against the applied load),
# Euler-Bernoulli beam theory in the prismatic handle, $\sigma_{xx}=M\,y/I$ with
# $M=F\,(x_F-x)$, and a mesh study of the peak von Mises stress in the two throat fillets.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacement,
    LinearElasticity,
    Traction,
    TractionFree,
)
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the solid with its faces; the named groups of the file are available as
# `wrench.boundaries`. Three meshes of curved quadratic tetrahedra differ in the cell size on the
# two throat fillets; the other round faces get 1.2 mm cells, the rest 3 mm.

wrench = load("wrench_stress_analysis.step")
faces = wrench.boundaries
other_round_faces = (
    wrench.select(faces=lambda face: face.kind == "cylinder") - faces.throat
)
sizes = [0.0008, 0.00053, 0.00036]
meshes = [
    TetrahedralMesher(
        target_size=0.003,
        geometry_order=2,
        refinements=(
            LocalRefinement(region=other_round_faces, target_size=0.0012),
            LocalRefinement(region=faces.throat, target_size=size),
        ),
    ).generate(wrench)
    for size in sizes
]

view = plots.Camera(elevation=48, azimuth=-72)
fig = plots.figure(1, 2, size=(16, 6))
plots.geometry(
    wrench,
    selection={"fixed": faces.fixed, "hand": faces.hand, "throat": faces.throat},
    labels=False,
    camera=view,
    title="Wrench and bolt with the named faces",
    ax=fig[0],
)
plots.mesh(
    meshes[-1],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Finest mesh: {meshes[-1].cell_count} curved tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Structural steel. The hand force is spread evenly over the upper half of the box-end rim, so the
# traction is the force divided by half the area of that face. The same problem is solved on the
# three meshes.

force, thickness, handle_height, ring_centre, ring_radius = (
    150.0,
    0.005,
    0.011,
    0.140,
    0.0105,
)
steel = IsotropicSolid(
    density=7850.0,
    specific_heat_capacity=475.0,
    thermal_conductivity=44.5,
    youngs_modulus=200e9,
    poisson_ratio=0.30,
    thermal_expansion_coefficient=12.3e-6,
)
model = LinearElasticity(domain=wrench.volume, material=steel)
hand_area = sum(face.area for face in wrench.faces if face.id in faces.hand.members)
pressure = force / (hand_area / 2)
problem = Problem(
    name="Wrench",
    geometry=wrench,
    physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=faces.fixed, physics=model, value=(0.0, 0.0, 0.0)),
        Traction(
            boundary=faces.hand,
            physics=model,
            value=lambda x, y, z: np.stack(
                (0 * y, np.where(y > 0, -pressure, 0.0), 0 * y), axis=-1
            ),
        ),
        TractionFree(boundary=faces.all - faces.fixed - faces.hand, physics=model),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
results = [
    Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    for mesh in meshes
]
displacement, stress, mises = (
    results[-1].field(f)
    for f in (model.displacement, model.stress, model.von_mises_stress)
)

# ## Results
#
# The reaction at the fixed shank end must balance the hand force. In the handle the bending stress
# follows beam theory; the line of action of the hand force is the centroid of the loaded arc.

half_angle = hand_area / 2 / (ring_radius * thickness)
line_of_action = ring_centre + ring_radius * np.sin(half_angle) / half_angle
y = np.linspace(-handle_height / 2 + 1e-6, handle_height / 2 - 1e-6, 23)
bending = stress.evaluate(
    np.column_stack((np.full_like(y, 0.0736), y, np.zeros_like(y)))
)[:, 0]
beam = force * (line_of_action - 0.0736) * y / (thickness * handle_height**3 / 12)
peaks = [
    result.field(model.von_mises_stress)
    .sample(along=faces.throat, subdivisions=4)
    .values.max()
    for result in results
]
study = convergence(peaks, sizes)

reaction = displacement.reaction(faces.fixed)
print(
    f"reaction at the fixed face: ({reaction[0]:.2f}, {reaction[1]:.2f}, {reaction[2]:.2f}) N against the hand force of {force:.0f} N"
)
print(
    f"largest displacement: {1e3 * np.abs(displacement.sample(along=faces.hand).values).max():.3f} mm"
)
print(
    f"bending stress at the handle surface: simulation {np.polyfit(y, bending, 1)[0] * handle_height / 2 / 1e6:.2f} MPa, "
    f"beam theory {beam[-1] / 1e6:.2f} MPa"
)
print("fillet cell size [mm]   tetrahedra   peak von Mises stress in the throat [MPa]")
for size, mesh, peak in zip(sizes, meshes, peaks):
    print(f"{1e3 * size:20.2f} {mesh.cell_count:12d} {peak / 1e6:28.2f}")
print(
    f"change between the two finest meshes {100 * study.change:.1f} %, observed order {study.order:.1f}, "
    f"extrapolated peak {study.extrapolated / 1e6:.0f} MPa"
)

# ## Figure
#
# The von Mises stress on the deformed wrench (displacement magnified 8 times) and in the mid-plane
# of the open end, where the two throat fillets carry the peak; the bending stress across the
# handle and the mesh study of the fillet peak.

megapascal = lambda s, position: s / 1e6
fig = plots.figure(2, 2, size=(16, 11))
plots.surface(
    mises,
    component=megapascal,
    deformation=displacement,
    deformation_scale=8.0,
    limits=(0, 200),
    camera=view,
    label="von Mises stress [MPa]",
    title="Deformed wrench",
    ax=fig[0],
)
plots.section(
    mises,
    plane=Plane(origin=(0, 0, 0), normal=(0, 0, 1)),
    component=megapascal,
    limits=(0, 200),
    subdivisions=3,
    label="von Mises stress [MPa]",
    title="Mid-plane of the open end with the throat fillets",
    xlimits=(-0.016, 0.036),
    ylimits=(-0.019, 0.019),
    ax=fig[1],
)
plots.line(
    points={"simulation": (1e3 * y, bending / 1e6)},
    reference={"beam theory": (1e3 * y, beam / 1e6)},
    xlabel="y [mm]",
    ylabel="bending stress [MPa]",
    title="Handle section at x = 73.6 mm",
    ax=fig[2],
)
plots.line(
    1e3 * np.array(sizes),
    {"simulation": np.array(peaks) / 1e6},
    markers=True,
    hlines={"extrapolated to zero cell size": study.extrapolated / 1e6},
    xlabel="cell size in the throat fillets [mm]",
    ylabel="peak von Mises stress [MPa]",
    title="Mesh study of the throat fillets",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Reduce the throat fillet radius in `wrench_stress_analysis_geometry.py`, write the STEP file again
# and follow the peak stress; shorten the shank to see the support act on the torsion field; or
# replace the bonded jaws by frictional contact (`onesim.physics.solid_mechanics`).
