"""OneSimulation example: submodel with plasticity at the hole of a steel plate.
Requires the installed, activated OneSimulation runtime.

A steel plate with a hole is computed elastically as a whole; the peak stress at the hole exceeds
the yield stress. A finer submodel of the region around the hole, read from its own STEP file,
takes the displacements of the plate model on its cut faces and is computed with J2 plasticity.
The elastic stress concentration is compared with Peterson's chart, the submodel with the plate
model in the elastic range, and the local stress and plastic strain with Neuber's rule.
"""

# # Submodel with plasticity at a hole
#
# An elastic analysis of a whole structure often shows a few small spots where the stress lies
# above the yield limit. Making the whole model elastoplastic and fine enough for those spots is
# expensive; a **submodel** is cheaper. A small region around the spot is cut out, meshed finely,
# given the plastic material, and driven on its cut faces by the displacements of the global
# model. As long as the plastic zone stays small, the displacements far from it are hardly
# changed by plasticity, which is what makes the transfer valid.
#
# **Setup.** A steel plate 100 mm wide, 200 mm long and 10 mm thick with a central hole of 20 mm
# diameter, pulled by $S=150$ MPa at its ends. Steel: $E=200$ GPa, $\nu=0.3$, J2 plasticity
# with yield stress $\sigma_y=250$ MPa and linear isotropic hardening $H=2$ GPa. One eighth of the
# plate (three symmetry planes) comes from `plate_hole_plastic_submodel.step`, the submodel, a
# 25 mm × 25 mm block around the hole cut from the same eighth, from
# `plate_hole_plastic_submodel_detail.step`; `plate_hole_plastic_submodel_geometry.py` writes both
# with the CAD objects of OneSimulation.
#
# | face | plate model (elastic) | submodel (J2 plasticity) |
# |---|---|---|
# | `x_min`, `y_min`, `z_min` | symmetry | symmetry |
# | `y_max` | traction $S$ | displacement of the plate model × load factor |
# | `x_max` | free | displacement of the plate model × load factor |
# | `z_max`, `hole.outer_wall` | free | free |
#
# **References.** The elastic stress concentration of a finite-width plate with a hole,
# $K_t=\big(2+(1-d/W)^3\big)/(1-d/W)$ on the gross stress (Heywood's fit to Peterson's chart; W.D.
# Pilkey, D.F. Pilkey, *Peterson's Stress Concentration Factors*, 3rd ed., Wiley 2008). For the
# elastoplastic notch root, Neuber's rule $\sigma\,\varepsilon=(K_tS)^2/E$ with the bilinear
# stress-strain curve $\varepsilon=\sigma/E+(\sigma-\sigma_y)/H$ (H. Neuber, J. Appl. Mech. 28,
# 1961, 544-550), an engineering estimate that is usually somewhat conservative.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials.solids import IsotropicHardening, J2Plasticity
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns both parts with their named faces. Curved quadratic tetrahedra: 8 mm in the
# plate with 3 mm at the hole; the submodel with 1.2 mm and, for the mesh study, 0.8 mm at the hole.

plate, detail = load("plate_hole_plastic_submodel.step"), load(
    "plate_hole_plastic_submodel_detail.step"
)
outer, inner = plate.boundaries, detail.boundaries
radius, width = 0.010, 0.100
plate_mesh = TetrahedralMesher(
    target_size=0.008,
    geometry_order=2,
    refinements=(LocalRefinement(region=outer["hole.outer_wall"], target_size=0.003),),
).generate(plate)
hole_sizes = [0.0012, 0.0008]
detail_meshes = [
    TetrahedralMesher(
        target_size=0.003,
        geometry_order=2,
        refinements=(
            LocalRefinement(region=inner["hole.outer_wall"], target_size=size),
        ),
    ).generate(detail)
    for size in hole_sizes
]

view = plots.Camera(elevation=40, azimuth=-60)
fig = plots.figure(2, 2, size=(14, 11))
plots.geometry(
    plate,
    selection={
        "pulled end": outer.y_max,
        "hole": outer["hole.outer_wall"],
        "symmetry": outer.x_min | outer.y_min | outer.z_min,
    },
    labels=False,
    camera=view,
    title="One eighth of the plate",
    ax=fig[0],
)
plots.geometry(
    detail,
    selection={
        "cut faces": inner.x_max | inner.y_max,
        "hole": inner["hole.outer_wall"],
    },
    labels=False,
    camera=view,
    title="Submodel around the hole",
    ax=fig[1],
)
plots.mesh(
    plate_mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Plate: {plate_mesh.cell_count} tetrahedra",
    ax=fig[2],
)
plots.mesh(
    detail_meshes[-1],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Submodel: {detail_meshes[-1].cell_count} tetrahedra",
    ax=fig[3],
)
plots.show()

# ## Global model
#
# The whole plate, elastic, under the full load.

youngs_modulus, poisson, stress, yield_stress, hardening = 200e9, 0.3, 150e6, 250e6, 2e9
whole = SmallStrainSolid(
    youngs_modulus=youngs_modulus, poisson_ratio=poisson, kinematics="volume"
).solve(
    mesh=plate_mesh,
    times=(1.0,),
    order=2,
    displacements=(
        (outer.x_min, "x", 0.0),
        (outer.y_min, "y", 0.0),
        (outer.z_min, "z", 0.0),
    ),
    tractions=((outer.y_max, (0.0, stress, 0.0)),),
)

# ## Submodel
#
# The cut faces take the displacement of the plate model, evaluated at their own nodes and scaled
# by the load factor $t$, which rises in eight steps to 1. The steel is elastoplastic.

transfer = (
    lambda component: lambda x, y, z, t: t
    * whole.evaluate_displacement(np.column_stack((x, y, z)))[:, component]
)
factors = np.linspace(0.0, 1.0, 9)
steel = SmallStrainSolid(
    youngs_modulus=youngs_modulus,
    poisson_ratio=poisson,
    kinematics="volume",
    material=J2Plasticity(
        shear_modulus=youngs_modulus / (2 * (1 + poisson)),
        hardening=IsotropicHardening(yield_stress, hardening),
    ),
)
submodels = [
    steel.solve(
        mesh=mesh,
        times=factors,
        order=2,
        displacements=(
            (inner.x_min, "x", 0.0),
            (inner.y_min, "y", 0.0),
            (inner.z_min, "z", 0.0),
        )
        + tuple(
            (face, axis, transfer(k))
            for face in (inner.x_max, inner.y_max)
            for k, axis in enumerate("xyz")
        ),
    )
    for mesh in detail_meshes
]
plastic = [
    np.array(
        [
            [[point.equivalent_plastic_strain for point in cell] for cell in step]
            for step in result.material_history
        ]
    ).reshape(len(factors), -1)
    for result in submodels
]

# ## Comparison with the references
#
# The elastic peak stress of the plate model and of the submodel in its last elastic step against
# Peterson's chart, then at full load the largest plastic strain and the stress at the same point
# against Neuber's rule.

peterson = (2 + (1 - 2 * radius / width) ** 3) / (1 - 2 * radius / width)
neuber = lambda elastic: np.where(
    elastic <= yield_stress,
    elastic,
    (
        yield_stress / hardening
        + np.sqrt(
            (yield_stress / hardening) ** 2
            + 4 * (1 / youngs_modulus + 1 / hardening) * elastic**2 / youngs_modulus
        )
    )
    / (2 * (1 / youngs_modulus + 1 / hardening)),
)
print(
    f"elastic stress concentration: plate model {whole.stress[0, :, 1, 1].max() / stress:.3f}, "
    f"submodel at t = 0.25 {[result.stress[2, :, 1, 1].max() / (0.25 * stress) for result in submodels][-1]:.3f}, Peterson's chart {peterson:.3f}"
)
print(
    f"elastic peak at full load K_t S = {peterson * stress / 1e6:.0f} MPa, yield stress {yield_stress / 1e6:.0f} MPa; "
    f"Neuber at full load: σ = {neuber(peterson * stress) / 1e6:.1f} MPa, ε_p = {1e3 * (neuber(peterson * stress) - yield_stress) / hardening:.3f}e-3"
)
print(
    f"{'hole mesh':>9s} {'tetrahedra':>10s} {'largest ε_p [1e-3]':>19s} {'σ_yy there [MPa]':>17s} {'plastic points [%]':>19s} {'first yield at t':>17s}"
)
for size, mesh, result, values in zip(hole_sizes, detail_meshes, submodels, plastic):
    worst = values[-1].argmax()
    print(
        f"{1e3 * size:6.1f} mm {mesh.cell_count:10d} {1e3 * values[-1].max():19.3f} {result.stress[-1, worst, 1, 1] / 1e6:17.1f}"
        f" {100 * np.mean(values[-1] > 0):19.2f} {factors[np.argmax(values.max(axis=1) > 0)]:17.3f}"
    )

# ## Figure
#
# The largest plastic strain over the load factor for both submodel meshes, and the stress at the
# point where it occurs, with Neuber's rule and the purely elastic stress.

line = np.linspace(0.0, 1.0, 201)
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    points={
        f"submodel, {mesh.cell_count} tetrahedra": (factors, 1e3 * values.max(axis=1))
        for mesh, values in zip(detail_meshes, plastic)
    },
    reference={
        "Neuber's rule": (
            line,
            1e3
            * (
                neuber(peterson * stress * line)
                - np.minimum(neuber(peterson * stress * line), yield_stress)
            )
            / hardening,
        )
    },
    xlabel="load factor t",
    ylabel="largest plastic strain ε_p [1e-3]",
    title="Plastic strain at the hole",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    points={
        f"submodel, {mesh.cell_count} tetrahedra": (
            factors,
            result.stress[:, values[-1].argmax(), 1, 1] / 1e6,
        )
        for mesh, result, values in zip(detail_meshes, submodels, plastic)
    },
    reference={
        "Neuber's rule": (line, neuber(peterson * stress * line) / 1e6),
        "elastic, K_t S t": (line, peterson * stress * line / 1e6),
    },
    hlines={"yield stress": yield_stress / 1e6},
    xlabel="load factor t",
    ylabel="σ_yy at the point of largest plastic strain [MPa]",
    title="Stress at the hole",
    legend="upper left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Raise the load until the plastic zone reaches the cut faces: then the submodel assumption fails
# and the submodel must grow. Use kinematic hardening and unload to see the residual compressive
# stress that a notch keeps after an overload; or replace the plate by your own part and cut the
# submodel from it in the geometry script.
