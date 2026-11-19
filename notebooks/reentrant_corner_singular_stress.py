"""OneSimulation example: the singular stress at a sharp re-entrant corner and how a fillet removes it.
Requires the installed, activated OneSimulation runtime.

An L-shaped plate is clamped at one leg and pulled at the other. The plate is read from a STEP
file and cut to its mid-plane; plane-stress solutions on successively refined meshes show that the
peak stress at the sharp inner corner grows without bound, with the power of the distance that
Williams' corner analysis predicts, while the peak at a rounded corner converges.
"""

# # Singular stress at a re-entrant corner
#
# At a sharp inner corner the material surrounds the corner point over an angle of $270°$. The
# elastic field near such a corner was found by Williams (1952): the stresses vary like
#
# $$\sigma_{ij}\sim K\,r^{\lambda-1}f_{ij}(\theta),$$
#
# where $r$ is the distance from the corner and $\lambda$ the smallest root of the eigenvalue
# equation of the wedge. For the symmetric (opening) field of a wedge of opening $2\alpha$ it is
# $\lambda\sin2\alpha+\sin(2\alpha\lambda)=0$; with $2\alpha=3\pi/2$ the root is
# $\lambda=0.5445$, so the stress grows like $r^{-0.4555}$. The stress at the corner is infinite:
# a finer mesh always shows a higher peak, roughly by $2^{0.4555}=1.37$ for every halving of the
# cells next to the corner. A real part has a fillet, and with it the peak converges to a finite
# value.
#
# The plate, 100 mm x 100 mm with legs 50 mm wide and 5 mm thick, comes in two versions from
# `reentrant_corner_singular_stress.step` (sharp corner) and
# `reentrant_corner_singular_stress_fillet.step` (corner rounded to 2 mm); the script
# `reentrant_corner_singular_stress_geometry.py` builds both with the CAD objects of OneSimulation.
#
# | boundary | condition |
# |---|---|
# | end of the horizontal leg, $x=100$ mm (`fixed`) | $\mathbf u=0$ |
# | end of the vertical leg, $y=100$ mm (`loaded`) | shear traction $(-10\ \mathrm{MPa},\ 0)$, which opens the corner |
# | all other edges | traction free |
#
# Aluminium, $E=70$ GPa, $\nu=0.33$, plane stress.
#
# **Reference.** M. L. Williams, *Stress singularities resulting from various boundary conditions
# in angular corners of plates in extension*, J. Appl. Mech. 19 (1952) 526; the exponent
# $\lambda=0.5445$ for the $270°$ wedge is tabulated there.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.results.derived import voigt, von_mises
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the plates with their named faces, `section` the mid-plane regions. The base mesh
# has 5 mm cells, 0.2 mm along the two edges that meet at the corner (and along the fillet); every
# refinement level splits each triangle into four.

plates = {
    "sharp corner": load("reentrant_corner_singular_stress.step"),
    "2 mm fillet": load("reentrant_corner_singular_stress_fillet.step"),
}
faces = plates["sharp corner"].boundaries
thickness = plates["sharp corner"].size[2]
regions = {
    name: plate.section((0, 0, 1), origin=(0, 0, thickness / 2))
    for name, plate in plates.items()
}
levels = (0, 1, 2, 3)
meshes = {
    name: [
        TriangularMesher(
            target_size=0.005,
            grading=0.25,
            refinements=level,
            local_sizes={
                side: 0.0002
                for side in region.boundary_groups()
                if side.startswith(("cut", "fillet"))
            },
        ).generate(region)
        for level in levels
    ]
    for name, region in regions.items()
}

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    plates["sharp corner"],
    selection={"fixed": faces.fixed, "loaded": faces.loaded, "corner": faces.corner},
    labels=False,
    camera=plots.Camera(elevation=50, azimuth=-60),
    title="L-shaped plate with its named faces",
    ax=fig[0],
)
plots.mesh(
    meshes["sharp corner"][1],
    groups=("all",),
    legend=False,
    title=f"Mid-plane mesh after one refinement: {meshes['sharp corner'][1].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Plane stress with quadratic triangles; the same model on all eight meshes.

plate = SmallStrainSolid(
    youngs_modulus=70e9,
    poisson_ratio=0.33,
    kinematics="plane_stress",
    thickness=thickness,
)
runs = {
    name: [
        plate.solve(
            mesh=mesh,
            times=(0.0,),
            order=2,
            tractions=((regions[name].boundaries.y_max, (-10e6, 0.0)),),
            displacements=(
                (regions[name].boundaries.x_max, "x", 0.0),
                (regions[name].boundaries.x_max, "y", 0.0),
            ),
        )
        for mesh in meshes[name]
    ]
    for name in regions
}

# ## Results
#
# The von Mises stress at the integration points. The peak is taken within 5 mm of the corner,
# away from the clamped end, which has a singularity of its own. Along the bisector of the corner,
# the line into the material at $45°$, the slope of the stress against the distance on a log-log
# scale is the exponent $\lambda-1$.

corner = np.array([0.050, 0.050])
mises = {
    name: [von_mises(voigt(result.stress[0])) for result in results]
    for name, results in runs.items()
}
near = {
    name: [
        np.linalg.norm(result.quadrature_coordinates - corner, axis=1) < 0.005
        for result in results
    ]
    for name, results in runs.items()
}
peaks = {
    name: [
        stress[inside].max() / 1e6 for stress, inside in zip(mises[name], near[name])
    ]
    for name in runs
}
offset = runs["sharp corner"][-1].quadrature_coordinates - corner
distance = np.linalg.norm(offset, axis=1)
bisector = (
    np.abs(np.arctan2(-offset[:, 1], -offset[:, 0]) - np.pi / 4) < np.radians(4)
) & (distance < 0.01)
fitted = bisector & (distance > 5e-5) & (distance < 1e-3)
slope = np.polyfit(
    np.log(distance[fitted]), np.log(mises["sharp corner"][-1][fitted]), 1
)[0]

print(
    "refinement   triangles   peak at the sharp corner [MPa]   peak at the 2 mm fillet [MPa]"
)
for level in levels:
    print(
        f"{level:10d} {meshes['sharp corner'][level].cell_count:11d} {peaks['sharp corner'][level]:32.1f} {peaks['2 mm fillet'][level]:31.1f}"
    )
print(
    f"growth of the sharp-corner peak per refinement: {', '.join(f'{b / a:.3f}' for a, b in zip(peaks['sharp corner'], peaks['sharp corner'][1:]))}"
    f" (Williams: 2^0.4555 = {2**0.4555:.3f})"
)
print(
    f"change of the fillet peak on the last refinement: {100 * (peaks['2 mm fillet'][-1] / peaks['2 mm fillet'][-2] - 1):+.2f} %"
)
print(
    f"exponent along the bisector between 0.05 and 1 mm: {slope:.4f}, Williams λ - 1 = -0.4555, deviation {100 * (slope / -0.4555 - 1):+.1f} %"
)

# ## Figure
#
# The von Mises stress around the corner on the finest mesh of each plate, the stress along the
# bisector against the Williams power law, and the peak stress over the refinement: the sharp
# corner climbs without end, the fillet settles.

cells = {
    name: mises[name][-1].reshape(meshes[name][-1].cell_count, -1).max(axis=1) / 1e6
    for name in runs
}
fig = plots.figure(1, 3, size=(19, 5.5))
plots.mesh(
    meshes["sharp corner"][-1],
    cell_values=cells["sharp corner"],
    edges=False,
    limits=(0, 300),
    label="von Mises stress [MPa]",
    title="Sharp corner, finest mesh",
    ax=fig[0],
)
plots.line(
    points={
        "simulation, sharp corner": (
            1e3 * distance[bisector],
            mises["sharp corner"][-1][bisector] / 1e6,
        )
    },
    reference={
        "Williams, r^-0.4555": (
            np.geomspace(0.01, 10, 50),
            np.exp(
                np.mean(
                    np.log(mises["sharp corner"][-1][fitted] / 1e6)
                    + 0.4555 * np.log(1e3 * distance[fitted])
                )
            )
            * np.geomspace(0.01, 10, 50) ** -0.4555,
        )
    },
    scale="loglog",
    xlabel="distance from the corner along the bisector [mm]",
    ylabel="von Mises stress [MPa]",
    title="Power law along the bisector",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    [meshes["sharp corner"][level].cell_count for level in levels],
    peaks,
    markers=True,
    scale="logx",
    xlabel="triangles",
    ylabel="peak von Mises stress near the corner [MPa]",
    title="Diverging and converging peaks",
    legend="upper left",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the fillet radius in `reentrant_corner_singular_stress_geometry.py` and write the STEP
# files again: the converged peak follows the stress concentration charts for shoulder fillets.
# Evaluate the generalised stress intensity $K$ from the fitted line, or open the corner angle and
# watch the exponent go to zero at $180°$.
