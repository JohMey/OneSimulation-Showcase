"""OneSimulation example: magnetic prospecting of a buried iron ore body (3D magnetostatics).
Requires the installed, activated OneSimulation runtime.

A magnetite body 100 m below the surface is magnetized by the inclined field of the Earth and
disturbs it. The ground and the ore body are read from STEP files; the total-field anomaly that a
magnetometer would record on the surface is computed with edge elements and compared with the
exact solution for a permeable sphere.
"""

# # Magnetic prospecting of an iron ore body
#
# Magnetic prospecting finds ore bodies rich in magnetite by the small changes, the anomalies, they
# cause in the magnetic field of the Earth. A magnetometer on the surface records the total-field
# anomaly
#
# $$\Delta T=|\mathbf B|-|\mathbf B_0|\approx(\mathbf B-\mathbf B_0)\cdot\hat{\mathbf B}_0 ,$$
#
# the change of the field strength along the direction of the undisturbed field $\mathbf B_0$.
#
# Here the Earth's field has a strength of 50 µT and an inclination of 60° (a mid-latitude of the
# northern hemisphere), pointing north ($+x$) and downwards. The ore body is a sphere of radius
# $a=50$ m with its centre 100 m below the surface and a magnetic susceptibility $\chi=0.5$
# ($\mu_r=1.5$, a magnetite-rich ore). Rock and air are non-magnetic. Without currents,
# $\nabla\times\mathbf H=0$, $\nabla\cdot\mathbf B=0$; OneSimulation solves for the vector potential
# with second-order Nédélec elements and a Coulomb gauge.
#
# The field is symmetric about the magnetic meridian plane through the body, so half of the
# surroundings is modelled: a half ball of 800 m radius about the centre of the body, from
# `magnetic_prospecting_ore_body.step` with the face groups `meridian` and `outer`; the half ore
# body is in `magnetic_prospecting_ore_body_ore.step`. The script
# `magnetic_prospecting_ore_body_geometry.py` builds both with the CAD objects of OneSimulation.
#
# | faces | condition |
# |---|---|
# | `meridian` (the plane $y=0$) | field tangential, $\mathbf n\times\mathbf A=0$ |
# | `outer`, 800 m from the body | undisturbed Earth field, $\mathbf A=\tfrac12\mathbf B_0\times\mathbf r$ |
#
# **Reference.** A permeable sphere in a uniform field is magnetized uniformly; outside, it acts as
# a dipole at its centre (Telford, Geldart and Sheriff, *Applied Geophysics*, 2nd ed., 1990,
# Sec. 3.7; Blakely, *Potential Theory in Gravity and Magnetic Applications*, 1995, Ch. 4):
#
# $$\mathbf B=\mathbf B_0+K\,\frac{a^3}{r^3}\big[3(\mathbf B_0\cdot\hat{\mathbf r})\hat{\mathbf r}-\mathbf B_0\big],
#   \qquad K=\frac{\mu_r-1}{\mu_r+2}.$$
#
# The outer boundary sees the undisturbed field only; the dipole field it misses there is about
# 1 nT, a thousandth of the anomaly.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.d3 import Box
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    CoulombGauge,
    MagneticInsulation,
    MagneticPotentialBoundary,
    MagneticVectorPotential,
    Magnetostatics,
)
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the half ball with its named faces and the half ore body. The mesher embeds the
# body and refines a layer around the surface where the magnetometer measures; elsewhere the cells
# grow with the distance from the body. Two meshes differ in their cell sizes.

region, ore = load("magnetic_prospecting_ore_body.step"), load(
    "magnetic_prospecting_ore_body_ore.step"
)
faces = region.boundaries
radius, depth = ore.size[0] / 2, 100.0
sizes = (0.36 * radius, 0.25 * radius)
meshes = [
    TetrahedralMesher(
        target_size=150.0,
        embedded_regions=(ore,),
        refinements=(
            LocalRefinement(region=ore, target_size=size),
            LocalRefinement(
                region=Box(
                    length=800.0,
                    width=400.0,
                    height=60.0,
                    origin=(-400.0, 0.0, depth - 30.0),
                ),
                target_size=2.5 * size,
            ),
        ),
        size_field=lambda points, size=size: np.clip(
            size * (np.linalg.norm(points, axis=1) / radius) ** 1.5, size, 150.0
        ),
    ).generate(region)
    for size in sizes
]

view = plots.Camera(elevation=20, azimuth=-120)
fig = plots.figure(1, 2, size=(14, 6.5))
plots.geometry(
    [region.named("ground and air"), ore.named("ore body")],
    camera=view,
    title="Half of the surroundings of the ore body",
    ax=fig[0],
)
plots.mesh(
    meshes[0],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Coarsest mesh: {meshes[0].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `regionwise` gives the ore body its permeability; the Earth's field enters through the vector
# potential on the outer face.

earth, inclination, mu_r = 50e-6, np.radians(60.0), 1.5
B0 = earth * np.array((np.cos(inclination), 0.0, -np.sin(inclination)))
mu0 = float(VacuumPermeability().evaluate())
model = Magnetostatics(
    domain=region.volume,
    formulation=MagneticVectorPotential(gauge=CoulombGauge()),
    permeability=regionwise(
        ((lambda points: np.linalg.norm(points, axis=1) < radius, mu0 * mu_r),),
        default=mu0,
        unit="H/m",
    ),
)
problem = Problem(
    name="Magnetic prospecting",
    geometry=region,
    physics=(model,),
    boundary_conditions=(
        MagneticPotentialBoundary(
            boundary=faces.outer,
            physics=model,
            value=lambda x, y, z: 0.5 * np.cross(B0, np.stack((x, y, z), axis=-1)),
        ),
        MagneticInsulation(boundary=faces.meridian, physics=model),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=SparseDirect()
)
fields = [
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(model.magnetic_flux_density)
    for mesh in meshes
]

# ## Results
#
# The anomaly along the north-south profile over the body on the surface, against the dipole field
# of the sphere.

K = (mu_r - 1) / (mu_r + 2)


def sphere_field(points):
    """Exact field outside a permeable sphere in the uniform field B0."""
    r = np.linalg.norm(points, axis=1, keepdims=True)
    return B0 + K * (radius / r) ** 3 * (
        3 * (points @ B0)[:, None] * points / r**2 - B0
    )


anomaly = lambda B, position: 1e9 * (B @ B0) / earth - 1e9 * earth
x = np.linspace(-400.0, 400.0, 321)
profile = np.column_stack((x, np.full_like(x, 1e-3), np.full_like(x, depth)))
exact = anomaly(sphere_field(profile), None)
computed = [anomaly(field.evaluate(profile), None) for field in fields]
peaks = [values.max() for values in computed]
print(
    f"exact: largest anomaly {exact.max():.1f} nT at x = {x[np.argmax(exact)]:+.1f} m, smallest {exact.min():.1f} nT at x = {x[np.argmin(exact)]:+.1f} m"
)
print(
    "cell size in the body [m]   tetrahedra   largest anomaly [nT]   deviation   largest deviation on the profile"
)
for size, mesh, values in zip(sizes, meshes, computed):
    print(
        f"{size:25.1f} {mesh.cell_count:12d} {values.max():20.1f} {100 * (values.max() / exact.max() - 1):+10.2f} % "
        f"{np.max(np.abs(values - exact)):12.1f} nT ({100 * np.max(np.abs(values - exact)) / exact.max():.2f} % of the peak)"
    )
print(
    f"field inside the body: {1e6 * np.linalg.norm(fields[-1].evaluate(np.array([[5.0, 5.0, 0.0]]))):.3f} µT, "
    f"exact {1e6 * 3 * mu_r / (mu_r + 2) * earth:.3f} µT"
)

# ## Figure
#
# The anomaly map on the surface (the half mirrored about the meridian), and the north-south
# profile over the body.

fig = plots.figure(1, 2, size=(15, 6.5))
plots.section(
    fields[-1],
    plane=Plane(origin=(0.0, 0.0, depth), normal=(0.0, 0.0, 1.0)),
    component=anomaly,
    symmetry={"planes": [((0, 1, 0), (0, 0, 0))]},
    limits=(-800, 2000),
    contours=15,
    subdivisions=2,
    label="ΔT [nT]",
    title="Total-field anomaly on the surface",
    xlimits=(-400, 400),
    ylimits=(-400, 400),
    xlabel="north x [m]",
    ylabel="east y [m]",
    ax=fig[0],
)
plots.line(
    points={"simulation": (x[::8], computed[-1][::8])},
    reference={"permeable sphere, exact": (x, exact)},
    hlines=0.0,
    xlabel="north x [m]",
    ylabel="ΔT [nT]",
    title="North-south profile over the body",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Replace the sphere in `magnetic_prospecting_ore_body_geometry.py` by a dipping slab or an
# irregular body and write the STEP files again; the exact sphere then serves as a first estimate.
# Add a remanent magnetization (`magnetization_provider`) to see how it shifts and turns the anomaly,
# or move the body to the magnetic equator (inclination 0°), where the anomaly over it is negative.
