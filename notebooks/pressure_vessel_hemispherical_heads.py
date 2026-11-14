"""OneSimulation example: stresses in a cylindrical pressure vessel with hemispherical heads.
Requires the installed, activated OneSimulation runtime.

A closed steel vessel under internal pressure is computed as a 3D elastic solid with curved
quadratic hexahedra. The stresses are compared with Lamé's thick-wall solution far from the
junction and with thin-shell theory at the junction between cylinder and head.
"""


# # Pressure vessel with hemispherical heads
#
# A steel cylinder (inner radius 0.5 m, wall 25 mm, 2 m long) is closed by two hemispherical heads
# of the same wall thickness and loaded by an internal pressure of 2 MPa. Under pressure the
# cylinder wants to grow more than the head, so the junction bends the wall locally.
#
# One octant is modelled, with symmetry conditions on its three cut planes.
#
# **Reference.** Far from the junction Lamé's solution holds: hoop stress
# $A+B/r^2$ and axial stress $A$ in the cylinder with $A=pa^2/(b^2-a^2)$, $B=Ab^2$, and the
# tangential stress $C\,(1+b^3/2r^3)$ in the sphere with $C=pa^3/(b^3-a^3)$. At the junction,
# thin-shell theory adds a bending moment that decays over the length $1/\beta$,
#
# $$M(s)=\pm\frac{p}{8\beta^2}\,e^{-\beta|s|}\sin\beta|s|,\qquad \beta=\frac{[3(1-\nu^2)]^{1/4}}{\sqrt{Rt}},$$
#
# which changes the meridional surface stresses by $\pm6M/t^2$ (Timoshenko and
# Woinowsky-Krieger, *Theory of Plates and Shells*).

import numpy as np

from onesim.materials import IsotropicSolid
from onesim.meshing import ExplicitGeometry, Geometric, VolumeBlock, merge_volume_blocks
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacementComponents, LinearElasticity, Pressure, TractionFree
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()

inner, wall, length, pressure = 0.5, 0.025, 1.0, 2.0e6
outer, middle = inner + wall, inner + wall / 2
steel = IsotropicSolid(density=7850.0, specific_heat_capacity=475.0, thermal_conductivity=44.5,
                       youngs_modulus=200e9, poisson_ratio=0.3, thermal_expansion_coefficient=12e-6)
membrane = pressure * middle / wall


def reference(s, r):
    """Meridional stress at the radius r, the distance s along the wall from the junction (head: s > 0)."""
    beta = (3 * (1 - steel.poisson_ratio**2)) ** 0.25 / np.sqrt(middle * wall)
    lame = np.where(s > 0, pressure * inner**3 / (outer**3 - inner**3) * (1 + outer**3 / (2 * r**3)),
                    pressure * inner**2 / (outer**2 - inner**2))
    moment = np.where(s > 0, -1, 1) * pressure / (8 * beta**2) * np.exp(-beta * np.abs(s)) * np.sin(beta * np.abs(s))
    return lame + np.sign(r - middle) * 6 * moment / wall**2


# ## Geometry and mesh
#
# Four blocks cover the octant of the wall: the cylinder and three patches of the head. Their
# maps place every node, including the mid-side nodes, on the exact cylinder and sphere. The cells
# are graded towards the junction.

def face_name(point, normal):
    """Symmetry plane, or inner and outer wall surface, from the position of a face."""
    cut = [axis for axis in range(3) if abs(point[axis]) < 1e-12]
    if cut:
        return "symmetry_" + "xyz"[cut[0]]
    radius = np.hypot(point[0], point[1]) if point[2] <= length else np.linalg.norm(np.subtract(point, (0, 0, length)))
    return "inside" if radius < middle else "outside"


vessel = ExplicitGeometry(dimension=3, boundary_names=("inside", "outside", "symmetry_x", "symmetry_y", "symmetry_z"),
                          classifier=face_name)
faces = vessel.boundaries


def head_patch(axes):
    """Map of one head patch: the direction (1, tan u, tan v), permuted by `axes`, scaled to the radius r."""
    def patch(u, v, r):
        direction = np.stack((np.ones_like(u), np.tan(u), np.tan(v)))[list(axes)]
        x, y, z = r * direction / np.linalg.norm(direction, axis=0)
        return x, y, z + length
    return patch


def vessel_mesh(layers):
    """Curved hexahedral mesh with the given number of cells through the wall."""
    radii, quarter = np.linspace(inner, outer, layers + 1), np.linspace(0, np.pi / 4, 5)
    towards_pole = np.pi / 4 * Geometric(10, first=wall / 2 / middle).reference(np.pi / 4)
    blocks = [VolumeBlock(lambda angle, z, r: (r * np.cos(angle), r * np.sin(angle), z), np.linspace(0, np.pi / 2, 9),
                          length * Geometric(16, first=wall / 2, toward="end").reference(length), radii),
              VolumeBlock(head_patch((0, 1, 2)), quarter, towards_pole, radii),
              VolumeBlock(head_patch((1, 0, 2)), quarter, towards_pole, radii),
              VolumeBlock(head_patch((1, 2, 0)), quarter, quarter, radii)]
    return merge_volume_blocks(blocks, vessel, order=2)


# ## Model

def solve(layers):
    """Stress tensor and von Mises stress on the mesh with `layers` cells through the wall."""
    model = LinearElasticity(domain=vessel.domain(), material=steel)
    problem = Problem(name="Pressure vessel", geometry=vessel, physics=(model,), boundary_conditions=(
        Pressure(boundary=faces.inside, physics=model, value=pressure),
        TractionFree(boundary=faces.outside, physics=model),
        FixedDisplacementComponents(boundary=faces.symmetry_x, physics=model, components=("x",)),
        FixedDisplacementComponents(boundary=faces.symmetry_y, physics=model, components=("y",)),
        FixedDisplacementComponents(boundary=faces.symmetry_z, physics=model, components=("z",))))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    result = Simulation(problem=problem, mesh=vessel_mesh(layers), numerics=numerics).solve_stationary()
    return result.field(model.stress), result.field(model.von_mises_stress)


runs = {layers: solve(layers) for layers in (1, 2, 4)}
stress, mises = runs[4]


# ## Results
#
# The stresses are read along the wall just inside the symmetry plane $y=0$. There the hoop
# stress is $\sigma_{yy}$, and the meridional stress follows from rotating the tensor by the
# latitude $\psi=s/R$ of the head.

s = np.linspace(-0.45, 0.35, 321)
latitude = np.where(s > 0, s / middle, 0.0)


def along_wall(field, r):
    """Meridional and hoop stress of a stress field along the wall at the radius r."""
    points = np.column_stack((r * np.cos(latitude), np.full_like(s, 1e-6), length + np.where(s > 0, r * np.sin(latitude), s)))
    xx, yy, zz, _, _, xz = field.evaluate(points).T
    return xx * np.sin(latitude) ** 2 + zz * np.cos(latitude) ** 2 - 2 * xz * np.sin(latitude) * np.cos(latitude), yy


surfaces = {"outer surface": outer - 2e-5, "inner surface": inner + 2e-5}
meridional = {label: along_wall(stress, r)[0] for label, r in surfaces.items()}
hoop_far = along_wall(stress, inner + 2e-5)[1][0]
lame_hoop = pressure * inner**2 / (outer**2 - inner**2) * (1 + outer**2 / (inner + 2e-5) ** 2)
peaks = {layers: along_wall(field, outer - 2e-5)[0][s < 0].max() for layers, (field, _) in runs.items()}

print(f"hoop stress at the inner surface far from the junction: simulation {hoop_far / 1e6:.3f} MPa, Lamé {lame_hoop / 1e6:.3f} MPa")
print(f"largest meridional stress on the outer surface of the cylinder, in units of pR/t: "
      f"thin-shell theory {reference(s[s < 0], outer).max() / membrane:.4f}")
for layers, peak in peaks.items():
    print(f"   {layers} cell(s) through the wall: {peak / membrane:.4f}")


# ## Figure
#
# The von Mises stress on the vessel with one half cut away, and the meridional stress on both
# wall surfaces along the junction: the bending flips sides between cylinder and head.

scaled = s / np.sqrt(middle * wall)
fig = plots.figure(1, 2, size=(15, 6))
plots.surface(mises, boundary=faces.outside | faces.inside, component=lambda value, position: value / 1e6, subdivisions=3,
              symmetry={"planes": [((0, 0, 1), (0, 0, 0)), ((1, 0, 0), (0, 0, 0))]}, limits=(12, 42),
              camera=plots.Camera(elevation=22, azimuth=-58, up="y"), label="von Mises stress [MPa]",
              title="von Mises stress, half cut away", ax=fig[0])
plots.line(scaled, {label: values / membrane for label, values in meridional.items()},
           reference={"outer surface": reference(s, outer) / membrane, "inner surface": reference(s, inner) / membrane},
           vlines={"junction": 0.0}, xlabel="distance from the junction / √(R t)   (cylinder < 0 < head)",
           ylabel="meridional stress / (pR/t)", title="Simulation (solid) and Lamé with thin-shell bending (dashed)",
           legend="lower left", ax=fig[1])
plots.show()


# ## Going further
#
# Give the head half the wall thickness of the cylinder, as the membrane stresses suggest, and
# the junction bending grows; replace the hemisphere by a torispherical head to see the high
# stresses in its knuckle.
