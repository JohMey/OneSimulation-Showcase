"""OneSimulation example: static stress analysis of a sheet-metal bracket.
Requires the installed, activated OneSimulation runtime.

A steel bracket is bolted to a ceiling and carries a pin load in the lugs at the end of its two
arms. The 3D stress field is computed with curved quadratic hexahedra and compared with beam
theory in the arms and with the handbook stress concentration factor of a pin-loaded lug.
"""


# # Static stress analysis of a bracket
#
# A bracket made of 5 mm steel sheet hangs from a ceiling. Its two vertical **arms** are joined at
# the back by a horizontal web; two **mounting tabs** bent outwards from the top of the arms carry
# two bolt holes each. At the front end each arm has a downward **lug** with a pin hole. A pin
# through both lugs carries a load of 2 kN straight down.
#
# Dimensions in mm, $x$ along the arms, $y$ across, $z$ up; the bracket is symmetric about $y=0$
# and one half is modelled.
#
# | part | geometry |
# |---|---|
# | arms | $0\le x\le130$, $0\le z\le40$, at $20\le|y|\le25$ |
# | lugs | $100\le x\le130$, $-45\le z\le0$, pin hole $\varnothing12$ at $x=115$, $z=-15$ |
# | web and tabs | $0\le x\le40$, $|y|\le45$, $35\le z\le40$, bolt holes $\varnothing9$ |
#
# | face | condition |
# |---|---|
# | bolt holes | fixed |
# | pin hole | bearing pressure $p_{max}\cos\varphi$ on the lower half, resultant 1 kN per lug |
# | symmetry plane $y=0$ | $u_y=0$ |
# | all other faces | free |
#
# **Reference.** In the arm, beam theory gives the bending stress $M(z-H/2)/I$ with
# $M=(F/2)(x_{pin}-x)$. At the pin hole, Peterson's chart for a pin-loaded lug gives the stress
# concentration factor $K_t=2.85$ on the net-section stress.

import numpy as np

from onesim.cad.d2 import Plane
from onesim.materials import IsotropicSolid
from onesim.meshing import ExplicitGeometry, Geometric, VolumeBlock, merge_volume_blocks, o_grid, rectangle_block
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, FixedDisplacementComponents, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()

sheet, height, half_gap, tab = 0.005, 0.040, 0.020, 0.040
lug_start, lug_end, lug_bottom = 0.100, 0.130, -0.045
pin, pin_radius, bolts, bolt_radius = (0.115, -0.015), 0.006, ((0.010, 0.035), (0.030, 0.035)), 0.0045
load = 2000.0
steel = IsotropicSolid(density=7850.0, specific_heat_capacity=470.0, thermal_conductivity=50.0,
                       youngs_modulus=200e9, poisson_ratio=0.30, thermal_expansion_coefficient=1.2e-5)


# ## Geometry and mesh
#
# The bracket is assembled from two sheets of blocks: the arm with its lug in the $x$-$z$ plane and
# the web with the tab in the $x$-$y$ plane, each with an O-grid around every hole. The sheets are
# extruded through the thickness and merged into one mesh of curved 27-node hexahedra.

def face_name(point, normal):
    """Name of a boundary face: hole surfaces lie on their circle with a normal in the sheet plane."""
    x, y, z = point
    n = np.asarray(normal) / np.linalg.norm(normal)
    if abs(y) < 1e-12:
        return "symmetry"
    if abs(n[1]) < 0.5 and abs(np.hypot(x - pin[0], z - pin[1]) - pin_radius) < 0.05 * pin_radius:
        return "pin"
    if abs(n[2]) < 0.5 and any(abs(np.hypot(x - cx, y - cy) - bolt_radius) < 0.05 * bolt_radius for cx, cy in bolts):
        return "bolts"
    return "free"


bracket = ExplicitGeometry(dimension=3, boundary_names=("bolts", "pin", "symmetry", "free"), classifier=face_name)
faces = bracket.boundaries


def bracket_mesh(g, e):
    """Mesh of the half bracket; g sets the cells of arm and tab, e those of the lug."""
    x = np.unique(np.r_[np.linspace(0, tab, 8 * g + 1), np.linspace(tab, lug_start, 8 * g + 1), np.linspace(lug_start, lug_end, 6 * e + 1)])
    z = np.unique(np.r_[np.linspace(0, height - sheet, 5 * g + 1), np.linspace(height - sheet, height, 3)])
    arm = [rectangle_block((0, 0), lug_end, height, (len(x) - 1, len(z) - 1), grading=(x, z)),
           rectangle_block((lug_start, lug_bottom), 0.030, pin[1] - 0.015 - lug_bottom, (6 * e, 2 * e)),
           *o_grid(pin, pin_radius, 0.015, 6 * e, 3 * e, radial_grading=Geometric(3 * e, ratio=4.0))]
    top = [rectangle_block((0, 0), tab, half_gap, (8 * g, 4 * g))]
    for centre in bolts:
        top += o_grid(centre, bolt_radius, 0.010, 4 * g, 2 * g + 1, radial_grading=Geometric(2 * g + 1, ratio=3.0))
    blocks = [VolumeBlock.from_block(block, np.linspace(half_gap, half_gap + sheet, 3), lambda u, v, w: (u, w, v)) for block in arm]
    blocks += [VolumeBlock.from_block(block, np.linspace(height - sheet, height, 3), lambda u, v, w: (u, v, w)) for block in top]
    return merge_volume_blocks(blocks, bracket, order=2)


# ## Model
#
# The pin presses on the lower half of the hole with the pressure $p_{max}\cos\varphi$,
# $p_{max}=F/(\pi r t)$, which adds up to half the load per lug.

def bearing(x, y, z):
    """Traction of the pin on the hole surface."""
    nx, nz = (pin[0] - np.asarray(x)) / pin_radius, (pin[1] - np.asarray(z)) / pin_radius
    pressure = load / (np.pi * pin_radius * sheet) * np.maximum(nz, 0.0)
    return np.stack((-pressure * nx, np.zeros_like(nx), -pressure * nz), axis=-1)


def solve(g, e):
    """Displacement, stress tensor and von Mises stress on the mesh with the densities g and e."""
    mesh = bracket_mesh(g, e)
    model = LinearElasticity(domain=bracket.domain(), material=steel)
    problem = Problem(name="Bracket", geometry=bracket, physics=(model,), boundary_conditions=(
        FixedDisplacement(boundary=faces.bolts, physics=model, value=(0.0, 0.0, 0.0)),
        FixedDisplacementComponents(boundary=faces.symmetry, physics=model, components=("y",)),
        Traction(boundary=faces.pin, physics=model, value=bearing),
        TractionFree(boundary=faces.free, physics=model)))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    return result.field(model.displacement), result.field(model.stress), result.field(model.von_mises_stress), mesh


runs = [solve(g, e) for g, e in ((1, 1), (2, 2), (2, 3))]
displacement, stress, mises, mesh = runs[-1]


# ## Results
#
# The bending stress across the arm at $x=70$ mm, and the hoop stress around the pin hole, both at
# mid-thickness of the sheet.

mid = half_gap + sheet / 2
z = np.linspace(1e-6, height - 1e-6, 21)
bending = stress.evaluate(np.column_stack((np.full_like(z, 0.070), np.full_like(z, mid), z)))[:, 0]
beam = load / 2 * (pin[0] - 0.070) * (z - height / 2) / (sheet * height**3 / 12)
angle = np.radians(np.arange(-180, 180, 3))
ring = stress.evaluate(np.column_stack((pin[0] + 1.0001 * pin_radius * np.cos(angle), np.full_like(angle, mid),
                                        pin[1] + 1.0001 * pin_radius * np.sin(angle))))
hoop = ring[:, 0] * np.sin(angle) ** 2 + ring[:, 2] * np.cos(angle) ** 2 - 2 * ring[:, 5] * np.sin(angle) * np.cos(angle)
net_stress = load / 2 / ((0.030 - 2 * pin_radius) * sheet)

print("hexahedra   pin deflection [mm]   largest von Mises stress at the pin hole [MPa]")
for u, _, s, level in runs:
    print(f"{level.cell_count:9d} {1e3 * u.evaluate([[pin[0], mid, pin[1] - pin_radius]])[0, 2]:17.4f}"
          f" {s.sample(along=faces.pin, subdivisions=4).values.max() / 1e6:22.2f}")
print(f"outer-fibre bending stress in the arm: simulation {np.polyfit(z - height / 2, bending, 1)[0] * height / 2 / 1e6:.2f} MPa, "
      f"beam theory {beam[-1] / 1e6:.2f} MPa")
print(f"stress concentration at the pin hole: simulation {hoop.max() / net_stress:.2f}, Peterson's chart 2.85")


# ## Figure
#
# The von Mises stress on the deformed bracket (displacement magnified 40 times), in the mid-plane
# of the lug, the bending stress across the arm and the hoop stress around the pin hole.

megapascal = lambda s, position: s / 1e6
fig = plots.figure(2, 2, size=(15, 11))
plots.surface(mises, component=megapascal, deformation=displacement, deformation_scale=40.0, limits=(0, 60),
              symmetry={"planes": [((0, 1, 0), (0, 0, 0))]}, camera=plots.Camera(elevation=26, azimuth=-128),
              label="von Mises stress [MPa]", title="Deformed bracket", ax=fig[0])
plots.section(mises, plane=Plane(origin=(0, mid, 0), normal=(0, 1, 0)), component=megapascal, limits=(0, 32), subdivisions=3,
              label="von Mises stress [MPa]", title="Lug with the pin hole", xlimits=(0.095, 0.135), ylimits=(-0.05, 0.005),
              ax=fig[1])
plots.line(points={"simulation": (bending / 1e6, 1e3 * z)}, reference={"beam theory": (beam / 1e6, 1e3 * z)},
           xlabel="bending stress [MPa]", ylabel="z [mm]", title="Arm section at x = 70 mm", ax=fig[2])
plots.line(np.degrees(angle), {"simulation": hoop / 1e6}, hlines={"Peterson's chart": 2.852 * net_stress / 1e6},
           ylimits=(-10, 36), xlabel="angle around the pin hole [°]", ylabel="hoop stress [MPa]",
           title="Hoop stress around the pin hole",
           legend="upper left", ax=fig[3])
plots.show()


# ## Going further
#
# Thicken the sheet or enlarge the lug and watch the stress concentration fall; replace the fixed
# bolt holes by elastic supports, or add a sideways load on the pin.
