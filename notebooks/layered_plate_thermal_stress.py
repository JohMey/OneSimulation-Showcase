"""OneSimulation example: thermal stresses in a coated plate (plane strain).
Requires the installed, activated OneSimulation runtime.

The 2D plane-strain stress field of a two-layer plate cooled after coating is
computed in OneSimulation and compared with the closed-form bilayer (Timoshenko)
solution at mid-span (for the actual plate and for a three times longer one),
with the section equilibrium and a mesh study at the free edge.
"""


# # Thermal stresses in a coated plate
#
# A substrate plate ($10$ mm thick) carries a coating ($2$ mm thick); the plate
# is $20$ mm long. The coating is deposited at $800\,^\circ$C, where both
# layers are stress free, and the plate is then cooled to $150\,^\circ$C. The
# substrate contracts more than the coating
# ($\alpha_s=3\cdot10^{-6}$ 1/K against $\alpha_c=5\cdot10^{-7}$ 1/K), so the
# coating ends up in compression, the substrate next to it in tension, and the
# plate bends. The plate is thick in the third direction and is in **plane
# strain**: the field depends on $(x,y)$ only and $u_z=0$. Near the two free
# ends the stresses must turn around to leave the end faces traction free,
# which creates the classical free-edge concentration of peel and shear stress
# at the interface.
#
# | layer | $E$ | $\nu$ | $\alpha$ |
# |---|---|---|---|
# | substrate | 130 GPa | 0.28 | $3\cdot10^{-6}$ 1/K |
# | coating | 70 GPa | 0.17 | $5\cdot10^{-7}$ 1/K |
#
# **Reference.** Away from the free ends the plate behaves like a bilayer
# beam (Timoshenko, *Analysis of bi-metal thermostats*, J. Opt. Soc. Am. 11,
# 233, 1925). With the plane-strain modulus $E_i'=E_i/(1-\nu_i^2)$ and the
# plane-strain thermal strain $e_i=(1+\nu_i)\alpha_i\Delta T$, the axial strain
# is linear across the section, $\varepsilon_{xx}=\varepsilon_0+\kappa y$, and
#
# $$\sigma_{xx}(y)=E_i'\,\bigl(\varepsilon_0+\kappa y-e_i\bigr),\qquad
#   \sum_i\int_i\sigma_{xx}\,dy=0,\qquad \sum_i\int_i\sigma_{xx}\,y\,dy=0,$$
#
# which fixes $\varepsilon_0$ and $\kappa$. This is the exact interior
# (Saint-Venant) solution of the 2D problem for a long plate; the mid-span of
# this short plate ($20$ mm long, $12$ mm thick) lies only $0.83$ thicknesses
# from each end, so the end disturbance has not decayed completely there.
# The comparison therefore combines (1) the mid-span stresses against this
# closed form, for the actual plate and for the same model with a plate three
# times as long, where the end disturbance has decayed at mid-span and the
# field must reduce to the closed form, (2) exact equilibrium identities of the
# computed field (zero net force and moment over the mid-span section;
# interface shear force equal to the coating force) and (3) a three-mesh study
# (observed order, Richardson estimate) of the stresses near the free edge and
# at mid-span.
#
# **How plane strain is obtained.** OneSimulation's thermoelastic solid
# (`LinearElasticity` with a thermal eigenstrain) is formulated in 3D. The
# section is extruded into a thin slab, one quadratic hexahedron thick. Plane
# strain means $\varepsilon_{zz}=0$ and therefore
# $\sigma_{zz}=\nu(\sigma_{xx}+\sigma_{yy})-E\alpha\Delta T$; the two $z$ faces
# receive exactly this normal traction, computed from the in-plane stresses of
# the previous solve. At the fixed point of this update the slab carries a
# $z$-independent field with $\varepsilon_{zz}=0$, i.e. the exact plane-strain
# solution; the printed $\max|\varepsilon_{zz}|$ shows how closely it is reached.
# The plate is free in its plane, as in the original set-up, so the rigid-body
# motions are removed by OneSimulation's exact rigid-motion gauge (no supports).
#
# **Method.** Continuous Galerkin finite elements: tensor-product quadratic
# ($Q_2$, 27-node) Lagrange displacement on hexahedra, $3\times3\times3$ Gauss
# points per cell, with a cell-wise layered material law
# (`PropertyLaw` in $y$) and the isotropic thermal eigenstrain
# $\alpha(T-T_\text{ref})\mathbf I$. The temperature comes from a trilinear
# ($Q_1$) heat-conduction solve. The problem is stationary (no time
# integrator); each solve uses a sparse direct (LU) solver with the six
# rigid-body modes eliminated exactly, and four plane-strain traction updates
# (a fixed-point iteration) are made per mesh.

from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np

from onesim.cad.d3 import TrilinearHexahedron
from onesim.materials import IsotropicSolid
from onesim.meshing import TrilinearHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.field_transfer import FiniteElementEvaluation
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction
from onesim.physics.solid_mechanics import IsotropicThermalExpansion, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling
from onesim.simulation.field_constraints import RigidMotionGauge

print("OneSimulation:", version("onesimulation"))


# ## Parameters

length = 0.020                              # m, plate length (x)
h_sub, h_coat = 0.010, 0.002                # m, substrate and coating thickness (y)
height = h_sub + h_coat
slab = 0.0005                               # m, extrusion depth (z); any value gives the same plane-strain field
E_sub, nu_sub, alpha_sub = 130e9, 0.28, 3.0e-6
E_coat, nu_coat, alpha_coat = 70e9, 0.17, 5.0e-7
T_stress_free = 1073.15                     # K (800 degC)
T_final = 423.15                            # K (150 degC)
delta_T = T_final - T_stress_free


def layer_constants(y):
    """E, nu, alpha of the layer at height y (coating above the interface)."""
    coat = np.asarray(y) > h_sub
    return (np.where(coat, E_coat, E_sub), np.where(coat, nu_coat, nu_sub),
            np.where(coat, alpha_coat, alpha_sub))


class LayerProperty:
    """Material property that takes one value in the substrate and another in the coating."""

    def __init__(self, substrate, coating, unit):
        self.substrate, self.coating, self.unit = substrate, coating, unit

    def evaluate(self, y):
        return np.where(np.asarray(y) > h_sub, self.coating, self.substrate)


material = IsotropicSolid(
    density=3000.0, specific_heat_capacity=700.0, thermal_conductivity=30.0,
    youngs_modulus=LayerProperty(E_sub, E_coat, "Pa"),
    poisson_ratio=LayerProperty(nu_sub, nu_coat, "1"),
    thermal_expansion_coefficient=LayerProperty(alpha_sub, alpha_coat, "1/K"),
)

# Closed-form bilayer solution (plane strain): solve for eps0 and kappa.
layers = ((0.0, h_sub, E_sub, nu_sub, alpha_sub), (h_sub, height, E_coat, nu_coat, alpha_coat))
matrix, rhs = np.zeros((2, 2)), np.zeros(2)
for y0, y1, E, nu, alpha in layers:
    Ep, e = E / (1 - nu**2), (1 + nu) * alpha * delta_T
    m0, m1, m2 = y1 - y0, (y1**2 - y0**2) / 2, (y1**3 - y0**3) / 3
    matrix += Ep * np.array([[m0, m1], [m1, m2]])
    rhs += Ep * e * np.array([m0, m1])
eps0, kappa = np.linalg.solve(matrix, rhs)


def sigma_xx_bilayer(y, coat):
    E, nu, alpha = (E_coat, nu_coat, alpha_coat) if coat else (E_sub, nu_sub, alpha_sub)
    return E / (1 - nu**2) * (eps0 + kappa * np.asarray(y) - (1 + nu) * alpha * delta_T)


coat_force_bilayer = (h_coat / 2) * (sigma_xx_bilayer(h_sub, True) + sigma_xx_bilayer(height, True))
print(f"Bilayer solution: eps0 = {eps0:.4e}, curvature = {kappa:.4f} 1/m, "
      f"coating force = {coat_force_bilayer / 1e3:.2f} kN/m")


# ## Geometry and mesh
#
# The section is the rectangle $0\le x\le L$, $0\le y\le 12$ mm with the
# interface at $y=10$ mm, extruded by $0.5$ mm in $z$. `TrilinearHexahedralMesher`
# builds a tensor grid with prescribed (graded) node positions. The cells
# shrink geometrically toward both free ends (factor 50) and toward the
# interface (factor 40 in the substrate, 8 in the coating), so the mesh is
# finest at the two interface corners where the free-edge stresses
# concentrate; every layer has its own cell rows (each cell lies in one
# material), and the coating has several cells across its thickness.
# Displacements are quadratic (27-node Lagrange hexahedra).
#
# At the interface corner itself the elastic field of a bimaterial wedge is
# weakly singular (Bogy, J. Appl. Mech. 35, 460, 1968), so the corner value is
# not a meaningful number. The mesh study therefore follows the peel and shear
# stress at fixed distances from the free end, where the exact field is finite.


def geometric(n, ratio):
    """n cells on [0, 1]; the last cell is `ratio` times the first."""
    sizes = ratio ** (np.arange(n) / max(n - 1, 1))
    nodes = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    nodes[-1] = 1.0
    return nodes


def mesh_axes(nx, n_sub, n_coat):
    half = 0.5 * geometric(nx // 2, 50.0)                         # geometric toward each free end
    x = np.r_[half, 1.0 - half[::-1][1:]]
    y_sub = h_sub * (1 - geometric(n_sub, 40.0)[::-1])            # geometric toward the interface
    y_coat = h_sub + h_coat * geometric(n_coat, 8.0)              # geometric away from the interface
    y = np.r_[y_sub, y_coat[1:]] / height
    x[-1] = y[-1] = 1.0
    return x, y


# ## Model
#
# Boundary conditions on every face of the slab:
#
# | face | condition |
# |---|---|
# | end faces $x=0$, $x=L$ | traction-free |
# | lower and upper faces $y=0$, $y=H$ | traction-free |
# | slab faces $z=0$, $z=\text{depth}$ | normal traction $\mp\sigma_{zz}^{\text{plane strain}}$, no shear |
#
# plus the rigid-motion gauge (free plate). The temperature field enters through
# a field coupling from a heat-conduction solve with $T=150\,^\circ$C on every
# face, i.e. the uniform final temperature.


def build_plate(plate_length):
    """Geometry, uniform final temperature and thermoelastic model of a plate of the given length."""
    geometry = TrilinearHexahedron(vertices=(
        (0, 0, 0), (plate_length, 0, 0), (plate_length, height, 0), (0, height, 0),
        (0, 0, slab), (plate_length, 0, slab), (plate_length, height, slab), (0, height, slab),
    ))
    faces = geometry.boundaries
    heat = HeatConduction(domain=geometry.volume, material=material)
    temperature = Simulation(
        problem=Problem(
            name="Uniform final temperature", geometry=geometry, physics=(heat,),
            boundary_conditions=(FixedTemperature(boundary=faces.all, physics=heat, value=T_final),),
        ),
        mesh=TrilinearHexahedralMesher(reference_axes=((0.0, 1.0),) * 3).generate(geometry),
        numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=1)), linear_solver=SparseDirect()),
    ).solve_stationary().field(heat.temperature)
    solid = LinearElasticity(
        domain=geometry.volume, material=material,
        thermal_strain=IsotropicThermalExpansion(reference_temperature=T_stress_free),
    )
    return dict(length=plate_length, geometry=geometry, faces=faces, temperature=temperature, solid=solid)


def in_plane_stress(gradient, y):
    """Stress components from a displacement gradient at heights y (Hooke law with thermal strain)."""
    E, nu, alpha = layer_constants(y)
    lam, mu, th = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu)), alpha * delta_T
    exx, eyy, ezz = gradient[:, 0, 0], gradient[:, 1, 1], gradient[:, 2, 2]
    exy = 0.5 * (gradient[:, 0, 1] + gradient[:, 1, 0])
    trace = exx + eyy + ezz - 3 * th
    return dict(xx=lam * trace + 2 * mu * (exx - th), yy=lam * trace + 2 * mu * (eyy - th),
                zz=lam * trace + 2 * mu * (ezz - th), xy=2 * mu * exy, ezz=ezz)


def solve_plane_strain(plate, cells, iterations=4):
    """Solve the slab, updating the plane-strain traction on the z faces after every solve."""
    geometry, faces, solid, L = plate["geometry"], plate["faces"], plate["solid"], plate["length"]
    x_axis, y_axis = mesh_axes(*cells)
    mesh = TrilinearHexahedralMesher(reference_axes=(tuple(x_axis), tuple(y_axis), (0.0, 1.0))).generate(geometry)
    # The z faces of this tensor grid are integrated with 3 x 3 Gauss points per
    # cell; the traction is tabulated exactly at these points.
    g = (np.polynomial.legendre.leggauss(3)[0] + 1) / 2
    xg = np.sort(((x_axis[:-1, None] + np.diff(x_axis)[:, None] * g) * L).ravel())
    yg = np.sort(((y_axis[:-1, None] + np.diff(y_axis)[:, None] * g) * height).ravel())
    XG, YG = np.meshgrid(xg, yg, indexing="ij")
    E, nu, alpha = layer_constants(YG)
    table = {"szz": -E * alpha * delta_T}            # start: plane-strain value of a free layer

    def szz(x, y):
        i = np.abs(xg[:, None] - np.ravel(x)[None, :]).argmin(axis=0)
        j = np.abs(yg[:, None] - np.ravel(y)[None, :]).argmin(axis=0)
        return table["szz"][i, j].reshape(np.shape(x))

    def z_traction(sign):
        return lambda x, y, z, t: np.stack((0 * x, 0 * x, sign * szz(x, y)), axis=-1)

    problem = Problem(
        name="Coated plate cooled from the deposition temperature", geometry=geometry, physics=(solid,),
        boundary_conditions=(
            TractionFree(boundary=faces.x_min | faces.x_max | faces.y_min | faces.y_max, physics=solid),
            Traction(boundary=faces.z_min, physics=solid, value=z_traction(-1.0)),
            Traction(boundary=faces.z_max, physics=solid, value=z_traction(+1.0)),
        ),
        constraints=(RigidMotionGauge(solid.displacement),),
        couplings=(FieldCoupling(source=plate["temperature"], target=solid.temperature,
                                 transfer=FiniteElementEvaluation()),),
    )
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()))
    points = np.column_stack((XG.ravel(), YG.ravel(), np.full(XG.size, slab / 2)))
    thermal = abs(alpha_sub * delta_T)
    print(f"Plate {1e3 * L:.0f} mm, mesh {cells[0]} x {cells[1] + cells[2]} cells "
          f"(coating {cells[2]} cells, smallest cell {1e6 * np.diff(x_axis).min() * L:.0f} um)")
    for iteration in range(iterations):
        result = simulation.solve_stationary()
        operator = result.operators[solid.model_id]
        values = np.asarray(result.field(solid.displacement).values)
        s = in_plane_stress(operator.space.gradient(values, points), YG.ravel())
        new = (nu.ravel() * (s["xx"] + s["yy"]) - E.ravel() * alpha.ravel() * delta_T).reshape(XG.shape)
        print(f"  update {iteration + 1}: max|eps_zz| / |alpha_s dT| = {np.abs(s['ezz']).max() / thermal:.1e}, "
              f"max change of sigma_zz = {np.abs(new - table['szz']).max() / 1e6:.3f} MPa")
        table["szz"] = new
    return result


# ## Solve
#
# The 20 mm plate on three meshes, each 1.5 times finer than the previous one
# in both directions. In addition, the same model is solved for a plate three
# times as long (60 mm), whose mid-span lies far enough from the free ends for
# the end disturbance to have decayed: there the field must reduce to the
# bilayer solution itself.

plate = build_plate(length)
mesh_family = ((16, 8, 4), (24, 12, 6), (36, 18, 9))
runs = {cells: solve_plane_strain(plate, cells) for cells in mesh_family}
long_plate = build_plate(3 * length)
long_result = solve_plane_strain(long_plate, (36, 12, 6))


# ## Comparison with the reference
#
# Printed below: the mid-span stresses of the 20 mm plate and of the 60 mm
# plate next to the bilayer solution with their relative deviations, the
# equilibrium identities of the computed field, and the three-mesh study with
# the change between the two finest meshes, the observed order and the
# Richardson estimate. In the short plate the deviation from the bilayer
# solution is the physical end effect; in the long plate it must vanish. The
# free-edge values and the coating stress converge fastest; the mid-span value
# at the lower face converges most slowly, and its Richardson estimate shows
# where it is heading.

def evaluate_stress(plate, result, points):
    return np.asarray(result.field(plate["solid"].stress).evaluate(np.asarray(points)))


def gauss_line(a, b, n_cells, order=6):
    """Gauss points and weights on [a, b] split into n_cells equal pieces."""
    g, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, n_cells + 1)
    pts = (edges[:-1, None] + (g + 1) / 2 * np.diff(edges)[:, None]).ravel()
    return pts, np.tile(w / 2, n_cells) * np.repeat(np.diff(edges), order)


tiny = 1e-9
mid_heights = [0.0, h_sub - tiny, h_sub + tiny, height]
edge_distances = (0.5e-3, 1.0e-3)


def checks(plate, result):
    L = plate["length"]
    mid = evaluate_stress(plate, result, [[L / 2, y, slab / 2] for y in mid_heights])[:, 0]
    ys, ws = gauss_line(0.0, h_sub, 20)
    yc, wc = gauss_line(h_sub, height, 6)
    s_sub = evaluate_stress(plate, result, [[L / 2, y, slab / 2] for y in ys])[:, 0]
    s_coat = evaluate_stress(plate, result, [[L / 2, y, slab / 2] for y in yc])[:, 0]
    force = ws @ s_sub + wc @ s_coat
    moment = ws @ (s_sub * (ys - height / 2)) + wc @ (s_coat * (yc - height / 2))
    coat_force = wc @ s_coat
    xs, wx = gauss_line(0.0, L / 2, 40)
    tau = 0.5 * (evaluate_stress(plate, result, [[x, h_sub - tiny, slab / 2] for x in xs])[:, 3]
                 + evaluate_stress(plate, result, [[x, h_sub + tiny, slab / 2] for x in xs])[:, 3])
    edge = evaluate_stress(plate, result, [[d, h_sub - tiny, slab / 2] for d in edge_distances])
    return dict(mid=mid, force=force, moment=moment, coat_force=coat_force,
                shear_force=abs(wx @ tau), peel=edge[:, 1], shear=edge[:, 3])


results = {cells: checks(plate, runs[cells]) for cells in mesh_family}
long_checks = checks(long_plate, long_result)
fine = results[mesh_family[-1]]
labels = ("lower face", "substrate at interface", "coating at interface", "upper face")
reference_mid = np.array([sigma_xx_bilayer(0.0, False), sigma_xx_bilayer(h_sub, False),
                          sigma_xx_bilayer(h_sub, True), sigma_xx_bilayer(height, True)])

print("\nMid-span sigma_xx (MPa) against the bilayer (Timoshenko) solution")
print(f"  {'':24s} {'bilayer':>9s} {'20 mm plate':>12s} {'deviation':>10s} {'60 mm plate':>12s} {'deviation':>10s}")
for k, label in enumerate(labels):
    ref, short, long_ = reference_mid[k], fine["mid"][k], long_checks["mid"][k]
    print(f"  {label:24s} {ref / 1e6:9.2f} {short / 1e6:12.2f} {100 * (short / ref - 1):+9.2f}% "
          f"{long_ / 1e6:12.2f} {100 * (long_ / ref - 1):+9.2f}%")
print(f"  {'coating force (kN/m)':24s} {coat_force_bilayer / 1e3:9.2f} {fine['coat_force'] / 1e3:12.2f} "
      f"{100 * (fine['coat_force'] / coat_force_bilayer - 1):+9.2f}% {long_checks['coat_force'] / 1e3:12.2f} "
      f"{100 * (long_checks['coat_force'] / coat_force_bilayer - 1):+9.2f}%")

print("\nEquilibrium of the computed field (20 mm plate, finest mesh)")
print(f"  net axial force over the mid-span section / coating force   {fine['force'] / fine['coat_force']:+.2e}")
print(f"  net bending moment / (coating force x plate thickness)       "
      f"{fine['moment'] / (fine['coat_force'] * height):+.2e}")
print(f"  interface shear force from the free end to mid-span          {fine['shear_force'] / 1e3:.3f} kN/m")
print(f"  coating force at mid-span                                    {abs(fine['coat_force']) / 1e3:.3f} kN/m"
      f"   difference {100 * (fine['shear_force'] / abs(fine['coat_force']) - 1):+.2f} %")

print("\nMesh study, 20 mm plate (MPa); observed order and Richardson estimate from the three meshes")
quantities = {
    "peel at 0.5 mm from the end": [results[c]["peel"][0] for c in mesh_family],
    "shear at 0.5 mm from the end": [results[c]["shear"][0] for c in mesh_family],
    "peel at 1.0 mm from the end": [results[c]["peel"][1] for c in mesh_family],
    "shear at 1.0 mm from the end": [results[c]["shear"][1] for c in mesh_family],
    "mid-span, coating at interface": [results[c]["mid"][2] for c in mesh_family],
    "mid-span, lower face": [results[c]["mid"][0] for c in mesh_family],
}
print(f"  {'quantity':32s} {'coarse':>9s} {'medium':>9s} {'fine':>9s} {'last change':>12s} {'order':>6s} {'Richardson':>11s}")
for name, (f1, f2, f3) in quantities.items():
    order = np.log(abs(f1 - f2) / abs(f2 - f3)) / np.log(1.5)
    richardson = f3 + (f3 - f2) / (1.5 ** order - 1)
    print(f"  {name:32s} {f1 / 1e6:9.2f} {f2 / 1e6:9.2f} {f3 / 1e6:9.2f} {100 * abs(f3 / f2 - 1):11.2f}% "
          f"{order:6.2f} {richardson / 1e6:11.2f}")


# ## Figure
#
# Left: axial stress $\sigma_{xx}$ over the whole section. Middle: peel stress
# $\sigma_{yy}$ near the free end, concentrated at the interface corner.
# Right: mid-span profile of $\sigma_{xx}$ against the bilayer solution, for
# the 20 mm plate and for the 60 mm plate.

result = runs[mesh_family[-1]]
operator = result.operators[plate["solid"].model_id]
space = operator.space
values = np.asarray(result.field(plate["solid"].displacement).values)
m = 6
grid_1d = np.linspace(0.0, 1.0, m)
local = np.array([(a, b, 0.5) for a in grid_1d for b in grid_1d])
sub = np.array([(i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1)
                for i in range(m - 1) for j in range(m - 1)])
local_triangles = np.r_[sub[:, [0, 1, 2]], sub[:, [0, 2, 3]]]
points, sxx, syy, triangles = [], [], [], []
for cell in range(len(space.cell_dofs)):
    pts = space.geometry.map(cell, local)
    gradient = np.einsum("qnd,nc->qcd", space.gradients(cell, local), values[space.cell_dofs[cell]])
    s = in_plane_stress(gradient, np.full(len(pts), pts[:, 1].mean()))   # each cell lies in one layer
    points.append(pts); sxx.append(s["xx"]); syy.append(s["yy"])
    triangles.append(local_triangles + cell * m * m)
points, sxx, syy, triangles = (np.concatenate(v) for v in (points, sxx, syy, triangles))

cream = "#fbf5eb"
fig, panels = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True,
                           gridspec_kw={"width_ratios": (1.55, 0.85, 0.9)})
fig.patch.set_facecolor(cream)

ax = panels[0]
levels = np.linspace(-120, 120, 49)
im = ax.tricontourf(points[:, 0] * 1e3, points[:, 1] * 1e3, triangles, sxx / 1e6,
                    levels=levels, cmap="RdBu_r", extend="both")
ax.axhline(h_sub * 1e3, color="k", lw=0.6)
fig.colorbar(im, ax=ax, label="sigma_xx / MPa", ticks=np.arange(-120, 121, 40))
ax.set(xlabel="x / mm", ylabel="y / mm", title="Axial stress sigma_xx in the section", aspect="equal")

ax = panels[1]
levels = np.linspace(-100, 100, 41)
im = ax.tricontourf(points[:, 0] * 1e3, points[:, 1] * 1e3, triangles, syy / 1e6,
                    levels=levels, cmap="PuOr_r", extend="both")
ax.axhline(h_sub * 1e3, color="k", lw=0.6)
fig.colorbar(im, ax=ax, label="sigma_yy / MPa", ticks=np.arange(-100, 101, 25))
ax.set(xlabel="x / mm", ylabel="y / mm", xlim=(0, 3), ylim=(7, 12),
       title="Peel stress near the free end", aspect="equal")

ax = panels[2]
ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
for coat, (y0, y1) in ((False, (0.0, h_sub)), (True, (h_sub, height))):
    yy = np.linspace(y0, y1, 50)
    ax.plot(sigma_xx_bilayer(yy, coat) / 1e6, yy * 1e3, color="#A85539", lw=1.8,
            label="bilayer (Timoshenko)" if not coat else None)
    yq = np.linspace(y0 + tiny, y1 - tiny, 21)
    short = evaluate_stress(plate, result, [[length / 2, y, slab / 2] for y in yq])[:, 0]
    long_ = evaluate_stress(long_plate, long_result, [[1.5 * length, y, slab / 2] for y in yq])[:, 0]
    ax.plot(short / 1e6, yq * 1e3, "o", color="#39715e", ms=3.5,
            label="OneSimulation, 20 mm plate" if not coat else None)
    ax.plot(long_ / 1e6, yq * 1e3, "s", mfc="none", color="#214D43", ms=4.5,
            label="OneSimulation, 60 mm plate" if not coat else None)
ax.axhline(h_sub * 1e3, color="k", lw=0.6)
ax.set(xlabel="sigma_xx / MPa", ylabel="y / mm", title="Mid-span stress profile")
ax.legend(frameon=False, fontsize=9)
plt.show()


# ## Going further
#
# Vary the plate length between the two cases above to see how quickly the
# end disturbance decays toward the mid-span, or vary the coating thickness to
# see how the peel stress at the free end grows. The original process adds a
# third layer, a carrier bonded at $150\,^\circ$C and cooled with the plate to
# room temperature; it can be added as a second load step whose thermal strain
# in the carrier starts from the bonding temperature.
