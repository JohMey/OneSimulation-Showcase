"""OneSimulation example: heat transfer through a copper pipe with annular disk fins.
Requires the installed, activated OneSimulation runtime.

A periodic element of a finned pipe (grooved bore, one disk fin) is solved as a genuine
3D conduction problem with convective water and air sides. The fin efficiency is compared
with the classical Bessel-function solution for annular fins (Gardner 1945), the total
heat rate with an energy balance and a thermal-resistance network, and both are followed
through a three-level mesh study.
"""


# # Heat transfer through a finned pipe
#
# Warm water flows through a copper pipe; air blows across the outside. To raise the
# air-side area the pipe carries thin annular **disk fins**, and the bore carries
# twelve axial **grooves**. Far from the pipe ends the arrangement repeats with the fin
# pitch $p$, so one periodic element (one fin, centred) represents the whole pipe. The
# twelve grooves add a rotational period of $30^\circ$ with mirror symmetry, so the
# element is a $15^\circ$ wedge bounded by two symmetry planes. The temperature
# depends on all three coordinates: along the axis (fin versus gap), along the radius
# (wall, fin) and around the circumference (groove versus land).
#
# OneSimulation solves the stationary heat equation in the copper,
#
# $$\nabla\cdot(k\nabla T)=0,$$
#
# with quadratic (P2) Lagrange finite elements on a graded tetrahedral mesh, and the
# following condition on **every** boundary face:
#
# | face | condition |
# |---|---|
# | grooved bore | forced convection to the water, $-k\,\partial_n T=h_w\,(T-T_w)$ |
# | fin faces, fin tip, bare pipe surface | forced convection to the air, $-k\,\partial_n T=h_a\,(T-T_a)$ |
# | the two axial end planes (mid-gap) | zero flux (periodicity and mirror symmetry of the element) |
# | the two wedge planes $\theta=0$, $\theta=15^\circ$ | zero flux (mirror symmetry of the grooves) |
#
# **Reference.** Three independent checks:
#
# 1. *Fin efficiency.* For a thin annular fin of thickness $t$ between radii $r_1$ and
#    $r_2$ with uniform coefficient $h$, the one-dimensional fin equation has the
#    Bessel-function solution of K. A. Gardner, "Efficiency of extended surface",
#    Trans. ASME 67 (1945) 621-631. With $m=\sqrt{2h/(kt)}$ and a convective tip,
#    $\theta(r)=A\,I_0(mr)+B\,K_0(mr)$ with $\theta(r_1)=\theta_b$ and
#    $k\theta'(r_2)+h\theta(r_2)=0$. The Biot number $ht/(2k)$ is about $10^{-4}$, so
#    the 1D fin theory is essentially exact for the fin itself; the 3D model adds the
#    real fin root on the pipe wall. We also print Gardner's adiabatic-tip formula with
#    the corrected radius $r_2+t/2$.
# 2. *Energy balance.* The heat taken from the water must equal the heat given to the air.
# 3. *Thermal-resistance network.* Water film, cylindrical wall and finned air side in
#    series, with the air side weighted by the overall surface efficiency
#    $\eta_o=1-\tfrac{A_f}{A}(1-\eta_f)$ from the Gardner solution (Incropera et al.,
#    *Fundamentals of Heat and Mass Transfer*, Sec. 3.6.5).
#
# A three-level mesh study with Richardson extrapolation shows that the reported values
# are converged on the final mesh.

from dataclasses import dataclass
from importlib.metadata import version
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq
from scipy.special import i0, i1, k0, k1

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# A representative member of the parameterised disk-stacked pipe family: copper pipe
# with 20 mm bore and 2 mm wall, 0.6 mm disk fins of 70 mm diameter at 5 mm pitch, and
# twelve rounded axial grooves of 0.8 mm depth in the bore. Water at 30 °C flows at
# 0.5 m/s, air at 20 °C blows across the pipe. The water-side coefficient follows from
# the Dittus-Boelter correlation $Nu=0.023\,Re^{0.8}Pr^{0.4}$ (Dittus & Boelter, Univ.
# Calif. Publ. Eng. 2 (1930) 443); the air-side coefficient is a representative value
# for forced cross-flow over finned tubes at a few m/s.

mm = 1e-3
r_bore = 10.0 * mm         # m, bore radius at the lands (between the grooves)
r_wall = 12.0 * mm         # m, outer radius of the pipe wall = fin root radius r1
r_fin = 35.0 * mm          # m, fin tip radius r2
t_fin = 0.6 * mm           # m, fin thickness
pitch = 5.0 * mm           # m, fin pitch (length of the periodic element)
n_grooves = 12             # number of axial grooves in the bore
groove_depth = 0.8 * mm    # m, groove depth
wedge = np.pi / n_grooves  # rad, half a groove period (15 degrees)

k_cu = 401.0               # W/(m K), copper (Incropera, Table A.1, 300 K)
T_water, T_air = 303.15, 293.15   # K
h_air = 80.0               # W/(m^2 K)

# Water at 30 C: kinematic viscosity, conductivity, Prandtl number.
nu_w, k_w, Pr_w, u_w = 0.801e-6, 0.615, 5.42, 0.5
D_h = 2 * r_bore
Re_w = u_w * D_h / nu_w
h_water = 0.023 * Re_w**0.8 * Pr_w**0.4 * k_w / D_h

m_fin = np.sqrt(2 * h_air / (k_cu * t_fin))
print(f"Re_water = {Re_w:.0f},  h_water = {h_water:.0f} W/(m^2 K),  h_air = {h_air:.0f} W/(m^2 K)")
print(f"fin Biot number h t/(2k) = {h_air * t_fin / (2 * k_cu):.1e},  m (r2 - r1) = {m_fin * (r_fin - r_wall):.3f}")


# ## Geometry and mesh
#
# OneSimulation's CAD primitives (box, cylinder, annular cylinder) and its CAD meshers
# do not offer a revolved, finned solid, so the solid is built directly as a mesh: a
# block-structured, graded quadrilateral mesh of the meridian section $(x, r)$ is swept
# around the axis, every prism is split into three tetrahedra with a conforming
# diagonal rule, and the bore nodes are moved outward to cut the rounded grooves
# $r_\mathrm{in}(\theta)=r_b+d\,[\tfrac12(1+\cos 12\theta)]^2$. The tetrahedra and their
# named boundary faces are handed to `UnstructuredSimplexMesh`. The mesh is graded
# towards the two re-entrant fin-root corners, where the heat flux concentrates, and
# has several quadratic cells across the fin thickness and across the wall.

def geometric(a, b, n, growth):
    """n cells from a to b whose sizes grow by `growth` away from a."""
    sizes = growth ** np.arange(n)
    return a + (b - a) * np.concatenate(([0.0], np.cumsum(sizes) / sizes.sum()))


def profile_mesh(level):
    """Graded quadrilateral blocks of the meridian section for refinement level `level`."""
    n_gap, n_thick, n_wall, n_radial = 3 * level, 2 * level, 3 * level, 10 * level
    x_gap_right = geometric(t_fin / 2, pitch / 2, n_gap, 1.25 ** (2 / level))
    x_gap_left = -x_gap_right[::-1]
    x_fin = np.linspace(-t_fin / 2, t_fin / 2, n_thick + 1)
    r_w = geometric(r_wall, r_bore, n_wall, 1.15 ** (2 / level))[::-1]
    r_f = geometric(r_wall, r_fin, n_radial, 1.12 ** (2 / level))
    blocks = [(x_gap_left, r_w), (x_fin, r_w), (x_gap_right, r_w), (x_fin, r_f)]
    points, index, quads = [], {}, []
    for xs, rs in blocks:
        ids = np.empty((len(xs), len(rs)), dtype=int)
        for i, x in enumerate(xs):
            for j, r in enumerate(rs):
                key = (round(x / 1e-10), round(r / 1e-10))
                if key not in index:
                    index[key] = len(points)
                    points.append((x, r))
                ids[i, j] = index[key]
        for i in range(len(xs) - 1):
            for j in range(len(rs) - 1):
                quads.append((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))
    return np.array(points), np.array(quads)


# Conforming prism split (Dompierre et al., Proc. 8th Int. Meshing Roundtable, 1999):
# rotate the prism so that its smallest global vertex index comes first.
PRISM_ROTATIONS = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                            [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def split_prism(vertices):
    v = np.asarray(vertices)[PRISM_ROTATIONS[int(np.argmin(vertices))]]
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [(v[0], v[1], v[2], v[5]), (v[0], v[1], v[5], v[4]), (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]), (v[0], v[4], v[2], v[5]), (v[0], v[4], v[5], v[3])]


def groove(theta):
    return 0.25 * (1.0 + np.cos(n_grooves * theta)) ** 2


@dataclass(frozen=True, eq=False, kw_only=True)
class FinnedPipeElement(Geometry):
    """One periodic, 15-degree element of the finned pipe with named boundary faces."""

    dimension = 3

    def boundary_groups(self):
        names = ("bore", "fin_surface", "pipe_surface", "end_planes", "wedge_planes")
        return {**{name: (name,) for name in names}, "all": names}


def build_mesh(level):
    section, quads = profile_mesh(level)
    triangles = []
    for a, b, c, d in quads:
        if np.linalg.norm(section[a] - section[c]) <= np.linalg.norm(section[b] - section[d]):
            triangles += [(a, b, c), (a, c, d)]
        else:
            triangles += [(a, b, d), (b, c, d)]
    n_theta = 3 * level
    thetas = np.linspace(0.0, wedge, n_theta + 1)
    n_section = len(section)
    x = np.tile(section[:, 0], n_theta + 1)
    r = np.tile(section[:, 1], n_theta + 1)
    theta = np.repeat(thetas, n_section)
    # Cut the grooves: stretch the wall radially between r_in(theta) and r_wall.
    in_wall = r <= r_wall + 1e-12
    r_in = r_bore + groove_depth * groove(theta)
    r = np.where(in_wall, r_in + (r - r_bore) * (r_wall - r_in) / (r_wall - r_bore), r)
    coordinates = np.column_stack((x, r * np.cos(theta), r * np.sin(theta)))
    cells = []
    for j in range(n_theta):
        for a, b, c in triangles:
            lower, upper = j * n_section, (j + 1) * n_section
            cells += split_prism((a + lower, b + lower, c + lower, a + upper, b + upper, c + upper))
    cells = np.array(cells)
    p = coordinates[cells]
    signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    cells[signed < 0] = cells[signed < 0][:, [0, 2, 1, 3]]

    geometry = FinnedPipeElement()
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    tol = 1e-9
    for face in boundary_faces(cells, coordinates):
        v = np.asarray(face.vertices)
        layer, planar = v // n_section, section[v % n_section]
        if np.all(layer == 0) or np.all(layer == n_theta):
            name = "wedge_planes"
        elif np.all(np.abs(planar[:, 1] - r_bore) < tol):
            name = "bore"
        elif np.all(np.abs(np.abs(planar[:, 0]) - pitch / 2) < tol):
            name = "end_planes"
        elif np.all(np.abs(planar[:, 1] - r_wall) < tol):
            name = "pipe_surface"
        else:
            name = "fin_surface"
        groups[name].append(tuple(v))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells,
                                   boundary_facets=groups)
    return dict(geometry=geometry, mesh=mesh, coordinates=coordinates, groups=groups,
                section=section, n_section=n_section, n_theta=n_theta, level=level)


# ## Model
#
# Copper is homogeneous and isotropic; only its conductivity enters the stationary
# problem. Both convective sides use OneSimulation's `ConvectiveHeatFlux` (Robin)
# condition; the four symmetry planes carry an explicit zero `HeatFlux`.

copper = IsotropicSolid(density=8933.0, specific_heat_capacity=385.0, thermal_conductivity=k_cu,
                        youngs_modulus=110e9, poisson_ratio=0.34, thermal_expansion_coefficient=17e-6)

# Degree-2 exact quadrature on the flat boundary triangles (P2 traces are quadratic).
BARY = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def surface_integral(data, temperature, names, integrand):
    faces = np.array([f for name in names for f in data["groups"][name]])
    p = data["coordinates"][faces]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    q = np.einsum("qk,fkd->fqd", BARY, p).reshape(-1, 3)
    values = integrand(np.asarray(temperature.evaluate(q)).reshape(-1)).reshape(len(faces), 3).mean(axis=1)
    return float(np.sum(values * area)), float(area.sum())


def solve(data):
    geometry, mesh = data["geometry"], data["mesh"]
    model = HeatConduction(domain=geometry.domain(), material=copper, heating=0.0)
    b = geometry.boundaries
    problem = Problem(
        name="Finned pipe periodic element",
        geometry=geometry,
        physics=(model,),
        boundary_conditions=(
            ConvectiveHeatFlux(boundary=b.bore, physics=model,
                               heat_transfer_coefficient=h_water, ambient_temperature=T_water),
            ConvectiveHeatFlux(boundary=b.fin_surface | b.pipe_surface, physics=model,
                               heat_transfer_coefficient=h_air, ambient_temperature=T_air),
            HeatFlux(boundary=b.end_planes | b.wedge_planes, physics=model, value=0.0),
        ),
    )
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()))
    temperature = simulation.solve_stationary().field(model.temperature)

    q_water, a_bore = surface_integral(data, temperature, ("bore",), lambda T: h_water * (T_water - T))
    q_fin, a_fin = surface_integral(data, temperature, ("fin_surface",), lambda T: h_air * (T - T_air))
    q_pipe, a_pipe = surface_integral(data, temperature, ("pipe_surface",), lambda T: h_air * (T - T_air))
    # Fin base temperature: mean over the root cylinder r = r1 under the fin.
    xs = (np.arange(16) + 0.5) / 16 * t_fin - t_fin / 2
    ths = (np.arange(48) + 0.5) / 48 * wedge
    X, TH = np.meshgrid(xs, ths, indexing="ij")
    root = np.column_stack((X.ravel(), r_wall * np.cos(TH.ravel()), r_wall * np.sin(TH.ravel())))
    T_base = float(np.mean(temperature.evaluate(root)))
    nodal = np.asarray(temperature.evaluate(data["coordinates"])).reshape(-1)
    return dict(temperature=temperature, nodal=nodal, q_water=q_water, q_fin=q_fin, q_pipe=q_pipe,
                a_bore=a_bore, a_fin=a_fin, a_pipe=a_pipe, T_base=T_base,
                eta=q_fin / (h_air * a_fin * (T_base - T_air)),
                T_max=float(nodal.max()), T_min=float(nodal.min()))


# ## Solve
#
# Three meshes; the cell counts in every direction grow in the ratio 2 : 3 : 4. The
# reported heat rates are per periodic element (one pitch, full circumference), i.e.
# 24 wedges.

levels = (2, 3, 4)
runs = []
for level in levels:
    start = time.time()
    data = build_mesh(level)
    out = solve(data)
    runs.append((data, out))
    print(f"level {level}: {data['mesh'].cell_count:6d} tetrahedra, {len(data['coordinates']):6d} vertices, "
          f"Q_water = {24 * out['q_water']:.5f} W, eta_fin = {out['eta']:.5f}, "
          f"T in [{out['T_min']:.3f}, {out['T_max']:.3f}] K  ({time.time() - start:.0f} s)")
data, out = runs[-1]


# ## Comparison with the reference
#
# **Fin efficiency.** The 1D annular-fin solution needs only $m$, $r_1$, $r_2$, $t$ and
# the tip condition; it is evaluated for exactly the fin of the 3D model.

def annular_fin_convective_tip(r1, r2, t, h, k):
    m = np.sqrt(2 * h / (k * t))
    # theta = A I0(m r) + B K0(m r); theta(r1) = 1; k theta'(r2) + h theta(r2) = 0
    M = np.array([[i0(m * r1), k0(m * r1)],
                  [k * m * i1(m * r2) + h * i0(m * r2), -k * m * k1(m * r2) + h * k0(m * r2)]])
    A, B = np.linalg.solve(M, [1.0, 0.0])
    base_flux = -k * m * (A * i1(m * r1) - B * k1(m * r1))      # -k dtheta/dr at r1, per unit theta_b
    area = 2 * np.pi * (r2**2 - r1**2) + 2 * np.pi * r2 * t
    profile = lambda r: A * i0(m * r) + B * k0(m * r)
    return base_flux * 2 * np.pi * r1 * t / (h * area), profile


def gardner_adiabatic_tip(r1, r2, t, h, k):
    m = np.sqrt(2 * h / (k * t))
    num = k1(m * r1) * i1(m * r2) - i1(m * r1) * k1(m * r2)
    den = i0(m * r1) * k1(m * r2) + k0(m * r1) * i1(m * r2)
    return 2 * r1 / (m * (r2**2 - r1**2)) * num / den


eta_exact, fin_profile = annular_fin_convective_tip(r_wall, r_fin, t_fin, h_air, k_cu)
eta_gardner = gardner_adiabatic_tip(r_wall, r_fin + t_fin / 2, t_fin, h_air, k_cu)


def richardson(values, levels):
    """Observed order and extrapolated value from three meshes with sizes h ~ 1/level."""
    f1, f2, f3 = values
    h1, h2, h3 = (1.0 / np.asarray(levels, float))
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf

    def mismatch(p):
        return (h1**p - h2**p) / (h2**p - h3**p) - ratio
    try:
        p = brentq(mismatch, 0.2, 10.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


etas = [o["eta"] for _, o in runs]
p_eta, eta_ext = richardson(etas, levels)
print("Fin efficiency")
print(f"  3D model (finest mesh)                 eta = {out['eta']:.5f}")
print(f"  Richardson extrapolation (order {p_eta:.2f})  eta = {eta_ext:.5f}")
print(f"  Bessel solution, convective tip        eta = {eta_exact:.5f}   deviation {100 * (out['eta'] / eta_exact - 1):+.3f} %")
print(f"  Gardner (1945), corrected radius        eta = {eta_gardner:.5f}   deviation {100 * (out['eta'] / eta_gardner - 1):+.3f} %")
print(f"  change between the last two meshes: {100 * (etas[-1] / etas[-2] - 1):+.4f} %")

# **Energy balance and thermal-resistance network.**
q_air = out["q_fin"] + out["q_pipe"]
area_air = out["a_fin"] + out["a_pipe"]
eta_overall = 1.0 - out["a_fin"] / area_air * (1.0 - eta_exact)
mean_bore_radius = r_bore + groove_depth * 3.0 / 8.0          # mean of the groove function is 3/8
R_water = 1.0 / (h_water * out["a_bore"])
R_wall = np.log(r_wall / mean_bore_radius) / (wedge * k_cu * pitch)
R_air = 1.0 / (eta_overall * h_air * area_air)
q_network = (T_water - T_air) / (R_water + R_wall + R_air)
print("Heat rate per periodic element (one pitch, full circumference)")
print(f"  from the water  {24 * out['q_water']:.5f} W")
print(f"  to the air      {24 * q_air:.5f} W   (fins {100 * out['q_fin'] / q_air:.1f} %, bare pipe {100 * out['q_pipe'] / q_air:.1f} %)")
print(f"  energy imbalance {abs(out['q_water'] - q_air) / out['q_water']:.2e}")
print(f"  resistance network with Gardner efficiency  {24 * q_network:.5f} W   deviation {100 * (out['q_water'] / q_network - 1):+.3f} %")
print(f"  heat dissipation per length and kelvin  h_L = Q/(p dT) = {24 * out['q_water'] / (pitch * (T_water - T_air)):.2f} W/(m K)")

qs = [24 * o["q_water"] for _, o in runs]
print("Mesh study (sizes h ~ 1/level)")
for name, key in (("Q_water / W", "q_water"), ("eta_fin", "eta"), ("T_max / K", "T_max"), ("T_min / K", "T_min")):
    vals = [o[key] * (24 if key == "q_water" else 1) for _, o in runs]
    p, ext = richardson(vals, levels)
    print(f"  {name:12s} " + "  ".join(f"{v:.6f}" for v in vals)
          + f"   order {p:.2f}, extrapolated {ext:.6f}, last change {abs(vals[-1] - vals[-2]):.1e}")
print(f"  base temperature T_b = {out['T_base']:.4f} K, tip temperature (mean) "
      f"{np.mean(out['nodal'][np.isclose(np.hypot(data['coordinates'][:, 1], data['coordinates'][:, 2]), r_fin)]):.4f} K")


# ## Figure
#
# Left: the computed surface temperature of five pitches of the pipe, assembled from
# the periodic, mirror-symmetric element and cut open to show the bore and the wall.
# Middle: the temperature over the cross-section through the fin mid-plane, where the
# grooves imprint a circumferential variation on the wall. Right: the fin temperature
# along its mid-plane against the Bessel solution, and the mesh study.

coords, nodal, groups = data["coordinates"], out["nodal"], data["groups"]
norm = colors.Normalize(vmin=nodal.min(), vmax=nodal.max())
cmap = cm.plasma


def rotated(points, angle, mirror):
    y, z = points[..., 1], points[..., 2] * (-1 if mirror else 1)
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((points[..., 0], c * y - s * z, s * y + c * z), axis=-1)


cream = "#fbf5eb"
fig = plt.figure(figsize=(16, 5.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax3d = fig.add_subplot(1, 4, (1, 2), projection="3d")
ax3d.set_facecolor(cream)
faces = np.array(groups["fin_surface"] + groups["pipe_surface"] + groups["bore"])
cut = np.array([f for f in groups["wedge_planes"] if np.all(np.abs(coords[list(f), 2]) < 1e-12)])
polys, cols = [], []
for k in range(18):                                  # 270 degrees, open towards the viewer
    angle = np.pi + (k + (k % 2)) * wedge
    shown = np.concatenate((faces, cut)) if k in (0, 17) else faces
    face_T = nodal[shown].mean(axis=1)
    tri = rotated(coords[shown], angle, mirror=bool(k % 2))
    for shift in (-2, -1, 0, 1, 2):
        polys.append(tri + np.array([shift * pitch, 0.0, 0.0]))
        cols.append(face_T)
polys = np.concatenate(polys) / mm
ax3d.add_collection3d(Poly3DCollection(polys, facecolors=cmap(norm(np.concatenate(cols))),
                                       edgecolors="none", linewidths=0))
lim = r_fin / mm
ax3d.set(xlim=(-2.6 * pitch / mm, 2.6 * pitch / mm), ylim=(-lim, lim), zlim=(-lim, lim),
         xlabel="x / mm", ylabel="y / mm", zlabel="z / mm")
ax3d.set_box_aspect((5.2 * pitch, 2 * r_fin, 2 * r_fin), zoom=1.2)
ax3d.view_init(elev=25, azim=-62)
ax3d.set_xticks([-10, 0, 10])
ax3d.set_title("Surface temperature, five pitches (cut open)")
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3d, shrink=0.7, label="T / K")

# Cross-section through the fin mid-plane x = 0 (wall and fin), assembled to 360 degrees.
ax_cut = fig.add_subplot(1, 4, 3)
ax_cut.set_facecolor("#fffaf2")
r_show = 15.0 * mm
n_r, n_t = 40, 24
S, TH = np.meshgrid(np.linspace(0, 1, n_r + 1), np.linspace(0, wedge, n_t + 1), indexing="ij")
R_in = r_bore + groove_depth * groove(TH) + 0.02 * mm          # just inside the faceted bore
R = R_in + S * (r_show - R_in)
wedge_pts = np.column_stack((np.zeros(R.size), (R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()))
T_wedge = np.asarray(out["temperature"].evaluate(wedge_pts)).reshape(-1)
grid = np.arange(R.size).reshape(R.shape)
cell_tris = np.concatenate((np.stack((grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:]), -1).reshape(-1, 3),
                            np.stack((grid[:-1, :-1], grid[1:, 1:], grid[:-1, 1:]), -1).reshape(-1, 3)))
pts, T_cut, tris = [], [], []
for k in range(2 * n_grooves):                       # mirror-and-rotate the wedge to 360 degrees
    tris.append(cell_tris + k * R.size)
    pts.append(rotated(wedge_pts, (k + k % 2) * wedge, bool(k % 2)))
    T_cut.append(T_wedge)
pts, T_cut = np.concatenate(pts), np.concatenate(T_cut)
triang = Triangulation(pts[:, 1] / mm, pts[:, 2] / mm, np.concatenate(tris))
levels_T = np.linspace(T_cut.min(), T_cut.max(), 40)
ax_cut.tricontourf(triang, T_cut, levels=levels_T, cmap=cmap)
ax_cut.set_aspect("equal")
ax_cut.set(xlim=(-14, 14), ylim=(-14, 14), xlabel="y / mm", ylabel="z / mm",
           title=f"Mid-plane section near the bore\nT from {T_cut.min():.2f} to {T_cut.max():.2f} K")

ax_fin = fig.add_subplot(1, 4, 4)
ax_fin.set_facecolor("#fffaf2")
r_line = np.linspace(r_wall, r_fin * (1 - 1e-6), 80)
for th, style in ((0.0, "o"), (wedge, "s")):
    line_pts = np.column_stack((np.zeros_like(r_line), r_line * np.cos(th), r_line * np.sin(th)))
    T_line = np.asarray(out["temperature"].evaluate(line_pts)).reshape(-1)
    ax_fin.plot(r_line[::4] / mm, ((T_line - T_air) / (out["T_base"] - T_air))[::4], style, ms=4, mfc="none",
                label=f"3D model, theta = {np.degrees(th):.0f} deg")
ax_fin.plot(r_line / mm, fin_profile(r_line), "-", color="#7a3b12", lw=1.6, label="Bessel solution (1D fin)")
ax_fin.set(xlabel="r / mm", ylabel="(T - T_air) / (T_b - T_air)", title="Fin temperature along its mid-plane")
ax_fin.grid(alpha=0.3)
inset = ax_fin.inset_axes([0.42, 0.45, 0.53, 0.28])
inset.plot(levels, etas, "o-", color="#b5532a", label="3D model")
inset.axhline(eta_exact, color="#7a3b12", lw=1.2, ls="--", label="Bessel")
inset.set_xticks(levels)
inset.set_xlabel("mesh level", fontsize=8)
inset.set_ylabel("fin efficiency", fontsize=8)
inset.tick_params(labelsize=7)
inset.ticklabel_format(axis="y", useOffset=False)
inset.legend(fontsize=7)
ax_fin.legend(loc="lower left", fontsize=8)
plt.show()


# ## Going further
#
# Change the fin pitch, thickness or tip radius and watch the efficiency follow the
# Bessel solution; switch the fin material to aluminium ($k\approx 237$ W/(m K)) or
# stainless steel ($k\approx 15$ W/(m K)) to drive the efficiency down. Replace the
# uniform air-side coefficient by a circumferential distribution for cross-flow (higher
# at the front stagnation line) to see the field lose its rotational symmetry; the
# element must then span the full circumference.
