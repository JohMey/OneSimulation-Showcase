"""OneSimulation example: harmonic response of a viscoelastic structural damper.
Requires the installed, activated OneSimulation runtime.

Two viscoelastic layers bonded between three steel plates damp wind and earthquake
vibrations of buildings. The full 3D device is driven by pin forces at 3 Hz. The
18-branch generalized-Maxwell (Prony) material is solved in the frequency domain with
its exact complex shear modulus on quadratic tetrahedra, and independently in the time
domain with an exponential Prony integrator. The results are checked against the
closed-form storage and loss moduli and the simple-shear layer, the energy balance per
cycle, a three-level mesh study and the convergence of the time-domain solution to the
frequency-domain motion as the time step shrinks.
"""


# # Viscoelastic structural damper under harmonic load
#
# Viscoelastic dampers reduce the wind-induced sway and the seismic response of tall
# buildings: a stiff steel frame forces thin layers of a soft polymer into shear, and the
# polymer turns part of every vibration cycle into heat. The device studied here
# (K. L. Shen and T. T. Soong, *J. Eng. Mech.* 121, 694-701, 1995; S. W. Park,
# *Int. J. Solids Struct.* 38, 8065-8092, 2001) has a central steel plate, bolted to the structure
# through two holes at the bottom, and two outer steel plates, each bonded to the central
# plate by one viscoelastic layer and pinned to the bracing at the top. The plates lie in
# the $(y,z)$ plane and are stacked along $x$:
#
# | part | position along $x$ | footprint in $(y,z)$ |
# |---|---|---|
# | left plate, 10 mm | 0 ... 10 mm | 50 × 50 mm pad plus an upward tab with one Ø16 mm pin hole |
# | viscoelastic layer 1, 10 mm | 10 ... 20 mm | 50 × 50 mm |
# | central plate, 10 mm | 20 ... 30 mm | pad plus a downward tab with two Ø16 mm bolt holes |
# | viscoelastic layer 2, 10 mm | 30 ... 40 mm | 50 × 50 mm |
# | right plate, 10 mm | 40 ... 50 mm | like the left plate |
#
# The original description gives the device only as a drawing; the dimensions above are
# a reconstruction with the same layout: 10 mm plates and Ø16 mm pin holes (consistent with
# the 16 mm × 10 mm bearing area of the original load definition) and 50 × 50 × 10 mm layers.
#
# **Material.** The layers follow the generalized Maxwell model of Park (2001): an
# equilibrium spring $G_\infty$ in parallel with 18 spring-dashpot branches $(G_k,\tau_k)$
# acting on the strain deviator, with an elastic bulk modulus $K$,
#
# $$\boldsymbol\sigma = K\,\mathrm{tr}\,\boldsymbol\varepsilon\,\mathbf I
#   + 2\int_{0}^{t} G(t-s)\,\dot{\mathbf e}(s)\,\mathrm ds,\qquad
#   G(t)=G_\infty+\sum_{k=1}^{18}G_k\,e^{-t/\tau_k}.$$
#
# For harmonic motion $e^{i\omega t}$ the relaxation integral becomes the exact complex
# shear modulus (Ferry, *Viscoelastic Properties of Polymers*, 3rd ed., Wiley 1980)
#
# $$G^*(\omega)=G'+iG''=G_\infty+\sum_k G_k\,\frac{i\omega\tau_k}{1+i\omega\tau_k},\qquad
#   G'=G_\infty+\sum_k G_k\frac{\omega^2\tau_k^2}{1+\omega^2\tau_k^2},\quad
#   G''=\sum_k G_k\frac{\omega\tau_k}{1+\omega^2\tau_k^2}.$$
#
# The plates are steel (AISI 4340), linear elastic.
#
# **Loads and supports** (as in the original set-up). The two bottom bolt holes are fixed.
# The right pin hole carries the vertical load $8.5\,\text{MPa}\cdot\cos(\omega t)$ and is
# guided in $x$ and $y$; the left pin hole carries $(0.5,\,0,\,8.5)\,\text{MPa}\cdot
# \sin(\omega t)$ and is guided in $y$; $f=3$ Hz. The loads are tractions on the hole
# surfaces whose resultant is $8.5\,\text{MPa}\times\pi\times16\,\text{mm}\times10\,\text{mm}
# = 4.27$ kN per plate. All other faces are traction-free.
#
# **Methods.** (1) Frequency domain: by the elastic-viscoelastic correspondence principle
# the steady harmonic state solves one complex linear problem
# $(\mathbf K_\text{steel}+\mathbf K_K+G^*(\omega)\,\mathbf B-\omega^2\mathbf M)\,\hat{\mathbf u}
# =\hat{\mathbf f}$, where $\mathbf B$ is the deviatoric stiffness of the layers for a unit
# shear modulus. All matrices are assembled from OneSimulation's continuous Galerkin
# space of quadratic (P2) Lagrange tetrahedra on a block-structured, graded mesh of the
# whole device; the complex system is solved with a sparse direct LU factorisation
# (nested-dissection ordering from PETSc, one step of iterative refinement).
# (2) Time domain: each Prony branch keeps a history vector $\mathbf h_k$ with
# $\dot{\mathbf h}_k+\mathbf h_k/\tau_k=\dot{\mathbf u}$, integrated with the exponential
# (recursive-convolution) rule of R. L. Taylor, K. S. Pister and G. L. Goudreau
# (*Int. J. Numer. Meth. Eng.* 2, 45-59, 1970). It is exact for motion
# that is linear within a step and unconditionally stable for all 18 relaxation times from
# $10^{-7}$ to $10^{3}$ s; it is second-order accurate for the branches with
# $\tau_k\gg\Delta t$ and acts as a backward-Euler dashpot for the much faster ones, so the
# overall convergence rate lies between one and two. The effective stiffness is constant,
# so it is factorised once per time-step size. The start-up from rest is followed for 3 s (nine
# cycles) with the loads ramped in over 0.1 s, as in the original study; separate runs
# that start from the exact steady state measure the integrator error alone.
#
# **Reference.** There is no closed-form solution for the full device. The notebook uses
# (i) the exact storage and loss moduli of the Prony series above; (ii) the closed-form
# simple-shear model of one layer between rigid plates, $\hat w=\hat F h/(G^*A)$, phase lag
# $\delta=\arg G^*$ and dissipated energy per cycle $W=\pi\,G''\,\hat\gamma^2\,V
# =\pi\,\hat\tau^2\,G''V/|G^*|^2$, which the 3D field must approach (it differs by the
# free-edge, bending and steel-compliance effects that only the 3D model contains);
# (iii) the energy balance: the work of the pin forces per cycle (the hysteresis-loop area)
# must equal the dissipation integrated over the layers; (iv) a three-level mesh study with
# observed order and Richardson extrapolation; (v) the time-domain solution, which must
# converge to the frequency-domain motion as the time step shrinks; (vi) the
# assembled stiffness, checked against OneSimulation's `SmallStrainSolid` solver on the
# same mesh with the real storage modulus.

from dataclasses import dataclass
from importlib.metadata import version
import time

import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from petsc4py import PETSc

from onesim.cad.geometry.primitives import Geometry
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.physics.solid_mechanics.elasticity.models.declarations.linear_elasticity import IsotropicHookeMaterial
from onesim.physics.solid_mechanics.kinematics.models.voigt_strain import voigt_strain_matrix

print("OneSimulation:", version("onesimulation"))
start_time = time.perf_counter()


# ## Parameters
#
# Prony series of the damper polymer (Park 2001), steel AISI 4340, dimensions and loads.

mm = 1e-3
G_branch = np.array([13.3, 286, 291, 212, 112, 61.6, 29.8, 16.1, 7.83, 4.15, 2.03, 1.11,
                     0.491, 0.326, 0.0825, 0.126, 0.0373, 0.0118]) * 1e6        # Pa
tau_branch = np.array([1e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3,
                       1e-2, 3.16e-2, 0.1, 0.316, 1.0, 3.16, 10.0, 100.0, 1000.0])  # s
G_inf = 58.6e3            # Pa, long-time shear modulus
K_layer = 0.4e9           # Pa, bulk modulus of the polymer
rho_layer = 1060.0        # kg/m^3
E_steel, nu_steel, rho_steel = 205e9, 0.28, 7850.0

t_plate, h_layer = 10 * mm, 10 * mm
pad_w, pad_h = 50 * mm, 50 * mm                  # layer footprint (y, z)
A_layer, V_layer = pad_w * pad_h, pad_w * pad_h * h_layer
tab_y0, tab_y1 = 7 * mm, 43 * mm                 # tab width 36 mm, centred at y = 25 mm
y_c, r_hole = 25 * mm, 8 * mm
z_bolts, z_pin = (-50 * mm, -80 * mm), 127 * mm  # hole centres

frequency = 3.0                                  # Hz
omega = 2 * np.pi * frequency
q_load = 8.5e6                                   # Pa, documented hole traction amplitude
F_pin = q_load * np.pi * 2 * r_hole * t_plate    # N, resultant per pin (4.27 kN)
sweep = (0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)   # Hz
levels = (1, 2, 3)


def complex_modulus(f):
    """Exact complex shear modulus G* of the Prony series at frequency f (Hz)."""
    w = 2 * np.pi * np.atleast_1d(f)[:, None]
    return G_inf + np.sum(G_branch * 1j * w * tau_branch / (1 + 1j * w * tau_branch), axis=1)


G_star = complex_modulus(frequency)[0]
delta_material = np.angle(G_star)
print(f"Prony series at {frequency:g} Hz: G' = {G_star.real/1e6:.4f} MPa, G'' = {G_star.imag/1e6:.4f} MPa, "
      f"loss factor tan(delta) = {G_star.imag/G_star.real:.4f}, delta = {np.degrees(delta_material):.2f} deg")
print(f"glassy modulus G_inf + sum G_k = {(G_inf + G_branch.sum())/1e6:.0f} MPa, K/|G*| = {K_layer/abs(G_star):.0f}")
print(f"pin force amplitude F = {F_pin:.1f} N per outer plate")


# ## Geometry and mesh
#
# The in-plane shape (pad, tabs and the three holes) is meshed with quadrilateral blocks
# (transfinite interpolation): the pad is graded towards its four free edges, where the
# layers have their edge zones, and every hole is surrounded by an O-grid of four blocks
# graded towards the hole. Every quadrilateral is split into two triangles, the plane
# mesh is extruded along $x$ through the five parts (pad + up tab for the outer plates,
# pad + down tab for the central plate, pad only for the layers), and each prism is split
# into three tetrahedra with a vertex-ordering rule that keeps neighbouring faces
# conforming. The layers and plates therefore share their nodes on the bonded faces, and
# every tetrahedron lies in one material. Level $L$ uses $2L$ cells per quarter hole,
# $2L$ cells across each layer and $L+1$ across each plate; the displacement is quadratic
# (P2).


def line(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return lambda t: (1 - t)[:, None] * p + t[:, None] * q


def arc(centre, radius, a0, a1):
    c = np.asarray(centre, float)
    return lambda t: c + radius * np.column_stack((np.cos(a0 + (a1 - a0) * t), np.sin(a0 + (a1 - a0) * t)))


def transfinite(bottom, top, left, right, u, v):
    B, T, L, R = bottom(u), top(u), left(v), right(v)
    U, V = np.meshgrid(u, v, indexing="ij")
    U, V = U[..., None], V[..., None]
    return ((1 - V) * B[:, None] + V * T[:, None] + (1 - U) * L[None] + U * R[None]
            - (1 - U) * (1 - V) * B[0] - U * (1 - V) * B[-1] - (1 - U) * V * T[0] - U * V * T[-1])


def graded(n, ratio):
    """n cells on [0, 1]; the last cell is `ratio` times the first."""
    g = ratio ** (1.0 / max(n - 1, 1))
    sizes = g ** np.arange(n)
    x = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    x[-1] = 1.0
    return x


def two_sided(n, ratio):
    """n cells on [0, 1], refined towards both ends (centre cells `ratio` times larger)."""
    sizes = 1.0 + (ratio - 1.0) * np.sin(np.pi * (np.arange(n) + 0.5) / n)
    x = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    x[-1] = 1.0
    return x


def plane_blocks(L):
    """Quadrilateral blocks of the (y, z) footprint: (region, bottom, top, left, right, u, v)."""
    n_tab, n_radial, n_margin, n_pad = 2 * L, L + 1, L + 1, 4 * L + 2
    across = np.linspace(0.0, 1.0, n_tab + 1)
    margin = graded(n_margin, 3.0)                       # fine at the free pad edge
    z_pad = two_sided(n_pad, 3.0)
    radial = graded(n_radial, 1 / 2.5)                   # fine at the hole
    blocks = []
    for ya, yb, u in ((0.0, tab_y0, margin), (tab_y0, tab_y1, across), (tab_y1, pad_w, 1 - margin[::-1])):
        blocks.append(("pad", line((ya, 0), (yb, 0)), line((ya, pad_h), (yb, pad_h)),
                       line((ya, 0), (ya, pad_h)), line((yb, 0), (yb, pad_h)), u, z_pad))
    for region, za, zb, n in (("down", -35 * mm, 0.0, 2 * L + 1), ("up", pad_h, 112 * mm, 4 * L + 2)):
        blocks.append((region, line((tab_y0, za), (tab_y1, za)), line((tab_y0, zb), (tab_y1, zb)),
                       line((tab_y0, za), (tab_y0, zb)), line((tab_y1, za), (tab_y1, zb)),
                       across, np.linspace(0.0, 1.0, n + 1)))
    for region, zc in (("down", z_bolts[0]), ("down", z_bolts[1]), ("up", z_pin)):
        z0, z1 = zc - 15 * mm, zc + 15 * mm
        corners = ((tab_y0, z0), (tab_y1, z0), (tab_y1, z1), (tab_y0, z1))
        for k in range(4):
            a = (1.25 + 0.5 * k) * np.pi
            ends = [np.array((y_c, zc)) + r_hole * np.array((np.cos(t), np.sin(t))) for t in (a, a + np.pi / 2)]
            blocks.append((region, line(corners[k], corners[(k + 1) % 4]), arc((y_c, zc), r_hole, a, a + np.pi / 2),
                           line(corners[k], ends[0]), line(corners[(k + 1) % 4], ends[1]), across, radial))
    return blocks


def plane_mesh(L):
    points, index, quads, regions = [], {}, [], []
    for region, bottom, top, left, right, u, v in plane_blocks(L):
        X = transfinite(bottom, top, left, right, u, v)
        ids = np.empty(X.shape[:2], dtype=int)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                key = (round(X[i, j, 0] / 1e-9), round(X[i, j, 1] / 1e-9))
                if key not in index:
                    index[key] = len(points)
                    points.append(X[i, j])
                ids[i, j] = index[key]
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                quads.append((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))
                regions.append(region)
    return np.array(points), np.array(quads), np.array(regions)


PRISM_ROTATIONS = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                            [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def split_prism(vertices):
    v = np.asarray(vertices)[PRISM_ROTATIONS[int(np.argmin(vertices))]]
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [(v[0], v[1], v[2], v[5]), (v[0], v[1], v[5], v[4]), (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]), (v[0], v[4], v[2], v[5]), (v[0], v[4], v[5], v[3])]


PARTS = (("left", 0.0, t_plate, ("pad", "up")), ("layer_1", t_plate, h_layer, ("pad",)),
         ("centre", t_plate + h_layer, t_plate, ("pad", "down")),
         ("layer_2", 2 * t_plate + h_layer, h_layer, ("pad",)),
         ("right", 2 * t_plate + 2 * h_layer, t_plate, ("pad", "up")))


def layer_index(x):
    """1 or 2 inside the viscoelastic layers, 0 in the steel."""
    x = np.asarray(x)
    return np.where((x > t_plate) & (x < t_plate + h_layer), 1,
                    np.where((x > 2 * t_plate + h_layer) & (x < 2 * t_plate + 2 * h_layer), 2, 0))


@dataclass(frozen=True, eq=False, kw_only=True)
class Damper(Geometry):
    """Steel plates and viscoelastic layers with named hole surfaces."""

    dimension = 3

    @property
    def volume(self):
        return self.domain()

    def boundary_groups(self):
        names = ("bottom_holes", "left_hole", "right_hole", "free")
        return {**{name: (name,) for name in names}, "all": names}


def build_mesh(L):
    plane, quads, regions = plane_mesh(L)
    triangles, tri_regions = [], []
    for (a, b, c, d), region in zip(quads, regions):
        if np.linalg.norm(plane[a] - plane[c]) <= np.linalg.norm(plane[b] - plane[d]):
            triangles += [(a, b, c), (a, c, d)]
        else:
            triangles += [(a, b, d), (b, c, d)]
        tri_regions += [region, region]
    triangles, tri_regions = np.array(triangles), np.array(tri_regions)
    x_nodes, slabs = [0.0], []
    for name, x0, thickness, footprint in PARTS:
        n = 2 * L if name.startswith("layer") else L + 1
        for k in range(1, n + 1):
            x_nodes.append(x0 + thickness * k / n)
            slabs.append(footprint)
    n_plane, node_of, coordinates, cells = len(plane), {}, [], []

    def node(key):
        if key not in node_of:
            node_of[key] = len(coordinates)
            coordinates.append((x_nodes[key // n_plane], *plane[key % n_plane]))
        return node_of[key]

    for k, footprint in enumerate(slabs):
        for a, b, c in triangles[np.isin(tri_regions, footprint)]:
            prism = (a + n_plane * k, b + n_plane * k, c + n_plane * k,
                     a + n_plane * (k + 1), b + n_plane * (k + 1), c + n_plane * (k + 1))
            cells += [tuple(node(int(v)) for v in tet) for tet in split_prism(prism)]
    coordinates, cells = np.array(coordinates), np.array(cells)
    p = coordinates[cells]
    signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    cells[signed < 0] = cells[signed < 0][:, [0, 2, 1, 3]]
    geometry = Damper()
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    for face in boundary_faces(cells, coordinates):
        q = coordinates[np.asarray(face.vertices)]
        on_hole = lambda zc: np.all(np.abs(np.hypot(q[:, 1] - y_c, q[:, 2] - zc) - r_hole) < 1e-7)
        if on_hole(z_bolts[0]) or on_hole(z_bolts[1]):
            name = "bottom_holes"
        elif on_hole(z_pin):
            name = "left_hole" if q[:, 0].max() <= t_plate + 1e-9 else "right_hole"
        else:
            name = "free"
        groups[name].append(tuple(face.vertices))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells, boundary_facets=groups)
    return geometry, mesh, groups


# ## Model
#
# The stiffness and mass matrices are assembled cell by cell from OneSimulation's P2
# tetrahedral Lagrange space (27-point quadrature, exact for the quadratic stiffness and
# the quartic mass integrands). Steel cells contribute $\int\mathbf S^T\mathbf C_\text{steel}
# \mathbf S$; layer cells contribute the bulk part $K\,\mathbf m\mathbf m^T$ and, separately
# for each layer, the unit-shear-modulus deviatoric part $\mathbf B_j$, so that any shear
# modulus, real or complex, enters as $G\,\mathbf B_j$. Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | two bottom bolt holes (central plate) | fixed, $\mathbf u=0$ |
# | right pin hole | $u_x=u_y=0$, traction $t_z=\tilde q\cos\omega t$ |
# | left pin hole | $u_y=0$, traction $(t_x,t_z)=(\tilde q/17,\ \tilde q)\sin\omega t$ |
# | all other faces (plate faces, edges, layer edges, tab ends) | traction-free |
#
# The straight-sided P2 faces approximate each hole by a polygonal cylinder, so the hole
# traction $\tilde q$ is scaled from 8.5 MPa by the ratio of the exact to the polygonal
# hole area: the resultant is exactly the documented 4.27 kN on every mesh.

D_steel = np.asarray(IsotropicHookeMaterial(E_steel, nu_steel).tangent_voigt())
m_vec = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
D_bulk = K_layer * np.outer(m_vec, m_vec)
D_shear = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]) - 2.0 / 3.0 * np.outer(m_vec, m_vec)   # sigma = G * D_shear eps


def assemble(L):
    """Mesh, P2 space, matrices, pin loads and supports of mesh level L."""
    geometry, mesh, groups = build_mesh(L)
    space = FiniteElement(element=Lagrange(order=2)).bind_space(mesh)
    n = 3 * space.dof_count
    corner_x = np.asarray(space.dof_coordinates)[space.cell_dofs[:, :4], 0].mean(axis=1)
    owner = layer_index(corner_x)
    rows, cols = [], []
    blocks = {key: [] for key in ("steel", "bulk", "B1", "B2", "mass")}
    zero = np.zeros((30, 30))
    for cell, nodes in enumerate(space.cell_dofs):
        _, weights, (basis, gradient, _) = space.element_data(cell, order=3)
        S = voigt_strain_matrix(gradient)                                  # (q, 30, 6)
        SW = (S * weights[:, None, None]).transpose(1, 0, 2).reshape(30, -1)
        local = lambda D: SW @ (S @ D).transpose(1, 0, 2).reshape(30, -1).T
        index = (np.asarray(nodes)[:, None] * 3 + np.arange(3)).ravel()
        rows.append(np.repeat(index, 30))
        cols.append(np.tile(index, 30))
        density = rho_steel if owner[cell] == 0 else rho_layer
        blocks["mass"].append(np.kron(density * (basis.T * weights) @ basis, np.eye(3)))
        blocks["steel"].append(local(D_steel) if owner[cell] == 0 else zero)
        blocks["bulk"].append(local(D_bulk) if owner[cell] else zero)
        blocks["B1"].append(local(D_shear) if owner[cell] == 1 else zero)
        blocks["B2"].append(local(D_shear) if owner[cell] == 2 else zero)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    matrices = {key: coo_matrix((np.asarray(value).ravel(), (rows, cols)), shape=(n, n)).tocsr()
                for key, value in blocks.items()}
    b = geometry.boundaries

    def pin_load(boundary, direction):
        faces = mesh.facets(boundary)
        traces = [space.trace_data(face) for face in faces]
        polygon_area = sum(trace[2].sum() for trace in traces)
        traction = q_load * (np.pi * 2 * r_hole * t_plate) / polygon_area * np.asarray(direction)
        f = np.zeros(n)
        for cell, _, weights, _, basis, _ in traces:
            index = (np.asarray(space.cell_dofs[cell])[:, None] * 3 + np.arange(3)).ravel()
            f[index] += np.outer(basis.T @ weights, traction).ravel()
        return f

    def nodes_of(boundary):
        return np.asarray(space.boundary_dofs(mesh.facets(boundary)))

    f_right = pin_load(b.right_hole, (0.0, 0.0, 1.0))           # multiplies cos(wt)
    f_left = pin_load(b.left_hole, (0.5 / 8.5, 0.0, 1.0))       # multiplies sin(wt)
    fixed = np.unique(np.concatenate(
        [3 * nodes_of(b.bottom_holes) + c for c in range(3)]
        + [3 * nodes_of(b.right_hole) + c for c in (0, 1)] + [3 * nodes_of(b.left_hole) + 1]))
    return dict(level=L, geometry=geometry, mesh=mesh, groups=groups, space=space, n=n, M=matrices,
                f_right=f_right, f_left=f_left, free=np.setdiff1d(np.arange(n), fixed),
                left_nodes=nodes_of(b.left_hole), right_nodes=nodes_of(b.right_hole))


def direct_solver(A):
    """Sparse direct LU of A with a nested-dissection ordering and one refinement step."""
    pattern = abs(A).tocsr()
    pattern.sort_indices()
    graph = PETSc.Mat().createAIJ(size=pattern.shape, csr=(pattern.indptr.astype(PETSc.IntType),
                                                           pattern.indices.astype(PETSc.IntType), pattern.data))
    order = graph.getOrdering("nd")[0].getIndices()
    graph.destroy()
    lu = splu(A[order][:, order].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
              options=dict(SymmetricMode=True))

    def solve(rhs):
        x = np.empty_like(rhs, dtype=np.result_type(rhs, A.dtype))
        x[order] = lu.solve(rhs[order])
        r = rhs - A @ x
        x[order] += lu.solve(r[order])
        return x

    return solve


def harmonic(model, G, w, inertia=True):
    """Complex displacement amplitude for shear modulus G (complex) at angular frequency w."""
    M, free = model["M"], model["free"]
    A = M["steel"] + M["bulk"] + G * (M["B1"] + M["B2"]) - (w**2 * M["mass"] if inertia else 0.0)
    f = model["f_right"] - 1j * model["f_left"]          # cos -> 1, sin -> -i
    u = np.zeros(model["n"], complex)
    u[free] = direct_solver(A.tocsr()[free][:, free])(f[free])
    return u


def observables(model, u, G):
    """Pin-hole displacements, dissipated energy per cycle per layer and work of the loads."""
    U = u.reshape(-1, 3)
    f = model["f_right"] - 1j * model["f_left"]
    return dict(
        w_right=U[model["right_nodes"], 2].mean(), w_left=U[model["left_nodes"], 2].mean(),
        W_1=np.pi * G.imag * np.real(np.conj(u) @ (model["M"]["B1"] @ u)),
        W_2=np.pi * G.imag * np.real(np.conj(u) @ (model["M"]["B2"] @ u)),
        W_loads=-np.pi * np.imag(np.conj(f) @ u))


# ## Solve
#
# **Frequency domain, mesh study.** Each level is assembled and solved once at 3 Hz
# (with inertia); the steady loop follows as $w(t)=\Re(\hat w\,e^{i\omega t})$. The
# middle level is then reused for the library cross-check, the time-domain runs and a
# frequency sweep from 0.125 to 5 Hz.

runs = {}
for L in levels:
    tic = time.perf_counter()
    model = assemble(L)
    u = harmonic(model, G_star, omega)
    runs[L] = dict(model=model, u=u, **observables(model, u, G_star))
    r = runs[L]
    print(f"level {L}: {model['mesh'].cell_count:6d} tetrahedra, {model['n']:7d} unknowns, "
          f"|w_right| = {abs(r['w_right'])/mm:.5f} mm, W_1 + W_2 = {r['W_1'] + r['W_2']:.4f} J "
          f"({time.perf_counter() - tic:.0f} s)")
fine = runs[levels[-1]]
model, u_fine = fine["model"], fine["u"]
middle = runs[levels[1]]

# Library cross-check: the assembled matrices with the real storage modulus G' against
# OneSimulation's own small-strain solver with the same layered stiffness and supports.
check = middle["model"]
D_layer = np.asarray(IsotropicHookeMaterial(9 * K_layer * G_star.real / (3 * K_layer + G_star.real),
                                            (3 * K_layer - 2 * G_star.real) / (2 * (3 * K_layer + G_star.real))
                                            ).tangent_voigt())
b = check["geometry"].boundaries
library = SmallStrainSolid(youngs_modulus=E_steel, poisson_ratio=nu_steel, kinematics="volume").solve(
    mesh=check["mesh"], times=(0.0,), order=2,
    elastic_stiffness=lambda p, t: np.where((layer_index(p[:, 0]) > 0)[:, None, None], D_layer, D_steel),
    displacements=((b.bottom_holes, "x", 0.0), (b.bottom_holes, "y", 0.0), (b.bottom_holes, "z", 0.0),
                   (b.right_hole, "x", 0.0), (b.right_hole, "y", 0.0), (b.left_hole, "y", 0.0)),
    tractions=((b.right_hole, (0.0, 0.0, q_load)), (b.left_hole, (0.0, 0.0, 0.0)), (b.free, (0.0, 0.0, 0.0))))
A_real = check["M"]["steel"] + check["M"]["bulk"] + G_star.real * (check["M"]["B1"] + check["M"]["B2"])
q_right = np.zeros(check["n"])
for face in check["mesh"].facets(b.right_hole):
    cell, _, weights, _, basis, _ = check["space"].trace_data(face)
    index = (np.asarray(check["space"].cell_dofs[cell])[:, None] * 3 + np.arange(3)).ravel()
    q_right[index] += np.outer(basis.T @ weights, (0.0, 0.0, q_load)).ravel()
u_real = np.zeros(check["n"])
u_real[check["free"]] = direct_solver(A_real.tocsr()[check["free"]][:, check["free"]])(q_right[check["free"]])
library_difference = np.abs(u_real - library.displacement[-1].ravel()).max() / np.abs(u_real).max()

# **Time domain** on the middle mesh, quasi-static. With a_k = exp(-dt/tau_k) and
# b_k = tau_k (1 - a_k)/dt the branch histories obey h_k(n+1) = a_k h_k(n) + b_k (u(n+1) - u(n)),
# and equilibrium at the new time reads
# (K_steel + K_K + (G_inf + sum G_k b_k) B) u(n+1) = f(n+1) - B sum_k G_k (a_k h_k(n) - b_k u(n)).
# Two runs: (a) the start-up from rest with the ramped loads, over 3 s; (b) the integrator
# accuracy: starting from the exact steady state (u and every h_k at their phasor values) and
# integrating two periods with 16, 32, 64 and 128 steps per period.


def ramp(t):
    """Load ramp of the original set-up: 0 before 0.02 s, rising linearly to 1 at 0.12 s."""
    return np.clip(10.0 * (np.asarray(t) - 0.02), 0.0, 1.0)


def time_domain(model, steps_per_period, end_time, start=None):
    """Exponential Prony integration; start=None is rest with ramped loads, else a steady phasor."""
    M, free = model["M"], model["free"]
    B = M["B1"] + M["B2"]
    dt = 1.0 / frequency / steps_per_period
    times = np.arange(0.0, end_time + 0.5 * dt, dt)
    a = np.exp(-dt / tau_branch)
    b_ = tau_branch * (1 - a) / dt
    solve = direct_solver((M["steel"] + M["bulk"] + (G_inf + np.sum(G_branch * b_)) * B).tocsr()[free][:, free])
    load = (lambda t: ramp(t)) if start is None else (lambda t: 1.0)
    if start is None:
        u, h = np.zeros(model["n"]), np.zeros((len(tau_branch), model["n"]))
    else:
        u = np.real(start)
        h = np.real((1j * omega * tau_branch / (1 + 1j * omega * tau_branch))[:, None] * start[None, :])
    history = [u.reshape(-1, 3)]
    for t in times[1:]:
        f = load(t) * (np.cos(omega * t) * model["f_right"] + np.sin(omega * t) * model["f_left"])
        memory = B @ ((G_branch * a) @ h - np.sum(G_branch * b_) * u)
        u_new = np.zeros(model["n"])
        u_new[free] = solve((f - memory)[free])
        h = a[:, None] * h + b_[:, None] * (u_new - u)[None, :]
        u = u_new
        history.append(u.reshape(-1, 3))
    w_left = np.array([U[model["left_nodes"], 2].mean() for U in history])
    w_right = np.array([U[model["right_nodes"], 2].mean() for U in history])
    return times, w_left, w_right


u_qs = harmonic(middle["model"], G_star, omega, inertia=False)
w_left_qs = u_qs.reshape(-1, 3)[middle["model"]["left_nodes"], 2].mean()
w_right_qs = u_qs.reshape(-1, 3)[middle["model"]["right_nodes"], 2].mean()
inertia_effect = np.abs(u_qs - middle["u"]).max() / np.abs(middle["u"]).max()


def deviation(times, w_left, w_right, window):
    """Largest deviation of the left and right pins from the steady motion after `window`, per amplitude."""
    sel = times >= window - 1e-12
    return (np.abs(w_left[sel] - np.real(w_left_qs * np.exp(1j * omega * times[sel]))).max() / abs(w_left_qs),
            np.abs(w_right[sel] - np.real(w_right_qs * np.exp(1j * omega * times[sel]))).max() / abs(w_right_qs))


tic = time.perf_counter()
startup = time_domain(middle["model"], 64, 3.0)
startup_offset = [deviation(*startup, window) for window in (1.0 / frequency, 1.0, 2.0, 3.0 - 1.0 / frequency)]
integrator_error = {steps: max(deviation(*time_domain(middle["model"], steps, 2.0 / frequency, start=u_qs), 0.0))
                    for steps in (16, 32, 64, 128)}
print(f"time domain: start-up {len(startup[0]) - 1} steps and four steady-state runs "
      f"({time.perf_counter() - tic:.0f} s)")

# **Frequency sweep** of the right damper unit on the middle mesh.
sweep_k, sweep_eta = [], []
for f in sweep:
    G = complex_modulus(f)[0]
    w_r = harmonic(check, G, 2 * np.pi * f).reshape(-1, 3)[check["right_nodes"], 2].mean()
    k = F_pin / w_r                       # complex stiffness of the right plate at its pin
    sweep_k.append(k.real)
    sweep_eta.append(k.imag / k.real)
sweep_k, sweep_eta = np.array(sweep_k), np.array(sweep_eta)
print(f"total run time so far {time.perf_counter() - start_time:.0f} s")


# ## Comparison with the reference
#
# The closed-form simple-shear layer is the limit of rigid plates, no free edges and no
# bending; the 3D device must come out somewhat softer. The right plate is guided in $x$ at
# its pin, so its deviation measures the free-edge zones and the steel compliance only. The
# left plate is guided in $y$ only and also carries the small $x$ force, so it tilts out of
# its plane and works its layer harder than simple shear predicts.

def richardson(values, h):
    """Observed order and extrapolated value from three meshes with sizes h (h1 > h2 > h3)."""
    f1, f2, f3 = values
    r21, r32 = h[0] / h[1], h[1] / h[2]
    eps21, eps32 = f2 - f1, f3 - f2
    if eps21 == 0 or eps32 == 0 or np.sign(eps21) != np.sign(eps32):
        return np.nan, f3
    g = lambda p: eps21 / eps32 - (r21**p - 1) / (1 - r32**-p)
    p = brentq(g, 0.05, 12.0) if g(0.05) * g(12.0) < 0 else np.nan
    return p, f3 + eps32 / (r32**p - 1)


h_rel = np.array([1.0 / L for L in levels])
G_m = abs(G_star)
w_simple = F_pin * h_layer / (G_star * A_layer)                # rigid-plate simple shear (complex)
W_simple = np.pi * (F_pin / A_layer) ** 2 * G_star.imag / G_m**2 * V_layer

print("Material (exact Prony moduli at 3 Hz, the input to both solvers)")
print(f"  G' = {G_star.real/1e6:.4f} MPa, G'' = {G_star.imag/1e6:.4f} MPa, delta = {np.degrees(delta_material):.2f} deg")
print()
print("Right damper unit: 3D field against the closed-form simple-shear layer between rigid plates")
w_r = fine["w_right"]
print(f"  pin displacement amplitude  FEM {abs(w_r)/mm:.4f} mm   simple shear {abs(w_simple)/mm:.4f} mm   "
      f"deviation {100*(abs(w_r)/abs(w_simple)-1):+.2f} %")
lag = np.angle(F_pin / w_r)
print(f"  phase lag of w behind F      FEM {np.degrees(lag):.3f} deg  material {np.degrees(delta_material):.3f} deg  "
      f"deviation {np.degrees(lag - delta_material):+.3f} deg")
print(f"  energy dissipated per cycle  FEM {fine['W_2']:.4f} J   simple shear {W_simple:.4f} J   "
      f"deviation {100*(fine['W_2']/W_simple-1):+.2f} %")
print(f"Left damper unit (loaded 90 deg later, plus a small x force):")
print(f"  energy dissipated per cycle  FEM {fine['W_1']:.4f} J   simple shear {W_simple:.4f} J   "
      f"deviation {100*(fine['W_1']/W_simple-1):+.2f} %")
print()
print("Energy balance per cycle (finest mesh): hysteresis-loop work of the pin forces against the")
print("dissipation integrated over both layers")
print(f"  work of the loads {fine['W_loads']:.6f} J, dissipation {fine['W_1'] + fine['W_2']:.6f} J, "
      f"relative difference {abs(fine['W_loads']/(fine['W_1'] + fine['W_2']) - 1):.1e}")
print()
print("Mesh study")
for name, key, unit, scale in (("total dissipation per cycle", "W", "J", 1.0),
                               ("right pin amplitude |w_right|", "w", "mm", 1 / mm),
                               ("left pin amplitude |w_left|", "wl", "mm", 1 / mm)):
    values = [(runs[L]["W_1"] + runs[L]["W_2"]) if key == "W" else
              abs(runs[L]["w_right" if key == "w" else "w_left"]) * scale for L in levels]
    p, extrapolated = richardson(values, h_rel)
    print(f"  {name:31s} " + "  ".join(f"L{L}: {v:.5f}" for L, v in zip(levels, values))
          + f" {unit};  change L{levels[1]}->L{levels[2]} {100*abs(values[2]/values[1]-1):.3f} %;"
          + f"  observed order {p:.2f};  Richardson {extrapolated:.5f} {unit}"
          + f" (finest mesh {100*abs(values[2]/extrapolated-1):.3f} % from it)")
print()
print("Time domain (exponential Prony integrator, middle mesh) against the quasi-static")
print("frequency-domain motion on the same mesh, largest deviation per amplitude")
for steps, error in integrator_error.items():
    print(f"  from the steady state, {steps:2d} steps per period, two periods: {100*error:.4f} %")
errors = np.array(list(integrator_error.values()))
orders = np.log2(errors[:-1] / errors[1:])
print("  observed temporal order " + ", ".join(f"{o:.2f}" for o in orders)
      + " (successive halvings; 2 for slow branches, 1 for the dashpot-like fast ones)")
print("  start-up from rest with the ramped loads (64 steps per period): remaining start-up")
print("  transient (largest deviation from the steady motion after the given time), left / right pin")
for window, (left, right) in zip(("1 period", "1 s", "2 s", "8 periods"), startup_offset):
    print(f"    after {window:9s}: {100*left:6.2f} % / {100*right:6.2f} %")
print()
print(f"Inertia effect at 3 Hz (with vs without the mass matrix): {inertia_effect:.1e} of the amplitude")
print(f"Assembled real stiffness vs OneSimulation SmallStrainSolid (same mesh, G = G'): "
      f"max difference {library_difference:.1e} of the maximum displacement")
print()
print("Frequency sweep of the right unit (middle mesh) against the material loss factor G''/G'")
eta_material = (complex_modulus(np.array(sweep)).imag / complex_modulus(np.array(sweep)).real)
for f, eta_d, eta_m, k in zip(sweep, sweep_eta, eta_material, sweep_k):
    print(f"  f = {f:5.3f} Hz: k' = {k/1e3:8.2f} N/mm, damper loss factor {eta_d:.4f}, material {eta_m:.4f} "
          f"({100*(eta_d/eta_m-1):+.2f} %)")
print(f"total run time {time.perf_counter() - start_time:.0f} s")


# ## Figure
#
# Top left: the device at the instant of maximum right-pin force, coloured by the vertical
# displacement $u_z$ (deformation magnified 3x). Top middle: amplitude of the shear strain
# $|\hat\gamma_{xz}|$ in the vertical section $y=25$ mm through both layers: it is nearly
# uniform in the core, peaks a few millimetres inside the upper and lower free edges and
# falls to zero at the edges themselves, where the shear traction must vanish. Top right:
# the normal (peel) stress $\sigma_{xx}$ across the right layer at the same instant: the
# eccentric pin force bends the joint, so the layer is pulled at one edge and pressed at the
# other, a purely 3D effect absent from the simple-shear model. The map stops 1 mm inside the
# layer edges and the strain colour scale is capped at its 99.5th percentile, because the
# stresses are singular at the bonded corners of the layers.
# Bottom: hysteresis loop of the left pin (time domain from rest, steady frequency-domain loop
# and the rigid-plate simple-shear ellipse); the start-up loop drifts slowly onto the steady
# loop because the long-time modulus $G_\infty$ is about 75 times smaller than $|G^*|$ at 3 Hz,
# so the mean force of the first cycles creeps away only over seconds. Then damper stiffness
# and loss factor against frequency, and the mesh study of the dissipated energy.

space, mesh, groups = model["space"], model["mesh"], model["groups"]
U = u_fine.reshape(-1, 3)
coords = np.asarray(mesh.coordinates)
nodal = np.real(U[: len(coords)])                       # P2 vertex values at the instant t = 0

fig = plt.figure(figsize=(17, 10.5), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.0, 1.0))

ax3d = fig.add_subplot(grid[0, 0], projection="3d")
faces = np.array([f for name in groups for f in groups[name]])
deformed = coords + 3.0 * nodal
tri = deformed[faces] / mm
uz_face = nodal[faces, 2].mean(axis=1) / mm
lim = np.abs(uz_face).max()
norm3d = colors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
light = np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ np.array([0.5, -0.4, 0.75]))
light /= np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) + 1e-30
shade = DIVERGING_CMAP(norm3d(uz_face))
shade[:, :3] *= (0.65 + 0.35 * light)[:, None]
ax3d.add_collection3d(Poly3DCollection(tri, facecolors=shade, edgecolors="none", linewidths=0))
ax3d.set(xlim=(-30, 80), ylim=(-30, 80), zlim=(-100, 150), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm")
ax3d.set_box_aspect((110, 110, 250))
ax3d.view_init(elev=18, azim=-62)
ax3d.set_title("Vertical displacement at peak right-pin force\n(deformation x3)")
fig.colorbar(cm.ScalarMappable(norm=norm3d, cmap=DIVERGING_CMAP), ax=ax3d, shrink=0.6, pad=0.1, label="u_z / mm")

# Smooth strain fields in the layers: the P2 displacement gradient is averaged at the
# nodes over the layer cells (nodal recovery) and interpolated with the P2 basis.
cell_x = coords[np.asarray(mesh.cells)].mean(axis=1)[:, 0]
gradient_sum = np.zeros((space.dof_count, 3, 3), complex)
count = np.zeros(space.dof_count)
for cell in np.flatnonzero(layer_index(cell_x) > 0):
    dofs = space.cell_dofs[cell]
    bary = space.reference_coordinates(cell, np.asarray(space.dof_coordinates)[dofs])
    gradient_sum[dofs] += np.einsum("qnd,nc->qcd", space.gradients(cell, bary), U[dofs])
    count[dofs] += 1
recovered = (gradient_sum / np.maximum(count, 1)[:, None, None]).reshape(space.dof_count, 9)


def layer_strain(points):
    """Recovered complex strain tensor at points inside the layers."""
    g = space.evaluate(recovered, points).reshape(-1, 3, 3)
    return 0.5 * (g + np.swapaxes(g, 1, 2))


ax_s = fig.add_subplot(grid[0, 1])
xs = np.linspace(0.0, 50.0, 251) * mm
zs = np.linspace(-6.0, 56.0, 311) * mm
XX, ZZ = np.meshgrid(xs, zs)
in_layers = (layer_index(XX) > 0) & (ZZ > 0.02 * mm) & (ZZ < pad_h - 0.02 * mm)
eps = layer_strain(np.column_stack((XX[in_layers], np.full(in_layers.sum(), y_c), ZZ[in_layers])))
gamma_xz = np.full(XX.shape, np.nan)
gamma_xz[in_layers] = np.abs(2 * eps[:, 0, 2])
cap = np.ceil(np.nanpercentile(gamma_xz, 99.5) * 10) / 10      # the bonded corners are singular
im = ax_s.contourf(XX / mm, ZZ / mm, gamma_xz, levels=np.linspace(0, cap, 31), cmap=FIELD_CMAP, extend="max")
ax_s.contour(XX / mm, ZZ / mm, gamma_xz, levels=np.linspace(0, cap, 11)[1:-1], colors="k", linewidths=0.35, alpha=0.5)
for name, x0, thickness, footprint in PARTS:
    if not name.startswith("layer"):
        z_low = -6.0 if "down" in footprint else 0.0
        z_high = 56.0 if "up" in footprint else 50.0
        ax_s.add_patch(plt.Rectangle((x0 / mm, z_low), thickness / mm, z_high - z_low, color="#C9CCBF", lw=0))
        ax_s.text((x0 + thickness / 2) / mm, 25.0, name if name != "centre" else "central plate",
                  rotation=90, ha="center", va="center", fontsize=8, color=INK)
ax_s.set_aspect("equal")
ax_s.set(xlim=(0, 50), ylim=(-6, 56), xlabel="x / mm", ylabel="z / mm",
         title="Shear strain amplitude |gamma_xz| in both layers\nsection y = 25 mm (steel in grey)")
fig.colorbar(im, ax=ax_s, shrink=0.8, label="|gamma_xz|", ticks=np.linspace(0, cap, 6))

ax_c = fig.add_subplot(grid[0, 2])
yy, zz = np.meshgrid(np.linspace(1.0, 49.0, 193) * mm, np.linspace(1.0, 49.0, 193) * mm)
mid_x = 2 * t_plate + 1.5 * h_layer
eps = layer_strain(np.column_stack((np.full(yy.size, mid_x), yy.ravel(), zz.ravel())))
volumetric = np.trace(eps, axis1=1, axis2=2)
sigma_xx = np.real(K_layer * volumetric + 2 * G_star * (eps[:, 0, 0] - volumetric / 3)).reshape(yy.shape) / 1e6
lim = np.ceil(np.abs(sigma_xx).max() * 10) / 10
im = ax_c.contourf(yy / mm, zz / mm, sigma_xx, levels=np.linspace(-lim, lim, 31), cmap=DIVERGING_CMAP)
ax_c.contour(yy / mm, zz / mm, sigma_xx, levels=np.linspace(-lim, lim, 11), colors="k", linewidths=0.35, alpha=0.5)
ax_c.set_aspect("equal")
ax_c.set(xlabel="y / mm", ylabel="z / mm",
         title="Peel stress sigma_xx at peak right-pin force\nmid-plane of the right layer (1 mm inside the edges)")
fig.colorbar(im, ax=ax_c, shrink=0.8, label="sigma_xx / MPa", ticks=np.linspace(-lim, lim, 5))

ax_l = fig.add_subplot(grid[1, 0])
times, w_left_t, _ = startup
force_t = ramp(times) * F_pin * np.sin(omega * times)
ax_l.plot(w_left_t / mm, force_t / 1e3, color="#8B9A8D", lw=0.8, label="time domain from rest (64 steps/period)")
phase = np.linspace(0, 2 * np.pi, 400)
w_steady = np.real(fine["w_left"] * np.exp(1j * phase))
ax_l.plot(w_steady / mm, F_pin * np.sin(phase) / 1e3, color=COPPER, lw=2.2, label="frequency domain, steady state")
ax_l.plot(np.real(-1j * w_simple * np.exp(1j * phase)) / mm, F_pin * np.sin(phase) / 1e3, "--", color=INK, lw=1.2,
          label="simple-shear layer, rigid plates")
ax_l.set(xlabel="left pin displacement w / mm", ylabel="left pin force F_z / kN",
         title=f"Hysteresis loop at {frequency:g} Hz (loop area = energy per cycle)")
ax_l.legend(fontsize=8, loc="upper left")
ax_l.grid(alpha=0.3)

ax_f = fig.add_subplot(grid[1, 1])
fine_f = np.geomspace(0.1, 5.5, 200)
Gf = complex_modulus(fine_f)
ax_f.semilogx(fine_f, Gf.real * A_layer / h_layer / 1e6, color=GREEN, lw=1.5, label="k' simple shear  G' A / h")
ax_f.semilogx(sweep, sweep_k / 1e6, "o", color=GREEN, mfc="none", label="k' 3D damper")
ax_f.set(xlabel="frequency / Hz", ylabel="storage stiffness k' / (kN/mm)", title="Damper stiffness and loss factor")
ax_f.grid(alpha=0.3, which="both")
ax_e = ax_f.twinx()
ax_e.semilogx(fine_f, Gf.imag / Gf.real, color=COPPER, lw=1.5, label="material G''/G'")
ax_e.semilogx(sweep, sweep_eta, "s", color=COPPER, mfc="none", label="3D damper loss factor")
ax_e.set_ylabel("loss factor")
handles = ax_f.get_legend_handles_labels()[0] + ax_e.get_legend_handles_labels()[0]
ax_f.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="center left")

ax_m = fig.add_subplot(grid[1, 2])
W_levels = np.array([runs[L]["W_1"] + runs[L]["W_2"] for L in levels])
p_W, W_ext = richardson(W_levels, h_rel)
ax_m.plot(h_rel, W_levels, "o-", color=COPPER, label="3D FEM, P2 tetrahedra")
ax_m.axhline(W_ext, color=INK, lw=1.0, ls=":", label=f"Richardson estimate {W_ext:.3f} J (order {p_W:.1f})")
ax_m.axhline(2 * W_simple, color=GREEN, lw=1.0, ls="--", label=f"simple shear, two layers {2*W_simple:.3f} J")
ax_m.set_xticks(h_rel, [f"L{L}" for L in levels])
ax_m.invert_xaxis()
ax_m.set(xlabel="mesh level (relative cell size 1/L)", ylabel="energy dissipated per cycle / J",
         title="Mesh study of the dissipated energy")
ax_m.legend(fontsize=8)
ax_m.grid(alpha=0.3)
plt.show()


# ## Going further
#
# Raise the excitation frequency towards 30 Hz: the polymer stiffens (the branches with
# $\tau\approx 5$ ms freeze) and the loop narrows in displacement but its loss factor stays
# above one over a wide band, which is why such materials are chosen. Replace the 18
# branches by a pruned set (only branches with $0.3\ \text{Hz}<1/(2\pi\tau_k)<30$ Hz, the
# faster ones lumped into a dashpot, the slower ones into the equilibrium spring) and
# compare the loop with the full series. Doubling the layer thickness halves the stiffness
# and, for the same force, doubles the energy dissipated per cycle.
