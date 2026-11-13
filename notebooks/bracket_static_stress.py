"""OneSimulation example: static stress analysis of a sheet-metal bracket.
Requires the installed, activated OneSimulation runtime.

A U-shaped steel bracket is bolted to a ceiling through four holes in its mounting tabs and
carries a pin load in the lugs at the end of its two arms. The 3D linear-elastic field is
computed with curved quadratic hexahedra (Q2) on a block-structured mesh with O-grids around
every hole and compared with trilinear (Q1) elements, global equilibrium, beam theory in the
arms, Peterson's stress concentration factor of a pin-loaded lug and a mesh study.
"""


# # Static stress analysis of a bracket
#
# A bracket made of 5 mm steel sheet hangs from a ceiling. Its two vertical **arms** are
# joined at the back by a horizontal **web**; two horizontal **mounting tabs** bent outwards
# from the top of the arms carry two bolt holes each. At the front end each arm has a
# downward **lug** with a pin hole. A pin through both lugs carries a load of $F=2$ kN
# straight down. We want the stress field, the deformed shape, the stress concentration at
# the loaded pin holes and the bending stress in the arms.
#
# **Geometry.** The original tutorial bracket is a CAD file; here a representative bracket
# of the same layout is built from blocks (all dimensions in mm; $x$ along the arms from the
# mounting end, $y$ across, $z$ up; the bracket is symmetric about the plane $y=0$):
#
# | part | geometry |
# |---|---|
# | arms | plates $0\le x\le130$, $0\le z\le40$, thickness $t=5$ at $20\le|y|\le25$ |
# | lugs | $100\le x\le130$, $-45\le z\le0$ below each arm (width $W=30$), pin hole $d=12$ at $x=115$, $z=-15$ |
# | web | $0\le x\le40$, $|y|\le20$, $35\le z\le40$ |
# | mounting tabs | $0\le x\le40$, $25\le|y|\le45$, $35\le z\le40$, bolt holes $\varnothing9$ at $x=10$ and $30$, $|y|=35$ |
#
# Material: structural steel, $E=200$ GPa, $\nu=0.3$. By symmetry the half $y\ge0$ is
# modelled. Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | the two bolt-hole surfaces of the tab | fixed, $\mathbf u=0$ (bolted to a rigid ceiling) |
# | pin-hole surface | bearing pressure $p=p_{max}\cos\varphi$ on the lower half ($\varphi$ from the downward direction), resultant $F/2$ per lug, $p_{max}=F/(\pi r_p t)$ |
# | symmetry plane $y=0$ (cut through the web) | $u_y=0$ |
# | all other faces | traction free |
#
# The field is genuinely three-dimensional: in-plane bending and shear of the arms, plate
# bending of the tabs and the web around the bolts, and the pin-loaded lugs.
#
# **Numerical method.**
# - Spatial: continuous Galerkin finite elements for small-strain linear elasticity with
#   **27-node quadratic hexahedra (Q2)**: tensor-product quadratic displacement on cells
#   whose geometry is also quadratic. The mesh is block-structured: plain blocks for the
#   arms, web and lug ends, and **O-grids** of four blocks around every hole, graded radially
#   towards the hole. The mid-edge and mid-face nodes on the hole surfaces lie on the exact
#   circles, so the holes are represented by curved cells. Two cell layers through the
#   5 mm thickness (a third layer is checked separately).
# - Comparison: trilinear displacement (**Q1**) on the same cells.
# - Solver: one linear stationary system, assembled with a 5-point Gauss rule per direction
#   on the curved cells and solved with a sparse direct LU factorization.
#
# **Reference.** There is no closed-form solution for the whole bracket. The notebook uses
# (i) global equilibrium (reactions at the supports against the applied load and moment),
# (ii) Euler-Bernoulli and Jourawski beam theory in the free span of the arms, where each arm
# is a statically determinate cantilever, (iii) Peterson's stress concentration factor for a
# lug with a closely fitting pin as a plausibility check of the peak stress at the pin hole,
# and (iv) a three-level mesh study of the peak von Mises stress at the pin hole and of the
# pin deflection, with observed order and Richardson extrapolation. The stresses at the
# fixed bolt holes and in the sharp re-entrant corners where the plates meet are singular in
# linear elasticity and are not used for grading.

from importlib.metadata import version
from itertools import product
from types import SimpleNamespace
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
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq

from onesim.meshing.generation.contracts.hexahedral_geometry import FACES
from onesim.meshing.generation.contracts.polynomial.hex import HEX27_REFERENCE, QuadraticHexahedralGeometry
from onesim.physics.solid_mechanics import SmallStrainSolid

print("OneSimulation:", version("onesimulation"))


# ## Parameters

mm = 1e-3
t = 5 * mm                                   # sheet thickness
H, w2, L_arm = 40 * mm, 20 * mm, 130 * mm    # arm height, half inner width, arm length
xe0, xe1, z_lug = 100 * mm, 130 * mm, -45 * mm   # lug between x = 100 and 130 mm, bottom at z = -45 mm
xp, zp, rp = 115 * mm, -15 * mm, 6 * mm      # pin-hole centre and radius
a_tab, c_tab, rb = 40 * mm, 20 * mm, 4.5 * mm    # tab length (x), width (y), bolt-hole radius
bolts = ((10 * mm, w2 + t + 10 * mm), (30 * mm, w2 + t + 10 * mm))
E, nu = 200e9, 0.30
F = 2000.0                                   # N, total pin load (F/2 per lug)

lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
p_max = F / (np.pi * rp * t)                 # bearing pressure amplitude, one lug carries F/2
W, d_pin = xe1 - xe0, 2 * rp
x_station = 70 * mm                          # beam-theory station in the free span of the arm
I_arm = t * H**3 / 12
print(f"bearing pressure amplitude {p_max / 1e6:.3f} MPa; lug net section stress "
      f"(F/2)/((W-d) t) = {F / 2 / ((W - d_pin) * t) / 1e6:.3f} MPa")


# ## Geometry and mesh
#
# Every part is a stack of blocks extruded through the sheet thickness. A block is a
# tensor grid of nodes (cell corners plus mid-edge, mid-face and centre nodes) and becomes
# 27-node hexahedra; nodes of neighbouring blocks that coincide are merged.
#
# - Arm: one block on node lines that contain every junction ($x=40$, $x=100$, $z=35$).
# - Lug: an O-grid around the pin hole in the square $100\le x\le130$, $-30\le z\le0$ (four
#   blocks, each between a quarter of the hole and one side of the square; the nodes on the
#   hole are the central projections of the nodes on the square side) and a plain block
#   below it.
# - Tab: two O-grids around the bolt holes, squares of 20 mm.
# - Web: one block.
#
# The node distributions on shared faces coincide, so the mesh is conforming. Mesh level
# $g$ sets the global resolution, level $e$ the resolution of the lug ($6e$ cells along each
# side of the pin-hole square, $3e$ cells radially, graded 4:1 towards the hole), $n_t$ the
# number of cell layers through the thickness.

NODE = np.rint(2 * HEX27_REFERENCE).astype(int)


def with_midpoints(nodes):
    nodes = np.asarray(nodes, float)
    out = np.empty(2 * len(nodes) - 1)
    out[::2], out[1::2] = nodes, 0.5 * (nodes[:-1] + nodes[1:])
    return out


def graded(start, stop, n, ratio=1.0):
    """n cells from start to stop, the last cell `ratio` times the first."""
    sizes = ratio ** (np.arange(n) / max(n - 1, 1))
    nodes = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    nodes[-1] = 1.0
    return start + (stop - start) * nodes


def rectangle(xs, ys):
    X, Y = np.meshgrid(with_midpoints(xs), with_midpoints(ys), indexing="ij")
    return np.stack((X, Y), -1)


def ogrid(centre, radius, half, n_side, radial):
    """Four blocks around a hole of the given radius in the square centre +- half."""
    c = np.asarray(centre, float)
    corners = c + half * np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], float)
    u, v = with_midpoints(np.linspace(0, 1, n_side + 1)), with_midpoints(radial)
    blocks = []
    for k in range(4):
        edge = corners[k] + u[:, None] * (corners[(k + 1) % 4] - corners[k])
        arc = c + radius * (edge - c) / np.linalg.norm(edge - c, axis=1)[:, None]
        blocks.append(arc[:, None, :] + v[None, :, None] * (edge - arc)[:, None, :])
    return blocks


def extrude(sheet, layers, plane):
    """Sheet coordinates -> (x, z) with layers along y ('xz'), or -> (x, y) with layers along z ('xy')."""
    w = with_midpoints(layers)
    out = np.empty(sheet.shape[:2] + (len(w), 3))
    out[..., 0] = sheet[..., 0, None]
    out[..., 2 if plane == "xz" else 1] = sheet[..., 1, None]
    out[..., 1 if plane == "xz" else 2] = w[None, None, :]
    return out


def assemble_blocks(blocks):
    points, cells, offset = [], [], 0
    for grid in blocks:
        e0, e1, e2 = grid[2, 0, 0] - grid[0, 0, 0], grid[0, 2, 0] - grid[0, 0, 0], grid[0, 0, 2] - grid[0, 0, 0]
        if np.dot(np.cross(e0, e1), e2) < 0:
            grid = grid[:, :, ::-1]                  # right-handed cells
        shape = np.array(grid.shape[:3])
        ids = np.arange(np.prod(shape)).reshape(tuple(shape)) + offset
        for index in product(*(range(k) for k in (shape - 1) // 2)):
            cells.append([ids[tuple(2 * np.asarray(index) + m)] for m in NODE])
        points.append(grid.reshape(-1, 3))
        offset += np.prod(shape)
    X = np.concatenate(points)
    _, first, inverse = np.unique(np.round(X / 1e-10).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    return X[first], inverse.ravel()[np.asarray(cells)]


def on_circle(P, centre, radius, axes):
    return np.all(np.abs(np.hypot(P[:, axes[0]] - centre[0], P[:, axes[1]] - centre[1]) - radius) < 1e-9)


def build_bracket(g, e, n_t=2):
    n_o, n_e = 4 * g, 6 * e
    z_layers = np.linspace(H - t, H, n_t + 1)
    y_layers = np.linspace(w2, w2 + t, n_t + 1)
    xa = np.unique(np.r_[np.linspace(0, a_tab, 2 * n_o + 1), graded(a_tab, xe0, 8 * g), np.linspace(xe0, xe1, n_e + 1)])
    za = np.unique(np.r_[graded(0, H - t, 5 * g), z_layers])
    blocks = [extrude(rectangle(xa, za), y_layers, "xz")]
    blocks += [extrude(s, y_layers, "xz") for s in ogrid((xp, zp), rp, W / 2, n_e, graded(0, 1, 3 * e, 4.0))]
    blocks.append(extrude(rectangle(np.linspace(xe0, xe1, n_e + 1), graded(z_lug, zp - W / 2, 2 * e)), y_layers, "xz"))
    for centre in bolts:
        blocks += [extrude(s, z_layers, "xy") for s in ogrid(centre, rb, 10 * mm, n_o, graded(0, 1, 2 * g + 1, 3.0))]
    blocks.append(extrude(rectangle(np.linspace(0, a_tab, 2 * n_o + 1), graded(0, w2, 4 * g)), z_layers, "xy"))
    X, C = assemble_blocks(blocks)
    owners = {}
    for cell, vertices in enumerate(C):
        for side, face in enumerate(FACES):
            corners = tuple(int(v) for v in vertices[list(face)])
            owners.setdefault(tuple(sorted(corners)), []).append((corners, cell, side))
    exterior = [found[0] for found in owners.values() if len(found) == 1]
    groups = {name: [] for name in ("bolts", "pin", "symmetry", "free")}
    for corners, cell, side in exterior:
        P = X[list(corners)]
        if np.all(np.abs(P[:, 1]) < 1e-12):
            name = "symmetry"
        elif on_circle(P, (xp, zp), rp, (0, 2)):
            name = "pin"
        elif any(on_circle(P, centre, rb, (0, 1)) for centre in bolts):
            name = "bolts"
        else:
            name = "free"
        groups[name].append((corners, cell, side))
    volume = QuadraticHexahedralGeometry(X, C).volumes.sum()
    exact_volume = t * (L_arm * H + W * (-z_lug) - np.pi * rp**2 + a_tab * c_tab - 2 * np.pi * rb**2 + a_tab * w2)
    return dict(g=g, e=e, n_t=n_t, X=X, C=C, groups=groups, volume_error=volume / exact_volume - 1)


# ## Model
#
# `SmallStrainSolid` with volume kinematics. The bearing pressure is a position-dependent
# traction on the pin-hole faces: with the solid's outward normal $\mathbf n$ (pointing into
# the hole) it is $-p\,\mathbf n$ with $p=p_{max}\max(0,n_z)$.

solid = SmallStrainSolid(youngs_modulus=E, poisson_ratio=nu, kinematics="volume")


def bearing(x, y, z):
    nx, nz = (xp - np.asarray(x)) / rp, (zp - np.asarray(z)) / rp
    p = p_max * np.maximum(nz, 0.0)
    value = np.zeros(np.shape(x) + (3,))
    value[..., 0], value[..., 2] = -p * nx, -p * nz
    return value


def solve(data, order=2):
    start = time.perf_counter()
    groups = {name: [corners for corners, _, _ in faces] for name, faces in data["groups"].items()}
    mesh = SimpleNamespace(geometry=SimpleNamespace(dimension=3), coordinates=data["X"], cells=data["C"],
                           facets=lambda name: groups[name])
    result = solid.solve(
        mesh=mesh, times=(0.0,), order=order,
        displacements=(("bolts", "x", 0.0), ("bolts", "y", 0.0), ("bolts", "z", 0.0),   # bolted tab
                       ("symmetry", "y", 0.0)),                                         # symmetry plane
        tractions=(("pin", bearing),                                                    # pin load
                   ("free", (0.0, 0.0, 0.0))))                                          # free faces
    u, reactions, space = result.displacement[-1], result.reactions[-1], result.space
    nodes = space.dof_coordinates
    return dict(data=data, order=order, space=space, u=u, dofs=u.size, cells=len(data["C"]),
                reaction=reactions.sum(axis=0), moment=np.cross(nodes, reactions).sum(axis=0),
                seconds=time.perf_counter() - start)


def face_samples(run, faces, k=5):
    """Positions, displacements and stresses on cell faces, evaluated inside the owning cells."""
    space, u = run["space"], run["u"]
    g1 = np.linspace(0, 1, k)
    A, B = (m.ravel() for m in np.meshgrid(g1, g1, indexing="ij"))
    points, displacements, stresses = [], [], []
    for _, cell, side in faces:
        ref = np.empty((k * k, 3))
        axis, free = side // 2, [d for d in range(3) if d != side // 2]
        ref[:, axis], ref[:, free[0]], ref[:, free[1]] = side % 2, A, B
        local = u[space.cell_dofs[cell]]
        gradient = np.einsum("qna,nb->qba", space.gradients(cell, ref), local)
        strain = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
        stresses.append(lam * np.trace(strain, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) + 2 * mu * strain)
        points.append(space.geometry.map(cell, ref))
        displacements.append(space.basis(ref) @ local)
    return np.concatenate(points), np.concatenate(displacements), np.concatenate(stresses)


def stress_at(run, points):
    gradient = run["space"].gradient(run["u"], points)
    strain = 0.5 * (gradient + np.swapaxes(gradient, -1, -2))
    return lam * np.trace(strain, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) + 2 * mu * strain


def von_mises(s):
    dev = s - np.trace(s, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))


z_line = np.linspace(1e-6, H - 1e-6, 21)
arm_line = np.column_stack((np.full_like(z_line, x_station), np.full_like(z_line, w2 + t / 2), z_line))
centre_line = np.array([[x_station + dx, w2 + t / 2, H / 2] for dx in (-10 * mm, 0.0, 10 * mm)])
pin_bottom = np.array([[xp, w2 + t / 2, zp - rp]])


def evaluate(run):
    P, _, s = face_samples(run, run["data"]["groups"]["pin"])
    run["pin_vm"] = von_mises(s)
    run["pin_peak"] = run["pin_vm"].max()
    angle = np.arctan2(P[:, 2] - zp, P[:, 0] - xp)
    tangent = np.column_stack((-np.sin(angle), np.zeros_like(angle), np.cos(angle)))
    hoop = np.einsum("qi,qij,qj->q", tangent, s, tangent)
    mid = np.abs(P[:, 1] - (w2 + t / 2)) < 1e-9
    order = np.argsort(angle[mid])
    run["hoop_angle"], run["hoop"] = angle[mid][order], hoop[mid][order]
    s_arm = stress_at(run, arm_line)
    run["sxx_arm"], run["sxz_arm"] = s_arm[:, 0, 0], s_arm[:, 0, 2]
    run["w_line"] = run["space"].evaluate(run["u"], centre_line)[:, 2]
    run["w_pin"] = run["space"].evaluate(run["u"], pin_bottom)[0, 2]
    return run


# ## Solve
#
# Three meshes for the study: $(g,e)=(1,1)$, $(2,2)$, $(2,3)$ with two layers through the
# thickness; the middle mesh again with three layers (thickness check) and with trilinear
# Q1 elements (element-order comparison).

cases = {"A": (1, 1, 2), "B": (2, 2, 2), "C": (2, 3, 2), "B, 3 layers": (2, 2, 3)}
runs = {}
for name, (g, e, n_t) in cases.items():
    data = build_bracket(g, e, n_t)
    run = evaluate(solve(data))
    if name != "C":
        run.pop("space")                     # keep only the evaluated quantities
    runs[name] = run
    print(f"mesh {name:12s}: {run['cells']:5d} hexahedra, {run['dofs']:6d} unknowns, geometry volume error "
          f"{data['volume_error']:+.1e}; peak von Mises at the pin hole {run['pin_peak'] / 1e6:.3f} MPa, "
          f"pin deflection {run['w_pin'] / mm:.5f} mm ({run['seconds']:.0f} s)")
q1 = evaluate(solve(build_bracket(2, 2, 2), order=1))
q1.pop("space")
print(f"mesh B with Q1 elements: {q1['dofs']} unknowns, peak von Mises at the pin hole {q1['pin_peak'] / 1e6:.3f} MPa, "
      f"pin deflection {q1['w_pin'] / mm:.5f} mm")


# ## Comparison with the reference

fine = runs["C"]
print("Global equilibrium (sum of all support reactions of the half model, finest mesh)")
print(f"   expected: force (0, 0, {F / 2:.1f}) N, moment ({F / 2 * (w2 + t / 2):.4f}, {-F / 2 * xp:.4f}, 0) N m")
for name, run in (("Q2", fine), ("Q1 (mesh B)", q1)):
    R, M = run["reaction"], run["moment"]
    print(f"   {name:12s}: force ({R[0]:+.1e}, {R[1]:+.1e}, {R[2]:.4f}) N, moment ({M[0]:.4f}, {M[1]:.4f}, {M[2]:+.1e}) N m; "
          f"deviation {R[2] / (F / 2) - 1:+.1e} (force), {M[1] / (-F / 2 * xp) - 1:+.1e} (moment about y)")

M_station = F / 2 * (xp - x_station)
V_station = F / 2
print(f"Arm at x = {x_station / mm:.0f} mm: Euler-Bernoulli bending M = (F/2)(x_pin - x) = {M_station:.3f} N m, "
      f"Jourawski shear V = F/2")
print(f"   outer-fibre stress M c / I = {M_station * H / 2 / I_arm / 1e6:.4f} MPa, curvature M/(EI) = "
      f"{M_station / (E * I_arm):.6f} 1/m, peak shear 3V/(2A) = {1.5 * V_station / (H * t) / 1e6:.4f} MPa")
for name, run in (("Q2", fine), ("Q1 (mesh B)", q1)):
    slope = np.polyfit(z_line - H / 2, run["sxx_arm"], 1)[0]
    w = run["w_line"]
    curvature = (w[0] - 2 * w[1] + w[2]) / (10 * mm) ** 2
    tau_mid = float(np.interp(H / 2, z_line, -run["sxz_arm"]))
    print(f"   {name:12s}: bending stress slope x c = {slope * H / 2 / 1e6:.4f} MPa "
          f"({100 * (slope / (M_station / I_arm) - 1):+.3f} %), curvature from the deflection {-curvature:.6f} 1/m "
          f"({100 * (-curvature / (M_station / (E * I_arm)) - 1):+.3f} %), shear at mid-height {tau_mid / 1e6:.4f} MPa "
          f"({100 * (tau_mid / (1.5 * V_station / (H * t)) - 1):+.2f} %)")

ratio = d_pin / W
Kt_peterson = 12.882 - 52.714 * ratio + 89.762 * ratio**2 - 51.667 * ratio**3
sigma_net = F / 2 / ((W - d_pin) * t)
print(f"Pin-loaded lug (d/W = {ratio:.2f}, end distance h/W = {(zp - z_lug) / W:.2f}): Peterson/Frocht-Hill "
      f"K_t = {Kt_peterson:.3f}, sigma_max = K_t (F/2)/((W-d)t) = {Kt_peterson * sigma_net / 1e6:.2f} MPa")
for name, run in (("Q2", fine), ("Q1 (mesh B)", q1)):
    k = int(np.argmax(run["hoop"]))
    print(f"   {name:12s}: largest hoop stress at mid-thickness {run['hoop'][k] / 1e6:.2f} MPa at "
          f"{np.degrees(run['hoop_angle'][k]):+.0f} deg from the horizontal, K_t = {run['hoop'][k] / sigma_net:.3f} "
          f"({100 * (run['hoop'][k] / (Kt_peterson * sigma_net) - 1):+.1f} % against the chart value)")

print("Mesh study of the peak von Mises stress at the pin hole (lug cell size h ~ 1/e)")
sizes = [1.0, 1 / 2, 1 / 3]
f1, f2, f3 = (runs[k]["pin_peak"] for k in ("A", "B", "C"))
p_obs = brentq(lambda q: (sizes[0]**q - sizes[1]**q) / (sizes[1]**q - sizes[2]**q) - (f1 - f2) / (f2 - f3), 0.3, 12.0)
peak_extrapolated = f3 + (f3 - f2) / ((sizes[1] / sizes[2]) ** p_obs - 1)
print(f"   meshes A, B, C: {f1 / 1e6:.4f}  {f2 / 1e6:.4f}  {f3 / 1e6:.4f} MPa; observed order {p_obs:.2f}, Richardson "
      f"{peak_extrapolated / 1e6:.4f} MPa; change between the last two meshes {100 * abs(f3 / f2 - 1):.3f} %; "
      f"three layers instead of two (mesh B): {100 * (runs['B, 3 layers']['pin_peak'] / f2 - 1):+.3f} %")
w1, w2_, w3 = (runs[k]["w_pin"] for k in ("A", "B", "C"))
print("Pin deflection (vertical displacement at the bottom of the pin hole)")
print(f"   meshes A, B, C: {w1 / mm:.5f}  {w2_ / mm:.5f}  {w3 / mm:.5f} mm; change A -> B {100 * (w2_ / w1 - 1):+.3f} %, "
      f"B -> C (lug refined only) {100 * (w3 / w2_ - 1):+.4f} %, three layers instead of two (mesh B) "
      f"{100 * (runs['B, 3 layers']['w_pin'] / w2_ - 1):+.3f} %")


# ## Figure
#
# Top: von Mises stress on the deformed bracket (both halves, displacements magnified 40
# times; finest mesh) and on the outer face of the lug. Bottom: bending and shear stress
# across the arm against beam theory (with Q1 for comparison), hoop stress around the pin
# hole on the three Q2 meshes against Peterson's peak value, and the mesh study. The Q1
# surface stresses at the loaded hole oscillate from point to point and are left out of the
# hoop-stress plot.

space = fine["space"]
X, C = fine["data"]["X"], fine["data"]["C"]
all_faces = [f for faces in fine["data"]["groups"].values() for f in faces]
k_sub = 3
P, U, S_ = face_samples(fine, all_faces, k_sub)
vm = von_mises(S_)
quads = [(i * k_sub + j, (i + 1) * k_sub + j, (i + 1) * k_sub + j + 1, i * k_sub + j + 1)
         for i in range(k_sub - 1) for j in range(k_sub - 1)]
magnify = 40.0
polys, shades = [], []
for f in range(len(all_faces)):
    base = f * k_sub * k_sub
    for qd in quads:
        idx = [base + i for i in qd]
        for mirror in (1.0, -1.0):
            Q = (P[idx] + magnify * U[idx]) / mm
            Q[:, 1] *= mirror
            polys.append(Q)
            shades.append(vm[idx].mean())
vmax = 60.0
norm = colors.Normalize(vmin=0.0, vmax=vmax)

fig = plt.figure(figsize=(20, 11.5), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.25, 1.0, 1.0))
ax = fig.add_subplot(grid[0, :2], projection="3d")
shade = FIELD_CMAP(norm(1e-6 * np.asarray(shades)))
ax.add_collection3d(Poly3DCollection(polys, facecolors=shade, edgecolors=shade, linewidths=0.1))
ax.set(xlim=(-5, 135), ylim=(-50, 50), zlim=(-50, 45), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm",
       title=f"von Mises stress on the deformed bracket (Q2, {fine['dofs']} unknowns in the half model, "
             f"displacement x{magnify:.0f})")
ax.set_box_aspect((140, 100, 95), zoom=1.3)
ax.view_init(elev=26, azim=-128)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax, shrink=0.6, extend="max",
             label="von Mises stress / MPa (colour scale capped)")

ax = fig.add_subplot(grid[0, 2])
outer = [f for f in fine["data"]["groups"]["free"]
         if np.allclose(X[list(f[0]), 1], w2 + t, atol=1e-12) and X[list(f[0]), 2].max() <= 1e-12]
Po, _, So = face_samples(fine, outer, 5)
tri_cells = []
for f in range(len(outer)):
    base = f * 25
    for i in range(4):
        for j in range(4):
            a0, a1, a2, a3 = base + i * 5 + j, base + (i + 1) * 5 + j, base + (i + 1) * 5 + j + 1, base + i * 5 + j + 1
            tri_cells += [(a0, a1, a2), (a0, a2, a3)]
tri = Triangulation(Po[:, 0] / mm, Po[:, 2] / mm, np.array(tri_cells))
vm_outer = von_mises(So) / 1e6
cs = ax.tricontourf(tri, vm_outer, levels=np.linspace(0, 32, 33), cmap=FIELD_CMAP, extend="max")
ax.set(aspect="equal", xlabel="x / mm", ylabel="z / mm", title="Outer face of the lug: von Mises stress")
fig.colorbar(cs, ax=ax, shrink=0.85, label="von Mises stress / MPa")

ax = fig.add_subplot(grid[1, 0])
zz = np.linspace(0, H, 50)
ax.plot(M_station * (zz - H / 2) / I_arm / 1e6, zz / mm, color=INK, lw=1.3, label="beam theory  M (z - H/2) / I")
ax.plot(1.5 * V_station / (H * t) * (1 - (2 * (zz - H / 2) / H) ** 2) / 1e6, zz / mm, color=INK, lw=1.0, ls="--",
        label="Jourawski shear  (3V/2A)(1 - (2z'/H)$^2$)")
ax.plot(fine["sxx_arm"] / 1e6, z_line / mm, "s", ms=4, color=GREEN, label="Q2: bending stress sigma_xx")
ax.plot(-fine["sxz_arm"] / 1e6, z_line / mm, "o", ms=4, color=COPPER, label="Q2: shear stress -sigma_xz")
ax.plot(q1["sxx_arm"] / 1e6, z_line / mm, "x", ms=4, color="#8B9A8D", label="Q1 (mesh B): sigma_xx")
ax.set(xlabel="stress / MPa", ylabel="z / mm", title=f"Arm section x = {x_station / mm:.0f} mm, mid-thickness")
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 1])
for name, color, style in (("A", "#A8BC91", "-"), ("B", "#5F9175", "-"), ("C", GREEN, "-")):
    ax.plot(np.degrees(runs[name]["hoop_angle"]), runs[name]["hoop"] / 1e6, style, color=color, lw=1.3,
            label=f"Q2, mesh {name} ({runs[name]['cells']} cells)")
ax.axhline(Kt_peterson * sigma_net / 1e6, color=INK, lw=1.0, ls="--",
           label=f"Peterson K$_t$ sigma$_{{net}}$ = {Kt_peterson * sigma_net / 1e6:.1f} MPa")
ax.set(xlabel="angle on the pin hole from the +x direction / deg", ylabel="hoop stress / MPa",
       title="Hoop stress around the pin hole (mid-thickness)", xlim=(-180, 180), ylim=(-15, 40))
ax.legend(fontsize=8, loc="upper left")

ax = fig.add_subplot(grid[1, 2])
cells = [runs[k]["cells"] for k in ("A", "B", "C")]
ax.plot(cells, [runs[k]["pin_peak"] / 1e6 for k in ("A", "B", "C")], "s-", color=GREEN, label="peak von Mises at the pin hole")
ax.set(xlabel="hexahedra in the half model", ylabel="peak von Mises stress / MPa", title="Mesh study")
ax2 = ax.twinx()
ax2.plot(cells, [-runs[k]["w_pin"] / mm for k in ("A", "B", "C")], "o--", color=COPPER, label="pin deflection")
ax2.set_ylabel("pin deflection / mm")
ax.legend(handles=ax.get_lines() + ax2.get_lines(), fontsize=8, loc="center right")
plt.show()


# ## Going further
#
# Replace the rigid bolt holes by bolt heads and a frictional contact with the ceiling, add
# fillets at the bends to remove the corner singularities and follow the converging
# stresses there, apply the pin load with a contact model instead of the cosine bearing
# pressure, or compute the eigenfrequencies of the bracket on the same Q2 mesh.
