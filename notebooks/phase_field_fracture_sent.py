"""OneSimulation example: phase-field fracture of the single-edge-notched tension specimen.
Requires the installed, activated OneSimulation runtime.

The classic benchmark of Miehe, Hofacker and Welschinger (2010): a 1 mm square plate with a
horizontal notch to its centre is pulled apart under displacement control. OneSimulation's
AT2 phase-field model with an irreversible damage field and a tension/compression energy
split is solved by alternate minimisation on a graded triangular mesh. The
force-displacement curve is compared with the published curve, the dissipated energy with
Griffith's G_c times the crack length, and the results are checked for their sensitivity to
the regularisation length and the element size.
"""


# # Phase-field fracture: single-edge-notched tension test
#
# A phase-field model replaces a sharp crack by a damage field $d(\mathbf x)\in[0,1]$
# ($d=1$: broken) smeared over a width set by the regularisation length $\ell$. The
# displacement $\mathbf u$ and the damage minimise the total energy (AT2 model)
#
# $$\Pi(\mathbf u,d)=\int_\Omega\Big[g(d)\,\psi^+(\boldsymbol\varepsilon)+\psi^-(\boldsymbol\varepsilon)\Big]\,\mathrm dV
#   +\frac{G_c}{2\ell}\int_\Omega\big(d^2+\ell^2|\nabla d|^2\big)\,\mathrm dV
#   -W_\mathrm{ext},\qquad g(d)=(1-d)^2(1-k)+k ,$$
#
# subject to irreversibility, $\dot d\ge0$. The second integral converges to the Griffith
# surface energy $G_c\,A_\mathrm{crack}$ as $\ell\to0$. The strain energy is split into a
# part $\psi^+$ that drives the crack (deviatoric energy plus the volumetric energy in
# expansion) and a part $\psi^-$ that does not (volumetric energy in compression): the
# volumetric-deviatoric split of Amor, Marigo and Maurini (J. Mech. Phys. Solids 57,
# 2009, 1209-1229).
#
# **Setup** (C. Miehe, M. Hofacker, F. Welschinger, *A phase field model for
# rate-independent crack propagation: robust algorithmic implementation based on
# operator splits*, Comput. Methods Appl. Mech. Eng. 199 (2010) 2765-2778): a
# 1 mm x 1 mm plate with a straight horizontal notch (a slit) from the left edge to the
# centre at mid-height; $E=210$ GPa, $\nu=0.3$, $G_c=2.7$ N/mm, plane strain; the
# bottom edge is clamped, the top edge is moved upwards by $u$ with its horizontal
# motion suppressed, the left and right edges and both notch faces are traction-free.
# Forces are per 1 mm thickness.
#
# **Dimension.** The specimen is a plane-strain 2D problem; the displacement and the
# damage vary over the whole $(x,y)$ section, and the crack path is a result, not an
# assumption (no symmetry plane is imposed).
#
# **Method.** Continuous Galerkin FEM with linear (P1) Lagrange triangles for both fields,
# on a tensor-product mesh graded towards the crack path with element size
# $h\le\ell/2$ in the crack band, the notch modelled geometrically as a slit (duplicated
# nodes). Quasi-static displacement stepping; each load step is solved by the staggered
# scheme (alternate minimisation): a damped Newton solve of the equilibrium at fixed
# damage (sparse direct LU, line search for the split), then the damage at fixed
# displacement as a bound-constrained quadratic problem (projected Newton / active set
# with the previous damage as lower bound, which enforces irreversibility exactly, in
# place of a history-field approximation), repeated with Aitken relaxation until the
# damage changes by less than $10^{-4}$. Residual stiffness $k=10^{-7}$.
#
# **Reference.** (1) The published force-displacement curve of Miehe et al. (2010):
# peak force about 0.70 kN at $u\approx5.5\times10^{-3}$ mm and complete separation at
# about $5.8\times10^{-3}$ mm (points digitised from their figure), and the straight
# crack path across the ligament. (2) The Griffith energy balance: the energy dissipated
# in forming the crack must equal $G_c$ times the new crack length (the 0.5 mm ligament)
# times the thickness, up to the discrete overestimate $h/2\ell$ of the regularised
# functional, which vanishes as $h/\ell\to0$. (3) Sensitivity of the peak force to
# $\ell$ and to $h$.

from importlib.metadata import version
import time
from dataclasses import dataclass

import numpy as np
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
from matplotlib.tri import Triangulation

from onesim.cad.geometry.primitives import Geometry
from onesim.meshing import UnstructuredSimplexMesh
from onesim.physics.solid_mechanics import PhaseFieldFracture

print("OneSimulation:", version("onesimulation"))


# ## Parameters

mm = 1e-3
side = 1.0 * mm                 # plate edge
notch_tip = 0.5 * mm            # notch from x = 0 to the centre, at y = 0.5 mm
thickness = 1.0 * mm            # forces per mm thickness
E, nu, G_c = 210e9, 0.3, 2700.0   # Pa, -, J/m^2 (= 2.7 N/mm)

# Published curve of Miehe et al. (2010), digitised: (u in mm, F in kN).
miehe_curve = np.array([
    (0.0, 0.0), (0.00055, 0.0768), (0.00128, 0.1774), (0.00197, 0.2716), (0.00290, 0.3955),
    (0.00369, 0.4961), (0.00421, 0.5619), (0.00471, 0.6200), (0.00501, 0.6535), (0.00524, 0.6780),
    (0.00543, 0.6960), (0.00555, 0.7012), (0.00557, 0.6584), (0.00562, 0.5613), (0.00568, 0.4629),
    (0.00575, 0.3192), (0.00580, 0.1833), (0.00583, 0.0214), (0.00583, 0.0007)])
F_peak_ref, u_peak_ref = 0.70, 0.0055e-3     # kN, m

# Load steps: coarse in the nearly linear range, fine around the peak and the drop.
load_steps = np.r_[0.0, np.arange(0.5e-3, 4.01e-3, 0.5e-3), np.arange(4.25e-3, 5.0e-3, 0.25e-3),
                   np.arange(5.0e-3, 6.31e-3, 0.05e-3)] * mm

# Runs: (regularisation length, element size in the crack band)
runs_spec = {
    "l = 0.0075 mm, h = l/2": (0.0075 * mm, 0.00375 * mm),
    "l = 0.015 mm, h = l/2": (0.015 * mm, 0.0075 * mm),
    "l = 0.015 mm, h = l/4": (0.015 * mm, 0.00375 * mm),
}
main_run = "l = 0.015 mm, h = l/4"


# ## Geometry and mesh
#
# The mesh is a tensor-product grid split into triangles with alternating diagonals.
# In $x$ the element size is $h$ from 6$\ell$ left of the notch tip to the right edge and
# grows geometrically (ratio 1.15, at most 0.05 mm) towards the left edge; in $y$ it is
# $h$ in a band of $\pm5\ell$ around the crack plane and grows in the same way towards
# the top and bottom. The nodes of the notch faces are duplicated, so the notch is a
# genuine slit whose faces separate. The boundary groups are `bottom`, `top`, `left`,
# `right` and `notch` (both faces).

def graded(start, end, h_fine, h_coarse=0.05 * mm, growth=1.15):
    """Nodes from start (size h_fine) to end, growing geometrically up to h_coarse."""
    nodes, step, s, sign = [start], h_fine, start, np.sign(end - start)
    while abs(end - s) > 1e-15:
        s = s + sign * step
        if (end - s) * sign <= 0.5 * step:
            s = end
        nodes.append(s)
        step = min(step * growth, h_coarse)
    return np.array(nodes)


@dataclass(frozen=True, eq=False, kw_only=True)
class NotchedPlate(Geometry):
    """Square plate with an edge slit; five boundary groups."""

    dimension = 2

    def boundary_groups(self):
        names = ("bottom", "top", "left", "right", "notch")
        return {**{name: (name,) for name in names}, "all": names}


def build_mesh(ell, h):
    n_band = int(np.ceil(5 * ell / h))
    band = 0.5 * side + h * np.arange(-n_band, n_band + 1)
    y_nodes = np.r_[graded(band[0], 0.0, h)[::-1], band[1:-1], graded(band[-1], side, h)]
    n_fine = int(np.ceil((side - (notch_tip - 6 * ell)) / h))
    fine = side - h * np.arange(n_fine, -1, -1)
    x_nodes = np.r_[graded(fine[0], 0.0, h)[::-1], fine[1:]]
    x_nodes[np.argmin(abs(x_nodes - notch_tip))] = notch_tip
    j_crack = int(np.argmin(abs(y_nodes - 0.5 * side)))
    y_nodes[j_crack] = 0.5 * side
    nx, ny = len(x_nodes), len(y_nodes)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    coordinates = [np.c_[X.ravel(), Y.ravel()]]
    node = np.arange(nx * ny).reshape(nx, ny)
    upper_face = {}                          # duplicated nodes of the upper notch face
    for i in np.flatnonzero(x_nodes < notch_tip - 1e-12):
        upper_face[i] = nx * ny + len(upper_face)
        coordinates.append([[x_nodes[i], 0.5 * side]])
    coordinates = np.vstack(coordinates)
    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            q = [node[i, j], node[i + 1, j], node[i + 1, j + 1], node[i, j + 1]]
            if j == j_crack:                 # cells just above the notch use the upper face
                q[0], q[1] = upper_face.get(i, q[0]), upper_face.get(i + 1, q[1])
            if (i + j) % 2 == 0:
                cells += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            else:
                cells += [(q[0], q[1], q[3]), (q[1], q[2], q[3])]
    cells = np.array(cells)
    edges = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    geometry = NotchedPlate()
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    for edge in unique[counts == 1]:
        p = coordinates[edge]
        if np.all(np.abs(p[:, 1]) < 1e-12):
            groups["bottom"].append(tuple(edge))
        elif np.all(np.abs(p[:, 1] - side) < 1e-12):
            groups["top"].append(tuple(edge))
        elif np.all(np.abs(p[:, 0]) < 1e-12):
            groups["left"].append(tuple(edge))
        elif np.all(np.abs(p[:, 0] - side) < 1e-12):
            groups["right"].append(tuple(edge))
        else:
            groups["notch"].append(tuple(edge))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells,
                                   boundary_facets=groups)
    return geometry, mesh, cells


# ## Model
#
# Boundary conditions on every boundary:
#
# | boundary | displacement | damage |
# |---|---|---|
# | bottom $y=0$ | $u_x=u_y=0$ | zero flux |
# | top $y=1$ mm | $u_x=0$, $u_y=u(t)$ | zero flux |
# | left, right | traction-free | zero flux |
# | notch faces | traction-free | zero flux |
#
# The load parameter $t$ of the solver is the prescribed top displacement itself.

def solve(ell, h):
    geometry, mesh, cells = build_mesh(ell, h)
    model = PhaseFieldFracture(youngs_modulus=E, poisson_ratio=nu, fracture_energy=G_c,
                               length_scale=ell, residual_stiffness=1e-7, thickness=thickness,
                               plane_stress=False)
    b = geometry.boundaries
    start = time.time()
    result = model.solve(
        mesh=mesh, times=load_steps, initial_damage=0.0,
        displacements=((b.bottom, "x", 0.0), (b.bottom, "y", 0.0),
                       (b.top, "x", 0.0), (b.top, "y", lambda t: t)),
        tractions=((b.left, (0.0, 0.0)), (b.right, (0.0, 0.0)), (b.notch, (0.0, 0.0))),
        tolerance=1e-4, max_iterations=5000)
    x, y = result.coordinates.T
    force = result.reactions[:, np.abs(y - side) < 1e-12, 1].sum(axis=1)
    return dict(result=result, cells=cells, force=force, time=time.time() - start, ell=ell, h=h,
                nodes=len(x), triangles=len(cells))


# ## Solve

runs = {}
for label, (ell, h) in runs_spec.items():
    runs[label] = solve(ell, h)
    run = runs[label]
    print(f"{label}: {run['triangles']} triangles, {run['nodes']} nodes, {len(load_steps)} load steps, "
          f"{int(run['result'].staggered_iterations.sum())} staggered iterations, {run['time']:.0f} s")
main = runs[main_run]
res = main["result"]


# ## Comparison with the reference

def peak(run):
    i = int(np.argmax(run["force"]))
    return run["force"][i] / 1e3, load_steps[i]


def separation(run, fraction=0.02):
    """First displacement after the peak at which the force has fallen below 2 % of the peak."""
    i = int(np.argmax(run["force"]))
    below = np.flatnonzero(run["force"][i:] < fraction * run["force"][i])
    return load_steps[i + below[0]] if len(below) else np.nan


print("Force-displacement (per mm thickness)")
print(f"  {'run':24s} {'peak force':>12s} {'u at peak':>12s} {'separation':>12s}")
print(f"  {'Miehe et al. 2010':24s} {F_peak_ref:9.3f} kN {u_peak_ref / mm * 1e3:9.3f} um {0.00583e3:9.3f} um")
for label, run in runs.items():
    F_max, u_max = peak(run)
    print(f"  {label:24s} {F_max:9.3f} kN {u_max / mm * 1e3:9.3f} um {separation(run) / mm * 1e3:9.3f} um"
          f"   peak {100 * (F_max / F_peak_ref - 1):+.1f} %, u_peak {100 * (u_max / u_peak_ref - 1):+.1f} %")
linear = load_steps <= 3e-3 * mm
slope = np.polyfit(load_steps[linear], main["force"][linear], 1)[0]
ref_linear = miehe_curve[(miehe_curve[:, 0] > 0) & (miehe_curve[:, 0] <= 0.003)]
slope_ref = np.polyfit(ref_linear[:, 0], ref_linear[:, 1], 1)[0]
print(f"  initial stiffness: {slope / 1e6:.1f} kN/mm vs published {slope_ref:.1f} kN/mm "
      f"({100 * (slope / 1e6 / slope_ref - 1):+.1f} %)")
pre_peak = miehe_curve[miehe_curve[:, 0] <= 0.005]
F_interp = np.interp(pre_peak[:, 0], load_steps / mm, main["force"] / 1e3)
print(f"  pre-peak curve (u <= 5 um), max deviation from the published points: "
      f"{np.max(np.abs(F_interp - pre_peak[:, 1])):.3f} kN")

# **Sensitivity to h and l.** Halving h at l = 0.015 mm lowers the peak by the amount
# printed below. The regularised crack resistance on the mesh is $G_c(1+h/2\ell)$
# (see the Griffith balance), and a crack-driven peak scales with its square root,
# which explains most of the change. Extrapolating linearly in $h$ (first order, the
# order of that term) gives an estimate of the mesh-independent peak for this $\ell$.
F_h2, F_h4 = peak(runs["l = 0.015 mm, h = l/2"])[0], peak(runs["l = 0.015 mm, h = l/4"])[0]
print(f"  l = 0.015 mm: peak h = l/2 {F_h2:.3f} kN, h = l/4 {F_h4:.3f} kN, change {100 * (F_h4 / F_h2 - 1):+.1f} %;"
      f" ratio {F_h2 / F_h4:.3f} vs sqrt((1 + 1/4)/(1 + 1/8)) = {np.sqrt(1.25 / 1.125):.3f};"
      f" first-order extrapolation h -> 0: {2 * F_h4 - F_h2:.3f} kN ({100 * ((2 * F_h4 - F_h2) / F_peak_ref - 1):+.1f} % vs published)")
F_l = peak(runs["l = 0.0075 mm, h = l/2"])[0]
print(f"  h = l/2: peak l = 0.0075 mm {F_l:.3f} kN vs l = 0.015 mm {F_h2:.3f} kN ({100 * (F_l / F_h2 - 1):+.1f} %)")

# Crack path: in every column of nodes right of the notch tip, the height of the
# maximum damage at full separation.
x, y = res.coordinates.T
d_end = res.damage[-1]
columns = np.unique(x[x > notch_tip + 1e-9])
path = []
for xc in columns:
    in_column = np.flatnonzero(np.abs(x - xc) < 1e-12)
    broken = in_column[d_end[in_column] > 0.95]          # fully broken nodes of this column
    path.append(y[broken].mean())
path = np.array(path)
print(f"Crack path: centre of the broken band in each of {len(columns)} node columns of the ligament; "
      f"mean y = {path.mean() / mm:.5f} mm, max |y - 0.5 mm| = {np.max(np.abs(path - 0.5 * side)) / mm * 1e3:.2f} um, "
      f"spread {np.ptp(path) / mm * 1e3:.2f} um (element size {main['h'] / mm * 1e3:.2f} um)")

# **Griffith balance.** The notch faces are free surfaces, so the new crack is the
# 0.5 mm ligament and the energy dissipated by its formation must be
# $G_c\cdot0.5\,\mathrm{mm}\cdot t$. On the mesh the separated crack is one fully broken
# layer of elements ($d=1$ on two node rows, $g=k$ in between) flanked by the two
# exponential tails of the AT2 profile; the regularised functional then gives
# $G_c\,(1+h/2\ell)$ per unit crack length (the layer contributes $h/2\ell$, each tail
# $1/2$), which tends to $G_c$ as $h/\ell\to0$. The energy dissipated after the peak (the
# diffuse sub-critical damage that AT2 produces before the peak is excluded) is compared
# with both.
print("Griffith energy balance (new crack = 0.5 mm ligament)")
griffith = G_c * (side - notch_tip) * thickness
for label, run in runs.items():
    r = run["result"]
    i_max = int(np.argmax(run["force"]))
    dissipated = r.fracture_energy[-1] - r.fracture_energy[i_max]
    discrete = griffith * (1 + run["h"] / (2 * run["ell"]))
    print(f"  {label:24s} dissipated after the peak {dissipated * 1e3:.4f} mJ;  G_c a t = {griffith * 1e3:.4f} mJ "
          f"(ratio {dissipated / griffith:.3f});  G_c a t (1 + h/2l) = {discrete * 1e3:.4f} mJ "
          f"({100 * (dissipated / discrete - 1):+.1f} %);  sub-critical damage before the peak "
          f"{r.fracture_energy[i_max] * 1e3:.4f} mJ")
# **Energy bookkeeping.** Before the peak the external work equals elastic plus crack
# energy. The crack then runs through the ligament within one load step (unstable
# growth under displacement control, as in the reference): the elastic energy released
# exceeds what the crack consumes, and in a quasi-static model the surplus has no place
# to go (physically it becomes kinetic energy), so the balance closes with a negative
# residual equal to that surplus.
W = res.external_work
i_max = int(np.argmax(main["force"]))
print(f"Energy bookkeeping (main run): at the peak work {W[i_max] * 1e3:.4f} mJ, elastic + crack "
      f"{(res.elastic_energy[i_max] + res.fracture_energy[i_max]) * 1e3:.4f} mJ "
      f"(residual {res.energy_balance_residual[i_max] / W[i_max]:+.1e} of the work);")
print(f"  at the end work {W[-1] * 1e3:.4f} mJ = crack {res.fracture_energy[-1] * 1e3:.4f} mJ "
      f"+ elastic {res.elastic_energy[-1] * 1e3:.4f} mJ + residual {res.energy_balance_residual[-1] * 1e3:+.4f} mJ "
      f"(released in the unstable step)")
print(f"Irreversibility: smallest damage change between steps {np.min(np.diff(res.damage, axis=0)):.1e}")


# ## Figure

def plot_damage(ax, run, step, title):
    r = run["result"]
    tri = Triangulation(r.coordinates[:, 0] / mm, r.coordinates[:, 1] / mm, run["cells"])
    image = ax.tripcolor(tri, r.damage[step], shading="gouraud", cmap=FIELD_CMAP, vmin=0, vmax=1, rasterized=True)
    ax.plot([0, notch_tip / mm], [0.5, 0.5], color=CREAM, lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    return image


fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
grid = fig.add_gridspec(2, 4)
i_peak = int(np.argmax(main["force"]))
drop = i_peak + int(np.argmax(main["force"][i_peak:] < 0.5 * main["force"][i_peak]))
shown = (int(np.argmin(np.abs(load_steps - 4e-6))), i_peak, drop)
for k, step in enumerate(shown):
    ax = fig.add_subplot(grid[0, k])
    plot_damage(ax, main, step, f"{main_run}\nu = {load_steps[step] / mm * 1e3:.2f} um, "
                                f"F = {main['force'][step] / 1e3:.3f} kN")
    ax.set_xlabel("x (mm)")
    if k == 0:
        ax.set_ylabel("y (mm)")
ax = fig.add_subplot(grid[0, 3])
other = runs["l = 0.0075 mm, h = l/2"]
image = plot_damage(ax, other, len(load_steps) - 1, f"l = 0.0075 mm, h = l/2\nu = {load_steps[-1] / mm * 1e3:.2f} um "
                                                     f"(separated)")
ax.set_xlabel("x (mm)")
fig.colorbar(image, ax=ax, label="phase field d", shrink=0.8)

ax_f = fig.add_subplot(grid[1, :2])
ax_f.plot(miehe_curve[:, 0] * 1e3, miehe_curve[:, 1], "o", mfc="none", color=INK, ms=6,
          label="Miehe et al. (2010), digitised")
styles = {"l = 0.0075 mm, h = l/2": (COPPER, "-"), "l = 0.015 mm, h = l/2": ("#8B9A8D", "--"),
          "l = 0.015 mm, h = l/4": (GREEN, "-")}
for label, run in runs.items():
    color, ls = styles[label]
    ax_f.plot(load_steps / mm * 1e3, run["force"] / 1e3, ls, color=color, lw=2.4 if label == main_run else 1.5,
              label=f"OneSimulation, {label}")
for step in shown:
    ax_f.plot(load_steps[step] / mm * 1e3, main["force"][step] / 1e3, "s", color=INK, ms=6)
ax_f.set_xlabel("top displacement u (um)")
ax_f.set_ylabel("force F (kN per mm thickness)")
ax_f.set_title("force-displacement curve")
ax_f.legend(fontsize=8, loc="upper left")
ax_f.grid(alpha=0.3)

ax_e = fig.add_subplot(grid[1, 2])
ax_e.plot(load_steps / mm * 1e3, res.external_work * 1e3, color=INK, label="external work")
ax_e.plot(load_steps / mm * 1e3, res.elastic_energy * 1e3, color=GREEN, label="elastic energy")
ax_e.plot(load_steps / mm * 1e3, res.fracture_energy * 1e3, color=COPPER, label="crack energy")
i_max = int(np.argmax(main["force"]))
ax_e.axhline((res.fracture_energy[i_max] + griffith * (1 + main["h"] / (2 * main["ell"]))) * 1e3, color=COPPER,
             ls="--", lw=1, label="pre-peak + G_c a t (1 + h/2l)")
ax_e.axhline((res.fracture_energy[i_max] + griffith) * 1e3, color=COPPER, ls=":", lw=1, label="pre-peak + G_c a t")
ax_e.set_xlabel("top displacement u (um)")
ax_e.set_ylabel("energy (mJ)")
ax_e.set_title(f"energies ({main_run})")
ax_e.legend(fontsize=8)
ax_e.grid(alpha=0.3)

ax_p = fig.add_subplot(grid[1, 3])
tri = Triangulation(x / mm, y / mm, main["cells"])
ax_p.tricontourf(tri, d_end, levels=np.linspace(0, 1, 21), cmap=FIELD_CMAP)
ax_p.plot(columns / mm, path / mm, ".", color=CREAM, ms=2.5, label="centre of broken band")
ax_p.axhline(0.5, color=INK, lw=0.6, ls="--")
ax_p.set_xlim(0.4, 1.0)
ax_p.set_ylim(0.44, 0.56)
ax_p.set_xlabel("x (mm)")
ax_p.set_ylabel("y (mm)")
ax_p.set_title("crack band at separation (zoom)")
ax_p.legend(fontsize=8, loc="upper left")
plt.show()


# ## Going further
#
# Switch the load to horizontal shear of the top edge (the second test of Miehe et al.):
# the crack then curves downwards, and the energy split decides whether spurious
# cracking appears in the compressed region. Try smaller $\ell$ together with a finer
# band, or replace the slit by an initial phase field $d=1$ on the notch line and see how
# much stiffer the initially notched plate becomes.
