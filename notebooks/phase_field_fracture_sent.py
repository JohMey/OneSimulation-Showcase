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

import time
import numpy as np

from onesim.meshing import ExplicitGeometry, explicit_mesh, merge_blocks, rectangle_block
from onesim.physics.solid_mechanics import PhaseFieldFracture
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

mm = 1e-3
side = 1.0 * mm
notch_tip = 0.5 * mm
thickness = 1.0 * mm
E, nu, G_c = 210e9, 0.3, 2700.0

miehe_curve = np.array([
    (0.0, 0.0), (0.00055, 0.0768), (0.00128, 0.1774), (0.00197, 0.2716), (0.00290, 0.3955),
    (0.00369, 0.4961), (0.00421, 0.5619), (0.00471, 0.6200), (0.00501, 0.6535), (0.00524, 0.6780),
    (0.00543, 0.6960), (0.00555, 0.7012), (0.00557, 0.6584), (0.00562, 0.5613), (0.00568, 0.4629),
    (0.00575, 0.3192), (0.00580, 0.1833), (0.00583, 0.0214), (0.00583, 0.0007)])
F_peak_ref, u_peak_ref = 0.70, 0.0055e-3

load_steps = np.r_[0.0, np.arange(0.5e-3, 4.01e-3, 0.5e-3), np.arange(4.25e-3, 5.0e-3, 0.25e-3),
                   np.arange(5.0e-3, 6.31e-3, 0.05e-3)] * mm

runs_spec = {
    "l = 0.0075 mm, h = l/2": (0.0075 * mm, 0.00375 * mm),
    "l = 0.015 mm, h = l/2": (0.015 * mm, 0.0075 * mm),
    "l = 0.015 mm, h = l/4": (0.015 * mm, 0.00375 * mm),
}
main_run = "l = 0.015 mm, h = l/4"


# ## Geometry and mesh
#
# The mesh is made of rectangular blocks split into triangles with alternating diagonals
# (`merge_blocks` and `triangulated("alternating")`). In $x$ the element size is $h$ from
# 6$\ell$ left of the notch tip to the right edge and grows geometrically (ratio 1.15, at
# most 0.05 mm) towards the left edge; in $y$ it is $h$ in a band of $\pm5\ell$ around the
# crack plane and grows in the same way towards the top and bottom. The notch is declared
# as a slit (`slits=("notch",)`, closed at the notch tip): its two faces get their own
# nodes, so it is a genuine slit whose faces separate. The boundary groups are `bottom`,
# `top`, `left`, `right` and `notch` (both faces).

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


plate = ExplicitGeometry(dimension=2, boundary_names=("bottom", "top", "left", "right", "notch"))


def build_mesh(ell, h):
    """Twelve rectangular blocks; the notch is a slit between the two blocks rows at y = 0.5 mm."""
    n_band = int(np.ceil(5 * ell / h))
    y0, y1 = 0.5 * side - n_band * h, 0.5 * side + n_band * h
    n_fine = int(np.ceil((side - (notch_tip - 6 * ell)) / h))
    x0 = side - n_fine * h
    x_segments = (graded(x0, 0.0, h)[::-1], np.linspace(x0, notch_tip, int(round((notch_tip - x0) / h)) + 1),
                  np.linspace(notch_tip, side, int(round((side - notch_tip) / h)) + 1))
    y_segments = (graded(y0, 0.0, h)[::-1], np.linspace(y0, 0.5 * side, n_band + 1),
                  np.linspace(0.5 * side, y1, n_band + 1), graded(y1, side, h))
    blocks = []
    for i, xs in enumerate(x_segments):
        for j, ys in enumerate(y_segments):
            slit = i < 2
            names = ("bottom" if j == 0 else "notch" if (j == 2 and slit) else None,
                     "right" if i == 2 else None,
                     "top" if j == 3 else "notch" if (j == 1 and slit) else None,
                     "left" if i == 0 else None)
            blocks.append(rectangle_block((xs[0], ys[0]), xs[-1] - xs[0], ys[-1] - ys[0], (len(xs) - 1, len(ys) - 1),
                                          grading=(xs, ys), names=names))
    section = merge_blocks(blocks, slits=("notch",), slit_tips=((notch_tip, 0.5 * side),)).triangulated("alternating")
    mesh = explicit_mesh(plate, section.coordinates, section.cells, boundary=section.edges)
    return plate, mesh, np.asarray(section.cells)


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
    broken = in_column[d_end[in_column] > 0.95]
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

i_peak = int(np.argmax(main["force"]))
drop = i_peak + int(np.argmax(main["force"][i_peak:] < 0.5 * main["force"][i_peak]))
shown = [int(np.argmin(np.abs(load_steps - 4e-6))), i_peak, drop]
other = runs["l = 0.0075 mm, h = l/2"]
u_um = load_steps / mm * 1e3
nodes = lambda run: (run["result"].coordinates[:, 0] / mm, run["result"].coordinates[:, 1] / mm)
fig = plots.figure(layout=[["d0", "d1", "d2", "d3"], ["force", "force", "energy", "band"]], size=(15, 9.5))
for panel, step in zip(("d0", "d1", "d2"), shown):
    plots.heatmap(*nodes(main), main["result"].damage[step], triangles=main["cells"], limits=(0, 1), label=False,
                  xlabel="x [mm]", ylabel="y [mm]", ax=fig[panel],
                  title=f"{main_run}\nu = {u_um[step]:.2f} µm, F = {main['force'][step] / 1e3:.3f} kN")
plots.heatmap(*nodes(other), other["result"].damage[-1], triangles=other["cells"], limits=(0, 1), label="phase field d",
              xlabel="x [mm]", ylabel="y [mm]", title=f"l = 0.0075 mm, h = l/2\nu = {u_um[-1]:.2f} µm (separated)", ax=fig["d3"])
plots.line(u_um, {f"OneSimulation, {label}": run["force"] / 1e3 for label, run in runs.items()},
           points={"Miehe et al. (2010), digitised": (miehe_curve[:, 0] * 1e3, miehe_curve[:, 1]),
                   "load steps shown above": (u_um[shown], main["force"][shown] / 1e3)},
           xlabel="top displacement u [µm]", ylabel="force F [kN per mm thickness]", title="Force-displacement curve",
           legend="upper left", ax=fig["force"])
plots.line(u_um, {"external work": res.external_work * 1e3, "elastic energy": res.elastic_energy * 1e3,
                  "crack energy": res.fracture_energy * 1e3},
           hlines={"pre-peak + G_c a t (1 + h/2l)": (res.fracture_energy[i_peak] + griffith * (1 + main["h"] / (2 * main["ell"]))) * 1e3,
                   "pre-peak + G_c a t": (res.fracture_energy[i_peak] + griffith) * 1e3},
           xlabel="top displacement u [µm]", ylabel="energy [mJ]", title=f"Energies ({main_run})", ax=fig["energy"])
plots.heatmap(x / mm, y / mm, d_end, triangles=main["cells"], limits=(0, 1), label=False, aspect="auto",
              xlimits=(0.4, 1.0), ylimits=(0.44, 0.56), xlabel="x [mm]", ylabel="y [mm]",
              title="Crack band at separation (zoom)", ax=fig["band"])
plots.show()


# ## Going further
#
# Switch the load to horizontal shear of the top edge (the second test of Miehe et al.):
# the crack then curves downwards, and the energy split decides whether spurious
# cracking appears in the compressed region. Try smaller $\ell$ together with a finer
# band, or replace the slit by an initial phase field $d=1$ on the notch line and see how
# much stiffer the initially notched plate becomes.
