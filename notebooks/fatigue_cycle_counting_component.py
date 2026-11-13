"""OneSimulation example: rainflow cycle counting and Palmgren-Miner damage in a flat fatigue specimen.
Requires the installed, activated OneSimulation runtime.

A flat aluminium tensile specimen with fillet transitions carries a variable-amplitude
load block. The plane-stress stress field per unit load is computed with quadratic
finite elements; because the problem is linear, the stress history at every point is
the unit-load field times the load history. The history is counted with the ASTM E1049
rainflow algorithm (implemented here), the R-ratio-dependent S-N curve is applied and
the Palmgren-Miner damage is accumulated at the benchmark point and over the whole
specimen. The cycle table is checked against the ASTM E1049 worked example, the damage
against the published bin-based benchmark value and closed-form sub-histories, and the
fillet stress concentration by a three-level mesh study.
"""


# # Cycle counting and fatigue damage in a flat specimen
#
# A load spectrum that is not a single sinusoid has to be broken into closed stress
# cycles before an S-N curve can be applied. The standard way of doing this is
# **rainflow counting** (ASTM E1049-85, *Standard Practices for Cycle Counting in
# Fatigue Analysis*, Sec. 5.4.4). Each counted cycle $i$ (full or half) has an
# amplitude $\sigma_{a,i}$, a mean $\sigma_{m,i}$ and a load ratio
# $R_i=\sigma_{\min}/\sigma_{\max}$; the Palmgren-Miner rule then accumulates
#
# $$D=m\sum_i \frac{n_i}{N(\sigma_{a,i},R_i)} ,$$
#
# where $n_i\in\{\tfrac12,1\}$, $m$ is the number of repetitions of the load block and
# $N$ the life from the S-N curve. Failure is predicted at $D=1$.
#
# **Setup.** The benchmark specimen is a flat bar of thickness 6.25 mm: a central
# section 12.5 mm wide and 82.1 mm long, circular fillets of radius 12.5 mm, and
# 20 mm wide grips; overall length 200 mm. Aluminium, $E=69$ GPa, $\nu=0.34$. One unit
# of the load history is the axial force $F_u=781.25$ N (390.6 N on the modelled
# quarter) that produces 10 MPa in the central section. The load block is the ASTM E1049 example sequence
# $-2,1,-3,5,-1,3,-4,4,-2$ units, repeated $m=10^5$ times. The S-N curve depends on the
# stress amplitude and on $R$,
#
# $$\sigma_a = 94\,\mathrm{MPa}\,\Big(\frac{R}{-0.36}\Big)^{1.15} N^{-0.119},
#   \qquad -2.5\le R\le -0.2 ,$$
#
# so the mean-stress effect is contained in the $R$ dependence (a tensile mean makes
# $R$ less negative and lowers the allowable amplitude); lives $N\ge 10^8$ count as
# infinite and do not contribute damage.
#
# **Dimension.** The specimen is thin and loaded in its plane, so it is a plane-stress
# 2D problem. The stress field genuinely varies in $x$ and $y$ at the fillets, which is
# where the critical point lies. A quarter of the specimen is modelled with two
# symmetry lines.
#
# **Method.** Continuous Galerkin finite elements with quadratic (P2) Lagrange
# triangles, plane stress, on a graded unstructured triangulation (Netgen/OpenCASCADE),
# solved with a sparse direct LU factorisation; nodal stresses from an $L^2$
# projection of the quadrature-point stresses. Linear (P1) triangles on the same meshes
# serve as a method comparison. Rainflow counting (ASTM E1049, three-point algorithm
# with half cycles for the residue) and Miner accumulation through OneSimulation's
# `StructuralDamageFractureFatigueProvider`.
#
# **Reference.** (1) The rainflow cycle table of the ASTM E1049 worked example (exact,
# Fig. 6 of the standard). (2) The bin-based fatigue usage of this benchmark,
# $D=0.182$ (finite elements) and $D=0.184$ (hand evaluation of the ASTM cycles on
# the same bins). (3) Closed-form Miner damage for constant-amplitude sub-histories.
# (4) The nominal stress $F/A$ and the reaction force in the central section, and a
# three-level mesh study (P1 and P2): observed order and Richardson estimate of the
# specimen elongation, and the change of the fillet stress concentration between the
# meshes.

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
import netgen.occ as occ
from matplotlib.tri import Triangulation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from onesim.cad.geometry.primitives import Geometry
from onesim.meshing import UnstructuredSimplexMesh
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.physics.solid_mechanics.fatigue.models.fatigue_data import FatigueCycle, FatigueState, SNDataset
from onesim.physics.solid_mechanics.inelasticity.models.damage_laws import DamageLaw
from onesim.physics.solid_mechanics.inelasticity.models.damage_provider import (
    StructuralDamageFractureFatigueProvider,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters

mm = 1e-3
half_width, half_neck, half_length = 10 * mm, 6.25 * mm, 100 * mm   # grip, central section, specimen
fillet_radius = 12.5 * mm
neck_end = 50 * mm - np.sqrt(fillet_radius**2 - (fillet_radius - (half_width - half_neck)) ** 2)
thickness = 6.25 * mm
E, nu = 69e9, 0.34

unit_force = 10e6 * (2 * half_neck) * thickness        # N per load unit (10 MPa in the central section)
grip_traction = (unit_force / 2) / (half_width * thickness)   # quarter model: half the force on the half grip
load_block = np.array([-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0])   # load units
blocks = 1e5
infinite_life = 1e8

print(f"central section ends at y = {neck_end / mm:.3f} mm, fillet up to y = 50 mm")
print(f"unit load F_u = {unit_force:.2f} N, grip traction per unit = {grip_traction / 1e6:.4f} MPa")


def sn_life(amplitude, R):
    """Cycles to failure from the R-dependent S-N curve (amplitude in Pa)."""
    return (amplitude / (94e6 * (R / -0.36) ** 1.15)) ** (-1.0 / 0.119)


# ## Geometry and mesh
#
# The quarter specimen $x\ge0$, $y\ge0$ is built exactly as described: a $10\times100$ mm
# rectangle minus the region beside the central section and minus the fillet circle of
# radius 12.5 mm centred at $(18.75, 41.07)$ mm. Netgen triangulates it with 0.2 mm
# edges along the fillet and the central-section edge (about 60 elements along the
# fillet arc), 1.6 mm elsewhere, and a smooth grading. Finer meshes come from uniform
# refinement (each triangle split into four, new boundary points placed on the exact
# circle), so the element size halves exactly: 0.2, 0.1 and 0.05 mm on the fillet.

def quarter_face():
    outer = occ.WorkPlane().Rectangle(half_width, half_length).Face()
    beside = occ.WorkPlane().MoveTo(half_neck, 0).Rectangle(half_width - half_neck, neck_end).Face()
    circle = occ.WorkPlane().Circle(half_neck + fillet_radius, neck_end, fillet_radius).Face()
    face = outer - beside - circle
    for edge in face.edges:
        c = edge.center
        if c.y < 50 * mm + 1e-9 and c.x > half_neck - 1e-9:
            edge.maxh = 0.2 * mm
    return face


@dataclass(frozen=True, eq=False, kw_only=True)
class QuarterSpecimen(Geometry):
    """Quarter of the flat specimen with its four boundary groups."""

    dimension = 2

    def boundary_groups(self):
        names = ("symmetry_x", "symmetry_y", "grip", "free")
        return {**{name: (name,) for name in names}, "all": names}


def classify(points):
    if np.all(np.abs(points[:, 0]) < 1e-9):
        return "symmetry_x"
    if np.all(np.abs(points[:, 1]) < 1e-9):
        return "symmetry_y"
    if np.all(np.abs(points[:, 1] - half_length) < 1e-9):
        return "grip"
    return "free"


def build_mesh(level):
    ng = occ.OCCGeometry(quarter_face(), dim=2).GenerateMesh(maxh=1.6 * mm, grading=0.2)
    for _ in range(level):
        ng.Refine()
    xy = np.array([p.p for p in ng.Points()])[:, :2]
    triangles = np.array([[v.nr for v in e.vertices] for e in ng.Elements2D()]) - 1
    geometry = QuarterSpecimen()
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    for edge in ng.Elements1D():
        v = np.array([p.nr for p in edge.vertices]) - 1
        groups[classify(xy[v])].append(tuple(v))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=xy, cells=triangles, boundary_facets=groups)
    return geometry, mesh, triangles


# ## Model
#
# Boundary conditions on every boundary of the quarter:
#
# | boundary | condition |
# |---|---|
# | $x=0$ (longitudinal symmetry line) | $u_x=0$, zero shear traction |
# | $y=0$ (transverse symmetry line) | $u_y=0$, zero shear traction |
# | grip end $y=100$ mm | uniform traction $t_y=F_u/(2\cdot10\,\mathrm{mm}\cdot t)$ per load unit |
# | central-section edge, fillet, grip edge | traction-free |
#
# The material is linear elastic, so one solve with a unit load gives the stress per
# load unit everywhere; the stress at time step $k$ of the history is
# $\boldsymbol\sigma_k(\mathbf x)=L_k\,\hat{\boldsymbol\sigma}(\mathbf x)$.

solid = SmallStrainSolid(youngs_modulus=E, poisson_ratio=nu, kinematics="plane_stress", thickness=thickness)


def solve_unit_load(mesh, geometry, order):
    b = geometry.boundaries
    return solid.solve(mesh=mesh, times=(1.0,), order=order,
                       displacements=((b.symmetry_x, "x", 0.0), (b.symmetry_y, "y", 0.0)),
                       tractions=((b.grip, (0.0, grip_traction)), (b.free, (0.0, 0.0))))


def recover_stress(space, stress_q):
    """L2 projection of the quadrature-point stresses (xx, yy, xy) onto the continuous space."""
    rows, cols, vals = [], [], []
    rhs = np.zeros((space.dof_count, stress_q.shape[-1]))
    start = 0
    for cell, dofs in enumerate(space.cell_dofs):
        _, weights, (basis, _, _) = space.element_data(cell)
        stop = start + len(weights)
        rows.append(np.repeat(dofs, len(dofs)))
        cols.append(np.tile(dofs, len(dofs)))
        vals.append(((basis.T * weights) @ basis).ravel())
        np.add.at(rhs, dofs, (basis.T * weights) @ stress_q[start:stop])
        start = stop
    mass = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(space.dof_count,) * 2).tocsc()
    return spsolve(mass, rhs)


def principal_signed(s):
    """Plane-stress principal stress of largest magnitude, with its sign."""
    centre = 0.5 * (s[..., 0] + s[..., 1])
    radius = np.sqrt((0.5 * (s[..., 0] - s[..., 1])) ** 2 + s[..., 2] ** 2)
    return np.where(np.abs(centre + radius) >= np.abs(centre - radius), centre + radius, centre - radius)


# ## Solve
#
# Three refinement levels with P2 elements (the reported solution) and P1 elements
# (method comparison). For each run we record the stress per load unit at the benchmark
# point in the central section, $(3, 20)$ mm, the elongation of the quarter (grip-end
# displacement) and the largest principal stress along the fillet, which is the
# critical point of the specimen. The fillet stress is sampled at 800 points on the
# exact arc, moved 2 µm into the material so that every point lies inside the
# straight-sided triangles of all meshes.

levels = (0, 1, 2)
bench_point = np.array([[3.0 * mm, 20.0 * mm]])
arc_angle = np.linspace(np.pi, np.pi - np.arcsin((50 * mm - neck_end) / fillet_radius), 800)
arc_points = np.c_[half_neck + fillet_radius + (fillet_radius + 2e-6) * np.cos(arc_angle),
                   neck_end + (fillet_radius + 2e-6) * np.sin(arc_angle)]
runs = {}
for order in (1, 2):
    for level in levels:
        start = time.time()
        geometry, mesh, triangles = build_mesh(level)
        result = solve_unit_load(mesh, geometry, order)
        space = result.space
        nodal = recover_stress(space, result.stress[0][:, [0, 1, 0], [0, 1, 1]])
        s1 = principal_signed(nodal)
        xy = space.dof_coordinates
        along_arc = principal_signed(space.evaluate(nodal, arc_points))
        hot = np.argmax(along_arc)
        bench = space.evaluate(nodal, bench_point)[0]
        reaction = -result.reactions[0][np.abs(xy[:, 1]) < 1e-9, 1].sum()
        elongation = result.displacement[0][np.abs(xy[:, 1] - half_length) < 1e-9, 1].mean()
        runs[order, level] = dict(space=space, nodal=nodal, s1=s1, peak=along_arc[hot], hot_xy=arc_points[hot],
                                  bench=bench, reaction=reaction, elongation=elongation,
                                  cells=len(triangles), dofs=2 * space.dof_count)
        print(f"P{order} level {level}: {len(triangles):6d} triangles, {2 * space.dof_count:6d} unknowns, "
              f"fillet peak {along_arc[hot] / 1e6:.5f} MPa at ({arc_points[hot, 0] / mm:.3f}, "
              f"{arc_points[hot, 1] / mm:.3f}) mm, benchmark point {bench[1] / 1e6:.6f} MPa, "
              f"elongation {elongation * 1e9:.4f} nm ({time.time() - start:.1f} s)")
fine = runs[2, levels[-1]]
space, nodal, s1 = fine["space"], fine["nodal"], fine["s1"]
Kt = fine["peak"] / 10e6
print(f"stress concentration of the fillet (P2, finest): Kt = {Kt:.4f}")


# **Rainflow counting (ASTM E1049-85, Sec. 5.4.4).** The history is first reduced to
# its turning points (peaks and valleys). Points are then read one by one onto a stack;
# with $X$ the range between the last two points and $Y$ the range before it, a range
# $Y\le X$ is counted: as a half cycle if $Y$ contains the starting point (and the
# starting point is discarded), otherwise as a full cycle (both points of $Y$ are
# removed). Ranges left on the stack at the end are counted as half cycles.

def turning_points(history):
    """Indices and values of the peaks and valleys of a history."""
    points = [(0, float(history[0]))]
    for index, value in enumerate(history[1:], start=1):
        if value == points[-1][1]:
            continue
        if len(points) >= 2 and (points[-1][1] - points[-2][1]) * (value - points[-1][1]) > 0:
            points[-1] = (index, float(value))
        else:
            points.append((index, float(value)))
    return points


def rainflow(history, with_positions=False):
    """Return (minimum, maximum, count) of every counted cycle; count is 0.5 or 1.

    With positions, also return the history indices of the two points of each range.
    """
    stack, cycles, positions = [], [], []

    def count(a, b, n):
        cycles.append((min(a[1], b[1]), max(a[1], b[1]), n))
        positions.append((a[0], b[0]))

    for point in turning_points(history):
        stack.append(point)
        while len(stack) >= 3:
            X = abs(stack[-1][1] - stack[-2][1])
            Y = abs(stack[-2][1] - stack[-3][1])
            if X < Y:
                break
            if len(stack) == 3:
                count(stack[0], stack[1], 0.5)
                stack.pop(0)
            else:
                count(stack[-3], stack[-2], 1.0)
                last = stack.pop()
                del stack[-2:]
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):
        count(a, b, 0.5)
    return (np.array(cycles), positions) if with_positions else np.array(cycles)


cycles_units, cycle_positions = rainflow(load_block, with_positions=True)


# **Miner accumulation.** OneSimulation's `StructuralDamageFractureFatigueProvider`
# accumulates $n/N$ cycle by cycle from an `SNDataset` (log-log interpolation of
# stress-life points). The documented curve is a power law at fixed $R$, so for each
# counted cycle the dataset holds that curve sampled at $N=10^3$ and $N=10^8$ for the
# cycle's own $R$; log-log interpolation then reproduces the curve exactly. Cycles whose
# life exceeds $10^8$ are infinite-life cycles and are skipped. Miner's rule is linear,
# so the damage of $m$ blocks is $m$ times the damage of one block.

provider = StructuralDamageFractureFatigueProvider(
    damage_law=DamageLaw(law_id="miner", kind="smeared_cracking", critical_history=1.0))


def cycle_stresses(cycles, stress_per_unit):
    low, high = stress_per_unit * cycles[:, 0], stress_per_unit * cycles[:, 1]
    s_min, s_max = np.minimum(low, high), np.maximum(low, high)
    return 0.5 * (s_max - s_min), 0.5 * (s_max + s_min), s_min / s_max


def block_damage(cycles, stress_per_unit, label):
    """Miner damage of one load block at a point with the given stress per load unit."""
    amplitude, mean, R = cycle_stresses(cycles, stress_per_unit)
    state = FatigueState()
    for i, (sa, sm, r, n) in enumerate(zip(amplitude, mean, R, cycles[:, 2])):
        if sn_life(sa, r) >= infinite_life:
            continue
        curve = SNDataset(dataset_id=f"{label}-R{r:.3f}", cycles_to_failure=(1e3, infinite_life),
                          stress_amplitudes=(sa * (sn_life(sa, r) / 1e3) ** 0.119,
                                             sa * (sn_life(sa, r) / infinite_life) ** 0.119))
        state = provider.update_fatigue(state, FatigueCycle(cycle_id=f"{label}-{i}", stress_amplitude=sa,
                                                            strain_amplitude=sa / E, mean_stress=sm, count=n),
                                        sn_data=curve)
    return state.damage


def damage_field(cycles, stress_per_unit):
    """Vectorised Miner damage of m blocks for an array of stresses per load unit."""
    s = np.asarray(stress_per_unit, float)[:, None]
    low, high = s * cycles[None, :, 0], s * cycles[None, :, 1]
    s_min, s_max = np.minimum(low, high), np.maximum(low, high)
    amplitude = 0.5 * (s_max - s_min)
    with np.errstate(divide="ignore", invalid="ignore"):
        life = sn_life(amplitude, s_min / s_max)
        terms = np.where((life < infinite_life) & (amplitude > 0), cycles[None, :, 2] / life, 0.0)
    return blocks * np.nansum(terms, axis=1)


sigma_bench = fine["bench"][1]            # axial stress per unit at the benchmark point
sigma_hot = fine["peak"]                  # largest principal stress per unit at the fillet
D_bench = blocks * block_damage(cycles_units, sigma_bench, "bench")
D_hot = blocks * block_damage(cycles_units, sigma_hot, "fillet")
D_map = damage_field(cycles_units, s1)

# The benchmark also evaluates the damage on a 7 x 5 grid of amplitude and mean-stress
# bins (amplitudes 15-45 MPa, means -10-10 MPa) with all stresses taken at the bin
# centres; we reproduce that evaluation from the counted cycles.
amplitude, mean, R = cycle_stresses(cycles_units, sigma_bench)
a_edges, m_edges = np.linspace(15e6, 45e6, 8), np.linspace(-10e6, 10e6, 6)
ia = np.clip(np.searchsorted(a_edges, amplitude + 1e-6, side="right") - 1, 0, 6)
im = np.clip(np.searchsorted(m_edges, mean + 1e-6, side="right") - 1, 0, 4)
a_bin, m_bin = 0.5 * (a_edges[ia] + a_edges[ia + 1]), 0.5 * (m_edges[im] + m_edges[im + 1])
R_bin = (m_bin - a_bin) / (m_bin + a_bin)
N_bin = sn_life(a_bin, R_bin)
D_binned = blocks * np.sum(np.where(N_bin < infinite_life, cycles_units[:, 2] / N_bin, 0.0))
print(f"Damage after {blocks:.0e} blocks: benchmark point {D_bench:.4f} (true cycles), "
      f"{D_binned:.4f} (bin centres); fillet {D_hot:.3f}")


# ## Comparison with the reference
#
# **1. Rainflow table.** ASTM E1049-85, Fig. 6: ranges of 3, 4, 6, 8 and 9 units with
# 0.5, 1.5, 0.5, 1.0 and 0.5 cycles; the benchmark lists the individual cycles with
# their means (amplitude/mean in MPa at 10 MPa per unit).

astm_table = [(20, -10, 0.5), (15, -5, 0.5), (40, 0, 0.5), (45, 5, 0.5), (40, 10, 0.5), (30, 10, 0.5), (20, 10, 1.0)]
counted = {(round(sa / 1e6, 3), round(sm / 1e6, 3)): n for sa, sm, n in zip(amplitude, mean, cycles_units[:, 2])}
print("Rainflow cycles at the benchmark point (10 MPa per unit)")
print("  ASTM E1049: amplitude/MPa  mean/MPa  count   |  this notebook: count")
for sa, sm, n in astm_table:
    print(f"  {sa:24.1f}  {sm:8.1f}  {n:5.1f}   |  {counted.get((float(sa), float(sm)), 0.0):19.1f}")
print(f"  cycles counted in total: {len(cycles_units)} entries, {cycles_units[:, 2].sum():.1f} cycles "
      f"(ASTM: 7 entries, {sum(c[2] for c in astm_table):.1f} cycles)")
ranges = np.round(cycles_units[:, 1] - cycles_units[:, 0], 9)
for r_units, ref in ((3, 0.5), (4, 1.5), (6, 0.5), (8, 1.0), (9, 0.5)):
    print(f"  range {r_units} units: counted {cycles_units[ranges == r_units, 2].sum():.1f} cycles, ASTM {ref:.1f}")

# **2. Stress per load unit.** In the central section the stress must equal the nominal
# $F_u/A=10$ MPa; the reaction on the symmetry line must equal half the unit force.
print(f"Benchmark point: sigma_yy = {sigma_bench / 1e6:.6f} MPa per unit, nominal 10 MPa, "
      f"deviation {100 * (sigma_bench / 10e6 - 1):+.4f} %")
print(f"Reaction on y = 0: {fine['reaction']:.4f} N, applied F_u/2 = {unit_force / 2:.4f} N, "
      f"deviation {100 * (fine['reaction'] / (unit_force / 2) - 1):+.2e} %")

# **3. Bin-based fatigue usage of the benchmark.** Lives at the bin centres and the
# usage; reference values $N_b$ from the benchmark table and $D=0.182$ (finite
# elements) / $0.184$ (hand evaluation).
ref_N = {(21.4, 8.0): 2.47e6, (38.6, 8.0): 5.84e5, (42.9, 4.0): 2.31e6, (38.6, 0.0): 3.44e7, (30.0, 8.0): 1.45e6}
print("Damaging bins:  bin amplitude  bin mean    R_b      N_b      reference N_b")
for ab, mb, rb, nb in zip(a_bin / 1e6, m_bin / 1e6, R_bin, N_bin):
    key = (round(ab, 1), round(mb, 1))
    if key in ref_N:
        print(f"               {ab:8.2f}     {mb:6.2f}   {rb:7.3f}  {nb:9.3e}   {ref_N[key]:9.3e} "
              f"({100 * (nb / ref_N[key] - 1):+.1f} %)")
print(f"Bin-based usage D = {D_binned:.4f}; benchmark 0.182 ({100 * (D_binned / 0.182 - 1):+.2f} %), "
      f"hand evaluation 0.184 ({100 * (D_binned / 0.184 - 1):+.2f} %)")

# **4. Constant-amplitude sub-histories (closed form).** Repeating one counted cycle
# $k$ times must give $D=m\,k/N(\sigma_a,R)$ exactly; the whole pipeline (rainflow,
# provider, block scaling) is run on such histories.
for lo, hi in ((-4.0, 4.0), (-1.0, 3.0), (-3.0, 5.0)):
    sub = np.r_[np.tile([lo, hi], 20), lo]
    counted = rainflow(sub)
    D_sub = blocks * block_damage(counted, sigma_bench, "sub")
    sa, R_sub = 0.5 * (hi - lo) * sigma_bench, lo / hi
    exact = blocks * 20 / sn_life(sa, R_sub)
    print(f"  {lo:+.0f} <-> {hi:+.0f} units, 20 cycles: counted {counted[:, 2].sum():.1f} cycles, "
          f"D = {D_sub:.6f}, closed form {exact:.6f} ({100 * (D_sub / exact - 1):+.1e} %)")

# The binned usage is far below the damage of the true cycles: the S-N curve is very
# sensitive to $R$ near the tensile end, and the bin centres shift the cycle
# $-1\to3$ units from $R=-0.333$ to $R=-0.456$. This is the non-conservatism of bin
# evaluation noted in the benchmark.
contrib = np.where(sn_life(amplitude, R) < infinite_life, cycles_units[:, 2] / sn_life(amplitude, R), 0) * blocks
for (sa, sm, r), d in zip(np.c_[amplitude / 1e6, mean / 1e6, R], contrib):
    print(f"  cycle {sa:4.0f}/{sm:4.0f} MPa, R = {r:6.3f}: true-cycle damage {d:.4f}")

# **5. Critical point at the fillet and mesh study.** The load is proportional, so $R$
# is the same at every point and the damage scales exactly as $(\sigma/\sigma_0)^{1/0.119}$
# with the local stress per load unit; the fillet damage must equal
# $D_\mathrm{bench}\,(\sigma_\mathrm{hot}/\sigma_\mathrm{bench})^{8.403}$.
#
# In the mesh study the elongation converges monotonically, so its observed order and
# Richardson estimate are printed. The P2 fillet peak changes only in the fourth
# significant digit between the meshes and not monotonically (each refinement places new
# vertices on the circle, and the straight-sided triangles represent the arc as a
# polygon), so no order is defined for it; its spread over the three meshes is the
# uncertainty we quote.

def richardson(values, refinement=2.0):
    f1, f2, f3 = values
    if (f1 - f2) * (f2 - f3) <= 0:
        return np.nan, np.nan
    p = np.log(abs(f1 - f2) / abs(f2 - f3)) / np.log(refinement)
    return p, f3 + (f3 - f2) / (refinement**p - 1.0)


print(f"Fillet damage {D_hot:.4f}, scaling law {D_bench * (sigma_hot / sigma_bench) ** (1 / 0.119):.4f}; "
      f"damage map maximum (nodes) {D_map.max():.4f}; predicted life {blocks / D_hot:.0f} blocks "
      f"(benchmark point {blocks / D_bench:.0f} blocks)")
print("Mesh study (element size on the fillet 0.2, 0.1, 0.05 mm)")
for order in (1, 2):
    for key, label, scale, unit in (("elongation", "elongation", 1e9, "nm"), ("peak", "fillet peak", 1e-6, "MPa")):
        values = [runs[order, level][key] * scale for level in levels]
        p, extrapolated = richardson(values)
        trend = (f"observed order {p:.2f}, Richardson {extrapolated:.5f}" if np.isfinite(p)
                 else "non-monotone, no observed order")
        print(f"  P{order} {label:11s}: " + "  ".join(f"{v:.5f}" for v in values) + f" {unit}   {trend}, "
              f"change last two meshes {100 * abs(values[-1] / values[-2] - 1):.4f} %")
p2_peaks = [runs[2, level]["peak"] for level in levels]
print(f"  spread of the P2 fillet peak over the three meshes: "
      f"{100 * (max(p2_peaks) - min(p2_peaks)) / p2_peaks[-1]:.4f} %")
print(f"  P1 vs P2 fillet peak on the finest mesh: {100 * (runs[1, levels[-1]]['peak'] / p2_peaks[-1] - 1):+.3f} %")
print(f"  life sensitivity: a 0.1 % stress change alters the fillet damage by {100 * (1.001 ** (1 / 0.119) - 1):.2f} %")


# ## Figure

cells_p2 = np.asarray(space.cell_dofs)
sub = np.vstack([cells_p2[:, [0, 3, 5]], cells_p2[:, [3, 1, 4]], cells_p2[:, [5, 4, 2]], cells_p2[:, [3, 4, 5]]])
X, Y = space.dof_coordinates.T / mm


def mirrored(ax, values, **kwargs):
    for sx in (1, -1):
        for sy in (1, -1):
            tri = Triangulation(sx * X, sy * Y, sub if sx * sy > 0 else sub[:, ::-1])
            image = ax.tripcolor(tri, values, shading="gouraud", rasterized=True, **kwargs)
    return image


fig = plt.figure(figsize=(15, 9.6), constrained_layout=True)
grid = fig.add_gridspec(2, 4, width_ratios=(0.55, 1.0, 1.0, 1.0))
ax_full = fig.add_subplot(grid[:, 0])
ax_zoom = fig.add_subplot(grid[0, 1])
ax_dam = fig.add_subplot(grid[0, 2])
ax_hist = fig.add_subplot(grid[0, 3])
ax_rf = fig.add_subplot(grid[1, 1])
ax_bar = fig.add_subplot(grid[1, 2])
ax_mesh = fig.add_subplot(grid[1, 3])

image = mirrored(ax_full, s1 / 1e6, cmap=FIELD_CMAP, vmin=0, vmax=s1.max() / 1e6)
ax_full.set_aspect("equal")
ax_full.set_title("principal stress\nper load unit")
ax_full.set_xlabel("x (mm)")
ax_full.set_ylabel("y (mm)")
fig.colorbar(image, ax=ax_full, shrink=0.95, location="bottom", label="MPa per unit (F = 781 N)")

zoom = (X < 11) & (Y > 30) & (Y < 56)
tri_q = Triangulation(X, Y, sub)
levels_c = np.linspace(6.0, s1.max() / 1e6, 25)
image = ax_zoom.tricontourf(tri_q, s1 / 1e6, levels=levels_c, cmap=FIELD_CMAP, extend="min")
ax_zoom.triplot(Triangulation(*(runs[2, 0]["space"].dof_coordinates.T / mm),
                              np.asarray(runs[2, 0]["space"].cell_dofs)[:, :3]), color=INK, lw=0.25, alpha=0.35)
ax_zoom.plot(*fine["hot_xy"] / mm, marker="*", ms=14, color=COPPER, mec=INK)
ax_zoom.set_xlim(0, 11)
ax_zoom.set_ylim(30, 56)
ax_zoom.set_aspect("equal")
ax_zoom.set_title(f"fillet, coarsest mesh: Kt = {Kt:.3f}")
ax_zoom.set_xlabel("x (mm)")
ax_zoom.set_ylabel("y (mm)")
fig.colorbar(image, ax=ax_zoom, label="MPa per unit")

log_damage = np.log10(np.maximum(D_map, 1e-6))
image = ax_dam.tricontourf(tri_q, log_damage, levels=np.linspace(-2.0, np.log10(D_map.max()), 25),
                           cmap=FIELD_CMAP, extend="min")
ax_dam.plot(*fine["hot_xy"] / mm, marker="*", ms=14, color=COPPER, mec=INK)
ax_dam.plot(*bench_point[0] / mm, marker="o", ms=8, color=CREAM, mec=INK)
ax_dam.set_xlim(0, 11)
ax_dam.set_ylim(15, 56)
ax_dam.set_aspect("equal")
ax_dam.set_title(f"Miner damage after 1e5 blocks (max {D_map.max():.2f})")
ax_dam.set_xlabel("x (mm)")
fig.colorbar(image, ax=ax_dam, label="log10 D")

steps = np.arange(1, len(load_block) + 1)
ax_hist.plot(steps, load_block * sigma_hot / 1e6, "-", color=COPPER, lw=1.2, label="fillet hot spot")
ax_hist.plot(steps, load_block * sigma_bench / 1e6, "-", color="#9FA593", lw=1.0, label="benchmark point")
cycle_colors = [GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]
for k, ((i_a, i_b), (lo, hi, n)) in enumerate(zip(cycle_positions, cycles_units)):
    ya, yb = load_block[i_a] * sigma_bench / 1e6, load_block[i_b] * sigma_bench / 1e6
    ax_hist.plot([i_a + 1, i_b + 1], [ya, yb], color=cycle_colors[k], lw=4.5 if n == 1.0 else 2.2,
                 ls="-" if n == 1.0 else "--", alpha=0.85,
                 label=f"{'full' if n == 1.0 else 'half'} cycle {5 * (hi - lo):.0f}/{5 * (hi + lo):.0f} MPa")
ax_hist.axhline(0, color=INK, lw=0.6)
ax_hist.set_xlabel("load step in block")
ax_hist.set_ylabel("stress (MPa)")
ax_hist.set_title("rainflow ranges in the load block")
ax_hist.legend(fontsize=7, ncol=2, loc="lower center")
ax_hist.set_ylim(-125, 72)
ax_hist.grid(alpha=0.3)

ref = np.array(astm_table, float)
ax_rf.scatter(ref[:, 1], ref[:, 0], s=900 * ref[:, 2], facecolors="none", edgecolors=COPPER, lw=2,
              label="ASTM E1049 example")
ax_rf.scatter(mean / 1e6, amplitude / 1e6, s=250 * cycles_units[:, 2], color=GREEN, label="rainflow (this notebook)")
for x0 in m_edges / 1e6:
    ax_rf.axvline(x0, color="#9FA593", lw=0.6)
for y0 in a_edges / 1e6:
    ax_rf.axhline(y0, color="#9FA593", lw=0.6)
ax_rf.set_xlabel("mean stress (MPa)")
ax_rf.set_ylabel("stress amplitude (MPa)")
ax_rf.set_title("rainflow matrix with the benchmark bins")
ax_rf.legend(fontsize=8, loc="upper left", markerscale=0.5)

labels = [f"{a:.0f}/{m:.0f}" for a, m in zip(amplitude / 1e6, mean / 1e6)]
binned_terms = np.where(N_bin < infinite_life, cycles_units[:, 2] / N_bin, 0) * blocks
xpos = np.arange(len(labels))
ax_bar.bar(xpos - 0.2, contrib, width=0.4, color=GREEN, label=f"true cycles, D = {D_bench:.3f}")
ax_bar.bar(xpos + 0.2, binned_terms, width=0.4, color=COPPER, label=f"bin centres, D = {D_binned:.3f} (ref. 0.182)")
ax_bar.set_xticks(xpos, labels, rotation=35, fontsize=8)
ax_bar.set_xlabel("cycle amplitude/mean (MPa)")
ax_bar.set_ylabel("damage contribution")
ax_bar.set_title("damage per cycle at the benchmark point")
ax_bar.legend(fontsize=8)
ax_bar.grid(alpha=0.3, axis="y")

for order, color in ((1, COPPER), (2, GREEN)):
    dofs = [runs[order, level]["dofs"] for level in levels]
    peaks = [runs[order, level]["peak"] / 1e6 for level in levels]
    ax_mesh.semilogx(dofs, peaks, "-o", color=color, label=f"P{order} triangles")
ax_mesh.axhspan(p2_peaks[-1] / 1e6 * (1 - 2e-4), p2_peaks[-1] / 1e6 * (1 + 2e-4), color=GREEN, alpha=0.15,
                label=f"P2 finest {p2_peaks[-1] / 1e6:.4f} ± 0.02 %")
ax_mesh.set_xlabel("unknowns")
ax_mesh.set_ylabel("fillet peak stress (MPa per unit)")
ax_mesh.set_title("mesh study of the critical stress")
ax_mesh.legend(fontsize=8)
ax_mesh.grid(alpha=0.3)
plt.show()


# ## Going further
#
# Replace the R-dependent curve by a fully reversed S-N curve with a Goodman or Gerber
# mean-stress correction (`SNDataset(mean_stress_correction="goodman", ...)`) and compare
# the predicted lives; use a measured, non-proportional two-channel load history and
# count the cycles point by point on a critical-plane stress; or let the fillet radius
# vary and find the radius at which the fillet stops being the critical location.
