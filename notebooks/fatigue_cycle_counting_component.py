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

import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from onesim.cad.d2 import Circle, Rectangle
from onesim.cad.d3 import Regrouped
from onesim.meshing import TriangularMesher
from onesim.numerics.time import rainflow_cycles, scalar_turning_points
from onesim.results.comparisons import convergence
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.physics.solid_mechanics import FatigueCycle, FatigueState, SNDataset
from onesim.physics.solid_mechanics import DamageLaw
from onesim.physics.solid_mechanics import StructuralDamageFractureFatigueProvider
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

mm = 1e-3
half_width, half_neck, half_length = 10 * mm, 6.25 * mm, 100 * mm
fillet_radius = 12.5 * mm
neck_end = 50 * mm - np.sqrt(fillet_radius**2 - (fillet_radius - (half_width - half_neck)) ** 2)
thickness = 6.25 * mm
E, nu = 69e9, 0.34

unit_force = 10e6 * (2 * half_neck) * thickness
grip_traction = (unit_force / 2) / (half_width * thickness)
load_block = np.array([-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0])
blocks = 1e5
infinite_life = 1e8

print(f"central section ends at y = {neck_end / mm:.3f} mm, fillet up to y = 50 mm")
print(f"unit load F_u = {unit_force:.2f} N, grip traction per unit = {grip_traction / 1e6:.4f} MPa")


def sn_life(amplitude, R):
    """Cycles to failure from the R-dependent S-N curve (amplitude in Pa)."""
    return (amplitude / (94e6 * (R / -0.36) ** 1.15)) ** (-1.0 / 0.119)


# ## Geometry and mesh
#
# The quarter specimen $x\ge0$, $y\ge0$ is declared exactly as described: a $10\times100$ mm
# rectangle minus the region beside the central section and minus the fillet circle of
# radius 12.5 mm centred at $(18.75, 41.07)$ mm (negative bodies). `TriangularMesher`
# triangulates it with 0.2 mm edges along the fillet and the central-section edge (about
# 60 elements along the fillet arc), 1.6 mm elsewhere, and a smooth grading. Finer meshes come from uniform
# refinement (each triangle split into four, new boundary points placed on the exact
# circle), so the element size halves exactly: 0.2, 0.1 and 0.05 mm on the fillet.

quarter = Regrouped(
    source=(Rectangle(length=half_width, width=half_length)
            - Rectangle(length=half_width - half_neck, width=neck_end, origin=(half_neck, 0.0, 0.0)).named("beside")
            - Circle(radius=fillet_radius, origin=(half_neck + fillet_radius, neck_end, 0.0)).named("fillet")),
    groups={"symmetry_x": ("x_min",), "symmetry_y": ("y_min",), "grip": ("y_max",),
            "free": ("x_max", "beside.x_min", "fillet.circumference")})


def build_mesh(level):
    mesher = TriangularMesher(target_size=1.6 * mm, grading=0.2, refinements=level,
                              local_sizes={"beside.x_min": 0.2 * mm, "fillet.circumference": 0.2 * mm})
    mesh = mesher.generate(quarter)
    return quarter, mesh, np.asarray(mesh.cells)


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


# **Rainflow counting (ASTM E1049-85, Sec. 5.4.4)** with OneSimulation's
# `scalar_turning_points` and `rainflow_cycles`. The history is first reduced to its
# turning points (peaks and valleys). Points are then read one by one onto a stack;
# with $X$ the range between the last two points and $Y$ the range before it, a range
# $Y\le X$ is counted: as a half cycle if $Y$ contains the starting point (and the
# starting point is discarded), otherwise as a full cycle (both points of $Y$ are
# removed). Ranges left on the stack at the end are counted as half cycles.

def rainflow(history, with_positions=False):
    """(minimum, maximum, count) of every counted cycle; count is 0.5 or 1.

    With positions, also the history indices of the two points of each range.
    """
    counted = list(rainflow_cycles(scalar_turning_points(np.asarray(history, dtype=float))))
    cycles = np.array([(min(c.start_value, c.end_value), max(c.start_value, c.end_value), c.count) for c in counted])
    positions = [(c.start_index, c.end_index) for c in counted]
    return (cycles, positions) if with_positions else cycles


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


sigma_bench = fine["bench"][1]
sigma_hot = fine["peak"]
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
    estimate = convergence(values, [refinement**2, refinement, 1.0])
    return (estimate.order, estimate.extrapolated) if estimate.order is not None else (np.nan, values[-1])


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
triangles = np.vstack([cells_p2[:, [0, 3, 5]], cells_p2[:, [3, 1, 4]], cells_p2[:, [5, 4, 2]], cells_p2[:, [3, 4, 5]]])
X, Y = space.dof_coordinates.T / mm
steps = np.arange(1, len(load_block) + 1)
history = {"fillet hot spot": (steps, load_block * sigma_hot / 1e6), "benchmark point": (steps, load_block * sigma_bench / 1e6)}
half_cycles = {}
for (i_a, i_b), (lo, hi, n) in zip(cycle_positions, cycles_units):
    label = f"{'full' if n == 1.0 else 'half'} cycle {5 * (hi - lo):.0f}/{5 * (hi + lo):.0f} MPa"
    history[label] = ([i_a + 1, i_b + 1], [load_block[i_a] * sigma_bench / 1e6, load_block[i_b] * sigma_bench / 1e6])
    if n != 1.0:
        half_cycles[label] = "dashed"
astm = np.array(astm_table, float)
bars = [f"{a:.0f}/{m:.0f}" for a, m in zip(amplitude / 1e6, mean / 1e6)]
binned_terms = np.where(N_bin < infinite_life, cycles_units[:, 2] / N_bin, 0) * blocks
fig = plots.figure(layout=[["part", "zoom", "damage", "history"], ["part", "matrix", "bars", "mesh"]], size=(21, 10))
plots.heatmap(np.r_[X, -X, X, -X], np.r_[Y, Y, -Y, -Y], np.tile(s1 / 1e6, 4),
              triangles=np.vstack([triangles + k * len(X) for k in range(4)]), limits=(0, s1.max() / 1e6),
              label="MPa per unit (F = 781 N)", xlabel="x [mm]", ylabel="y [mm]", title="Principal stress per load unit", ax=fig["part"])
plots.heatmap(X, Y, s1 / 1e6, triangles=triangles, limits=(6.0, s1.max() / 1e6), label="MPa per unit",
              xlimits=(0, 11), ylimits=(30, 56), xlabel="x [mm]", ylabel="y [mm]", title=f"Fillet: Kt = {Kt:.3f}", ax=fig["zoom"])
plots.heatmap(X, Y, np.log10(np.maximum(D_map, 1e-6)), triangles=triangles, limits=(-2.0, np.log10(D_map.max())),
              label="log10 D", xlimits=(0, 11), ylimits=(15, 56), xlabel="x [mm]", ylabel="y [mm]",
              title=f"Miner damage after 1e5 blocks (max {D_map.max():.2f})", ax=fig["damage"])
plots.line(y=history, styles=half_cycles, hlines=0.0, ylimits=(-125, 72), xlabel="load step in block", ylabel="stress [MPa]",
           title="Rainflow ranges in the load block", legend="lower center", ax=fig["history"])
plots.line(points={"ASTM E1049 example": (astm[:, 1], astm[:, 0]), "rainflow (this notebook)": (mean / 1e6, amplitude / 1e6)},
           vlines=m_edges / 1e6, hlines=a_edges / 1e6, xlabel="mean stress [MPa]", ylabel="stress amplitude [MPa]",
           title="Rainflow matrix and benchmark bins", legend="upper left", ax=fig["matrix"])
plots.bars(bars, {f"true cycles, D = {D_bench:.3f}": contrib, f"bin centres, D = {D_binned:.3f} (ref. 0.182)": binned_terms},
           ylabel="damage contribution (cycles as amplitude/mean in MPa)", title="Damage per cycle at the benchmark point", ax=fig["bars"])
plots.line(y={f"P{order} triangles": ([runs[order, level]["dofs"] for level in levels], [runs[order, level]["peak"] / 1e6 for level in levels])
              for order in (1, 2)},
           hlines={f"P2 finest {p2_peaks[-1] / 1e6:.4f} MPa": p2_peaks[-1] / 1e6}, markers=True, scale="logx",
           xlabel="unknowns", ylabel="fillet peak stress [MPa per unit]", title="Mesh study of the critical stress", ax=fig["mesh"])
plots.show()


# ## Going further
#
# Replace the R-dependent curve by a fully reversed S-N curve with a Goodman or Gerber
# mean-stress correction (`SNDataset(mean_stress_correction="goodman", ...)`) and compare
# the predicted lives; use a measured, non-proportional two-channel load history and
# count the cycles point by point on a critical-plane stress; or let the fillet radius
# vary and find the radius at which the fillet stops being the critical location.
