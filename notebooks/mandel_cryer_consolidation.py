"""OneSimulation example: the Mandel-Cryer effect in a consolidating poroelastic sphere.
Requires the installed, activated OneSimulation runtime.

A saturated soil sphere is suddenly loaded by a uniform surface pressure and drains through
its surface. Fully coupled 3D Biot poroelasticity (mixed fluid content / displacement / pore
pressure, quadratic hexahedra on a curved octant mesh, Crank-Nicolson with a backward-Euler
start) reproduces the rise of the centre pore pressure above its initial value and is graded
against Cryer's exact series solution, a mesh and a time-step study and the fluid balance.
"""


# # The Mandel-Cryer effect in a sphere
#
# A sphere of saturated soil, radius $a=1$ m, is loaded at $t=0$ by a uniform surface
# pressure $P=100$ kPa and is free to drain through its surface. At the instant of loading
# the water cannot escape, so the load is carried by an **undrained** pore pressure $p_0$
# that is uniform in the whole sphere. Then the outer shell drains first, consolidates and
# contracts. The contracting shell squeezes the still undrained core, and the core pore
# pressure **rises above $p_0$** before it finally decays. This non-monotonic response,
# impossible in uncoupled (Terzaghi-type) diffusion, is the *Mandel-Cryer effect*
# (J. Mandel, Géotechnique 3, 287, 1953; C.W. Cryer, Q. J. Mech. Appl. Math. 16, 401, 1963).
#
# The physics is Biot's theory of poroelasticity (M.A. Biot, J. Appl. Phys. 12, 155, 1941):
#
# $$\nabla\cdot\big(\mathbf C:\boldsymbol\varepsilon(\mathbf u)-\alpha\,p\,\mathbf I\big)=0,\qquad
#   \zeta=S\,p+\alpha\,\nabla\cdot\mathbf u,\qquad
#   \partial_t\zeta-\nabla\cdot\Big(\frac{\kappa}{\mu_f}\nabla p\Big)=0 ,$$
#
# with displacement $\mathbf u$, pore pressure $p$, fluid content $\zeta$, Biot coefficient
# $\alpha$, storage coefficient $S$ and mobility $\kappa/\mu_f$.
#
# **Geometry, dimension and boundary conditions.** The problem is solved in 3D. By symmetry
# one octant $x,y,z\ge 0$ of the sphere is modelled; the three coordinate planes are
# planes of symmetry. Boundary conditions on every face:
#
# | face | mechanics | flow |
# |---|---|---|
# | spherical surface $r=a$ | total normal traction $-P$ (compression) | drained, $p=0$ |
# | symmetry plane $x=0$ | roller $u_x=0$ | no flow |
# | symmetry plane $y=0$ | roller $u_y=0$ | no flow |
# | symmetry plane $z=0$ | roller $u_z=0$ | no flow |
#
# **Material.** A soft clayey silt: drained Young's modulus $E=10$ MPa, drained Poisson's
# ratio $\nu=0$ (the case with the strongest Mandel-Cryer effect), Biot coefficient
# $\alpha=1$, intrinsic permeability $\kappa=10^{-14}$ m$^2$, water viscosity
# $\mu_f=10^{-3}$ Pa s. The pore water is slightly compressible: porosity $0.4$ and water
# bulk modulus $2.2$ GPa give $S=0.4/2.2\ \mathrm{GPa}^{-1}=1.82\cdot10^{-10}$ Pa$^{-1}$
# (Skempton coefficient $B=0.998$). Cryer's classical solution assumes incompressible
# constituents ($S=0$); the exact solution below is the same derivation with $S>0$ kept,
# and the notebook also prints the $S=0$ values to show that the difference is small.
#
# **Numerical method.**
# - Spatial: continuous Galerkin finite elements on a block-structured mesh of curved
#   27-node hexahedra (a "cubed-sphere" octant: a cube at the centre and three blocks that
#   map its faces onto the sphere, nodes on the exact sphere). Mixed three-field Biot
#   formulation with the equal-order pair **Q2 fluid content / Q2 displacement / Q2 pore
#   pressure**; the drained condition $p=0$ is imposed weakly by Nitsche's method, the
#   rollers strongly.
# - Time: **Crank-Nicolson** (trapezoidal rule, second order) on the fluid content, started
#   with four **backward-Euler** steps (Rannacher start) that damp the high-frequency content
#   of the sudden load, which Crank-Nicolson alone would carry along as oscillations. The
#   step grows piecewise: $\Delta T=2.5\cdot10^{-4}$ in the backward-Euler start, then
#   $5\cdot10^{-4}$, $2\cdot10^{-3}$, $5\cdot10^{-3}$, $0.02$, $0.05$ in units of the
#   consolidation time $a^2/c$. Each constant-step segment is a separate run that starts
#   from the fluid content at the end of the previous one.
# - Solvers: Newton's method on every stage (one iteration for this linear problem) with a
#   sparse direct LU solver (SuperLU). The LU factor is reused as long as the assembled
#   matrix does not change (a small factory plugged into `SparseDirect`): the equilibrium
#   matrix is constant and the stage matrix only changes with the step size, so only a
#   handful of factorizations are needed for the whole history.
#
# **Reference.** (i) The exact series solution for the centre pore pressure (Cryer 1963),
# including the size and time of its peak, and the exact radial pore-pressure profiles,
# surface displacement and expelled fluid volume of the same solution. (ii) A three-level
# mesh study and a time-step study. (iii) The fluid balance: the change of the fluid volume
# in the sphere against the time-integrated outflow through the drained surface.

import time
import numpy as np
from scipy.optimize import brentq

from onesim.meshing import ExplicitGeometry, Geometric, VolumeBlock, exterior_faces, merge_volume_blocks
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, Nitsche
from onesim.numerics.linear import SparseDirect
from onesim.numerics.linear import ReusedFactorization
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import TimeMethod
from onesim.physics.geotechnics import (BiotConsolidation, BiotFixedComponent, BiotNormalTraction,
                                        PoreFluidFlux, PrescribedPorePressure)
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# **Units.** Lengths in m, stresses and pressures in MPa, time in hours. This is a consistent
# rescaling of the SI equations (every coefficient is converted) that keeps the entries of
# the coupled matrices within a few orders of magnitude of each other.

hour = 3600.0
a = 1.0
P = 0.1
E, nu, alpha = 10.0, 0.0, 1.0
S = 0.4 / 2.2e3
mobility = 1e-14 / 1e-3 * 1e6 * hour

mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
M_c = lam + 2 * mu
K = lam + 2 * mu / 3
beta = alpha / (M_c * S + alpha**2)
eta = 4 * mu * alpha * beta / (3 * K)
c = mobility / (S + alpha**2 / M_c)
t_c = a**2 / c
p0 = beta * M_c * P / (K + 4 * mu * alpha * beta / 3)
eta_cryer = 4 * mu / (3 * K)
print(f"consolidation coefficient c = {c:.4f} m^2/h, a^2/c = {t_c:.3f} h, "
      f"undrained pore pressure p0 = {1e3 * p0:.3f} kPa (= {p0 / P:.5f} P), eta = {eta:.5f}")

SCHEDULE = (("backward_euler", 2.5e-4, 1e-3),
            ("crank_nicolson", 5e-4, 6e-3), ("crank_nicolson", 2e-3, 0.03),
            ("crank_nicolson", 5e-3, 0.15), ("crank_nicolson", 2e-2, 0.5), ("crank_nicolson", 5e-2, 1.0))
FIGURE_TIMES = (0.006, 0.03, 0.065, 0.25)


# ## Geometry and mesh
#
# The octant is covered by four volume blocks of 27-node hexahedra: the cube $[0,b]^3$ with
# $b=0.55\,a$ ($n^3$ cells) and three blocks ($n\times n\times n$ cells) that connect each outer
# face of the cube to the part of the sphere onto which that face projects from the centre.
# Each block map places a point on the straight line from the cube point to its projection
# on the sphere, so the outer nodes, including the mid-edge and mid-face nodes, lie exactly
# on the sphere; `merge_volume_blocks` joins the blocks. The radial cells shrink towards the drained surface (last cell
# half the first), where the early-time pressure boundary layer forms. The mesh level $n$
# sets the number of cells along every block edge; $n=3,4,6$ give 108, 256 and 864 curved
# hexahedra.

def octant_face(point, normal):
    planes = [k for k in range(3) if abs(point[k]) < 1e-12]
    return "plane_" + "xyz"[planes[0]] if planes else "surface"


octant = ExplicitGeometry(dimension=3, boundary_names=("surface", "plane_x", "plane_y", "plane_z"), classifier=octant_face)


def projection_block(k, b):
    """Block from cube face x_k = b to the part of the sphere it projects onto from the centre."""
    def mapping(u, v, rho):
        cube = [None] * 3
        cube[k], cube[(k + 1) % 3], cube[(k + 2) % 3] = b * np.ones_like(u), b * u, b * v
        length = np.sqrt(sum(c**2 for c in cube))
        return tuple(c + rho * (a * c / length - c) for c in cube)
    return mapping


def octant_mesh(n, b=0.55 * a, ratio=2.0):
    s = np.linspace(0.0, 1.0, n + 1)
    rho = Geometric(n, ratio=ratio, toward="end").reference()
    blocks = [VolumeBlock(lambda u, v, w: (b * u, b * v, b * w), s, s, s)]
    blocks += [VolumeBlock(projection_block(k, b), s, s, rho) for k in range(3)]
    mesh = merge_volume_blocks(blocks, octant, order=2)
    faces = exterior_faces(np.asarray(mesh.coordinates), np.asarray(mesh.cells), dimension=3)
    return dict(n=n, domain=octant, mesh=mesh, X=np.asarray(mesh.coordinates), C=np.asarray(mesh.cells), faces=faces)


MESH_LEVELS = (3, 4, 6)
meshes = {n: octant_mesh(n) for n in MESH_LEVELS}
for n, data in meshes.items():
    print(f"n = {n}: {len(data['C'])} hexahedra, {len(data['X'])} nodes, volume error of the quadratic geometry "
          f"{data['mesh'].cell_volumes.sum() / (np.pi * a**3 / 6) - 1:+.1e}")


# ## Model
#
# The factory below plugs SuperLU with factor reuse into `SparseDirect`: a factor is kept
# for each of the last three distinct matrices (identified by their sparsity pattern and
# values) and the solver always checks the residual of the solved system against the
# matrix it was given.

def biot_problem(data, initial_content):
    domain = data["domain"]
    b = domain.boundaries
    model = BiotConsolidation(domain=domain.domain(), youngs_modulus=E, poisson_ratio=nu, biot_coefficient=alpha,
                              specific_storage=S, permeability=mobility, fluid_viscosity=1.0)
    problem = Problem(
        name="Cryer's sphere", geometry=domain, physics=(model,),
        boundary_conditions=(
            BiotFixedComponent(physics=model, boundary=b.plane_x, component="x"),
            BiotFixedComponent(physics=model, boundary=b.plane_y, component="y"),
            BiotFixedComponent(physics=model, boundary=b.plane_z, component="z"),
            BiotNormalTraction(physics=model, boundary=b.surface, value=-P),
            PrescribedPorePressure(physics=model, boundary=b.surface, value=0.0),
            PoreFluidFlux(physics=model, boundary=b.plane_x | b.plane_y | b.plane_z, flux=0.0),
        ),
        initial_conditions=(InitialValue(model.fluid_content, initial_content),))
    return model, problem


def numerics(method, factory):
    return Numerics(
        spatial=FiniteElement(element=Lagrange(2), boundary_enforcement=Nitsche(penalty=16)),
        linear_solver=SparseDirect(factory=factory), time_integrator=TimeMethod(method),
        nonlinear_solver=Newton(linear_solver=SparseDirect(factory=factory), jacobian="analytic",
                                relative_tolerance=1e-6, absolute_tolerance=1e-10))


def response_operators(data, space):
    """Linear functionals on the nodal coefficients: fluid volume, surface outflow, point values."""
    volume = np.zeros(space.dof_count)
    for cell, dofs in enumerate(space.cell_dofs):
        _, weights, (basis, _, _) = space.element_data(cell)
        np.add.at(volume, dofs, basis.T @ weights)
    outflow = np.zeros(space.dof_count)
    for corners in data["mesh"].facets(data["domain"].boundaries.surface):
        cell, _, weights, normal, _, gradient = space.trace_data(corners)
        normal = np.broadcast_to(normal, (len(weights), 3))
        np.add.at(outflow, space.cell_dofs[cell], -mobility * np.einsum("q,qnd,qd->n", weights, gradient, normal))
    nodes = np.asarray(space.dof_coordinates)
    surface = np.array([[a, 0, 0], [0, a, 0], [a, a, a] / np.sqrt(3)])
    node_of = lambda point: int(np.argmin(np.linalg.norm(nodes - point, axis=1)))
    return dict(volume=volume, outflow=outflow, centre=node_of(np.zeros(3)),
                surface=[node_of(point) for point in surface], normals=surface / a)


def run_schedule(data, schedule, keep_fields=False):
    """Run the constant-step segments one after another and record the response at every step."""
    factory = ReusedFactorization(capacity=3, significant_digits=7)
    start, content, operators = 0.0, 0.0, None
    record = dict(T=[], centre=[], volume=[], outflow=[], u_surface=[], method=[], fields={})
    clock = time.time()
    for method, step, end in schedule:
        model, problem = biot_problem(data, content)
        result = Simulation(problem=problem, mesh=data["mesh"], numerics=numerics(method, factory)).run(
            start_time=start * t_c, end_time=end * t_c, time_step=step * t_c)
        pressure, zeta = result.field(model.pore_pressure), result.field(model.fluid_content)
        displacement = result.field(model.displacement)
        space = zeta.space
        if operators is None:
            operators = response_operators(data, space)
        times = np.asarray(result.times)
        for k, t in enumerate(times):
            if k == 0 and record["T"]:
                continue
            p_nodes = np.asarray(pressure.coefficients_at(t), dtype=float)
            u_nodes = np.asarray(displacement.coefficients_at(t), dtype=float).reshape(space.dof_count, 3)
            record["T"].append(t / t_c)
            record["centre"].append(p_nodes[operators["centre"]])
            record["volume"].append(operators["volume"] @ np.asarray(zeta.coefficients_at(t), dtype=float))
            record["outflow"].append(operators["outflow"] @ p_nodes)
            record["u_surface"].append(np.mean(np.einsum("qi,qi->q", u_nodes[operators["surface"]], operators["normals"])))
            record["method"].append(method)
            if keep_fields and np.min(np.abs(np.asarray(FIGURE_TIMES) - t / t_c)) < 1e-9:
                record["fields"][round(t / t_c, 6)] = p_nodes
        final = np.array(zeta.coefficients_at(times[-1]), dtype=float)
        content = (lambda values, sp: lambda x, y, z: np.asarray(sp.evaluate(
            values, np.column_stack((np.ravel(x), np.ravel(y), np.ravel(z)))), dtype=float).ravel())(final, space)
        start = end
    record = {k: (np.asarray(v) if k not in ("fields", "method") else v) for k, v in record.items()}
    record.update(space=space, seconds=time.time() - clock, factorizations=factory.factorizations,
                  steps=len(record["T"]) - 1)
    return record


# ## Solve
#
# The full history $0\le T=ct/a^2\le1$ on the three meshes; on the coarsest mesh a second
# run with every step halved (time-step study); and, on the coarsest mesh up to $T=0.03$,
# Crank-Nicolson from $t=0$ with a constant step $\Delta T=2\cdot10^{-3}$ once without and
# once with a backward-Euler start (four quarter steps), to show why the start is needed.

runs = {}
for n in MESH_LEVELS:
    runs[n] = run_schedule(meshes[n], SCHEDULE, keep_fields=(n == MESH_LEVELS[-1]))
    print(f"n = {n}: {runs[n]['steps']} steps, {runs[n]['factorizations']} LU factorizations, "
          f"{runs[n]['seconds']:.0f} s")
halved = tuple((method, step / 2, end) for method, step, end in SCHEDULE)
fine_steps = run_schedule(meshes[MESH_LEVELS[0]], halved)
print(f"n = {MESH_LEVELS[0]}, all steps halved: {fine_steps['steps']} steps, {fine_steps['seconds']:.0f} s")
cn_plain = run_schedule(meshes[MESH_LEVELS[0]], (("crank_nicolson", 2e-3, 0.03),))
cn_started = run_schedule(meshes[MESH_LEVELS[0]], (("backward_euler", 5e-4, 2e-3), ("crank_nicolson", 2e-3, 0.03)))
print(f"Crank-Nicolson start comparison: {cn_plain['seconds'] + cn_started['seconds']:.0f} s")


# ## Comparison with the reference
#
# **Exact solution.** For the sphere the Biot equations reduce to (Cryer 1963; with storage
# $S$ kept) $\partial_t p-c\nabla^2p=-\eta\,\mathrm d\langle p\rangle/\mathrm dt$, where
# $\langle p\rangle$ is the volume mean of the pore pressure,
# $c=(\kappa/\mu_f)/(S+\alpha^2/M_c)$, $M_c=\lambda+2\mu$,
# $\eta=4\mu\alpha\beta/(3K)$ and $\beta=\alpha/(M_cS+\alpha^2)$. With $p=0$ at $r=a$ and
# $p=p_0$ at $t=0^+$ the Laplace transform gives, in $T=ct/a^2$,
#
# $$\frac{p(r,T)}{p_0}=\sum_{k}A_k\Big(1-\frac{\sin(\xi_k r/a)}{(r/a)\sin\xi_k}\Big)e^{-\xi_k^2T},\qquad
#   (1+\eta)\,\xi^2\sin\xi=3\eta\,(\sin\xi-\xi\cos\xi),$$
#
# with $A_k=(1+\eta)/(s_kD'(s_k))$, $s_k=-\xi_k^2$, $D(s)=1+\eta-3\eta(\sqrt s\coth\sqrt s-1)/s$.
# The radial surface displacement is $u_r(a)=-a\,(P-\alpha\langle p\rangle)/(3K)$ and the
# fluid content in the octant is
# $\int\zeta\,dV=\frac{\pi a^3}{6}\big[(S+\alpha^2/M_c)\langle p\rangle+\alpha f/M_c\big]$ with
# $f=-(M_c/K)P+\tfrac{4\mu\alpha}{3K}\langle p\rangle$. 400 terms are summed.

def characteristic(xi, e):
    return (1 + e) * xi**2 * np.sin(xi) - 3 * e * (np.sin(xi) - xi * np.cos(xi))


def exact(T, e=eta, r=None, terms=400):
    """Centre pressure p(0,T)/p0, mean <p>/p0 and, if r is given, p(r,T)/p0 (rows: r, columns: T)."""
    grid = np.linspace(1e-6, (terms + 2) * np.pi, 200 * (terms + 2))
    values = characteristic(grid, e)
    xi = np.array([brentq(characteristic, grid[i], grid[i + 1], args=(e,), xtol=1e-15)
                   for i in np.flatnonzero(values[:-1] * values[1:] < 0)[:terms]])
    cot, dcot = 1 / np.tan(xi), -1 / np.sin(xi) ** 2
    g = (1 - xi * cot) / xi**2
    dD_ds = -3 * e * ((-(cot + xi * dcot)) / xi**2 - 2 * g / xi) / (-2 * xi)
    A = (1 + e) / (-xi**2 * dD_ds)
    decay = np.exp(-np.outer(np.atleast_1d(T), xi**2))
    centre, mean = decay @ (A * (1 - xi / np.sin(xi))), decay @ (A * (1 - 3 * g))
    if r is None:
        return centre, mean
    rr = np.maximum(np.atleast_1d(r) / a, 1e-12)[:, None]
    return centre, mean, (A * (1 - np.sin(xi * rr) / (rr * np.sin(xi)))) @ decay.T


def richardson(values, sizes):
    estimate = convergence(values, sizes)
    return (estimate.order, estimate.extrapolated) if estimate.order is not None else (np.nan, values[-1])


fine = runs[MESH_LEVELS[-1]]
T_dense = np.linspace(0.03, 0.12, 9001)
centre_dense, _ = exact(T_dense)
peak_exact, T_peak = centre_dense.max(), T_dense[np.argmax(centre_dense)]
centre_cryer, _ = exact(T_dense, e=eta_cryer)
print("Centre pore pressure: exact series (Cryer 1963, with the water compressibility kept)")
print(f"   peak p/p0 = {peak_exact:.5f} at T = {T_peak:.4f}  (incompressible constituents, S = 0: "
      f"{centre_cryer.max():.5f} at T = {T_dense[np.argmax(centre_cryer)]:.4f})")


def fe_peak(run):
    """Peak of the computed centre pressure: parabola through the largest sample and its neighbours."""
    k = int(np.argmax(run["centre"]))
    Tk, pk = run["T"][k - 1:k + 2], run["centre"][k - 1:k + 2] / p0
    c2, c1, c0 = np.polyfit(Tk, pk, 2)
    return c0 - c1**2 / (4 * c2), -c1 / (2 * c2)


for n in MESH_LEVELS:
    peak, T_at = fe_peak(runs[n])
    ex, _ = exact(runs[n]["T"][1:])
    err = np.abs(runs[n]["centre"][1:] / p0 - ex)
    print(f"   n = {n}: peak {peak:.5f} at T = {T_at:.4f}  (peak deviation {100 * (peak / peak_exact - 1):+.3f} %, "
          f"time {100 * (T_at / T_peak - 1):+.2f} %); largest |p_c/p0 - exact| for T >= 0.005: "
          f"{err[runs[n]['T'][1:] >= 0.005].max():.2e}")
peak_h, _ = fe_peak(fine_steps)
print(f"   n = {MESH_LEVELS[0]} with all steps halved: peak {peak_h:.5f}; change against the standard steps "
      f"{100 * (peak_h / fe_peak(runs[MESH_LEVELS[0]])[0] - 1):+.4f} %")

print("Mesh study (h ~ 1/n) of the centre pressure")
sizes = [1.0 / n for n in MESH_LEVELS]
for label, T_probe in (("at the peak", None), ("at T = 0.03", 0.03), ("at T = 0.25", 0.25)):
    if T_probe is None:
        vals = [fe_peak(runs[n])[0] for n in MESH_LEVELS]
        reference = peak_exact
    else:
        vals = [float(np.interp(T_probe, runs[n]["T"], runs[n]["centre"])) / p0 for n in MESH_LEVELS]
        reference = float(exact(T_probe)[0][0])
    p_obs, extrap = richardson(vals, sizes)
    print(f"   {label:12s}: " + "  ".join(f"{v:.6f}" for v in vals) + f"; exact {reference:.6f}; observed order "
          f"{p_obs:.2f}, Richardson {extrap:.6f}; change between the last two meshes {100 * abs(vals[2] / vals[1] - 1):.4f} %")

print("Start of Crank-Nicolson with a constant step 0.002 (n = 3), deviation from the exact centre pressure")
for name, run in (("without backward-Euler start", cn_plain), ("with four backward-Euler quarter steps", cn_started)):
    sel = run["T"] >= 2e-3
    dev = run["centre"][sel] / p0 - exact(run["T"][sel])[0]
    print(f"   {name:40s}: largest deviation {np.abs(dev).max():.4f}, at the last step {dev[-1]:+.4f}")

_, mean_fine = exact(fine["T"])
u_exact = -a * (P - alpha * p0 * mean_fine) / (3 * K)
zeta_exact = np.pi * a**3 / 6 * ((S + alpha**2 / M_c) * p0 * mean_fine
                                 + alpha * (-(M_c / K) * P + 4 * mu * alpha / (3 * K) * p0 * mean_fine) / M_c)
print("Surface displacement and fluid volume on the finest mesh")
for T_probe in (0.03, 0.065, 0.25, 1.0):
    k = int(np.argmin(np.abs(fine["T"] - T_probe)))
    print(f"   T = {fine['T'][k]:.3f}: u_r(a) = {1e3 * fine['u_surface'][k]:.4f} mm (exact {1e3 * u_exact[k]:.4f} mm, "
          f"{100 * (fine['u_surface'][k] / u_exact[k] - 1):+.3f} %); fluid content {1e3 * fine['volume'][k]:.4f} L "
          f"(exact {1e3 * zeta_exact[k]:.4f} L, {100 * (fine['volume'][k] / zeta_exact[k] - 1):+.3f} %)")

# Fluid balance: change of the stored fluid against the outflow through the drained surface,
# integrated in time with the rule of each step (backward Euler: end value; Crank-Nicolson:
# trapezoid). The outflow is the flux of the recovered Darcy velocity through the surface.
q, T = fine["outflow"], fine["T"]
weights = np.where(np.array(fine["method"][1:]) == "backward_euler", q[1:], 0.5 * (q[1:] + q[:-1]))
expelled = np.r_[0.0, np.cumsum(weights * np.diff(T) * t_c)]
print("Fluid balance on the finest mesh (volumes per octant)")
for T_probe in (0.03, 0.065, 0.25, 1.0):
    k = int(np.argmin(np.abs(T - T_probe)))
    print(f"   T = {T[k]:.3f}: loss of stored fluid {-1e3 * fine['volume'][k]:.4f} L, time-integrated outflow "
          f"{1e3 * expelled[k]:.4f} L, imbalance {100 * (expelled[k] / -fine['volume'][k] - 1):+.2f} %")


# ## Figure
#
# Top: pore pressure on the three symmetry planes of the octant (shown mirrored to
# $x,y,z\le0$ so that the planes face the viewer) at four times; the drained outer surface
# stays at $p=0$ while the core pressure first rises above $p_0$ and then decays. Bottom: the
# centre pressure against the exact series, the radial profiles along three directions,
# the error history on the three meshes, and the Crank-Nicolson start.

space = fine["space"]
radius, angle = np.meshgrid(np.linspace(0.0, 0.999 * a, 40), np.linspace(0.0, np.pi / 2, 31), indexing="ij")
plane_x, plane_y = radius * np.cos(angle), radius * np.sin(angle)
plane_points = np.column_stack((plane_x.ravel(), plane_y.ravel(), np.zeros(plane_x.size)))
T_line, T_short = np.geomspace(1e-4, 1.0, 600), np.linspace(1e-4, 0.03, 300)
r_line, r_probe = np.linspace(0, a, 200), np.linspace(0, 0.999 * a, 15)
directions = (np.array([1, 0, 0]), np.array([1, 1, 0]) / np.sqrt(2), np.ones(3) / np.sqrt(3))
profile_times = (0.006, 0.065, 0.25)
errors = {f"n = {n}": (runs[n]["T"][1:], np.abs(runs[n]["centre"][1:] / p0 - exact(runs[n]["T"][1:])[0])) for n in MESH_LEVELS}
errors[f"n = {MESH_LEVELS[0]}, steps halved"] = (fine_steps["T"][1:], np.abs(fine_steps["centre"][1:] / p0 - exact(fine_steps["T"][1:])[0]))
fig = plots.figure(2, 4, size=(20, 10.5))
for panel, T_fig in zip(list(fig)[:4], FIGURE_TIMES):
    pressure = space.evaluate(fine["fields"][round(T_fig, 6)], plane_points).reshape(plane_x.shape) / p0
    plots.heatmap(plane_x, plane_y, pressure, limits=(0.0, 1.6), contours=8, label="pore pressure p / p0", ax=panel,
                  title=f"Plane z = 0, T = c t / a² = {T_fig:g} (t = {T_fig * t_c * 60:.0f} min), "
                        f"centre p / p0 = {np.interp(T_fig, fine['T'], fine['centre']) / p0:.3f}")
plots.line(points={f"Q2 Biot, n = {n} ({len(meshes[n]['C'])} hexahedra)": (runs[n]["T"][1:], runs[n]["centre"][1:] / p0)
                   for n in (MESH_LEVELS[0], MESH_LEVELS[-1])},
           reference={"exact series (Cryer)": (T_line, exact(T_line)[0])}, hlines=1.0, vlines={"exact peak time": T_peak},
           scale="logx", xlabel="T = c t / a²", ylabel="centre pore pressure p / p0",
           title=f"Mandel-Cryer effect: peak {peak_exact:.3f} p0 at T = {T_peak:.3f}", legend="lower left", ax=fig[4])
plots.line(points={f"T = {T_fig:g}": (np.tile(r_probe / a, 3), np.concatenate(
                       [space.evaluate(fine["fields"][round(T_fig, 6)], r_probe[:, None] * d[None, :]) / p0 for d in directions]))
                   for T_fig in profile_times},
           reference={f"T = {T_fig:g}": (r_line / a, exact(T_fig, r=r_line)[2][:, 0]) for T_fig in profile_times},
           xlabel="r / a", ylabel="p / p0",
           title="Radial profiles along axis, face and body diagonal (markers), exact (dashed)", ax=fig[5])
plots.line(y=errors, styles={f"n = {MESH_LEVELS[0]}, steps halved": "dashed"}, scale="loglog", xlabel="T = c t / a²",
           ylabel="|p_c / p0 - exact|", title="Error of the centre pressure", ax=fig[6])
plots.line(y={"Crank-Nicolson from t = 0": (cn_plain["T"][1:], cn_plain["centre"][1:] / p0),
              "4 backward-Euler quarter steps, then Crank-Nicolson": (cn_started["T"][1:], cn_started["centre"][1:] / p0)},
           reference={"exact series": (T_short, exact(T_short)[0])}, markers=True, xlabel="T = c t / a²",
           ylabel="centre pore pressure p / p0", title="Starting Crank-Nicolson after a sudden load (n = 3, step 0.002)",
           ax=fig[7])
plots.show()


# ## Going further
#
# Raise Poisson's ratio to see the overshoot shrink (Cryer's solution gives about 1.31
# $p_0$ for $\nu=0.25$); make the constituents more compressible (smaller Skempton
# coefficient) to weaken the effect; replace the sphere by a long slab between rigid
# frictionless platens, Mandel's original problem; or swap the time integrator for
# `TimeMethod("dirk_sdirk2")`, an L-stable second-order scheme that needs no damping start.
