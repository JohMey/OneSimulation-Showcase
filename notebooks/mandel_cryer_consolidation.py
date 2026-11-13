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

from dataclasses import dataclass, replace
from hashlib import blake2b, sha256
from importlib.metadata import version
from itertools import product
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
from collections import OrderedDict
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq

from onesim.cad.geometry.primitives import Geometry
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES
from onesim.meshing.generation.contracts.polynomial.hex import HEX27_REFERENCE, QuadraticHexahedralGeometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, Nitsche
from onesim.numerics.linear import SparseDirect
from onesim.numerics.linear.selection.execution.backends.scipy.sparse_direct_solver import ScipySparseDirectSolver
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import TimeMethod
from onesim.physics.geotechnics import (BiotConsolidation, BiotFixedComponent, BiotNormalTraction,
                                        PoreFluidFlux, PrescribedPorePressure)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# **Units.** Lengths in m, stresses and pressures in MPa, time in hours. This is a consistent
# rescaling of the SI equations (every coefficient is converted) that keeps the entries of
# the coupled matrices within a few orders of magnitude of each other.

hour = 3600.0
a = 1.0                                   # m, sphere radius
P = 0.1                                   # MPa, surface load (100 kPa)
E, nu, alpha = 10.0, 0.0, 1.0             # MPa, -, -
S = 0.4 / 2.2e3                           # 1/MPa, porosity / water bulk modulus
mobility = 1e-14 / 1e-3 * 1e6 * hour      # kappa/mu_f in m^2/(MPa h)

mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
M_c = lam + 2 * mu                        # oedometric (constrained) modulus
K = lam + 2 * mu / 3                      # drained bulk modulus
beta = alpha / (M_c * S + alpha**2)
eta = 4 * mu * alpha * beta / (3 * K)     # coupling parameter of the sphere solution
c = mobility / (S + alpha**2 / M_c)       # consolidation coefficient, m^2/h
t_c = a**2 / c                            # consolidation time, h
p0 = beta * M_c * P / (K + 4 * mu * alpha * beta / 3)   # undrained pore pressure at t = 0+
eta_cryer = 4 * mu / (3 * K)              # the same parameter for incompressible constituents (S = 0)
print(f"consolidation coefficient c = {c:.4f} m^2/h, a^2/c = {t_c:.3f} h, "
      f"undrained pore pressure p0 = {1e3 * p0:.3f} kPa (= {p0 / P:.5f} P), eta = {eta:.5f}")

# Time schedule in units of the consolidation time: (method, step, end of segment).
SCHEDULE = (("backward_euler", 2.5e-4, 1e-3),
            ("crank_nicolson", 5e-4, 6e-3), ("crank_nicolson", 2e-3, 0.03),
            ("crank_nicolson", 5e-3, 0.15), ("crank_nicolson", 2e-2, 0.5), ("crank_nicolson", 5e-2, 1.0))
FIGURE_TIMES = (0.006, 0.03, 0.065, 0.25)  # dimensionless times of the field plots


# ## Geometry and mesh
#
# The octant is covered by four blocks of 27-node hexahedra: the cube $[0,b]^3$ with
# $b=0.55\,a$ ($n^3$ cells) and three blocks ($n\times n\times n$ cells) that connect each outer
# face of the cube to the part of the sphere onto which that face projects from the centre.
# Inside a block a node is placed on the straight line from the cube point to its
# projection on the sphere, so the outer nodes, including the mid-edge and mid-face nodes,
# lie exactly on the sphere. The radial cells shrink towards the drained surface (last cell
# half the first), where the early-time pressure boundary layer forms. The mesh level $n$
# sets the number of cells along every block edge; $n=3,4,6$ give 108, 256 and 864 curved
# hexahedra.

@dataclass(frozen=True, eq=False, kw_only=True)
class SphereOctant(Geometry):
    """Octant of the sphere: spherical surface and three symmetry planes."""

    dimension = 3

    def boundary_groups(self):
        names = ("surface", "plane_x", "plane_y", "plane_z")
        return {**{name: (name,) for name in names}, "all": names}


class Hex27Mesh(StructuredHexahedralMesh):
    """27-node hexahedral mesh from explicit nodes, cells and named boundary faces."""

    def __init__(self, geometry, coordinates, cells, facets):
        self.geometry, self.shape = geometry, (len(cells),)
        data = QuadraticHexahedralGeometry(coordinates, cells)
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.mesh_id = "hex27:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        self._facets = {name: list(faces) for name, faces in facets.items()}
        self._carrier = self._build_carrier()

    def _build_carrier(self):
        carrier = super()._build_carrier()
        return replace(carrier, connectivity=replace(carrier.connectivity, entity_kind="hexahedron27"))


NODE = np.rint(2 * HEX27_REFERENCE).astype(int)      # position of each of the 27 nodes in a 3x3x3 grid


def with_midpoints(nodes):
    nodes = np.asarray(nodes, float)
    out = np.empty(2 * len(nodes) - 1)
    out[::2], out[1::2] = nodes, 0.5 * (nodes[:-1] + nodes[1:])
    return out


def graded(n, ratio):
    """n cells on [0, 1]; the last cell is `ratio` times the first."""
    sizes = ratio ** (np.arange(n) / max(n - 1, 1))
    nodes = np.r_[0.0, np.cumsum(sizes)] / sizes.sum()
    nodes[-1] = 1.0
    return nodes


def assemble_blocks(blocks):
    """Merge node grids (with mid-nodes) of several blocks into one Hex27 mesh."""
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


def exterior_faces(cells):
    """(corner vertices, cell, side) of every hexahedron face owned by one cell only."""
    owners = {}
    for cell, vertices in enumerate(cells):
        for side, face in enumerate(FACES):
            corners = tuple(int(v) for v in vertices[list(face)])
            owners.setdefault(tuple(sorted(corners)), []).append((corners, cell, side))
    return [found[0] for found in owners.values() if len(found) == 1]


def octant_mesh(n, b=0.55 * a, ratio=0.5):
    s = with_midpoints(np.linspace(0.0, 1.0, n + 1))
    blocks = [np.stack(np.meshgrid(b * s, b * s, b * s, indexing="ij"), -1)]
    rho = with_midpoints(graded(n, ratio))
    U, V = np.meshgrid(s, s, indexing="ij")
    for k in range(3):
        cube = np.zeros(U.shape + (3,))
        cube[..., k], cube[..., (k + 1) % 3], cube[..., (k + 2) % 3] = b, b * U, b * V
        sphere = a * cube / np.linalg.norm(cube, axis=-1)[..., None]
        blocks.append(cube[:, :, None, :] + rho[None, None, :, None] * (sphere - cube)[:, :, None, :])
    X, C = assemble_blocks(blocks)
    faces = exterior_faces(C)
    groups = {name: [] for name in ("surface", "plane_x", "plane_y", "plane_z")}
    for corners, _, _ in faces:
        on_plane = [k for k in range(3) if np.all(np.abs(X[list(corners), k]) < 1e-12)]
        groups["plane_" + "xyz"[on_plane[0]] if on_plane else "surface"].append(corners)
    domain = SphereOctant()
    return dict(n=n, domain=domain, mesh=Hex27Mesh(domain, X, C, groups), X=X, C=C, faces=faces, groups=groups)


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

class ReusedSuperLU:
    """Linear-solver factory: SuperLU whose LU factor is reused while the matrix is unchanged."""

    def __init__(self):
        self.factors, self.factorizations = OrderedDict(), 0

    def create_components(self, declaration, *, context=None):
        return BoundReusedSuperLU(self, ScipySparseDirectSolver(), declaration.component_configuration())


class BoundReusedSuperLU:
    def __init__(self, owner, provider, configuration):
        self.owner, self.provider, self.configuration = owner, provider, configuration

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def factor(self, matrix):
        carrier = getattr(matrix, "matrix", matrix)
        offsets, indices, values = (carrier.csr_components() if hasattr(carrier, "csr_components")
                                    else (lambda m: (m.indptr, m.indices, m.data))(carrier.tocsr()))
        key = blake2b(np.ascontiguousarray(offsets).tobytes() + np.ascontiguousarray(indices).tobytes()
                      + np.asarray(values, dtype=np.float32).tobytes(), digest_size=16).hexdigest()
        factors = self.owner.factors
        if key in factors:
            factors.move_to_end(key)
        else:
            factors[key] = self.provider.factorize(matrix)
            self.owner.factorizations += 1
            if len(factors) > 3:
                factors.popitem(last=False)
        return factors[key]

    def solve_configured(self, matrix, right_hand_side, configuration, **ignored):
        return self.provider.solve_configured(matrix, right_hand_side, configuration, factorization=self.factor(matrix))

    def solve_stationary(self, matrix, right_hand_side, configuration, **ignored):
        return self.solve_configured(matrix, right_hand_side, configuration)

    def solve(self, matrix, right_hand_side):
        return self.solve_configured(matrix, right_hand_side, self.configuration)


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
    outflow = np.zeros(space.dof_count)            # outflow = -mobility * (grad p . n) integrated over r = a
    for corners in data["groups"]["surface"]:
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
    factory = ReusedSuperLU()
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
                continue                               # first state of a segment = last state of the previous one
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
        # The next segment starts from this fluid content (the nodes of both runs coincide).
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
    f1, f2, f3 = values
    h1, h2, h3 = sizes
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda q: (h1**q - h2**q) / (h2**q - h3**q) - ratio, 0.3, 12.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


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
X, C = meshes[MESH_LEVELS[-1]]["X"], meshes[MESH_LEVELS[-1]]["C"]
plane_faces = [(corners, cell, side) for corners, cell, side in meshes[MESH_LEVELS[-1]]["faces"]
               if any(np.all(np.abs(X[list(corners), k]) < 1e-12) for k in range(3))]
k_sub = 5
g1 = np.linspace(0, 1, k_sub)
A_, B_ = (m.ravel() for m in np.meshgrid(g1, g1, indexing="ij"))
quads = [(i * k_sub + j, (i + 1) * k_sub + j, (i + 1) * k_sub + j + 1, i * k_sub + j + 1)
         for i in range(k_sub - 1) for j in range(k_sub - 1)]
face_points, face_basis = [], []
for corners, cell, side in plane_faces:
    ref = np.empty((k_sub * k_sub, 3))
    axis, free = side // 2, [d for d in range(3) if d != side // 2]
    ref[:, axis], ref[:, free[0]], ref[:, free[1]] = side % 2, A_, B_
    face_points.append(space.geometry.map(cell, ref))
    face_basis.append((space.cell_dofs[cell], space.basis(ref)))
norm = colors.Normalize(vmin=0.0, vmax=1.6)

fig = plt.figure(figsize=(20, 10.5), constrained_layout=True)
grid = fig.add_gridspec(2, 4, height_ratios=(1.05, 1.0))
for col, T_fig in enumerate(FIGURE_TIMES):
    ax = fig.add_subplot(grid[0, col], projection="3d")
    coeff = fine["fields"][round(T_fig, 6)] / p0
    polys, shades = [], []
    for pts, (dofs, basis) in zip(face_points, face_basis):
        vals = basis @ coeff[dofs]
        for qd in quads:
            polys.append(-pts[list(qd)])
            shades.append(vals[list(qd)].mean())
    shade = FIELD_CMAP(norm(np.asarray(shades)))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=shade, edgecolors=shade, linewidths=0.2))
    for axis in np.eye(3):                       # edges where the symmetry planes meet
        ax.plot(*np.column_stack((np.zeros(3), -axis)), color=INK, lw=1.2)
    arc = np.linspace(0, np.pi / 2, 60)
    for i, j in ((0, 1), (1, 2), (0, 2)):         # rim of the drained spherical surface
        rim = np.zeros((60, 3))
        rim[:, i], rim[:, j] = -np.cos(arc), -np.sin(arc)
        ax.plot(*rim.T, color=INK, lw=0.8)
    ax.set(xlim=(-1, 0), ylim=(-1, 0), zlim=(-1, 0), xlabel="x / m", ylabel="y / m", zlabel="z / m",
           title=f"T = c t / a$^2$ = {T_fig:g}   (t = {T_fig * t_c * 60:.0f} min)\n"
                 f"centre p / p$_0$ = {np.interp(T_fig, fine['T'], fine['centre']) / p0:.3f}")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=35)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=fig.axes[:4], shrink=0.7, pad=0.02,
             label="pore pressure p / p$_0$")

ax = fig.add_subplot(grid[1, 0])
T_line = np.geomspace(1e-4, 1.0, 600)
ax.semilogx(T_line, exact(T_line)[0], color=INK, lw=1.4, label="exact series (Cryer)")
for n, marker, color in ((MESH_LEVELS[0], "o", COPPER), (MESH_LEVELS[-1], "s", GREEN)):
    ax.semilogx(runs[n]["T"][1:], runs[n]["centre"][1:] / p0, marker, ms=3.2, color=color, mfc="none",
                label=f"Q2 Biot, n = {n} ({len(meshes[n]['C'])} hexahedra)")
ax.axhline(1.0, color="#9FA593", lw=0.8, ls="--")
ax.axvline(T_peak, color="#9FA593", lw=0.8, ls=":")
ax.set(xlabel="T = c t / a$^2$", ylabel="centre pore pressure p / p$_0$",
       title=f"Mandel-Cryer effect: peak {peak_exact:.3f} p$_0$ at T = {T_peak:.3f}")
ax.legend(fontsize=8, loc="lower left")

ax = fig.add_subplot(grid[1, 1])
r_line = np.linspace(0, a, 200)
directions = {"x axis": np.array([1, 0, 0]), "face diagonal": np.array([1, 1, 0]) / np.sqrt(2),
              "body diagonal": np.ones(3) / np.sqrt(3)}
r_probe = np.linspace(0, 0.999 * a, 15)
for T_fig, color in zip((0.006, 0.065, 0.25), (COPPER, GREEN, "#5F9175")):
    ax.plot(r_line / a, exact(T_fig, r=r_line)[2][:, 0], color=color, lw=1.3, label=f"exact, T = {T_fig:g}")
    coeff = fine["fields"][round(T_fig, 6)]
    for (name, d), marker in zip(directions.items(), ("o", "^", "s")):
        values = space.evaluate(coeff, r_probe[:, None] * d[None, :]) / p0
        ax.plot(r_probe / a, values, marker, ms=3.5, color=color, mfc="none",
                label=f"Q2 along the {name}" if T_fig == 0.006 else None)
ax.set(xlabel="r / a", ylabel="p / p$_0$", title="Radial pressure profiles along three directions (n = 6)")
ax.legend(fontsize=7.5, ncol=2)

ax = fig.add_subplot(grid[1, 2])
for n, color in zip(MESH_LEVELS, (COPPER, "#5F9175", GREEN)):
    ax.semilogx(runs[n]["T"][1:], np.abs(runs[n]["centre"][1:] / p0 - exact(runs[n]["T"][1:])[0]),
                color=color, lw=1.2, label=f"n = {n}")
ax.semilogx(fine_steps["T"][1:], np.abs(fine_steps["centre"][1:] / p0 - exact(fine_steps["T"][1:])[0]),
            color=INK, lw=0.9, ls="--", label=f"n = {MESH_LEVELS[0]}, steps halved")
ax.set_yscale("log")
ax.set(xlabel="T = c t / a$^2$", ylabel="|p$_c$ / p$_0$ - exact|", title="Error of the centre pressure")
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 3])
T_short = np.linspace(1e-4, 0.03, 300)
ax.plot(T_short, exact(T_short)[0], color=INK, lw=1.3, label="exact series")
ax.plot(cn_plain["T"][1:], cn_plain["centre"][1:] / p0, "o-", ms=3, lw=0.8, color=COPPER,
        label="Crank-Nicolson from t = 0")
ax.plot(cn_started["T"][1:], cn_started["centre"][1:] / p0, "s-", ms=3, lw=0.8, color=GREEN,
        label="4 backward-Euler quarter steps, then Crank-Nicolson")
ax.set(xlabel="T = c t / a$^2$", ylabel="centre pore pressure p / p$_0$",
       title="Starting Crank-Nicolson after a sudden load (n = 3, step 0.002)")
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Raise Poisson's ratio to see the overshoot shrink (Cryer's solution gives about 1.31
# $p_0$ for $\nu=0.25$); make the constituents more compressible (smaller Skempton
# coefficient) to weaken the effect; replace the sphere by a long slab between rigid
# frictionless platens, Mandel's original problem; or swap the time integrator for
# `TimeMethod("dirk_sdirk2")`, an L-stable second-order scheme that needs no damping start.
