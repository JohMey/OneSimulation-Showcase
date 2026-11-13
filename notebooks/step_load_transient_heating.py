"""OneSimulation example: transient heating of a disc by a heat load with a step change.
Requires the installed, activated OneSimulation runtime.

A silicon disc is heated on the central part of its top face by a uniform heat flux that
drops abruptly to a lower level after 0.25 s. The 3D temperature field is computed with
quadratic 27-node hexahedral Lagrange elements on an O-grid of the full disc and integrated
in time with an adaptive IMEX-SDIRK2 scheme. It is graded against the exact Fourier-Bessel
series of the same linear problem, the global energy balance and a three-level mesh study,
and the effect of the time-step control (tolerance, fixed steps, alignment with the load
step) is measured against the series.
"""


# # Transient heating with a step change in the load
#
# A cylindrical sample of radius $R$ and thickness $H$ is heated over a central circular
# spot of radius $a$ on its top face. The absorbed flux is uniform on the spot, high at
# first and then stepped down to a lower value at $t_1$:
#
# $$q(r,t)=\begin{cases}q_1, & t<t_1\\ q_2, & t\ge t_1\end{cases}\qquad (r<a),\qquad q=0\ (r>a).$$
#
# The load therefore jumps in space (at the spot edge $r=a$) and in time (at $t=0$ and at
# $t_1$). Such loads are the classic test of how a transient solution depends on the mesh
# near the jump and on the time-step control: the error concentrates at the jumps and is
# then smoothed out by diffusion. The temperature obeys the heat equation
#
# $$\rho c_p\,\partial_t T=\nabla\cdot(k\nabla T)$$
#
# with constant silicon properties. The whole top face loses heat by radiation to the
# surroundings at $T_a$, written in its linearised form
# $-k\,\partial_n T = h_r\,(T-T_a)$ with $h_r=4\varepsilon\sigma T_a^3$ (at the
# temperatures reached here the radiative loss is below 0.1 % of the absorbed power, so
# the linearisation is immaterial, but it keeps the problem linear so that its exact
# solution is available). Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | heated spot, top face $r<a$ | absorbed flux $q(t)$ and radiative loss $h_r(T-T_a)$ |
# | rest of the top face, $a<r<R$ | radiative loss $h_r(T-T_a)$ |
# | bottom face | thermally insulated (zero flux) |
# | cylindrical side face | thermally insulated (zero flux) |
#
# The disc starts at $T_a$. The temperature depends on $r$ and $z$ (and the full 3D disc
# is computed, without using the axial symmetry).
#
# **Numerical method.**
# - Space: continuous Galerkin finite elements with quadratic **27-node hexahedral
#   Lagrange elements (Q2)**, isoparametric (the circular arcs are quadratic), on a
#   block-structured O-grid of the whole disc: a centre block, a ring of four blocks up to
#   the spot edge $r=a$ (a mesh circle, so the spatial jump of the load lies on element
#   faces) and a second ring out to $R$. The mesh is graded towards the spot edge and
#   towards the top face. OneSimulation assembles the consistent mass matrix $M$, the
#   stiffness matrix $K$ (conduction plus the radiative Robin term) and the load vectors.
# - Time: the semi-discrete system $M\dot\theta = f(t)-K\theta$, $\theta=T-T_a$, is
#   integrated with OneSimulation's **adaptive IMEX-SDIRK2** integrator (the additive
#   ARS(2,2,2) Runge-Kutta pair: L-stable, stiffly accurate SDIRK for the stiff conduction
#   operator, explicit stages for the time-dependent load), with an embedded first-order
#   error estimate and step-size control on the maximum nodal temperature error.
# - Linear solver: each implicit stage solves $(M+\gamma\Delta t\,K)\,\theta=M\,b$ by
#   conjugate gradients with Jacobi preconditioning (relative residual $10^{-12}$).
#
# **Reference.** The linear problem has an exact solution: the Fourier-Bessel series in
# $J_0(\lambda_m r)\cos(\mu_n z)$ with $J_1(\lambda_m R)=0$ and
# $\mu_n\tan(\mu_n H)=h_r/k$, superposed for the two load steps (Duhamel's principle for
# a piecewise-constant flux; H. S. Carslaw and J. C. Jaeger, *Conduction of Heat in
# Solids*, 2nd ed., Oxford 1959, Sec. 1.14 and Ch. VIII). The notebook compares the
# computed temperatures pointwise with the series, checks the global energy balance
# against the exactly integrated absorbed and radiated energy, and runs a three-level
# mesh study (observed order and Richardson estimate).

from dataclasses import dataclass
from hashlib import sha256
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
from matplotlib import cm, colors
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from scipy.special import j0, j1, jn_zeros

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES
from onesim.meshing.generation.contracts.polynomial.hex import HEX27_REFERENCE, QuadraticHexahedralGeometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time.methods.backends.imex_sdirk2 import AdaptiveIMEXSDIRK2Provider, IMEXSDIRK2Options
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Silicon at 300 K: $k=148$ W/(m K), $\rho=2330$ kg/m³, $c_p=712$ J/(kg K)
# (F. P. Incropera et al., *Fundamentals of Heat and Mass Transfer*, 6th ed., Table A.1).
# Disc of 20 mm diameter and 4 mm thickness, heated spot of 8 mm diameter. Emissivity
# 0.7, surroundings at 293.15 K.

mm = 1e-3
R, H, a = 10.0 * mm, 4.0 * mm, 4.0 * mm       # m: disc radius, thickness, spot radius
k, rho, cp = 148.0, 2330.0, 712.0             # W/(m K), kg/m^3, J/(kg K)
alpha = k / (rho * cp)
T_a = 293.15                                  # K, surroundings and initial temperature
sigma, emissivity = 5.670e-8, 0.7
h_r = 4 * emissivity * sigma * T_a**3         # W/(m^2 K), linearised radiation
q1, q2, t1 = 2.0e6, 0.5e6, 0.25               # W/m^2, W/m^2, s: load levels and step time
t_end = 2.0                                   # s


def load(t):
    return q1 if t < t1 else q2


print(f"alpha = {alpha:.4e} m^2/s, h_r = {h_r:.3f} W/(m^2 K), Biot h_r H/k = {h_r * H / k:.2e}")
print(f"diffusion times: H^2/alpha = {H**2 / alpha:.3f} s, R^2/alpha = {R**2 / alpha:.3f} s")


# ## Geometry and mesh
#
# OneSimulation's CAD cylinder is meshed by its structured meshers either with an
# annulus (hole on the axis) or with cylindrical control volumes, so the full disc with an
# embedded spot edge is built directly as a block-structured 27-node hexahedral mesh.
# In the plane: a centre block (a slightly bulged square), four blocks from the square to
# the circle $r=a$, four blocks from $r=a$ to $R$; radially graded towards $r=a$ on both
# sides. The plane mesh is swept through the thickness with layers graded towards the top
# face (first layer four times thinner than the last). The mid-edge, mid-face and centre
# nodes come from the same block maps, so the arcs are quadratic. The boundary faces are
# sorted into the four named groups of a small `Geometry` subclass.

@dataclass(frozen=True, eq=False, kw_only=True)
class HeatedDisc(Geometry):
    """Disc with its top face split at the spot edge r = a."""

    dimension = 3

    def boundary_groups(self):
        names = ("spot", "top_ring", "bottom", "side")
        return {**{name: (name,) for name in names}, "all": names}


def graded(n, ratio):
    """n intervals on [0, 1] with last/first interval length = ratio."""
    if n == 1 or ratio == 1.0:
        return np.linspace(0.0, 1.0, n + 1)
    sizes = ratio ** (np.arange(n) / (n - 1))
    return np.concatenate(([0.0], np.cumsum(sizes) / sizes.sum()))


def with_midpoints(t):
    out = np.empty(2 * len(t) - 1)
    out[::2], out[1::2] = t, 0.5 * (t[1:] + t[:-1])
    return out


def plane_blocks(n, m1, m2, square=0.45, bulge=0.12):
    """Node lattices (with mid-nodes) of the nine plane blocks."""
    s = square * a
    u = with_midpoints(np.linspace(0.0, 1.0, n + 1))
    U, V = np.meshgrid(2 * u - 1, 2 * u - 1, indexing="ij")
    blocks = [(s * U * (1 + bulge * (1 - V**2)), s * V * (1 + bulge * (1 - U**2)))]
    r1 = with_midpoints(graded(m1, 1 / 3.0))      # square -> spot edge, finer at r = a
    r2 = with_midpoints(graded(m2, 6.0))          # spot edge -> rim, finer at r = a
    t = 2 * u - 1
    angle = t * np.pi / 4
    xi, yi = s * (1 + bulge * (1 - t**2)), s * t
    xa, ya = a * np.cos(angle), a * np.sin(angle)
    radius = a + (R - a) * r2
    inner = (xi[:, None] + (xa - xi)[:, None] * r1, yi[:, None] + (ya - yi)[:, None] * r1)
    outer = (radius * np.cos(angle)[:, None], radius * np.sin(angle)[:, None])
    for quarter in range(4):
        c, sn = np.cos(quarter * np.pi / 2), np.sin(quarter * np.pi / 2)
        for X, Y in (inner, outer):
            blocks.append((c * X - sn * Y, sn * X + c * Y))
    return blocks


class DiscMesh(StructuredHexahedralMesh):
    """Hex27 O-grid of the disc, passed to OneSimulation as a native hexahedral mesh."""

    def __init__(self, geometry, n, m1, m2, nz, top_ratio=4.0):
        self.geometry = geometry
        z_nodes = with_midpoints(H * graded(nz, 1 / top_ratio))
        reference = np.rint(2 * HEX27_REFERENCE).astype(int)
        index, coordinates, cells = {}, [], []
        for X, Y in plane_blocks(n, m1, m2):
            ids = np.empty((*X.shape, len(z_nodes)), dtype=np.int64)
            for i, j, l in product(range(X.shape[0]), range(X.shape[1]), range(len(z_nodes))):
                point = (X[i, j], Y[i, j], z_nodes[l])
                key = tuple(np.rint(np.asarray(point) * 1e9).astype(np.int64))
                if key not in index:
                    index[key] = len(coordinates)
                    coordinates.append(point)
                ids[i, j, l] = index[key]
            for i, j, l in product(range((X.shape[0] - 1) // 2), range((X.shape[1] - 1) // 2), range(nz)):
                cells.append([ids[2 * i + r[0], 2 * j + r[1], 2 * l + r[2]] for r in reference])
        coordinates, cells = np.asarray(coordinates), np.asarray(cells, dtype=np.int64)
        p = coordinates[cells[:, :8]]
        signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 3] - p[:, 0], p[:, 4] - p[:, 0]))
        lookup = {tuple(r): q for q, r in enumerate(reference)}
        mirror = [lookup[(2 - r[0], r[1], r[2])] for r in reference]
        cells[signed < 0] = cells[signed < 0][:, mirror]
        data = QuadraticHexahedralGeometry(coordinates, cells)
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (n, m1, m2, nz)
        self.mesh_id = "disc-hex27:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        # 3 x 3 node grid of each of the six faces of a Hex27 cell (used for drawing)
        grids = []
        for side in range(6):
            axis, value = side // 2, 2 * (side % 2)
            others = [d for d in range(3) if d != axis]
            grid = np.empty((3, 3), dtype=int)
            for i, j in product(range(3), range(3)):
                address = [0, 0, 0]
                address[axis], address[others[0]], address[others[1]] = value, i, j
                grid[i, j] = lookup[tuple(address)]
            grids.append(grid)
        owners = {}
        for cell in self.cells:
            for face, grid in zip(FACES, grids):
                corners = tuple(cell[list(face)])
                owners.setdefault(tuple(sorted(corners)), []).append((corners, cell[grid]))
        self._facets = {name: [] for name in geometry.boundary_groups()["all"]}
        self.face_grids = []
        for matches in owners.values():
            self.face_grids.append(matches[0][1])
            matches = [m[0] for m in matches]
            if len(matches) == 1:
                q = self.coordinates[list(matches[0])]
                if np.allclose(q[:, 2], H):
                    name = "spot" if np.all(np.hypot(q[:, 0], q[:, 1]) <= a * (1 + 1e-9)) else "top_ring"
                elif np.allclose(q[:, 2], 0.0):
                    name = "bottom"
                else:
                    name = "side"
                self._facets[name].append(matches[0])
        self._carrier = self._build_carrier()

    def _build_carrier(self):
        from dataclasses import replace
        carrier = super()._build_carrier()
        return replace(carrier, connectivity=replace(carrier.connectivity, entity_kind="hexahedron27"))


# ## Model
#
# The spot carries a `ConvectiveHeatFlux` with coefficient $h_r$ and the effective
# ambient temperature $T_a+q(t)/h_r$, which gives exactly
# $-k\partial_nT=h_r(T-T_a)-q(t)$; the rest of the top face a plain radiative
# `ConvectiveHeatFlux`; bottom and side an explicit zero `HeatFlux`. OneSimulation's
# stationary solve binds and assembles the Q2 operator (its stationary solution, the
# far-off steady state, is not used); the transient integration then works on the
# assembled matrices. Because $q$ is piecewise constant, the load vector takes exactly
# two values, assembled at a time before and after the step.

silicon = IsotropicSolid(density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
                         youngs_modulus=130e9, poisson_ratio=0.28, thermal_expansion_coefficient=2.6e-6)


def assemble(level):
    geometry = HeatedDisc()
    start = time.time()
    mesh = DiscMesh(geometry, *level)
    model = HeatConduction(domain=geometry.domain(), material=silicon, heating=0.0)
    b = geometry.boundaries
    problem = Problem(name="Disc heated by a stepped spot load", geometry=geometry, physics=(model,),
                      boundary_conditions=(
                          ConvectiveHeatFlux(boundary=b.spot, physics=model, heat_transfer_coefficient=h_r,
                                             ambient_temperature=lambda t: T_a + load(t) / h_r),
                          ConvectiveHeatFlux(boundary=b.top_ring, physics=model, heat_transfer_coefficient=h_r,
                                             ambient_temperature=T_a),
                          HeatFlux(boundary=b.bottom | b.side, physics=model, value=0.0)))
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()))
    operator = simulation.solve_stationary().operators[model.model_id]
    base, system, load_before, _ = operator.weak_assembly(0.5 * t1)
    _, _, load_after, _ = operator.weak_assembly(0.5 * (t1 + t_end))
    K = system.matrix.tocsr()
    M = base.mass.matrix.tocsr()
    ones = np.ones(K.shape[0])
    # in theta = T - T_a the ambient part h_r*T_a cancels against K @ T_a
    f_before = np.asarray(load_before) - K @ (T_a * ones)
    f_after = np.asarray(load_after) - K @ (T_a * ones)
    nodes = np.asarray(operator.space.dof_coordinates)
    return dict(level=level, mesh=mesh, M=M, K=K, f=(f_before, f_after), nodes=nodes,
                seconds=time.time() - start)


# ## Solve
#
# **Time integration.** The ARS(2,2,2) pair advances $M\dot\theta=f(t)-K\theta$ with the
# load $M^{-1}f(t)$ as the explicit part and $-M^{-1}K\theta$ as the implicit part. The
# local error estimate is the difference to the embedded first-order solution, measured
# in the maximum nodal temperature; a step is accepted if it is below
# $\mathrm{atol}+\mathrm{rtol}\,\max|\theta|$ and the next step follows the usual
# controller $\Delta t\,\min(5,\max(0.2,0.9\,e^{-1/3}))$. Three ways of treating the
# load step are compared:
#
# - *adaptive, aligned*: the integration stops exactly at $t_1$ and restarts there with a
#   small step (the analogue of an explicit time event);
# - *adaptive, free*: the controller is left to find the jump by itself (it only sees the
#   load at its stage times $t_n$ and $t_n+\gamma\Delta t$);
# - *fixed step*: the same integrator with a constant step $\Delta t$ (minimum = maximum
#   step, every step accepted), once with $t_1$ on a step boundary and once between two
#   steps.
#
# All runs also land exactly on the report times 0.4 s, 1 s and 2 s. The time-step study
# runs on the coarse mesh, where its time-discretisation error is measured against a run
# with rtol $10^{-5}$; the three-level mesh study uses aligned steps with rtol
# $10^{-4}$, whose time error (see the printout) is far below the spatial differences.


class MaxNorm:
    """State algebra of the integrator: arrays, maximum nodal difference in kelvin."""

    def weighted_sum(self, states, weights):
        out = states[0] * weights[0]
        for state, weight in zip(states[1:], weights[1:]):
            out = out + state * weight
        return out

    def distance(self, a_, b_):
        return float(np.max(np.abs(a_ - b_)))


def integrator_system(data):
    M, K = data["M"], data["K"]
    jacobi = diags(1.0 / M.diagonal())

    def solve(A, rhs, guess, precond):
        x, info = cg(A, rhs, x0=guess, rtol=1e-12, atol=0.0, M=precond, maxiter=5000)
        return x

    w = tuple(solve(M, f, None, jacobi) for f in data["f"])   # M^{-1} f before / after the step
    explicit = lambda theta, t: w[0] if t < t1 - 1e-12 else w[1]
    implicit = lambda theta, t: -solve(M, K @ theta, None, jacobi)

    def stage(base, c, _rhs, _t, guess):
        A = M + c * K
        return solve(A, M @ base, guess, diags(1.0 / A.diagonal())), 1

    return explicit, implicit, stage


report_times = (0.4, 1.0, t_end)      # every run lands exactly on these times


def integrate(data, mode, rtol=1e-4, step=None, atol=1e-6):
    explicit, implicit, stage = integrator_system(data)
    if mode == "fixed":
        options = IMEXSDIRK2Options(relative_tolerance=1.0, absolute_tolerance=1.0,
                                    minimum_step=step, maximum_step=step)
    else:
        options = IMEXSDIRK2Options(relative_tolerance=rtol, absolute_tolerance=atol,
                                    minimum_step=1e-9, maximum_step=0.1)
    provider = AdaptiveIMEXSDIRK2Provider(options, algebra=MaxNorm())
    theta = np.zeros(data["K"].shape[0])
    t, dt = 0.0, (step if mode == "fixed" else 1e-5)
    times, states, rejected = [0.0], [theta], 0
    stops = sorted(report_times + ((t1,) if mode == "aligned" else ()))
    start = time.time()
    for stop in stops:
        if mode == "aligned" and abs(t - t1) < 1e-12:
            dt = 1e-5                     # restart after the load step
        while t < stop - 1e-12:
            outcome = provider.advance(theta, explicit, implicit, t, min(dt, stop - t),
                                       implicit_stage_solver=stage)
            dt = outcome.next_step if mode != "fixed" else step
            if outcome.accepted:
                theta, t = outcome.state, outcome.end_time
                times.append(t)
                states.append(theta)
            else:
                rejected += 1
    return dict(mode=mode, rtol=rtol, step=step, times=np.array(times), states=np.array(states),
                accepted=len(times) - 1, rejected=rejected, seconds=time.time() - start)


# **Exact series.** With $\lambda_0=0$, $\lambda_m$ the roots of $J_1(\lambda R)=0$ and
# $\mu_n$ the roots of $\mu\tan(\mu H)=h_r/k$,
#
# $$\theta(r,z,t)=\sum_{m,n}\frac{\alpha}{k}\,\frac{\cos(\mu_nH)\,S_m}{N_mN_n}\,
#   J_0(\lambda_mr)\cos(\mu_nz)\sum_j\Delta q_j\,\frac{1-e^{-\kappa_{mn}(t-t_j)}}{\kappa_{mn}},
#   \qquad \kappa_{mn}=\alpha(\lambda_m^2+\mu_n^2),$$
#
# with $S_m=\int_0^aJ_0(\lambda_mr)\,r\,dr$, $N_m=\int_0^RJ_0^2r\,dr$,
# $N_n=\int_0^H\cos^2(\mu_nz)\,dz$, the load increments $\Delta q=(q_1,\,q_2-q_1)$ at
# $t_j=(0,\,t_1)$. The time-independent part $\sum 1/\kappa$ converges slowly at the
# heated face (its truncation error falls like $1/N$ for $N\times N$ terms); it is summed
# with $N=4000$ and $N=8000$ and extrapolated, $2S_{8000}-S_{4000}$ (the extrapolated
# top-centre value agrees with $2S_{4000}-S_{2000}$ to $2\times10^{-5}$ K). The
# decaying part converges fast and uses the first $300\times300$ terms (neglected terms
# are below $e^{-30}$ for all times compared, which are at least 1 ms after a load jump).

def mu_roots(count):
    bi = h_r * H / k
    # the n-th root lies in (n pi, n pi + 1) because tan(1) > Bi
    return np.array([brentq(lambda x: x * np.tan(x) - bi, n * np.pi + 1e-300, n * np.pi + 1.0)
                     for n in range(count)]) / H


class Series:
    def __init__(self, n_steady=8000, n_decay=300):
        self.lam = np.r_[0.0, jn_zeros(1, n_steady - 1) / R]
        self.mu = mu_roots(n_steady)
        lam, mu = self.lam, self.mu
        S = np.where(lam > 0, a * j1(lam * a) / np.where(lam > 0, lam, 1.0), a**2 / 2)
        Nm = R**2 / 2 * j0(lam * R) ** 2
        Nn = H / 2 + np.sin(2 * mu * H) / (4 * mu)
        self.cr = S / Nm                                  # radial weights
        self.cz = alpha / k * np.cos(mu * H) / Nn         # axial weights
        self.nd = n_decay

    def steady_shape(self, r, z):
        """sum of C J0 cos / kappa per unit flux, N x N and N/2 x N/2 terms, extrapolated."""
        full, half = np.zeros(len(r)), np.zeros(len(r))
        n_half = len(self.mu) // 2
        Cz = np.cos(np.outer(z, self.mu)) * self.cz                               # points x n
        for i0 in range(0, len(self.lam), 500):
            lam = self.lam[i0:i0 + 500]
            Jr = j0(np.outer(r, lam)) * self.cr[i0:i0 + 500]                      # points x m
            kap = alpha * (lam[:, None] ** 2 + self.mu[None, :] ** 2)            # m x n
            full += np.sum((Jr @ (1.0 / kap)) * Cz, axis=1)
            if i0 < n_half:
                half += np.sum((Jr @ (1.0 / kap[:, :n_half])) * Cz[:, :n_half], axis=1)
        return 2 * full - half

    def evaluate(self, r, z, times):
        r, z = np.asarray(r, float), np.asarray(z, float)
        steady = self.steady_shape(r, z)
        lam, mu = self.lam[:self.nd], self.mu[:self.nd]
        Jr = j0(np.outer(r, lam)) * self.cr[:self.nd]
        Cz = np.cos(np.outer(z, mu)) * self.cz[:self.nd]
        kap = alpha * (lam[:, None] ** 2 + mu[None, :] ** 2)
        out = []
        for t in np.atleast_1d(times):
            value = np.zeros(len(r))
            for tj, dq in ((0.0, q1), (t1, q2 - q1)):
                if t > tj:
                    decay = np.exp(-kap * (t - tj)) / kap
                    value += dq * (steady - np.sum((Jr @ decay) * Cz, axis=1))
            out.append(value)
        return np.array(out)

    def energy(self, times):
        """rho c_p * integral of theta over the disc (only the m = 0 terms survive)."""
        mu, cz = self.mu, self.cz
        kap = alpha * mu**2
        weight = rho * cp * np.pi * R**2 * self.cr[0] * cz * np.sin(mu * H) / mu
        out = []
        for t in np.atleast_1d(times):
            e = 0.0
            for tj, dq in ((0.0, q1), (t1, q2 - q1)):
                if t > tj:
                    e += dq * np.sum(weight * -np.expm1(-kap * (t - tj)) / kap)
            out.append(e)
        return np.array(out)


def absorbed_energy(t):
    return np.pi * a**2 * (q1 * min(t, t1) + q2 * max(t - t1, 0.0))


series = Series()

levels = ((4, 4, 6, 4), (6, 6, 9, 6), (8, 8, 12, 8))     # (centre, inner ring, outer ring, layers)
meshes = []
for level in levels:
    data = assemble(level)
    meshes.append(data)
    print(f"mesh {level}: {data['mesh'].cell_count:5d} Hex27 cells, {data['K'].shape[0]:6d} nodes, "
          f"volume error {data['M'].sum() / (rho * cp) / (np.pi * R**2 * H) - 1:+.1e}  ({data['seconds']:.0f} s)")


def sample_sets(data):
    x, y, z = data["nodes"].T
    on_plane = (np.abs(y) < 1e-12) & (x > -1e-12)
    centre = int(np.argmin(x**2 + y**2 + (z - H) ** 2))
    surface = np.flatnonzero(on_plane & (np.abs(z - H) < 1e-12))
    surface = surface[np.argsort(x[surface])]
    return dict(centre=centre, surface=surface, plane=np.flatnonzero(on_plane))


for data in meshes:
    data["samples"] = sample_sets(data)

# time-step study on the coarse mesh (its time-step reference: rtol 1e-5), mesh study at rtol 1e-4
study_mesh = meshes[0]
reference_run = integrate(study_mesh, "aligned", rtol=1e-5)
print(f"time reference (coarse mesh, aligned, rtol 1e-5): {reference_run['accepted']} accepted steps "
      f"({reference_run['seconds']:.1f} s)")
runs = {}
for label, kwargs in (("adaptive aligned, rtol 1e-2", dict(mode="aligned", rtol=1e-2)),
                      ("adaptive aligned, rtol 1e-3", dict(mode="aligned", rtol=1e-3)),
                      ("adaptive aligned, rtol 1e-4", dict(mode="aligned", rtol=1e-4)),
                      ("adaptive free, rtol 1e-3", dict(mode="free", rtol=1e-3)),
                      ("fixed dt = 50 ms", dict(mode="fixed", step=0.05)),
                      ("fixed dt = 10 ms", dict(mode="fixed", step=0.01)),
                      ("fixed dt = 40 ms (t1 between steps)", dict(mode="fixed", step=0.04))):
    run = integrate(study_mesh, **kwargs)
    runs[label] = run
    print(f"{label:36s}: {run['accepted']:4d} accepted, {run['rejected']:3d} rejected steps ({run['seconds']:.1f} s)")

mesh_runs = [runs["adaptive aligned, rtol 1e-4"]] + [integrate(data, "aligned", rtol=1e-4) for data in meshes[1:]]
for data, run in zip(meshes, mesh_runs):
    print(f"mesh {data['level']}: aligned, rtol 1e-4, {run['accepted']} accepted, {run['rejected']} rejected "
          f"steps ({run['seconds']:.1f} s)")


# ## Comparison with the reference
#
# **Pointwise against the exact series.** Largest deviation of the top-centre temperature
# from the series over all accepted times at least 1 ms after a load jump, and the largest
# deviation along the whole top-surface radius at $t=t_1$ (just before the step,
# $\theta(t_1^-)$ is continuous) and at $t=0.4$ s and $t=2$ s.
#
# What to expect from the time-step study: with the steps aligned to $t_1$ the time error
# shrinks with the tolerance and is soon far below the spatial error. If a step straddles
# $t_1$, the integrator samples the load only at its stage times $t_n$ and
# $t_n+\gamma\Delta t$; the explicit ARS(2,2,2) weights are $(1-1/(2\gamma),\,1/(2\gamma))
# \approx(-0.71,\,1.71)$, so a jump between the two stage times is integrated with a wrong
# amount (even a wrong sign) of heat. The error estimate cannot see a jump it never
# samples, and because the disc is nearly insulated the wrong energy stays in the solution.
# This is why load steps are best placed on step boundaries (time events).

def accepted_times(run, gap=1e-3):
    t = run["times"]
    return np.flatnonzero(((t >= gap) & (t < t1)) | (t >= t1 + gap))


def centre_errors(data, run):
    keep = accepted_times(run)
    r0 = np.zeros(1)
    exact = series.evaluate(r0, np.full(1, H), run["times"][keep])[:, 0]
    computed = run["states"][keep, data["samples"]["centre"]]
    return run["times"][keep], computed, exact


def at_time(run, t):
    i = int(np.argmin(np.abs(run["times"] - t)))
    return run["states"][i], run["times"][i]


ref_peak = series.evaluate(np.zeros(1), np.full(1, H), [t1])[0, 0]
print(f"Exact top-centre temperature rise at t1 = {t1} s: {ref_peak:.4f} K,"
      f" at t_end = {t_end} s: {series.evaluate(np.zeros(1), np.full(1, H), [t_end])[0, 0]:.4f} K")
print("Time-step control on the coarse mesh: top-centre deviation from the series over all accepted times,")
print("and time-discretisation error = largest nodal deviation from the rtol 1e-5 run at t = 0.4, 1.0, 2.0 s")
for label, run in runs.items():
    t, computed, exact = centre_errors(study_mesh, run)
    temporal = max(np.max(np.abs(at_time(run, tr)[0] - at_time(reference_run, tr)[0])) for tr in report_times)
    print(f"  {label:36s}: vs series {np.max(np.abs(computed - exact)):.2e} K "
          f"({100 * np.max(np.abs(computed - exact)) / ref_peak:.3f} % of the peak rise), time error {temporal:.2e} K")

print("Mesh study (aligned, rtol 1e-4): top-surface radius, largest deviation from the series")
surface_err = []
for data, run in zip(meshes, mesh_runs):
    s = data["samples"]["surface"]
    r_s = data["nodes"][s, 0]
    errs = []
    for t in (t1, 0.4, t_end):
        state, t_hit = at_time(run, t)
        exact = series.evaluate(r_s, np.full(len(s), H), [t_hit])[0]
        errs.append(np.max(np.abs(state[s] - exact)))
    surface_err.append(errs)
    t, computed, exact = centre_errors(data, run)
    print(f"  mesh {str(data['level']):16s}: t = 0.25 s {errs[0]:.2e} K, t = 0.40 s {errs[1]:.2e} K, "
          f"t = 2.0 s {errs[2]:.2e} K;  top centre over time {np.max(np.abs(computed - exact)):.2e} K")

print("Energy balance: rho c_p * integral of (T - T_a) dV against the exact series")
for data, run in zip(meshes, mesh_runs):
    for t in (t1, t_end):
        state, t_hit = at_time(run, t)
        e_fe = float(np.sum(data["M"] @ state))
        e_ex = float(series.energy([t_hit])[0])
        print(f"  mesh {str(data['level']):16s} t = {t_hit:.2f} s: {e_fe:.6f} J vs exact {e_ex:.6f} J "
              f"({100 * (e_fe / e_ex - 1):+.4f} %);  absorbed {absorbed_energy(t_hit):.6f} J")


def richardson(values, refinement=(4, 6, 8)):
    f1, f2, f3 = values
    h1, h2, h3 = 1.0 / np.asarray(refinement, float)
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.2, 12.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


print("Mesh study of the top-centre temperature rise (cell counts 4 : 6 : 8 per direction)")
for t in (t1, 0.4, t_end):
    vals = []
    for data, run in zip(meshes, mesh_runs):
        state, t_hit = at_time(run, t)
        vals.append(state[data["samples"]["centre"]])
    exact = series.evaluate(np.zeros(1), np.full(1, H), [t])[0, 0]
    p, ext = richardson(vals)
    print(f"  t = {t:.2f} s: " + "  ".join(f"{v:.5f}" for v in vals)
          + f" K, order {p:.2f}, Richardson {ext:.5f} K, exact {exact:.5f} K, "
          f"change last two meshes {abs(vals[-1] - vals[-2]):.1e} K")


# ## Figure

fine, fine_run = meshes[-1], mesh_runs[-1]
cmap = FIELD_CMAP
fig = plt.figure(figsize=(15, 11), constrained_layout=True)
grid = fig.add_gridspec(3, 4)

# 3D cut-away of the disc at t = t1
state, _ = at_time(fine_run, t1)
values = state + T_a
norm = colors.Normalize(vmin=values.min(), vmax=values.max())
ax3 = fig.add_subplot(grid[0:2, 0:2], projection="3d")
xyz = fine["nodes"]
node_of = {tuple(np.rint(p * 1e9).astype(np.int64)): i for i, p in enumerate(xyz)}
mesh_to_dof = np.array([node_of[tuple(np.rint(p * 1e9).astype(np.int64))] for p in fine["mesh"].coordinates])
polys, shade = [], []
for g in fine["mesh"].face_grids:
    p = fine["mesh"].coordinates[g]                       # 3 x 3 x 3 face nodes
    xs, ys, zs = p[..., 0], p[..., 1], p[..., 2]
    in_removed_quarter = np.all(xs >= -1e-12) and np.all(ys <= 1e-12)
    outer = np.allclose(zs, H) or np.allclose(np.hypot(xs, ys), R, rtol=1e-3)
    cut = ((np.allclose(ys, 0.0, atol=1e-12) and np.all(xs >= -1e-12))
           or (np.allclose(xs, 0.0, atol=1e-12) and np.all(ys <= 1e-12)))
    if cut or (outer and not in_removed_quarter):
        for i, j in product(range(2), range(2)):          # four sub-quads per Hex27 face
            quad = [g[i, j], g[i + 1, j], g[i + 1, j + 1], g[i, j + 1]]
            polys.append(fine["mesh"].coordinates[quad] * 1e3)
            shade.append(values[mesh_to_dof[quad]].mean())
collection = Poly3DCollection(polys, facecolors=cmap(norm(np.array(shade))), edgecolors="none",
                              linewidths=0.0, antialiased=False)
ax3.add_collection3d(collection)
ax3.set(xlim=(-10, 10), ylim=(-10, 10), zlim=(0, 4), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm",
        title=f"T at t = {t1} s, cut-away (thickness drawn 1.5x)")
ax3.set_box_aspect((1, 1, 0.3))
ax3.view_init(elev=30, azim=-60)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3, shrink=0.6, label="T / K")

# r-z sections at several times
plane = fine["samples"]["plane"]
tri = Triangulation(fine["nodes"][plane, 0] * 1e3, fine["nodes"][plane, 2] * 1e3)
for col, t in enumerate((0.05, t1, 0.35, t_end)):
    ax = fig.add_subplot(grid[2, col])
    state, t_hit = at_time(fine_run, t)
    T = state[plane] + T_a
    cs = ax.tricontourf(tri, T, levels=24, cmap=cmap)
    ax.tricontour(tri, T, levels=12, colors="w", linewidths=0.4)
    ax.axvline(a / mm, color=CREAM, lw=0.8, ls="--")
    ax.set(xlabel="r / mm", ylabel="z / mm", title=f"r-z section, t = {t_hit:.2f} s", aspect="equal")
    fig.colorbar(cs, ax=ax, shrink=0.8, label="T / K")

# top-centre temperature and its deviation from the series
ax = fig.add_subplot(grid[0, 2])
t_c, computed, exact = centre_errors(fine, fine_run)
t_dense = np.r_[np.linspace(1e-3, t1, 60), np.linspace(t1 + 1e-3, t_end, 120)]
ax.plot(t_dense, series.evaluate(np.zeros(1), np.full(1, H), t_dense)[:, 0] + T_a, color=GREEN, lw=2,
        label="exact series")
ax.plot(t_c, computed + T_a, ".", color=COPPER, ms=3, label="OneSimulation, accepted steps")
ax.set(xlabel="t / s", ylabel="T at top centre / K", title="Top-centre temperature")
ax.legend(loc="lower right")
ax = fig.add_subplot(grid[0, 3])
for label, run in runs.items():
    t_c, computed, exact = centre_errors(study_mesh, run)
    ax.semilogy(t_c, np.abs(computed - exact) + 1e-9, lw=1.2, label=label)
ax.set(xlabel="t / s", ylabel="|T - T_exact| at top centre / K", title="Effect of the time-step control",
       ylim=(1e-4, 1e5))
ax.legend(fontsize=6, ncol=2, loc="upper center")

# top-surface profiles
ax = fig.add_subplot(grid[1, 2])
s = fine["samples"]["surface"]
r_s = fine["nodes"][s, 0]
r_dense = np.linspace(0, R, 200)
for t, color in ((t1, COPPER), (0.4, GREEN), (t_end, INK)):
    state, t_hit = at_time(fine_run, t)
    ax.plot(r_dense * 1e3, series.evaluate(r_dense, np.full(200, H), [t_hit])[0] + T_a, color=color, lw=1.5)
    ax.plot(r_s * 1e3, state[s] + T_a, "o", color=color, ms=3, label=f"t = {t_hit:.2f} s")
ax.axvline(a / mm, color="#9FA593", lw=0.8, ls="--")
ax.set(xlabel="r / mm", ylabel="T on the top face / K", title="Top face: nodes (dots) vs series (lines)")
ax.legend()

# mesh convergence
ax = fig.add_subplot(grid[1, 3])
cells = [d["level"][0] for d in meshes]
for i, t in enumerate((t1, 0.4, t_end)):
    ax.loglog(cells, [e[i] for e in surface_err], "o-", label=f"top face, t = {t:.2f} s")
ax.set(xlabel="cells per side of the centre block", ylabel="max |T - T_exact| on the top face / K",
       title="Mesh convergence against the series")
ax.set_xticks(cells, [str(c) for c in cells])
ax.minorticks_off()
ax.legend()
plt.show()


# ## Going further
#
# Replace the linearised radiation by the full $\varepsilon\sigma(T^4-T_a^4)$ law (a
# Newton or chord iteration inside the implicit stages) and raise the load until the
# loss matters; add temperature-dependent silicon properties; replace the step by a
# train of pulses and use the aligned restarts at every switching time; or compare the
# Q2 hexahedra with quadratic tetrahedra on the same disc at equal node count.
