"""OneSimulation example: viscous flow in a journal bearing (Wannier's eccentric rotating cylinders).
Requires the installed, activated OneSimulation runtime.

A journal rotates inside a fixed, eccentric bearing shell. The two-dimensional flow in the
crescent-shaped gap is computed with Taylor-Hood finite elements on a curved, boundary-fitted
bipolar grid and graded against Wannier's exact Stokes solution: the velocity field, the torque,
the load on the journal, the flow rate, the wall pressure and the recirculation eddy in the wide
gap, with a three-level mesh study, a second eccentricity, the narrow-gap limit of lubrication
theory (Sommerfeld) and a run at a moderate Reynolds number.
"""


# # Journal bearing: flow between eccentric rotating cylinders
#
# A journal of radius $R_1$ rotates with angular velocity $\Omega$ inside a fixed bearing shell
# of radius $R_2$ whose centre is displaced by the eccentricity $e$. The viscous fluid in the
# crescent-shaped gap is dragged round by the journal; because the gap narrows on one side, a
# pressure field builds up that pushes the journal sideways - the **load-carrying mechanism of a
# hydrodynamic bearing**. Along the cylinders nothing changes, so the flow is two-dimensional: it
# depends on both coordinates of the cross-section, $(x, y)$.
#
# For slow, viscous flow (Stokes flow, Reynolds number $\to0$) the problem has an **exact
# solution** found by G. H. Wannier, "A contribution to the hydrodynamics of lubrication",
# *Quarterly of Applied Mathematics* 8, 1-32 (1950), building on G. B. Jeffery, *Proc. R. Soc.
# London A* 101, 169-174 (1922); see also B. Y. Ballal and R. S. Rivlin, "Flow of a Newtonian
# fluid between eccentric rotating cylinders: inertial effects", *Arch. Ration. Mech. Anal.* 62,
# 237-294 (1976). In **bipolar coordinates** $(\xi,\eta)$,
#
# $$x=\frac{c\,\sinh\xi}{\cosh\xi-\cos\eta},\qquad y=\frac{c\,\sin\eta}{\cosh\xi-\cos\eta},$$
#
# both cylinders are coordinate lines, $\xi=\xi_1$ (journal) and $\xi=\xi_2$ (bearing). The
# stream function ($u=\partial_y\psi$, $v=-\partial_x\psi$) is
#
# $$\psi=\frac{c\,F(\xi,\eta)}{\cosh\xi-\cos\eta},\quad
#   F=A\cosh\xi+B\,\xi\cosh\xi+C\sinh\xi+D\,\xi\sinh\xi
#     +\left(E\cosh2\xi+G\sinh2\xi+H+K\xi\right)\cos\eta ,$$
#
# a biharmonic function for any constants. The eight constants and the flow rate $Q$ through the
# gap follow from no slip on both walls ($\psi=Q$ and $\partial_n\psi=-\Omega R_1$ on the
# journal, $\psi=\partial_n\psi=0$ on the bearing) and from the requirement that the pressure is
# single-valued around the journal.
#
# **Lubrication limit.** For a narrow clearance $C_r=R_2-R_1\ll R_1$ and relative eccentricity
# $\varepsilon=e/C_r$, Sommerfeld's solution of the Reynolds equation gives the torque and the
# load per unit length
#
# $$T=\frac{4\pi\mu\Omega R_1^3}{C_r}\,\frac{1+2\varepsilon^2}{(2+\varepsilon^2)\sqrt{1-\varepsilon^2}},
#   \qquad
#   W=\frac{12\pi\mu\Omega R_1^3}{C_r^2}\,\frac{\varepsilon}{(2+\varepsilon^2)\sqrt{1-\varepsilon^2}},$$
#
# with the load perpendicular to the line of centres (A. Sommerfeld, *Z. Math. Phys.* 50, 97-155,
# 1904; A. Cameron, *Basic Lubrication Theory*, Ellis Horwood 1981, ch. 5). In terms of the
# Sommerfeld number $S=(R_1/C_r)^2\,\mu N/P$ with $N=\Omega/2\pi$ and the projected pressure
# $P=W/(2R_1)$ this is the classical relation $S=(2+\varepsilon^2)\sqrt{1-\varepsilon^2}/(12\pi^2\varepsilon)$.
#
# **Numerical method.**
#
# * Spatial: continuous Galerkin finite elements, Taylor-Hood P2/P1 (quadratic velocity, linear
#   pressure) on **curved, isoparametric ten-node tetrahedra**. The full 3D Navier-Stokes
#   equations are solved on a slab one cell thick whose two faces normal to $z$ form a periodic
#   pair, so the computed flow depends on $(x, y)$ only.
# * Mesh: a **boundary-fitted bipolar grid**: grid lines $\xi=$ const are circles, so both
#   cylinders are exact grid lines, and every edge on them carries its mid-node on the exact
#   circle. The grid is naturally graded: every gap width - from the narrow to the wide side - is
#   crossed by the same number of cells.
# * Solver: stationary Newton iteration with a sparse direct LU solver.
# * Loads: torque and force are taken from the weak (volume) form of the momentum balance - the
#   reaction of the discrete equations - which converges faster than integrating the wall
#   traction.
#
# **Reference.** Wannier's exact solution, evaluated to machine precision below. It is itself
# checked against the concentric (Couette) limit and Sommerfeld's narrow-gap limit. The finite
# element results are graded against it field-wise and in the integral quantities, with a
# three-level mesh study (observed order and Richardson estimate). The run at a moderate Reynolds
# number has no exact solution; it is checked by a mesh comparison and by mass conservation.

from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
import gc
import time

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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation
from scipy.optimize import brentq
import sympy as sp

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import NewtonianFluid
from onesim.meshing import PeriodicPair
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.polynomial.tetrahedron import EDGES, QuadraticTetrahedralGeometry
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import IncompressibleNavierStokes, NoSlipWall, RotatingWall
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions.values import InitialValue
from onesim.simulation.problem_definition.boundaries import PeriodicBoundary

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Lengths are measured in units of the journal radius $R_1$, velocities in $\Omega R_1$ and
# stresses in $\mu\Omega$; in Stokes flow every consistent set of physical units scales out.
# The bearing has the radius ratio $R_2/R_1=2$ (a wide-gap case, where the eddy appears) and the
# relative eccentricities $\varepsilon=0.5$ and $0.8$. The Stokes limit is obtained with a
# negligible density ($Re=10^{-6}$); the inertial run uses $Re=\rho\,\Omega R_1C_r/\mu=50$.

R1, R2 = 1.0, 2.0          # journal and bearing radius
mu, Omega = 1.0, 1.0       # viscosity, journal angular velocity (counter-clockwise)
clearance = R2 - R1
eccentricities = (0.5, 0.8)
re_stokes, re_inertial = 1.0e-6, 50.0
levels = ((8, 48), (12, 72), (16, 96))   # bipolar grid: cells across the gap x cells around
thickness = 0.02           # slab thickness (one periodic cell)
print(f"R2/R1 = {R2 / R1:.1f}, clearance C_r = {clearance:.2f} R1, eccentricities {eccentricities}")


# ## Exact solution (Wannier 1950)
#
# The journal centre lies at $x=d_1$, the bearing centre at $x=d_2=d_1+e$, both on the
# $x$-axis (the line of centres); the poles of the bipolar system are at $x=\pm c$ with
# $c^2=d_1^2-R_1^2=d_2^2-R_2^2$. The narrow gap lies to the left of the journal, the wide gap to
# its right; the journal turns counter-clockwise. The flow is mirror-symmetric about the line of
# centres, so only $\cos\eta$ terms appear. The nine unknowns follow from the wall conditions at
# 48 collocation points on each circle (they are satisfied exactly, because the conditions
# contain only the Fourier modes 0 and 1 in $\eta$) and from the single-valued pressure,
# $\oint\partial_n\omega\,ds=0$ around the journal ($\omega$ is the vorticity; $\mu\omega+ip$ is an analytic function in Stokes flow). The
# wall pressure follows by integrating $\partial_s p=\mu\,\partial_n\omega$ along the journal, the
# loads by integrating the traction $\boldsymbol\sigma\cdot\mathbf n$ with the trapezoidal rule,
# which is spectrally accurate for these smooth periodic integrands.

X_, Y_, C_ = sp.symbols("x y c", real=True)


def wannier_basis():
    r1s, r2s = (X_ + C_) ** 2 + Y_**2, (X_ - C_) ** 2 + Y_**2
    r12 = sp.sqrt(r1s * r2s)
    ch, sh = (r1s + r2s) / (2 * r12), (r1s - r2s) / (2 * r12)       # cosh xi, sinh xi
    co = (X_**2 + Y_**2 - C_**2) / r12                              # cos eta
    xi = sp.log(r1s / r2s) / 2
    terms = [ch, xi * ch, sh, xi * sh, (2 * ch**2 - 1) * co, 2 * sh * ch * co, co, xi * co]
    functions = []
    for term in terms:
        psi = C_ * term / (ch - co)
        u, v = sp.diff(psi, Y_), -sp.diff(psi, X_)
        w = sp.diff(v, X_) - sp.diff(u, Y_)
        derived = [psi, u, v, sp.diff(u, X_), sp.diff(u, Y_), sp.diff(v, X_), sp.diff(v, Y_), w,
                   sp.diff(w, X_), sp.diff(w, Y_)]
        functions.append(sp.lambdify((X_, Y_, C_), derived, "numpy", cse=True))
    return functions


BASIS = wannier_basis()
# component index in the evaluated stack
PSI, U, V, UX, UY, VX, VY, W, WX, WY = range(10)


def basis_values(x, y, c):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.array([[np.broadcast_to(np.asarray(f_, float), x.shape) for f_ in f(x, y, c)] for f in BASIS])


def exact_solution(R_in, R_out, ecc, n=48):
    """Coefficients of Wannier's stream function for the journal R_in in the bearing R_out."""
    e = ecc * (R_out - R_in)
    d1 = ((R_out**2 - R_in**2) / e - e) / 2
    d2, c = d1 + e, np.sqrt(d1**2 - R_in**2)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows, rhs = [], []
    for centre, radius, wall_speed, flux in ((d1, R_in, Omega * R_in, -1.0), (d2, R_out, 0.0, 0.0)):
        B = basis_values(centre + radius * np.cos(theta), radius * np.sin(theta), c)
        dpsi_dr = -B[:, V] * np.cos(theta) + B[:, U] * np.sin(theta)
        rows += [np.column_stack((B[:, PSI].T, np.full(n, flux))), np.column_stack((-dpsi_dr.T, np.zeros(n)))]
        rhs += [np.zeros(n), np.full(n, wall_speed)]
    B = basis_values(d1 + R_in * np.cos(theta), R_in * np.sin(theta), c)
    vorticity_flux = (B[:, WX] * np.cos(theta) + B[:, WY] * np.sin(theta)).mean(axis=1)
    rows.append(np.r_[vorticity_flux, 0.0][None, :])
    rhs.append(np.zeros(1))
    A, b = np.vstack(rows), np.concatenate(rhs)
    scale = np.linalg.norm(A, axis=0)
    solution = np.linalg.lstsq(A / scale, b, rcond=None)[0] / scale
    return dict(R1=R_in, R2=R_out, eps=ecc, e=e, d1=d1, d2=d2, c=c, coef=solution[:8], Q=solution[8],
                residual=float(np.max(np.abs(A @ solution - b))))


def exact_field(sol, x, y):
    return np.einsum("k,k...->...", sol["coef"], basis_values(x, y, sol["c"]))


def exact_wall(sol, n=512):
    """Wall pressure (zero mean), torque and force on the journal from the exact solution."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    nx, ny = np.cos(theta), np.sin(theta)
    F = exact_field(sol, sol["d1"] + sol["R1"] * nx, sol["R1"] * ny)
    dp_dtheta = mu * sol["R1"] * (F[WX] * nx + F[WY] * ny)          # mu * d(omega)/dn along the wall
    k = np.fft.fftfreq(n, 1.0 / n)
    modes = np.fft.fft(dp_dtheta)
    modes[1:] /= 1j * k[1:]
    modes[0] = 0.0
    p = np.real(np.fft.ifft(modes))
    sxx, syy, sxy = -p + 2 * mu * F[UX], -p + 2 * mu * F[VY], mu * (F[UY] + F[VX])
    tx, ty = sxx * nx + sxy * ny, sxy * nx + syy * ny
    ds = 2 * np.pi * sol["R1"] / n
    return dict(theta=theta, p=p, p_modes=modes / n, Fx=tx.sum() * ds, Fy=ty.sum() * ds,
                T=(sol["R1"] * (nx * ty - ny * tx)).sum() * ds)


def wall_pressure_at(wall, theta):
    """Trigonometric interpolation of the exact wall pressure."""
    k = np.fft.fftfreq(len(wall["theta"]), 1.0 / len(wall["theta"]))
    return np.real(np.exp(1j * np.outer(theta, k)) @ wall["p_modes"])


def sommerfeld(ecc, R_in, cr):
    root = np.sqrt(1 - ecc**2)
    return (4 * np.pi * mu * Omega * R_in**3 / cr * (1 + 2 * ecc**2) / ((2 + ecc**2) * root),
            12 * np.pi * mu * Omega * R_in**3 / cr**2 * ecc / ((2 + ecc**2) * root))


exact = {ecc: exact_solution(R1, R2, ecc) for ecc in eccentricities}
walls = {ecc: exact_wall(sol) for ecc, sol in exact.items()}
for ecc, sol in exact.items():
    wl = walls[ecc]
    print(f"eps = {ecc}: torque T = {wl['T']:.8f}, force (F_x, F_y) = ({wl['Fx']:.2e}, {wl['Fy']:.8f}), "
          f"flow rate Q = {sol['Q']:.8f}   (wall-condition residual {sol['residual']:.1e})")

# Check of the exact solution itself. (i) Concentric limit: the torque must approach that of
# circular Couette flow, with a deviation of order eps^2 (so the last column must settle to a
# constant). (ii) Narrow-gap limit: torque, load and Sommerfeld number must approach Sommerfeld's
# lubrication result as C_r/R1 -> 0.
T_couette = -4 * np.pi * mu * Omega * R1**2 * R2**2 / (R2**2 - R1**2)
print(f"\nconcentric limit, Couette torque {T_couette:.8f}:")
for ecc in (0.2, 0.1, 0.05, 0.025):
    T_ecc = exact_wall(exact_solution(R1, R2, ecc))["T"]
    print(f"  eps = {ecc:5.3f}: T = {T_ecc:.8f},  (T / T_Couette - 1) / eps^2 = {(T_ecc / T_couette - 1) / ecc**2:.5f}")
gap_ratios = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
sweep = []
for cr in gap_ratios:
    wl = exact_wall(exact_solution(R1, R1 + cr, 0.5))
    T_s, W_s = sommerfeld(0.5, R1, cr)
    sweep.append((cr, -wl["T"] / T_s, np.hypot(wl["Fx"], wl["Fy"]) / W_s, wl))
print("narrow-gap limit, eps = 0.5:   C_r/R1   T_exact/T_Sommerfeld   W_exact/W_Sommerfeld   "
      "S_exact / S_Sommerfeld(eps)")
S_som = (2 + 0.25) * np.sqrt(0.75) / (12 * np.pi**2 * 0.5)
for cr, t_ratio, w_ratio, wl in sweep:
    S_exact = (R1 / cr) ** 2 * mu * (Omega / (2 * np.pi)) / (np.hypot(wl["Fx"], wl["Fy"]) / (2 * R1))
    print(f"{'':31s}{cr:6.2f}   {t_ratio:20.5f}   {w_ratio:20.5f}   {S_exact / S_som:24.5f}")


# ## Geometry and mesh
#
# The bipolar grid has $n_\xi$ cells across the gap (uniform in $\xi$) and $n_\eta$ cells
# around; the $\eta$ nodes are spaced so that the cells along the mid-gap circle have similar
# lengths. Each grid quadrilateral is cut into two triangles; the triangles are extruded into one
# layer of prisms, each split into three tetrahedra, and every edge gets its mid-node at the
# image of its parameter midpoint - on the cylinders exactly on the circle.
#
# Boundary conditions on every face:
#
# | face | condition |
# |---|---|
# | journal $\xi=\xi_1$ | rotating wall, $\mathbf u=\Omega\,\mathbf e_z\times(\mathbf x-\mathbf x_1)$ |
# | bearing shell $\xi=\xi_2$ | no slip, $\mathbf u=0$ |
# | slab faces normal to $z$ | periodic pair |
#
# The pressure is fixed by a zero mean value (the domain is closed).

@dataclass(frozen=True, eq=False, kw_only=True)
class BearingSlab(Geometry):
    """Gap between the journal and the bearing shell, extruded into a slab one cell thick."""

    dimension = 3

    def boundary_groups(self):
        names = ("journal", "bearing", "front", "back")
        return {**{name: (name,) for name in names}, "all": names}


class CurvedTetMesh(StructuredHexahedralMesh):
    """Ten-node (quadratic, isoparametric) tetrahedra with named boundary facets."""

    def __init__(self, geometry, coordinates, cells, facets):
        data = QuadraticTetrahedralGeometry(coordinates, cells)
        self.geometry = geometry
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.shape = (self.cell_count,)
        self.mesh_id = "bearing-tet10:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        self._facets = facets
        self._carrier = self._build_carrier()

    def _build_carrier(self):
        carrier = super()._build_carrier()
        return replace(carrier, connectivity=replace(carrier.connectivity, entity_kind="tetrahedron10"),
                       mesh_kind="unstructured")

    def to_mapping(self):
        return {**super().to_mapping(), "generator": "bearing-tet10", "geometry_mapping": "quadratic_tet10"}


def build_mesh(sol, n_xi, n_eta):
    c, d1 = sol["c"], sol["d1"]
    xi1, xi2 = np.arcsinh(c / sol["R1"]), np.arcsinh(c / sol["R2"])

    def bipolar(xi, eta):
        den = np.cosh(xi) - np.cos(eta)
        return np.column_stack((c * np.sinh(xi) / den, c * np.sin(eta) / den))

    s = np.linspace(0, 2 * np.pi, 8001)
    weight = 1.0 / np.sqrt(np.cosh(0.5 * (xi1 + xi2)) - np.cos(s))
    arc = np.r_[0.0, np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(s))]
    etas = np.interp(np.linspace(0, arc[-1], n_eta + 1), arc, s)[:-1]
    XI, ETA = np.meshgrid(np.linspace(xi1, xi2, n_xi + 1), etas, indexing="ij")
    xi_v, eta_v = XI.ravel(), ETA.ravel()
    xy = bipolar(xi_v, eta_v)
    index = np.arange(xi_v.size).reshape(XI.shape)
    triangles = []
    for i in range(n_xi):
        for j in range(n_eta):
            a, b, cc, d = index[i, j], index[i + 1, j], index[i + 1, (j + 1) % n_eta], index[i, (j + 1) % n_eta]
            triangles += [(a, b, cc), (a, cc, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, cc, d)]
    triangles = np.array(triangles)
    n2 = len(xy)
    coords = np.vstack([np.column_stack((xy, np.full(n2, z))) for z in (0.0, thickness)])
    i, j, k = np.sort(triangles, axis=1).T
    I, J, K = i + n2, j + n2, k + n2
    cells = np.vstack([np.column_stack((i, j, k, K)), np.column_stack((i, j, J, K)),
                       np.column_stack((i, I, J, K))])
    p = coords[cells]
    negative = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) < 0
    cells[negative] = cells[negative][:, [0, 2, 1, 3]]
    xi3, eta3 = np.tile(xi_v, 2), np.tile(eta_v, 2)
    geometry = BearingSlab()
    facets = {name: [] for name in geometry.boundary_groups()["all"]}
    for face in boundary_faces(cells, coords):
        v = np.asarray(face.vertices)
        if np.all(coords[v, 2] < 1e-12):
            name = "front"
        elif np.all(coords[v, 2] > thickness - 1e-12):
            name = "back"
        else:
            name = "journal" if np.all(np.abs(xi3[v] - xi1) < 1e-12) else "bearing"
        facets[name].append(tuple(int(t) for t in v))

    def midpoints(edges, xi_n, eta_n):
        a, b = edges[:, 0], edges[:, 1]
        deta = (eta_n[b] - eta_n[a] + np.pi) % (2 * np.pi) - np.pi
        return bipolar(0.5 * (xi_n[a] + xi_n[b]), eta_n[a] + 0.5 * deta)

    edges = np.sort(cells[:, list(EDGES)].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    middle = 0.5 * (coords[unique[:, 0]] + coords[unique[:, 1]])
    middle[:, :2] = midpoints(unique, xi3, eta3)
    mesh = CurvedTetMesh(geometry, np.vstack([coords, middle]),
                         np.hstack([cells, 2 * n2 + inverse.reshape(len(cells), 6)]), facets)
    # six-node section triangles on z = 0 for post-processing
    edges2 = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique2, inverse2 = np.unique(edges2, axis=0, return_inverse=True)
    nodes = np.vstack([xy, midpoints(unique2, xi_v, eta_v)])
    t6 = np.column_stack((triangles, n2 + inverse2.reshape(3, len(triangles)).T))
    on_journal = np.r_[np.abs(xi_v - xi1) < 1e-12,
                       (np.abs(xi_v[unique2[:, 0]] - xi1) < 1e-12) & (np.abs(xi_v[unique2[:, 1]] - xi1) < 1e-12)]
    return dict(sol=sol, n_xi=n_xi, n_eta=n_eta, geometry=geometry, mesh=mesh, xy=xy, triangles=triangles,
                nodes=nodes, t6=t6, on_journal=on_journal, xi=xi_v, eta=eta_v, xi1=xi1, xi2=xi2, bipolar=bipolar)


for n_xi, n_eta in levels:
    print(f"bipolar grid {n_xi:2d} x {n_eta:3d}: {2 * n_xi * n_eta:5d} section triangles, "
          f"{6 * n_xi * n_eta:6d} curved tetrahedra")


# ## Model
#
# One `IncompressibleNavierStokes` model per mesh and Reynolds number, with the boundary
# conditions of the table; the journal rotates about its own axis through $(d_1, 0)$.

def solve(data, reynolds, start=None):
    begin = time.perf_counter()
    geometry, mesh, sol = data["geometry"], data["mesh"], data["sol"]
    b = geometry.boundaries
    rho = reynolds * mu / (Omega * R1 * (sol["R2"] - sol["R1"]))
    model = IncompressibleNavierStokes(domain=geometry.domain(),
                                       material=NewtonianFluid(density=rho, dynamic_viscosity=mu))
    problem = Problem(
        name="Journal bearing", geometry=geometry, physics=(model,),
        boundary_conditions=(
            RotatingWall(boundary=b.journal, physics=model, angular_velocity=(0.0, 0.0, Omega),
                         rotation_origin=(sol["d1"], 0.0, 0.0)),
            NoSlipWall(boundary=b.bearing, physics=model),
            PeriodicBoundary(pair=PeriodicPair(first=b.front, second=b.back), physics=model),
        ),
        constraints=(MeanValue(model.pressure, 0.0),),
        initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
    )
    guess = (0.0, 0.0, 0.0) if start is None else start
    numerics = Numerics(
        spatial=FiniteElement(element=TaylorHood(), quadrature_order=5),
        nonlinear_solver=Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-11,
                                relative_tolerance=1e-11, maximum_iterations=25),
        initial_guess={model.velocity: guess},
    )
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    run = dict(data, reynolds=reynolds, rho=rho, velocity=result.field(model.velocity),
               pressure=result.field(model.pressure))
    run.update(post_process(run))
    run["seconds"] = time.perf_counter() - begin
    return run


# ## Post-processing on the section
#
# On the slab face $z=0$ every ten-node tetrahedron has a section triangle as a face, and the
# velocity trace there is the six-node curved quadratic triangle interpolant; the pressure trace
# is linear. The loads on the journal are the **reaction functionals** of the discrete momentum
# balance: with a discrete vector field $\mathbf g$ equal to a rigid motion on the journal nodes
# and zero elsewhere,
#
# $$\oint_{\text{journal}}(\boldsymbol\sigma\mathbf n)\cdot\mathbf g\,ds=-\int_\Omega\Big[\rho\,
#   (\mathbf u\cdot\nabla)\mathbf u\cdot\mathbf g+\mu(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T}):
#   \nabla\mathbf g-p\,\nabla\cdot\mathbf g\Big]\,dA ,$$
#
# with $\mathbf g=\mathbf e_x$, $\mathbf e_y$ for the force and $\mathbf g=\mathbf e_z\times
# (\mathbf x-\mathbf x_1)$ for the torque ($\mathbf n$ points from the journal into the fluid).
# The flow rate is the flux through the line of centres in the narrow and in the wide gap; both
# must equal the same $Q$ (mass conservation). The stream function for the figure is obtained by
# integrating $d\psi=u\,dy-v\,dx$ along the grid lines $\eta=$ const from the bearing ($\psi=0$).

A1, B1 = 0.059715871789770, 0.470142064105115
A2, B2 = 0.797426985353087, 0.101286507323456
TRI_POINTS = np.array([[1 / 3, 1 / 3], [B1, B1], [A1, B1], [B1, A1], [B2, B2], [A2, B2], [B2, A2]])
TRI_WEIGHTS = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3) / 2
DL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
GAUSS, GWEIGHT = np.polynomial.legendre.leggauss(6)
GAUSS, GWEIGHT = (GAUSS + 1) / 2, GWEIGHT / 2


def p2_shape(s, t):
    l0, l1, l2 = 1 - s - t, s, t
    N = np.array([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0])
    dN = np.array([(4 * l0 - 1) * DL[0], (4 * l1 - 1) * DL[1], (4 * l2 - 1) * DL[2],
                   4 * (l0 * DL[1] + l1 * DL[0]), 4 * (l1 * DL[2] + l2 * DL[1]), 4 * (l2 * DL[0] + l0 * DL[2])])
    return N, dN


def sample(run, points):
    return np.asarray(run["velocity"].evaluate(np.column_stack((points, np.zeros(len(points))))))


def reaction(run, U, P, G):
    X, Ut, Pt, Gt = run["nodes"][run["t6"]], U[run["t6"]][:, :, :2], P[run["triangles"]], G[run["t6"]]
    touch = np.abs(Gt).max(axis=(1, 2)) > 0
    X, Ut, Pt, Gt = X[touch], Ut[touch], Pt[touch], Gt[touch]
    total = 0.0
    for (s, t), w in zip(TRI_POINTS, TRI_WEIGHTS):
        N, dN = p2_shape(s, t)
        J = np.einsum("tkd,ke->tde", X, dN)
        grad = np.einsum("ke,tef->tkf", dN, np.linalg.inv(J))
        u, gu = np.einsum("k,tkc->tc", N, Ut), np.einsum("tkf,tkc->tcf", grad, Ut)
        g, gg = np.einsum("k,tkc->tc", N, Gt), np.einsum("tkf,tkc->tcf", grad, Gt)
        p = Pt @ np.array([1 - s - t, s, t])
        integrand = (run["rho"] * np.einsum("tc,tf,tcf->t", g, u, gu)
                     + mu * np.einsum("tcf,tcf->t", gu + gu.transpose(0, 2, 1), gg) - p * np.einsum("tcc->t", gg))
        total -= np.sum(w * np.abs(np.linalg.det(J)) * integrand)
    return total


def post_process(run):
    sol, nodes = run["sol"], run["nodes"]
    U = sample(run, nodes)
    P = np.asarray(run["pressure"].evaluate(np.column_stack((run["xy"], np.zeros(len(run["xy"])))))).ravel()
    phi = run["on_journal"].astype(float)
    zero = np.zeros_like(phi)
    Fx = reaction(run, U, P, np.column_stack((phi, zero)))
    Fy = reaction(run, U, P, np.column_stack((zero, phi)))
    T = reaction(run, U, P, np.column_stack((-phi * nodes[:, 1], phi * (nodes[:, 0] - sol["d1"]))))
    # flux through the line of centres: narrow gap (x < d1 - R1) and wide gap (x > d1 + R1)
    narrow = (sol["d2"] - sol["R2"], sol["d1"] - sol["R1"])
    wide = (sol["d1"] + sol["R1"], sol["d2"] + sol["R2"])
    Q = []
    for a, b in (narrow, wide):
        x = (a + (b - a) * (np.arange(400)[:, None] + GAUSS[None, :]) / 400).ravel()
        Q.append(float(np.sum(sample(run, np.column_stack((x, 0 * x)))[:, 1] * np.tile(GWEIGHT, 400)) * (b - a) / 400))
    Q_narrow, Q_wide = -Q[0], Q[1]     # psi(journal) - psi(bearing), positive for counter-clockwise rotation
    # stream function at the grid nodes, integrated along eta = const from the bearing
    n_xi, n_eta, bip = run["n_xi"], run["n_eta"], run["bipolar"]
    XI = run["xi"].reshape(n_xi + 1, n_eta)
    ETA = run["eta"].reshape(n_xi + 1, n_eta)
    psi = np.zeros_like(XI)
    for i in range(n_xi - 1, -1, -1):
        xs = XI[i + 1][:, None] + (XI[i] - XI[i + 1])[:, None] * GAUSS[None, :]
        es = np.repeat(ETA[i][:, None], len(GAUSS), axis=1)
        h = 1e-6
        p0, dp = bip(xs.ravel(), es.ravel()), (bip(xs.ravel() + h, es.ravel()) - bip(xs.ravel() - h, es.ravel())) / (2 * h)
        uv = sample(run, p0)
        dpsi = (uv[:, 0] * dp[:, 1] - uv[:, 1] * dp[:, 0]).reshape(xs.shape) @ GWEIGHT * (XI[i] - XI[i + 1])
        psi[i] = psi[i + 1] + dpsi
    # eddy on the wide-gap part of the line of centres: zero of v (centre) and minimum of psi
    xw = np.linspace(wide[0] + 1e-9, wide[1] - 1e-9, 3001)
    vw = sample(run, np.column_stack((xw, 0 * xw)))[:, 1]
    psi_w = np.r_[np.cumsum((0.5 * (vw[1:] + vw[:-1]) * np.diff(xw))[::-1])[::-1], 0.0]
    crossings = np.flatnonzero(np.diff(np.sign(vw)) != 0)
    if len(crossings):
        k = crossings[-1]
        eddy_x = xw[k] - vw[k] * (xw[k + 1] - xw[k]) / (vw[k + 1] - vw[k])
    else:
        eddy_x = np.nan
    return dict(U=U, P=P, Fx=Fx, Fy=Fy, T=T, Q_narrow=Q_narrow, Q_wide=Q_wide, psi=psi.ravel(),
                eddy_x=eddy_x, eddy_psi=float(psi_w.min()), w_max=float(np.abs(U[:, 2]).max()))


# ## Solve
#
# Stokes flow at $\varepsilon=0.5$ on the three grids, each of the two finer ones followed by the
# inertial run at $Re=50$ on the same grid (started from the Stokes solution); then Stokes flow at
# $\varepsilon=0.8$ on the finest grid and a narrow-clearance bearing ($C_r/R_1=0.1$) to approach
# the lubrication limit. Only the sampled section data of finished runs are kept.

def release(run):
    """Drop the solver objects of a finished run; the sampled section data are kept."""
    for key in ("velocity", "pressure", "mesh", "geometry"):
        run.pop(key, None)
    gc.collect()


def stokes_start(run):
    """Velocity of the Stokes solution as Newton starting value."""
    return lambda x, y, z: np.asarray(run["velocity"].evaluate(np.column_stack((x, y, np.zeros_like(x)))))


study, inertial = [], []
for n_xi, n_eta in levels:
    run = solve(build_mesh(exact[0.5], n_xi, n_eta), re_stokes)
    print(f"Stokes, eps = 0.5, grid {n_xi:2d} x {n_eta:3d}: T = {run['T']:.7f}, F = ({run['Fx']:+.2e}, "
          f"{run['Fy']:.7f}), Q = {run['Q_narrow']:.7f} / {run['Q_wide']:.7f}  ({run['seconds']:.0f} s)")
    if (n_xi, n_eta) != levels[0]:
        r = solve(build_mesh(exact[0.5], n_xi, n_eta), re_inertial, stokes_start(run))
        print(f"Re = {re_inertial:.0f}, eps = 0.5, grid {n_xi:2d} x {n_eta:3d}: T = {r['T']:.6f}, "
              f"F = ({r['Fx']:+.6f}, {r['Fy']:.6f}), Q = {r['Q_narrow']:.6f} / {r['Q_wide']:.6f}  ({r['seconds']:.0f} s)")
        release(r)
        inertial.append(r)
    release(run)
    study.append(run)
stokes = {0.5: study[-1]}
stokes[0.8] = solve(build_mesh(exact[0.8], *levels[-1]), re_stokes)
release(stokes[0.8])
r = stokes[0.8]
print(f"Stokes, eps = 0.8, grid {levels[-1][0]:2d} x {levels[-1][1]:3d}: T = {r['T']:.7f}, F = ({r['Fx']:+.2e}, "
      f"{r['Fy']:.7f})  ({r['seconds']:.0f} s)")

narrow_exact = exact_solution(R1, R1 + 0.1, 0.5)
narrow_wall = exact_wall(narrow_exact)
narrow = solve(build_mesh(narrow_exact, 6, 128), re_stokes)
release(narrow)
print(f"C_r/R1 = 0.1, eps = 0.5, grid 6 x 128: T = {narrow['T']:.5f}, F_y = {narrow['Fy']:.5f}  "
      f"({narrow['seconds']:.0f} s)")


# ## Comparison with the reference
#
# **Stokes flow against Wannier's exact solution** on the finest grid: velocity field at all
# section nodes, torque, load, flow rate (narrow and wide gap), journal wall pressure (zero-mean,
# relative to its range) and the eddy in the wide gap (position of its centre on the line of
# centres and its strength $\psi_{\min}$). The torque is the moment of the fluid on the journal;
# it is negative because it opposes the rotation. The out-of-plane velocity $w$ measures the
# two-dimensionality of the slab solution.

def eddy_exact(sol):
    x = np.linspace(sol["d1"] + sol["R1"] + 1e-9, sol["d2"] + sol["R2"] - 1e-9, 20001)
    F = exact_field(sol, x, 0 * x)
    k = np.flatnonzero(np.diff(np.sign(F[V])) != 0)[-1]
    return x[k] - F[V][k] * (x[k + 1] - x[k]) / (F[V][k + 1] - F[V][k]), float(F[PSI].min())


print(f"{'quantity':38s} {'OneSimulation':>14s} {'exact':>14s} {'deviation':>11s}")
for ecc, run in stokes.items():
    sol, wl = exact[ecc], walls[ecc]
    Fe = exact_field(sol, run["nodes"][:, 0], run["nodes"][:, 1])
    field_error = np.max(np.hypot(run["U"][:, 0] - Fe[U], run["U"][:, 1] - Fe[V])) / (Omega * R1)
    jw = np.flatnonzero(np.abs(run["xi"] - run["xi1"]) < 1e-12)
    theta = np.arctan2(run["xy"][jw, 1], run["xy"][jw, 0] - sol["d1"])
    p_fem, p_ex = run["P"][jw] - run["P"][jw].mean(), wall_pressure_at(wl, theta)
    p_error = np.max(np.abs(p_fem - (p_ex - p_ex.mean()))) / np.ptp(p_ex)
    x_eddy, psi_eddy = eddy_exact(sol)
    print(f"-- eps = {ecc} (grid {run['n_xi']} x {run['n_eta']})")
    for name, value, ref in (("torque T / (mu Omega R1^2)", run["T"], wl["T"]),
                             ("load F_y / (mu Omega R1)", run["Fy"], wl["Fy"]),
                             ("flow rate Q, narrow gap", run["Q_narrow"], sol["Q"]),
                             ("flow rate Q, wide gap", run["Q_wide"], sol["Q"]),
                             ("eddy centre x_c / R1", run["eddy_x"], x_eddy),
                             ("eddy strength psi_min", run["eddy_psi"], psi_eddy)):
        print(f"{name:38s} {value:14.7f} {ref:14.7f} {value / ref - 1:+11.2e}")
    print(f"{'load along line of centres F_x':38s} {run['Fx']:14.2e} {wl['Fx']:14.2e}   (exact: zero)")
    print(f"{'max |u - u_exact| / (Omega R1)':38s} {field_error:14.2e}")
    print(f"{'max wall pressure error / range':38s} {p_error:14.2e}")
    print(f"{'max |w| / (Omega R1) (two-dimensionality)':38s} {run['w_max']:14.2e}")

# **Mesh study** for $\varepsilon=0.5$ (grid spacing $h\propto 1/n_\xi$): observed order from the
# three grids and Richardson estimate, compared with the exact values.

h = 1.0 / np.array([lv[0] for lv in levels], float)


def richardson(values):
    f1, f2, f3 = values
    ratio = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda q: (h[0]**q - h[1]**q) / (h[1]**q - h[2]**q) - ratio, 0.3, 14.0)
    except ValueError:
        return np.nan, np.nan
    return p, f3 + (f3 - f2) / ((h[1] / h[2]) ** p - 1)


print("\nMesh study, eps = 0.5 (grids " + ", ".join(f"{a}x{b}" for a, b in levels) + ")")
for key, ref in (("T", walls[0.5]["T"]), ("Fy", walls[0.5]["Fy"]), ("Q_wide", exact[0.5]["Q"]),
                 ("eddy_psi", eddy_exact(exact[0.5])[1])):
    values = [r[key] for r in study]
    p, extrapolated = richardson(values)
    errors = "  ".join(f"{v / ref - 1:+.2e}" for v in values)
    print(f"  {key:9s} errors {errors}   change last two grids {abs(values[2] / values[1] - 1):.1e}, "
          f"observed order {p:.2f}, Richardson {extrapolated:.8f} ({extrapolated / ref - 1:+.1e} from exact)")

# **Narrow gap.** For $C_r/R_1=0.1$ the finite-element result is compared with Wannier and with
# Sommerfeld's lubrication formula (which neglects the curvature terms of order $C_r/R_1$).

T_s, W_s = sommerfeld(0.5, R1, 0.1)
print(f"\nC_r/R1 = 0.1, eps = 0.5: torque  FEM {-narrow['T']:.4f}, exact {-narrow_wall['T']:.4f} "
      f"({narrow['T'] / narrow_wall['T'] - 1:+.1e}), Sommerfeld {T_s:.4f} ({-narrow_wall['T'] / T_s - 1:+.1e})")
print(f"                         load    FEM {abs(narrow['Fy']):.4f}, exact {abs(narrow_wall['Fy']):.4f} "
      f"({narrow['Fy'] / narrow_wall['Fy'] - 1:+.1e}), Sommerfeld {W_s:.4f} ({abs(narrow_wall['Fy']) / W_s - 1:+.1e})")

# **Eddy onset.** The eddy appears when the wall shear stress on the bearing at the widest point
# of the gap changes sign; from the exact solution:

def wall_shear_widest(ecc):
    sol = exact_solution(R1, R2, ecc)
    return float(exact_field(sol, np.array([sol["d2"] + R2 - 1e-9]), np.array([0.0]))[VX][0])


eps_onset = brentq(wall_shear_widest, 0.2, 0.5, xtol=1e-6)
print(f"\nexact eddy onset for R2/R1 = 2: eps = {eps_onset:.4f}; at eps = 0.5 and 0.8 the eddy is present")

# **Inertia ($Re=50$).** No exact solution exists; the two finest grids are compared, together
# with the change against Stokes flow. Inertia turns the load away from the perpendicular to the
# line of centres (attitude angle $\phi=\arctan(F_y/F_x)$, $90°$ in Stokes flow) and shifts the
# eddy.

coarse_i, fine_i = inertial
print(f"\n{'Re = 50 quantity':30s} {'grid 12x72':>12s} {'grid 16x96':>12s} {'change':>9s} {'Stokes':>12s}")
for name, key in (("torque T", "T"), ("load F_x (line of centres)", "Fx"), ("load F_y", "Fy"),
                  ("flow rate Q (wide gap)", "Q_wide"), ("eddy centre x_c", "eddy_x")):
    print(f"{name:30s} {coarse_i[key]:12.6f} {fine_i[key]:12.6f} {abs(fine_i[key] / coarse_i[key] - 1):9.1e} "
          f"{stokes[0.5][key]:12.6f}")
attitude = np.degrees(np.arctan2(abs(fine_i["Fy"]), abs(fine_i["Fx"])))
print(f"attitude angle at Re = 50: {attitude:.2f} deg (Stokes: 90 deg); mass balance Q_narrow / Q_wide - 1 = "
      f"{fine_i['Q_narrow'] / fine_i['Q_wide'] - 1:+.1e}")


# ## Figure
#
# Top: streamlines (contours of $\psi$) over the speed for Stokes flow at $\varepsilon=0.5$ and
# $0.8$ and for $Re=50$ - the eddy in the wide gap is the closed set of streamlines along the
# bearing shell - and the pressure field at $\varepsilon=0.5$ with the load on the journal.
# Bottom: journal wall pressure against the exact solution, the pointwise velocity error over the
# section, mesh convergence, and the approach of the exact torque and load to Sommerfeld's
# lubrication theory as the clearance shrinks (open symbols: OneSimulation).

def section_tri(run):
    t6 = run["t6"]
    sub = np.vstack([t6[:, [0, 3, 5]], t6[:, [3, 1, 4]], t6[:, [5, 4, 2]], t6[:, [3, 4, 5]]])
    return Triangulation(run["nodes"][:, 0], run["nodes"][:, 1], sub)


def stream_panel(ax, run, title):
    tri = section_tri(run)
    speed = np.hypot(run["U"][:, 0], run["U"][:, 1])
    filled = ax.tricontourf(tri, speed, levels=np.linspace(0, Omega * R1, 31), cmap=FIELD_CMAP, extend="max")
    vertex_tri = Triangulation(run["xy"][:, 0], run["xy"][:, 1], run["triangles"])
    psi, Q = run["psi"], run["Q_wide"]
    ax.tricontour(vertex_tri, psi, levels=np.linspace(0.04 * Q, 0.96 * Q, 12), colors=INK, linewidths=0.6)
    if psi.min() < 0:
        ax.tricontour(vertex_tri, psi, levels=np.linspace(psi.min(), 0, 6)[1:-1], colors=COPPER, linewidths=0.8)
        ax.tricontour(vertex_tri, psi, levels=[0.0], colors=COPPER, linewidths=0.8, linestyles="--")
    sol = run["sol"]
    ax.add_patch(plt.Circle((sol["d1"], 0), sol["R1"], color="#D5D7C8", ec=INK, lw=0.8))
    ax.add_patch(plt.Circle((sol["d2"], 0), sol["R2"], fill=False, ec=INK, lw=1.2))
    ax.annotate("", xy=(sol["d1"] - 0.45, 0.45), xytext=(sol["d1"] + 0.45, 0.45),
                arrowprops=dict(arrowstyle="->", color=INK, connectionstyle="arc3,rad=0.5"))
    ax.set_aspect("equal")
    ax.set(xlim=(sol["d2"] - 2.1, sol["d2"] + 2.1), ylim=(-2.1, 2.1), xlabel="x / R1", ylabel="y / R1", title=title)
    return filled


fig = plt.figure(figsize=(19, 9.4), constrained_layout=True)
grid = fig.add_gridspec(2, 4)
ax = fig.add_subplot(grid[0, 0])
stream_panel(ax, stokes[0.5], "Stokes flow, eps = 0.5\nstreamlines (eddy in copper) over the speed")
ax = fig.add_subplot(grid[0, 1])
stream_panel(ax, stokes[0.8], "Stokes flow, eps = 0.8")
ax = fig.add_subplot(grid[0, 2])
filled = stream_panel(ax, fine_i, f"Re = {re_inertial:.0f}, eps = 0.5")
fig.colorbar(filled, ax=ax, label="|u| / (Omega R1)", shrink=0.8, format="%.1f")

ax = fig.add_subplot(grid[0, 3])
run = stokes[0.5]
tri = section_tri(run)
P_nodes = np.zeros(len(run["nodes"]))
P_nodes[run["t6"][:, :3].ravel()] = run["P"][run["triangles"]].ravel()
for (a, b), m in (((0, 1), 3), ((1, 2), 4), ((2, 0), 5)):
    P_nodes[run["t6"][:, m]] = 0.5 * (run["P"][run["triangles"][:, a]] + run["P"][run["triangles"][:, b]])
limit = np.abs(P_nodes).max()
filled = ax.tricontourf(tri, P_nodes, levels=np.linspace(-limit, limit, 41), cmap=DIVERGING_CMAP,
                        norm=TwoSlopeNorm(0.0, -limit, limit))
vertex_tri = Triangulation(run["xy"][:, 0], run["xy"][:, 1], run["triangles"])
ax.tricontour(vertex_tri, run["psi"], levels=np.linspace(0.08, 0.92, 7) * run["Q_wide"], colors=INK, linewidths=0.4)
sol = run["sol"]
ax.add_patch(plt.Circle((sol["d1"], 0), R1, color="#D5D7C8", ec=INK, lw=0.8))
ax.arrow(sol["d1"], 0, 0, 0.75 * np.sign(run["Fy"]), width=0.03, color=COPPER, length_includes_head=True)
ax.text(sol["d1"] + 0.1, 0.45 * np.sign(run["Fy"]), "load", color=COPPER, fontsize=9)
ax.set_aspect("equal")
ax.set(xlim=(sol["d2"] - 2.1, sol["d2"] + 2.1), ylim=(-2.1, 2.1), xlabel="x / R1", ylabel="y / R1",
       title="Pressure, eps = 0.5 (Stokes)")
fig.colorbar(filled, ax=ax, label="p / (mu Omega)", shrink=0.8, format="%.1f")

ax = fig.add_subplot(grid[1, 0])
wl = walls[0.5]
order = np.argsort(wl["theta"])
ax.plot(np.degrees(wl["theta"]), wl["p"] - wl["p"].mean(), color=INK, lw=1.4, label="Wannier (exact), Re = 0")
for r, marker, colour, label in ((stokes[0.5], "o", GREEN, "OneSimulation, Re = 0"),
                                 (fine_i, "s", COPPER, f"OneSimulation, Re = {re_inertial:.0f}")):
    jw = np.flatnonzero(np.abs(r["xi"] - r["xi1"]) < 1e-12)
    th = np.mod(np.arctan2(r["xy"][jw, 1], r["xy"][jw, 0] - r["sol"]["d1"]), 2 * np.pi)
    k = np.argsort(th)
    ax.plot(np.degrees(th[k]), (r["P"][jw] - r["P"][jw].mean())[k], marker, ms=3.2, color=colour, mfc="none",
            label=label, ls="-" if r is fine_i else "none", lw=0.8)
ax.set(xlabel="angle on the journal from the wide gap / deg", ylabel="wall pressure / (mu Omega)",
       title="Pressure on the journal, eps = 0.5", xlim=(0, 360))
ax.set_xticks([0, 90, 180, 270, 360])
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 1])
Fe = exact_field(run["sol"], run["nodes"][:, 0], run["nodes"][:, 1])
error = np.hypot(run["U"][:, 0] - Fe[U], run["U"][:, 1] - Fe[V]) / (Omega * R1)
filled = ax.tricontourf(tri, np.log10(np.maximum(error, 1e-9)), levels=np.linspace(-9, -4, 26), cmap=FIELD_CMAP,
                        extend="both")
ax.add_patch(plt.Circle((sol["d1"], 0), R1, color="#D5D7C8", ec=INK, lw=0.8))
ax.add_patch(plt.Circle((sol["d2"], 0), R2, fill=False, ec=INK, lw=1.0))
ax.set_aspect("equal")
ax.set(xlim=(sol["d2"] - 2.1, sol["d2"] + 2.1), ylim=(-2.1, 2.1), xlabel="x / R1", ylabel="y / R1",
       title=f"Velocity error against Wannier, eps = 0.5\n(grid {run['n_xi']} x {run['n_eta']}, max {error.max():.1e})")
fig.colorbar(filled, ax=ax, label="log10 |u - u_exact| / (Omega R1)", shrink=0.8, format="%.0f")

ax = fig.add_subplot(grid[1, 2])
for key, ref, marker, colour, name in (("T", walls[0.5]["T"], "o-", GREEN, "torque"),
                                       ("Fy", walls[0.5]["Fy"], "s-", COPPER, "load"),
                                       ("Q_wide", exact[0.5]["Q"], "^-", "#214D43", "flow rate"),
                                       ("eddy_psi", eddy_exact(exact[0.5])[1], "D--", "#8B9A8D", "eddy strength")):
    ax.loglog(h, [abs(r[key] / ref - 1) for r in study], marker, color=colour, label=name)
ax.loglog(h, abs(study[0]["Fy"] / walls[0.5]["Fy"] - 1) * (h / h[0]) ** 4, ":", color=INK, lw=0.8, label="slope 4")
ax.set_xticks(h, [f"1/{lv[0]}" for lv in levels])
ax.minorticks_off()
ax.set(xlabel="grid spacing across the gap (1 / n_xi)", ylabel="relative error against Wannier",
       title="Mesh convergence, eps = 0.5")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 3])
ax.semilogx(gap_ratios, [s[1] for s in sweep], "o-", color=GREEN, label="torque: exact / Sommerfeld")
ax.semilogx(gap_ratios, [s[2] for s in sweep], "s-", color=COPPER, label="load: exact / Sommerfeld")
for value, cr, wl_, marker in ((stokes[0.5], 1.0, walls[0.5], "o"), (narrow, 0.1, narrow_wall, "o")):
    T_s, W_s = sommerfeld(0.5, R1, cr)
    ax.semilogx([cr], [-value["T"] / T_s], marker, ms=11, mfc="none", color=INK)
    ax.semilogx([cr], [abs(value["Fy"]) / W_s], "s", ms=11, mfc="none", color=INK)
ax.plot([], [], "o", ms=9, mfc="none", color=INK, label="OneSimulation")
ax.axhline(1.0, color=INK, lw=0.8, ls=":")
ax.invert_xaxis()
ax.set(xlabel="clearance ratio C_r / R1", ylabel="ratio to lubrication theory",
       title="Narrow-gap limit (Sommerfeld), eps = 0.5")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Raise the Reynolds number further to follow how inertia rotates the load and moves the eddy;
# in a long bearing resolved along the axis (many cells in $z$ instead of one periodic cell),
# Taylor vortices appear beyond a critical Taylor number. Let the bearing shell rotate as well
# (a second `RotatingWall`) - Wannier's solution covers that case too, and counter-rotation
# produces a stagnation point inside the gap. Replace the prescribed eccentricity by the force balance of a
# loaded journal to find its equilibrium position, or narrow the clearance further and compare
# the pressure profile with the Reynolds-equation solution of lubrication theory.
