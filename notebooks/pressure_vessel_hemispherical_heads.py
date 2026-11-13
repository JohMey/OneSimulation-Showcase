"""OneSimulation example: stresses in a cylindrical pressure vessel with hemispherical heads.
Requires the installed, activated OneSimulation runtime.

A closed steel vessel - a cylinder with two hemispherical heads of the same wall thickness -
is loaded by internal pressure and computed as a 3D elastic solid (one octant with three
symmetry planes) with quadratic hexahedra whose nodes lie on the exact cylinder and sphere
surfaces. The field is graded against Lame's thick-wall solutions far from the junction, the
classical thin-shell solution for the cylinder-hemisphere junction (edge shear, bending
moment and its decay over the length sqrt(R t)), global equilibrium and a mesh study through
the wall thickness.
"""


# # Pressure vessel with hemispherical heads: the junction stresses in 3D
#
# The textbook pressure-vessel formulas give the membrane stresses of a thin cylinder,
# hoop $pR/t$ and axial $pR/(2t)$, and of a thin sphere, $pR/(2t)$ in every direction.
# A real vessel joins a cylinder and two heads. Under pressure the free cylinder would
# grow radially by $\delta_c=\frac{pR^2}{Et}(1-\tfrac{\nu}{2})$, the free hemisphere only by
# $\delta_s=\frac{pR^2}{2Et}(1-\nu)$. At the junction both must move together, so an edge
# shear force and a local bending moment appear in the wall; they die out along the
# meridian over a length of the order of $\sqrt{Rt}$. This notebook computes the vessel as
# a **3D elastic solid**, without any shell assumption, and compares it with the
# classical solutions.
#
# **Vessel.** Inner radius $a=0.5$ m, wall thickness $t=25$ mm (mid-surface radius
# $R=a+t/2=0.5125$ m, $R/t=20.5$), cylindrical part of length $2L$ with $L=1.0$ m between
# the two hemispherical heads of the same thickness; structural steel, $E=200$ GPa,
# $\nu=0.3$; internal pressure $p=2$ MPa (20 bar).
#
# **Geometry, dimension and boundary conditions.** The vessel is modelled in 3D. Its
# symmetry planes $x=0$, $y=0$ (through the axis) and $z=0$ (the middle of the
# cylinder) cut out one octant: a quarter of the half-cylinder $0\le z\le L$ and a
# quarter of one hemispherical head centred at $(0,0,L)$. Boundary conditions on every
# face:
#
# | face | condition |
# |---|---|
# | inner surface (cylinder and head) | internal pressure: traction $-p\,\mathbf n$ ($\mathbf n$ = outward normal of the wall, pointing into the vessel) |
# | outer surface (cylinder and head) | traction-free |
# | symmetry plane $x=0$ | roller $u_x=0$ |
# | symmetry plane $y=0$ | roller $u_y=0$ |
# | symmetry plane $z=0$ | roller $u_z=0$ |
#
# Load and geometry are axisymmetric, so the solution depends on the radius and on the
# position along the meridian, $\sigma(r,z)$: the 3D model resolves the variation through
# the thickness (Lame) and along the meridian (junction bending) at the same time.
#
# **Numerical method.**
# - Spatial: continuous Galerkin finite elements, **isoparametric quadratic hexahedra
#   (27-node Q2)** of linear elasticity on a block-structured mesh: the cylinder is a
#   polar block, each head a "cubed-sphere" octant (two equatorial patches and a polar
#   cap, no degenerate cells at the pole). Every node, including the mid-edge, mid-face
#   and centre nodes, lies on the exact cylinder or sphere at its radius. The cells are
#   graded along the meridian from $t/2$ at the junction.
# - Mesh study: 1, 2 and 4 elements through the thickness on the same surface mesh, then
#   the surface mesh refined (first cell $t/3$, 1.5 times the cells along the meridian)
#   with 4 elements through the thickness; this finest mesh (1312 hexahedra, about 34 000
#   unknowns) is used for all comparisons and figures.
# - Linear static problem, sparse direct LU solver.
#
# **References.**
# - Thick-wall solutions (exact 3D elasticity away from the junction; G. Lame, *Leçons sur
#   la théorie mathématique de l'élasticité des corps solides*, 1852; S. Timoshenko,
#   J. N. Goodier, *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970):
#   closed-end cylinder $\sigma_r=A-B/r^2$, $\sigma_\theta=A+B/r^2$, $\sigma_z=A$ with
#   $A=pa^2/(b^2-a^2)$, $B=pa^2b^2/(b^2-a^2)$, $b=a+t$; sphere
#   $\sigma_r=C(1-b^3/r^3)$, $\sigma_t=C(1+b^3/(2r^3))$ with $C=pa^3/(b^3-a^3)$.
# - Junction (S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed.,
#   McGraw-Hill 1959, chapters on the symmetrical deformation of cylindrical shells and on
#   shells of revolution): with $\beta^4=3(1-\nu^2)/(R^2t^2)$ and equal
#   thickness, compatibility of the radial displacement gives an edge shear
#   $Q_0=p/(8\beta)$ and no edge moment. Along the meridian, at the distance $s$ from the
#   junction ($\xi=\beta|s|$),
#   $M=\pm\frac{p}{8\beta^2}e^{-\xi}\sin\xi$ (cylinder +, head -; positive = tension on the
#   outer surface), $N_\theta=pR\mp\frac{pR}{4}e^{-\xi}\cos\xi$ in the cylinder / head
#   ($pR$ replaced by $pR/2$ in the head), $N_s=pR/2$. The largest bending moment
#   $0.0403\,p/\beta^2$ occurs at $\beta|s|=\pi/4$, and the surface stress there is
#   $pR/(2t)\pm6M/t^2=(0.5\pm0.146)\,pR/t$. Thin-shell theory neglects terms of order
#   $t/R\approx5$ %. The comparison therefore superposes the shell perturbation on the
#   exact Lame state (see *Model*); remaining deviations of a few per cent (for example a
#   small edge moment) are thick-wall effects, not errors of the 3D model.
# - Global equilibrium of the octant: the reaction on $z=0$ equals $p\,\pi a^2/4$, the
#   reactions on $x=0$ and $y=0$ equal $p\,(aL+\pi a^2/4)$.

from dataclasses import dataclass, replace
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing.generation.backends.structured.volume.hexahedral import StructuredHexahedralMesh
from onesim.meshing.generation.contracts.hexahedral_geometry import FACES
from onesim.meshing.generation.contracts.polynomial.hex import HEX27_REFERENCE, QuadraticHexahedralGeometry
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacementComponents, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters

a, t, L = 0.5, 0.025, 1.0              # inner radius, wall thickness, half-length of the cylinder / m
b, R = a + t, a + t / 2                # outer and mid-surface radius / m
E, nu, p = 200e9, 0.3, 2.0e6           # Pa, -, Pa
lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
beta = (3 * (1 - nu**2)) ** 0.25 / np.sqrt(R * t)
S0 = p * R / t                          # membrane hoop stress of the cylinder, the stress unit below
A_cyl, B_cyl = p * a**2 / (b**2 - a**2), p * a**2 * b**2 / (b**2 - a**2)
C_sph = p * a**3 / (b**3 - a**3)
print(f"R/t = {R / t:.1f}, sqrt(R t) = {np.sqrt(R * t) * 1e3:.1f} mm, beta = {beta:.3f} 1/m, "
      f"pi/beta = {np.pi / beta * 1e3:.0f} mm; cylinder L = {L * beta:.1f}/beta long")
print(f"membrane: hoop pR/t = {S0 / 1e6:.2f} MPa, axial pR/2t = {S0 / 2e6:.2f} MPa; Lame: inner hoop "
      f"{(A_cyl + B_cyl / a**2) / 1e6:.2f} MPa, outer hoop {(A_cyl + B_cyl / b**2) / 1e6:.2f} MPa, axial {A_cyl / 1e6:.2f} MPa")
print(f"free radial growth: cylinder {p * R**2 / (E * t) * (1 - nu / 2) * 1e6:.1f} um, hemisphere "
      f"{p * R**2 / (2 * E * t) * (1 - nu) * 1e6:.1f} um; edge shear Q0 = p/(8 beta) = {p / (8 * beta) / 1e3:.1f} kN/m")


# ## Geometry and mesh
#
# Each block is a structured grid of nodes (with the mid-edge, mid-face and centre nodes
# of the 27-node cells) in parameter coordinates, mapped to the exact surfaces:
# cylinder $(r\cos\varphi, r\sin\varphi, z)$ with $2n$ cells over $\varphi\in[0,\pi/2]$;
# head patches along the directions $(1,\tan\alpha,\tan\gamma)$ and
# $(\tan\alpha,1,\tan\gamma)$ (equiangular, $n$ cells in $\alpha$, graded in $\gamma$ from
# the equator) and the polar cap $(\tan u,\tan v,1)$, each normalised and scaled to the
# radius $a+\rho t$ about the head centre. The equatorial nodes of the head patches fall
# exactly on the cylinder nodes, so the blocks are joined node to node.

@dataclass(frozen=True, eq=False, kw_only=True)
class VesselOctant(Geometry):
    """One octant of the vessel wall: inner and outer surface and three symmetry planes."""

    dimension = 3

    def boundary_groups(self):
        names = ("inner", "outer", "sym_x", "sym_y", "sym_z")
        return {**{name: (name,) for name in names}, "all": names}


class Hex27Mesh(StructuredHexahedralMesh):
    """Curved 27-node hexahedral mesh from explicit nodes, cells and named boundary faces."""

    def __init__(self, geometry, coordinates, cells, facets):
        self.geometry, self.shape = geometry, (len(cells),)
        data = QuadraticHexahedralGeometry(coordinates, cells)
        self.coordinates, self.cells, self.cell_volumes = data.coordinates, data.cells, data.volumes
        self.cell_count = len(self.cells)
        self.mesh_id = "vessel-hex27:" + sha256(self.coordinates.tobytes() + self.cells.tobytes()).hexdigest()
        self._facets = {name: list(faces) for name, faces in facets.items()}
        self._carrier = self._build_carrier()

    def _build_carrier(self):
        carrier = super()._build_carrier()
        return replace(carrier, connectivity=replace(carrier.connectivity, entity_kind="hexahedron27"))


NODE = np.rint(2 * HEX27_REFERENCE).astype(int)      # position of each of the 27 nodes in a 3x3x3 grid
CENTRE = np.array([0.0, 0.0, L])                      # centre of the hemispherical head


def with_midpoints(nodes):
    out = np.empty(2 * len(nodes) - 1)
    out[::2], out[1::2] = nodes, 0.5 * (nodes[:-1] + nodes[1:])
    return out


def geometric(length, n, first):
    """n cells on [0, length], the first one of width `first`, geometric growth."""
    if n * first >= length:
        return np.linspace(0.0, length, n + 1)
    r = brentq(lambda r: first * (r**n - 1) / (r - 1) - length, 1 + 1e-12, 50.0)
    s = np.concatenate(([0.0], np.cumsum(first * r ** np.arange(n))))
    return s * length / s[-1]


def assemble_blocks(blocks):
    """Merge node grids of several blocks into one 27-node mesh (right-handed cells)."""
    points, cells, offset = [], [], 0
    for grid in blocks:
        e0, e1, e2 = grid[2, 0, 0] - grid[0, 0, 0], grid[0, 2, 0] - grid[0, 0, 0], grid[0, 0, 2] - grid[0, 0, 0]
        if np.dot(np.cross(e0, e1), e2) < 0:
            grid = grid[:, :, ::-1]
        shape = np.array(grid.shape[:3])
        ids = np.arange(np.prod(shape)).reshape(tuple(shape)) + offset
        for index in product(*(range(k) for k in (shape - 1) // 2)):
            cells.append([ids[tuple(2 * np.asarray(index) + m)] for m in NODE])
        points.append(grid.reshape(-1, 3))
        offset += np.prod(shape)
    X = np.concatenate(points)
    _, first, inverse = np.unique(np.round(X / 1e-10).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    return X[first], inverse.ravel()[np.asarray(cells)]


def wall_radius(q):
    """Distance from the axis (cylinder) or from the head centre (head)."""
    return np.where(q[..., 2] <= L, np.hypot(q[..., 0], q[..., 1]), np.linalg.norm(q - CENTRE, axis=-1))


def vessel_mesh(n_t, n=4, m_c=16, m_h=10, first=t / 2):
    rad = a + with_midpoints(np.linspace(0.0, 1.0, n_t + 1)) * t
    phi = with_midpoints(np.linspace(0.0, np.pi / 2, 2 * n + 1))
    z = with_midpoints(L - geometric(L, m_c, first)[::-1])
    P, Z, Rr = np.meshgrid(phi, z, rad, indexing="ij")
    blocks = [np.stack((Rr * np.cos(P), Rr * np.sin(P), Z), axis=-1)]
    alpha = with_midpoints(np.linspace(0.0, np.pi / 4, n + 1))
    gamma = with_midpoints(geometric(np.pi / 4, m_h, first / R))
    for kind in ("x", "y", "cap"):
        U, V, Rr = np.meshgrid(alpha, alpha if kind == "cap" else gamma, rad, indexing="ij")
        one, tu, tv = np.ones_like(U), np.tan(U), np.tan(V)
        d = {"x": (one, tu, tv), "y": (tu, one, tv), "cap": (tu, tv, one)}[kind]
        d = np.stack(d, axis=-1)
        blocks.append(CENTRE + Rr[..., None] * d / np.linalg.norm(d, axis=-1)[..., None])
    X, C = assemble_blocks(blocks)
    owners = {}
    for cell, vertices in enumerate(C):
        for side, face in enumerate(FACES):
            corners = tuple(int(v) for v in vertices[list(face)])
            owners.setdefault(tuple(sorted(corners)), []).append((corners, cell, side))
    groups = {name: [] for name in ("inner", "outer", "sym_x", "sym_y", "sym_z")}
    for found in owners.values():
        if len(found) == 1:
            q = X[list(found[0][0])]
            planes = [k for k in range(3) if np.all(np.abs(q[:, k]) < 1e-12)]
            name = "sym_" + "xyz"[planes[0]] if planes else (
                "inner" if np.all(np.abs(wall_radius(q) - a) < 1e-9) else "outer")
            groups[name].append(found[0])
    return dict(n_t=n_t, X=X, C=C, groups=groups, surface=(n, m_c, m_h, first))


MESHES = {"t1": vessel_mesh(1), "t2": vessel_mesh(2), "t4": vessel_mesh(4),
          "t4 fine": vessel_mesh(4, m_c=24, m_h=15, first=t / 3)}
exact_volume = np.pi / 4 * (b**2 - a**2) * L + np.pi / 6 * (b**3 - a**3)
for key, data in MESHES.items():
    data["mesh"] = Hex27Mesh(VesselOctant(), data["X"], data["C"],
                             {k: [f[0] for f in v] for k, v in data["groups"].items()})
    print(f"mesh {key:8s}: {data['n_t']} element(s) through the thickness, {len(data['C'])} hexahedra, "
          f"{len(data['X'])} nodes, wall volume {data['mesh'].cell_volumes.sum():.6f} m^3 (exact {exact_volume:.6f})")


# ## Model
#
# `LinearElasticity` with structural steel. The internal pressure is a traction
# $p\,\mathbf e$ on the inner faces, with $\mathbf e$ the unit vector from the axis
# (cylinder) or from the head centre (head) - the exact inward normal of the vessel
# cavity. The reactions on the symmetry planes are the residual of the unconstrained
# system at the supported degrees of freedom.

steel = IsotropicSolid(density=7850.0, specific_heat_capacity=475.0, thermal_conductivity=44.5,
                       youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6)


def internal_pressure(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    e = np.stack((x, y, np.where(z > L, z - L, 0.0)), axis=-1)
    return p * e / np.linalg.norm(e, axis=-1)[..., None]


def solve(key):
    start = time.time()
    data = MESHES[key]
    mesh = data["mesh"]
    geometry = mesh.geometry
    solid = LinearElasticity(domain=geometry.domain(), material=steel)
    bnd = geometry.boundaries
    problem = Problem(name="pressure vessel octant", geometry=geometry, physics=(solid,), boundary_conditions=(
        Traction(boundary=bnd.inner, physics=solid, value=internal_pressure),
        TractionFree(boundary=bnd.outer, physics=solid),
        FixedDisplacementComponents(boundary=bnd.sym_x, physics=solid, components=("x",)),
        FixedDisplacementComponents(boundary=bnd.sym_y, physics=solid, components=("y",)),
        FixedDisplacementComponents(boundary=bnd.sym_z, physics=solid, components=("z",))))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())).solve_stationary()
    op = result.operators[solid.model_id]
    index = np.asarray(op.field_binding.indices)
    u = np.zeros(op.problem_size)
    u[index] = np.asarray(result.field(solid.displacement).values)
    reaction = None
    if key != "t4 fine":       # reactions on the coarser meshes (re-assembly doubles the memory on the finest)
        _, system, rhs, constraints = op.weak_assembly(0.0)
        residual = system.matrix.tocsr() @ u - np.asarray(rhs)
        fixed = np.zeros(op.problem_size, dtype=bool)
        fixed[[c.dof_index for c in constraints]] = True
        reaction = np.where(fixed[index], residual[index], 0.0).sum(axis=0)
    run = dict(key=key, data=data, op=op, u=u, U=u[index], dofs=op.problem_size, cells=mesh.cell_count,
               reaction=reaction)
    evaluate(run)
    run["seconds"] = time.time() - start
    return run


# **Evaluation.** Stresses are taken from the displacement gradient inside the owning
# cell. Along the meridian (at $\varphi=22.5°$) the arc length $s$ is measured from the
# junction (negative in the cylinder, positive in the head); the local axes are the
# meridional direction $\mathbf e_s$, the hoop direction $\mathbf e_\theta$ and the wall
# normal. Surface values are taken 20 µm inside the wall (the quadratic geometry
# reproduces the exact surfaces to about a micrometre between the nodes). The stress
# resultants per unit length of the mid-surface follow from an 8-point Gauss rule
# through the thickness, $N=\int\sigma\,(r/R)\,d\zeta$ and $M=\int\sigma\,\zeta\,(r/R)\,d\zeta$
# ($\zeta=r-R$; in the cylinder the hoop resultant has no factor $r/R$).

PHI = np.pi / 8
S_LINE = np.linspace(-0.45, 0.35, 321)
ZETA, ZETA_W = np.polynomial.legendre.leggauss(8)
ZETA, ZETA_W = 0.5 * t * ZETA, 0.5 * t * ZETA_W
INSET = 2e-5


def tensor(s):
    xx, yy, zz, xy, yz, xz = np.moveaxis(np.asarray(s), -1, 0)
    return np.stack((np.stack((xx, xy, xz), -1), np.stack((xy, yy, yz), -1), np.stack((xz, yz, zz), -1)), -2)


def meridian_frame(s):
    psi = np.where(s > 0, s / R, 0.0)
    base = np.column_stack((np.zeros_like(s), np.zeros_like(s), np.where(s > 0, L, L + s)))
    e_r = np.column_stack((np.cos(psi) * np.cos(PHI), np.cos(psi) * np.sin(PHI), np.sin(psi)))
    e_s = np.column_stack((-np.sin(psi) * np.cos(PHI), -np.sin(psi) * np.sin(PHI), np.cos(psi)))
    e_t = np.broadcast_to(np.array([-np.sin(PHI), np.cos(PHI), 0.0]), e_r.shape)
    return base, e_r, e_s, e_t


def local_stress(op, u, s, radius):
    base, e_r, e_s, e_t = meridian_frame(s)
    S = tensor(op.stress(u, base + np.asarray(radius)[..., None] * e_r))
    comp = lambda e1, e2: np.einsum("qi,qij,qj->q", e1, S, e2)
    return comp(e_s, e_s), comp(e_t, e_t), comp(e_r, e_r)


def von_mises(S):
    dev = S - np.trace(S, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))


def evaluate(run):
    op, u = run["op"], run["u"]
    run["s_out"], run["t_out"], _ = local_stress(op, u, S_LINE, np.full_like(S_LINE, b - INSET))
    run["s_in"], run["t_in"], _ = local_stress(op, u, S_LINE, np.full_like(S_LINE, a + INSET))
    N_s, N_t, M_s = np.zeros_like(S_LINE), np.zeros_like(S_LINE), np.zeros_like(S_LINE)
    for z, w in zip(ZETA, ZETA_W):
        ss, st, _ = local_stress(op, u, S_LINE, np.full_like(S_LINE, R + z))
        f = 1 + z / R
        N_s += w * f * ss
        M_s += w * f * z * ss
        N_t += w * np.where(S_LINE > 0, f, 1.0) * st
    run.update(N_s=N_s, N_t=N_t, M_s=M_s)
    # far field: radial lines through the cylinder at z = 0 and through the head at 75 deg latitude
    r = np.linspace(a + INSET, b - INSET, 9)
    for name, s0 in (("cyl", -L + 1e-4), ("sph", np.radians(75) * R)):
        ss, st, sr = local_stress(op, u, np.full_like(r, s0), r)
        run[name] = dict(r=r, s=ss, t=st, r_=sr)
    pts = np.column_stack((R * np.cos(PHI), R * np.sin(PHI), 0.0))
    run["u_mid"] = float(np.asarray(op.evaluate_field(op.field, u, pts))[0] @ np.array([np.cos(PHI), np.sin(PHI), 0.0]))


# **Thin-shell reference in a thick wall.** Shell theory splits the state into the
# membrane state and the junction perturbation. In a wall with $t/R=0.05$ the exact
# membrane state is Lame's, whose resultants differ from $pR$, $pR/2$ by terms of order
# $t/R$ (e.g. the hoop resultant of the cylinder is exactly $pa$, and the uniform axial
# stress has a small first moment $At^3/(12R)$ about the mid-surface). The junction
# perturbation of the 3D field is therefore taken as the 3D resultant minus the Lame
# resultant of the same side (computed with the same quadrature), and the predicted
# surface stresses are Lame's plus the shell perturbation, $\Delta N_\theta/t\pm6M/t^2$ and
# $\pm6\nu M/t^2$.

def lame_resultants():
    """Resultants of the exact thick-wall states (cylinder, sphere) with the 8-point rule."""
    r, f = R + ZETA, 1 + ZETA / R
    hoop_c, sph = A_cyl + B_cyl / r**2, C_sph * (1 + b**3 / (2 * r**3))
    return dict(cyl=dict(N_t=np.sum(ZETA_W * hoop_c), N_s=np.sum(ZETA_W * f * A_cyl), M=np.sum(ZETA_W * f * ZETA * A_cyl)),
                sph=dict(N_t=np.sum(ZETA_W * f * sph), N_s=np.sum(ZETA_W * f * sph), M=np.sum(ZETA_W * f * ZETA * sph)))


LAME = lame_resultants()


def lame_side(s, key):
    return np.where(s > 0, LAME["sph"][key], LAME["cyl"][key])


def shell_theory(s):
    """Junction perturbation of thin-shell theory (equal thickness: edge shear p/(8 beta), no edge moment)
    superposed on the exact Lame surface stresses."""
    xi = beta * np.abs(s)
    head = s > 0
    dM = np.where(head, -1.0, 1.0) * p / (8 * beta**2) * np.exp(-xi) * np.sin(xi)
    dN_t = np.where(head, 1.0, -1.0) * p * R / 4 * np.exp(-xi) * np.cos(xi)
    sph = lambda r: C_sph * (1 + b**3 / (2 * r**3))
    base = dict(s_out=np.where(head, sph(b), A_cyl), s_in=np.where(head, sph(a), A_cyl),
                t_out=np.where(head, sph(b), A_cyl + B_cyl / b**2), t_in=np.where(head, sph(a), A_cyl + B_cyl / a**2))
    return dict(dM=dM, dN_t=dN_t,
                s_out=base["s_out"] + 6 * dM / t**2, s_in=base["s_in"] - 6 * dM / t**2,
                t_out=base["t_out"] + dN_t / t + 6 * nu * dM / t**2, t_in=base["t_in"] + dN_t / t - 6 * nu * dM / t**2)


# ## Solve

runs = {}
for key in MESHES:
    runs[key] = solve(key)
    if key != "t4 fine":
        for name in ("op", "u", "U"):     # keep only the finest field in memory
            del runs[key][name]
    print(f"mesh {key:8s}: {runs[key]['cells']} curved hexahedra, {runs[key]['dofs']} unknowns, "
          f"{runs[key]['seconds']:.0f} s")
fine = runs["t4 fine"]


# ## Comparison with the reference

shell = shell_theory(S_LINE)


def peak(values, sign=1):
    k = np.argmax(sign * values)
    return values[k], S_LINE[k]


print(f"Lame thick-wall solution, cylinder at the symmetry plane z = 0 ({L * beta:.1f}/beta from the junction), "
      "finest mesh")
c = fine["cyl"]
for i in (0, 4, 8):
    r = c["r"][i]
    ref = (A_cyl + B_cyl / r**2, A_cyl, A_cyl - B_cyl / r**2)
    print(f"   r = {r:.5f} m: hoop {c['t'][i] / 1e6:8.4f} MPa (Lame {ref[0] / 1e6:8.4f}, {100 * (c['t'][i] / ref[0] - 1):+.3f} %),"
          f" axial {c['s'][i] / 1e6:8.4f} MPa (Lame {ref[1] / 1e6:8.4f}, {100 * (c['s'][i] / ref[1] - 1):+.3f} %),"
          f" radial {c['r_'][i] / 1e6:8.4f} MPa (Lame {ref[2] / 1e6:8.4f})")
u_ref = R / E * ((A_cyl + B_cyl / R**2) - nu * (A_cyl - B_cyl / R**2 + A_cyl))
print(f"   radial displacement of the mid-surface: {fine['u_mid'] * 1e6:.3f} um (Lame {u_ref * 1e6:.3f} um, "
      f"{100 * (fine['u_mid'] / u_ref - 1):+.3f} %)")
print(f"   thin-wall formulas for comparison: hoop pR/t = {S0 / 1e6:.3f} MPa, pa/t = {p * a / t / 1e6:.3f} MPa "
      f"(= Lame mean hoop stress), axial pR/2t = {S0 / 2e6:.3f} MPa")
print("Lame thick-wall solution, head at 75 deg latitude")
c = fine["sph"]
for i in (0, 4, 8):
    r = c["r"][i]
    ref = (C_sph * (1 + b**3 / (2 * r**3)), C_sph * (1 - b**3 / r**3))
    tang = 0.5 * (c["t"][i] + c["s"][i])
    print(f"   r = {r:.5f} m: tangential {tang / 1e6:8.4f} MPa (Lame {ref[0] / 1e6:8.4f}, {100 * (tang / ref[0] - 1):+.3f} %),"
          f" hoop/meridional {c['t'][i] / 1e6:.4f}/{c['s'][i] / 1e6:.4f} MPa, radial {c['r_'][i] / 1e6:8.4f} MPa "
          f"(Lame {ref[1] / 1e6:8.4f})")
print(f"   thin-wall formula pR/2t = {S0 / 2e6:.3f} MPa")

dM = fine["M_s"] - lame_side(S_LINE, "M")
dN_t = fine["N_t"] - lame_side(S_LINE, "N_t")
fine.update(dM=dM, dN_t=dN_t)
M_ref = p / (8 * beta**2) * np.exp(-np.pi / 4) * np.sin(np.pi / 4)
print("Junction: perturbation of the 3D stress resultants (3D minus Lame) against thin-shell theory, finest mesh")
for label, (v, sv), (r_, sr) in (
        ("largest bending moment, cylinder / (N m/m)", peak(dM), peak(shell["dM"])),
        ("largest bending moment, head / (N m/m)", peak(dM, -1), peak(shell["dM"], -1))):
    print(f"   {label:46s} 3D {v:9.2f} at s = {sv * 1e3:6.1f} mm; shell {r_:9.2f} at s = {sr * 1e3:6.1f} mm "
          f"(beta s = pi/4; {100 * (v / r_ - 1):+.2f} %)")
dN0 = np.interp(0.0, S_LINE, fine["N_t"]) - LAME["cyl"]["N_t"]
print(f"   hoop resultant at the junction minus p a / (kN/m)  3D {dN0 / 1e3:9.3f}; shell -pR/4 = {-p * R / 4e3:9.3f} "
      f"({100 * (dN0 / (-p * R / 4) - 1):+.2f} %)")
crossing = S_LINE[1:][(dM[1:] * dM[:-1] < 0) & (S_LINE[1:] < -0.05)]
print(f"   bending moment in the cylinder changes sign at s = {crossing[-1] * 1e3:.1f} mm "
      f"(shell: beta s = -pi, s = {-np.pi / beta * 1e3:.1f} mm)")
print(f"   edge moment at the junction: 3D {np.interp(0.0, S_LINE, dM):.2f} N m/m = "
      f"{np.interp(0.0, S_LINE, dM) / M_ref:.3f} x the peak (thin-shell theory: 0; a thick-wall effect of order t/R)")
print(f"   Lame resultants: cylinder N_theta = {LAME['cyl']['N_t'] / 1e3:.3f} kN/m (= p a = {p * a / 1e3:.3f}; shell pR = "
      f"{p * R / 1e3:.3f}), first moment of the axial stress {LAME['cyl']['M']:.2f} N m/m")
print(f"   meridional resultant far from the junction {np.interp(-0.4, S_LINE, fine['N_s']) / 1e3:.3f} kN/m "
      f"(exact axial equilibrium p a^2/(2R) = {p * a**2 / (2 * R) / 1e3:.3f} kN/m)")
print("Junction: surface stresses in units of pR/t (finest mesh) against Lame + thin-shell perturbation")
for label, key, sign, region in (("max. meridional, outer surface (cylinder)", "s_out", 1, S_LINE < 0),
                                 ("max. meridional, inner surface (head)", "s_in", 1, S_LINE > 0),
                                 ("min. meridional, inner surface (cylinder)", "s_in", -1, S_LINE < 0),
                                 ("min. meridional, outer surface (head)", "s_out", -1, S_LINE > 0),
                                 ("max. hoop, inner surface (cylinder)", "t_in", 1, S_LINE < 0)):
    v, sv = peak(np.where(region, fine[key], -sign * np.inf), sign)
    r_, sr = peak(np.where(region, shell[key], -sign * np.inf), sign)
    print(f"   {label:44s} 3D {v / S0:.4f} at s = {sv * 1e3:6.1f} mm; reference {r_ / S0:.4f} at s = {sr * 1e3:6.1f} mm "
          f"({100 * (v / r_ - 1):+.2f} %)")
for key, label in (("t_out", "hoop at the junction, outer"), ("t_in", "hoop at the junction, inner")):
    v, r_ = np.interp(0.0, S_LINE, fine[key]), np.interp(-1e-9, S_LINE, shell[key])
    print(f"   {label:44s} 3D {v / S0:.4f}; reference {r_ / S0:.4f} ({100 * (v / r_ - 1):+.2f} %)")
print("   (textbook thin-wall values: 0.5 +- 0.146 pR/t meridional at beta s = pi/4, 0.75 pR/t hoop at the junction)")
vm_in = von_mises(tensor(fine["op"].stress(fine["u"], meridian_frame(S_LINE)[0]
                                            + (a + INSET) * meridian_frame(S_LINE)[1])))
print(f"   largest von Mises stress on the meridian: {vm_in.max() / 1e6:.2f} MPa ({vm_in.max() / S0:.3f} pR/t) on the "
      f"inner surface at s = {S_LINE[np.argmax(vm_in)] * 1e3:.1f} mm")

print("Global equilibrium of the octant (reaction forces on the symmetry planes)")
ref_force = np.array([p * (a * L + np.pi * a**2 / 4), p * (a * L + np.pi * a**2 / 4), p * np.pi * a**2 / 4])
for run in (runs["t1"], runs["t4"]):
    print(f"   mesh {run['key']}: |F_x|, |F_y|, |F_z| = "
          + ", ".join(f"{abs(f) / 1e3:.3f}" for f in run["reaction"]) + " kN; exact "
          + ", ".join(f"{f / 1e3:.3f}" for f in ref_force) + " kN; largest deviation "
          + f"{np.max(np.abs(np.abs(run['reaction']) / ref_force - 1)):.1e}")

print("Mesh study (Q2, curved): 1, 2, 4 elements through the thickness, then the finer surface mesh (4 elements)")
quantities = (("M max, cylinder / (N m/m)", lambda r: peak(r["M_s"] - lame_side(S_LINE, "M"))[0]),
              ("M max, head / (N m/m)", lambda r: -peak(r["M_s"] - lame_side(S_LINE, "M"), -1)[0]),
              ("sigma_s outer max / (pR/t)", lambda r: peak(np.where(S_LINE < 0, r["s_out"], -np.inf))[0] / S0),
              ("sigma_s inner max / (pR/t)", lambda r: peak(np.where(S_LINE > 0, r["s_in"], -np.inf))[0] / S0),
              ("hoop at z = 0, inner / MPa", lambda r: r["cyl"]["t"][0] / 1e6))
for label, f in quantities:
    vals = [f(runs[k]) for k in ("t1", "t2", "t4", "t4 fine")]
    print(f"   {label:28s} " + "  ".join(f"{v:10.4f}" for v in vals)
          + f"   change 2 -> 4 elements {abs(vals[2] / vals[1] - 1):.1e}, finer surface mesh {abs(vals[3] / vals[2] - 1):.1e}")


# ## Figure
#
# Top: von Mises stress on the vessel (the octant mirrored to the whole vessel, one quarter
# cut away to show the inner surface; finest mesh), the von Mises stress in the wall
# across the junction (the meridian unrolled, thickness stretched 8 times) and the mesh
# study. Bottom: meridional and hoop surface stresses along the meridian, the stress
# resultants against thin-shell theory, and the through-thickness stresses far from the
# junction against Lame.

op, u, U, data = fine["op"], fine["u"], fine["U"], fine["data"]
space = op.space
grid5 = np.linspace(0.0, 1.0, 4)
GA, GB = (m.ravel() for m in np.meshgrid(grid5, grid5, indexing="ij"))


def face_samples(faces):
    polys, vals = [], []
    for _, cell, side in faces:
        ref = np.empty((len(GA), 3))
        axis, free = side // 2, [d for d in range(3) if d != side // 2]
        ref[:, axis], ref[:, free[0]], ref[:, free[1]] = side % 2, GA, GB
        ref = np.clip(ref, 1e-9, 1 - 1e-9)
        grad = np.einsum("qna,nb->qba", space.gradients(cell, ref), U[space.cell_dofs[cell]])
        eps = 0.5 * (grad + np.swapaxes(grad, -1, -2))
        S = lam * np.trace(eps, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) + 2 * mu * eps
        P = space.geometry.map(cell, ref).reshape(4, 4, 3)
        V = von_mises(S).reshape(4, 4)
        for i in range(3):
            for j in range(3):
                polys.append(np.stack((P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1])))
                vals.append(V[i:i + 2, j:j + 2].mean())
    return np.array(polys), np.array(vals)


fig = plt.figure(figsize=(18, 11), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.1, 1.2, 1.0))
ax3 = fig.add_subplot(grid[0, 0], projection="3d")
norm = colors.Normalize(vmin=0.3 * S0 / 1e6, vmax=1.0 * S0 / 1e6)
for name in ("outer", "inner"):
    polys, vals = face_samples(data["groups"][name])
    for sx, sy, sz in product((1, -1), repeat=3):
        if sx == 1 and sy == 1:
            continue                                     # quarter cut away
        mirrored = polys * np.array([sx, sy, sz])
        shade = FIELD_CMAP(norm(vals / 1e6))
        ax3.add_collection3d(Poly3DCollection(mirrored[..., [0, 2, 1]], facecolors=shade, edgecolors=shade,
                                              linewidths=0.15, antialiased=False))
ax3.set(xlim=(-0.8, 0.8), ylim=(-1.6, 1.6), zlim=(-0.8, 0.8), xlabel="x / m", ylabel="z (axis) / m", zlabel="y / m",
        xticks=[-0.5, 0, 0.5], yticks=[-1.5, -0.75, 0, 0.75, 1.5], zticks=[-0.5, 0, 0.5],
        title="von Mises stress on the vessel, quarter cut away\n(Q2, 4 elements through the wall)")
ax3.set_box_aspect((1.6, 3.2, 1.6))
ax3.view_init(elev=22, azim=-58)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax3, shrink=0.55, label="von Mises stress / MPa")

ax = fig.add_subplot(grid[0, 1])
s_map = np.linspace(-0.3, 0.3, 241)
z_map = np.linspace(-t / 2 + INSET, t / 2 - INSET, 25)
SS, ZZ = np.meshgrid(s_map, z_map, indexing="ij")
base, e_r, _, _ = meridian_frame(SS.ravel())
_, e_r, e_s, _ = meridian_frame(SS.ravel())
SIG = np.einsum("qi,qij,qj->q", e_s, tensor(op.stress(u, base + (R + ZZ.ravel())[:, None] * e_r)), e_s)
SIG = SIG.reshape(SS.shape) / S0
cs = ax.contourf(SS / np.sqrt(R * t), ZZ * 1e3, SIG, levels=np.linspace(0.30, 0.66, 37), cmap=FIELD_CMAP, extend="both")
ax.contour(SS / np.sqrt(R * t), ZZ * 1e3, SIG, levels=np.arange(0.32, 0.66, 0.04), colors=CREAM, linewidths=0.4)
ax.axvline(0.0, color=INK, lw=0.8, ls="--")
ax.text(-2.5, 10.5, "cylinder", color=CREAM, fontsize=10)
ax.text(1.6, 10.5, "head", color=CREAM, fontsize=10)
ax.set(xlabel="meridional distance from the junction s / sqrt(R t)", ylabel="position in the wall zeta / mm (outer +)",
       title="Meridional stress in the wall across the junction\n(thickness stretched 8 times): the bending flips sides")
ax.set_aspect(8e-3 / np.sqrt(R * t), adjustable="box")
fig.colorbar(cs, ax=ax, shrink=0.8, label="meridional stress / (pR/t)", orientation="horizontal")

ax = fig.add_subplot(grid[0, 2])
for (label, f), marker in zip(quantities[:4], "osD^"):
    ref_value = f(fine)
    vals = np.array([f(runs[k]) for k in ("t1", "t2", "t4")])
    ax.plot([1, 2, 4], 100 * (vals / ref_value - 1), marker + "-", label=label)
    ax.plot([4.25], [0.0], marker, mfc="none", ms=10, color=ax.lines[-1].get_color())
ax.axhline(0.0, color="#9FA593", lw=0.6)
ax.set(xticks=[1, 2, 4], xlabel="Q2 elements through the thickness",
       ylabel="difference from the finest mesh / %",
       title="Mesh study (open symbols at 4: finest mesh,\nsurface refined 1.5 times, 4 elements)")
ax.legend(fontsize=7)
ax.grid(alpha=0.5)

ax = fig.add_subplot(grid[1, 0])
x = S_LINE / np.sqrt(R * t)
for key, colour, label in (("s_out", GREEN, "meridional, outer"), ("s_in", COPPER, "meridional, inner"),
                           ("t_out", "#5F9175", "hoop, outer"), ("t_in", "#A85539", "hoop, inner")):
    ax.plot(x, fine[key] / S0, color=colour, lw=2, label="3D: " + label)
    ax.plot(x, shell[key] / S0, color=colour, lw=1, ls="--")
ax.plot([], [], color=INK, lw=1, ls="--", label="Lame + thin-shell perturbation")
ax.axvline(0.0, color=INK, lw=0.6)
ax.set(xlabel="s / sqrt(R t)  (cylinder < 0 < head)", ylabel="surface stress / (pR/t)",
       title="Surface stresses along the meridian")
ax.set_ylim(0.1, 1.1)
ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.grid(alpha=0.5)

ax = fig.add_subplot(grid[1, 1])
ax.plot(x, fine["dM"] / (p * R * t), color=GREEN, lw=2, label="3D: bending moment M / (pRt)")
ax.plot(x, shell["dM"] / (p * R * t), color=GREEN, lw=1, ls="--", label="shell: M")
ax2 = ax.twinx()
ax2.plot(x, fine["dN_t"] / (p * R), color=COPPER, lw=2, label="3D: hoop perturbation dN_theta / (pR)")
ax2.plot(x, shell["dN_t"] / (p * R), color=COPPER, lw=1, ls="--", label="shell: dN_theta")
ax.axvline(0.0, color=INK, lw=0.6)
for xi in (np.pi / 4, np.pi):
    ax.axvline(-xi / (beta * np.sqrt(R * t)), color="#9FA593", lw=0.6, ls=":")
ax.set(xlabel="s / sqrt(R t)", ylabel="M / (pRt)", ylim=(-0.03, 0.03),
       title="Junction perturbation of the resultants (3D minus Lame)\nagainst thin-shell theory (dotted: beta s = -pi/4, -pi)")
ax2.set(ylabel="dN_theta / (pR)", ylim=(-0.3, 0.3))
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
ax.grid(alpha=0.5)

ax = fig.add_subplot(grid[1, 2])
rr = np.linspace(a, b, 50)
c, h = fine["cyl"], fine["sph"]
ax.plot((rr - a) * 1e3, (A_cyl + B_cyl / rr**2) / S0, color=GREEN, lw=1, ls="--", label="Lame: cylinder hoop")
ax.plot((c["r"] - a) * 1e3, c["t"] / S0, "o", color=GREEN, label="3D: cylinder hoop (z = 0)")
ax.plot((rr - a) * 1e3, np.full_like(rr, A_cyl) / S0, color=COPPER, lw=1, ls="--", label="Lame: cylinder axial")
ax.plot((c["r"] - a) * 1e3, c["s"] / S0, "s", color=COPPER, label="3D: cylinder axial")
ax.plot((rr - a) * 1e3, C_sph * (1 + b**3 / (2 * rr**3)) / S0, color="#A85539", lw=1, ls="--", label="Lame: sphere")
ax.plot((h["r"] - a) * 1e3, 0.5 * (h["t"] + h["s"]) / S0, "^", color="#A85539", label="3D: head (75 deg latitude)")
ax.axhline(1.0, color="#9FA593", lw=0.8, ls=":")
ax.axhline(0.5, color="#9FA593", lw=0.8, ls=":")
ax.text(0.5, 1.008, "membrane pR/t", fontsize=8, color=INK)
ax.text(0.5, 0.508, "membrane pR/2t", fontsize=8, color=INK)
ax.set(xlabel="depth from the inner surface / mm", ylabel="stress / (pR/t)",
       title="Far field: stresses through the wall vs Lame")
ax.legend(fontsize=7, loc="center right")
ax.grid(alpha=0.5)
plt.show()


# ## Going further
#
# Make the heads thinner (the optimum of membrane theory is half the cylinder thickness)
# and watch the edge moment appear; replace the hemispheres by torispherical or
# ellipsoidal heads, whose knuckle produces compressive hoop stresses; add a nozzle on the
# cylinder (the octant then becomes a quarter model with a genuinely 3D stress
# concentration); or increase $t/R$ to see where thin-shell theory fails and Lame's
# solution remains exact.
