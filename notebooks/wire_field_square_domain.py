"""OneSimulation example: magnetic field of an infinitely long wire in a square, magnetically insulated air domain.
Requires the installed, activated OneSimulation runtime.

The field of a straight round conductor inside a square box with magnetic insulation is a genuinely
two-dimensional field B(x, y). It is computed with 3D Nedelec elements on a slab that is one cell thick
and compared with the exact conformal-mapping solution for the square, with Ampere's law and with the
exact inductance per unit length.
"""


# # Magnetic field of an infinite wire in a square air domain
#
# A straight round conductor of radius $a=1$ cm carries $I=1$ A uniformly over its cross section.
# It runs through the centre of a square air region of side $s=20$ cm, whose outer boundary carries
# magnetic insulation, $\mathbf n\times\mathbf A=0$ (the return current flows in the boundary). The
# wire is infinitely long, so the field depends only on the cross-section coordinates $(x,y)$: near
# the wire it is the familiar circular field, but towards the square walls it is pushed into the
# shape of the boundary - stronger towards the side midpoints, weaker towards the corners.
#
# OneSimulation's magnetostatics is three-dimensional. The cross section is therefore extruded into a
# slab that is a single tetrahedral layer thick; magnetic insulation on the two slab faces is exact
# for this problem, because $\mathbf A=A_z(x,y)\,\hat{\mathbf z}$ has no tangential component there.
# The model solves $\nabla\times(\mu_0^{-1}\nabla\times\mathbf A)=\mathbf J$ with second-order Nedelec
# elements and an exact (Lagrange-multiplier) Coulomb gauge.
#
# **Numerical method.** Spatial: curl-conforming finite elements (Nedelec, first kind, order 2) on
# tetrahedra, with the Coulomb gauge enforced by a nodal Lagrange multiplier (mixed saddle-point
# formulation). Time integration: none, the problem is stationary. Solver: sparse direct LU
# factorisation of the saddle-point system. Mesh: graded O-grid triangulation extruded to one layer of
# tetrahedra, built in numpy and passed to OneSimulation as an unstructured simplex mesh.
#
# **Reference (exact solution of the 2D problem).** With $A_z=0$ on the square, $A_z/(\mu_0 I)$ outside
# the wire is the Dirichlet Green's function $G$ of the square for a source at its centre (the mean
# value property makes a uniformly filled round wire act like a line current outside itself). $G$
# follows from the Schwarz-Christoffel map of the unit disc onto the square,
#
# $$z(w)=R_c\int_0^w\frac{\mathrm dt}{\sqrt{1+t^4}}=R_c\,w\,{}_2F_1\!\left(\tfrac14,\tfrac12;\tfrac54;-w^4\right),
#   \qquad G=-\frac{1}{2\pi}\ln|w(z)|,$$
#
# where $R_c=(s/2)/\int_0^1(1+t^4)^{-1/2}\mathrm dt=0.53935\,s$ is the conformal radius of the square
# about its centre (Polya and Szego, *Isoperimetric Inequalities in Mathematical Physics*, 1951). The
# exact inductance per unit length then follows without any approximation for a centred round wire,
#
# $$L'=\frac{\mu_0}{2\pi}\left[\frac14+\ln\frac{R_c}{a}\right],$$
#
# the $\tfrac14$ being the internal inductance. Inside the wire the field is $\mu_0 I r/(2\pi a^2)$ plus a
# harmonic correction that the same map supplies. Further checks: Ampere's circulation
# $\oint\mathbf B\cdot\mathrm d\boldsymbol\ell=\mu_0 I_\mathrm{enc}$ on circles, the flux-linkage form of
# the energy, and a three-mesh convergence study. A second, deliberately small box shows how magnetic
# insulation deforms the field when the air region is too tight.

from dataclasses import dataclass
from importlib.metadata import version
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.spatial import Delaunay
from scipy.special import hyp2f1

from onesim.cad.d3 import Box
from onesim.materials import VacuumPermeability
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    CoulombGauge, MagneticInsulation, MagneticVectorPotential, Magnetostatics,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters

wire_radius = 0.01          # m, conductor radius a
current = 1.0               # A, total current along +z
side = 0.20                 # m, side of the square air domain (half side = 10 radii)
small_side = 0.04           # m, deliberately tight square (half side = 2 radii)
mesh_levels = (32, 64, 128) # cells around the wire circumference
mu0 = float(VacuumPermeability().evaluate())

K0 = quad(lambda t: (1.0 + t**4) ** -0.5, 0.0, 1.0)[0]       # centre-to-side distance of the unit map


def conformal_radius(s):
    return 0.5 * s / K0


def inductance_exact(s):
    """Exact inductance per unit length of the centred round wire in the insulated square."""
    return mu0 / (2 * np.pi) * (0.25 + np.log(conformal_radius(s) / wire_radius))


print(f"conformal radius R_c = {conformal_radius(1.0):.5f} s")
print(f"exact L' (s = {100*side:.0f} cm) = {inductance_exact(side)*1e9:.3f} nH/m,"
      f"  (s = {100*small_side:.0f} cm) = {inductance_exact(small_side)*1e9:.3f} nH/m")


# ## Exact field from the conformal map
#
# The map $z(w)$ is inverted by Newton's method. With $F=\partial_x A_z-i\,\partial_y A_z$ the field is
# $B_x=-\operatorname{Im}F$, $B_y=-\operatorname{Re}F$.

def unit_disc_point(zeta):
    w = np.array(zeta, dtype=complex)
    for _ in range(80):
        residual = w * hyp2f1(0.25, 0.5, 1.25, -w**4) - zeta
        w = w - residual * np.sqrt(1.0 + w**4)
        if np.max(np.abs(residual)) < 1e-14:
            break
    return w


def exact_B(x, y, s, centre):
    zp = (np.asarray(x) - centre) + 1j * (np.asarray(y) - centre)
    Rc = conformal_radius(s)
    w = unit_disc_point(zp / Rc)
    r = np.abs(zp)
    F = -mu0 * current / (2 * np.pi * Rc) * np.sqrt(1.0 + w**4) / w
    Bx, By = -F.imag, -F.real
    inside = r < wire_radius
    if np.any(inside):
        zi, wi = zp[inside], w[inside]
        correction = -mu0 * current / (2 * np.pi) * (np.sqrt(1.0 + wi**4) / (wi * Rc) - 1.0 / zi)
        correction[np.abs(zi) < 1e-9] = 0.0
        B_in = mu0 * current * r[inside] / (2 * np.pi * wire_radius**2)
        angle = np.angle(zi)
        Bx[inside] = -B_in * np.sin(angle) - correction.imag
        By[inside] = B_in * np.cos(angle) - correction.real
    return Bx, By


# ## Geometry and mesh
#
# The cross section is meshed with a graded O-grid point cloud: uniform rings inside the conductor, a
# geometric radial progression outside it that matches the circumferential spacing (cells are nearly
# square everywhere), and rings that blend from circles near the wire into the square at the wall. A
# Delaunay triangulation of these points keeps the 32/64/128-gon of the wire surface as mesh edges.
# Each triangle is extruded into a prism that is split into three tetrahedra with a vertex-ordering
# rule, which keeps the 3D mesh conforming. The slab thickness follows the in-plane spacing at the
# wire.

def cross_section(s, n_theta):
    spacing = 2 * np.pi * wire_radius / n_theta
    points = [[0.0, 0.0]]
    n_inner = int(np.ceil(wire_radius / spacing))
    for j in range(1, n_inner + 1):
        r = wire_radius * j / n_inner
        n = n_theta if j == n_inner else max(6, int(round(2 * np.pi * r / spacing)))
        shift = 0.0 if j == n_inner else 0.5 * (j % 2)
        angle = 2 * np.pi * (np.arange(n) + shift) / n
        points += list(np.column_stack((r * np.cos(angle), r * np.sin(angle))))
    half = s / 2
    growth = 2 * np.pi / n_theta
    n_rings = int(np.ceil(np.log(half / wire_radius) / growth))
    for j in range(1, n_rings + 1):
        t = j / n_rings
        shift = 0.0 if j == n_rings else 0.5 * (j % 2)
        angle = 2 * np.pi * (np.arange(n_theta) + shift) / n_theta
        direction = np.column_stack((np.cos(angle), np.sin(angle)))
        on_square = direction / np.max(np.abs(direction), axis=1)[:, None]
        blend = t**3
        points += list(wire_radius * (half / wire_radius) ** t * ((1 - blend) * direction + blend * on_square))
    points = np.vstack([points, [[half, half], [-half, half], [-half, -half], [half, -half]]])
    points = np.unique(np.round(points, 13), axis=0)
    return points, Delaunay(points).simplices


def slab_mesh(s, n_theta):
    points, triangles = cross_section(s, n_theta)
    thickness = 2.5 * 2 * np.pi * wire_radius / n_theta
    n2 = len(points)
    xy = points + s / 2
    coordinates = np.vstack([np.column_stack((xy, np.full(n2, z))) for z in (0.0, thickness)])
    i, j, k = np.sort(triangles, axis=1).T
    I, J, K = i + n2, j + n2, k + n2
    cells = np.vstack([np.column_stack((i, j, k, K)), np.column_stack((i, j, J, K)),
                       np.column_stack((i, I, J, K))])
    geometry = Box(length=s, width=s, height=thickness)
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells)
    return geometry, mesh, thickness, len(triangles)


@dataclass(frozen=True)
class RoundWire:
    """Uniform axial current density in the meshed polygonal wire cross section (exactly I in total)."""

    centre: float
    sides: int
    support: object

    def current_density(self, points):
        p = np.asarray(points, dtype=float)
        shape = p.shape
        p = p.reshape(-1, 3)
        x, y = p[:, 0] - self.centre, p[:, 1] - self.centre
        angle, r = np.arctan2(y, x), np.hypot(x, y)
        mid = (np.floor(angle * self.sides / (2 * np.pi)) + 0.5) * 2 * np.pi / self.sides
        inside = r * np.cos(angle - mid) <= wire_radius * np.cos(np.pi / self.sides) * (1 + 1e-12)
        area = 0.5 * self.sides * wire_radius**2 * np.sin(2 * np.pi / self.sides)
        values = np.zeros_like(p)
        values[inside, 2] = current / area
        return values.reshape(shape)


# ## Model and solve
#
# Magnetic insulation is applied explicitly to all six faces: the four walls of the square and the
# two slab faces. The source is defined on every cell of the domain and is non-zero only inside the
# meshed wire polygon.

GAUSS_BARY = np.full((4, 4), 0.1381966011250105) + np.eye(4) * (0.5854101966249685 - 0.1381966011250105)


def solve_wire(s, n_theta):
    geometry, mesh, thickness, n_triangles = slab_mesh(s, n_theta)
    wire = RoundWire(centre=s / 2, sides=n_theta, support=geometry.volume)
    model = Magnetostatics(
        domain=geometry.volume, permeability=VacuumPermeability(),
        formulation=MagneticVectorPotential(gauge=CoulombGauge()), current_sources=(wire,),
    )
    problem = Problem(
        name="Infinite wire in an insulated square",
        geometry=geometry, physics=(model,),
        boundary_conditions=(MagneticInsulation(
            boundary=geometry.boundaries.x_min | geometry.boundaries.x_max | geometry.boundaries.y_min
            | geometry.boundaries.y_max | geometry.boundaries.z_min | geometry.boundaries.z_max,
            physics=model),),
    )
    start = time.perf_counter()
    result = Simulation(
        problem=problem, mesh=mesh,
        numerics=Numerics(spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=SparseDirect()),
    ).solve_stationary()
    B_field = result.field(model.magnetic_flux_density)
    A_field = result.field(model.vector_potential)

    # Energy and flux linkage per unit length; B is linear and A quadratic on each tetrahedron, so the
    # 4-point Gauss rule integrates |B|^2 and J.A exactly.
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.cells)]
    volumes = np.abs(np.einsum("ni,ni->n", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                               corners[:, 3] - corners[:, 0])) / 6.0
    gauss_points = np.einsum("qv,nvd->nqd", GAUSS_BARY, corners).reshape(-1, 3)
    B_gauss = np.asarray(B_field.evaluate(gauss_points)).reshape(-1, 4, 3)
    energy = np.sum(volumes * (B_gauss**2).sum(axis=2).mean(axis=1)) / (2 * mu0) / thickness
    J_gauss = wire.current_density(gauss_points).reshape(-1, 4, 3)
    A_gauss = np.asarray(A_field.evaluate(gauss_points)).reshape(-1, 4, 3)
    linkage = np.sum(volumes * np.einsum("nqc,nqc->nq", J_gauss, A_gauss).mean(axis=1)) / thickness
    elapsed = time.perf_counter() - start
    print(f"s = {100*s:4.0f} cm, {n_theta:3d} cells around the wire: {n_triangles} triangles, "
          f"{mesh.cell_count} tetrahedra, slab {1e3*thickness:.2f} mm, solved in {elapsed:.0f} s")
    return dict(s=s, n=n_theta, B=B_field, A=A_field, z_mid=thickness / 2, thickness=thickness,
                L_energy=2 * energy / current**2, L_linkage=linkage / current**2, cells=mesh.cell_count)


runs = [solve_wire(side, n) for n in mesh_levels]
small = solve_wire(small_side, mesh_levels[1])
final = runs[-1]


# ## Comparison with the reference
#
# **Inductance per unit length** from the field energy, $L'=2W'/I^2$, and from the flux linkage,
# $L'=\int\mathbf J\cdot\mathbf A\,\mathrm dA/I^2$, against the exact value; the observed order and a
# Richardson estimate come from the three meshes (refinement ratio 2). For the Galerkin solution the
# two routes are identical up to round-off, so their agreement checks the post-processing, not the
# discretisation; the discretisation error shows in the comparison with the exact value.

L_ref = inductance_exact(side)
print("\nInductance per unit length, s = 20 cm   (exact %.4f nH/m)" % (L_ref * 1e9))
print(" cells around wire   tetrahedra   L' energy [nH/m]   dev        L' linkage [nH/m]   dev")
for run in runs:
    print(f" {run['n']:>17d}   {run['cells']:>10d}   {run['L_energy']*1e9:>16.4f}   {100*(run['L_energy']/L_ref-1):+.4f} %"
          f"   {run['L_linkage']*1e9:>17.4f}   {100*(run['L_linkage']/L_ref-1):+.4f} %")
e1, e2, e3 = (abs(run["L_energy"] - L_ref) for run in runs)
order = np.log2(e1 / e2), np.log2(e2 / e3)
L1, L2, L3 = (run["L_energy"] for run in runs)
p_obs = np.log2(abs(L1 - L2) / abs(L2 - L3))
L_rich = L3 + (L3 - L2) / (2**p_obs - 1)
print(f"observed order from the errors: {order[0]:.2f}, {order[1]:.2f};  from successive differences: {p_obs:.2f}")
print(f"change between the last two meshes: {100*abs(L3/L2-1):.4f} %;  Richardson estimate {L_rich*1e9:.4f} nH/m "
      f"({100*(L_rich/L_ref-1):+.4f} % from exact)")
L_small_ref = inductance_exact(small_side)
print(f"tight square s = 4 cm: L' energy {small['L_energy']*1e9:.4f} nH/m vs exact {L_small_ref*1e9:.4f} nH/m "
      f"({100*(small['L_energy']/L_small_ref-1):+.4f} %); the free-space-like value for s = 20 cm is "
      f"{100*(1-L_small_ref/L_ref):.1f} % higher")

# **Field B(x, y)** on the final mesh along two lines from the centre: towards a side midpoint (+x) and
# towards a corner (diagonal), against the exact field and the textbook law $\mu_0 I/(2\pi r)$. Inside
# the conductor the computed field is uniformly about 0.04 % high: the meshed wire is a 128-gon whose
# area is 0.04 % smaller than the circle, and it carries the full current.

c = side / 2
r_axis = np.array([0.3, 0.6, 0.9, 1.5, 2.0, 3.0, 5.0, 7.0, 8.5, 9.5]) * wire_radius
r_diag = np.array([1.5, 3.0, 6.0, 9.0, 11.0, 13.0]) * wire_radius
lines = {"towards side": (c + r_axis, np.full_like(r_axis, c), r_axis),
         "towards corner": (c + r_diag / np.sqrt(2), c + r_diag / np.sqrt(2), r_diag)}
print("\n|B| on the final mesh (s = 20 cm)")
print(" line            r/a    |B| FEM [uT]   |B| exact [uT]   dev FEM-exact   exact / textbook")
for name, (x, y, r) in lines.items():
    Bf = np.asarray(final["B"].evaluate(np.column_stack((x, y, np.full_like(x, final["z_mid"])))))
    Bx, By = exact_B(x, y, side, c)
    B_exact = np.hypot(Bx, By)
    B_text = np.where(r < wire_radius, mu0 * current * r / (2 * np.pi * wire_radius**2), mu0 * current / (2 * np.pi * r))
    B_fem = np.hypot(Bf[:, 0], Bf[:, 1])
    for k in range(len(r)):
        print(f" {name:<14s} {r[k]/wire_radius:5.1f}   {1e6*B_fem[k]:12.4f}   {1e6*B_exact[k]:14.4f}   "
              f"{100*(B_fem[k]/B_exact[k]-1):+12.4f} %   {B_exact[k]/B_text[k]:12.4f}")
    print(f"   largest |Bz| / |B| on this line: {np.max(np.abs(Bf[:, 2]) / B_fem):.1e}")

# Two-dimensionality: the field at the two slab faces must equal the mid-plane field.
probe = np.array([[c + 2 * wire_radius, c + wire_radius, z] for z in (0.0, final["z_mid"], final["thickness"])])
Bp = np.asarray(final["B"].evaluate(probe))
print(f"z-variation of B at (2a, a) across the slab: {np.max(np.abs(Bp - Bp[1])) / np.linalg.norm(Bp[1]):.1e} (relative)")

# **Ampere's law** on circles around the wire (2000 points per circle).

print("\nAmpere's circulation on the final mesh, mu0*I_enc = reference")
phi = np.linspace(0.0, 2 * np.pi, 2000, endpoint=False)
for rho in np.array([0.5, 0.9, 2.0, 5.0, 9.0]) * wire_radius:
    ring = np.column_stack((c + rho * np.cos(phi), c + rho * np.sin(phi), np.full_like(phi, final["z_mid"])))
    Bring = np.asarray(final["B"].evaluate(ring))
    circulation = np.mean(-Bring[:, 0] * np.sin(phi) + Bring[:, 1] * np.cos(phi)) * 2 * np.pi * rho
    enclosed = mu0 * current * min(1.0, (rho / wire_radius) ** 2)
    print(f" rho = {rho/wire_radius:3.1f} a:  circulation {circulation*1e6:.6f} uT m   reference {enclosed*1e6:.6f} uT m"
          f"   ({100*(circulation/enclosed-1):+.4f} %)")


# ## Figure

def field_map(run, n_grid=241):
    s = run["s"]
    g = np.linspace(0.0, s, n_grid)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, run["z_mid"])))
    pts[:, :2] = np.clip(pts[:, :2], 1e-9 * s, s * (1 - 1e-9))
    B = np.asarray(run["B"].evaluate(pts))
    Az = np.asarray(run["A"].evaluate(pts))[:, 2]
    return X, Y, np.hypot(B[:, 0], B[:, 1]).reshape(X.shape), Az.reshape(X.shape)


cream = "#fbf5eb"
fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in axes.ravel():
    ax.set_facecolor("#fffaf2")

for ax, run, title in ((axes[0, 0], final, "s = 20 cm (half side 10 a)"),
                       (axes[0, 1], small, "tight box s = 4 cm (half side 2 a)")):
    X, Y, Bmag, Az = field_map(run)
    off = run["s"] / 2
    im = ax.contourf(1e2 * (X - off), 1e2 * (Y - off), np.log10(1e6 * np.maximum(Bmag, 1e-9)),
                     levels=np.linspace(-0.6, 1.35, 60), cmap="magma", extend="both")
    ax.contour(1e2 * (X - off), 1e2 * (Y - off), Az, levels=14, colors="white", linewidths=0.7)
    ax.add_patch(plt.Circle((0, 0), 1e2 * wire_radius, fill=False, color="#7fd3c4", lw=1.2, ls="--"))
    ax.set_aspect("equal")
    ax.set(xlabel="x / cm", ylabel="y / cm", title=f"|B| and flux lines, {title}")
    fig.colorbar(im, ax=ax, label="log10 (|B| / uT)", shrink=0.85, ticks=np.arange(-0.5, 1.5, 0.25))

ax = axes[1, 0]
ax.grid(alpha=0.25)
r_line = np.linspace(0.02, 9.9, 300) * wire_radius
text = np.where(r_line < wire_radius, mu0 * current * r_line / (2 * np.pi * wire_radius**2), mu0 * current / (2 * np.pi * r_line))
ax.plot(r_line / wire_radius, 1e6 * text, color="#214D43", lw=1.2, ls=":", label="textbook law (unbounded)")
for name, (dx, dy), colour in (("towards side", (1, 0), "#A85539"), ("towards corner", (1 / np.sqrt(2), 1 / np.sqrt(2)), "#3887c5")):
    rr = np.linspace(0.02, 9.9 if dy == 0 else 13.9, 300) * wire_radius
    Bx, By = exact_B(c + rr * dx, c + rr * dy, side, c)
    ax.plot(rr / wire_radius, 1e6 * np.hypot(Bx, By), color=colour, lw=1.8, label=f"exact, {name}")
    rs = np.linspace(0.25, 9.75 if dy == 0 else 13.75, 26) * wire_radius
    Bf = np.asarray(final["B"].evaluate(np.column_stack((c + rs * dx, c + rs * dy, np.full_like(rs, final["z_mid"])))))
    ax.plot(rs / wire_radius, 1e6 * np.hypot(Bf[:, 0], Bf[:, 1]), "o", ms=4, color=colour, mfc="none",
            label=f"OneSimulation, {name}")
rr = np.linspace(0.02, 1.98, 200) * wire_radius
Bx, By = exact_B(small_side / 2 + rr, np.full_like(rr, small_side / 2), small_side, small_side / 2)
ax.plot(rr / wire_radius, 1e6 * np.hypot(Bx, By), color="#6b8e23", lw=1.5, ls="--", label="exact, tight box, towards side")
ax.axvline(1.0, color="#6b442f", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set(xlabel="distance from the wire axis r / a", ylabel="|B| / uT", title="Field along lines from the centre")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.grid(alpha=0.25, which="both")
n_vals = np.array([run["n"] for run in runs])
ax.loglog(n_vals, [abs(run["L_energy"] / L_ref - 1) for run in runs], "o-", color="#A85539", label="L' from field energy")
ax.loglog(n_vals, [abs(run["L_linkage"] / L_ref - 1) for run in runs], "s--", color="#3887c5", label="L' from flux linkage")
ax.loglog(n_vals, abs(runs[0]["L_energy"] / L_ref - 1) * (n_vals[0] / n_vals) ** 2, color="grey", lw=0.8, ls=":", label="slope -2")
ax.set(xlabel="cells around the wire circumference", ylabel="relative deviation from exact L'",
       title=f"Mesh convergence (exact L' = {L_ref*1e9:.2f} nH/m)")
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Move the wire off centre: the conformal map still gives the exact field, now with the source mapped to an
# interior point $w_0$ of the disc and the disc automorphism $(w-w_0)/(1-\bar w_0 w)$. Replace the
# square by a rectangle, or give the wire a permeability $\mu_r\neq1$: the field inside scales with
# $\mu_r$ while Ampere's circulation outside stays $\mu_0 I$.
