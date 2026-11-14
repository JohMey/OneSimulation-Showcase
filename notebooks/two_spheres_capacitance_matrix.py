"""OneSimulation example: capacitance matrix of two conducting spheres.
Requires the installed, activated OneSimulation runtime.

The full 2x2 Maxwell capacitance matrix of two unequal spheres is computed from a
3D finite-element field (curved quadratic tetrahedra) and compared with the exact
bispherical-coordinate series solution, through two independent routes: field
energy and Gauss-law charge.
"""


# # Capacitance matrix of two spheres
#
# Two conducting spheres in free space, radii $a=0.1$ m and $b=0.3$ m, centres
# $c=0.5$ m apart (a gap of only $0.1$ m), carry charges that depend linearly on both
# potentials:
#
# $$\begin{pmatrix}Q_1\\Q_2\end{pmatrix}=
#   \begin{pmatrix}C_{11}&C_{12}\\C_{21}&C_{22}\end{pmatrix}
#   \begin{pmatrix}V_1\\V_2\end{pmatrix}.$$
#
# This **Maxwell capacitance matrix** is symmetric ($C_{12}=C_{21}$), its diagonal is
# positive and its off-diagonal term negative. The circuit-style **mutual capacitance
# matrix** follows from it: each sphere has a capacitance to ground (infinity)
# $C_{11}+C_{12}$ and $C_{22}+C_{21}$, and the two spheres are coupled by $-C_{12}$.
#
# Column $j$ of the matrix is obtained by holding sphere $j$ at $1$ V and the other at
# $0$ V, solving Laplace's equation $\nabla\cdot(\varepsilon_0\nabla V)=0$ in the air
# around them and measuring the charge on both spheres. The field is genuinely
# three-dimensional: it is strongly concentrated in the narrow gap and decays like
# $1/r$ far away.
#
# **Reference (exact solution).** In bispherical coordinates the problem separates and
# the capacitance coefficients are rapidly converging series (W. R. Smythe, *Static and
# Dynamic Electricity*, McGraw-Hill 1950, ch. 5; J. Lekner, "Capacitance coefficients
# of two spheres", *J. Electrostatics* 69 (2011) 11-14):
#
# $$C_{11}=4\pi\varepsilon_0\,ab\sinh u\sum_{n=0}^{\infty}\frac{1}{a\sinh nu+b\sinh(n+1)u},\qquad
#   C_{22}=4\pi\varepsilon_0\,ab\sinh u\sum_{n=0}^{\infty}\frac{1}{b\sinh nu+a\sinh(n+1)u},$$
#
# $$C_{12}=-4\pi\varepsilon_0\,\frac{ab}{c}\sinh u\sum_{n=1}^{\infty}\frac{1}{\sinh nu},\qquad
#   \cosh u=\frac{c^2-a^2-b^2}{2ab}.$$
#
# For a very distant partner $C_{11}\to4\pi\varepsilon_0 a$, the capacitance of an
# isolated sphere.

from dataclasses import dataclass
import time
import numpy as np

from onesim.cad.d3 import Box, Regrouped, Sphere
from onesim.materials import VacuumPermittivity
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement,
)
from onesim.results.derived import derived_field
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()
start_time = time.perf_counter()


# ## Parameters

a, b_radius = 0.1, 0.3
c = 0.5
x1, x2 = -c / 2, c / 2
outer_radius = 4.0
eps0 = float(VacuumPermittivity().evaluate())
F = 4 * np.pi * eps0

mesh_levels = (0.034, 0.024, 0.017)
study_radii = (2.0, 3.0)
print(f"a = {a} m, b = {b_radius} m, c = {c} m, gap = {c - a - b_radius:.2f} m, outer radius {outer_radius} m")


# ## Reference: exact series solution

def series_capacitance(a, b, c, terms=200):
    """Maxwell capacitance matrix of two spheres from the bispherical series."""
    u = np.arccosh((c * c - a * a - b * b) / (2 * a * b))
    n = np.arange(terms)
    m = np.arange(1, terms)
    with np.errstate(over="ignore"):
        c11 = F * a * b * np.sinh(u) * np.sum(1 / (a * np.sinh(n * u) + b * np.sinh((n + 1) * u)))
        c22 = F * a * b * np.sinh(u) * np.sum(1 / (b * np.sinh(n * u) + a * np.sinh((n + 1) * u)))
        c12 = -F * a * b / c * np.sinh(u) * np.sum(1 / np.sinh(m * u))
    return np.array([[c11, c12], [c12, c22]])


C_ref = series_capacitance(a, b_radius, c)
C_far = series_capacitance(a, b_radius, 1000.0)
print("Exact Maxwell capacitance matrix / pF:")
print(f"  C11 = {C_ref[0, 0]*1e12:.4f}   C12 = {C_ref[0, 1]*1e12:.4f}   C22 = {C_ref[1, 1]*1e12:.4f}")
print(f"Isolated-sphere limit of the series (c = 1000 m): C11 / (4 pi eps0 a) = {C_far[0, 0]/(F*a):.6f}, "
      f"C22 / (4 pi eps0 b) = {C_far[1, 1]/(F*b_radius):.6f}")


# ## Geometry and mesh
#
# The arrangement is rotationally symmetric about the line of centres (the $x$ axis),
# so the planes $y=0$ and $z=0$ are symmetry planes on which the normal electric
# displacement vanishes. The air domain is therefore one quarter of a large sphere of
# radius $R$ with the two conducting spheres removed. It is declared with OneSimulation's
# CAD solids and Boolean operators: the outer sphere intersected with a box gives the
# quarter, and the conductors are cut out as named negative bodies, so their surfaces
# become the boundary groups `sphere_1` and `sphere_2`. `TetrahedralMesher` meshes it
# with **curved quadratic tetrahedra** (edges projected onto the exact spherical
# surfaces). The mesh is refined on both spheres, in a ball that contains the gap and,
# more mildly, in the near field; it coarsens towards the outer sphere. All sizes scale
# with one level parameter $h$ (cell size on the small sphere and across the gap).

def air_domain(spheres, radius):
    """Quarter of a grounded outer sphere with the conducting spheres [(x, r), ...] removed."""
    air = Sphere(radius=radius) & Box(length=2 * radius, width=radius, height=radius,
                                      origin=(-radius, 0.0, 0.0)).named("plane")
    for i, (centre, r) in enumerate(spheres):
        air = air - Sphere(radius=r, origin=(centre, 0.0, 0.0)).named(f"sphere_{i + 1}")
    return Regrouped(source=air, groups={"outer": ("surface",), "section": ("plane.z_min",),
                                         "symmetry": ("plane.y_min", "plane.z_min")})


def quarter_mesh(spheres, radius, h):
    domain = air_domain(spheres, radius)
    b = domain.boundaries
    refinements = [LocalRefinement(region=getattr(b, f"sphere_{i + 1}"), target_size=h * min(r / a, 2.0))
                   for i, (_, r) in enumerate(spheres)]
    if len(spheres) == 2:
        gap_centre = 0.5 * ((x1 + a) + (x2 - b_radius))
        refinements.append(LocalRefinement(region=Sphere(radius=0.12, origin=(gap_centre, 0.0, 0.0)), target_size=h))
    refinements.append(LocalRefinement(region=Sphere(radius=0.6), target_size=6 * h))
    mesher = TetrahedralMesher(target_size=0.15 * radius, refinements=tuple(refinements), geometry_order=2,
                               max_candidate_cells=1_000_000)
    return domain, mesher.generate(domain)


# ## Model
#
# **Numerical method.**
# - Spatial method: continuous Galerkin finite elements, isoparametric quadratic Lagrange
#   elements (P2) on curved 10-node tetrahedra whose geometry is also quadratic, with a
#   quadrature of order 5 for the curved-cell integrals.
# - Time integration: none (stationary electrostatics).
# - Solver: one linear system per excitation, solved with a sparse direct factorisation.
#
# Boundary conditions on every face of the quarter domain: fixed potential on both
# spheres, zero potential on the outer sphere, zero normal electric displacement on the
# two symmetry planes.

def solve(domain, mesh, potentials):
    """Solve Laplace's equation with the given sphere potentials (outer sphere grounded)."""
    b = domain.boundaries
    model = PoissonEquation(domain=domain.volume, permittivity=VacuumPermittivity())
    conditions = [FixedPotential(boundary=getattr(b, f"sphere_{i + 1}"), physics=model, value=v)
                  for i, v in enumerate(potentials)]
    conditions += [
        FixedPotential(boundary=b.outer, physics=model, value=0.0),
        ZeroNormalElectricDisplacement(boundary=b.symmetry, physics=model),
    ]
    problem = Problem(name="Two-sphere capacitance", geometry=domain, physics=(model,),
                      boundary_conditions=tuple(conditions))
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2), quadrature_order=5),
                        linear_solver=SparseDirect())
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    return result.field(model.electric_field), result.field(model.potential)


# Two independent routes to the charges:
#
# 1. **Field energy.** The reaction of the solved potential on a sphere,
#    `potential.reaction(sphere)`, is the discrete mutual field energy
#    $\varepsilon_0\int\nabla V_i\cdot\nabla V_j\,dV$ of the two unit solutions, i.e. the
#    charge on sphere $i$ when sphere $j$ is at 1 V (factor 4 for the quarter domain).
# 2. **Gauss's law over a shell.** $Q=-\varepsilon_0\int\mathbf E\cdot\nabla\eta\,dV$ with a
#    smooth ramp $\eta$ that is 1 on the sphere and 0 beyond a shell around it: Gauss's
#    law weighted over many closed surfaces, integrated from the solved field.

def shell_charge(E, centre, r_in, r_out):
    def weighted_radial_field(values, points):
        offset = points - np.array([centre, 0.0, 0.0])
        r = np.linalg.norm(offset, axis=1)
        s = np.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
        d_eta = -6.0 * s * (1.0 - s) / (r_out - r_in)
        return np.einsum("ij,ij->i", values, offset) / r * d_eta

    integrand = derived_field(E, weighted_radial_field, name="gauss_weight", unit="V/m^2")
    return -4.0 * eps0 * float(integrand.integrate().value)


def capacitance_matrices(domain, mesh):
    """Maxwell matrix from (i) the mutual field energy and (ii) Gauss-law charges."""
    b = domain.boundaries
    fields = [solve(domain, mesh, (1.0, 0.0)), solve(domain, mesh, (0.0, 1.0))]
    spheres = (b.sphere_1, b.sphere_2)
    C_energy = np.array([[4.0 * fields[j][1].reaction(spheres[i]) for j in range(2)] for i in range(2)])
    shells = ((x1, a, c - b_radius - 0.01), (x2, b_radius, c - a - 0.01))
    C_charge = np.array([[shell_charge(fields[j][0], *shells[i]) for j in range(2)] for i in range(2)])
    return C_energy, C_charge, fields


def free_space(C, radius, centres=(x1, x2)):
    """Remove the grounded outer sphere: Kelvin-image correction of the potential coefficients.

    For charges near the centre of a grounded sphere of radius R, the image charge adds
    -1/(4 pi eps0 R) - (x.x')/(4 pi eps0 R^3) + ... to every potential coefficient.
    Adding these terms back to P = C^-1 gives the free-space matrix to O(R^-5).
    """
    d = np.asarray(centres, float)
    P = np.linalg.inv(C) + 1.0 / (F * radius) + np.outer(d, d) / (F * radius**3)
    return np.linalg.inv(P)


# ## Solve
#
# **Isolated-sphere check.** A single sphere of radius $a$ at the centre of the grounded
# outer sphere has the exact capacitance $4\pi\varepsilon_0\,aR/(R-a)$ (a spherical
# capacitor); after the image correction this is the isolated-sphere value
# $4\pi\varepsilon_0 a$.

domain_1, mesh_1 = quarter_mesh(((0.0, a),), outer_radius, mesh_levels[0])
E_single, V_single = solve(domain_1, mesh_1, (1.0,))
C_single_energy = 4.0 * V_single.reaction(domain_1.boundaries.sphere_1)
C_single_charge = shell_charge(E_single, 0.0, a, 2 * a)
C_single_exact = F * a * outer_radius / (outer_radius - a)
print(f"Single sphere ({mesh_1.cell_count} curved tetrahedra): energy {C_single_energy*1e12:.4f} pF, "
      f"charge {C_single_charge*1e12:.4f} pF, exact spherical capacitor {C_single_exact*1e12:.4f} pF")

# **Mesh study** at $R=4$ m: three meshes, each solved twice (one sphere at 1 V).

study = []
for h in mesh_levels:
    domain, mesh = quarter_mesh(((x1, a), (x2, b_radius)), outer_radius, h)
    C_energy, C_charge, fields = capacitance_matrices(domain, mesh)
    study.append(dict(h=h, cells=mesh.cell_count, nodes=len(mesh.coordinates),
                      energy=free_space(C_energy, outer_radius),
                      charge=free_space(C_charge, outer_radius), raw=C_energy))
    print(f"h = {h:.3f} m: {mesh.cell_count:6d} curved tetrahedra, {len(mesh.coordinates):6d} quadratic nodes")
final_fields, final_domain = fields, domain

# **Outer-radius study** on the coarsest mesh level: the raw (truncated-domain) matrix
# changes like $1/R$, the image-corrected matrix must not.

radius_study = []
for radius in study_radii + (outer_radius,):
    if radius == outer_radius:
        C_raw = study[0]["raw"]
    else:
        domain, mesh = quarter_mesh(((x1, a), (x2, b_radius)), radius, mesh_levels[0])
        C_raw = capacitance_matrices(domain, mesh)[0]
    radius_study.append((radius, C_raw, free_space(C_raw, radius)))


# ## Comparison with the reference
#
# Reference: the exact bispherical series (Smythe 1950; Lekner 2011) for the same radii
# and separation. The finite-element matrices are shown after the image correction of
# the grounded outer sphere. The two independent routes are compared: the field energy
# (by Dirichlet's principle the conforming finite-element energy approaches the exact
# diagonal terms from above) and the charges from Gauss's law, whose off-diagonal
# terms $C_{12}$ and $C_{21}$ come from two different solves and therefore also test
# reciprocity.


def rel(value, reference):
    return 100.0 * (value - reference) / abs(reference)


entries = (("C11", 0, 0), ("C22", 1, 1), ("C12", 0, 1), ("C21", 1, 0))
print("\nMesh study (energy route | charge route), values in pF, deviation from the series in %:")
for row in study:
    text = "  ".join(f"{name} {row['energy'][i, j]*1e12:8.4f} ({rel(row['energy'][i, j], C_ref[i, j]):+.3f}) | "
                     f"{row['charge'][i, j]*1e12:8.4f} ({rel(row['charge'][i, j], C_ref[i, j]):+.3f})"
                     for name, i, j in entries[:3])
    print(f"  h = {row['h']:.3f} m, {row['cells']:6d} cells:  {text}")
fine, medium = study[-1], study[-2]
print("Change between the last two meshes (energy route): "
      + ", ".join(f"{name} {rel(fine['energy'][i, j], medium['energy'][i, j]):+.3f} %" for name, i, j in entries[:3]))
print("Change between the last two meshes (charge route): "
      + ", ".join(f"{name} {rel(fine['charge'][i, j], medium['charge'][i, j]):+.3f} %" for name, i, j in entries[:3]))

C_fem = fine["energy"]
C_q = fine["charge"]
print("\nFinest mesh against the exact series:")
print(f"{'':>6} {'series':>10} {'energy':>10} {'dev %':>8} {'charge':>10} {'dev %':>8}")
for name, i, j in entries:
    print(f"{name:>6} {C_ref[i, j]*1e12:10.4f} {C_fem[i, j]*1e12:10.4f} {rel(C_fem[i, j], C_ref[i, j]):+8.3f} "
          f"{C_q[i, j]*1e12:10.4f} {rel(C_q[i, j], C_ref[i, j]):+8.3f}")
print(f"Reciprocity from the independent charge columns: C12 = {C_q[0, 1]*1e12:.4f} pF, "
      f"C21 = {C_q[1, 0]*1e12:.4f} pF ({rel(C_q[0, 1], C_q[1, 0]):+.3f} %)")

mutual_ref = np.array([C_ref[0, 0] + C_ref[0, 1], C_ref[1, 1] + C_ref[1, 0], -C_ref[0, 1]])
mutual_fem = np.array([C_fem[0, 0] + C_fem[0, 1], C_fem[1, 1] + C_fem[1, 0], -C_fem[0, 1]])
print("\nMutual capacitances (energy route): "
      + ", ".join(f"{name} {v*1e12:.4f} pF (series {r*1e12:.4f}, {rel(v, r):+.3f} %)"
                  for name, v, r in zip(("sphere 1 to ground", "sphere 2 to ground", "sphere 1 to sphere 2"),
                                        mutual_fem, mutual_ref)))
print(f"Stored energy with both spheres at 1 V: {0.5*np.sum(C_fem)*1e12:.4f} pJ "
      f"(series {0.5*np.sum(C_ref)*1e12:.4f} pJ)")

print(f"\nIsolated sphere: energy route {rel(C_single_energy, C_single_exact):+.3f} %, "
      f"charge route {rel(C_single_charge, C_single_exact):+.3f} % against 4 pi eps0 aR/(R-a); "
      f"image-corrected {free_space(np.array([[C_single_energy]]), outer_radius, (0.0,))[0, 0]*1e12:.4f} pF "
      f"vs 4 pi eps0 a = {F*a*1e12:.4f} pF")

print("\nOuter-radius study (coarsest mesh level, energy route), C11 / C22 / C12 in pF:")
for radius, C_raw, C_corr in radius_study:
    print(f"  R = {radius:.0f} m: raw {C_raw[0, 0]*1e12:8.4f} {C_raw[1, 1]*1e12:8.4f} {C_raw[0, 1]*1e12:8.4f}   "
          f"image-corrected {C_corr[0, 0]*1e12:8.4f} {C_corr[1, 1]*1e12:8.4f} {C_corr[0, 1]*1e12:8.4f}")
print(f"Total run time: {time.perf_counter() - start_time:.0f} s")


# ## Figure
#
# Left: the potential on the symmetry plane $z=0$ with sphere 1 at 1 V and sphere 2
# grounded, mirrored about $y=0$ to show the full section, with equipotentials every
# 0.05 V. Right: mesh convergence of both routes against the exact series, and the
# outer-boundary study.

V_field = final_fields[0][1]
cells = np.array([row["cells"] for row in study])
radii = np.array([item[0] for item in radius_study])
index = {name: (i, j) for name, i, j in entries[:3]}
colour = {name: n for n, name in enumerate(index)}
fig = plots.figure(layout=[["map", "mesh"], ["map", "radius"]], size=(14, 7.6))
plots.surface(V_field, boundary=final_domain.boundaries.section, symmetry={"planes": [((0, 1, 0), (0, 0, 0))]},
              limits=(0.0, 1.0), contours=19, subdivisions=3, label="potential [V]",
              title="Section z = 0 of the 3D field (sphere 1 at 1 V, sphere 2 grounded)",
              xlimits=(-0.55, 0.65), ylimits=(-0.6, 0.6), ax=fig["map"])
plots.line(cells, {**{f"{name}, field energy": [abs(rel(row["energy"][ij], C_ref[ij])) for row in study] for name, ij in index.items()},
                   **{f"{name}, Gauss-law charge": [abs(rel(row["charge"][ij], C_ref[ij])) for row in study] for name, ij in index.items()}},
           colors={f"{name}, {kind}": colour[name] for name in index for kind in ("field energy", "Gauss-law charge")},
           styles={f"{name}, Gauss-law charge": "dashed" for name in index}, markers=True, scale="loglog",
           xlabel="curved quadratic tetrahedra (quarter domain)", ylabel="|deviation from exact series| [%]",
           title="Mesh convergence at R = 4 m", ax=fig["mesh"])
plots.line(1 / radii, {**{f"{name}, truncated domain": [rel(item[1][ij], C_ref[ij]) for item in radius_study] for name, ij in index.items()},
                       **{f"{name}, image-corrected": [rel(item[2][ij], C_ref[ij]) for item in radius_study] for name, ij in index.items()}},
           colors={f"{name}, {kind}": colour[name] for name in index for kind in ("truncated domain", "image-corrected")},
           styles={f"{name}, truncated domain": "dotted" for name in index}, markers=True, hlines=0.0,
           xlabel="1 / R (outer radius R in m)", ylabel="deviation from exact series [%]",
           title="Outer-boundary study (coarsest mesh)", ax=fig["radius"])
plots.show()


# ## Going further
#
# Move the spheres closer together: the gap field and the coupling $-C_{12}$ grow
# rapidly and the series needs more terms, while the mesh needs a finer ball in the
# gap. Give the spheres equal radii to recover the symmetric case, or place a
# dielectric between them by meshing an additional region. With more than two
# conductors the same recipe (one unit-potential solve per conductor) gives the full
# $N\times N$ matrix.
