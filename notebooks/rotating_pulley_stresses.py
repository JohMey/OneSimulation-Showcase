"""OneSimulation example: stresses in a belt-driven pulley as a function of speed.
Requires the installed, activated OneSimulation runtime.

A spoked aluminium pulley transmits a constant shaft power to a flat belt. The 2D
plane-stress field is solved with quadratic triangles over a speed sweep and is
checked by global equilibrium, the closed-form rotating-disc solution, a three-level
mesh study and an independent cell-centred finite-volume solution of the 3D plate.
"""


# # Stresses in a rotating pulley
#
# A driving pulley of outer radius $R=125$ mm sits on a shaft (bore radius 25 mm) and
# drives a flat belt that wraps it over the angle $\beta=170^\circ$. Eight rounded,
# trapezoidal cut-outs leave a hub, eight tapered spokes and a rim. The shaft power $P$
# is constant, so the torque $M=P/\omega$ falls as the speed rises, while the
# centrifugal load grows with $\omega^2$: at low speed the belt dominates, at high
# speed the inertia does.
#
# **Belt load.** With friction coefficient $\mu$ and impending slip, the strand
# tensions obey the capstan (Euler-Eytelwein) relation $F_1/F_2=e^{\mu\beta}$ and
# $F_1-F_2=M/R$. Along the contact, measured by the angle $\theta$ from the point
# where the slack strand leaves, the belt presses on the rim with the line loads
#
# $$F_n(\theta)=\frac{F_2}{R}\,e^{\mu\theta}\ \text{(inward)},\qquad
#   F_t(\theta)=\mu F_n(\theta)\ \text{(against the rotation)},$$
#
# which follow from the equilibrium of a belt element. Divided by the pulley thickness
# $d$ they become the boundary traction. **Inertia.** In the rotating frame every
# material point carries the centrifugal body force $\rho\,\omega^2\,\mathbf r$.
# The bore is clamped to the (much stiffer) shaft. Dynamic effects are ignored.
#
# **Method.** The pulley is thin, so it is a plane-stress problem in the $(x, y)$
# plane. OneSimulation's `SmallStrainSolid` with `kinematics="plane_stress"` solves it
# by the continuous Galerkin finite-element method with quadratic (P2) Lagrange
# triangles on a graded unstructured mesh; the problem is linear and stationary, so
# the Newton solver converges in one step with a sparse direct LU factorisation. As an
# independent second discretisation the same pulley is solved as a thin 3D plate with
# traction-free faces by OneSimulation's cell-centred finite-volume elasticity
# operator (`FiniteVolume` with boundary-aware least-squares gradients and a central
# full-stress face flux), whose field depends on $(x, y)$ only.
#
# **Reference.** There is no closed-form solution for the spoked pulley. The notebook
# uses (i) global equilibrium: the shaft reaction must balance the two strand forces
# $F_1\,\mathbf t_1+F_2\,\mathbf t_2$ and the torque $M$, (ii) the exact plane-stress
# solution of a rotating annular disc clamped at the bore (Timoshenko & Goodier,
# *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970, Sec. 32) on the pulley blank
# without cut-outs, solved with the same mesher and solver, (iii) a three-level mesh
# study with observed order and Richardson extrapolation, and (iv) the agreement of
# the finite-element and finite-volume solutions.

from dataclasses import dataclass
from importlib.metadata import version
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
import netgen.occ as occ
import numpy as np
from matplotlib.tri import Triangulation
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fvm import BoundaryLeastSquaresGradient, CentralElasticStress, FiniteVolume
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (BodyForce, FixedDisplacement, LinearElasticity, SmallStrainSolid,
                                            Traction, TractionFree)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Pulley dimensions in millimetres: outer radius 125, bore radius 25; each cut-out is
# a trapezoid between the chords at 48 mm and 100 mm from the centre whose sides make
# $\pm 14.8^\circ$ with its axis, with corner fillets of 11 mm (inner) and 16 mm
# (outer); the cut-outs repeat every $45^\circ$. Aluminium: $E=70$ GPa, $\nu=0.33$,
# $\rho=2700$ kg/m³. Drive: $P=30$ kW, $\mu=0.3$, $\beta=170^\circ$, thickness
# $d=10$ mm, speeds 1000-9000 rpm. The belt runs onto the pulley at the top (tight
# strand $F_1$), wraps the right-hand side and leaves at the bottom (slack strand
# $F_2$), so the pulley turns clockwise and the contact spans the polar angles
# $-85^\circ\ldots+85^\circ$.

mm = 1e-3
R_out, r_bore = 125 * mm, 25 * mm
y_inner, y_outer = 48 * mm, 100 * mm            # chords bounding a cut-out
side_slope, side_ref = 0.265, (19.3 * mm, 56 * mm)   # cut-out side: x = x0 + slope (y - y0)
r_fillet_in, r_fillet_out = 11 * mm, 16 * mm
n_cutouts = 8
thickness = 10 * mm

E, nu, rho = 70e9, 0.33, 2700.0
power, mu_belt, beta = 30e3, 0.3, np.radians(170.0)
phi_slack, phi_tight = np.radians(-85.0), np.radians(85.0)   # contact arc
speeds_shown = (1000.0, 9000.0)                              # rpm
probe = np.array([0.019, 0.054])                             # m, on the inner fillet of the top cut-out


def omega(rpm):
    return 2 * np.pi * rpm / 60.0


def strand_forces(rpm):
    """Peripheral, tight and slack strand forces for constant power (capstan relation)."""
    F_u = power / omega(rpm) / R_out
    F_2 = F_u / (np.exp(mu_belt * beta) - 1.0)
    return F_u, F_2 * np.exp(mu_belt * beta), F_2


for rpm in speeds_shown:
    F_u, F_1, F_2 = strand_forces(rpm)
    print(f"{rpm:6.0f} rpm: torque M = {power / omega(rpm):7.2f} N m, F1 = {F_1:7.1f} N, F2 = {F_2:6.1f} N")


# ## Geometry and mesh
#
# OneSimulation's CAD primitives do not include this spoked section, so the face is
# drawn with the OpenCASCADE kernel shipped with Netgen (`netgen.occ`): a disc whose
# rim is split at the two belt tangent points, minus the bore and eight cut-outs made
# of straight sides and tangent circular fillets. Netgen triangulates the face with
# local edge sizes (finest on the inner fillets, where the stress concentrates, coarser
# on the rim) and a smooth grading. The triangles and the named boundary edges are
# handed to `UnstructuredSimplexMesh` with a small `Geometry` subclass that declares
# the boundary groups `bore`, `belt` (the contact arc), `rim_free` and `cutouts`.
# The base mesh has 1.4 mm edges on the inner fillets, 2.2 mm on the outer fillets,
# 3.5 mm on the straight cut-out sides and the bore and 7 mm on the rim. Finer meshes
# come from uniform refinement (every triangle split into four, new boundary points
# placed on the exact curves), so the element size halves exactly from level to level.

def rounded_cutout(angle, h_fillet, h_line):
    x_a = side_ref[0] + side_slope * (y_inner - side_ref[1])
    x_b = side_ref[0] + side_slope * (y_outer - side_ref[1])
    corners = np.array([(x_a, y_inner), (x_b, y_outer), (-x_b, y_outer), (-x_a, y_inner)])
    radii = (r_fillet_in, r_fillet_out, r_fillet_out, r_fillet_in)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    turns, entries, exits = [], [], []
    for i, corner in enumerate(corners):
        d_in = corner - corners[i - 1]
        d_out = corners[(i + 1) % 4] - corner
        d_in, d_out = d_in / np.linalg.norm(d_in), d_out / np.linalg.norm(d_out)
        turn = np.arctan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], d_in @ d_out)
        cut = radii[i] * np.tan(abs(turn) / 2)
        turns.append(turn)
        entries.append(corner - cut * d_in)
        exits.append(corner + cut * d_out)
    first = (corners[0] - corners[-1]) / np.linalg.norm(corners[0] - corners[-1])
    wp = occ.WorkPlane().MoveTo(*(rot @ exits[-1])).Direction(*(rot @ first))
    current = exits[-1]
    for i in range(4):
        wp = wp.Line(float(np.linalg.norm(entries[i] - current)), name="cut_line")
        wp = wp.Arc(radii[i], float(np.degrees(turns[i])), name="fillet_in" if i in (0, 3) else "fillet_out")
        current = exits[i]
    face = wp.Close().Face()
    for edge in face.edges:
        edge.maxh = {"fillet_in": h_fillet, "fillet_out": 1.6 * h_fillet}.get(edge.name, h_line)
    return face


def pulley_face(cutouts=True):
    h_fillet, h_line, h_rim = 1.4 * mm, 3.5 * mm, 7.0 * mm
    wp = occ.WorkPlane().MoveTo(R_out * np.cos(phi_slack), R_out * np.sin(phi_slack))
    wp = wp.Direction(-np.sin(phi_slack), np.cos(phi_slack))
    disc = wp.Arc(R_out, 170.0, name="belt").Arc(R_out, 190.0, name="rim").Close().Face()
    for edge in disc.edges:
        edge.maxh = h_rim
    bore = occ.WorkPlane().Circle(0.0, 0.0, r_bore).Face()
    for edge in bore.edges:
        edge.maxh = h_line
    shape = disc - bore
    if cutouts:
        for k in range(n_cutouts):
            shape = shape - rounded_cutout(2 * np.pi * k / n_cutouts, h_fillet, h_line)
    return shape, 8.4 * mm


@dataclass(frozen=True, eq=False, kw_only=True)
class PulleySection(Geometry):
    """Plane pulley section with its four boundary groups."""

    dimension = 2

    def boundary_groups(self):
        names = ("bore", "belt", "rim_free", "cutouts")
        return {**{name: (name,) for name in names}, "all": names}


def classify(points):
    """Boundary group of an edge or face from its vertex coordinates."""
    r = np.hypot(points[:, 0], points[:, 1])
    if np.all(np.abs(r - r_bore) < 1e-6):
        return "bore"
    if np.all(np.abs(r - R_out) < 1e-6):
        angle = np.arctan2(points[:, 1].mean(), points[:, 0].mean())
        return "belt" if phi_slack < angle < phi_tight else "rim_free"
    return "cutouts"


def build_section(level, cutouts=True):
    shape, maxh = pulley_face(cutouts)
    ng = occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh, grading=0.3)
    for _ in range(level):
        ng.Refine()
    xy = np.array([p.p for p in ng.Points()])[:, :2]
    triangles = np.array([[v.nr for v in e.vertices] for e in ng.Elements2D()]) - 1
    geometry = PulleySection()
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    for edge in ng.Elements1D():
        v = np.array([p.nr for p in edge.vertices]) - 1
        groups[classify(xy[v])].append(tuple(v))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=xy, cells=triangles, boundary_facets=groups)
    return dict(geometry=geometry, mesh=mesh, xy=xy, triangles=triangles, level=level, cutouts=cutouts)


# ## Model
#
# Boundary conditions on every boundary of the section:
#
# | boundary | condition |
# |---|---|
# | bore $r=25$ mm | clamped, $u_x=u_y=0$ |
# | belt contact arc $-85^\circ<\varphi<85^\circ$ | traction $(-F_n\,\mathbf e_r+F_t\,\mathbf e_\varphi)/d$ |
# | rest of the rim | traction-free |
# | cut-out edges | traction-free |
#
# With $\theta=\varphi-\varphi_\mathrm{slack}$ the tangential load points along
# $+\mathbf e_\varphi$, i.e. against the clockwise rotation. Load and speed enter
# through the load parameter `t` of the solver, which is the rotational speed in rpm.

def centrifugal(x, y, t):
    w2 = omega(t) ** 2
    return np.stack((rho * w2 * x, rho * w2 * y), axis=-1)


def belt_traction(x, y, t):
    _, _, F_2 = strand_forces(t)
    phi = np.arctan2(y, x)
    F_n = F_2 / (R_out * thickness) * np.exp(mu_belt * (phi - phi_slack))
    return np.stack((-F_n * np.cos(phi) - mu_belt * F_n * np.sin(phi),
                     -F_n * np.sin(phi) + mu_belt * F_n * np.cos(phi)), axis=-1)


plane_stress = SmallStrainSolid(youngs_modulus=E, poisson_ratio=nu, kinematics="plane_stress",
                                thickness=thickness)


def solve_section(data, speeds, with_belt=True):
    b = data["geometry"].boundaries
    tractions = ((b.rim_free, (0.0, 0.0)), (b.belt, belt_traction if with_belt else (0.0, 0.0)))
    if data["cutouts"]:
        tractions += ((b.cutouts, (0.0, 0.0)),)
    return plane_stress.solve(mesh=data["mesh"], times=tuple(speeds), order=2,
                              displacements=((b.bore, "x", 0.0), (b.bore, "y", 0.0)),
                              tractions=tractions, body_force=centrifugal)


def recover_stress(space, stress_q):
    """L2 projection of the quadrature-point stresses onto the continuous P2 space."""
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


def von_mises(s):
    """Plane-stress von Mises stress from (sxx, syy, sxy)."""
    return np.sqrt(s[..., 0] ** 2 + s[..., 1] ** 2 - s[..., 0] * s[..., 1] + 3 * s[..., 2] ** 2)


# ## Solve
#
# Each mesh is solved at 1000 and 9000 rpm. The problem is linear, so the field at any
# speed is the exact superposition $\boldsymbol\sigma(\omega)=\omega^2\boldsymbol\sigma_c
# +(P/\omega)\,\boldsymbol\sigma_b$ of a unit centrifugal case and a unit-torque belt
# case, which are recovered from the two solved speeds. Stresses are taken from the
# quadrature points and recovered as a continuous P2 field by an $L^2$ projection.

levels = (0, 1, 2)                     # number of uniform refinements of the base mesh
w_lo, w_hi = omega(speeds_shown[0]), omega(speeds_shown[1])
basis_matrix = np.array([[w_lo**2, power / w_lo], [w_hi**2, power / w_hi]])
sweep_rpm = np.arange(1000.0, 9001.0, 250.0)
runs = []
for level in levels:
    start = time.time()
    data = build_section(level)
    result = solve_section(data, speeds_shown)
    space = result.space
    nodal = [recover_stress(space, result.stress[i][:, [0, 1, 0], [0, 1, 1]]) for i in range(2)]
    unit = np.linalg.solve(basis_matrix, np.stack([n.reshape(-1) for n in nodal])).reshape(2, *nodal[0].shape)
    vm = [von_mises(n) for n in nodal]
    probe_stress = space.evaluate(np.stack(unit, axis=-1).reshape(space.dof_count, -1), probe[None, :])
    probe_unit = probe_stress.reshape(3, 2).T
    sweep_probe = np.array([von_mises(omega(n) ** 2 * probe_unit[0] + power / omega(n) * probe_unit[1])
                            for n in sweep_rpm])
    sweep_peak = np.array([von_mises(omega(n) ** 2 * unit[0] + power / omega(n) * unit[1]).max()
                           for n in sweep_rpm])
    xy_dofs = space.dof_coordinates
    on_bore = np.hypot(xy_dofs[:, 0], xy_dofs[:, 1]) < r_bore + 1e-9
    reactions = [result.reactions[i][on_bore] for i in range(2)]
    torque = [float(np.sum(xy_dofs[on_bore, 0] * R[:, 1] - xy_dofs[on_bore, 1] * R[:, 0])) for R in reactions]
    rim_point = np.array([[0.0, R_out - 1e-4]])
    rim_u = [space.evaluate(result.displacement[i], rim_point)[0] for i in range(2)]
    runs.append(dict(data=data, result=result, nodal=nodal, vm=vm, unit=unit, sweep_probe=sweep_probe,
                     sweep_peak=sweep_peak, reactions=[R.sum(axis=0) for R in reactions], torque=torque,
                     peak=[v.max() for v in vm],
                     probe=[von_mises(space.evaluate(nodal[i], probe[None, :])[0]) for i in range(2)],
                     rim_u=rim_u, dofs=2 * space.dof_count))
    print(f"level {level}: {len(data['triangles']):6d} triangles, {2 * space.dof_count:6d} unknowns, "
          f"peak von Mises {vm[0].max() / 1e6:6.2f} / {vm[1].max() / 1e6:6.2f} MPa at 1000 / 9000 rpm "
          f"({time.time() - start:.0f} s)")
fine = runs[-1]

# Pulley blank without cut-outs (rotating annular disc clamped at the bore), finest
# level, centrifugal load only at 9000 rpm.
blank = build_section(levels[-1], cutouts=False)
blank_result = solve_section(blank, (speeds_shown[1],), with_belt=False)
blank_stress = recover_stress(blank_result.space, blank_result.stress[0][:, [0, 1, 0], [0, 1, 1]])


# **Finite-volume solution of the 3D plate.** The base triangulation is swept
# through the thickness in two layers of prisms, each split into three tetrahedra by
# a conforming diagonal rule (Dompierre et al., Proc. 8th Int. Meshing Roundtable,
# 1999). Boundary conditions on every face: bore clamped ($\mathbf u=0$), belt
# traction on the contact arc, traction-free on the rest of the rim, on the cut-outs
# and on the two flat faces $z=\pm d/2$.

PRISM_ROTATIONS = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                            [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def split_prism(vertices):
    v = np.asarray(vertices)[PRISM_ROTATIONS[int(np.argmin(vertices))]]
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [(v[0], v[1], v[2], v[5]), (v[0], v[1], v[5], v[4]), (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]), (v[0], v[4], v[2], v[5]), (v[0], v[4], v[5], v[3])]


@dataclass(frozen=True, eq=False, kw_only=True)
class PulleyPlate(Geometry):
    """Thin 3D pulley plate with its five boundary groups."""

    dimension = 3

    def boundary_groups(self):
        names = ("bore", "belt", "rim_free", "cutouts", "faces")
        return {**{name: (name,) for name in names}, "all": names}


def build_plate(section, layers=2):
    xy, triangles = section["xy"], section["triangles"]
    n = len(xy)
    z = np.linspace(-thickness / 2, thickness / 2, layers + 1)
    coordinates = np.column_stack((np.tile(xy, (layers + 1, 1)), np.repeat(z, n)))
    cells = np.array([tet for j in range(layers) for a, b, c in triangles
                      for tet in split_prism((a + j * n, b + j * n, c + j * n,
                                              a + (j + 1) * n, b + (j + 1) * n, c + (j + 1) * n))])
    p = coordinates[cells]
    negative = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) < 0
    cells[negative] = cells[negative][:, [0, 2, 1, 3]]
    geometry = PulleyPlate()
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    for face in boundary_faces(cells, coordinates):
        v = np.asarray(face.vertices)
        q = coordinates[v]
        flat = np.all(np.abs(q[:, 2] - q[0, 2]) < 1e-12) and abs(abs(q[0, 2]) - thickness / 2) < 1e-12
        groups["faces" if flat else classify(q)].append(tuple(v))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells, boundary_facets=groups)
    return geometry, mesh


aluminium = IsotropicSolid(density=rho, specific_heat_capacity=900.0, thermal_conductivity=200.0,
                           youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=23e-6)
plate_geometry, plate_mesh = build_plate(runs[0]["data"])
fv_runs = []
for rpm in speeds_shown:
    start = time.time()

    def plate_loads(rpm):
        def body(x, y, z):
            return np.concatenate((centrifugal(x, y, rpm), 0 * z[..., None]), axis=-1)

        def traction(x, y, z):
            return np.concatenate((belt_traction(x, y, rpm), 0 * z[..., None]), axis=-1)
        return body, traction

    body, traction = plate_loads(rpm)
    solid = LinearElasticity(domain=plate_geometry.domain(), material=aluminium, loading=BodyForce(value=body))
    b = plate_geometry.boundaries
    problem = Problem(name="Pulley plate", geometry=plate_geometry, physics=(solid,), boundary_conditions=(
        FixedDisplacement(physics=solid, boundary=b.bore, value=(0.0, 0.0, 0.0)),
        Traction(physics=solid, boundary=b.belt, value=traction),
        TractionFree(physics=solid, boundary=b.rim_free | b.cutouts | b.faces)))
    fv = Simulation(problem=problem, mesh=plate_mesh, numerics=Numerics(
        spatial=FiniteVolume(gradient=BoundaryLeastSquaresGradient(), diffusion=CentralElasticStress()),
        linear_solver=SparseDirect())).solve_stationary()
    operator = fv.operators[solid.model_id]
    u_fv = fv.field(solid.displacement)
    face_force = operator.conservative_face_tractions(u_fv.values.ravel())
    bore_faces = [i for i, face in enumerate(operator.geometry.faces) if face.boundary in b.bore.members]
    fv_runs.append(dict(u=u_fv, vm=fv.field(solid.von_mises_stress), centres=operator.space.dof_coordinates,
                        reaction=face_force[bore_faces].sum(axis=0)))
    print(f"finite volumes, {rpm:.0f} rpm: {plate_mesh.cell_count} tetrahedra, "
          f"{3 * plate_mesh.cell_count} unknowns ({time.time() - start:.0f} s)")


# ## Comparison with the reference
#
# **Global equilibrium.** The strand forces act along the belt tangents at the two
# contact ends, $\mathbf t_{1,2}=(-\sin\varphi,\cos\varphi)$ pointing away from the
# pulley, so the shaft must supply $-(F_1\mathbf t_1+F_2\mathbf t_2)$ and the
# torque $-M$; the centrifugal load is self-balanced.

def strand_resultant(rpm):
    _, F_1, F_2 = strand_forces(rpm)
    t_1 = np.array([-np.sin(phi_tight), np.cos(phi_tight)])
    t_2 = np.array([np.sin(phi_slack), -np.cos(phi_slack)])
    return F_1 * t_1 + F_2 * t_2


print("Global equilibrium, finite elements on the finest mesh (shaft reaction + belt load = 0)")
for i, rpm in enumerate(speeds_shown):
    belt_force = strand_resultant(rpm)
    R = fine["reactions"][i]
    print(f"  {rpm:5.0f} rpm: reaction ({R[0]:9.3f}, {R[1]:8.3f}) N vs -(F1 t1 + F2 t2) = "
          f"({-belt_force[0]:9.3f}, {-belt_force[1]:8.3f}) N, deviation {np.linalg.norm(R + belt_force) / np.linalg.norm(belt_force):.1e}; "
          f"torque {fine['torque'][i]:8.3f} vs -M = {-power / omega(rpm):8.3f} N m "
          f"({abs(fine['torque'][i] / (power / omega(rpm)) + 1):.1e})")
print("Global equilibrium, finite volumes (conservative face forces on the bore)")
for i, rpm in enumerate(speeds_shown):
    belt_force = strand_resultant(rpm)
    R = fv_runs[i]["reaction"]
    print(f"  {rpm:5.0f} rpm: reaction ({R[0]:9.3f}, {R[1]:8.3f}, {R[2]:.1e}) N, "
          f"deviation from -(F1 t1 + F2 t2) {np.linalg.norm(R[:2] + belt_force) / np.linalg.norm(belt_force):.1e}")

# **Rotating disc (closed form).** For the blank, $u(r)=-\frac{(1-\nu^2)\rho\omega^2}{8E}r^3
# +C_1 r+C_2/r$ with $u(a)=0$ and $\sigma_r(b)=0$; stresses are compared along the
# radius at $\varphi=20^\circ$, slightly inside the faceted boundary circles.


def rotating_disc(r, w, a=r_bore, b=R_out):
    k = (1 - nu**2) * rho * w**2 / (8 * E)
    c = E / (1 - nu**2)
    C1, C2 = np.linalg.solve(np.array([[a, 1 / a], [c * (1 + nu), -c * (1 - nu) / b**2]]),
                             np.array([k * a**3, c * k * (3 + nu) * b**2]))
    sigma_r = c * (-k * (3 + nu) * r**2 + C1 * (1 + nu) - C2 * (1 - nu) / r**2)
    sigma_t = c * (-k * (1 + 3 * nu) * r**2 + C1 * (1 + nu) + C2 * (1 - nu) / r**2)
    return sigma_r, sigma_t


ray = np.radians(20.0)
r_line = np.linspace(r_bore + 3e-4, R_out - 3e-4, 80)
line_stress = blank_result.space.evaluate(blank_stress, np.column_stack((r_line * np.cos(ray), r_line * np.sin(ray))))
c, s = np.cos(ray), np.sin(ray)
fe_sr = line_stress[:, 0] * c * c + line_stress[:, 1] * s * s + 2 * line_stress[:, 2] * c * s
fe_st = line_stress[:, 0] * s * s + line_stress[:, 1] * c * c - 2 * line_stress[:, 2] * c * s
ex_sr, ex_st = rotating_disc(r_line, w_hi)
scale = np.abs(ex_st).max()
print(f"Rotating annular disc at 9000 rpm ({len(blank['triangles'])} triangles)")
print(f"  hoop stress near the bore: computed {fe_st[0] / 1e6:.4f} MPa, exact {ex_st[0] / 1e6:.4f} MPa "
      f"({100 * (fe_st[0] / ex_st[0] - 1):+.3f} %)")
print(f"  hoop stress near the rim:  computed {fe_st[-1] / 1e6:.4f} MPa, exact {ex_st[-1] / 1e6:.4f} MPa "
      f"({100 * (fe_st[-1] / ex_st[-1] - 1):+.3f} %)")
print(f"  largest deviation along the radius: radial {np.abs(fe_sr - ex_sr).max() / scale:.2e}, "
      f"hoop {np.abs(fe_st - ex_st).max() / scale:.2e} (relative to the peak hoop stress)")

# **Mesh study.** Uniform refinement halves the element size, so the observed order is
# $p=\log_2[(f_1-f_2)/(f_2-f_3)]$ and the Richardson estimate $f_3+(f_3-f_2)/(2^p-1)$.
# Point stresses on the fillets are evaluated on straight-sided triangles whose
# boundary vertices move onto the exact curves on refinement, so their sequence need
# not be monotone; the change between the last two meshes is then the error measure.


def richardson(values, h):
    f1, f2, f3 = values
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf
    try:
        p = brentq(lambda p: (h[0]**p - h[1]**p) / (h[1]**p - h[2]**p) - ratio, 0.2, 10.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h[1] / h[2]) ** p - 1.0)


print("Mesh study (P2 triangles)")
study = {}
for label, key, idx, unit_scale, unit in (
        ("peak von Mises, 1000 rpm", "peak", 0, 1e6, "MPa"), ("peak von Mises, 9000 rpm", "peak", 1, 1e6, "MPa"),
        ("von Mises at the probe, 1000 rpm", "probe", 0, 1e6, "MPa"),
        ("von Mises at the probe, 9000 rpm", "probe", 1, 1e6, "MPa"),
        ("radial rim displacement, 9000 rpm", "rim_u", 1, 1e-6, "um")):
    vals = [float(r[key][idx][1] if key == "rim_u" else r[key][idx]) for r in runs]
    p, ext = richardson(vals, [2.0**-l for l in levels])
    study[(key, idx)] = (vals, p, ext)
    trend = (f"order {p:4.2f}, extrapolated {ext / unit_scale:9.4f}" if np.isfinite(p)
             else "non-monotone sequence, no extrapolation")
    print(f"  {label:36s} " + "  ".join(f"{v / unit_scale:9.4f}" for v in vals)
          + f" {unit}; {trend}, change between the last two meshes {100 * abs(vals[2] / vals[1] - 1):.3f} %")

# **Finite elements versus finite volumes.** The FV solution is sampled at 1500
# randomly chosen cell centres; the FE solution (middle mesh) is evaluated at the same
# $(x, y)$.
print("Finite volumes (3D plate, cell-centred) vs finite elements (plane stress, P2, middle mesh)")
fe_mid = runs[1]
for i, rpm in enumerate(speeds_shown):
    sample = np.random.default_rng(1).choice(len(fv_runs[i]["centres"]), 1500, replace=False)
    centres = fv_runs[i]["centres"][sample]
    u_fv = fv_runs[i]["u"].values[sample, :2]
    u_fe = fe_mid["result"].space.evaluate(fe_mid["result"].displacement[i], centres[:, :2])
    diff = np.linalg.norm(u_fv - u_fe) / np.linalg.norm(u_fe)
    rim_fv = fv_runs[i]["u"].evaluate(np.array([[0.0, R_out - 1e-3, 0.0]]))[0, :2]
    rim_fe = fe_mid["result"].space.evaluate(fe_mid["result"].displacement[i], np.array([[0.0, R_out - 1e-3]]))[0]
    vm_fv = float(fv_runs[i]["vm"].evaluate(np.array([[*probe, 0.0]]))[0])
    print(f"  {rpm:5.0f} rpm: relative L2 difference of the in-plane displacement {diff:.2e}; "
          f"rim top (u_x, u_y) FV ({rim_fv[0] * 1e6:7.3f}, {rim_fv[1] * 1e6:7.3f}) um vs "
          f"FE ({rim_fe[0] * 1e6:7.3f}, {rim_fe[1] * 1e6:7.3f}) um; "
          f"von Mises in the probe cell FV {vm_fv / 1e6:6.2f} MPa vs FE {fe_mid['probe'][i] / 1e6:6.2f} MPa")


# ## Figure
#
# Top: von Mises stress over the pulley at 1000 and 9000 rpm (finest mesh, recovered
# P2 field). Bottom left: von Mises stress at the probe point on the inner fillet of
# the top cut-out and the peak over the pulley, over the speed sweep, with the
# finite-volume values at the two solved speeds. Bottom middle: rotating-disc check on
# the blank. Bottom right: mesh study of the peak stresses.

space = fine["result"].space
cells = space.cell_dofs
sub = np.concatenate([cells[:, [0, 3, 5]], cells[:, [3, 1, 4]], cells[:, [5, 4, 2]], cells[:, [3, 4, 5]]])
xy_nodes = space.dof_coordinates
tri = Triangulation(xy_nodes[:, 0] / mm, xy_nodes[:, 1] / mm, sub)
fig = plt.figure(figsize=(16, 11), constrained_layout=True)
grid = fig.add_gridspec(2, 6, height_ratios=(1.25, 1.0))
for i, rpm in enumerate(speeds_shown):
    ax = fig.add_subplot(grid[0, 3 * i:3 * i + 3])
    values = fine["vm"][i] / 1e6
    levels_vm = np.linspace(0.0, values.max(), 41)
    filled = ax.tricontourf(tri, values, levels=levels_vm, cmap=FIELD_CMAP)
    ax.tricontour(tri, values, levels=levels_vm[::4], colors=INK, linewidths=0.25, alpha=0.4)
    fig.colorbar(filled, ax=ax, shrink=0.85, label="von Mises stress / MPa", ticks=np.linspace(0, values.max(), 6).round(0))
    arc = np.linspace(phi_slack, phi_tight, 100)
    ax.plot(1.04 * R_out / mm * np.cos(arc), 1.04 * R_out / mm * np.sin(arc), color=INK, lw=3, alpha=0.6)
    ax.plot(*(probe / mm), "w*", ms=10, mec=INK)
    ax.set(aspect="equal", xlabel="x / mm", ylabel="y / mm",
           title=f"{rpm:.0f} rpm: torque {power / omega(rpm):.0f} N m, peak {values.max():.1f} MPa")

ax = fig.add_subplot(grid[1, 0:2])
ax.plot(sweep_rpm, fine["sweep_probe"] / 1e6, "-", color=COPPER, lw=2, label="probe point, finite elements")
ax.plot(sweep_rpm, fine["sweep_peak"] / 1e6, "--", color=GREEN, lw=1.6, label="peak over the pulley, finite elements")
for i, rpm in enumerate(speeds_shown):
    vm_fv = float(fv_runs[i]["vm"].evaluate(np.array([[*probe, 0.0]]))[0])
    ax.plot(rpm, vm_fv / 1e6, "s", color="#5F9175", mfc="none", ms=8, label="probe cell, finite volumes" if i == 0 else None)
ax.set(xlabel="rotational speed / rpm", ylabel="von Mises stress / MPa", title="Constant power: belt vs inertia")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 2:4])
ax.plot(r_line / mm, ex_st / 1e6, color=COPPER, lw=2, label="hoop, exact")
ax.plot(r_line / mm, ex_sr / 1e6, color=GREEN, lw=2, label="radial, exact")
ax.plot(r_line[::4] / mm, fe_st[::4] / 1e6, "o", color=COPPER, mfc="none", ms=5, label="hoop, finite elements")
ax.plot(r_line[::4] / mm, fe_sr[::4] / 1e6, "s", color=GREEN, mfc="none", ms=5, label="radial, finite elements")
ax.set(xlabel="radius / mm", ylabel="stress / MPa", title="Blank without cut-outs at 9000 rpm")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 4:6])
dofs = [r["dofs"] for r in runs]
for (key, idx), color, label in ((("peak", 0), COPPER, "peak von Mises, 1000 rpm"),
                                 (("peak", 1), GREEN, "peak von Mises, 9000 rpm"),
                                 (("probe", 0), "#CD9561", "probe von Mises, 1000 rpm"),
                                 (("probe", 1), "#5F9175", "probe von Mises, 9000 rpm"),
                                 (("rim_u", 1), INK, "radial rim displacement, 9000 rpm")):
    vals = np.array(study[(key, idx)][0])
    ax.plot(dofs, 100 * (vals / vals[-1] - 1), "o-", color=color, label=label)
ax.set_xscale("log")
ax.minorticks_off()
ax.set_xticks(dofs, [f"{n:,}" for n in dofs])
ax.set(xlabel="number of displacement unknowns", ylabel="deviation from the finest mesh / %",
       title="Mesh study (P2, uniform refinement)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Halve the power at fixed geometry to move the stress minimum to a lower speed, or
# widen the spokes and compare how the peak at the inner fillets responds at low and at
# high speed. Because the problem is linear, a single pair of solves gives the whole
# speed and power map; a natural-frequency analysis of the same mesh would show where
# the neglected dynamic effects start to matter.
