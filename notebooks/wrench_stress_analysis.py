"""OneSimulation example: stresses and strains in a combination wrench turning a bolt.
Requires the installed, activated OneSimulation runtime.

A 13 mm combination wrench grips the hexagon head of an M8 bolt with its open end; a hand force
of 150 N acts on the box end. The 3D linear-elastic field of wrench and bolt is computed with
curved quadratic tetrahedra from the CAD kernel (and linear tetrahedra for comparison) and checked
against global equilibrium, Euler-Bernoulli bending of the handle, Saint-Venant torsion of the bolt
shank and a mesh study of the peak stress in the jaw fillets.
"""


# # Stresses and strains in a wrench
#
# A combination wrench has an open end and a box end. Here the open end grips the hexagon head
# of a bolt whose shank is held fixed at its far end (as if screwed into a rigid part), and the
# hand pushes on the box end with $F=150$ N perpendicular to the handle, in the plane of the
# wrench. The torque on the bolt is carried through the jaws, the throat of the open end and the
# neck between head and handle. We want the stress field, the deformed shape and the peak stress
# in the rounded throat of the jaws.
#
# **Geometry** (a representative 13 mm wrench built from boxes and cylinders in the CAD kernel;
# all dimensions in mm, wrench plane $x$-$y$, thickness along $z$):
#
# | part | geometry |
# |---|---|
# | open-end head | disc of radius 16, thickness 5, centred on the bolt axis ($x=y=0$) |
# | jaw opening | 13 mm wide (= bolt across-flats), turned 15° against the handle; flat throat 8.5 from the bolt axis with two fillets of radius $r_f=2$ |
# | handle | $11\times5$ rectangular section from the head to the box end, neck fillets of radius 8 at both ends |
# | box end | ring of outer radius 10.5 and bore radius 7, centre at $x=L=140$ |
# | bolt | M8 hexagon head, 13 across flats (ISO 4017), height equal to the jaw thickness, shank $\varnothing8$ and 15 long below the wrench, end face fixed |
#
# Material: structural steel, $E=200$ GPa, $\nu=0.30$. As in the original problem statement the
# contact between jaw and bolt is taken as perfect (the hexagon flats are bonded to the jaw
# faces), so wrench and bolt form one solid.
#
# **Boundary conditions** on every face:
#
# | face | condition |
# |---|---|
# | end face of the bolt shank ($z=-17.5$) | fixed, $\mathbf u=0$ |
# | upper half ($y>0$) of the outer cylinder of the box end | uniform traction $-p\,\mathbf e_y$ with total force $F=150$ N |
# | every other face | traction free |
#
# The field is genuinely three-dimensional: in-plane bending of the handle, bending and
# twisting of the head, torsion and bending of the bolt shank.
#
# **Numerical method.** Continuous Galerkin finite elements for small-strain linear elasticity.
# The OpenCASCADE kernel builds the solid by Boolean operations; the conforming tetrahedral
# mesher refines it towards the throat fillets and returns 10-node tetrahedra, whose mid-edge
# nodes on the cylindrical faces (fillets, bolt shank, discs) are placed on the exact CAD
# cylinders. The main discretisation is quadratic (P2) Lagrange on these curved cells; for
# comparison the same vertices are used with straight-sided linear (P1) tetrahedra. Each solve is linear and
# stationary: sparse direct LU, no time integration.
#
# **Reference.** There is no closed-form solution for the whole wrench. The notebook uses
# (i) global equilibrium (reaction force and moment at the fixed face against the applied load),
# (ii) Euler-Bernoulli beam theory in the prismatic handle: the bending stress
# $\sigma_{xx}=M\,y/I$ with $M=F\,(x_F-x)$ ($x_F$ the line of action of the hand force) and the
# curvature $w''=-M/(EI)$ obtained from second differences of the 3D centre-line deflection
# (exact for a linear moment, and unaffected by shear deformation), (iii) Saint-Venant torsion plus bending of the round bolt shank, and (iv) a
# three-level mesh study of the peak von Mises stress in the throat fillets (observed order and
# Richardson extrapolation where the sequence is monotone). Stresses at the re-entrant edges
# where the bonded hexagon meets the jaw are singular in linear elasticity and are excluded from
# the convergence study.

from dataclasses import dataclass, replace
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
import numpy as np
from matplotlib import cm, colors
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.optimize import brentq

from onesim.cad.geometry.backends.occt.provider import OcctCadProvider
from onesim.cad.geometry.ports.kernel_port import HealingRequest
from onesim.cad.geometry.primitives import Geometry
from onesim.cad.operations.execution.model_geometry import GeometryService
from onesim.materials import IsotropicSolid
from onesim.meshing import UnstructuredSimplexMesh
from onesim.meshing.generation.backends.cad.in_memory_cad import (
    NativeConstructionRegistry, build_file_free_mesher, default_native_constructions)
from onesim.meshing.generation.contracts.polynomial.tetrahedron import QuadraticTetrahedralGeometry
from onesim.meshing.generation.contracts.requests import (
    HighOrderMeshControl, MeshGenerationRequest, MeshRefinement, MeshRefinementSupport, SpatialMeshControl)
from onesim.meshing.generation.operations.geometry.model_mesh import CadMeshingService
from onesim.meshing.meshes.mesh_declarations import TetrahedralMesh
from onesim.meshing.meshes.models.geometry.discretization_data import CoordinateSet
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation
from onesim.simulation.problem_definition.values.declarations.immutable_values import FrozenMap

print("OneSimulation:", version("onesimulation"))


# ## Parameters

mm = 1e-3
t = 5.0 * mm                        # plate thickness, z in [-t/2, t/2]
R_head, s_af = 16.0 * mm, 13.0 * mm  # open-end head radius, jaw opening = bolt across flats
throat, r_f = 8.5 * mm, 2.0 * mm      # throat distance from the bolt axis, throat fillet radius
jaw_angle = np.radians(15.0)
L, h_handle = 140.0 * mm, 11.0 * mm   # bolt axis to box-end centre, handle height (in-plane)
R_ring, r_bore = 10.5 * mm, 7.0 * mm
rho = 8.0 * mm                       # neck fillet radius
r_shank, l_shank = 4.0 * mm, 15.0 * mm
E, nu = 200e9, 0.30
F = 150.0                            # N, hand force on the box end

a_h = h_handle / 2
d_head = np.sqrt((R_head + rho) ** 2 - (a_h + rho) ** 2)     # neck fillet centres (x offsets)
d_ring = np.sqrt((R_ring + rho) ** 2 - (a_h + rho) ** 2)
theta_load = np.pi - np.arctan2(a_h + rho, d_ring)           # loaded arc of the box end: 0 <= theta <= theta_load
A_load = R_ring * t * theta_load
p_load = F / A_load
x_load = L + R_ring * np.sin(theta_load) / theta_load        # line of action of the resultant
I_handle = t * h_handle**3 / 12
print(f"loaded area {A_load / mm**2:.2f} mm^2, traction {p_load / 1e6:.4f} MPa, "
      f"resultant at x = {x_load / mm:.3f} mm, torque on the bolt F x = {F * x_load:.4f} N m")


# ## Geometry and mesh
#
# **Solid model.** The OpenCASCADE kernel builds the open-end head as a disc minus a
# rounded-end slot (a box for the parallel jaw faces, a shorter box and two cylinders for the
# flat throat with its two fillets), adds the bolt as the intersection of three rotated slabs
# (the hexagon) plus a cylinder (the shank), turns head and bolt by 15° about the bolt axis,
# and unites them with the handle box, the ring of the box end and the four neck fillets
# (each a box minus a tangent cylinder). A final healing step merges coplanar faces. Every
# curved surface is an exact cylinder.
#
# **Mesh.** The conforming tetrahedral mesher works on the exact CAD faces: global size
# 3 mm (two quadratic elements through the 5 mm thickness), 1.2 mm on every other cylinder
# (shank, discs, neck fillets, box end), a size $h_f$ on the two throat fillets and $3h_f$ in a
# ball of radius 5 mm around the throat; $h_f$ is refined in the mesh study. It returns 10-node tetrahedra; the mid-edge nodes of every edge that lies on
# one of the cylindrical faces are then placed on the exact cylinder, so the quadratic cells
# follow the fillets, the shank and the discs. The linear comparison uses the same vertices
# with straight-sided 4-node cells.

GROUPS = ("fixed", "load", "free")


@dataclass(frozen=True, eq=False, kw_only=True)
class WrenchAndBolt(Geometry):
    """Wrench and bolt as one solid; fixed shank end, loaded box-end rim, free faces."""

    dimension = 3

    def boundary_groups(self):
        return {**{name: (name,) for name in GROUPS}, "all": GROUPS}


def rotate(point):
    c, s = np.cos(jaw_angle), np.sin(jaw_angle)
    return (c * point[0] - s * point[1], s * point[0] + c * point[1])


w_jaw = s_af / 2
CYLINDERS = {   # (centre x, centre y), radius; all axes along z
    "shank": ((0.0, 0.0), r_shank), "head rim": ((0.0, 0.0), R_head),
    "box-end rim": ((L, 0.0), R_ring), "box-end bore": ((L, 0.0), r_bore),
    "throat fillet +": (rotate((throat - r_f, w_jaw - r_f)), r_f),
    "throat fillet -": (rotate((throat - r_f, -(w_jaw - r_f))), r_f),
    **{f"neck fillet {k}": (c, rho) for k, c in enumerate(
        ((d_head, a_h + rho), (d_head, -(a_h + rho)), (L - d_ring, a_h + rho), (L - d_ring, -(a_h + rho))))},
}


def build_solid():
    kernel = OcctCadProvider()
    cad = GeometryService("wrench", kernel)
    count = [0]

    def name(base):
        count[0] += 1
        return f"{base}{count[0]}"

    def box(x0, x1, y0, y1, z0, z1):
        return cad.place(cad.box(width=x1 - x0, height=y1 - y0, depth=z1 - z0, name=name("box")),
                         translation=(x0, y0, z0), name=name("placed"))

    def cylinder(xc, yc, r, z0, z1):
        return cad.place(cad.cylinder(radius=r, height=z1 - z0, name=name("cylinder")),
                         translation=(xc, yc, z0), name=name("placed"))

    def boolean(kind, a, b):
        return cad.boolean(kind, a, b, name=name(kind))

    lo, hi = -t / 2, t / 2
    head = cylinder(0.0, 0.0, R_head, lo, hi)
    slot = box(-3 * R_head, throat - r_f, -w_jaw, w_jaw, 2 * lo, 2 * hi)
    slot = boolean("union", slot, box(throat - r_f - 0.5 * mm, throat, -(w_jaw - r_f), w_jaw - r_f, 2 * lo, 2 * hi))
    for sign in (1, -1):
        slot = boolean("union", slot, cylinder(throat - r_f, sign * (w_jaw - r_f), r_f, 2 * lo, 2 * hi))
    head = boolean("difference", head, slot)
    hexagon = None
    for k in range(3):
        slab = box(-s_af, s_af, -w_jaw, w_jaw, lo, hi)
        if k:
            slab = cad.place(slab, rotation_radians=(0.0, 0.0, k * np.pi / 3), name=name("turned"))
        hexagon = slab if hexagon is None else boolean("intersection", hexagon, slab)
    bolt = boolean("union", hexagon, cylinder(0.0, 0.0, r_shank, lo - l_shank, lo))
    head = cad.place(boolean("union", head, bolt), rotation_radians=(0.0, 0.0, jaw_angle), name=name("turned"))
    body = boolean("union", head, box(R_head - 3 * mm, L - 8 * mm, -a_h, a_h, lo, hi))
    body = boolean("union", body, boolean("difference", cylinder(L, 0.0, R_ring, lo, hi),
                                          cylinder(L, 0.0, r_bore, 2 * lo, 2 * hi)))
    for x_centre, R, direction in ((0.0, R_head, 1), (L, R_ring, -1)):
        d = np.sqrt((R + rho) ** 2 - (a_h + rho) ** 2)
        xc = x_centre + direction * d
        x_tangent, y_tangent = x_centre + direction * d * R / (R + rho), (a_h + rho) * R / (R + rho)
        for sign in (1, -1):
            x0, x1 = sorted((x_tangent, xc))
            y0, y1 = sorted((sign * a_h, sign * y_tangent))
            fill = boolean("difference", box(x0, x1, y0, y1, lo, hi), cylinder(xc, sign * (a_h + rho), rho, 2 * lo, 2 * hi))
            body = boolean("union", body, fill)
    body = cad.construct(kernel.heal, HealingRequest(cad.reference("healed"), body.reference))
    return kernel, body


# The mesher replays the construction in its own OpenCASCADE session; the healing step is
# replayed there as the equivalent merge of coplanar faces.
NATIVE = NativeConstructionRegistry({**default_native_constructions().constructors,
                                     HealingRequest: lambda request, shapes, occ: shapes[request.source_reference].UnifySameDomain()})
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def exterior_faces(cells, points):
    """Outward-oriented vertex triples of the tetrahedron faces that belong to exactly one cell."""
    owners = {}
    for cell in cells:
        for face in TET_FACES:
            raw = tuple(int(v) for v in cell[list(face)])
            opposite = int(cell[[k for k in range(4) if k not in face][0]])
            owners.setdefault(tuple(sorted(raw)), []).append((raw, opposite))
    result = []
    for matches in owners.values():
        if len(matches) == 1:
            (a, b, c), opposite = matches[0]
            normal = np.cross(points[b] - points[a], points[c] - points[a])
            result.append((a, c, b) if normal @ (points[opposite] - points[a]) > 0 else (a, b, c))
    return result


def mesh_wrench(kernel, body, h_global, h_fillet):
    start = time.perf_counter()
    topology = kernel.structured_meshing_topology(body.reference)
    body_id = next(e.entity_id for e in body.data.entities if e.kind == "body")
    fillet_area = np.pi / 2 * r_f * t
    controls = tuple(MeshRefinement(f.entity_id, h_fillet if abs(f.area - fillet_area) < 1e-9 else 1.2 * mm, "surface")
                     for f in topology.faces if f.surface_type == "cylinder")
    throat_centre = (*rotate((throat - r_f, 0.0)), 0.0)
    controls += (MeshRefinement(body_id, 3 * h_fillet, "volume", MeshRefinementSupport("ball", throat_centre, 5 * mm)),)
    request = MeshGenerationRequest(
        body.reference.artifact_id, body.reference.version, global_size=h_global, refinements=controls,
        max_candidate_cells=2_000_000, high_order=HighOrderMeshControl(polynomial_order=2, project_to_geometry=False),
        spatial=SpatialMeshControl("free_tetrahedral", body_id, engine="cad_native"))
    native = CadMeshingService(provider_factory=lambda g, r: build_file_free_mesher(g, r, constructions=NATIVE)
                               ).generate(body, request)
    carrier = native.carrier
    X10, C10 = np.array(carrier.coordinates.values), np.asarray(carrier.connectivity.entities)
    C4 = C10[:, :4]
    # Place the mid-edge nodes of edges on cylindrical faces onto the exact cylinders.
    midpoint = {tuple(sorted((c[i], c[j]))): c[4 + m] for c in C10 for m, (i, j) in enumerate(TET_EDGES)}
    facets = {name: [] for name in GROUPS}
    curved = set()
    for face in exterior_faces(C4, X10):
        v = np.asarray(face)
        P = X10[v]
        normal = np.cross(P[1] - P[0], P[2] - P[0])
        on = None
        if abs(normal[2]) < 0.5 * np.linalg.norm(normal):
            for key, ((cx, cy), r) in CYLINDERS.items():
                if np.all(np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) < 1e-6):
                    on = key
                    for i, j in ((0, 1), (0, 2), (1, 2)):
                        node = midpoint[tuple(sorted((v[i], v[j])))]
                        q = X10[node]
                        scale = r / np.hypot(q[0] - cx, q[1] - cy)
                        X10[node, :2] = (cx + (q[0] - cx) * scale, cy + (q[1] - cy) * scale)
                        curved.add(node)
                    break
        if np.allclose(P[:, 2], -t / 2 - l_shank, atol=1e-9):
            facets["fixed"].append(tuple(v))
        elif on == "box-end rim":
            facets["load"].append(tuple(v))
        else:
            facets["free"].append(tuple(v))
    volumes = QuadraticTetrahedralGeometry(X10, C10).volumes
    carrier = replace(carrier, coordinates=CoordinateSet(tuple(map(tuple, X10)), unit="m"),
                      measures=tuple(map(float, volumes)))
    # CAD face groups for the curved mesh, position groups for the straight one.
    cad_groups = {name: [] for name in GROUPS}
    for f in topology.faces:
        c = np.array(f.center)
        if f.surface_type == "plane" and abs(c[2] + t / 2 + l_shank) < 1e-9:
            cad_groups["fixed"].append(f.entity_id)
        elif f.surface_type == "cylinder" and c[0] > L + 1 * mm:
            cad_groups["load"].append(f.entity_id)
        else:
            cad_groups["free"].append(f.entity_id)
    domain = WrenchAndBolt()
    mesh10 = TetrahedralMesh(domain, replace(native, carrier=carrier),
                             FrozenMap({k: tuple(v) for k, v in cad_groups.items()}), ())
    used = np.unique(C4)
    renumber = np.full(len(X10), -1)
    renumber[used] = np.arange(len(used))
    mesh4 = UnstructuredSimplexMesh(geometry=domain, coordinates=X10[used], cells=renumber[C4],
                                    boundary_facets={k: [tuple(renumber[list(f)]) for f in v] for k, v in facets.items()})
    return dict(domain=domain, mesh10=mesh10, mesh4=mesh4, X10=X10, C10=C10, h_fillet=h_fillet,
                cells=len(C10), curved=len(curved), volume=float(volumes.sum()), seconds=time.perf_counter() - start)


# ## Model
#
# `LinearElasticity` with structural steel; the box-end traction is a position-dependent value
# that is $-p\,\mathbf e_y$ on the upper half of the rim and zero on the lower half.

steel = IsotropicSolid(density=7850.0, specific_heat_capacity=475.0, thermal_conductivity=44.5,
                       youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12.3e-6)


def hand_traction(x, y, z):
    y = np.asarray(y)
    value = np.zeros(y.shape + (3,))
    value[..., 1] = np.where(y > 0.0, -p_load, 0.0)
    return value


def solve(data, order):
    start = time.perf_counter()
    domain = data["domain"]
    solid = LinearElasticity(domain=domain.domain(), material=steel)
    b = domain.boundaries
    problem = Problem(name="combination wrench on an M8 bolt", geometry=domain, physics=(solid,), boundary_conditions=(
        FixedDisplacement(boundary=b.fixed, physics=solid, value=(0.0, 0.0, 0.0)),
        Traction(boundary=b.load, physics=solid, value=hand_traction),
        TractionFree(boundary=b.free, physics=solid)))
    mesh = data["mesh10"] if order == 2 else data["mesh4"]
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=order)), linear_solver=SparseDirect())).solve_stationary()
    operator = result.operators[solid.model_id]
    index = np.asarray(operator.field_binding.indices)
    u = np.zeros(operator.problem_size)
    u[index] = np.asarray(result.field(solid.displacement).values)
    # Reactions: residual of the unconstrained system at the fixed degrees of freedom.
    base, system, rhs, constraints = operator.weak_assembly(0.0)
    residual = system.matrix.tocsr() @ u - np.asarray(rhs)
    fixed = np.zeros(operator.problem_size, dtype=bool)
    fixed[[c.dof_index for c in constraints]] = True
    reaction = np.where(fixed[index], residual[index], 0.0)
    X = operator.space.dof_coordinates
    run = dict(order=order, h_fillet=data["h_fillet"], operator=operator, u=u, index=index, dofs=operator.problem_size,
               reaction=reaction.sum(axis=0), moment=np.cross(X, reaction).sum(axis=0),
               u_max=float(np.linalg.norm(u[index], axis=1).max()))
    run["seconds"] = time.perf_counter() - start
    return run


def stress_at(run, points):
    """Cauchy stress (xx, yy, zz, xy, yz, xz) of a solved run at physical points."""
    return np.asarray(run["operator"].stress(run["u"], np.asarray(points)))


def von_mises(s):
    xx, yy, zz, xy, yz, xz = np.moveaxis(s, -1, 0)
    return np.sqrt(((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) / 2 + 3 * (xy**2 + yz**2 + xz**2))


def displacement_at(run, points):
    return np.asarray(run["operator"].evaluate_field(run["operator"].field, run["u"], np.asarray(points)))


# Sample points: both throat fillets (5 um inside the material), the handle centre line and a
# vertical line across the handle, and a circle of radius 3.9 mm in the bolt shank half-way
# between head and support (0.1 mm below the surface, inside the straight-sided cells too).
phi = np.linspace(0.0, 0.5 * np.pi, 46)
zs = np.linspace(-t / 2, t / 2, 11)
fillet_points = []
for sign, key in ((1, "throat fillet +"), (-1, "throat fillet -")):
    (cx, cy), r = CYLINDERS[key]
    # In the head frame the fillet arc runs from the jaw face (normal -y) to the throat (normal +x).
    angles = jaw_angle + sign * (0.5 * np.pi - phi)
    P, Z = np.meshgrid(angles, zs, indexing="ij")
    fillet_points.append(np.column_stack(((cx + (r + 5e-6) * np.cos(P)).ravel(),
                                          (cy + (r + 5e-6) * np.sin(P)).ravel(), Z.ravel())))
x_stations = np.array([53.6, 73.6, 93.6]) * mm
centre_line = np.column_stack((x_stations, np.zeros(3), np.zeros(3)))
y_line = np.linspace(-a_h + 1e-6, a_h - 1e-6, 23)
handle_line = np.column_stack((np.full_like(y_line, x_stations[1]), y_line, np.zeros_like(y_line)))
theta = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
z_shank = -t / 2 - l_shank / 2
r_probe = r_shank - 0.1 * mm        # inside the faceted surface of the linear mesh as well
shank_circle = np.column_stack((r_probe * np.cos(theta), r_probe * np.sin(theta), np.full_like(theta, z_shank)))


def evaluate(run):
    vm = [von_mises(stress_at(run, P)) for P in fillet_points]
    run["fillet_vm"] = vm
    run["fillet_peak"] = float(max(v.max() for v in vm))
    run["w_centre"] = displacement_at(run, centre_line)[:, 1]
    run["sxx_handle"] = stress_at(run, handle_line)[:, 0]
    run["vm_shank"] = von_mises(stress_at(run, shank_circle))
    return run


# ## Solve
#
# Three meshes with throat-fillet sizes $h_f=0.8$, $0.53$ and $0.36$ mm (refinement ratio 1.5;
# global size 3 mm, 1.2 mm on the other cylinders), each solved with curved quadratic and with
# linear tetrahedra.

FILLET_SIZES = (0.8 * mm, 0.8 / 1.5 * mm, 0.8 / 1.5**2 * mm)
kernel, body = build_solid()
print(f"CAD solid: volume {body.volume / mm**3:.1f} mm^3, "
      f"{len(kernel.structured_meshing_topology(body.reference).faces)} faces")
runs = {2: [], 1: []}
meshes = []
for h_f in FILLET_SIZES:
    data = mesh_wrench(kernel, body, 3.0 * mm, h_f)
    meshes.append(data)
    print(f"h_f = {h_f / mm:.2f} mm: {data['cells']} tetrahedra, {data['curved']} mid-edge nodes placed on "
          f"cylinders, volume {data['volume'] / mm**3:.2f} mm^3 ({data['seconds']:.0f} s)")
    for order in (2, 1):
        run = evaluate(solve(data, order))
        runs[order].append(run)
        print(f"   P{order}: {run['dofs']:6d} unknowns, peak fillet von Mises {run['fillet_peak'] / 1e6:7.2f} MPa, "
              f"max |u| {run['u_max'] / mm:.4f} mm ({run['seconds']:.0f} s)")


# ## Comparison with the reference
#
# Equilibrium, the handle and the bolt shank are regions where the 3D field must reduce to a
# closed form, so the quadratic elements are expected to match to a fraction of a percent; the
# linear elements are too stiff in bending and torsion at the same mesh. The throat fillets are
# a smooth stress concentration (no singularity), so their peak must converge; the bonded
# hexagon corners on the jaw faces are singular and are not part of this study.

def richardson(values, sizes):
    """Observed order and extrapolated value from three meshes with sizes h."""
    f1, f2, f3 = values
    h1, h2, h3 = sizes
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.2, 12.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


fine2, fine1 = runs[2][-1], runs[1][-1]
G = E / (2 * (1 + nu))
torque = F * x_load

print("Global equilibrium on the finest mesh (reactions at the fixed shank face)")
print(f"   expected: force (0, {F:.3f}, 0) N, moment about the bolt axis {torque:.4f} N m")
for run in (fine2, fine1):
    Rf, M = run["reaction"], run["moment"]
    print(f"   P{run['order']}: force ({Rf[0]:+.2e}, {Rf[1]:.4f}, {Rf[2]:+.2e}) N, moment ({M[0]:+.2e}, {M[1]:+.2e}, "
          f"{M[2]:.5f}) N m; deviation {Rf[1] / F - 1:+.1e} (force), {M[2] / torque - 1:+.1e} (torque)")

print("Handle: Euler-Bernoulli beam theory at x = 73.6 mm (bending moment M = F (x_load - x))")
M_station = F * (x_load - x_stations[1])
curvature = -F * (x_load - x_stations[1]) / (E * I_handle)
dx = x_stations[1] - x_stations[0]
print(f"   outer-fibre stress M c / I = {M_station * a_h / I_handle / 1e6:.3f} MPa, curvature M/(EI) = {curvature:.5f} 1/m")
for run in (fine2, fine1):
    slope = np.polyfit(y_line, run["sxx_handle"], 1)[0]
    w = run["w_centre"]
    kappa = (w[0] - 2 * w[1] + w[2]) / dx**2
    print(f"   P{run['order']}: sigma_xx slope {slope * a_h / 1e6:.3f} MPa at y = c "
          f"({100 * (slope * a_h / (M_station * a_h / I_handle) - 1):+.2f} %), curvature from the 3D deflection "
          f"{kappa:.5f} 1/m ({100 * (kappa / curvature - 1):+.2f} %)")

print("Bolt shank half-way between head and support: Saint-Venant torsion plus bending")
J_shank, I_shank = np.pi * r_shank**4 / 2, np.pi * r_shank**4 / 4
tau_t = torque * r_probe / J_shank                       # = 16 T/(pi d^3) at the surface
M_b = F * (0.0 - z_shank)                                # moment of the hand force about the shank section
sigma_b = M_b * r_probe * np.sin(theta) / I_shank
# Transverse shear of the round bar at its surface (flexure solution), largest on the neutral axis,
# where it adds to the torsional shear on one side (theta = 0) and subtracts on the other.
tau_v = (1 + 2 * nu) / (1 + nu) * F / (np.pi * r_shank**2) * np.cos(theta)   # value at the surface
vm_theory = np.sqrt(sigma_b**2 + 3 * (tau_t + tau_v) ** 2)
print(f"   at r = {r_probe / mm:.1f} mm: torsional shear T r / J = {tau_t / 1e6:.2f} MPa, "
      f"bending stress amplitude {M_b * r_probe / I_shank / 1e6:.2f} MPa")
for run in (fine2, fine1):
    deviation = run["vm_shank"] / vm_theory - 1
    print(f"   P{run['order']}: von Mises on the shank surface, mean {run['vm_shank'].mean() / 1e6:.2f} MPa "
          f"(theory {vm_theory.mean() / 1e6:.2f} MPa), largest deviation around the circumference {100 * np.abs(deviation).max():.2f} %")

print("Peak von Mises stress in the throat fillets (mesh study, h_f = "
      + ", ".join(f"{h / mm:.2f}" for h in [r["h_fillet"] for r in runs[2]]) + " mm)")
sizes = [r["h_fillet"] for r in runs[2]]
extrapolated_peak = {}
for order in (2, 1):
    vals = [r["fillet_peak"] for r in runs[order]]
    p, extrapolated_peak[order] = richardson(vals, sizes)
    steps = np.diff(vals)
    if np.isfinite(p):
        trend = f"observed order {p:.2f}, Richardson {extrapolated_peak[order] / 1e6:.2f} MPa"
    elif steps[0] * steps[1] < 0:
        trend = "not monotone, so no observed order"
    else:
        trend = "monotone but not yet in the asymptotic range (no observed order between 0.2 and 12)"
    print(f"   P{order}: " + "  ".join(f"{v / 1e6:.2f}" for v in vals) + f" MPa; {trend}; change between the "
          f"last two meshes {100 * abs(vals[-1] / vals[-2] - 1):.2f} %")
    where = np.argmax([v.max() for v in runs[order][-1]["fillet_vm"]])
    print(f"      peak on the fillet on the {'+y' if where == 0 else '-y'} side of the jaw opening")

print("Load-point deflection budget (lower-bound estimate, rigid head and jaws)")
twist = torque * l_shank / (G * np.pi * (2 * r_shank) ** 4 / 32)
x0 = d_head                                             # handle clamped where the head neck fillets end
bending = F * ((x_load - x0) ** 3) / (3 * E * I_handle)
estimate = twist * x_load + bending
for run in (fine2, fine1):
    tip = displacement_at(run, np.array([[x_load, 8.0 * mm, 0.0]]))[0, 1]
    print(f"   P{run['order']}: deflection at the box end {abs(tip) / mm:.3f} mm; estimate {estimate / mm:.3f} mm = shank twist "
          f"{twist * x_load / mm:.3f} mm + handle bending {bending / mm:.3f} mm")


# ## Figure
#
# Top: the von Mises stress on the deformed wrench and bolt (displacements magnified 8 times,
# the undeformed feature edges in grey) and on the upper face of the open end. Bottom: the
# von Mises stress around the bolt shank against Saint-Venant torsion with bending, the bending
# stress across the handle against beam theory, and the mesh study of the fillet peak.

data, run = meshes[-1], fine2
X10, C10 = data["X10"], data["C10"]
faces = exterior_faces(C10[:, :4], X10)
midpoint = {tuple(sorted((c[i], c[j]))): c[4 + m] for c in C10 for m, (i, j) in enumerate(TET_EDGES)}
nodes = np.unique([n for f in faces for n in (*f, *(midpoint[tuple(sorted(e))] for e in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2]))))])
vm_nodes = np.zeros(len(X10))
vm_nodes[nodes] = von_mises(stress_at(run, X10[nodes]))
u_nodes = np.zeros_like(X10)
u_nodes[nodes] = displacement_at(run, X10[nodes])
sub = []
for a, b, c in faces:
    ab, bc, ca = (midpoint[tuple(sorted(e))] for e in ((a, b), (b, c), (c, a)))
    sub += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
sub = np.array(sub)
vmax = 1e-6 * np.percentile(vm_nodes[nodes], 99.5)
norm = colors.Normalize(vmin=0.0, vmax=vmax)
magnify = 8.0

fig = plt.figure(figsize=(19, 11), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.25, 1.0, 1.0))
ax = fig.add_subplot(grid[0, :2], projection="3d")
deformed = (X10 + magnify * u_nodes) / mm
shade = FIELD_CMAP(norm(1e-6 * vm_nodes[sub].mean(axis=1)))
ax.add_collection3d(Poly3DCollection(deformed[sub], facecolors=shade, edgecolors=shade, linewidths=0.1))
normals = {}
for a, b, c in faces:
    n = np.cross(X10[b] - X10[a], X10[c] - X10[a])
    for e in ((a, b), (b, c), (c, a)):
        normals.setdefault(tuple(sorted(e)), []).append(n / np.linalg.norm(n))
feature = [e for e, ns in normals.items() if len(ns) == 2 and ns[0] @ ns[1] < np.cos(np.radians(30))]
ax.add_collection3d(Line3DCollection([X10[list(e)] / mm for e in feature], colors="#9FA593", linewidths=0.4))
ax.set(xlim=(-20, 160), ylim=(-40, 25), zlim=(-20, 6), xlabel="x / mm", ylabel="y / mm", zlabel="z / mm",
       title=f"von Mises stress on the deformed shape (P2, {run['dofs']} unknowns, displacement x{magnify:.0f})")
ax.set_box_aspect((180, 65, 26 * 1.3), zoom=1.2)
ax.view_init(elev=28, azim=-62)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=FIELD_CMAP), ax=ax, shrink=0.6, pad=0.12, label="von Mises stress / MPa")

ax = fig.add_subplot(grid[0, 2])
top = [f for f in faces if np.allclose(X10[list(f), 2], t / 2, atol=1e-9)]
top_sub = []
for a, b, c in top:
    ab, bc, ca = (midpoint[tuple(sorted(e))] for e in ((a, b), (b, c), (c, a)))
    top_sub += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
top_sub = np.array(top_sub)
tri = Triangulation(X10[:, 0] / mm, X10[:, 1] / mm, top_sub)
vmax_top = 1e-6 * np.percentile(vm_nodes[np.unique(top_sub)], 99.8)
cs = ax.tricontourf(tri, 1e-6 * vm_nodes, levels=np.linspace(0, vmax_top, 41), cmap=FIELD_CMAP, extend="max")
ax.tricontour(tri, 1e-6 * vm_nodes, levels=np.linspace(0, vmax_top, 11)[1:-1], colors=CREAM, linewidths=0.4)
ax.set(xlim=(-18, 32), ylim=(-18, 18), aspect="equal", xlabel="x / mm", ylabel="y / mm",
       title="Upper face of the open end and the bolt head")
fig.colorbar(cs, ax=ax, shrink=0.8, label="von Mises stress / MPa")

ax = fig.add_subplot(grid[1, 0])
ax.plot(np.degrees(theta), vm_theory / 1e6, color=INK, lw=1.2, label="Saint-Venant torsion + bending")
for r, marker, color in ((fine2, "s", GREEN), (fine1, "o", COPPER)):
    ax.plot(np.degrees(theta), r["vm_shank"] / 1e6, marker, ms=3.5, color=color, label=f"P{r['order']} elements")
ax.set(xlabel="angle around the shank / deg", ylabel="von Mises stress / MPa",
       title=f"Bolt shank, r = {r_probe / mm:.1f} mm, {abs(z_shank) / mm:.1f} mm below the wrench mid-plane")
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 1])
ax.plot(F * (x_load - x_stations[1]) * y_line / I_handle / 1e6, y_line / mm, color=INK, lw=1.2,
        label="beam theory  sigma = M y / I")
for r, marker, color in ((fine2, "s", GREEN), (fine1, "o", COPPER)):
    ax.plot(r["sxx_handle"] / 1e6, y_line / mm, marker, ms=3.5, color=color, label=f"P{r['order']} elements")
ax.set(xlabel="bending stress sigma_xx / MPa", ylabel="y across the handle / mm",
       title=f"Handle section x = {x_stations[1] / mm:.1f} mm")
ax.legend(fontsize=8)

ax = fig.add_subplot(grid[1, 2])
for order, marker, color in ((2, "s", GREEN), (1, "o", COPPER)):
    ax.plot([s / mm for s in sizes], [r["fillet_peak"] / 1e6 for r in runs[order]], marker + "-", color=color,
            label=f"P{order} elements")
ax.set_xticks([s / mm for s in sizes])
if np.isfinite(richardson([r["fillet_peak"] for r in runs[1]], sizes)[0]):
    ax.axhline(extrapolated_peak[1] / 1e6, color=COPPER, lw=0.8, ls="--",
               label=f"P1 Richardson estimate {extrapolated_peak[1] / 1e6:.1f} MPa")
ax.set(xlabel="element size on the throat fillets h_f / mm", ylabel="peak von Mises stress in the fillets / MPa",
       title="Mesh study of the throat-fillet peak")
ax.invert_xaxis()
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Replace the bonded jaws by frictional contact (`onesim.physics.solid_mechanics.contact`) and
# watch the load move to the hexagon corners; reduce the throat fillet radius and follow the
# peak stress; shorten the shank to see the support influence on the torsion field; or compute
# the first principal strain to check the small-strain assumption.
