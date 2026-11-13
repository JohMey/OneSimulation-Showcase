"""OneSimulation example: temperature field in a glass cooling flange.
Requires the installed, activated OneSimulation runtime.

A hot process fluid flows through a glass pipe with three cooling flanges. Forced
convection heats the bore, angle- and temperature-dependent natural convection cools the
outside. The 3D temperature field is checked by the global energy balance, a three-level
mesh study with Richardson extrapolation, and a 1D radial-wall estimate near the bore.
"""


# # Temperature field in a cooling flange
#
# Reaction fluids are often cooled by passing them through a glass pipe with cooling
# flanges: the fluid heats the bore, the heat is conducted through the glass into the
# flanges, and the surrounding room air carries it away by **natural convection**. The
# pipe axis is horizontal ($x$), gravity points along $-y$.
#
# The glass is described by the stationary heat equation
#
# $$\nabla\cdot(k\nabla T)=0 .$$
#
# The bore sees forced convection to the process fluid with a constant coefficient,
# $-k\,\partial_n T=h_i\,(T-T_i)$. On all outer surfaces the free-convection film
# coefficient is taken from the laminar correlation for a horizontal cylinder of
# diameter $D$ (the flange outer diameter), with a local factor $f(\theta)$ that depends
# on the angle $\theta$ measured from the lowest point of the cylinder (B. Sundén,
# *Kompendium i värmeöverföring*, Sec. 10-3, Lund Institute of Technology, 2003):
#
# $$h_a=\frac{k_\mathrm{air}}{D}\,f(\theta)\,Gr^{1/4},\qquad
#   Gr=\frac{g\,\beta\,(T-T_a)\,D^3}{\nu^2},\qquad \beta=1/T_a ,$$
#
# so the outward flux $h_a(T-T_a)\propto f(\theta)\,(T-T_a)^{5/4}$ is nonlinear in the
# surface temperature and varies around the circumference. The temperature therefore
# depends on all three coordinates: axially (pipe, flanges), radially (wall, fins) and
# around the axis (stronger cooling at the bottom than at the top).
#
# OneSimulation solves the problem with quadratic (P2) Lagrange finite elements on a
# graded tetrahedral mesh of the half flange $z\ge 0$; the vertical plane through the
# axis is a plane of symmetry because gravity lies in it. Boundary conditions on every
# face:
#
# | face | condition |
# |---|---|
# | bore | forced convection to the process fluid, $h_i=15$ W/(m² K), $T_i=363$ K |
# | all outer surfaces (pipe, flange shoulders, fins, fin tips) | natural convection $h_a(\theta, T)$ to air at $T_a=298$ K |
# | the two annular end faces of the pipe | thermally insulated (zero flux) |
# | vertical symmetry plane $z=0$ | symmetry (zero flux) |
#
# The nonlinear air-side law is applied by a chord (fixed-tangent) iteration: in every
# step each of the angular sectors of the outer surface carries a `ConvectiveHeatFlux`
# whose coefficient is the local tangent $\partial q/\partial T$ and whose effective
# ambient temperature reproduces the exact nonlinear flux at the previous iterate. At
# convergence the solved flux equals $h_a(\theta,T)\,(T-T_a)$ pointwise.
#
# **Reference.** There is no closed-form solution for this geometry. The notebook
# uses (i) the global energy balance (heat entering through the bore equals heat leaving
# to the air), (ii) a three-level mesh-convergence study of the total heat rate and of
# the peak and minimum temperatures with observed order and Richardson extrapolation,
# and (iii) as a sanity bound only, the 1D radial conduction estimate through the pipe
# wall near the insulated pipe end, where the field is closest to one-dimensional.

from dataclasses import dataclass
from importlib.metadata import version
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from onesim.cad.geometry.primitives import Geometry
from onesim.materials import IsotropicSolid
from onesim.meshing import UnstructuredSimplexMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import boundary_faces
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Flange geometry: a glass pipe of 16 mm bore with a 4 mm wall carries three flanges,
# 4 mm wide and 10 mm tall, with filleted shoulders and rounded tips; a 12 mm long
# connecting pipe with a 3 mm wall joins on one side and widens smoothly to the flange
# wall. Fused silica: $k=1.38$ W/(m K) (Incropera et al., *Fundamentals of Heat and Mass
# Transfer*, Table A.3, 300 K). Air at 298 K: $k_\mathrm{air}=0.0262$ W/(m K),
# $\nu=18\times10^{-6}$ m²/s.

mm = 1e-3
r_in = 8.0 * mm          # m, bore radius
l_pipe = 12.0 * mm       # m, connecting pipe length
t_pipe = 3.0 * mm        # m, wall thickness of the connecting pipe
t_wall = 4.0 * mm        # m, wall thickness under the flanges
h_flange = 10.0 * mm     # m, flange height above the wall
w_flange = 4.0 * mm      # m, flange width
n_flanges = 3
D = 2 * (r_in + t_wall + h_flange)   # m, outer flange diameter (44 mm)

k_glass = 1.38           # W/(m K)
h_in, T_in = 15.0, 363.0          # W/(m^2 K), K: process fluid
T_air = 298.0                      # K
k_air, nu_air, g = 0.0262, 18e-6, 9.81
theta_table = np.radians([0, 90, 100, 110, 120, 130, 140, 150, 160, 180])
f_table = np.array([0.48, 0.46, 0.45, 0.435, 0.42, 0.38, 0.35, 0.28, 0.22, 0.15])
grashof_per_kelvin = g * (1.0 / T_air) * D**3 / nu_air**2
print(f"outer diameter D = {D / mm:.0f} mm,  Gr = {grashof_per_kelvin:.3e} * (T - T_air)/K")


def incidence(points):
    """Angle from the lowest point of the horizontal cylinder (gravity along -y)."""
    p = np.atleast_2d(points)
    return np.arctan2(np.abs(p[:, 2]), -p[:, 1])


def air_prefactor(points):
    """C(theta) in q = C(theta) (T - T_air)^(5/4)."""
    return k_air / D * np.interp(incidence(points), theta_table, f_table) * grashof_per_kelvin**0.25


def air_flux(points, T):
    return air_prefactor(points) * np.maximum(np.asarray(T) - T_air, 0.0) ** 1.25


# ## Geometry and mesh
#
# OneSimulation's CAD primitives (box, cylinder, annular cylinder) and its CAD meshers
# do not offer this revolved, filleted solid, so it is built directly as a mesh: the
# meridian section $(x, r)$ is covered by curved quadrilateral blocks (transfinite
# interpolation between straight lines, quadratic and cubic Bezier fillets and circular
# fin tips), the section is swept around the axis over $180^\circ$, every prism is split
# into three tetrahedra by a conforming diagonal rule, and the tetrahedra with their
# named boundary faces are handed to `UnstructuredSimplexMesh`. The outer surface is
# split into one boundary group per angular sector so that each sector can carry its
# own film coefficient.

def line(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return lambda t: (1 - t)[:, None] * p + t[:, None] * q


def bezier(*control):
    P = np.asarray(control, float)
    n = len(P) - 1
    binom = [1, n, n * (n - 1) // 2, 1][: n + 1] if n <= 3 else None
    return lambda t: sum(binom[i] * ((1 - t) ** (n - i) * t**i)[:, None] * P[i] for i in range(n + 1))


def arc(centre, radius, a0, a1):
    c = np.asarray(centre, float)
    return lambda t: c + radius * np.column_stack((np.cos(a0 + (a1 - a0) * t), np.sin(a0 + (a1 - a0) * t)))


def transfinite(bottom, top, left, right, u, v):
    B, T, L, R = bottom(u), top(u), left(v), right(v)
    U, V = np.meshgrid(u, v, indexing="ij")
    U, V = U[..., None], V[..., None]
    return ((1 - V) * B[:, None] + V * T[:, None] + (1 - U) * L[None] + U * R[None]
            - (1 - U) * (1 - V) * B[0] - U * (1 - V) * B[-1] - (1 - U) * V * T[0] - U * V * T[-1])


def graded(n, growth):
    sizes = growth ** np.arange(n)
    return np.concatenate(([0.0], np.cumsum(sizes) / sizes.sum()))


def section_blocks(level):
    """Curved blocks of the meridian section; neighbouring blocks share node rows."""
    L = level
    wall = graded(2 * L, 1.0)                  # radial, through the wall
    side = graded(L, 1.0)                      # axial, under a fillet (2 mm)
    mid = graded(2 * L, 1.0)                   # axial, under a fin (4 mm)
    fin = graded(3 * L, 1.15 ** (-2 / L))      # radial along the fin, finer towards the tip
    tip = graded(L, 1.0)
    y0, y1 = r_in, r_in + t_wall
    blocks = [
        (line((-l_pipe, y0), (-2 * l_pipe / 3, y0)), line((-l_pipe, y0 + t_pipe), (-2 * l_pipe / 3, y0 + t_pipe)),
         line((-l_pipe, y0), (-l_pipe, y0 + t_pipe)), line((-2 * l_pipe / 3, y0), (-2 * l_pipe / 3, y0 + t_pipe)),
         graded(L, 1.0), wall),
        (line((-2 * l_pipe / 3, y0), (0, y0)),
         bezier((-2 * l_pipe / 3, y0 + t_pipe), (-l_pipe / 3, y1), (-l_pipe / 3, y1), (0, y1)),
         line((-2 * l_pipe / 3, y0), (-2 * l_pipe / 3, y0 + t_pipe)), line((0, y0), (0, y1)),
         graded(2 * L, 1.0), wall),
    ]
    w = w_flange
    for n in range(n_flanges):
        x0 = 2 * w * n
        shoulder = y1 + w / 2
        top = y1 + h_flange - w / 2
        blocks += [
            (line((x0, y0), (x0 + w / 2, y0)), bezier((x0, y1), (x0 + w / 2, y1), (x0 + w / 2, shoulder)),
             line((x0, y0), (x0, y1)), line((x0 + w / 2, y0), (x0 + w / 2, shoulder)), side, wall),
            (line((x0 + w / 2, y0), (x0 + 3 * w / 2, y0)), line((x0 + w / 2, shoulder), (x0 + 3 * w / 2, shoulder)),
             line((x0 + w / 2, y0), (x0 + w / 2, shoulder)), line((x0 + 3 * w / 2, y0), (x0 + 3 * w / 2, shoulder)),
             mid, wall),
            (line((x0 + 3 * w / 2, y0), (x0 + 2 * w, y0)),
             bezier((x0 + 3 * w / 2, shoulder), (x0 + 3 * w / 2, y1), (x0 + 2 * w, y1)),
             line((x0 + 3 * w / 2, y0), (x0 + 3 * w / 2, shoulder)), line((x0 + 2 * w, y0), (x0 + 2 * w, y1)),
             side, wall),
            (line((x0 + w / 2, shoulder), (x0 + 3 * w / 2, shoulder)), line((x0 + w / 2, top), (x0 + 3 * w / 2, top)),
             line((x0 + w / 2, shoulder), (x0 + w / 2, top)), line((x0 + 3 * w / 2, shoulder), (x0 + 3 * w / 2, top)),
             mid, fin),
            (line((x0 + w / 2, top), (x0 + 3 * w / 2, top)), arc((x0 + w, top), w / 2, 3 * np.pi / 4, np.pi / 4),
             arc((x0 + w, top), w / 2, np.pi, 3 * np.pi / 4), arc((x0 + w, top), w / 2, 0.0, np.pi / 4),
             mid, tip),
        ]
    return blocks


def section_mesh(level):
    points, index, quads = [], {}, []
    for bottom, top, left, right, u, v in section_blocks(level):
        X = transfinite(bottom, top, left, right, u, v)
        ids = np.empty(X.shape[:2], dtype=int)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                key = (round(X[i, j, 0] / 1e-10), round(X[i, j, 1] / 1e-10))
                if key not in index:
                    index[key] = len(points)
                    points.append(X[i, j])
                ids[i, j] = index[key]
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                quads.append((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))
    return np.array(points), np.array(quads)


# Conforming prism split (Dompierre et al., Proc. 8th Int. Meshing Roundtable, 1999).
PRISM_ROTATIONS = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                            [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]])


def split_prism(vertices):
    v = np.asarray(vertices)[PRISM_ROTATIONS[int(np.argmin(vertices))]]
    if min(v[1], v[5]) < min(v[2], v[4]):
        return [(v[0], v[1], v[2], v[5]), (v[0], v[1], v[5], v[4]), (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]), (v[0], v[4], v[2], v[5]), (v[0], v[4], v[5], v[3])]


@dataclass(frozen=True, eq=False, kw_only=True)
class HalfFlange(Geometry):
    """Half cooling flange (z >= 0) with the outer surface split into angular sectors."""

    sectors: int
    dimension = 3

    def boundary_groups(self):
        names = ("bore", "pipe_end", "flange_end", "symmetry_plane") + tuple(
            f"outer_{s:02d}" for s in range(self.sectors))
        return {**{name: (name,) for name in names}, "all": names}


def build_mesh(level):
    section, quads = section_mesh(level)
    triangles = []
    for a, b, c, d in quads:
        if np.linalg.norm(section[a] - section[c]) <= np.linalg.norm(section[b] - section[d]):
            triangles += [(a, b, c), (a, c, d)]
        else:
            triangles += [(a, b, d), (b, c, d)]
    triangles = np.array(triangles)
    n_theta = 4 * level
    thetas = np.linspace(0.0, np.pi, n_theta + 1)      # theta = 0 at the bottom (y < 0)
    n_section = len(section)
    x = np.tile(section[:, 0], n_theta + 1)
    r = np.tile(section[:, 1], n_theta + 1)
    th = np.repeat(thetas, n_section)
    coordinates = np.column_stack((x, -r * np.cos(th), r * np.sin(th)))
    cells = []
    for j in range(n_theta):
        lower, upper = j * n_section, (j + 1) * n_section
        for a, b, c in triangles:
            cells += split_prism((a + lower, b + lower, c + lower, a + upper, b + upper, c + upper))
    cells = np.array(cells)
    p = coordinates[cells]
    signed = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    cells[signed < 0] = cells[signed < 0][:, [0, 2, 1, 3]]

    geometry = HalfFlange(sectors=n_theta)
    groups = {name: [] for name in geometry.boundary_groups()["all"]}
    tol = 1e-9
    for face in boundary_faces(cells, coordinates):
        v = np.asarray(face.vertices)
        layer, planar = v // n_section, section[v % n_section]
        if np.all(layer == 0) or np.all(layer == n_theta):
            name = "symmetry_plane"
        elif np.all(np.abs(planar[:, 1] - r_in) < tol):
            name = "bore"
        elif np.all(np.abs(planar[:, 0] + l_pipe) < tol):
            name = "pipe_end"
        elif np.all(np.abs(planar[:, 0] - 2 * w_flange * n_flanges) < tol):
            name = "flange_end"
        else:
            name = f"outer_{layer.min():02d}"
        groups[name].append(tuple(v))
    mesh = UnstructuredSimplexMesh(geometry=geometry, coordinates=coordinates, cells=cells,
                                   boundary_facets=groups)
    return dict(geometry=geometry, mesh=mesh, coordinates=coordinates, groups=groups, section=section,
                triangles=triangles, n_section=n_section, n_theta=n_theta, thetas=thetas, level=level)


# ## Model
#
# Glass is homogeneous and isotropic. The bore carries a `ConvectiveHeatFlux`, the end
# faces and the symmetry plane an explicit zero `HeatFlux`, and every outer sector a
# `ConvectiveHeatFlux` linearised about the previous iterate $T^{(n)}$:
#
# $$q \approx q(T^{(n)}) + h_s\,(T-T^{(n)}) = h_s\,(T-T_\mathrm{eff}),\qquad
#   T_\mathrm{eff}=T^{(n)}-q(T^{(n)})/h_s ,$$
#
# with the sector tangent $h_s=\tfrac54\,C(\theta_s)\,\overline{(T^{(n)}-T_a)}^{1/4}$.

glass = IsotropicSolid(density=2203.0, specific_heat_capacity=703.0, thermal_conductivity=k_glass,
                       youngs_modulus=73.1e9, poisson_ratio=0.17, thermal_expansion_coefficient=5.5e-7)

BARY = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def face_quadrature(data, names):
    faces = np.array([f for name in names for f in data["groups"][name]])
    p = data["coordinates"][faces]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return np.einsum("qk,fkd->fqd", BARY, p).reshape(-1, 3), np.repeat(area / 3.0, 3)


class PreviousIterate:
    """T^(n) as a function of position: the FE field, or a coarser mesh's field."""

    def __init__(self, field=None, nodes=None, values=None, guess=330.0):
        self.field, self.guess = field, guess
        self.tree = None if nodes is None else cKDTree(nodes)
        self.values = values

    def __call__(self, points):
        points = np.atleast_2d(points)
        if self.field is None:
            return np.full(len(points), self.guess)
        try:
            return np.asarray(self.field.evaluate(points)).reshape(-1)
        except ValueError:
            # A point of the finer faceted surface may lie just outside the coarser
            # mesh: take the nearest coarse vertex (only used for the starting iterate).
            return self.values[self.tree.query(points)[1]]


def effective_ambient(previous, h_s):
    """T_eff(x) = T0 - q(T0)/h_s for one sector, as a coordinate callback."""
    def value(x, y, z):
        pts = np.column_stack((x, y, z))
        T0 = previous(pts)
        return T0 - air_flux(pts, T0) / h_s
    return value


def solve_level(data, previous, contraction=None, tolerance=2e-3, max_steps=12):
    geometry, mesh, groups = data["geometry"], data["mesh"], data["groups"]
    model = HeatConduction(domain=geometry.domain(), material=glass, heating=0.0)
    b = geometry.boundaries
    outer_names = [f"outer_{s:02d}" for s in range(data["n_theta"])]
    theta_mid = 0.5 * (data["thetas"][:-1] + data["thetas"][1:])
    check_points = {n: face_quadrature(data, (n,))[0] for n in outer_names}
    history = []
    for step in range(max_steps):
        conditions = [
            ConvectiveHeatFlux(boundary=b.bore, physics=model, heat_transfer_coefficient=h_in,
                               ambient_temperature=T_in),
            HeatFlux(boundary=b.pipe_end | b.flange_end | b.symmetry_plane, physics=model, value=0.0),
        ]
        for s, name in enumerate(outer_names):
            dT = max(float(np.mean(previous(check_points[name]))) - T_air, 0.5)
            point = np.array([[0.0, -np.cos(theta_mid[s]), np.sin(theta_mid[s])]])
            h_s = 1.25 * float(air_prefactor(point)[0]) * dT**0.25

            conditions.append(ConvectiveHeatFlux(boundary=getattr(b, name), physics=model,
                                                 heat_transfer_coefficient=h_s,
                                                 ambient_temperature=effective_ambient(previous, h_s)))
        problem = Problem(name="Glass cooling flange", geometry=geometry, physics=(model,),
                          boundary_conditions=tuple(conditions))
        simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
            spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()))
        field = simulation.solve_stationary().field(model.temperature)
        all_points = np.concatenate([check_points[n] for n in outer_names])
        change = float(np.max(np.abs(np.asarray(field.evaluate(all_points)).reshape(-1) - previous(all_points))))
        history.append(change)
        previous = PreviousIterate(field)
        # Remaining iteration error ~ contraction factor x last change; the factor is
        # measured from two successive changes (on this level, or on the previous one).
        if len(history) >= 2:
            contraction = history[-1] / history[-2]
        if contraction is not None and contraction < 0.5 and contraction * change < tolerance:
            break

    # Heat rates of the full flange (twice the half model).
    q_pts, q_w = face_quadrature(data, ("bore",))
    Q_in = 2 * float(np.sum(q_w * h_in * (T_in - np.asarray(field.evaluate(q_pts)).reshape(-1))))
    o_pts, o_w = face_quadrature(data, outer_names)
    T_outer = np.asarray(field.evaluate(o_pts)).reshape(-1)
    Q_out = 2 * float(np.sum(o_w * air_flux(o_pts, T_outer)))
    nodal = np.asarray(field.evaluate(data["coordinates"])).reshape(-1)
    return dict(field=field, nodal=nodal, Q_in=Q_in, Q_out=Q_out, T_max=float(nodal.max()),
                T_min=float(nodal.min()), history=history, contraction=contraction,
                h_outer=(T_outer, o_pts))


# ## Solve
#
# Three meshes with cell counts in the ratio 2 : 3 : 4 in every direction. Each finer
# level starts its chord iteration from the converged field of the previous level. The
# chord iteration contracts the error by a roughly constant factor $\rho$ per step,
# measured from two successive changes of the outer surface temperature; the iteration
# stops once the estimated remaining error $\rho\,\max|T^{(n+1)}-T^{(n)}|$ is below
# 2 mK, far below the discretisation error.

levels = (2, 3, 4)
runs = []
previous, contraction = PreviousIterate(guess=330.0), None
for level in levels:
    start = time.time()
    data = build_mesh(level)
    out = solve_level(data, previous, contraction)
    runs.append((data, out))
    previous = PreviousIterate(out["field"], data["coordinates"], out["nodal"])
    contraction = out["contraction"]
    steps = ", ".join(f"{c:.1e}" for c in out["history"])
    print(f"level {level}: {data['mesh'].cell_count:6d} tetrahedra, {len(data['coordinates']):6d} vertices, "
          f"chord-step changes [{steps}] K (contraction {contraction:.3f}"
          f"{', from the previous level' if len(out['history']) == 1 else ''}), "
          f"Q = {out['Q_in']:.5f} W, T in [{out['T_min']:.3f}, {out['T_max']:.3f}] K  ({time.time() - start:.0f} s)")
data, out = runs[-1]


# ## Comparison with the reference
#
# **Energy balance and mesh convergence.** For three meshes with sizes $h\propto 1/\ell$
# the observed order $p$ solves
# $(f_1-f_2)/(f_2-f_3)=(h_1^p-h_2^p)/(h_2^p-h_3^p)$, and the Richardson estimate is
# $f_3+(f_3-f_2)/((h_2/h_3)^p-1)$.

def richardson(values, levels):
    f1, f2, f3 = values
    h1, h2, h3 = 1.0 / np.asarray(levels, float)
    ratio = (f1 - f2) / (f2 - f3) if f2 != f3 else np.inf
    try:
        p = brentq(lambda p: (h1**p - h2**p) / (h2**p - h3**p) - ratio, 0.2, 10.0)
    except ValueError:
        return np.nan, f3
    return p, f3 + (f3 - f2) / ((h2 / h3) ** p - 1.0)


print("Energy balance on the finest mesh (full flange)")
print(f"  heat entering through the bore   Q_in  = {out['Q_in']:.5f} W")
print(f"  heat leaving to the air          Q_out = {out['Q_out']:.5f} W")
print(f"  relative imbalance               {abs(out['Q_in'] - out['Q_out']) / out['Q_in']:.2e}")
print("Mesh study (h ~ 1/level)")
study = {}
for label, key in (("Q_in / W", "Q_in"), ("Q_out / W", "Q_out"), ("T_max / K", "T_max"), ("T_min / K", "T_min")):
    vals = [o[key] for _, o in runs]
    p, ext = richardson(vals, levels)
    study[key] = (vals, p, ext)
    print(f"  {label:10s} " + "  ".join(f"{v:.6f}" for v in vals)
          + f"   order {p:.2f}, extrapolated {ext:.6f}, change last two meshes {abs(vals[-1] - vals[-2]):.1e}"
          + (f" ({100 * abs(vals[-1] / vals[-2] - 1):.3f} %)" if key.startswith("Q") else " K"))
Q_ext = study["Q_in"][2]
print(f"  finest-mesh heat rate vs Richardson estimate: {100 * (out['Q_in'] / Q_ext - 1):+.3f} %")

# **1D radial wall estimate (sanity bound).** On the insulated end face of the pipe the
# axial temperature gradient vanishes, so this is where the wall comes closest to pure
# radial conduction. For radial conduction the
# local bore flux $q_i$ fixes the drop across the wall,
# $\Delta T_\mathrm{1D}=q_i\,r_i\ln(r_o/r_i)/k$, and the outer flux must be
# $q_o=q_i\,r_i/r_o$. The connecting pipe is only 12 mm long, shorter than the axial
# conduction length $\sqrt{k\,t/(h_i+h_a)}\approx 15$ mm of the glass wall, so part of
# the heat entering the bore there still flows axially into the flanges: the 3D wall
# drop and outer flux must lie *below* the 1D values. The comparison is a plausibility
# bound, not the main reference.
field = out["field"]
r_out = r_in + t_pipe
print("1D radial estimate at the pipe end, x = -11.9 mm")
for theta_deg in (0.0, 90.0, 180.0):
    th = np.radians(theta_deg)
    inner = np.array([[-11.9 * mm, -r_in * np.cos(th) * (1 + 1e-6), r_in * np.sin(th) * (1 + 1e-6)]])
    outer = np.array([[-11.9 * mm, -r_out * np.cos(th) * (1 - 1e-6), r_out * np.sin(th) * (1 - 1e-6)]])
    T_i = float(np.asarray(field.evaluate(inner))[0])
    T_o = float(np.asarray(field.evaluate(outer))[0])
    q_i = h_in * (T_in - T_i)
    dT_1d = q_i * r_in * np.log(r_out / r_in) / k_glass
    q_o = float(air_flux(outer, T_o)[0])
    print(f"  theta = {theta_deg:5.1f} deg: wall drop 3D {T_i - T_o:.4f} K vs 1D {dT_1d:.4f} K "
          f"({100 * ((T_i - T_o) / dT_1d - 1):+.1f} %);  outer flux 3D {q_o:.2f} vs 1D {q_i * r_in / r_out:.2f} W/m^2 "
          f"({100 * (q_o / (q_i * r_in / r_out) - 1):+.1f} %)")


# ## Figure
#
# Left: surface temperature of the complete flange (the computed half mirrored about
# the symmetry plane), seen obliquely from the front; gravity points down. Middle: the vertical
# section through the axis, upper half at the top of the flange ($\theta=180^\circ$),
# lower half at the bottom ($\theta=0$). Right: total heat rate on the three meshes
# with its Richardson estimate, and the local air-side film coefficient over the
# outer surface.

coords, nodal, groups = data["coordinates"], out["nodal"], data["groups"]
outer_names = [f"outer_{s:02d}" for s in range(data["n_theta"])]
cmap = cm.plasma
norm = colors.Normalize(vmin=nodal.min(), vmax=nodal.max())
cream = "#fbf5eb"
fig = plt.figure(figsize=(17, 6.2), constrained_layout=True)
fig.patch.set_facecolor(cream)
layout = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.25, 1.0))

ax3d = fig.add_subplot(layout[:, 0], projection="3d")
ax3d.set_facecolor(cream)
faces = np.array([f for n in outer_names for f in groups[n]] + groups["pipe_end"] + groups["flange_end"])
tri = coords[faces]
mirror = tri * np.array([1.0, 1.0, -1.0])
face_T = nodal[faces].mean(axis=1)
polys = np.concatenate((tri, mirror)) / mm
ax3d.add_collection3d(Poly3DCollection(polys[:, :, [0, 2, 1]],
                                       facecolors=cmap(norm(np.concatenate((face_T, face_T)))),
                                       edgecolors="none", linewidths=0))
R = D / 2 / mm
ax3d.set(xlim=(-l_pipe / mm, 24), ylim=(-R, R), zlim=(-R, R), xlabel="x / mm", ylabel="z / mm", zlabel="y / mm")
ax3d.set_box_aspect((36, 2 * R, 2 * R))
ax3d.view_init(elev=14, azim=-58)
ax3d.set_title("Surface temperature (full flange)", pad=18)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3d, shrink=0.65, label="T / K", pad=0.08)

ax_cut = fig.add_subplot(layout[:, 1])
ax_cut.set_facecolor("#fffaf2")
section, triangles, n_section = data["section"], data["triangles"], data["n_section"]
top_T = nodal[data["n_theta"] * n_section + np.arange(n_section)]
bottom_T = nodal[np.arange(n_section)]
levels_T = np.linspace(nodal.min(), nodal.max(), 41)
for sign, values in ((1.0, top_T), (-1.0, bottom_T)):
    tri_cut = Triangulation(section[:, 0] / mm, sign * section[:, 1] / mm, triangles)
    filled = ax_cut.tricontourf(tri_cut, values, levels=levels_T, cmap=cmap)
    ax_cut.tricontour(tri_cut, values, levels=levels_T[::4], colors="k", linewidths=0.3, alpha=0.5)
ax_cut.axhline(0.0, color="0.4", lw=0.6, ls=":")
ax_cut.set_aspect("equal")
ax_cut.set(xlabel="x / mm", ylabel="vertical position / mm",
           title="Vertical section through the axis\n(top: theta = 180 deg, bottom: theta = 0)")

ax_q = fig.add_subplot(layout[0, 2])
ax_q.set_facecolor("#fffaf2")
Q_vals = study["Q_in"][0]
ax_q.plot(levels, Q_vals, "o-", color="#b5532a", label="Q through the bore")
ax_q.plot(levels, study["Q_out"][0], "s--", mfc="none", color="#5a4a8a", label="Q to the air")
ax_q.axhline(Q_ext, color="#7a3b12", lw=1.0, ls=":", label=f"Richardson estimate {Q_ext:.4f} W")
ax_q.set_xticks(levels)
ax_q.set(xlabel="mesh level", ylabel="heat rate / W", title="Mesh study of the cooling power")
ax_q.legend(fontsize=8)
ax_q.grid(alpha=0.3)

ax_h = fig.add_subplot(layout[1, 2])
ax_h.set_facecolor("#fffaf2")
T_outer, o_pts = out["h_outer"]
h_local = air_flux(o_pts, T_outer) / np.maximum(T_outer - T_air, 1e-9)
sc = ax_h.scatter(np.degrees(incidence(o_pts)), h_local, c=np.hypot(o_pts[:, 1], o_pts[:, 2]) / mm,
                  s=2, cmap="viridis")
fig.colorbar(sc, ax=ax_h, label="radius / mm")
ax_h.set(xlabel="angle from the bottom theta / deg", ylabel="h_air / (W/(m^2 K))",
         title="Local natural-convection coefficient")
ax_h.grid(alpha=0.3)
plt.show()


# ## Going further
#
# Sweep the bore radius between 6 and 10 mm at fixed flange dimensions: the bore-side
# film resistance dominates, so a wider bore raises the cooling power noticeably. Add
# a fourth flange, or replace glass by a metal, and compare how much of the temperature
# drop then sits in the wall versus the two films.
