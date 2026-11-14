"""OneSimulation example: 3D thermal bridge of a balcony slab between two floors (ISO 10211 case 3).
Requires the installed, activated OneSimulation runtime.

The steady 3D temperature field of the third test reference case of EN ISO 10211 - an
external building corner whose floor slab runs through the insulated wall and continues
outside as a balcony - is computed with cubic hexahedral finite elements (64-node Q3
Lagrange elements) on graded block meshes. All tabulated reference values of the standard
(heat flows, coupling coefficients, minimum surface temperatures and temperature weighting
factors) are graded against its tolerances, supported by an h-study with the cubic
elements, an element-order comparison (Q1, Q2, Q3 on the same mesh) and the energy balance.
"""


# # Thermal bridge in 3D: a balcony slab between two heated floors
#
# EN ISO 10211 (*Thermal bridges in building construction - Heat flows and surface
# temperatures - Detailed calculations*) validates numerical methods with four test
# reference cases; a program that reproduces all of them is classed as a
# *three-dimensional steady-state high-precision method*. Cases 1 and 2 are
# two-dimensional, cases 3 and 4 three-dimensional. This notebook computes **case 3**, a
# three-dimensional external corner of a building between two floors:
#
# - two external walls meet at a corner; each wall consists of an exterior leaf (brick),
#   an insulation layer and an interior leaf (aerated concrete);
# - the concrete floor slab between the lower and the upper room cuts through the
#   interior leaf and the insulation of one wall, crosses the brick leaf and continues
#   outside as a **balcony**, 0.6 m deep - the classical slab-edge thermal bridge;
# - a thin floor finish lies on the slab in the upper room.
#
# Coordinates (m): the outer corner of the building is the origin, the external faces
# are $x=0$ and $y=0$, the balcony sticks out to $y<0$, $z$ is vertical. The model ends at
# the cut planes $x=1.3$, $y=1.3$, $z=0$ and $z=2.15$.
#
# | no. | part | extent / m | $\lambda$ / (W/(m K)) |
# |---|---|---|---|
# | 3 | exterior leaf (brick) | wall region with $x<0.10$ or $y<0.10$ | 1.0 |
# | 2 | insulation | wall region with $0.10\le\min(x,y)<0.15$ | 0.04 |
# | 1 | interior leaf (aerated concrete) | wall region with $0.15\le\min(x,y)<0.30$ | 0.7 |
# | 4 | floor slab and balcony (concrete) | $0.15\le x\le1.3,\ -0.6\le y\le1.3,\ 1.0\le z\le1.15$ | 2.5 |
# | 5 | floor finish | $0.3\le x\le1.3,\ 0.3\le y\le1.3,\ 1.15\le z\le1.2$ | 1.0 |
#
# The wall region is $\{0\le x\le 1.3,\ 0\le y\le 1.3,\ \min(x,y)<0.3,\ 0\le z\le 2.15\}$;
# the slab replaces the wall materials where they overlap. The temperature solves
# $\nabla\cdot(\lambda\nabla T)=0$ with surface resistances $R_s$ (Robin condition
# $-\lambda\,\partial_nT=(T-\theta)/R_s$) towards three environments. Boundary conditions
# on every face:
#
# | face | environment | condition |
# |---|---|---|
# | $\alpha$: lower room - inner wall faces $x=0.3$ / $y=0.3$ below the slab and the slab soffit $z=1.0$ | lower room, $\theta_\alpha=20$ °C | $R_{s}=0.20$ m² K/W ($h=5$ W/(m² K)) |
# | $\beta$: upper room - inner wall faces above the floor and the floor surface $z=1.2$ | upper room, $\theta_\beta=15$ °C | $R_{s}=0.20$ m² K/W ($h=5$ W/(m² K)) |
# | $\gamma$: outside - external wall faces $x=0$, $y=0$ and all faces of the balcony | outside, $\theta_\gamma=0$ °C | $R_{s}=0.05$ m² K/W ($h=20$ W/(m² K)) |
# | $\delta$: the cut planes $x=1.3$, $y=1.3$, $z=0$, $z=2.15$ | - | adiabatic (zero flux) |
#
# The field is genuinely three-dimensional: it depends on the depth into each wall, on
# the distance from the corner and on the height relative to the slab.
#
# **Numerical method.**
# - Spatial: continuous Galerkin finite elements with **cubic tensor-product Lagrange
#   elements (Q3, 64 nodes per hexahedron)** on a block mesh that is aligned with every
#   material interface and geometrically graded towards the slab, the inner corner and
#   the balcony root. For comparison, the same problem is solved with Q1 (8-node) and Q2
#   (27-node) elements on the finest mesh.
# - Steady linear problem, sparse direct LU solver.
# - Because the problem is linear, a second solve with $\theta_\alpha=1$, $\theta_\beta=\theta_\gamma=0$
#   gives the thermal coupling coefficients $L_{ij}$ and the temperature weighting factors
#   $g$ of the standard by superposition.
#
# **Reference.** The test reference values of EN ISO 10211 (editions 2007 and 2017,
# normative annex on the validation of calculation methods), case 3: heat flows $\Phi_\alpha=46.09$ W, $\Phi_\beta=13.89$ W,
# $\Phi_\gamma=59.98$ W, coupling coefficients $L_{\alpha\gamma}=1.781$,
# $L_{\beta\gamma}=1.624$, $L_{\alpha\beta}=2.094$ W/K, minimum surface temperatures
# $\theta_{\min,\alpha}=11.32$ °C and $\theta_{\min,\beta}=11.11$ °C with their weighting
# factors. The standard requires heat flows within 1 % and temperatures within 0.1 K.
# Supporting checks: the energy balance and a three-level h-study of the cubic elements.

from dataclasses import replace
import gc
import time
import numpy as np

from onesim.cad.d2 import Plane
from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import (
    ExplicitGeometry, Geometric, Piecewise, StructuredHexahedralMesher, Uniform, region_mesh,
)
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

LAMBDA = {"brick": 1.0, "insulation": 0.04, "aerated": 0.7, "slab": 2.5, "finish": 1.0}
THETA = {"alpha": 20.0, "beta": 15.0, "gamma": 0.0}
H_S = {"alpha": 1 / 0.20, "beta": 1 / 0.20, "gamma": 1 / 0.05}
X_END, Y_BAL, Z_TOP = 1.3, -0.6, 2.15
Z_SLAB, Z_FLOOR, Z_FINISH = 1.0, 1.15, 1.2
REF = {"Phi_alpha": 46.09, "Phi_beta": 13.89, "Phi_gamma": 59.98,
       "L_ag": 1.781, "L_bg": 1.624, "L_ab": 2.094,
       "T_alpha": 11.32, "T_beta": 11.11,
       "g_alpha": (0.378, 0.399, 0.223), "g_beta": (0.331, 0.214, 0.455)}
TOL_PHI, TOL_T = 0.01, 0.1

# One-dimensional U-values of the undisturbed wall (between the surface resistances) as
# orientation: lower room to outside and upper room to outside.
R_wall = 0.10 / LAMBDA["brick"] + 0.05 / LAMBDA["insulation"] + 0.15 / LAMBDA["aerated"]
U_wall = 1 / (0.20 + R_wall + 0.05)
print(f"undisturbed wall: R = {R_wall:.4f} m^2 K/W, U = {U_wall:.4f} W/(m^2 K)")


# ## Geometry and mesh
#
# The construction is declared as a union of boxes: the two wall legs of the external
# corner, the slab that runs through the wall into the balcony, and the floor finish. A
# graded tensor-product block mesh of the bounding box has a node plane on every material
# interface ($x,y\in\{0,0.10,0.15,0.30\}$, $y=-0.6$, $z\in\{1.0,1.15,1.2\}$); `region_mesh`
# keeps the hexahedra inside the construction (the two rooms and the air around the
# balcony are dropped). Each layer gets several cells (brick 2, insulation 2, interior
# leaf 3, slab 3, floor finish 2 on the finest mesh), and the cells in the rooms and in the
# balcony are geometrically graded from 20 mm next to the wall, the slab and the balcony
# root. A mesh level $\ell$ scales the cell counts by $\ell$ and the first cell width by
# $1/\ell$.
#
# The boundary faces are assigned to the environments of the standard by their position:
# $\gamma$ outside (the external wall faces and the balcony), $\alpha$ the lower room
# (below the slab soffit), $\beta$ the upper room, and $\delta$ the adiabatic cut planes. The
# room faces are further split into walls and horizontal faces for the figure.

wall_x = Box(length=0.30, width=X_END, height=Z_TOP)
wall_y = Box(length=X_END - 0.30, width=0.30, height=Z_TOP, origin=(0.30, 0.0, 0.0))
slab = Box(length=X_END - 0.15, width=X_END - Y_BAL, height=Z_FLOOR - Z_SLAB, origin=(0.15, Y_BAL, Z_SLAB))
finish = Box(length=X_END - 0.30, width=X_END - 0.30, height=Z_FINISH - Z_FLOOR, origin=(0.30, 0.30, Z_FLOOR))
construction = wall_x | wall_y | slab.named("slab") | finish.named("finish")
bounding_box = Box(length=X_END, width=X_END - Y_BAL, height=Z_TOP, origin=(0.0, Y_BAL, 0.0))


def environment(point, normal):
    """Environment of a boundary face from its centre and normal."""
    x, y, z = point
    if min(abs(z), abs(z - Z_TOP), abs(x - X_END), abs(y - X_END)) < 1e-9:
        return "delta"
    if abs(x) < 1e-9 or abs(y) < 1e-9 or y < 0.0:
        return "gamma"
    room = "alpha" if z <= Z_SLAB + 1e-9 else "beta"
    return room + ("_wall" if abs(normal[2]) < 0.5 else "_horizontal")


environments = ExplicitGeometry(
    dimension=3, boundary_names=("alpha_wall", "alpha_horizontal", "beta_wall", "beta_horizontal", "gamma", "delta"),
    groups={"alpha": ("alpha_wall", "alpha_horizontal"), "beta": ("beta_wall", "beta_horizontal")},
    classifier=environment)


def graded(length, cells, first, toward="start"):
    """Geometric cells growing away from ``toward``; uniform when they cannot grow."""
    return (length, Uniform(cells) if cells * first >= length or cells == 1
            else Geometric(cells, first=first, toward=toward))


def mesh_for(level):
    c = lambda n: max(1, int(np.floor(n * level + 0.5)))
    h0 = 0.02 / level
    wall = (graded(0.10, c(2), h0), graded(0.05, c(2), h0), graded(0.15, c(3), h0, "end"),
            graded(X_END - 0.30, c(5), h0))
    grading = (Piecewise(wall), Piecewise((graded(-Y_BAL, c(4), h0, "end"),) + wall),
               Piecewise((graded(Z_SLAB, c(5), h0, "end"), graded(Z_FLOOR - Z_SLAB, c(3), h0),
                          graded(Z_FINISH - Z_FLOOR, c(2), h0), graded(Z_TOP - Z_FINISH, c(5), h0))))
    block = StructuredHexahedralMesher(cells=tuple(g.cells for g in grading), grading=grading).generate(bounding_box)
    return region_mesh(block, construction, geometry=environments)


LEVELS = (0.5, 0.75, 1.0)
meshes = {level: mesh_for(level) for level in LEVELS}
for level, mesh in meshes.items():
    print(f"level {level}: {mesh.cell_count} hexahedra, volume {mesh.cell_volumes.sum():.5f} m^3")


# ## Model
#
# The conductivity is region-wise (first match wins): the slab (it runs through the
# wall), the floor finish, the brick and insulation layers of the wall (L-shaped, at
# depths 0-0.10 m and 0.10-0.15 m from the outside), and the aerated concrete of the
# interior leaf everywhere else. Every quadrature point lies strictly inside one
# hexahedron and every hexahedron inside one material, so each element sees exactly one
# conductivity.

brick = Box(length=0.10, width=X_END, height=Z_TOP) | Box(length=X_END, width=0.10, height=Z_TOP)
insulation = (Box(length=0.05, width=X_END - 0.10, height=Z_TOP, origin=(0.10, 0.10, 0.0))
              | Box(length=X_END - 0.10, width=0.05, height=Z_TOP, origin=(0.10, 0.10, 0.0)))
conductivity = regionwise(((slab, LAMBDA["slab"]), (finish, LAMBDA["finish"]), (brick, LAMBDA["brick"]),
                           (insulation, LAMBDA["insulation"])), default=LAMBDA["aerated"], unit="W/m/K")
material = replace(IsotropicSolid(density=1500.0, specific_heat_capacity=1000.0, thermal_conductivity=1.0,
                                  youngs_modulus=1e9, poisson_ratio=0.3, thermal_expansion_coefficient=0.0),
                   thermal_conductivity=conductivity)
b = environments.boundaries


def solve(level, order, theta, keep_field=False):
    """One steady solve; returns heat flows and surface temperatures (and the field if asked)."""
    start = time.time()
    mesh = meshes[level]
    model = HeatConduction(domain=environments.domain(), material=material, heating=0.0)
    problem = Problem(
        name="ISO 10211 case 3", geometry=environments, physics=(model,),
        boundary_conditions=tuple(
            ConvectiveHeatFlux(boundary=getattr(b, name), physics=model, heat_transfer_coefficient=H_S[name],
                               ambient_temperature=273.15 + theta[name]) for name in ("alpha", "beta", "gamma"))
        + (HeatFlux(boundary=b.delta, physics=model, value=0.0),))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=order)), linear_solver=SparseDirect())).solve_stationary()
    field = result.field(model.temperature)
    out = dict(level=level, order=order, cells=mesh.cell_count,
               unknowns=result.operators[model.model_id].problem_size)
    if keep_field:
        out["field"] = field
    for name in ("alpha", "beta", "gamma"):
        boundary = getattr(b, name)
        out["Phi_" + name] = -field.boundary_flux(boundary)
        surface = field.sample(along=boundary, subdivisions=2 * order)
        T = np.asarray(surface.values).ravel() - 273.15
        out["T_" + name] = float(T.min())
        out["where_" + name] = np.asarray(surface.positions)[np.argmin(T)]
        out["surface_" + name] = T
    out["seconds"] = time.time() - start
    del result, field
    gc.collect()
    return out


# ## Solve
#
# The finest Q3 mesh is first solved with the unit load $\theta_\alpha=1$ °C,
# $\theta_\beta=\theta_\gamma=0$ for the coupling coefficients and weighting factors. Then the
# design load runs: the h-study with the cubic elements on the three mesh levels and the
# order comparison Q1/Q2 on the finest mesh. Only the finest cubic design field is kept in
# memory (for the figure).

unit = solve(LEVELS[-1], 3, {"alpha": 1.0, "beta": 0.0, "gamma": 0.0})
runs = []
for level, order in [(level, 3) for level in LEVELS] + [(LEVELS[-1], 1), (LEVELS[-1], 2)]:
    runs.append(solve(level, order, THETA, keep_field=(level, order) == (LEVELS[-1], 3)))
for r in runs:
    print(f"Q{r['order']} level {r['level']}: {r['cells']} cells, {r['unknowns']} unknowns, "
          f"Phi_alpha = {r['Phi_alpha']:.4f} W, Phi_beta = {r['Phi_beta']:.4f} W, Phi_gamma = {r['Phi_gamma']:.4f} W, "
          f"T_min,alpha = {r['T_alpha']:.4f} degC, T_min,beta = {r['T_beta']:.4f} degC  ({r['seconds']:.0f} s)")
fine = runs[2]


# ## Comparison with the reference
#
# Coupling coefficients from the unit solve (superscript $u$): $L_{\alpha\beta}=-\Phi^u_\beta$,
# $L_{\alpha\gamma}=-\Phi^u_\gamma$, and from the design solve
# $\Phi_\beta=L_{\alpha\beta}(\theta_\beta-\theta_\alpha)+L_{\beta\gamma}(\theta_\beta-\theta_\gamma)$.
# The weighting factor of an environment at a surface point is the temperature that point
# takes when that environment is at 1 and the others at 0; the points are the locations of
# the lowest surface temperatures found in the design solve.

L_ab = -unit["Phi_beta"]
L_ag = -unit["Phi_gamma"]
L_bg = (fine["Phi_beta"] - L_ab * (THETA["beta"] - THETA["alpha"])) / (THETA["beta"] - THETA["gamma"])
F = fine["field"]


def weights(name):
    k = np.argmin(fine["surface_" + name])
    g_a = unit["surface_" + name][k]
    g_b = (fine["T_" + name] - THETA["alpha"] * g_a) / THETA["beta"]
    return 1.0 - g_a - g_b, g_a, g_b


g_alpha, g_beta = weights("alpha"), weights("beta")

print("EN ISO 10211 case 3, Q3 elements on the finest mesh "
      f"({fine['cells']} hexahedra, {fine['unknowns']} unknowns)")
print("   quantity                       reference    computed    deviation   tolerance")
for key, label, value in (("Phi_alpha", "heat flow alpha / W", fine["Phi_alpha"]),
                          ("Phi_beta", "heat flow beta / W", fine["Phi_beta"]),
                          ("Phi_gamma", "heat flow gamma / W", -fine["Phi_gamma"]),
                          ("L_ag", "L alpha-gamma / (W/K)", L_ag),
                          ("L_bg", "L beta-gamma / (W/K)", L_bg),
                          ("L_ab", "L alpha-beta / (W/K)", L_ab)):
    print(f"   {label:28s} {REF[key]:9.3f}   {value:9.4f}    {100 * (value / REF[key] - 1):+7.3f} %    1 %")
for key, label in (("T_alpha", "min. temperature alpha / degC"), ("T_beta", "min. temperature beta / degC")):
    print(f"   {label:28s} {REF[key]:9.2f}   {fine[key]:9.4f}    {fine[key] - REF[key]:+7.4f} K    0.1 K")
for key, g in (("g_alpha", g_alpha), ("g_beta", g_beta)):
    print(f"   weighting factors at {key[2:]:5s} min. (gamma, alpha, beta): reference "
          + ", ".join(f"{v:.3f}" for v in REF[key]) + "; computed " + ", ".join(f"{v:.4f}" for v in g)
          + f"; largest difference {max(abs(a - b) for a, b in zip(g, REF[key])):.4f} "
          "(0.1 K on the 20 K span = 0.005)")
print(f"   heat flows between environments: alpha-gamma {L_ag * 20:.3f} W (ref. 35.62), "
      f"beta-gamma {L_bg * 15:.3f} W (ref. 24.36), alpha-beta {L_ab * 5:.3f} W (ref. 10.47)")
print("   location of the lowest temperatures: alpha at", np.round(fine["where_alpha"], 4),
      " beta at", np.round(fine["where_beta"], 4), "(the inner room corner at the slab soffit / floor surface)")

print("Energy balance (heat entering from the rooms = heat leaving to the outside)")
print(f"   Phi_alpha + Phi_beta = {fine['Phi_alpha'] + fine['Phi_beta']:.5f} W, -Phi_gamma = {-fine['Phi_gamma']:.5f} W, "
      f"relative difference {abs(fine['Phi_alpha'] + fine['Phi_beta'] + fine['Phi_gamma']) / -fine['Phi_gamma']:.1e}")


print("h-study of the cubic elements (h ~ 1/level; levels " + ", ".join(str(l) for l in LEVELS) + ")")
for key, label in (("Phi_alpha", "Phi_alpha / W"), ("Phi_beta", "Phi_beta / W"), ("Phi_gamma", "Phi_gamma / W"),
                   ("T_alpha", "T_min,alpha / degC"), ("T_beta", "T_min,beta / degC")):
    vals = [r[key] for r in runs[:3]]
    line = f"   {label:19s} " + "  ".join(f"{v:9.4f}" for v in vals)
    if key.startswith("Phi"):
        estimate = convergence(vals, [1.0 / level for level in LEVELS])
        order = f"{estimate.order:.2f}" if estimate.order is not None else "n/a"
        ext = f"{estimate.extrapolated:.4f}" if estimate.extrapolated is not None else "n/a"
        line += f"   change last two meshes {abs(vals[-1] / vals[-2] - 1):.1e}, observed order {order}, Richardson {ext}"
    else:
        line += f"   change last two meshes {abs(vals[-1] - vals[-2]):.4f} K"
    print(line)
print("Element order on the finest mesh")
for r in (runs[3], runs[4], runs[2]):
    print(f"   Q{r['order']}: {r['unknowns']:6d} unknowns, Phi_gamma {-r['Phi_gamma']:.4f} W "
          f"({100 * (-r['Phi_gamma'] / REF['Phi_gamma'] - 1):+.3f} %), T_min,alpha {r['T_alpha']:.4f} degC, "
          f"T_min,beta {r['T_beta']:.4f} degC")


# ## Figure
#
# Top: surface temperature of the room walls (left) and seen from outside with the balcony
# (middle), and the temperature along the inner room corner $x=y=0.3$ (right). Bottom:
# vertical section $x=0.8$ m through the wall, the slab and the balcony; horizontal section
# through the slab $z=1.075$ m; and the convergence of the heat flow and of the minimum
# temperatures against the tolerances of the standard.

celsius = lambda T, p: T - 273.15
zline = np.linspace(0.0, Z_TOP, 431)
rooms = {"alpha: lower room": zline[zline <= Z_SLAB], "beta: upper room": zline[zline >= Z_FINISH]}
corner = lambda zz: np.asarray(F.evaluate(np.column_stack((np.full_like(zz, 0.3), np.full_like(zz, 0.3), zz)))).ravel() - 273.15
cells = [run["unknowns"] for run in runs[:3]]
fig = plots.figure(2, 3, size=(18, 11))
plots.surface(F, boundary=b.alpha_wall | b.beta_wall, component=celsius, limits=(0.0, 20.0), subdivisions=3,
              camera=plots.Camera(elevation=20, azimuth=45), label="surface temperature [°C]",
              title="Room walls (alpha 20 °C below, beta 15 °C above the slab)", ax=fig[0])
plots.surface(F, boundary=b.gamma, component=celsius, limits=(0.0, 2.5), subdivisions=3,
              camera=plots.Camera(elevation=20, azimuth=225), label="surface temperature [°C]",
              title="Seen from outside: external faces and balcony", ax=fig[1])
plots.line(y={label: (corner(zz), zz) for label, zz in rooms.items()},
           points={"ISO 10211 minimum temperatures": ([REF["T_alpha"], REF["T_beta"]], [Z_SLAB, Z_FINISH])},
           hlines={"slab and floor finish": Z_SLAB, "": Z_FINISH}, ylimits=(0, Z_TOP),
           xlabel="surface temperature in the room corner x = y = 0.3 m [°C]", ylabel="z [m]",
           title="Inner room corner: the cold spots sit at the slab", legend="lower left", ax=fig[2])
plots.section(F, plane=Plane(origin=(0.8, 0.0, 0.0), normal=(1.0, 0.0, 0.0)), component=celsius, limits=(0.0, 20.0),
              contours=19, subdivisions=3, label="T [°C]", title="Section x = 0.8 m: wall, slab and balcony",
              xlabel="y [m] (outside at y < 0)", ylabel="z [m]", ax=fig[3])
plots.section(F, plane=Plane(origin=(0.0, 0.0, 0.5 * (Z_SLAB + Z_FLOOR)), normal=(0.0, 0.0, 1.0)), component=celsius,
              limits=(0.0, 20.0), contours=19, subdivisions=3, label="T [°C]",
              title="Section z = 1.075 m through the slab and balcony", ax=fig[4])
plots.line(y={"Q3 h-study: heat flow gamma": (cells, [100 * (-run["Phi_gamma"] / REF["Phi_gamma"] - 1) for run in runs[:3]])},
           points={"Q1, Q2 on the finest mesh: heat flow gamma":
                   ([run["unknowns"] for run in runs[3:5]], [100 * (-run["Phi_gamma"] / REF["Phi_gamma"] - 1) for run in runs[3:5]])},
           right={f"Q3: min. temperature {key[2:]}": (cells, [run[key] - REF[key] for run in runs[:3]]) for key in ("T_alpha", "T_beta")},
           hlines={"heat-flow tolerance ±1 %": 100 * TOL_PHI, "": -100 * TOL_PHI}, markers=True, scale="logx",
           xlabel="unknowns", ylabel="heat flow: deviation from 59.98 W [%]",
           right_label="min. surface temperature: deviation [K]", title="Convergence against the ISO values",
           legend="lower right", ax=fig[5])
plots.show()


# ## Going further
#
# Insulate the balcony slab with a thermal break (a 60 mm insulating element of
# $\lambda=0.15$ W/(m K) in the slab inside the brick leaf) and compare the coupling
# coefficient $L_{\alpha\gamma}$ and the minimum surface temperature with the standard's
# mould criterion $f_{R_{si}}\ge0.7$; compute the linear thermal transmittance $\Psi$ of the
# slab edge by subtracting the one-dimensional wall losses; or run the two-dimensional
# test cases 1 and 2 of the standard.
