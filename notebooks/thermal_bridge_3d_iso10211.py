"""OneSimulation example: 3D thermal bridge of an iron bar through an insulation layer (ISO 10211 case 4).
Requires the installed, activated OneSimulation runtime.

The steady 3D temperature field of the fourth test reference case of EN ISO 10211 is computed
twice on the same graded block meshes: with cell-centred finite volumes (two-point fluxes with
least-squares gradients) and with quadratic hexahedral finite elements. Both are graded against
the tabulated heat flow and maximum external surface temperature of the standard and their
tolerances, the far-field one-dimensional solution, the energy balance and mesh studies.
"""


# # Thermal bridge in 3D: iron bar through an insulation layer
#
# EN ISO 10211 (*Thermal bridges in building construction - Heat flows and surface
# temperatures - Detailed calculations*) validates numerical methods with four test
# reference cases. A program that reproduces all of them is classed as a
# *three-dimensional steady-state high-precision method*. Case 4 is the three-dimensional
# one with the steepest gradients: a square insulation layer, 1 m x 1 m and 0.2 m thick,
# is pierced by an iron bar of 0.1 m x 0.05 m cross section. The bar starts flush with
# the cold (external) surface, crosses the insulation and sticks out 0.4 m into the warm
# (internal) room:
#
# | part | extent / m | conductivity / (W/(m K)) |
# |---|---|---|
# | insulation layer | $0\le x\le1,\ 0\le y\le0.2,\ 0\le z\le1$ | 0.1 |
# | iron bar | $0.45\le x\le0.55,\ 0\le y\le0.6,\ 0.475\le z\le0.525$ | 50 |
#
# The temperature solves the steady heat equation $\nabla\cdot(k\nabla T)=0$. Both sides
# exchange heat with the air through a surface resistance $R_s=0.10$ m² K/W
# ($h=1/R_s=10$ W/(m² K)); the internal air is at 1 °C, the external air at 0 °C. Boundary
# conditions on every face:
#
# | face | condition |
# |---|---|
# | external face $y=0$ (insulation and bar end) | $-k\,\partial_nT=h\,(T-0\,^\circ\mathrm{C})$ |
# | internal face $y=0.2$ of the insulation and all faces of the protruding bar | $-k\,\partial_nT=h\,(T-1\,^\circ\mathrm{C})$ |
# | the four cut edges of the layer ($x=0,1$; $z=0,1$) | adiabatic (zero flux) |
#
# The field is genuinely three-dimensional: it depends on the depth $y$ and on both
# distances from the bar. The problem is symmetric about the planes $x=0.5$ and $z=0.5$,
# so one quarter ($x\ge0.5$, $z\ge0.5$) is computed with zero-flux symmetry planes and
# mirrored for the figures; heat flows are multiplied by four.
#
# **Numerical methods (both on the same meshes, same OneSimulation problem).**
# - **Cell-centred finite volumes**: one temperature per hexahedron, two-point diffusive
#   face fluxes (`CentralDifference`) with least-squares cell gradients
#   (`LeastSquaresGradient`) for the non-orthogonal correction, Robin faces closed by the
#   exact face balance. On faces between bar and insulation the face conductivity is the
#   harmonic mean, the classical series-resistance rule of finite volumes (S. V. Patankar,
#   *Numerical Heat Transfer and Fluid Flow*, Hemisphere 1980, Sec. 4.2-3); the mesh has
#   equal cell widths on both sides of every material interface, so the harmonic mean is
#   the exact series resistance of the two half cells.
# - **Quadratic finite elements**: continuous Galerkin with 27-node tensor-product
#   Lagrange elements (Q2) on the same hexahedra.
# - Linear solver: sparse direct LU in both cases (the problem is linear and steady).
#
# **Reference.** The test reference values of EN ISO 10211:2007, Annex A (2017 edition:
# Annex C), case 4: heat flow $\Phi=0.540$ W between the internal and the external
# environment and the highest external surface temperature $0.805\,^\circ$C, to be met
# within 1 % (heat flow) and 0.005 K (temperature). Supporting checks: the one-dimensional
# solution far from the bar, the energy balance (heat in = heat out) and three-level mesh
# studies with observed order and Richardson estimate for both methods.

from dataclasses import replace
import time
import numpy as np

from onesim.cad.d3 import Box, Regrouped
from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, Piecewise, StructuredHexahedralMesher, region_mesh
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.fvm import CentralDifference, FiniteVolume, LeastSquaresGradient
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.results.comparisons import convergence
from onesim.cad.d2 import Plane
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

k_ins, k_iron = 0.1, 50.0
h_s = 1.0 / 0.10
T_int, T_ext = 274.15, 273.15
x_bar, z_bar = 0.55, 0.525
t_ins, y_tip = 0.2, 0.6
Phi_ref, Phi_tol = 0.540, 0.01
T_ref, T_tol = 0.805, 0.005
U_1d = 1.0 / (0.10 + t_ins / k_ins + 0.10)
print(f"1D far field: U = {U_1d:.5f} W/(m^2 K), heat flux {U_1d * 1.0:.5f} W/m^2, "
      f"external surface temperature {U_1d * 0.10:.5f} degC")


# ## Geometry and mesh
#
# The quarter model ($x\ge0.5$, $z\ge0.5$) is the insulation slab joined with the bar
# (`|`); the joined faces keep the names of the part they lie on (`bar.` for the bar)
# and are grouped into the four boundary kinds of the standard. The mesh is a graded
# tensor-product block mesh of the bounding box with a node plane on every material
# interface, restricted to the solid (`region_mesh`): geometric grading away from the bar
# surfaces $x=0.55$ and $z=0.525$ (same first width on both sides), towards both faces
# of the insulation in $y$ and along the protruding bar. A mesh level $\ell$ scales the
# cell counts by $\ell$ and the smallest cell width ($8$ mm$/\ell$) by $1/\ell$.

slab = Box(length=0.5, width=t_ins, height=0.5, origin=(0.5, 0.0, 0.5))
bar = Box(length=x_bar - 0.5, width=y_tip, height=z_bar - 0.5, origin=(0.5, 0.0, 0.5))
construction = Regrouped(source=slab | bar.named("bar"), groups={
    "external": ("y_min",),
    "internal": ("y_max", "bar.x_max", "bar.z_max", "bar.y_max"),
    "symmetry": ("x_min", "z_min", "bar.x_min", "bar.z_min"),
    "layer_edges": ("x_max", "z_max"),
})
bounding_box = Box(length=0.5, width=y_tip, height=0.5, origin=(0.5, 0.0, 0.5))


def mesh_for(level):
    cells = lambda base: int(round(base * level))
    first = 0.008 / level
    grading = (
        Piecewise(((x_bar - 0.5, Geometric(cells(2), first=first, toward="end")),
                   (1.0 - x_bar, Geometric(cells(5), first=first)))),
        Piecewise(((t_ins / 2, Geometric(cells(3), first=first)),
                   (t_ins / 2, Geometric(cells(3), first=first, toward="end")),
                   (y_tip - t_ins, Geometric(cells(3), first=first)))),
        Piecewise(((z_bar - 0.5, Geometric(cells(1), first=first, toward="end")),
                   (1.0 - z_bar, Geometric(cells(5), first=first)))),
    )
    counts = tuple(g.cells for g in grading)
    block = StructuredHexahedralMesher(cells=counts, grading=grading).generate(bounding_box)
    return region_mesh(block, construction)


# ## Model
#
# One `HeatConduction` problem serves both methods. The conductivity is region-wise:
# 50 W/(m K) in the bar and 0.1 W/(m K) in the insulation. On the bar-insulation interface,
# where only the finite-volume faces sample it, `interfaces="harmonic"` returns the
# harmonic mean (the series resistance of the two equal half cells); the quadrature
# points of the finite elements always lie inside a cell and see the plain values.

conductivity = regionwise(((bar, k_iron),), default=k_ins, unit="W/m/K", interfaces="harmonic")
material = replace(IsotropicSolid(density=500.0, specific_heat_capacity=1700.0, thermal_conductivity=k_ins,
                                  youngs_modulus=1e9, poisson_ratio=0.3, thermal_expansion_coefficient=0.0),
                   thermal_conductivity=conductivity)

METHODS = {
    "FV": FiniteVolume(gradient=LeastSquaresGradient(), diffusion=CentralDifference()),
    "Q2": FiniteElement(element=Lagrange(order=2)),
}
b = construction.boundaries


def solve(level, method):
    start = time.time()
    mesh = mesh_for(level)
    model = HeatConduction(domain=construction.domain(), material=material, heating=0.0)
    problem = Problem(name="ISO 10211 case 4 thermal bridge", geometry=construction, physics=(model,),
                      boundary_conditions=(
                          ConvectiveHeatFlux(boundary=b.internal, physics=model, heat_transfer_coefficient=h_s,
                                             ambient_temperature=T_int),
                          ConvectiveHeatFlux(boundary=b.external, physics=model, heat_transfer_coefficient=h_s,
                                             ambient_temperature=T_ext),
                          HeatFlux(boundary=b.symmetry | b.layer_edges, physics=model, value=0.0)))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=METHODS[method], linear_solver=SparseDirect())).solve_stationary()
    field = result.field(model.temperature)
    external = field.sample(along=b.external, subdivisions=4)
    return dict(
        level=level, method=method, mesh=mesh, field=field,
        unknowns=result.operators[model.model_id].problem_size,
        Phi_out=4 * field.boundary_flux(b.external), Phi_in=-4 * field.boundary_flux(b.internal),
        T_max=float(np.max(external.values)) - 273.15,
        T_far=float(field.evaluate([[1.0, 0.0, 1.0]])[0]) - 273.15,
        seconds=time.time() - start,
    )


# ## Solve
#
# Three levels per method. The finite-volume levels $\ell=2,3,4$ and the finite-element
# levels $\ell=1,2,3$ use the same mesh family.

runs = {"FV": [], "Q2": []}
for method, levels in (("FV", (2, 3, 4)), ("Q2", (1, 2, 3))):
    for level in levels:
        out = solve(level, method)
        runs[method].append(out)
        print(f"{method} level {level}: {out['mesh'].cell_count:6d} cells, {out['unknowns']:6d} unknowns, "
              f"Phi = {out['Phi_out']:.5f} W, T_max,ext = {out['T_max']:.5f} degC  ({out['seconds']:.0f} s)")


# ## Comparison with the reference

print("EN ISO 10211 case 4, finest mesh of each method")
print("   quantity                       reference      FV (cells)     Q2 (elements)")
fv, fe = runs["FV"][-1], runs["Q2"][-1]
print(f"   heat flow Phi / W              {Phi_ref:9.3f}    {fv['Phi_out']:9.4f}      {fe['Phi_out']:9.4f}")
print(f"     deviation (tolerance 1 %)                 {100 * (fv['Phi_out'] / Phi_ref - 1):+8.2f} %     "
      f"{100 * (fe['Phi_out'] / Phi_ref - 1):+8.2f} %")
print(f"   max external T / degC          {T_ref:9.3f}    {fv['T_max']:9.4f}      {fe['T_max']:9.4f}")
print(f"     deviation (tolerance 0.005 K)              {fv['T_max'] - T_ref:+8.4f} K     {fe['T_max'] - T_ref:+8.4f} K")

print("Mesh studies (cell size ~ 1/level)")
for method, levels in (("FV", (2, 3, 4)), ("Q2", (1, 2, 3))):
    for key, label in (("Phi_out", "Phi / W"), ("T_max", "T_max / degC")):
        values = [r[key] for r in runs[method]]
        study = convergence(values, [1.0 / level for level in levels])
        order = "-" if study.order is None else f"{study.order:.2f}"
        extra = study.reason if study.extrapolated is None else f"{study.extrapolated:.5f}"
        print(f"   {method} {label:13s} " + "  ".join(f"{v:.5f}" for v in values)
              + f"   order {order}, Richardson {extra}, change last two meshes {abs(values[-1] - values[-2]):.1e}")

print("Energy balance (heat entering on the warm side = heat leaving on the cold side)")
for method in runs:
    r = runs[method][-1]
    print(f"   {method}: in {r['Phi_in']:.6f} W, out {r['Phi_out']:.6f} W, relative difference "
          f"{abs(r['Phi_in'] - r['Phi_out']) / r['Phi_out']:.1e}")
print(f"1D far field at the layer corner (x = z = 1): external surface temperature, closed form "
      f"{U_1d * 0.10:.5f} degC")
for method in runs:
    print(f"   {method}: {runs[method][-1]['T_far']:.5f} degC")
print(f"Thermal bridge effect: Phi - U A = {fe['Phi_out'] - U_1d * 1.0:.4f} W "
      f"(point thermal transmittance chi = {(fe['Phi_out'] - U_1d) / 1.0:.4f} W/K)")


# ## Figure
#
# Left: surface temperature of the whole construction (the quarter mirrored about both
# symmetry planes), seen from the warm side with the protruding bar and from the cold side
# with the warm spot where the bar ends. Middle: the section $z=0.5$ through the bar axis.
# Right: the external surface around the bar end, and the mesh convergence of both methods.

fine = runs["Q2"][-1]["field"]
celsius = lambda values, positions: values - 273.15
mirror = {"planes": [((1, 0, 0), (0.5, 0, 0)), ((0, 0, 1), (0, 0, 0.5))]}
cells = {method: [run["mesh"].cell_count for run in runs[method]] for method in ("FV", "Q2")}
fig = plots.figure(layout=[["warm", "cut", "outside"], ["cold", "cut", "convergence"]], size=(17, 10))
plots.surface(fine, component=celsius, symmetry=mirror, camera=plots.Camera(elevation=18, azimuth=40), limits=(0.0, 1.0),
              label="T [°C]", title="Warm side: surface temperature", ax=fig["warm"])
plots.surface(fine, component=celsius, symmetry=mirror, camera=plots.Camera(elevation=22, azimuth=230), limits=(0.0, 1.0),
              label="T [°C]", title="Cold side: surface temperature", ax=fig["cold"])
plots.section(fine, plane=Plane(origin=(0.5, 0.0, 0.5), normal=(0, 0, 1)), component=celsius,
              symmetry={"planes": [((1, 0, 0), (0.5, 0, 0))]}, limits=(0.0, 1.0), contours=19, label="T [°C]",
              title="Section z = 0.5 through the bar axis (Q2)", xlimits=(0.15, 0.85), ylimits=(0, 0.62),
              ylabel="y [m] (cold face at y = 0)", ax=fig["cut"])
plots.surface(fine, boundary=b.external, component=celsius, symmetry=mirror, contours=12, label="T [°C]", subdivisions=4,
              title=f"External surface around the bar end (max {runs['Q2'][-1]['T_max']:.3f} °C)",
              xlimits=(0.25, 0.75), ylimits=(0.35, 0.65), ax=fig["outside"])
plots.line(y={f"{method}: heat flow": (cells[method], [100 * (run["Phi_out"] / Phi_ref - 1) for run in runs[method]])
              for method in cells},
           right={f"{method}: max external T": (cells[method], [1e3 * (run["T_max"] - T_ref) for run in runs[method]])
                  for method in cells},
           hlines={"heat-flow tolerance ±1 %": 100 * Phi_tol, "": -100 * Phi_tol}, markers=True, scale="logx",
           xlabel="hexahedral cells (quarter model)", ylabel="heat flow: deviation from 0.540 W [%]",
           right_label="max external T: deviation from 0.805 °C [mK]", title="Mesh convergence against the ISO values",
           legend="lower right", ax=fig["convergence"])
plots.show()


