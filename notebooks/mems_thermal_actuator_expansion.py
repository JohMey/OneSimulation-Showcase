"""OneSimulation example: thermal expansion of a Joule-heated copper-beryllium MEMS meander.
Requires the installed, activated OneSimulation runtime.

A 3D serpentine microstructure of the copper-beryllium alloy UNS C17500 is heated volumetrically,
cooled by convection and clamped to a circuit board. The coupled thermal-structural problem is
solved with quadratic finite elements and temperature-dependent material laws, and checked against
the published temperature, the exact heat balance, the exact free-expansion solution and a
three-level mesh study of the maximum displacement.
"""


# # Thermal expansion in a MEMS device
#
# Microgyroscopes and similar MEMS devices must keep their geometry stable while they are
# heated by the drive current. The device studied here is a meander of four hairpins etched
# from a 10 um thick layer of the copper-beryllium alloy UNS C17500: legs 10 um wide and
# 80 um tall, half-ring arches with radii 10 and 20 um, and short bars that join the hairpins
# at the bottom. The bottom face ($y=0$) is bonded to a circuit board. The device spans
# $190\times100\times10$ um, and all fields depend on all three coordinates.
#
# Loads, as in the original problem set-up:
#
# | face | thermal condition | mechanical condition |
# |---|---|---|
# | bonded bottom face $y=0$ | thermally insulated | clamped to the board (three variants below) |
# | front and back faces $z=0$, $z=10$ um | convection, $h=10$ W/(m² K), $T_\infty=298$ K | traction-free |
# | all side walls (legs, arches, bars) | convection, $h=10$ W/(m² K), $T_\infty=298$ K | traction-free |
#
# The Joule heating is a uniform volumetric source $Q=10^8$ W/m³. The thermal strain is
# $\varepsilon_\mathrm{th}=\alpha(T)\,(T-T_\mathrm{ref})\,\mathbf I$ with the secant expansion
# coefficient $\alpha(T)$ and the strain-free temperature $T_\mathrm{ref}=298$ K.
#
# **Board variants.** (1) *Fixed board*: the bonded face cannot move. (2) *Expanding FR-4 board*:
# the bonded face follows the free thermal expansion of a glass-epoxy board at the local device
# temperature, $\mathbf u=\alpha_\mathrm{FR4}(T-T_\mathrm{ref})\,(x,0,z)$. (3) *Matched board*:
# the board expands exactly like the alloy itself.
#
# **Material data.** UNS C17500 (Cu-2.5Co-0.6Be): $k=208$ W/(m K) and $E=138$ GPa at 20 °C,
# mean expansion coefficient $17.6\times10^{-6}$ 1/K over 20-200 °C, $\rho=8830$ kg/m³,
# $c_p=420$ J/(kg K) (Copper Development Association, *Standard Designations for Wrought and
# Cast Copper Alloys*, alloy C17500; ASTM B441), and $\nu=0.30$ (Materion, *Guide to Copper
# Beryllium*). These values are made temperature-dependent with linear laws about 293.15 K.
# The slopes are typical of copper alloys. $E$ drops by $3.5\times10^{-4}$ per K (Ledbetter
# and Naimon, *J. Phys. Chem. Ref. Data* 3, 897, 1974, for copper), and $k$ of the alloy rises
# by $2\times10^{-4}$ per K. The tangent expansion coefficient of copper rises by about
# $5\times10^{-4}$ per K, so the 20-200 °C mean of $17.6\times10^{-6}$ corresponds to
# $16.8\times10^{-6}$ 1/K at 20 °C. The secant coefficient used in the thermal-strain law
# rises with half that slope. The tabulated library functions behind the original model are
# not public, so these laws stand in for them. Over the 26 K temperature rise they change $E$
# by 0.9 % and $\alpha$ by 0.7 %. FR-4 laminate:
# in-plane $\alpha_\mathrm{FR4}=18\times10^{-6}$ 1/K, the upper end of the IPC-4101 range.
#
# **Numerical method.** Spatial: continuous Galerkin finite elements, quadratic (Q2) Lagrange
# hexahedra for both the temperature and the displacement, on a block-structured mesh with
# exactly curved arches. Coupling: one-way staggered. The heat problem is solved first, and its Q2 temperature
# field is passed to the elastic solid by finite-element evaluation at the solid's quadrature
# points (`FieldCoupling` + `IsotropicThermalExpansion`). The displacement does not feed back
# into the heat problem, so one pass is the exact coupled solution. The temperature-dependent
# conductivity makes the heat problem nonlinear, and it is solved with Newton's method
# (sparse direct LU inner solves). Each elastic solve is linear (sparse direct LU). No time
# integration is needed: the problem is stationary.
#
# **Reference.** (i) The published result of the original set-up: the device reaches about
# 323.67 K and is isothermal to better than 0.01 K. (ii) The exact global heat balance
# $\int h(T-T_\infty)\,\mathrm dA=QV$, and the lumped-body limit $T=T_\infty+QV/(hA)$, which is
# exact as the Biot number $hL/k\approx5\times10^{-7}$ goes to zero. (iii) The exact 3D solution for the
# matched board, $\mathbf u=\alpha(T)(T-T_\mathrm{ref})\,(x,y,z)$, with zero stress (uniform
# free expansion). (iv) A three-level mesh study (observed order and Richardson estimate) of
# the maximum displacement and of the arch-top displacements for the fixed board. (v) The published qualitative result that an
# expanding board lowers the peak stress by about an order of magnitude.

import time
import numpy as np

from onesim.materials import IsotropicSolid, PolynomialLaw
from onesim.meshing import Block, arc, extrude, line, merge_blocks, rectangle_block
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.field_transfer import FiniteElementEvaluation
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.physics.solid_mechanics import (
    FixedDisplacement, IsotropicThermalExpansion, LinearElasticity, TractionFree,
)
from onesim.results.comparisons import convergence
from onesim.results.integration import measure
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_connections import FieldCoupling
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

um = 1e-6
leg_width = 10 * um
leg_height = 80 * um
arch_inner, arch_outer = 10 * um, 20 * um
depth = 10 * um
pitch = 50 * um
n_hairpins, x_first = 4, -100 * um

Q = 1e8
h_conv, T_amb = 10.0, 298.0
T_ref = 298.0
alpha_fr4 = 18e-6
T_published = 323.67


def linear_in_temperature(value_293, slope, unit):
    """Property law p(T) = p0 (1 + s (T - 293.15 K))."""
    return PolynomialLaw((value_293, value_293 * slope), unit=unit, reference=293.15)


conductivity = linear_in_temperature(208.0, 2.0e-4, "W/m/K")
youngs_modulus = linear_in_temperature(138e9, -3.5e-4, "Pa")
expansion = linear_in_temperature(16.8e-6, 2.5e-4, "1/K")
poisson = 0.30


def alloy(at_temperature=None):
    """C17500 with temperature-dependent laws, or with the laws evaluated at one temperature."""
    if at_temperature is None:
        k, E, a = conductivity, youngs_modulus, expansion
    else:
        k, E, a = (float(law.evaluate(temperature=at_temperature)) for law in (conductivity, youngs_modulus, expansion))
    return IsotropicSolid(density=8830.0, specific_heat_capacity=420.0, thermal_conductivity=k,
                          youngs_modulus=E, poisson_ratio=poisson, thermal_expansion_coefficient=a)


area_profile = (2 * n_hairpins * leg_width * leg_height + (n_hairpins - 1) * leg_width**2
                + n_hairpins * 0.5 * np.pi * (arch_outer**2 - arch_inner**2))
perimeter = (2 * n_hairpins * 2 * (leg_width + leg_height) + (n_hairpins - 1) * 4 * leg_width
             + n_hairpins * (np.pi * (arch_outer + arch_inner) + 2 * leg_width)
             - 2 * (2 * n_hairpins + 2 * (n_hairpins - 1)) * leg_width)
bonded_length = (2 * n_hairpins + n_hairpins - 1) * leg_width
volume_exact = area_profile * depth
cooled_exact = 2 * area_profile + (perimeter - bonded_length) * depth
T_lumped = T_amb + Q * volume_exact / (h_conv * cooled_exact)
print(f"profile area {area_profile / um**2:.2f} um^2, volume {volume_exact / um**3:.1f} um^3, "
      f"cooled surface {cooled_exact / um**2:.1f} um^2")
print(f"lumped-body temperature T_inf + QV/(hA) = {T_lumped:.4f} K   (published: {T_published} K)")
print(f"Biot number h*leg_width/k = {h_conv * leg_width / 208.0:.1e}")


# ## Geometry and mesh
#
# The profile is covered by quadrilateral blocks: each leg is split into a bottom block and
# an upper block, so the bottom bars join it conformally, and each arch is one block between
# two circular arcs. A refinement level $n$ puts $n$ cells across every 10 um width, $6n$
# cells along every arch and $n$ cells through the thickness. `merge_blocks` builds a
# quadratic (9-node) profile mesh whose arch nodes lie on the exact circles, and `extrude`
# sweeps it through the thickness into 27-node hexahedra. The bonded bottom sides are named
# `base`, all other profile edges become the side walls, and the extrusion names its two
# caps `back` ($z=0$) and `front` ($z=10$ um).

def build_mesh(n):
    w = leg_width
    blocks = []
    for i in range(n_hairpins):
        left = x_first + i * pitch
        for x0 in (left, left + 3 * w):
            blocks.append(rectangle_block((x0, 0.0), w, w, (n, n), names=("base", None, None, None)))
            blocks.append(rectangle_block((x0, w), w, leg_height - w, (n, 7 * n)))
        cx = left + 2 * w
        blocks.append(Block(arc((cx, leg_height), arch_inner, np.pi, 0.0),
                            line((cx + arch_inner, leg_height), (cx + arch_outer, leg_height)),
                            arc((cx, leg_height), arch_outer, np.pi, 0.0),
                            line((cx - arch_inner, leg_height), (cx - arch_outer, leg_height)),
                            cells=(6 * n, n)))
    for j in range(n_hairpins - 1):
        blocks.append(rectangle_block((x_first + 4 * w + j * pitch, 0.0), w, w, (n, n),
                                      names=("base", None, None, None)))
    profile = merge_blocks(blocks, order=2, unnamed="side_walls")
    return extrude(profile, depth, n, names=("back", "front"))


# ## Model
#
# The heat model carries the uniform Joule source, convection on every cooled face and an
# explicit zero flux on the bonded face. The solid is coupled to the solved temperature
# field. Boundary conditions are listed per board variant in the introduction. The heat
# leaving by convection is the convective law integrated with the solved surface
# temperature, $h\,(\int T\,dA - T_\infty A)$.

def cooled(b):
    return b.front | b.back | b.side_walls


def solve_heat(mesh, material, nonlinear):
    geometry = mesh.geometry
    b = geometry.boundaries
    heat = HeatConduction(domain=geometry.domain(), material=material, heating=Q)
    problem = Problem(name="Joule-heated meander", geometry=geometry, physics=(heat,), boundary_conditions=(
        ConvectiveHeatFlux(boundary=cooled(b), physics=heat, heat_transfer_coefficient=h_conv, ambient_temperature=T_amb),
        HeatFlux(boundary=b.base, physics=heat, value=0.0)))
    if nonlinear:
        numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)),
                            nonlinear_solver=Newton(linear_solver=SparseDirect(), relative_tolerance=0.0,
                                                    absolute_tolerance=1e-9, maximum_iterations=20),
                            initial_guess={heat.temperature: T_lumped})
    else:
        numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())
    field = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(heat.temperature)
    nodal = np.asarray(field.evaluate(mesh.coordinates)).reshape(-1)
    cooled_area = measure(field, boundary=cooled(b))
    heat_out = h_conv * (float(field.integrate(boundary=cooled(b)).value) - T_amb * cooled_area)
    volume = measure(field)
    return dict(field=field, nodal=nodal, T_max=float(nodal.max()), T_min=float(nodal.min()),
                heat_out=heat_out, heat_in=Q * volume, volume=volume, cooled_area=cooled_area)


def board_displacement(temperature_field, coefficient):
    """Bonded-face displacement of a board that expands freely at the local device temperature."""
    def value(x, y, z):
        points = np.column_stack((np.ravel(x), np.ravel(y), np.ravel(z)))
        T = np.asarray(temperature_field.evaluate(points)).reshape(-1)
        strain = np.asarray(coefficient(T)) * (T - T_ref)
        return np.stack((strain * points[:, 0], 0.0 * strain, strain * points[:, 2]), axis=-1).reshape(np.shape(x) + (3,))
    return value


arch_tops = np.array([[x_first + 2 * leg_width + i * pitch, leg_height + arch_outer, depth / 2]
                      for i in range(n_hairpins)])


def solve_solid(mesh, material, temperature_field, board="fixed"):
    geometry = mesh.geometry
    b = geometry.boundaries
    solid = LinearElasticity(domain=geometry.domain(), material=material,
                             thermal_strain=IsotropicThermalExpansion(reference_temperature=T_ref))
    if board == "fixed":
        support = (0.0, 0.0, 0.0)
    elif board == "FR-4":
        support = board_displacement(temperature_field, lambda T: alpha_fr4 + 0.0 * T)
    else:
        support = board_displacement(temperature_field, lambda T: expansion.evaluate(temperature=T))
    problem = Problem(name=f"Meander on a {board} board", geometry=geometry, physics=(solid,), boundary_conditions=(
        FixedDisplacement(boundary=b.base, physics=solid, value=support),
        TractionFree(boundary=cooled(b), physics=solid)),
        couplings=(FieldCoupling(source=temperature_field, target=solid.temperature,
                                 transfer=FiniteElementEvaluation()),))
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect())).solve_stationary()
    u = result.field(solid.displacement)
    nodal = np.asarray(u.evaluate(mesh.coordinates))
    magnitude = np.linalg.norm(nodal, axis=1)
    top = np.asarray(u.evaluate(arch_tops)).reshape(-1, 3)
    return dict(u=u, stress=result.field(solid.von_mises_stress), nodal=nodal, u_max=float(magnitude.max()),
                where_max=np.asarray(mesh.coordinates)[int(magnitude.argmax())], top=top)


# ## Solve
#
# **Temperature-dependent laws evaluated pointwise.** On the coarsest mesh ($n=1$) the
# complete model is solved with the laws evaluated at every quadrature point. Newton's method
# handles $k(T)$, and $E(T)$ and $\alpha(T)$ are evaluated at the coupled temperature. The
# same mesh is then solved again with the laws evaluated once, at the computed mean device
# temperature. The device is isothermal to within millikelvin, so the two must agree. The
# pointwise path is about ten times more expensive per assembly, so the mesh study below uses
# the laws evaluated at each mesh's own computed mean temperature.

start = time.perf_counter()
coarse = build_mesh(1)
heat_pointwise = solve_heat(coarse, alloy(), nonlinear=True)
solid_pointwise = solve_solid(coarse, alloy(), heat_pointwise["field"])
T_mean_coarse = float(heat_pointwise["nodal"].mean())
heat_frozen = solve_heat(coarse, alloy(T_mean_coarse), nonlinear=False)
solid_frozen = solve_solid(coarse, alloy(T_mean_coarse), heat_frozen["field"])
print(f"n = 1 ({coarse.cell_count} hexahedra): pointwise laws vs laws at T_mean = {T_mean_coarse:.4f} K:")
print(f"  T_max {heat_pointwise['T_max']:.6f} K vs {heat_frozen['T_max']:.6f} K;  "
      f"u_max {solid_pointwise['u_max'] / um * 1e3:.5f} nm vs {solid_frozen['u_max'] / um * 1e3:.5f} nm "
      f"(relative difference {solid_frozen['u_max'] / solid_pointwise['u_max'] - 1:+.1e})  [{time.perf_counter() - start:.0f} s]")

# **Mesh study, fixed board.** Levels $n=2,3,4$ (cells across every 10 um width and through
# the thickness).

levels = (2, 3, 4)
runs = []
for n in levels:
    start = time.perf_counter()
    mesh = build_mesh(n)
    heat = solve_heat(mesh, alloy(T_mean_coarse), nonlinear=False)
    T_mean = float(heat["nodal"].mean())
    material = alloy(T_mean)
    fixed = solve_solid(mesh, material, heat["field"], "fixed")
    runs.append(dict(n=n, mesh=mesh, heat=heat, T_mean=T_mean, material=material, fixed=fixed))
    print(f"n = {n}: {mesh.cell_count:6d} hexahedra, {len(mesh.coordinates):6d} nodes, "
          f"T in [{heat['T_min']:.5f}, {heat['T_max']:.5f}] K, u_max = {fixed['u_max'] / um * 1e3:.4f} nm  "
          f"[{time.perf_counter() - start:.0f} s]")
final = runs[-1]

# **Board variants** on the medium mesh: the expanding FR-4 board and the matched board.

start = time.perf_counter()
medium = runs[1]
fr4 = solve_solid(medium["mesh"], medium["material"], medium["heat"]["field"], "FR-4")
matched = solve_solid(medium["mesh"], medium["material"], medium["heat"]["field"], "matched")
print(f"board variants on n = {medium['n']}: [{time.perf_counter() - start:.0f} s]")


# ## Comparison with the reference
#
# **Thermal part.** The published device temperature, the lumped-body limit for the exact
# geometry, and the global heat balance of the computed field. The quadratic cells follow the
# arches closely, so the lumped limit of the meshed geometry agrees with the exact one.

print("\nTemperature and heat balance (fixed-board meshes)")
print(" level   T_min [K]     T_max [K]     spread [mK]   lumped, meshed geometry [K]   heat in [uW]   imbalance")
for run in runs:
    heat = run["heat"]
    lumped_mesh = T_amb + heat["heat_in"] / (h_conv * heat["cooled_area"])
    print(f"   {run['n']}    {heat['T_min']:.5f}   {heat['T_max']:.5f}   {1e3 * (heat['T_max'] - heat['T_min']):9.3f}"
          f"     {lumped_mesh:.5f}                    {1e6 * heat['heat_in']:.5f}      "
          f"{abs(heat['heat_out'] - heat['heat_in']) / heat['heat_in']:.1e}")
T_final = final["heat"]["T_max"]
print(f"T_max, finest mesh:         {T_final:.4f} K")
print(f"published device temperature: {T_published:.2f} K   deviation {T_final - T_published:+.4f} K "
      f"({100 * (T_final - T_amb) / (T_published - T_amb) - 100:+.3f} % of the temperature rise)")
print(f"lumped limit, exact geometry: {T_lumped:.4f} K   deviation {T_final - T_lumped:+.4f} K "
      f"({100 * (T_final - T_amb) / (T_lumped - T_amb) - 100:+.3f} % of the temperature rise)")

# **Matched board: exact free expansion.** With the board expanding like the alloy, the
# exact solution is $\mathbf u=\alpha(T)(T-T_\mathrm{ref})\,(x,y,z)$ with zero stress. The
# residual temperature spread of a few millikelvin is the only source of a deviation.

coords = np.asarray(medium["mesh"].coordinates)
T_nodes = medium["heat"]["nodal"]
strain_free = expansion.evaluate(temperature=T_nodes) * (T_nodes - T_ref)
u_exact = strain_free[:, None] * coords
scale_stress = float(youngs_modulus.evaluate(temperature=medium["T_mean"]) * expansion.evaluate(temperature=medium["T_mean"])
                     * (medium["T_mean"] - T_ref))
vm_matched = np.asarray(matched["stress"].evaluate(coords)).reshape(-1)
print("\nMatched board (exact uniform free expansion), medium mesh")
print(f"  max |u - u_exact| / max |u_exact|  = {np.abs(matched['nodal'] - u_exact).max() / np.abs(u_exact).max():.2e}")
print(f"  max von Mises / (E alpha dT)       = {vm_matched.max() / scale_stress:.2e}   (exact: 0)")
print(f"  top of the arches: u_y = {matched['nodal'][:, 1].max() / um * 1e3:.4f} nm, exact "
      f"{(strain_free * coords[:, 1]).max() / um * 1e3:.4f} nm")

# **Mesh study, fixed board.** Mesh size $\propto 1/n$. The edges where the clamped face
# meets the free side walls are singular: the elastic field of a clamped edge varies like
# $\rho^\lambda$ with $\lambda<1$ in the distance $\rho$ from the edge (Williams, *J. Appl.
# Mech.* 19, 526, 1952). This singularity limits the global convergence order of the
# displacement to about one, whatever the element order. The vertical expansion at the arch
# tops converges monotonically. The maximum displacement also contains the small sideways sway
# of the outer hairpins, which comes from the rotation at the clamped feet. Its successive
# changes are a few hundredths of a percent and not yet monotone, so no order is fitted for it.

study = {
    "max |u| over the device": [run["fixed"]["u_max"] for run in runs],
    "u_y at the top of arch 1": [run["fixed"]["top"][0, 1] for run in runs],
    "u_y at the top of arch 2": [run["fixed"]["top"][1, 1] for run in runs],
}
print("\nMesh study, fixed board (nm); h ~ 1/n")
print(f"  {'quantity':26s}" + "".join(f"{'n = ' + str(n):>12s}" for n in levels)
      + "   change last two   order   Richardson   finest vs Richardson")
for name, values in study.items():
    estimate = convergence(values, [1.0 / n for n in levels])
    p, extrapolated = estimate.order, estimate.extrapolated
    study[name] = (values, p, extrapolated)
    tail = (f"   {p:5.2f}   {extrapolated / um * 1e3:10.5f}   {100 * (values[2] / extrapolated - 1):+.3f} %"
            if p is not None else "     -   (successive changes not monotone: no order can be fitted)")
    print(f"  {name:26s}" + "".join(f"{v / um * 1e3:12.5f}" for v in values)
          + f"   {100 * abs(values[2] / values[1] - 1):12.3f} %" + tail)
print(f"  maximum displacement on the finest mesh at (x, y, z) = "
      f"({', '.join(f'{c / um:.1f}' for c in final['fixed']['where_max'])}) um")
uniform_rise = float(expansion.evaluate(temperature=final["T_mean"]) * (final["T_mean"] - T_ref)) * (leg_height + arch_outer)
print(f"  for scale: free vertical expansion of the 100 um tall device alpha dT H = {uniform_rise / um * 1e3:.4f} nm")

# **Expanding board.** The published result: including the board expansion lowers the
# maximum von Mises stress by about an order of magnitude. The peak sits at the re-entrant
# edges of the bonded face, where the elastic field of a clamped corner is singular, so the
# peak value depends on the mesh. The ratio is therefore also given at a regular point,
# half-way up the first leg on its centre line.

vm_fixed = np.asarray(medium["fixed"]["stress"].evaluate(coords)).reshape(-1)
vm_fr4 = np.asarray(fr4["stress"].evaluate(coords)).reshape(-1)
probe = np.array([[x_first + leg_width / 2, 20 * um, depth / 2]])
vm_probe = [float(np.asarray(s["stress"].evaluate(probe)).reshape(-1)[0]) for s in (medium["fixed"], fr4)]
print("\nFixed vs expanding FR-4 board, medium mesh")
print(f"  max von Mises: fixed {vm_fixed.max() / 1e6:.2f} MPa, FR-4 {vm_fr4.max() / 1e6:.3f} MPa, "
      f"ratio {vm_fixed.max() / vm_fr4.max():.1f}  (published: about one order of magnitude)")
print(f"  von Mises at y = 20 um on the first leg: fixed {vm_probe[0] / 1e6:.3f} MPa, FR-4 {vm_probe[1] / 1e6:.4f} MPa, "
      f"ratio {vm_probe[0] / vm_probe[1]:.1f}")
print(f"  u_max: fixed {medium['fixed']['u_max'] / um * 1e3:.4f} nm, FR-4 {fr4['u_max'] / um * 1e3:.4f} nm, "
      f"matched {matched['u_max'] / um * 1e3:.4f} nm")


# ## Figure
#
# Top left: surface temperature of the device (deviation from the mean, in mK). Top right:
# deformed device on the fixed board, displacement magnified 200 times, coloured by the
# displacement magnitude. Bottom left and middle: von Mises stress on the front face for the
# fixed and the expanding FR-4 board (same logarithmic colour scale). Right: mesh study of the
# maximum displacement and of the vertical displacement at the arch tops, plotted against
# the mesh size, with the Richardson estimates at zero mesh size.

mesh, heat, fixed = final["mesh"], final["heat"], final["fixed"]
b, b_medium = mesh.geometry.boundaries, medium["mesh"].geometry.boundaries
camera = plots.Camera(elevation=20, azimuth=-58, up="y")
T_mean_final = float(heat["nodal"].mean())
log_stress = lambda s, p: np.log10(np.maximum(s, 1e3))
stress_range = (np.log10(max(float(np.min(vm_fr4)), 1e3)), np.log10(vm_fixed.max()))
inverse = 1.0 / np.array(levels)
estimates = {name: f"{name}: Richardson estimate (order {p:.2f})" for name, (_, p, _) in study.items() if p is not None}
fig = plots.figure(layout=[["temperature", "displacement", "study"], ["fixed", "fr4", "study"]], size=(17, 10.5))
plots.surface(heat["field"], boundary=cooled(b), component=lambda T, p: 1e3 * (T - T_mean_final), camera=camera,
              label="T - T_mean [mK]", title=f"Temperature, mean {T_mean_final:.3f} K", ax=fig["temperature"])
plots.surface(fixed["u"], boundary=cooled(b), component=lambda u, p: np.linalg.norm(u, axis=1) / um * 1e3,
              deformation=fixed["u"], deformation_scale=200.0, camera=camera, label="|u| [nm]",
              title="Displacement on the fixed board (magnified 200x)", ax=fig["displacement"])
plots.surface(medium["fixed"]["stress"], boundary=b_medium.front, component=log_stress, subdivisions=3,
              limits=stress_range, label="log10(von Mises / Pa)", title="von Mises stress, fixed board (front face)",
              ax=fig["fixed"])
plots.surface(fr4["stress"], boundary=b_medium.front, component=log_stress, subdivisions=3, limits=stress_range,
              label="log10(von Mises / Pa)", title="von Mises stress, expanding FR-4 board (front face)", ax=fig["fr4"])
plots.line(inverse, {name: np.array(values) / um * 1e3 for name, (values, _, _) in study.items()},
           points={estimates[name]: ([0.0], [extrapolated / um * 1e3]) for name, (_, _, extrapolated) in study.items()
                   if name in estimates},
           colors={label: index for index, name in enumerate(study) for label in (name, estimates.get(name, name))},
           markers=True, xlimits=(-0.02, 0.55), xlabel="mesh size 1/n (n = cells across each 10 µm width)",
           ylabel="displacement [nm]", title="Mesh study, fixed board", ax=fig["study"])
plots.show()


# ## Going further
#
# Replace the uniform heat source by the Joule heat of an actual drive current (an electric
# conduction solve on the same mesh), add the board as a second, meshed body, or sweep the
# board's expansion coefficient to find the value that makes the bonded joint stress free.
