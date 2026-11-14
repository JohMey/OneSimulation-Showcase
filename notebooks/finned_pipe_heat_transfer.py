"""OneSimulation example: heat transfer through a copper pipe with annular disk fins.
Requires the installed, activated OneSimulation runtime.

A periodic element of a finned pipe (grooved bore, one disk fin) is solved as a genuine
3D conduction problem with convective water and air sides. The fin efficiency is compared
with the classical Bessel-function solution for annular fins (Gardner 1945), the total
heat rate with an energy balance and a thermal-resistance network, and both are followed
through a three-level mesh study.
"""


# # Heat transfer through a finned pipe
#
# Warm water flows through a copper pipe; air blows across the outside. To raise the
# air-side area the pipe carries thin annular **disk fins**, and the bore carries
# twelve axial **grooves**. Far from the pipe ends the arrangement repeats with the fin
# pitch $p$, so one periodic element (one fin, centred) represents the whole pipe. The
# twelve grooves add a rotational period of $30^\circ$ with mirror symmetry, so the
# element is a $15^\circ$ wedge bounded by two symmetry planes. The temperature
# depends on all three coordinates: along the axis (fin versus gap), along the radius
# (wall, fin) and around the circumference (groove versus land).
#
# OneSimulation solves the stationary heat equation in the copper,
#
# $$\nabla\cdot(k\nabla T)=0,$$
#
# with quadratic (P2) Lagrange finite elements on a graded tetrahedral mesh, and the
# following condition on **every** boundary face:
#
# | face | condition |
# |---|---|
# | grooved bore | forced convection to the water, $-k\,\partial_n T=h_w\,(T-T_w)$ |
# | fin faces, fin tip, bare pipe surface | forced convection to the air, $-k\,\partial_n T=h_a\,(T-T_a)$ |
# | the two axial end planes (mid-gap) | zero flux (periodicity and mirror symmetry of the element) |
# | the two wedge planes $\theta=0$, $\theta=15^\circ$ | zero flux (mirror symmetry of the grooves) |
#
# **Reference.** Three independent checks:
#
# 1. *Fin efficiency.* For a thin annular fin of thickness $t$ between radii $r_1$ and
#    $r_2$ with uniform coefficient $h$, the one-dimensional fin equation has the
#    Bessel-function solution of K. A. Gardner, "Efficiency of extended surface",
#    Trans. ASME 67 (1945) 621-631. With $m=\sqrt{2h/(kt)}$ and a convective tip,
#    $\theta(r)=A\,I_0(mr)+B\,K_0(mr)$ with $\theta(r_1)=\theta_b$ and
#    $k\theta'(r_2)+h\theta(r_2)=0$. The Biot number $ht/(2k)$ is about $10^{-4}$, so
#    the 1D fin theory is essentially exact for the fin itself; the 3D model adds the
#    real fin root on the pipe wall. We also print Gardner's adiabatic-tip formula with
#    the corrected radius $r_2+t/2$.
# 2. *Energy balance.* The heat taken from the water must equal the heat given to the air.
# 3. *Thermal-resistance network.* Water film, cylindrical wall and finned air side in
#    series, with the air side weighted by the overall surface efficiency
#    $\eta_o=1-\tfrac{A_f}{A}(1-\eta_f)$ from the Gardner solution (Incropera et al.,
#    *Fundamentals of Heat and Mass Transfer*, Sec. 3.6.5).
#
# A three-level mesh study with Richardson extrapolation shows that the reported values
# are converged on the final mesh.

import time
import numpy as np
from scipy.special import i0, i1, k0, k1

from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, mapped_mesh, merge_blocks, rectangle_block, revolve
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.results.comparisons import convergence
from onesim.results.integration import measure
from onesim.simulation import Problem, Simulation
from onesim.cad.d2 import Plane
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# A representative member of the parameterised disk-stacked pipe family: copper pipe
# with 20 mm bore and 2 mm wall, 0.6 mm disk fins of 70 mm diameter at 5 mm pitch, and
# twelve rounded axial grooves of 0.8 mm depth in the bore. Water at 30 °C flows at
# 0.5 m/s, air at 20 °C blows across the pipe. The water-side coefficient follows from
# the Dittus-Boelter correlation $Nu=0.023\,Re^{0.8}Pr^{0.4}$ (Dittus & Boelter, Univ.
# Calif. Publ. Eng. 2 (1930) 443); the air-side coefficient is a representative value
# for forced cross-flow over finned tubes at a few m/s.

mm = 1e-3
r_bore = 10.0 * mm
r_wall = 12.0 * mm
r_fin = 35.0 * mm
t_fin = 0.6 * mm
pitch = 5.0 * mm
n_grooves = 12
groove_depth = 0.8 * mm
wedge = np.pi / n_grooves

k_cu = 401.0
T_water, T_air = 303.15, 293.15
h_air = 80.0

nu_w, k_w, Pr_w, u_w = 0.801e-6, 0.615, 5.42, 0.5
D_h = 2 * r_bore
Re_w = u_w * D_h / nu_w
h_water = 0.023 * Re_w**0.8 * Pr_w**0.4 * k_w / D_h

m_fin = np.sqrt(2 * h_air / (k_cu * t_fin))
print(f"Re_water = {Re_w:.0f},  h_water = {h_water:.0f} W/(m^2 K),  h_air = {h_air:.0f} W/(m^2 K)")
print(f"fin Biot number h t/(2k) = {h_air * t_fin / (2 * k_cu):.1e},  m (r2 - r1) = {m_fin * (r_fin - r_wall):.3f}")


# ## Geometry and mesh
#
# One periodic element spans half a pitch on each side of a fin and half a groove
# period ($15^\circ$) around the axis. Its meridian section $(r, x)$ consists of four
# rectangular blocks (the wall left of, under and right of the fin, and the fin itself),
# graded towards the two re-entrant fin-root corners where the heat flux concentrates.
# `merge_blocks` joins them, the quadrilaterals are split into triangles, and `revolve`
# sweeps the section by $15^\circ$ about the pipe axis $x$ into tetrahedra. Finally
# `mapped_mesh` moves the wall nodes outward to cut the rounded grooves
# $r_\mathrm{in}(\theta)=r_b+d\,[\tfrac12(1+\cos 12\theta)]^2$, stretching the wall
# between the grooved bore and the unchanged outer radius. The block sides name the
# boundaries; the two ends of the revolution are the wedge (symmetry) planes.

def groove(theta):
    return 0.25 * (1.0 + np.cos(n_grooves * theta)) ** 2


def cut_grooves(points):
    """Stretch the wall radially between r_in(theta) and r_wall."""
    r, theta = np.hypot(points[:, 1], points[:, 2]), np.arctan2(points[:, 2], points[:, 1])
    r_in = r_bore + groove_depth * groove(theta)
    r_new = np.where(r <= r_wall + 1e-12, r_in + (r - r_bore) * (r_wall - r_in) / (r_wall - r_bore), r)
    points[:, 1], points[:, 2] = r_new * np.cos(theta), r_new * np.sin(theta)
    return points


def build_mesh(level):
    n_gap, n_thick, n_wall, n_radial = 3 * level, 2 * level, 3 * level, 10 * level
    gap_to_fin = Geometric(n_gap, growth=1.25 ** (2 / level), toward="end")
    gap_from_fin = Geometric(n_gap, growth=1.25 ** (2 / level), toward="start")
    wall = Geometric(n_wall, growth=1.15 ** (2 / level), toward="end")
    fin = Geometric(n_radial, growth=1.12 ** (2 / level), toward="start")
    wall_t, gap = r_wall - r_bore, pitch / 2 - t_fin / 2
    section = merge_blocks([
        rectangle_block((r_bore, -pitch / 2), wall_t, gap, (n_wall, n_gap), (wall, gap_to_fin),
                        names=("end_planes", "pipe_surface", None, "bore")),
        rectangle_block((r_bore, -t_fin / 2), wall_t, t_fin, (n_wall, n_thick), (wall, None),
                        names=(None, None, None, "bore")),
        rectangle_block((r_bore, t_fin / 2), wall_t, gap, (n_wall, n_gap), (wall, gap_from_fin),
                        names=(None, "pipe_surface", "end_planes", "bore")),
        rectangle_block((r_wall, -t_fin / 2), r_fin - r_wall, t_fin, (n_radial, n_thick), (fin, None),
                        names=("fin_surface", "fin_surface", "fin_surface", None)),
    ]).triangulated()
    element = revolve(section, 3 * level, angle=wedge, axis=(1.0, 0.0, 0.0), radial=(0.0, 1.0, 0.0),
                      names=("wedge_planes", "wedge_planes"))
    return mapped_mesh(element, cut_grooves)


# ## Model
#
# Copper is homogeneous and isotropic; only its conductivity enters the stationary
# problem. Both convective sides use OneSimulation's `ConvectiveHeatFlux` (Robin)
# condition; the four symmetry planes carry an explicit zero `HeatFlux`. The heat rates
# follow from the convective laws integrated over each surface with the solved surface
# temperature, $\int h\,(T-T_\infty)\,dA = h\,(\int T\,dA - T_\infty A)$.

copper = IsotropicSolid(density=8933.0, specific_heat_capacity=385.0, thermal_conductivity=k_cu,
                        youngs_modulus=110e9, poisson_ratio=0.34, thermal_expansion_coefficient=17e-6)


def solve(mesh):
    geometry = mesh.geometry
    model = HeatConduction(domain=geometry.domain(), material=copper, heating=0.0)
    b = geometry.boundaries
    problem = Problem(
        name="Finned pipe periodic element",
        geometry=geometry,
        physics=(model,),
        boundary_conditions=(
            ConvectiveHeatFlux(boundary=b.bore, physics=model,
                               heat_transfer_coefficient=h_water, ambient_temperature=T_water),
            ConvectiveHeatFlux(boundary=b.fin_surface | b.pipe_surface, physics=model,
                               heat_transfer_coefficient=h_air, ambient_temperature=T_air),
            HeatFlux(boundary=b.end_planes | b.wedge_planes, physics=model, value=0.0),
        ),
    )
    simulation = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()))
    temperature = simulation.solve_stationary().field(model.temperature)

    def convected(boundary, h, ambient):
        area = measure(temperature, boundary=boundary)
        return h * (float(temperature.integrate(boundary=boundary).value) - ambient * area), area

    q_water, a_bore = convected(b.bore, h_water, T_water)
    q_fin, a_fin = convected(b.fin_surface, h_air, T_air)
    q_pipe, a_pipe = convected(b.pipe_surface, h_air, T_air)
    q_water = -q_water
    xs = (np.arange(16) + 0.5) / 16 * t_fin - t_fin / 2
    ths = (np.arange(48) + 0.5) / 48 * wedge
    X, TH = np.meshgrid(xs, ths, indexing="ij")
    root = np.column_stack((X.ravel(), r_wall * np.cos(TH.ravel()), r_wall * np.sin(TH.ravel())))
    T_base = float(np.mean(temperature.evaluate(root)))
    nodal = np.asarray(temperature.evaluate(mesh.coordinates)).reshape(-1)
    return dict(temperature=temperature, b=b, nodal=nodal, q_water=q_water, q_fin=q_fin, q_pipe=q_pipe,
                a_bore=a_bore, a_fin=a_fin, a_pipe=a_pipe, T_base=T_base,
                eta=q_fin / (h_air * a_fin * (T_base - T_air)),
                T_max=float(nodal.max()), T_min=float(nodal.min()))


# ## Solve
#
# Three meshes; the cell counts in every direction grow in the ratio 2 : 3 : 4. The
# reported heat rates are per periodic element (one pitch, full circumference), i.e.
# 24 wedges.

levels = (2, 3, 4)
runs = []
for level in levels:
    start = time.time()
    mesh = build_mesh(level)
    out = solve(mesh)
    runs.append((mesh, out))
    print(f"level {level}: {mesh.cell_count:6d} tetrahedra, {len(mesh.coordinates):6d} vertices, "
          f"Q_water = {24 * out['q_water']:.5f} W, eta_fin = {out['eta']:.5f}, "
          f"T in [{out['T_min']:.3f}, {out['T_max']:.3f}] K  ({time.time() - start:.0f} s)")
mesh, out = runs[-1]


# ## Comparison with the reference
#
# **Fin efficiency.** The 1D annular-fin solution needs only $m$, $r_1$, $r_2$, $t$ and
# the tip condition; it is evaluated for exactly the fin of the 3D model.

def annular_fin_convective_tip(r1, r2, t, h, k):
    m = np.sqrt(2 * h / (k * t))
    M = np.array([[i0(m * r1), k0(m * r1)],
                  [k * m * i1(m * r2) + h * i0(m * r2), -k * m * k1(m * r2) + h * k0(m * r2)]])
    A, B = np.linalg.solve(M, [1.0, 0.0])
    base_flux = -k * m * (A * i1(m * r1) - B * k1(m * r1))
    area = 2 * np.pi * (r2**2 - r1**2) + 2 * np.pi * r2 * t
    profile = lambda r: A * i0(m * r) + B * k0(m * r)
    return base_flux * 2 * np.pi * r1 * t / (h * area), profile


def gardner_adiabatic_tip(r1, r2, t, h, k):
    m = np.sqrt(2 * h / (k * t))
    num = k1(m * r1) * i1(m * r2) - i1(m * r1) * k1(m * r2)
    den = i0(m * r1) * k1(m * r2) + k0(m * r1) * i1(m * r2)
    return 2 * r1 / (m * (r2**2 - r1**2)) * num / den


eta_exact, fin_profile = annular_fin_convective_tip(r_wall, r_fin, t_fin, h_air, k_cu)
eta_gardner = gardner_adiabatic_tip(r_wall, r_fin + t_fin / 2, t_fin, h_air, k_cu)


etas = [o["eta"] for _, o in runs]
sizes = [1.0 / level for level in levels]


def study_line(estimate):
    order = f"{estimate.order:.2f}" if estimate.order is not None else "n/a"
    extrapolated = f"{estimate.extrapolated:.6f}" if estimate.extrapolated is not None else "n/a"
    return order, extrapolated


eta_study = convergence(etas, sizes)
order, extrapolated = study_line(eta_study)
print("Fin efficiency")
print(f"  3D model (finest mesh)                 eta = {out['eta']:.5f}")
print(f"  Richardson extrapolation (order {order})  eta = {extrapolated}")
print(f"  Bessel solution, convective tip        eta = {eta_exact:.5f}   deviation {100 * (out['eta'] / eta_exact - 1):+.3f} %")
print(f"  Gardner (1945), corrected radius        eta = {eta_gardner:.5f}   deviation {100 * (out['eta'] / eta_gardner - 1):+.3f} %")
print(f"  change between the last two meshes: {100 * (etas[-1] / etas[-2] - 1):+.4f} %")

# **Energy balance and thermal-resistance network.**
q_air = out["q_fin"] + out["q_pipe"]
area_air = out["a_fin"] + out["a_pipe"]
eta_overall = 1.0 - out["a_fin"] / area_air * (1.0 - eta_exact)
mean_bore_radius = r_bore + groove_depth * 3.0 / 8.0
R_water = 1.0 / (h_water * out["a_bore"])
R_wall = np.log(r_wall / mean_bore_radius) / (wedge * k_cu * pitch)
R_air = 1.0 / (eta_overall * h_air * area_air)
q_network = (T_water - T_air) / (R_water + R_wall + R_air)
print("Heat rate per periodic element (one pitch, full circumference)")
print(f"  from the water  {24 * out['q_water']:.5f} W")
print(f"  to the air      {24 * q_air:.5f} W   (fins {100 * out['q_fin'] / q_air:.1f} %, bare pipe {100 * out['q_pipe'] / q_air:.1f} %)")
print(f"  energy imbalance {abs(out['q_water'] - q_air) / out['q_water']:.2e}")
print(f"  resistance network with Gardner efficiency  {24 * q_network:.5f} W   deviation {100 * (out['q_water'] / q_network - 1):+.3f} %")
print(f"  heat dissipation per length and kelvin  h_L = Q/(p dT) = {24 * out['q_water'] / (pitch * (T_water - T_air)):.2f} W/(m K)")

print("Mesh study (sizes h ~ 1/level)")
for name, key in (("Q_water / W", "q_water"), ("eta_fin", "eta"), ("T_max / K", "T_max"), ("T_min / K", "T_min")):
    vals = [o[key] * (24 if key == "q_water" else 1) for _, o in runs]
    order, extrapolated = study_line(convergence(vals, sizes))
    print(f"  {name:12s} " + "  ".join(f"{v:.6f}" for v in vals)
          + f"   order {order}, extrapolated {extrapolated}, last change {abs(vals[-1] - vals[-2]):.1e}")
coords = np.asarray(mesh.coordinates)
tip_nodes = np.isclose(np.hypot(coords[:, 1], coords[:, 2]), r_fin)
print(f"  base temperature T_b = {out['T_base']:.4f} K, tip temperature (mean) {np.mean(out['nodal'][tip_nodes]):.4f} K")


# ## Figure
#
# Left: the computed surface temperature of four pitches of the pipe, completed from the
# periodic element by its mirror planes (the element ends and the wedge planes) and its
# 12-fold rotational symmetry. Middle: the temperature over the cross-section through the fin mid-plane, where
# the grooves imprint a circumferential variation on the wall. Right: the fin
# temperature along its mid-plane against the Bessel solution, and the mesh study.

temperature, b = out["temperature"], out["b"]
limits = (out["T_min"], out["T_max"])
wedges = [((0, 0, 1), (0, 0, 0))]
pitches = [((1, 0, 0), (pitch / 2, 0, 0)), ((1, 0, 0), (-pitch / 2, 0, 0)), ((1, 0, 0), (1.5 * pitch, 0, 0))]
r_line = np.linspace(r_wall, r_fin * (1 - 1e-6), 80)
radial_lines = {f"3D model, theta = {np.degrees(angle):.0f} deg":
                np.column_stack((np.zeros_like(r_line), r_line * np.cos(angle), r_line * np.sin(angle))) for angle in (0.0, wedge)}
fig = plots.figure(layout=[["surface", "surface", "cut", "fin"], ["surface", "surface", "cut", "study"]], size=(17, 6.4))
plots.surface(temperature, boundary=b.fin_surface | b.pipe_surface | b.bore,
              symmetry={"planes": wedges + pitches, "sector": ((1, 0, 0), n_grooves)}, limits=limits, subdivisions=1,
              camera=plots.Camera(elevation=25, azimuth=-62, distance=0.85), label="T [K]",
              title="Surface temperature, four pitches", ax=fig["surface"])
plots.section(temperature, plane=Plane(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
              symmetry={"planes": wedges, "sector": ((1, 0, 0), n_grooves)}, subdivisions=3, contours=12, label="T [K]",
              limits=limits, title="Mid-plane section near the bore", xlimits=(-14 * mm, 14 * mm),
              ylimits=(-14 * mm, 14 * mm), ax=fig["cut"])
plots.line(points={label: (r_line[::4] / mm, ((np.asarray(temperature.evaluate(points)).reshape(-1) - T_air)
                                              / (out["T_base"] - T_air))[::4]) for label, points in radial_lines.items()},
           reference={"Bessel solution (1D fin)": (r_line / mm, fin_profile(r_line))}, xlabel="r [mm]",
           ylabel="(T - T_air) / (T_b - T_air)", title="Fin temperature along its mid-plane", legend="lower left", ax=fig["fin"])
plots.line(levels, {"3D model": etas}, hlines={"Bessel solution": eta_exact}, markers=True, xlabel="mesh level",
           ylabel="fin efficiency", title="Mesh study", ax=fig["study"])
plots.show()


# ## Going further
#
# Change the fin pitch, thickness or tip radius and watch the efficiency follow the
# Bessel solution; switch the fin material to aluminium ($k\approx 237$ W/(m K)) or
# stainless steel ($k\approx 15$ W/(m K)) to drive the efficiency down. Replace the
# uniform air-side coefficient by a circumferential distribution for cross-flow (higher
# at the front stagnation line) to see the field lose its rotational symmetry; the
# element must then span the full circumference.
