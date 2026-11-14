"""OneSimulation example: controlled diffusion separator - an H-shaped microchannel (3D).
Requires the installed, activated OneSimulation runtime.

Two laminar water streams meet in the 100 micrometre contact zone of an H-shaped
microchannel and part again. Small molecules diffuse across the interface, large ones
stay in their stream, so the device separates by molecular size. The 3D creeping flow is
computed with cubic/quadratic Taylor-Hood finite elements, the solute transport with cubic
Lagrange elements for three diffusivities. Graded against the published outlet
concentrations of this benchmark, the exact duct velocity, a hydraulic-resistance estimate
of the inlet pressure, a Graetz-type eigenfunction solution, the mass balances, a
three-level mesh study and a comparison with quadratic elements.
"""


# # Controlled diffusion separator: separation by molecular size in an H-cell
#
# The H-cell (formulated by A. Witarsa and B. A. Finlayson, University of Washington, as
# a test case for evaluating microfluidic patents; the "H-filter" of J. P. Brody and
# P. Yager, *Sensors and Actuators A* 58, 13, 1997) brings two laminar streams into
# contact for a controlled time. Stream A enters with a solute at $c_0=1$ mol/m³,
# stream B with pure water. Both flow side by side through a straight contact zone and
# are split again. At a Reynolds number of order $10^{-3}$ the streams do not mix by
# convection; the solute crosses the interface only by diffusion. A small molecule (large
# diffusivity $D$) leaves through both outlets almost equally, a large molecule stays
# in stream A. The outlet-B concentration as a function of $D$ is the design curve of
# the device.
#
# **Setup** (lengths in µm, the published benchmark "controlled diffusion separator"):
#
# | item | value |
# |---|---|
# | channel width (every arm, each half of the contact zone) | 10 |
# | channel depth | 20 (half depth 10 computed, symmetry plane at mid-depth) |
# | arms | straight, 40 long, then a quarter bend with inner radius 10 and outer radius 20 |
# | contact zone | straight, 100 long, 20 wide; the four bends meet tangentially at its ends |
# | flow rate through each inlet | 15 pl/s (mean velocity 0.075 mm/s) |
# | water | ρ = 1000 kg/m³, μ = 1 mPa s |
# | diffusivities | $D=10^{-10}$, $5\cdot10^{-11}$, $10^{-11}$ m²/s |
#
# The Reynolds number $\rho\bar uW/\mu=7.5\cdot10^{-4}$ makes the flow creeping; the
# Navier-Stokes equations are solved with their (negligible) inertia, followed by the
# steady convection-diffusion equation of the dilute solute in that flow:
#
# $$\rho(\mathbf u\cdot\nabla)\mathbf u=-\nabla p+\mu\nabla^2\mathbf u,\quad\nabla\cdot\mathbf u=0,
#   \qquad \mathbf u\cdot\nabla c=D\nabla^2 c .$$
#
# The geometry is genuinely three-dimensional: the streams turn through the bends,
# merge and split, and the velocity varies across the width *and* through the depth, so
# the solute front at the interface is curved (it advances fastest at mid-depth).
# Boundary conditions on every face:
#
# | face | flow | solute |
# |---|---|---|
# | inlet A (lower left arm) | fully developed duct profile, 15 pl/s | $c=c_0$ |
# | inlet B (upper left arm) | fully developed duct profile, 15 pl/s | $c=0$ |
# | outlets A and B (right arms) | traction-free, $\boldsymbol\sigma\cdot\mathbf n=0$ ($p=0$) | outflow: zero diffusive flux |
# | side walls and bottom $z=0$ | no slip | no flux |
# | mid-depth plane $z=10$ µm | symmetry: impermeable, shear-free | no flux |
#
# **Numerical methods.**
#
# * Flow: continuous Galerkin finite elements, **higher-order Taylor-Hood P3/P2
#   tetrahedra** (cubic velocity, quadratic pressure; inf-sup stable), Newton iteration
#   with a sparse direct LU solver.
# * Solute: continuous Galerkin finite elements with **cubic (P3) Lagrange tetrahedra** in
#   the computed velocity field, sparse LU (the problem is linear). The steep gradient is
#   across the streams, where convection does not act, so no stabilisation is added. The
#   price is visible for the large molecule: along the streams the cell Péclet number of
#   the cubic elements exceeds one for $D=10^{-11}$, and the unstabilised solution over-
#   and undershoots locally (printed below); the outlet concentrations, which are flux
#   integrals, are not affected, as the mesh study and the quadratic contrast show.
# * Contrast: the same mesh with quadratic Taylor-Hood P2/P1 flow and P2 solute (a common
#   default), graded against the same references.
#
# The solute equation is solved with OneSimulation's convection-diffusion operator
# `ConvectiveHeatTransfer`, which is the same equation: with unit volumetric capacity and
# "conductivity" $k=D$ its unknown is the concentration.
#
# **Reference.**
#
# * The published outlet-B concentrations of this benchmark, $c_B=0.492$, $0.448$ and
#   $0.235$ mol/m³ for $D=10^{-10}$, $5\cdot10^{-11}$ and $10^{-11}$ m²/s (read from its
#   design curve to about ±0.003).
# * Flow: the exact fully developed velocity in a rectangular duct (Fourier series;
#   R. K. Shah and A. L. London, *Laminar Flow Forced Convection in Ducts*, Academic Press
#   1978, ch. VII) in an arm and in the contact zone, the flow split, and the inlet
#   pressure from the exact hydraulic resistance of the straight duct sections along the
#   stream path (bends taken at their centre-line length, merging neglected; a 2-3 %
#   estimate).
# * Solute: a Graetz-type eigenfunction expansion of the parallel-flow problem in the
#   contact zone with the exact duct velocity and a step in concentration at its start
#   (the classical T-sensor/H-filter analysis of A. E. Kamholz, B. H. Weigl,
#   B. A. Finlayson and P. Yager, *Anal. Chem.* 71, 5340, 1999). It neglects the merging
#   and splitting at the ends of the contact zone and axial diffusion, so it is expected
#   to agree with the 3D result to one or two per cent, not exactly.
# * Mass balances of water and solute through the four arms, and a three-level mesh study
#   with observed order and Richardson estimate.

import time
import numpy as np
from scipy.linalg import eigh

from onesim.cad.d2 import Plane
from onesim.materials import IsotropicSolid, NewtonianFluid
from onesim.meshing import Block, Geometric, arc, extrude, line, merge_blocks, rectangle_block
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.heat_transfer import ConvectiveHeatTransfer, FixedTemperature, HeatFlux
from onesim.physics.hydrodynamics import (
    FixedVelocity, FlowTraction, FreeSlipLaw, IncompressibleNavierStokes, NoSlipWall, WallShear,
)
from onesim.results.comparisons import convergence
from onesim.results.derived import derived_field
from onesim.results.integration import integrate_plane, mean_value
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()


# ## Parameters

UM = 1e-6
W = 10 * UM
H = 20 * UM
L_ARM = 40 * UM
R_IN, R_OUT = 10 * UM, 20 * UM
X0, X1 = 20 * UM, 120 * UM
L_CONTACT = X1 - X0
Q_in = 15e-15
u_in = Q_in / (W * H)
rho, mu = 1000.0, 1.0e-3
c0 = 1.0
diffusivities = (1e-10, 5e-11, 1e-11)
c_B_published = {1e-10: 0.492, 5e-11: 0.448, 1e-11: 0.235}
p_in_published = 2.29
levels = (1, 2, 3)

print(f"mean inlet velocity {u_in*1e3:.4f} mm/s, Re = rho u W / mu = {rho*u_in*W/mu:.1e}")
for D in diffusivities:
    print(f"D = {D:.0e} m^2/s: Pe = u W / D = {u_in*W/D:6.1f}, contact time L/u = {L_CONTACT/u_in:.2f} s, "
          f"diffusion length sqrt(D L/u) = {np.sqrt(D*L_CONTACT/u_in)/UM:.2f} um")


def duct_velocity(s, z, a, b, mean, terms=60):
    """Exact fully developed velocity in a rectangular duct 0<s<a, 0<z<b with the given mean."""
    s, z = np.asarray(s, float), np.asarray(z, float)
    n = 2 * np.arange(terms) + 1
    coeff = 1.0 / (n[:, None] * n[None, :] * ((n[:, None] / a) ** 2 + (n[None, :] / b) ** 2))
    norm = np.sum(coeff * 4 / (n[:, None] * n[None, :] * np.pi**2))
    Ss = np.sin(np.pi * np.multiply.outer(s, n) / a)
    Sz = np.sin(np.pi * np.multiply.outer(z, n) / b)
    return mean * np.einsum("...i,ij,...j->...", Ss, coeff, Sz) / norm


def duct_resistance(a, b, terms=200):
    """Exact hydraulic resistance per length, dp/dx / Q, of a rectangular duct a x b (a >= b)."""
    n = 2 * np.arange(terms) + 1
    shape = 1 - 192 * b / (np.pi**5 * a) * np.sum(np.tanh(n * np.pi * a / (2 * b)) / n**5)
    return 12 * mu / (a * b**3 * shape)


path_arm = 2 * (L_ARM + np.pi / 2 * (R_IN + R_OUT) / 2)
p_in_estimate = Q_in * duct_resistance(H, W) * path_arm + 2 * Q_in * duct_resistance(2 * W, H) * L_CONTACT
print(f"inlet pressure from the duct resistances: {p_in_estimate:.3f} Pa (arms {path_arm/UM:.1f} um, contact zone "
      f"{L_CONTACT/UM:.0f} um)")


# ## Geometry and mesh
#
# Only the lower half of the depth, $0\le z\le H/2$, is computed; the mid-plane $z=H/2$ is
# the symmetry plane of the original model. The H-shaped section is covered by
# quadrilateral blocks: the straight arms, the four quarter-annulus bends (exact circular
# arcs) and the two halves of the contact zone. `merge_blocks` joins them into one
# conforming section mesh, and the cells across each stream are graded towards the
# interface (first cell one third of the last) where the concentration gradient is
# steepest. The quadrilaterals are split into triangles and `extrude` sweeps the section
# over the half depth with layers graded towards the bottom wall, splitting every prism
# into three tetrahedra. With cubic elements, $n=3$ cells across a stream give 9 nodal intervals across
# each stream and through the half depth.

def section_blocks(nw):
    """Planar blocks of the H-shaped section; the second block direction runs across a stream."""
    n_arm, n_bend, n_contact = 2 * nw, int(round(1.6 * nw)), 5 * nw
    across = Geometric(nw, ratio=3.0)
    nodes = across.reference()
    blocks = []
    for sy, stream in ((+1, "b"), (-1, "a")):
        for xc, sx, kind in ((X0, -1, "inlet"), (X1, +1, "outlet")):
            centre = (xc, sy * R_OUT)
            theta0 = -sy * np.pi / 2
            theta1 = theta0 + sy * sx * np.pi / 2
            at = lambda r, th: (centre[0] + r * np.cos(th), centre[1] + r * np.sin(th))
            blocks.append(Block(arc(centre, R_OUT, theta0, theta1), line(at(R_OUT, theta1), at(R_IN, theta1)),
                                arc(centre, R_IN, theta0, theta1), line(at(R_OUT, theta0), at(R_IN, theta0)),
                                cells=(n_bend, nw), grading=(None, across)))
            xs = np.sort(xc + sx * (R_OUT + (R_IN - R_OUT) * nodes))
            ys = np.sort(sy * (R_OUT + L_ARM * np.linspace(0.0, 1.0, n_arm + 1)))
            end = f"{kind}_{stream}"
            blocks.append(rectangle_block((xs[0], ys[0]), xs[-1] - xs[0], ys[-1] - ys[0], (nw, n_arm),
                                          grading=(xs, ys), names=(end if sy < 0 else None, None, end if sy > 0 else None, None)))
        ys = np.sort(sy * W * nodes)
        blocks.append(rectangle_block((X0, ys[0]), L_CONTACT, ys[-1] - ys[0], (n_contact, nw), grading=(None, ys)))
    return blocks


def build_mesh(nw):
    section = merge_blocks(section_blocks(nw), unnamed="walls").triangulated()
    mesh = extrude(section, H / 2, Geometric(nw, ratio=2.0), names=("walls", "mid_plane"))
    return mesh.geometry, mesh


# ## Model
#
# The inlets carry the exact duct profile scaled to 15 pl/s (7.5 pl/s through each half
# inlet of the computed half depth). The symmetry plane is a `WallShear` condition with a
# zero shear law: the normal velocity vanishes and no tangential stress acts.

fluid = NewtonianFluid(density=rho, dynamic_viscosity=mu)


def inlet_velocity(x, y, z):
    """Fully developed duct profile in both inlet arms (flowing towards the contact zone)."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    u = duct_velocity(np.clip(x, 0, W), np.clip(z, 0, H), W, H, u_in)
    return np.column_stack((np.zeros_like(u), -np.sign(y) * u, np.zeros_like(u)))


def no_traction(x, y, z):
    return np.zeros((len(np.asarray(x)), 3))


def solve_flow(geometry, mesh, order):
    model = IncompressibleNavierStokes(domain=geometry.domain(), material=fluid)
    b = geometry.boundaries
    problem = Problem(
        name="Controlled diffusion separator: flow", geometry=geometry, physics=(model,),
        boundary_conditions=(
            FixedVelocity(boundary=b.inlet_a | b.inlet_b, physics=model, value=inlet_velocity),
            FlowTraction(boundary=b.outlet_a | b.outlet_b, physics=model, value=no_traction),
            NoSlipWall(boundary=b.walls, physics=model),
            WallShear(boundary=b.mid_plane, physics=model, law=FreeSlipLaw()),
        ),
        initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
    )
    newton = Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-15, relative_tolerance=1e-10,
                    maximum_iterations=10)
    result = Simulation(problem=problem, mesh=mesh, numerics=Numerics(
        spatial=FiniteElement(element=TaylorHood(order=order)), nonlinear_solver=newton,
        initial_guess={model.velocity: (0.0, 0.0, 0.0)})).solve_stationary()
    return result.field(model.velocity), result.field(model.pressure), result.operators[model.model_id].problem_size


def solve_solute(geometry, mesh, velocity, D, order):
    solute = IsotropicSolid(density=1.0, specific_heat_capacity=1.0, thermal_conductivity=D,
                            youngs_modulus=1.0, poisson_ratio=0.3, thermal_expansion_coefficient=0.0)
    flow = lambda x, y, z: np.asarray(velocity.evaluate(np.column_stack((x, y, z))))
    model = ConvectiveHeatTransfer(domain=geometry.domain(), material=solute, velocity=flow)
    b = geometry.boundaries
    problem = Problem(
        name="Controlled diffusion separator: solute", geometry=geometry, physics=(model,),
        boundary_conditions=(
            FixedTemperature(physics=model, boundary=b.inlet_a, value=c0),
            FixedTemperature(physics=model, boundary=b.inlet_b, value=0.0),
            HeatFlux(physics=model, boundary=b.outlet_a | b.outlet_b, value=0.0),
            HeatFlux(physics=model, boundary=b.walls | b.mid_plane, value=0.0),
        ),
    )
    numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=order), convection=GalerkinAdvection()),
                        linear_solver=SparseDirect())
    return Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.temperature)


def plane_flux(velocity, concentration, axis, value, span):
    """Volume flow, solute flow and area-averaged concentration through a cross-section (full depth)."""
    component = 0 if axis == "x" else 1
    normal = np.eye(3)[component]
    plane = Plane(origin=tuple(value * normal), normal=tuple(normal))
    other = 1 - component
    within = lambda p: (p[:, other] >= span[0] - 1e-12) & (p[:, other] <= span[1] + 1e-12)
    volume_flow = 2 * float(np.asarray(integrate_plane(velocity, plane, component=component, within=within,
                                                       subdivisions=4).value))
    if concentration is None:
        return volume_flow, volume_flow, 1.0
    carried = derived_field(velocity, lambda u, p: u[:, component] * np.asarray(concentration.evaluate(p)).ravel(),
                            name="solute_flux", unit="mol/m^2/s")
    solute_flow = 2 * float(np.asarray(integrate_plane(carried, plane, within=within, subdivisions=4).value))
    return volume_flow, solute_flow, mean_value(concentration, plane=plane, within=within, subdivisions=4)


ARMS = {"inlet A": (-40 * UM, (0, W)), "inlet B": (40 * UM, (0, W)),
        "outlet A": (-40 * UM, (130 * UM, 140 * UM)), "outlet B": (40 * UM, (130 * UM, 140 * UM))}


# ## Solve
#
# Three meshes with 1, 2 and 3 cells across each stream (and proportionally along the
# channel and through the half depth), all with P3/P2 flow and P3 solute for the most
# demanding molecule ($D=10^{-11}$); on the finest mesh also the two larger
# diffusivities and the quadratic P2/P1 + P2 contrast.

runs = []
for nw in levels:
    start = time.perf_counter()
    geometry, mesh = build_mesh(nw)
    velocity, pressure, flow_dofs = solve_flow(geometry, mesh, 3)
    t_flow = time.perf_counter() - start
    run = dict(nw=nw, geometry=geometry, mesh=mesh, velocity=velocity, pressure=pressure, solute={})
    for D in (diffusivities if nw == levels[-1] else (1e-11,)):
        run["solute"][D] = solve_solute(geometry, mesh, velocity, D, 3)
    runs.append(run)
    print(f"mesh n = {nw}: {mesh.cell_count:6d} tetrahedra, {flow_dofs:7d} flow unknowns (P3/P2); flow {t_flow:4.0f} s, "
          f"total {time.perf_counter()-start:4.0f} s")
fine = runs[-1]
start = time.perf_counter()
v2, p2, dofs2 = solve_flow(fine["geometry"], fine["mesh"], 2)
c2 = solve_solute(fine["geometry"], fine["mesh"], v2, 1e-11, 2)
print(f"quadratic contrast on mesh n = {fine['nw']}: {dofs2} flow unknowns (P2/P1), {time.perf_counter()-start:.0f} s")


# ## Reference solutions
#
# **Graetz-type expansion.** In the contact zone the reference concentration is expanded
# in the cosine modes $\cos(p\pi s/2W)\cos(2q\pi z/H)$ ($s=y+W$), which satisfy the
# no-flux conditions at the walls and at the mid-plane. The Galerkin form of
# $u\,\partial_x c=D\nabla_\perp^2 c$ is $M\,\dot a=-D\,K a$ with the velocity-weighted
# mass matrix $M$ (exact duct velocity, Gauss quadrature) and the diagonal Laplacian $K$;
# the generalised eigenvalue problem $K v=\lambda M v$ gives
# $a(x)=\sum_n v_n e^{-D\lambda_n x}(v_n^{\mathsf T}Ma(0))$. Since both outlets carry the
# same flow, the flow-weighted (mixing-cup) concentration at outlet B equals the
# fraction of the solute that leaves through it, $c_B/c_0=\dot m_B/\dot m_{\rm in}$
# (0.5 means complete mixing). With 100 x 12 modes it is converged to better than
# $10^{-7}$.

def graetz_reference(P=100, Q=12):
    a, hz = 2 * W, H / 2
    edges = np.linspace(0, a, 9)
    gs, ws = np.polynomial.legendre.leggauss(30)
    S = (edges[:-1, None] + np.diff(edges)[:, None] * (gs + 1) / 2).ravel()
    WS = (np.diff(edges)[:, None] * ws / 2).ravel()
    gz, wz = np.polynomial.legendre.leggauss(60)
    Z, WZ = hz * (gz + 1) / 2, hz * wz / 2
    SS, ZZ = np.meshgrid(S, Z, indexing="ij")
    u = duct_velocity(SS, ZZ, a, H, u_in)
    p, q = np.arange(P), np.arange(Q)
    basis = np.einsum("ip,jq->ijpq", np.cos(np.pi * np.outer(S, p) / a),
                      np.cos(np.pi * np.outer(Z, q) / hz)).reshape(len(S), len(Z), P * Q)
    weighted = (np.outer(WS, WZ) * u)[:, :, None] * basis
    M = weighted.reshape(-1, P * Q).T @ basis.reshape(-1, P * Q)
    norms = np.outer(np.where(p == 0, a, a / 2), np.where(q == 0, hz, hz / 2))
    K = np.diag((((np.pi * p[:, None] / a) ** 2 + (np.pi * q[None, :] / hz) ** 2) * norms).ravel())
    lam, V = eigh(K, M)
    start = V.T @ np.einsum("ijk,ij->k", weighted, c0 * (SS < W))
    to_b = np.einsum("ijk,ij->k", weighted, (SS > W).astype(float))
    total = np.einsum("ijk->k", weighted)

    def coefficients(D, x):
        return V @ (np.exp(-D * lam * x) * start)

    def fraction_b(D, x=L_CONTACT):
        a_x = coefficients(D, x)
        return (a_x @ to_b) / (a_x @ total)

    def concentration(D, y, z, x=L_CONTACT):
        modes = np.einsum("...p,...q->...pq", np.cos(np.pi * np.multiply.outer(np.asarray(y) + W, p) / a),
                          np.cos(np.pi * np.multiply.outer(np.asarray(z), q) / hz))
        return modes.reshape(np.shape(y) + (P * Q,)) @ coefficients(D, x)

    return fraction_b, concentration


graetz_fraction, graetz_concentration = graetz_reference()


# ## Comparison with the reference
#
# **Flow.** Fully developed velocity on the centre line of an arm (half-way along its
# straight part) and of the contact zone (half-way along it), the flow through the
# contact zone, the split between the two outlets and the inlet pressure.

def centre_velocity(velocity, point, component):
    return float(np.asarray(velocity.evaluate(np.array([point])))[0, component])


def inlet_pressure(pressure):
    return float(np.asarray(pressure.evaluate(np.array([[W / 2, -(R_OUT + L_ARM) + 1e-9, H / 2 - 1e-12]])))[0])


u_arm_exact = float(duct_velocity(W / 2, H / 2 - 1e-12, W, H, u_in))
u_contact_exact = float(duct_velocity(W, H / 2 - 1e-12, 2 * W, H, u_in))
for run in runs:
    v = run["velocity"]
    run["u_arm"] = centre_velocity(v, (W / 2, -40 * UM, H / 2 - 1e-12), 1)
    run["u_contact"] = centre_velocity(v, (70 * UM, 0.0, H / 2 - 1e-12), 0)
    run["q_contact"] = plane_flux(v, None, "x", 70 * UM, (-W, W))[0] / (2 * Q_in)
    run["split"] = plane_flux(v, None, "y", *ARMS["outlet B"])[0] / -plane_flux(v, None, "y", *ARMS["outlet A"])[0]
    run["p_in"] = inlet_pressure(run["pressure"])
print(" mesh   u_max arm / exact [mm/s]       u_max contact / exact [mm/s]    Q contact / 2 Q_in   split B/A   inlet pressure [Pa]")
for run in runs:
    print(f"  {run['nw']}    {run['u_arm']*1e3:.5f} / {u_arm_exact*1e3:.5f} ({run['u_arm']/u_arm_exact-1:+.1e})"
          f"    {run['u_contact']*1e3:.5f} / {u_contact_exact*1e3:.5f} ({run['u_contact']/u_contact_exact-1:+.1e})"
          f"     {run['q_contact']:.6f}       {run['split']:.6f}   {run['p_in']:.4f}")
print(f"inlet pressure: {fine['p_in']:.3f} Pa; duct-resistance estimate {p_in_estimate:.3f} Pa "
      f"({fine['p_in']/p_in_estimate-1:+.1%}); published about {p_in_published:.2f} Pa ({fine['p_in']/p_in_published-1:+.1%})")

# **Water and solute balance** on the finest mesh: inflow through the two inlets against
# the outflow through the two outlets (cross-sections half-way along the arms).

flows = {name: plane_flux(fine["velocity"], None, "y", *where)[0] for name, where in ARMS.items()}
q_in_total, q_out_total = flows["inlet A"] - flows["inlet B"], flows["outlet B"] - flows["outlet A"]
print(f"\nwater: inflow {q_in_total*1e15:.4f} pl/s (prescribed {2*Q_in*1e15:.4f}), outflow {q_out_total*1e15:.4f} pl/s,"
      f" out/in - 1 = {q_out_total/q_in_total-1:+.1e}")
outlet = {}
for D, c in fine["solute"].items():
    m = {name: plane_flux(fine["velocity"], c, "y", *where) for name, where in ARMS.items()}
    m_in, m_a, m_b = m["inlet A"][1] - m["inlet B"][1], -m["outlet A"][1], m["outlet B"][1]
    outlet[D] = dict(cup=m_b / m["outlet B"][0], area=m["outlet B"][2], fraction=m_b / (m_a + m_b))
    print(f"solute D = {D:.0e}: inflow {m_in*1e15:.4f} fmol/s (prescribed {c0*Q_in*1e15:.4f}), outflow A "
          f"{m_a*1e15:.4f}, outflow B {m_b*1e15:.4f} fmol/s, (out A + out B)/in - 1 = {(m_a+m_b)/m_in-1:+.1e}")

# **Outlet-B concentration** on the finest mesh against the published design curve and
# the Graetz-type reference, as mixing-cup (flow-weighted) and as area average over the
# outlet cross-section, with the over- and undershoot of the Galerkin solution (the exact
# solution is bounded by $0\le c\le c_0$).

print("\n D [m^2/s]   c_B mixing-cup   c_B area avg   published   deviation   Graetz   deviation   min c / c0   max c / c0")
nodes = fine["mesh"].coordinates
for D, c in fine["solute"].items():
    o = outlet[D]
    values = np.asarray(c.evaluate(nodes)).ravel()
    print(f" {D:.0e}       {o['cup']:.4f}          {o['area']:.4f}        {c_B_published[D]:.3f}     "
          f"{o['cup']/c_B_published[D]-1:+.2%}     {graetz_fraction(D):.4f}   {o['cup']/graetz_fraction(D)-1:+.2%}"
          f"      {values.min()/c0:+.4f}      {values.max()/c0:.4f}")

# **Mesh study** with $n=1,2,3$ cells across a stream ($h\propto1/n$). For the flow
# quantities with an exact value the observed order follows from successive errors; for
# the inlet pressure and the outlet concentration (no closed form for the full 3D
# geometry) $p$ solves $(f_1-f_2)/(f_2-f_3)=(h_1^p-h_2^p)/(h_2^p-h_3^p)$ and the Richardson
# estimate is $f_3+(f_3-f_2)\,h_3^p/(h_2^p-h_3^p)$. The last line compares the quadratic
# P2/P1 + P2 discretization on the finest mesh.

for run in runs:
    c = run["solute"][1e-11]
    m_b = plane_flux(run["velocity"], c, "y", *ARMS["outlet B"])
    run["c_B_low"] = m_b[1] / m_b[0]


def richardson(values, hs):
    estimate = convergence(values, hs)
    return (estimate.order, estimate.extrapolated) if estimate.order is not None else (np.nan, np.nan)


hs = np.array([1.0 / run["nw"] for run in runs])
print("\nquantity                     n = 1        n = 2        n = 3      reference     change last two   observed order")
for label, key, ref, scale in (("u_max arm [mm/s]", "u_arm", u_arm_exact, 1e3),
                               ("u_max contact [mm/s]", "u_contact", u_contact_exact, 1e3),
                               ("Q contact / 2 Q_in", "q_contact", 1.0, 1.0)):
    f = np.array([run[key] for run in runs])
    e = np.abs(f / ref - 1)
    orders = np.log(e[:-1] / e[1:]) / np.log(hs[:-1] / hs[1:])
    print(f"{label:24s} " + "  ".join(f"{scale*v:11.6f}" for v in f) + f"   {scale*ref:10.6f}"
          f"      {abs(f[2]/f[1]-1):.1e}        {orders[0]:.2f}, {orders[1]:.2f}")
for label, key in (("inlet pressure [Pa]", "p_in"), ("c_B, D = 1e-11", "c_B_low")):
    f = np.array([run[key] for run in runs])
    p, rich = richardson(f, hs)
    estimate = f"Richardson {rich:.4f}   " if np.isfinite(rich) else "(not monotone)      "
    print(f"{label:24s} " + "  ".join(f"{v:11.6f}" for v in f) + f"   {estimate}{abs(f[2]/f[1]-1):.1e}        "
          + (f"{p:.2f}" if np.isfinite(p) else f"changes {f[1]-f[0]:+.1e}, {f[2]-f[1]:+.1e}"))
m_b2 = plane_flux(v2, c2, "y", *ARMS["outlet B"])
values2 = np.asarray(c2.evaluate(nodes)).ravel()
print(f"P2/P1 + P2 on mesh n = {fine['nw']}: u_max contact {centre_velocity(v2, (70*UM, 0.0, H/2-1e-12), 0)*1e3:.5f} mm/s "
      f"({centre_velocity(v2, (70*UM, 0.0, H/2-1e-12), 0)/u_contact_exact-1:+.1e}), inlet pressure {inlet_pressure(p2):.4f} Pa, "
      f"c_B (D = 1e-11) = {m_b2[1]/m_b2[0]:.4f} (P3: {fine['c_B_low']:.4f}), c in [{values2.min():+.4f}, {values2.max():.4f}]")


# ## Figure
#
# Top: the velocity magnitude and the concentration on the mid-depth plane of the whole
# separator, for the large molecule ($D=10^{-11}$, little mixing) and the small molecule
# ($D=10^{-10}$, almost complete mixing). Middle: cross-sections of the contact zone for
# $D=10^{-11}$ (full depth, mirrored at the symmetry plane). Bottom: velocity across the
# contact zone against the exact duct solution, the concentration at the end of the
# contact zone against the Graetz-type reference, the outlet-B concentration against the
# published design curve, and the mesh study.

b_fine = fine["geometry"].boundaries
cuts = (25, 45, 70, 95, 118)
clipped = lambda c, p: np.clip(c, 0, c0)
y_line = np.linspace(-W + 1e-9, W - 1e-9, 121)
z_line = np.linspace(1e-9, H / 2 - 1e-9, 61)
u_y = np.asarray(fine["velocity"].evaluate(np.column_stack((np.full(121, 70 * UM), y_line, np.full(121, H / 2 - 1e-9)))))[:, 0]
u_z = np.asarray(fine["velocity"].evaluate(np.column_stack((np.full(61, 70 * UM), np.zeros(61), z_line))))[:, 0]
graetz_fem, graetz_exact = {}, {}
for D in diffusivities:
    for z_probe, name in ((H / 2 - 1e-9, "z = H/2"), (2 * UM, "z = 2 µm")):
        points = np.column_stack((np.full(121, X1 - 1e-9), y_line, np.full(121, z_probe)))
        graetz_fem[f"D = {D:.0e}, {name}"] = (y_line[::8] / UM, np.asarray(fine["solute"][D].evaluate(points)).ravel()[::8])
        graetz_exact[f"D = {D:.0e}, {name}"] = (y_line / UM, graetz_concentration(D, y_line, np.full(121, z_probe)))
D_line = np.geomspace(5e-12, 2e-10, 120)
fig = plots.figure(layout=[["speed"] * 20 + ["slow"] * 20 + ["fast"] * 20,
                           [f"cut {x}" for x in cuts for _ in range(12)],
                           ["duct"] * 15 + ["graetz"] * 15 + ["outlet"] * 15 + ["mesh"] * 15], size=(17, 15.5))
plots.surface(fine["velocity"], boundary=b_fine.mid_plane, component=lambda u, p: 1e3 * np.linalg.norm(u, axis=1),
              contours=7, subdivisions=3, label="|u| [mm/s]", title="|u| on the mid-depth plane (P3/P2)",
              ylimits=(-70 * UM, 68 * UM), ax=fig["speed"])
for panel, D in (("slow", 1e-11), ("fast", 1e-10)):
    plots.surface(fine["solute"][D], boundary=b_fine.mid_plane, component=clipped, limits=(0.0, c0), contours=9,
                  subdivisions=3, label="c / c0", ylimits=(-70 * UM, 68 * UM), ax=fig[panel],
                  title=f"c / c0, D = {D:.0e} m²/s: c_B = {outlet[D]['cup']:.3f} mol/m³")
for x_cut in cuts:
    plots.section(fine["solute"][1e-11], plane=Plane(origin=(x_cut * UM, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
                  component=clipped, symmetry={"planes": [((0, 0, 1), (0, 0, H / 2))]}, limits=(0.0, c0), contours=9,
                  subdivisions=3, colorbar=False, title=f"D = 1e-11, x = {x_cut} µm", xlimits=(-W, W), ylimits=(0.0, H),
                  ax=fig[f"cut {x_cut}"])
plots.line(points={"across (z = H/2)": (y_line[::6] / UM, u_y[::6] * 1e3),
                   "through depth (y = 0)": ((z_line[::3] - H / 2) / UM, u_z[::3] * 1e3)},
           reference={"across (z = H/2)": (y_line / UM, duct_velocity(y_line + W, H / 2, 2 * W, H, u_in) * 1e3),
                      "through depth (y = 0)": ((z_line - H / 2) / UM, duct_velocity(W, z_line, 2 * W, H, u_in) * 1e3)},
           xlabel="y, or z - H/2 [µm]", ylabel="u_x [mm/s]", legend="lower center",
           title="Velocity at x = 70 µm: FEM (markers), exact duct flow (dashed)", ax=fig["duct"])
plots.line(points=graetz_fem, reference=graetz_exact, xlabel="y [µm]", ylabel="c / c0", legend="upper right",
           title="c at the end of the contact zone: FEM (markers), Graetz (dashed)", ax=fig["graetz"])
plots.line(D_line, {"Graetz-type reference": [graetz_fraction(D) for D in D_line]},
           points={"published benchmark": (list(c_B_published), list(c_B_published.values())),
                   "3D FEM, mixing-cup": (list(outlet), [o["cup"] for o in outlet.values()]),
                   "3D FEM, area average": (list(outlet), [o["area"] for o in outlet.values()])},
           hlines=0.5, scale="logx", ylimits=(0, 0.55), xlabel="diffusivity D [m²/s]", ylabel="c at outlet B [mol/m³]",
           title="Separation: outlet-B concentration", legend="lower right", ax=fig["outlet"])
plots.line([r["nw"] for r in runs], {"P3, D = 1e-11": [r["c_B_low"] for r in runs]},
           points={"P2, D = 1e-11": ([fine["nw"]], [m_b2[1] / m_b2[0]])},
           hlines={"published": c_B_published[1e-11], "Graetz-type reference": graetz_fraction(1e-11)}, markers=True,
           xlabel="cells across one stream", ylabel="c_B [mol/m³]", title="Mesh study", ax=fig["mesh"])
plots.show()


# ## Going further
#
# * Make the viscosity depend on the solute, $\mu=\mu_0(1+\alpha c^2)$ with
#   $\alpha=0.5$ m⁶/mol², as for macromolecules: flow and transport then have to be
#   solved together (a Picard iteration of the two solves above, with the viscosity as a
#   position-dependent material law). The velocity profile becomes asymmetric and a
#   higher pressure is needed at the inlet of the solute-laden stream, while the
#   published benchmark reports that the outlet concentration for
#   $D=5\cdot10^{-11}$ m²/s changes only from 0.448 to 0.450 mol/m³.
# * A cascade of separators, each fed with the B outlet of the previous one, enriches the
#   small molecule further; with the Graetz reference the number of stages for a target
#   purity follows directly.
# * Halve the flow rate: the contact time doubles and the large molecule's outlet
#   concentration rises roughly by $\sqrt2$ while the small molecule is already mixed.
