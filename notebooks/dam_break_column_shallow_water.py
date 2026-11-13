"""OneSimulation example: dam break on a square column with the shallow-water equations.
Requires the installed, activated OneSimulation runtime.

A 0.3 m deep water body is released by a gate and the bore strikes a square column in
a tank. The 2D shallow-water equations are solved by a second-order Godunov-type finite-
volume scheme (HLL flux, MUSCL, SSP-RK3) on three grids and checked against Stoker's exact
dam-break solution, the exact bore-reflection force at impact and mass conservation.
"""


# # Dam break on a column (shallow-water equations)
#
# A tank 1.60 m long and 0.61 m wide holds a 0.40 m long, 0.30 m deep body of water
# behind a gate; the rest of the floor is covered by a 0.01 m layer. A tall column
# with a 0.12 m square base stands 0.50 m downstream of the gate and 0.25 m from one
# side wall. At $t=0$ the gate is removed: a bore runs down the tank, strikes the
# column, wraps around it, is reflected by the far wall and strikes the column a
# second time from behind. The arrangement follows the dam-break experiment reported by
# Raad & Bidoae (J. Comput. Phys. 203, 668-699, 2005).
#
# The depth-averaged flow obeys the 2D shallow-water (Saint-Venant) equations over a
# flat, frictionless floor,
#
# $$\partial_t h+\nabla\cdot(h\mathbf u)=0,\qquad
#   \partial_t(h\mathbf u)+\nabla\cdot\Bigl(h\mathbf u\otimes\mathbf u+\tfrac12 g h^2\mathbf I\Bigr)=0,$$
#
# with depth $h$ and depth-averaged velocity $\mathbf u$. The pressure is hydrostatic,
# so the force per unit length on a wall is $\rho g h^2/2$ and the force on the column
# is the integral of $\rho g h^2\mathbf n/2$ over its four faces.
#
# **Method.** These are exactly the 2D gas-dynamics equations of a barotropic fluid
# with "density" $h$ and "pressure" $p=g h^2/2$ (the hydraulic analogy, $\gamma=2$).
# OneSimulation's compressible cell-face finite-volume provider
# (`CompressibleFaceFiniteVolume` on an `UnstructuredCompressibleGeometry`) accepts the
# equation of state, the face flux and the reconstruction as injected components, so
# the shallow-water system is solved with:
#
# * a cell-centred finite-volume method on a quadrilateral grid with the column cut
#   out, conservative face fluxes on every face;
# * a Godunov-type HLL Riemann flux with Einfeldt wave-speed estimates
#   (Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*, Wiley 2001);
# * MUSCL reconstruction of $(h, u, v)$ with OneSimulation's least-squares cell
#   gradients and the Barth-Jespersen limiter, velocity desingularisation in nearly
#   dry cells (Kurganov & Petrova, Commun. Math. Sci. 5, 2007) and first-order fluxes
#   next to them;
# * slip walls whose flux uses the exact solution of the wall Riemann problem;
# * explicit third-order strong-stability-preserving Runge-Kutta time stepping
#   (SSP-RK3, Shu & Osher 1988) with an adaptive CFL step; no linear solver is needed.
#
# The energy row of the gas-dynamics state is carried along passively: the closure
# below makes the pressure depend on $h$ only, so it never feeds back.
#
# **Reference.** (i) Until the bore reaches the column ($t<0.265$ s) the flow is
# exactly one-dimensional and equals Stoker's exact wet-bed dam-break solution
# (J. J. Stoker, *Water Waves*, Interscience 1957, Ch. 10; the dry-bed limit is
# Ritter's solution); the 2D field is compared with it over the whole tank.
# (ii) At the instant of impact the straight bore is reflected by the flat front face;
# the exact reflected depth gives the closed-form peak force. (iii) Mass is conserved
# to round-off. (iv) A three-grid study gives observed orders and Richardson estimates
# for the force history. The first impact converges well; the second impact, carried
# by bores that cross the nearly dry wake, still shifts by about 0.02 s per refinement
# on these grids. Tabulated measurements of the cited experiment are not
# available, so the measured forces are not compared.

from importlib.metadata import version
import time
from collections import Counter
from dataclasses import dataclass

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
from matplotlib.patches import Rectangle
from scipy.optimize import brentq

from onesim.physics.hydrodynamics import (CompressibleBoundaryPatch, CompressibleFaceFiniteVolume,
                                          ImpermeableCompressibleWall, UnstructuredCompressibleGeometry)
from onesim.physics.hydrodynamics.incompressible.numerics.operators.equations.navier_stokes.compressible_navier_stokes import (
    CompressibleNavierStokesSolver,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters

g, rho_w = 9.81, 1000.0            # m/s^2, kg/m^3
width, length = 0.61, 1.60         # m, tank (x across, y along)
y_gate = 0.40                      # m, water body occupies 0 <= y <= 0.40
column = (0.24, 0.90, 0.12)        # m, lower-left corner (x, y) and side of the column
h_up, h_bed = 0.30, 0.01           # m, initial depths behind and in front of the gate
t_end = 2.0                        # s
grids = (0.02, 0.01, 0.005)        # m, target cell size
x0, y0, side = column

# Stoker's wet-bed dam-break solution: a rarefaction into the reservoir, a uniform
# middle state (h_m, u_m) and a bore of speed S into the thin layer.
f_bore = lambda h: (h - h_bed) * np.sqrt(0.5 * g * (h + h_bed) / (h * h_bed))
f_rare = lambda h: 2 * (np.sqrt(g * h) - np.sqrt(g * h_up))
h_mid = brentq(lambda h: f_rare(h) + f_bore(h), h_bed, h_up)
u_mid = -f_rare(h_mid)
bore_speed = h_mid * u_mid / (h_mid - h_bed)
t_arrival = (y0 - y_gate) / bore_speed
# Reflection of the bore at the front face: shock back into the moving middle state.
h_reflected = brentq(lambda h: (h - h_mid) * np.sqrt(0.5 * g * (h + h_mid) / (h * h_mid)) - u_mid, h_mid, 1.0)
F_impact = rho_w * g * side * (h_reflected**2 - h_bed**2) / 2
print(f"Stoker middle state h_m = {h_mid:.5f} m, u_m = {u_mid:.5f} m/s, bore speed {bore_speed:.5f} m/s")
print(f"bore reaches the column at t = {t_arrival:.4f} s; reflected depth {h_reflected:.5f} m, "
      f"impact force {F_impact:.3f} N")


def stoker(y, t):
    """Exact depth of the 1D dam break at time t (gate at y_gate, flow towards +y)."""
    c_up, c_mid = np.sqrt(g * h_up), np.sqrt(g * h_mid)
    s = (y - y_gate) / t
    return np.where(s < -c_up, h_up, np.where(s < u_mid - c_mid, (2 * c_up - s) ** 2 / (9 * g),
                                               np.where(s < bore_speed, h_mid, h_bed)))


# ## Geometry and mesh
#
# The tank is covered by a tensor grid of quadrilaterals whose lines include the gate,
# the four column faces and the walls; cells inside the column are removed, so the
# column faces are exact mesh boundaries. Two boundary groups are declared: `tank` (the
# four walls) and `column` (the four column faces). The grid and its named exterior
# facets are handed to `UnstructuredCompressibleGeometry`, which builds the face
# normals, areas, cell volumes and least-squares gradient operators.

def axis_nodes(keys, h):
    nodes = [keys[0]]
    for a, b in zip(keys[:-1], keys[1:]):
        n = int(np.ceil((b - a) / h - 1e-9))
        nodes += list(np.linspace(a, b, n + 1)[1:])
    return np.array(nodes)


def build_grid(h):
    xs = axis_nodes([0.0, x0, x0 + side, width], h)
    ys = axis_nodes([0.0, y_gate, y0, y0 + side, length], h)
    nx, ny = len(xs) - 1, len(ys) - 1
    I, J = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy"))
    xc, yc = 0.5 * (xs[I] + xs[I + 1]), 0.5 * (ys[J] + ys[J + 1])
    fluid = ~((xc > x0) & (xc < x0 + side) & (yc > y0) & (yc < y0 + side))
    I, J = I[fluid], J[fluid]
    vid = lambda i, j: j * (nx + 1) + i
    cells = np.stack((vid(I, J), vid(I + 1, J), vid(I + 1, J + 1), vid(I, J + 1)), axis=1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.column_stack((X.ravel(), Y.ravel()))
    used = np.unique(cells)
    renumber = np.full(len(vertices), -1)
    renumber[used] = np.arange(len(used))
    vertices, cells = vertices[used], renumber[cells]
    edges = Counter(tuple(sorted((int(c[k]), int(c[(k + 1) % 4])))) for c in cells for k in range(4))
    groups = {}
    for edge, count in edges.items():
        if count == 1:
            mx, my = vertices[list(edge)].mean(axis=0)
            on_column = (x0 - 1e-9 <= mx <= x0 + side + 1e-9) and (y0 - 1e-9 <= my <= y0 + side + 1e-9)
            groups[edge] = "column" if on_column else "tank"
    geometry = UnstructuredCompressibleGeometry(artifact_id=f"dam-break-tank-{h}", vertices=vertices, cells=cells,
                                                boundary_groups=groups, cell_quadrature_order=2,
                                                face_quadrature_order=1)
    return dict(geometry=geometry, cells=geometry(geometry.carrier), xs=xs, ys=ys, I=I, J=J,
                h_min=min(np.diff(xs).min(), np.diff(ys).min()))


# ## Model
#
# **Closure.** Depth plays the role of density and $p=g h^2/2$ (per unit water
# density) the role of pressure; the sound speed is the wave celerity $\sqrt{g h}$.
# The specific internal energy is offset by a large constant so that the passive
# energy row stays admissible.

E_OFFSET, H_DRY = 1.0e6, 1.0e-3


@dataclass(frozen=True)
class HydrostaticClosure:
    gravity: float = g
    offset: float = E_OFFSET

    def pressure_from_density_internal_energy(self, h, e):
        return 0.5 * self.gravity * h * h

    def evaluate_density_internal_energy_fields(self, h, e):
        return 0.5 * self.gravity * h * h, np.ones_like(h)

    def temperature_from_density_internal_energy(self, h, e):
        return 1.0

    def internal_energy_from_density_pressure(self, h, p):
        return self.offset + p / h

    def evaluate_internal_energy_pressure_field(self, h, p):
        return self.offset + p / h

    def sound_speed(self, h, e):
        return np.sqrt(self.gravity * h)

    def evaluate_sound_speed_field(self, h, e):
        return np.sqrt(self.gravity * h)

    def pressure_from_density_temperature(self, h, T):
        return 0.5 * self.gravity * h * h

    def density_from_pressure_temperature(self, p, T):
        return np.sqrt(2 * p / self.gravity)

    def specific_enthalpy_from_density_pressure(self, h, p):
        return self.offset + 2 * p / h

    def specific_entropy(self, h, e):
        return 0.0


def velocity(h, hu):
    """Desingularised velocity hu/h, identical to hu/h for h >> H_DRY."""
    return np.sqrt(2.0) * h * hu / np.sqrt(h**4 + np.maximum(h**4, H_DRY**4))


def energy_row(h, u, v):
    return h * E_OFFSET + 0.5 * g * h * h + 0.5 * h * (u * u + v * v)


# **HLL flux.** The normal velocity of each face state follows from its mass flux;
# the outer wave speeds use the Einfeldt (Roe-average) estimates.

def hll_flux(left, right, flux_left, flux_right, wave):
    hl, hr = left[0], right[0]
    ul, ur = flux_left[0] / hl, flux_right[0] / hr
    cl, cr = np.sqrt(g * hl), np.sqrt(g * hr)
    u_roe = (np.sqrt(hl) * ul + np.sqrt(hr) * ur) / (np.sqrt(hl) + np.sqrt(hr))
    c_roe = np.sqrt(0.5 * g * (hl + hr))
    s_left = np.minimum(np.minimum(ul - cl, u_roe - c_roe), 0.0)
    s_right = np.maximum(np.maximum(ur + cr, u_roe + c_roe), 0.0)
    return (s_right * flux_left - s_left * flux_right + s_left * s_right * (right - left)) / (s_right - s_left)


# **Walls.** A slip wall is a mirror state; the exact Riemann solution between a cell
# state with normal velocity $u_n$ and its mirror image has $u^*=0$ and the depth
# $h^*=(\sqrt{gh}+u_n/2)^2/g$ for $u_n\le0$ (rarefaction) or the root of
# $(h^*-h)\sqrt{g(h^*+h)/(2h^*h)}=u_n$ for $u_n>0$ (reflected bore). The wall flux is
# the hydrostatic thrust $g h^{*2}/2\,\mathbf n$, and the same $h^*$ gives the force on
# the column.

def wall_depth(h, un):
    hs = np.maximum(np.sqrt(g * h) + 0.5 * np.minimum(un, 0.0), 0.0) ** 2 / g
    bore = un > 0
    x, hb, ub = h[bore] + un[bore] * np.sqrt(h[bore] / g), h[bore], un[bore]
    for _ in range(30):                                  # Newton iteration for the reflected bore
        q = np.sqrt(0.5 * g * (x + hb) / (x * hb))
        x = x - ((x - hb) * q - ub) / (q - (x - hb) * 0.25 * g / (q * x * x))
    hs[bore] = x
    return hs


def wall_flux(interior, trace, normal, points, time, turbulence, default):
    h = np.asarray(interior.density)
    un = velocity(h, normal @ (np.asarray(interior.velocity) * h))
    thrust = 0.5 * g * wall_depth(h, un) ** 2
    flux = np.zeros_like(default)
    flux[1], flux[2] = thrust * normal[0], thrust * normal[1]
    return flux


# **MUSCL reconstruction.** Cell gradients of $(h,u,v)$ come from the geometry's
# least-squares operator; the Barth-Jespersen limiter keeps every face value within
# the range of the cell and its face neighbours. Cells touching a nearly dry cell
# ($h<1$ mm) use first-order (piecewise-constant) values.

def muscl(cells, pressure, *, balance):
    geo, inside = balance.geometry, balance.internal
    owner, neighbour = geo.owner[inside], geo.neighbour[inside]
    h = cells[0]
    prim = np.stack((h, velocity(h, cells[1]), velocity(h, cells[2])))
    grads = np.stack([np.stack(balance.solver.operators.gradient(q.reshape(geo.shape))).reshape(2, -1)
                      for q in prim])
    lo, hi = prim.copy(), prim.copy()
    for k in range(3):
        np.minimum.at(lo[k], owner, prim[k, neighbour])
        np.minimum.at(lo[k], neighbour, prim[k, owner])
        np.maximum.at(hi[k], owner, prim[k, neighbour])
        np.maximum.at(hi[k], neighbour, prim[k, owner])
    limiter = np.ones_like(prim)
    d_owner = geo.points[inside] - geo.coordinates[owner]
    d_neighbour = geo.points[inside] - geo.coordinates[neighbour]
    for idx, offset in ((owner, d_owner), (neighbour, d_neighbour)):
        for k in range(3):
            delta = np.einsum("dq,qd->q", grads[k][:, idx], offset)
            room = np.where(delta > 0, hi[k, idx] - prim[k, idx], lo[k, idx] - prim[k, idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(delta) > 1e-300, room / delta, 1.0)
            np.minimum.at(limiter[k], idx, np.clip(ratio, 0.0, 1.0))
    limiter[:, lo[0] < H_DRY] = 0.0
    grads = grads * limiter[:, None, :]

    def face_state(idx, offset):
        hh, uu, vv = prim[:, idx] + np.einsum("kdq,qd->kq", grads[:, :, idx], offset)
        hh = np.maximum(hh, 1e-300)
        return np.stack((hh, hh * uu, hh * vv, energy_row(hh, uu, vv))), 0.5 * g * hh * hh

    left, p_left = face_state(owner, d_owner)
    right, p_right = face_state(neighbour, d_neighbour)
    return (left, right), (p_left, p_right)


def cfl_step(cfl, h_min):
    def step(state, *, time, species=None, balance):
        h = state[0]
        c = np.sqrt(g * h)
        rate = (np.abs(velocity(h, state[1])) + c + np.abs(velocity(h, state[2])) + c) / h_min
        return cfl / np.max(rate)
    return step


def make_solver(grid, second_order=True, cfl=0.45):
    wall = CompressibleBoundaryPatch(ImpermeableCompressibleWall(), advective_flux=wall_flux)
    balance = CompressibleFaceFiniteVolume(
        geometry_factory=grid["geometry"], boundaries={"tank": wall, "column": wall}, numerical_flux=hll_flux,
        reconstruction=muscl if second_order else None, stability=cfl_step(cfl, grid["h_min"]), cfl=0.5)
    return CompressibleNavierStokesSolver(
        dimension=2, viscosity=0.0, thermal_conductivity=0.0, density=1.0, pressure=1.0, final_time=0.01,
        time_step=1.0, equation_of_state=HydrostaticClosure(), spatial_balance_factory=balance,
        cell_geometry=grid["cells"], spatial_carrier=grid["geometry"].carrier, time_method="ssprk3")


def initial_state(grid):
    y = grid["cells"].coordinates[:, 1]
    h = np.where(y < y_gate, h_up, h_bed)
    return np.stack((h, 0 * h, 0 * h, energy_row(h, 0 * h, 0 * h)))


def column_force(grid, state):
    """y-force on the column from the wall depth h* on its four faces (N)."""
    cg = grid["cells"]
    on_column = np.array([key == "column" for key in cg.boundary_ids])
    owner, normal, area = cg.owner[on_column], cg.normals[on_column], cg.areas[on_column]
    h = state[0][owner]
    un = velocity(h, normal[:, 0] * state[1][owner] + normal[:, 1] * state[2][owner])
    return rho_w * np.sum(0.5 * g * wall_depth(h, un) ** 2 * area * normal[:, 1])


# ## Solve
#
# The solution is advanced in output intervals (5 ms, refined to 1 ms around the two
# impacts); after each interval the column force is evaluated. The depth is stored at
# $t=0.20$ s for the comparison with Stoker's solution and at four display times on
# the finest grid.

t_out = np.unique(np.round(np.r_[np.arange(0.0, t_end + 1e-9, 0.005), np.arange(0.255, 0.330, 0.001),
                                 np.arange(1.250, 1.450, 0.001), 0.20], 9))
t_maps = (0.22, 0.38, 0.76, 1.44)


def run(h, second_order=True, until=t_end):
    grid = build_grid(h)
    solver = make_solver(grid, second_order)
    state = initial_state(grid)
    volume0 = np.sum(state[0] * grid["cells"].volumes)
    times, forces, maps, steps, now = [0.0], [column_force(grid, state)], {}, 0, 0.0
    start = time.time()
    for t in t_out[(t_out > 0) & (t_out <= until + 1e-12)]:
        solver.final_time = t - now
        result = solver.run(initial_state=state)
        state, now, steps = result.conserved_state, t, steps + result.step_count
        times.append(t)
        forces.append(column_force(grid, state))
        if any(abs(t - m) < 1e-9 for m in (0.20, *t_maps)):
            maps[round(t, 3)] = state[0].copy()
    return dict(grid=grid, times=np.array(times), force=np.array(forces), maps=maps, steps=steps,
                volume_error=np.sum(state[0] * grid["cells"].volumes) / volume0 - 1.0, seconds=time.time() - start,
                cells=len(grid["cells"].volumes))


runs = []
for h in grids:
    out = run(h)
    runs.append(out)
    print(f"grid {h * 1e3:4.1f} mm: {out['cells']:6d} cells, {out['steps']:5d} SSP-RK3 steps, "
          f"peak force {out['force'].max():6.2f} N, relative volume change {out['volume_error']:+.1e} "
          f"({out['seconds']:.0f} s)")
first_order = run(0.01, second_order=False, until=0.6)
print(f"first-order Godunov (no MUSCL) on the 10 mm grid up to 0.6 s: {first_order['steps']} steps "
      f"({first_order['seconds']:.0f} s)")


# ## Comparison with the reference
#
# **Stoker's solution at $t=0.20$ s.** The $L^1$ error of the depth over the whole
# tank, $\sum|h_i-h_\mathrm{exact}(y_i)|A_i/A$, contains the rarefaction (smooth) and
# the bore (a discontinuity, first-order in $L^1$ at best), so an observed order
# between 1 and 2 is expected.

def stoker_error(out):
    cg = out["grid"]["cells"]
    exact = stoker(cg.coordinates[:, 1], 0.20)
    return np.sum(np.abs(out["maps"][0.2] - exact) * cg.volumes) / np.sum(cg.volumes)


errors = [stoker_error(out) for out in runs]
print("Depth at t = 0.20 s vs Stoker's exact dam-break solution (L1 error, m)")
for h, err in zip(grids, errors):
    print(f"  grid {h * 1e3:4.1f} mm: {err:.3e}")
print(f"  observed orders {np.log2(errors[0] / errors[1]):.2f} and {np.log2(errors[1] / errors[2]):.2f}")
print(f"  first-order Godunov on the 10 mm grid: {stoker_error(first_order):.3e} "
      f"({stoker_error(first_order) / errors[1]:.1f} x the MUSCL error)")

# **Impact force.** When the straight bore meets the flat front face, the depth there
# jumps to the exact reflected depth while the back face still sees the 1 cm layer,
# so the force is $\rho g\,b\,(h_r^2-h_0^2)/2$ at that instant. Lateral relief from the
# column edges lowers it immediately afterwards, so a finite grid, on which the bore
# is spread over a few cells, approaches this value from below.


def richardson(values, ratio=2.0):
    """Observed order and extrapolated value from three grids refined by `ratio`."""
    f1, f2, f3 = values
    r = (f1 - f2) / (f2 - f3)
    p = np.log(r) / np.log(ratio) if r > 1 else np.nan
    return p, (f3 + (f3 - f2) / (ratio**p - 1.0) if r > 1 else np.nan)


def window_impulse(out, t0, t1):
    t, F = out["times"], out["force"]
    keep = (t >= t0) & (t <= t1)
    return np.trapezoid(F[keep], t[keep])


print(f"Peak force at the first impact (closed form {F_impact:.3f} N at t = {t_arrival:.4f} s)")
for h, out in zip(grids, runs):
    i = np.argmax(out["force"])
    print(f"  grid {h * 1e3:4.1f} mm: {out['force'][i]:7.3f} N at t = {out['times'][i]:.3f} s, "
          f"deviation {100 * (out['force'][i] / F_impact - 1):+.2f} %")
i = np.argmax(first_order["force"])
print(f"  first-order Godunov, 10 mm: {first_order['force'][i]:7.3f} N "
      f"({100 * (first_order['force'][i] / F_impact - 1):+.2f} %)")

# **Grid study of the force history.** The impulse of the first impact is a smooth
# functional of the solution. The second impact is caused by bores that travel round
# the column over the nearly dry wake and back from the far wall; its extreme force and
# its onset time (first drop below -5 N) are listed as well.


def second_impact(out):
    t, F = out["times"], out["force"]
    late = t > 1.2
    return F[late].min(), t[late][np.argmax(F[late] < -5.0)]


print("Grid study of the force history")
quantities = (("impulse of the first impact, 0.25-0.80 s (N s)", [window_impulse(out, 0.25, 0.80) for out in runs]),
              ("extreme force of the second impact (N)", [second_impact(out)[0] for out in runs]),
              ("onset of the second impact (s)", [second_impact(out)[1] for out in runs]))
for label, vals in quantities:
    p, ext = richardson(vals)
    trend = f"order {p:4.2f}, Richardson {ext:9.4f}" if np.isfinite(p) else "not in the asymptotic range, no extrapolation"
    print(f"  {label:50s} " + "  ".join(f"{v:9.4f}" for v in vals)
          + f"; {trend}, change between the last two grids {100 * abs(vals[2] / vals[1] - 1):.2f} %")

print("Mass conservation (relative change of the water volume at t = 2 s)")
for h, out in zip(grids, runs):
    print(f"  grid {h * 1e3:4.1f} mm: {out['volume_error']:+.2e}")


# ## Figure
#
# Top: water depth on the finest grid at four times (impact, wrap-around, reflection
# from the far wall, second impact). Bottom left: depth along the tank at $t=0.20$ s
# against Stoker's solution. Bottom right: force on the column on the three grids with
# the closed-form impact force.

fine = runs[-1]
grid = fine["grid"]
xc = 0.5 * (grid["xs"][:-1] + grid["xs"][1:])
yc = 0.5 * (grid["ys"][:-1] + grid["ys"][1:])
fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
layout = fig.add_gridspec(2, 5, height_ratios=(1.55, 1.0), width_ratios=(1, 1, 1, 1, 0.07))
depth_levels = np.linspace(0.0, 0.30, 46)
for k, t in enumerate(t_maps):
    ax = fig.add_subplot(layout[0, k])
    field = np.ma.masked_all((len(yc), len(xc)))
    field[grid["J"], grid["I"]] = fine["maps"][t]
    filled = ax.contourf(xc, yc, field, levels=depth_levels, cmap=FIELD_CMAP, extend="max")
    ax.contour(xc, yc, field, levels=depth_levels[::4], colors="w", linewidths=0.3, alpha=0.6)
    ax.add_patch(Rectangle((x0, y0), side, side, color="0.35"))
    ax.set(aspect="equal", xlim=(0, width), ylim=(0, length), xlabel="x / m", title=f"t = {t:.2f} s")
    if k == 0:
        ax.set_ylabel("y / m (gate at 0.40 m)")
fig.colorbar(filled, cax=fig.add_subplot(layout[0, 4]), label="water depth h / m")

ax = fig.add_subplot(layout[1, :2])
y_line = np.linspace(0.0, y0, 600)
ax.plot(y_line, stoker(y_line, 0.20), color=INK, lw=1.5, label="Stoker, exact")
for out, h, color in zip(runs, grids, ("#CD9561", GREEN, INK)):
    cg = out["grid"]["cells"]
    strip = np.abs(cg.coordinates[:, 0] - 0.5 * width) < 0.6 * h + 1e-9
    order = np.argsort(cg.coordinates[strip, 1])
    ax.plot(cg.coordinates[strip, 1][order], out["maps"][0.2][strip][order], ".-", ms=3, lw=0.8, color=color,
            label=f"MUSCL-HLL, {h * 1e3:.0f} mm")
ax.set(xlabel="y / m", ylabel="h / m", title="Depth along the tank at t = 0.20 s (before impact)", xlim=(0, y0))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = fig.add_subplot(layout[1, 2:5])
for out, h, color in zip(runs, grids, ("#CD9561", GREEN, INK)):
    ax.plot(out["times"], out["force"], color=color, lw=1.2, label=f"MUSCL-HLL, {h * 1e3:.0f} mm")
ax.plot(first_order["times"], first_order["force"], color="0.5", lw=1.0, ls="--", label="first-order Godunov, 10 mm")
ax.plot([t_arrival], [F_impact], "*", color=INK, ms=12, label=f"exact impact force {F_impact:.1f} N")
ax.axhline(0.0, color="0.6", lw=0.6)
ax.set(xlabel="t / s", ylabel="F_y on the column / N", title="Pressure force on the column", xlim=(0, t_end))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Add Manning bed friction as a momentum source to see how much of the peak force the
# frictionless model over-predicts, move the column closer to the side wall, or replace
# the column by a raised bed block that the water can overtop. The same injected
# closure runs unchanged on triangular or mixed meshes for curved obstacles.
