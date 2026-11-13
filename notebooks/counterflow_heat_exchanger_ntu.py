"""OneSimulation example: counterflow heat exchanger against the effectiveness-NTU relation.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form counterflow effectiveness-NTU solution.
"""


# # Counterflow heat exchanger
#
# A hot and a cold water stream flow in opposite directions through two thin
# parallel channels (gap $t$, width $w$, length $L$) separated by a plastic
# wall of thickness $t_w$ and conductivity $k_w$. Heat crosses the wall from the
# hot to the cold stream; the outer faces of both channels are insulated.
#
# OneSimulation solves the steady material-temperature balance
# $\rho c_p\,\mathbf u\!\cdot\!\nabla T=k\,\nabla^2 T$ with finite elements in
# each channel on a real 3D mesh (plug flow $+u_h$ in the hot, $-u_c$ in the
# cold channel). The two channels are coupled through the wall: on the hot
# channel's wall face the outward heat flux is $h_w\,(T_h-T_c)$ with the wall
# conductance $h_w=k_w/t_w$ and the cold channel's wall-face temperature as the
# "ambient" value, and vice versa. The two stationary problems are solved
# alternately until the outlet temperatures no longer change (a partitioned
# fixed-point iteration between the two streams).
#
# **Reference.** With the capacity rates $C_h=\rho c_p u_h t w$ and
# $C_c=\rho c_p u_c t w$, $C_\min=\min(C_h,C_c)$, $C_r=C_\min/C_\max$ and the
# number of transfer units $\mathrm{NTU}=UA/C_\min$ (heat transfer area
# $A=Lw$), the effectiveness of a counterflow exchanger is
#
# $$\varepsilon=\frac{1-e^{-\mathrm{NTU}(1-C_r)}}{1-C_r\,e^{-\mathrm{NTU}(1-C_r)}}
#   \qquad\Big(\varepsilon=\tfrac{\mathrm{NTU}}{1+\mathrm{NTU}}\ \text{for } C_r=1\Big),$$
#
# the transferred power is $q=\varepsilon\,C_\min\,(T_{h,in}-T_{c,in})$ and the
# outlets follow from $q=C_h(T_{h,in}-T_{h,out})=C_c(T_{c,out}-T_{c,in})$. Along
# the exchanger the local temperature difference decays exponentially,
# $T_h-T_c=\Delta_0\,e^{-m x}$ with $m=Uw\,(1/C_h-1/C_c)$ and
# $\Delta_0=T_{h,in}-T_{c,out}$, which gives closed-form bulk profiles
# $T_h(x)$ and $T_c(x)$.
#
# The overall coefficient combines the wall with the two fluid films. For
# laminar plug flow in a slab that is heated through one face and insulated on
# the other, the developed bulk-to-face temperature difference is
# $q''\,t/(3k_f)$ (the plug-flow Nusselt number is 3), so
#
# $$\frac1U=\frac{t_w}{k_w}+\frac{2\,t}{3\,k_f}.$$
#
# The film part is resolved by the 3D mesh; only the wall enters as $h_w$.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux, ConvectiveHeatTransfer, FixedTemperature, HeatFlux,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# A small laboratory plate exchanger with slow laminar water flow: 1 mm gaps,
# 45 mm long, separated by a 1.2 mm polypropylene wall. The hot stream carries
# the smaller capacity rate ($C_r=0.6$) and the design point is NTU = 1.5.

L, t, w = 0.045, 1.0e-3, 2.0e-3      # m, channel length, gap and width
t_w, k_w = 1.2e-3, 0.12               # m, W/(m K), polypropylene wall
rho, cp, k_f = 1000.0, 4180.0, 0.6    # water
T_h_in, T_c_in = 350.0, 290.0         # K, inlet temperatures
NTU, C_r = 1.5, 0.6                   # design point

h_wall = k_w / t_w                                   # W/(m^2 K), wall conductance
U = 1.0 / (1.0 / h_wall + 2.0 * t / (3.0 * k_f))     # W/(m^2 K), overall coefficient
area = L * w
C_min = U * area / NTU                               # W/K, hot stream (smaller capacity rate)
C_max = C_min / C_r                                  # W/K, cold stream
u_h, u_c = C_min / (rho * cp * t * w), C_max / (rho * cp * t * w)
peclet = rho * cp * u_h * L / k_f
print(f"U = {U:.2f} W/(m^2 K) (wall {h_wall:.1f}, films {3*k_f/t:.0f} each),  "
      f"u_h = {u_h*1e3:.3f} mm/s,  u_c = {u_c*1e3:.3f} mm/s,  axial Peclet number {peclet:.0f}")


def effectiveness(ntu, cr):
    if abs(1.0 - cr) < 1e-12:
        return ntu / (1.0 + ntu)
    e = np.exp(-ntu * (1.0 - cr))
    return (1.0 - e) / (1.0 - cr * e)


eps = effectiveness(NTU, C_r)
q_ref = eps * C_min * (T_h_in - T_c_in)
T_h_out_ref = T_h_in - q_ref / C_min
T_c_out_ref = T_c_in + q_ref / C_max
print(f"effectiveness = {eps:.4f},  q = {q_ref:.4f} W,  "
      f"outlets: hot {T_h_out_ref:.3f} K,  cold {T_c_out_ref:.3f} K")


# ## Model
#
# Each channel is a slender box meshed with one quadratic element across the
# gap (enough for the parabolic film profile) and 80 elements along the flow.
# Every face carries an explicit condition: fixed inlet temperature, zero
# conductive flux at the outlet and on the insulated faces, and the wall face
# exchanges heat with the other stream.

cells = (80, 1, 1)
material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k_f,
    youngs_modulus=2e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
hot_box, cold_box = Box(length=L, width=t, height=w), Box(length=L, width=t, height=w)
hot_mesh = StructuredHexahedralMesher(cells=cells).generate(hot_box)
cold_mesh = StructuredHexahedralMesher(cells=cells).generate(cold_box)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2), convection=GalerkinAdvection()),
    linear_solver=SparseDirect(),
)
hot_wall_y, cold_wall_y = t, 0.0          # wall face of each channel (y_max of the hot, y_min of the cold box)


def solve_channel(box, mesh, velocity, inlet, outlet, wall, insulated, inlet_temperature, ambient):
    model = ConvectiveHeatTransfer(domain=box.volume, material=material, velocity=(velocity, 0.0, 0.0))
    problem = Problem(
        name="exchanger channel", geometry=box, physics=(model,),
        boundary_conditions=(
            FixedTemperature(boundary=inlet, physics=model, value=inlet_temperature),
            HeatFlux(boundary=outlet | insulated, physics=model, value=0.0),
            ConvectiveHeatFlux(boundary=wall, physics=model,
                               heat_transfer_coefficient=h_wall, ambient_temperature=ambient),
        ),
    )
    simulation = Simulation(problem=problem, mesh=mesh, numerics=numerics)
    simulation.validate().require_valid()
    return simulation.solve_stationary().field(model.temperature)


def face_temperature(field, y_face):
    # The other stream's wall-face temperature as a function of the axial position.
    def ambient(x):
        x = np.asarray(x, dtype=float)
        points = np.column_stack((x.ravel(), np.full(x.size, y_face), np.full(x.size, w / 2)))
        return np.asarray(field.evaluate(points)).reshape(x.shape)
    return ambient


def bulk_temperature(field, x_positions):
    # Plug flow: the bulk temperature is the plain mean over the gap.
    y_nodes = np.linspace(0.0, t, 41)
    values = []
    for x in np.atleast_1d(x_positions):
        points = np.column_stack((np.full_like(y_nodes, x), y_nodes, np.full_like(y_nodes, w / 2)))
        values.append(float(np.trapezoid(np.asarray(field.evaluate(points)).reshape(-1), y_nodes) / t))
    return np.asarray(values)


hb, cb = hot_box.boundaries, cold_box.boundaries
hot_arguments = dict(box=hot_box, mesh=hot_mesh, velocity=u_h, inlet=hb.x_min, outlet=hb.x_max,
                     wall=hb.y_max, insulated=hb.y_min | hb.z_min | hb.z_max, inlet_temperature=T_h_in)
cold_arguments = dict(box=cold_box, mesh=cold_mesh, velocity=-u_c, inlet=cb.x_max, outlet=cb.x_min,
                      wall=cb.y_min, insulated=cb.y_max | cb.z_min | cb.z_max, inlet_temperature=T_c_in)

cold_ambient = lambda x: T_c_in + 0.0 * np.asarray(x)     # first guess: cold stream at its inlet temperature
outlets, sweeps = (np.inf, np.inf), 0
for sweep in range(1, 41):
    hot_field = solve_channel(**hot_arguments, ambient=cold_ambient)
    cold_field = solve_channel(**cold_arguments, ambient=face_temperature(hot_field, hot_wall_y))
    cold_ambient = face_temperature(cold_field, cold_wall_y)
    new = (bulk_temperature(hot_field, L)[0], bulk_temperature(cold_field, 0.0)[0])
    change = max(abs(a - b) for a, b in zip(new, outlets))
    outlets, sweeps = new, sweep
    print(f"sweep {sweep:2d}: hot outlet {new[0]:.4f} K, cold outlet {new[1]:.4f} K, change {change:.2e} K")
    if change < 1e-4:
        break
assert change < 1e-4, "The alternating solution of the two streams did not converge."
T_h_out, T_c_out = outlets


# ## Comparison with the effectiveness-NTU solution
#
# Outlet temperatures, the transferred power on both sides and the bulk profiles
# along the exchanger are compared with the closed form.

dT_in = T_h_in - T_c_in
err_h, err_c = abs(T_h_out - T_h_out_ref), abs(T_c_out - T_c_out_ref)
q_hot, q_cold = C_min * (T_h_in - T_h_out), C_max * (T_c_out - T_c_in)
print(f"Hot outlet:  FEM {T_h_out:.3f} K,  eps-NTU {T_h_out_ref:.3f} K  ({100*err_h/dT_in:.2f} % of the inlet difference)")
print(f"Cold outlet: FEM {T_c_out:.3f} K,  eps-NTU {T_c_out_ref:.3f} K  ({100*err_c/dT_in:.2f} % of the inlet difference)")
print(f"Transferred power: hot side {q_hot:.4f} W,  cold side {q_cold:.4f} W,  eps-NTU {q_ref:.4f} W")
print(f"Effectiveness: FEM {q_hot/(C_min*dT_in):.4f},  closed form {eps:.4f}")
assert err_h < 0.01 * dT_in and err_c < 0.01 * dT_in, "Outlet temperatures deviate by more than 1 % of the inlet difference."
assert abs(q_hot - q_cold) < 0.01 * q_ref, "The two streams must exchange the same power."

# Bulk temperature profiles along the exchanger.
m_decay = U * w * (1.0 / C_min - 1.0 / C_max)
delta_0 = T_h_in - T_c_out_ref


def reference_profiles(x):
    x = np.asarray(x, dtype=float)
    integral = (1.0 - np.exp(-m_decay * x)) / m_decay if abs(m_decay) > 1e-12 else x
    hot = T_h_in - U * w / C_min * delta_0 * integral
    return hot, hot - delta_0 * np.exp(-m_decay * x)


sample_x = np.linspace(0.0, L, 31)
hot_profile, cold_profile = bulk_temperature(hot_field, sample_x), bulk_temperature(cold_field, sample_x)
hot_ref, cold_ref = reference_profiles(sample_x)
profile_error = max(float(np.max(np.abs(hot_profile - hot_ref))), float(np.max(np.abs(cold_profile - cold_ref))))
print(f"Largest deviation of the bulk profiles from the closed form: {profile_error:.3f} K  ({100*profile_error/dT_in:.2f} %)")
assert profile_error < 0.015 * dT_in, "Temperature profiles deviate by more than 1.5 % of the inlet difference."
# Physical sanity: the hot stream cools along +x, the cold stream warms along -x, and they never cross.
assert np.all(np.diff(hot_profile) < 0) and np.all(np.diff(cold_profile) < 0)
assert np.all(hot_profile > cold_profile)


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
dense_x = np.linspace(0.0, L, 300)
dense_hot, dense_cold = reference_profiles(dense_x)
ax.plot(dense_x * 1000, dense_hot, color="#A85539", lw=1.7, label="closed form, hot stream (flows to the right)")
ax.plot(dense_x * 1000, dense_cold, color="#3f6b8a", lw=1.7, label="closed form, cold stream (flows to the left)")
ax.plot(sample_x * 1000, hot_profile, "o", color="#c56838", ms=4, label="OneSimulation, hot bulk temperature")
ax.plot(sample_x * 1000, cold_profile, "s", color="#5b8fb3", ms=4, label="OneSimulation, cold bulk temperature")
ax.set(xlabel="Position along the exchanger x / mm", ylabel="Bulk temperature / K",
       title=f"Counterflow heat exchanger: NTU = {NTU:.1f}, C_r = {C_r:.1f}, effectiveness {eps:.3f}")
ax.legend(loc="center right", fontsize=9)
plt.show()


# ## Going further
#
# Make the two capacity rates equal ($C_r=1$): the temperature difference is
# then uniform along the exchanger and both profiles become straight lines.
# Reverse the cold stream to obtain a parallel-flow exchanger, whose
# effectiveness $\big(1-e^{-\mathrm{NTU}(1+C_r)}\big)/(1+C_r)$ is lower; or raise
# the flow rates until the axial Peclet number demands a finer mesh along the
# channel.
