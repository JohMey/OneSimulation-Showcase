"""OneSimulation example: Womersley pulsatile flow between parallel plates.
Requires the installed, activated OneSimulation runtime.

The periodic velocity profiles computed by OneSimulation are checked against the
closed-form Womersley solution for an oscillating pressure gradient.
"""


# # Womersley pulsatile flow between parallel plates
#
# A viscous fluid fills the gap between two fixed parallel plates a distance $2h$
# apart. A streamwise pressure gradient that oscillates harmonically in time,
# $-\partial p/\partial x = G\cos\omega t$ (applied here as an equivalent body
# force), drives a unidirectional flow $u=u(y,t)$. With $y$ measured from the
# channel centre, the Navier-Stokes momentum balance reduces to
#
# $$\rho\,\frac{\partial u}{\partial t}=G\cos\omega t+\mu\,\frac{\partial^2 u}{\partial y^2},
#   \qquad u(\pm h,t)=0 .$$
#
# OneSimulation integrates the full 3D time-dependent Navier-Stokes system with
# inf-sup stable Taylor-Hood finite elements (quadratic velocity, linear pressure),
# a second-order BDF time integrator and a Newton solver per time step. The flow is
# started from rest and settles into its periodic state after a few cycles.
#
# **Reference.** The periodic state is the Womersley solution. Writing
# $u=\operatorname{Re}\{U(y)\,e^{i\omega t}\}$ gives
#
# $$U(y)=\frac{G}{i\rho\omega}\left[1-\frac{\cosh(k y)}{\cosh(k h)}\right],
#   \qquad k=\sqrt{\frac{i\omega}{\nu}},$$
#
# governed by the Womersley number $\alpha=h\sqrt{\omega/\nu}$. For $\alpha\ll1$ the
# profile is a quasi-steady parabola in phase with the forcing; for $\alpha$ of
# order 3 and above the core is flattened and lags the pressure gradient while the
# near-wall layers (Stokes layer thickness $\delta=\sqrt{2\nu/\omega}$) respond
# first. We check the profiles at four phases of the last cycle, the amplitude and
# phase lag of the centreline velocity, and the transverse velocity components.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import PeriodicPair, StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.hydrodynamics import IncompressibleNavierStokes, NoSlipWall
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions.values import InitialValue
from onesim.simulation.problem_definition.boundaries import PeriodicBoundary

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Water in a 2 mm gap. The angular frequency is chosen from the Womersley number
# $\alpha=3$, which puts the Stokes layer at roughly half the half-gap, so the
# profile is clearly not parabolic. The start-up transient decays like
# $\exp(-\nu\pi^2 t/4h^2)$, i.e. by a factor $\exp(-\pi^3/2\alpha^2)$ per cycle; after
# three cycles less than 1 % of it is left, so the fourth cycle is compared with the
# periodic reference.

length = 0.0005       # m, streamwise extent (x)
gap = 0.002           # m, plate spacing 2h (y), the only physically resolved direction
span = 0.0005         # m, spanwise extent (z)
rho = 1000.0          # kg/m^3, density of water
mu = 1.0e-3           # Pa·s, dynamic viscosity of water
nu = mu / rho         # m^2/s, kinematic viscosity
half_gap = gap / 2.0  # m, h

womersley = 3.0                                   # alpha = h sqrt(omega / nu)
omega = (womersley / half_gap) ** 2 * nu          # rad/s
period = 2.0 * np.pi / omega                      # s
forcing_amplitude = 1.0                           # N/m^3, G (pressure-gradient amplitude)
steps_per_period = 40
time_step = period / steps_per_period
compared_cycle = 3                                # compare during the fourth cycle (t in [3T, 3.75T])
end_time = (compared_cycle + 0.75) * period

stokes_layer = np.sqrt(2.0 * nu / omega)
transient_per_cycle = np.exp(-np.pi**3 / (2.0 * womersley**2))
print(f"alpha = {womersley:.1f},  omega = {omega:.2f} rad/s,  period T = {period:.3f} s")
print(f"Stokes layer delta = {1e3*stokes_layer:.3f} mm  (h = {1e3*half_gap:.1f} mm)")
print(f"Start-up transient left after 3 cycles: {100*transient_per_cycle**3:.2f} %")


# ## Model
#
# The plate spacing is the box width (its $y$ extent); both plates (y_min, y_max)
# are no-slip walls. The flow is independent of $x$ and $z$, so the streamwise
# (x_min/x_max) and spanwise (z_min/z_max) faces are paired periodically; every one
# of the six boundary faces has an explicit condition. The oscillating pressure
# gradient enters as the time-dependent body force $G\cos\omega t\,\hat e_x$, and the
# fluid starts from rest.

geometry = Box(length=length, width=gap, height=span)
streamwise = PeriodicPair(geometry.boundaries.x_min, geometry.boundaries.x_max)
spanwise = PeriodicPair(geometry.boundaries.z_min, geometry.boundaries.z_max)
mesh = StructuredTetrahedralMesher(
    cells=(1, 12, 1), periodic_pairs=(streamwise, spanwise),
).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")

fluid = NewtonianFluid(density=rho, dynamic_viscosity=mu)


def pulsatile_forcing(x, t):
    # Oscillating streamwise pressure gradient as a body force; x fixes the array shape.
    zeros = np.zeros_like(x)
    return np.column_stack((np.full_like(x, forcing_amplitude * np.cos(omega * t)), zeros, zeros))


def womersley_profile(y, t):
    # Closed-form periodic solution; y is measured from the channel centre.
    k = np.sqrt(1j * omega / nu)
    complex_amplitude = forcing_amplitude / (1j * rho * omega) * (1.0 - np.cosh(k * y) / np.cosh(k * half_gap))
    return np.real(complex_amplitude * np.exp(1j * omega * t))


model = IncompressibleNavierStokes(domain=geometry.volume, material=fluid, body_force=pulsatile_forcing)
problem = Problem(
    name="Womersley pulsatile channel flow",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        NoSlipWall(boundary=geometry.boundaries.y_min, physics=model),
        NoSlipWall(boundary=geometry.boundaries.y_max, physics=model),
        PeriodicBoundary(pair=streamwise, physics=model),
        PeriodicBoundary(pair=spanwise, physics=model),
    ),
    constraints=(MeanValue(model.pressure, 0.0),),
    initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
)
simulation = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
        linear_solver=SparseDirect(),
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(), absolute_tolerance=1e-12, relative_tolerance=0.0,
        ),
    ),
)
result = simulation.run(start_time=0.0, end_time=end_time, time_step=time_step)
times = np.asarray(result.times)
print(f"Integrated {len(times) - 1} implicit time steps to t = {times[-1]:.3f} s = {times[-1]/period:.2f} T")


# ## Comparison with the Womersley solution
#
# During the fourth cycle we sample the velocity along a wall-normal line at
# mid-domain at four phases ($\omega t = 0, \pi/2, \pi, 3\pi/2$ modulo $2\pi$) and
# compare the streamwise component with the closed-form profile. Errors are
# measured relative to the centreline velocity amplitude $|U(0)|$. The wall-normal
# and spanwise components must vanish and the no-slip condition must hold exactly.

y_samples = np.linspace(0.0, gap, 49)
line = np.column_stack((np.full_like(y_samples, length / 2),
                        y_samples,
                        np.full_like(y_samples, span / 2)))
velocity = result.field(model.velocity)

k_complex = np.sqrt(1j * omega / nu)
centre_complex = forcing_amplitude / (1j * rho * omega) * (1.0 - 1.0 / np.cosh(k_complex * half_gap))
centre_amplitude = float(abs(centre_complex))
centre_lag = float(-np.angle(centre_complex))            # rad, lag of u(0,t) behind G cos(omega t)
quasi_steady_amplitude = forcing_amplitude * half_gap**2 / (2.0 * mu)
print(f"Centreline amplitude |U(0)| = {1e3*centre_amplitude:.4f} mm/s "
      f"({centre_amplitude/quasi_steady_amplitude:.3f} of the quasi-steady value G h^2/2mu)")
print(f"Centreline phase lag behind the pressure gradient: {np.degrees(centre_lag):.1f} deg")

profiles = {}
for phase_fraction in (0.0, 0.25, 0.5, 0.75):
    target = (compared_cycle + phase_fraction) * period
    accepted = float(times[np.argmin(np.abs(times - target))])
    sampled = np.asarray(velocity.evaluate(line, time=accepted))
    u_numeric = sampled[:, 0]
    u_exact = womersley_profile(y_samples - half_gap, accepted)
    error = float(np.max(np.abs(u_numeric - u_exact))) / centre_amplitude
    transverse = float(np.max(np.abs(sampled[:, 1:]))) / centre_amplitude
    profiles[accepted] = (u_numeric, u_exact, error)
    print(f"omega t = {np.degrees(omega*accepted) % 360:5.1f} deg (t = {accepted:.3f} s): "
          f"max |u - u_exact| / |U(0)| = {100*error:.2f} %,  |v|,|w| / |U(0)| <= {transverse:.1e}")
    assert error < 0.03, f"Profile at t={accepted:.3f} s deviates more than 3 % of the centreline amplitude."
    assert transverse < 1e-4, "The flow must stay unidirectional."
    assert abs(u_numeric[0]) < 1e-12 and abs(u_numeric[-1]) < 1e-12, "No-slip must hold exactly at both plates."
    assert abs(u_numeric[len(u_numeric)//2] - u_exact[len(u_exact)//2]) < 0.03 * centre_amplitude


# ### Centreline amplitude and phase lag
#
# The centreline velocity over the compared part of the fourth cycle (three quarters
# of a period, 31 time levels) is projected onto $\cos\omega t$ and $\sin\omega t$,
# which gives its amplitude and phase; both are compared with the
# complex amplitude $U(0)$. The phase lag and the flattened core (centreline amplitude
# well below the quasi-steady value $Gh^2/2\mu$) are the signature of the Womersley regime.

cycle_times = times[times >= compared_cycle * period - 1e-9]
centre_point = np.array([[length / 2, half_gap, span / 2]])
centre_series = np.array([float(np.asarray(velocity.evaluate(centre_point, time=float(t)))[0, 0])
                          for t in cycle_times])
basis = np.column_stack((np.cos(omega * cycle_times), np.sin(omega * cycle_times)))
cos_part, sin_part = np.linalg.lstsq(basis, centre_series, rcond=None)[0]
amplitude_numeric = float(np.hypot(cos_part, sin_part))
lag_numeric = float(np.arctan2(sin_part, cos_part))
amplitude_error = abs(amplitude_numeric - centre_amplitude) / centre_amplitude
lag_error = abs(lag_numeric - centre_lag)
print(f"Centreline amplitude: FEM {1e3*amplitude_numeric:.4f} mm/s,  reference {1e3*centre_amplitude:.4f} mm/s "
      f"({100*amplitude_error:.2f} %)")
print(f"Centreline phase lag: FEM {np.degrees(lag_numeric):.2f} deg,  reference {np.degrees(centre_lag):.2f} deg "
      f"(difference {np.degrees(lag_error):.2f} deg)")
assert amplitude_error < 0.02, "Centreline amplitude off by more than 2 %."
assert lag_error < np.radians(2.0), "Centreline phase lag off by more than 2 degrees."
assert centre_lag > np.radians(20.0), "The chosen Womersley number should give a clear phase lag."
assert centre_amplitude < 0.8 * quasi_steady_amplitude, "The core should be noticeably below quasi-steady."


# ## Figure

cream = "#fbf5eb"
colors = ("#214D43", "#A85539", "#1f5c73", "#8a6d1f")
fig, (ax_profile, ax_centre) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax_profile, ax_centre):
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
for color, (accepted, (u_numeric, u_exact, _)) in zip(colors, profiles.items()):
    phase_deg = np.degrees(omega * accepted) % 360
    ax_profile.plot(u_exact * 1000, (y_samples - half_gap) * 1000, color=color, lw=1.8,
                    label=f"Womersley, ωt = {phase_deg:.0f}°")
    ax_profile.plot(u_numeric[::2] * 1000, (y_samples[::2] - half_gap) * 1000, "o", color=color, ms=4,
                    label=f"OneSimulation, ωt = {phase_deg:.0f}°")
ax_profile.axhline(half_gap * 1000, color="#555", lw=2)
ax_profile.axhline(-half_gap * 1000, color="#555", lw=2)
ax_profile.set(xlabel="Streamwise velocity u / (mm/s)", ylabel="Distance from the centre y / mm",
               title=f"Pulsatile channel flow, α = {womersley:.0f}: profiles in the fourth cycle")
ax_profile.legend(fontsize=7, ncol=2)
fine_times = np.linspace(cycle_times[0], cycle_times[-1], 200)
ax_centre.plot(fine_times / period, womersley_profile(0.0, fine_times) * 1000, color="#214D43", lw=1.8,
               label="Womersley centreline u(0,t)")
ax_centre.plot(cycle_times / period, centre_series * 1000, "o", color="#A85539", ms=4, label="OneSimulation")
ax_centre.plot(fine_times / period, quasi_steady_amplitude * np.cos(omega * fine_times) * 1000, "--",
               color="#888", lw=1.2, label="Quasi-steady response (α → 0)")
ax_centre.set(xlabel="Time t / T", ylabel="Centreline velocity / (mm/s)",
              title="Centreline velocity: amplitude reduction and phase lag")
ax_centre.legend(fontsize=8)
plt.show()


# ## Going further
#
# Lower the Womersley number towards 1 and watch the profile become the quasi-steady
# parabola $G h^2(1-y^2/h^2)/2\mu\,\cos\omega t$ in phase with the forcing; raise it
# towards 10 and the core becomes a plug that lags by almost 90° while the motion
# concentrates in thin Stokes layers at the plates. Add a steady component to the
# pressure gradient to obtain pulsatile flow superposed on Poiseuille flow, or move
# to a round pipe, where the same analysis yields Bessel functions in place of the
# hyperbolic cosine.
