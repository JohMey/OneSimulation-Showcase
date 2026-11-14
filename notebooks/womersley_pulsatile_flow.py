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

import numpy as np

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
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation import PeriodicBoundary
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# Water in a 2 mm gap. The angular frequency is chosen from the Womersley number
# $\alpha=3$, which puts the Stokes layer at roughly half the half-gap, so the
# profile is clearly not parabolic. The start-up transient decays like
# $\exp(-\nu\pi^2 t/4h^2)$, i.e. by a factor $\exp(-\pi^3/2\alpha^2)$ per cycle; after
# three cycles less than 1 % of it is left, so the fourth cycle is compared with the
# periodic reference.

length = 0.0005
gap = 0.002
span = 0.0005
rho = 1000.0
mu = 1.0e-3
nu = mu / rho
half_gap = gap / 2.0

womersley = 3.0
omega = (womersley / half_gap) ** 2 * nu
period = 2.0 * np.pi / omega
forcing_amplitude = 1.0
steps_per_period = 40
time_step = period / steps_per_period
compared_cycle = 3
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
    zeros = np.zeros_like(x)
    return np.column_stack((np.full_like(x, forcing_amplitude * np.cos(omega * t)), zeros, zeros))


def womersley_profile(y, t):
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
centre_lag = float(-np.angle(centre_complex))
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


# ## Figure

phases = {f"ωt = {np.degrees(omega * accepted) % 360:.0f}°": values for accepted, values in profiles.items()}
fine_times = np.linspace(cycle_times[0], cycle_times[-1], 200)
fig = plots.figure(1, 2, size=(11, 4.8))
plots.line(points={label: (u_numeric[::2] * 1000, (y_samples[::2] - half_gap) * 1000) for label, (u_numeric, _, _) in phases.items()},
           reference={label: (u_exact * 1000, (y_samples - half_gap) * 1000) for label, (_, u_exact, _) in phases.items()},
           hlines=[-half_gap * 1000, half_gap * 1000], xlabel="streamwise velocity u [mm/s]",
           ylabel="distance from the centre y [mm]",
           title=f"Pulsatile channel flow, α = {womersley:.0f}: OneSimulation (markers), Womersley (dashed)", ax=fig[0])
plots.line(points={"OneSimulation": (cycle_times / period, centre_series * 1000)},
           reference={"Womersley centreline u(0, t)": (fine_times / period, womersley_profile(0.0, fine_times) * 1000),
                      "quasi-steady response (α → 0)": (fine_times / period, quasi_steady_amplitude * np.cos(omega * fine_times) * 1000)},
           xlabel="time t / T", ylabel="centreline velocity [mm/s]",
           title="Centreline velocity: amplitude reduction and phase lag", ax=fig[1])
plots.show()


# ## Going further
#
# Lower the Womersley number towards 1 and watch the profile become the quasi-steady
# parabola $G h^2(1-y^2/h^2)/2\mu\,\cos\omega t$ in phase with the forcing; raise it
# towards 10 and the core becomes a plug that lags by almost 90° while the motion
# concentrates in thin Stokes layers at the plates. Add a steady component to the
# pressure gradient to obtain pulsatile flow superposed on Poiseuille flow, or move
# to a round pipe, where the same analysis yields Bessel functions in place of the
# hyperbolic cosine.
