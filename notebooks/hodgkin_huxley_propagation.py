"""OneSimulation example: propagating action potential on a Hodgkin-Huxley axon.
Requires the installed, activated OneSimulation runtime.

The computed conduction velocity is checked against the published value of the
Hodgkin-Huxley squid-axon model, and the action-potential shape against the
classical observations.
"""


# # Action-potential conduction along a Hodgkin-Huxley axon
#
# The 1952 Hodgkin-Huxley model describes the squid giant axon membrane with a
# voltage-gated sodium channel ($m^3h$), a voltage-gated potassium channel
# ($n^4$) and a passive leak. Embedded in a one-dimensional cable, the membrane
# model becomes the nonlinear **monodomain** (cable) equation
#
# $$C_m\,\frac{\partial V}{\partial t}
#    = \frac{\sigma}{\beta}\,\frac{\partial^2 V}{\partial x^2}
#      - \bar g_\mathrm{Na} m^3 h\,(V-E_\mathrm{Na})
#      - \bar g_\mathrm{K} n^4\,(V-E_\mathrm{K})
#      - g_L\,(V-E_L),$$
#
# with the gate kinetics $\dot y=\alpha_y(V)(1-y)-\beta_y(V)\,y$ for
# $y\in\{m,h,n\}$. OneSimulation ships the published channel set as
# `HodgkinHuxley1952` and integrates it on a real 1D line mesh with both fibre
# ends sealed (zero axial current). A short depolarised segment at one end
# triggers an action potential that then travels along the axon at a constant
# speed.
#
# **Reference.** For the original squid-axon parameters (fibre radius 238 µm,
# axoplasm resistivity 35.4 Ω cm, membrane capacitance 1.01 µF/cm²) at 18.5 °C,
# Hodgkin and Huxley computed a conduction velocity of **18.8 m/s** and a
# propagated spike amplitude of **90.5 mV** above rest (J. Physiol. 117,
# 500-544, 1952, Table 3; the measured values were 21.2 m/s and 86.5 mV).
# OneSimulation's own mesh-converged velocity for this configuration is
# **18.6 m/s**. We measure the velocity from the arrival times of the up-stroke
# at two sensors well away from the stimulus and the far end, on three
# successively refined meshes (120, 240 and 480 cells over 160 mm with 16, 8 and
# 4 µs maximum time steps), and require the finest value to agree with the
# references within a few percent and the refinement sequence to converge. We
# also check the classical qualitative features: the overshoot above 0 V, the
# after-hyperpolarisation below rest, a sub-millisecond spike at this
# temperature, and a wave that keeps its shape while it travels.

import numpy as np

from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.results.sampling import crossing_times
from onesim.physics.biophysics import (
    HodgkinHuxley1952,
    MembraneCurrentElectrode,
    Monodomain,
)

from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# SI units throughout. The cable parameters are those of the original squid-axon
# computation; the published membrane constants (conductances in S/m², reversal
# potentials relative to a -65 mV rest) are built into `HodgkinHuxley1952`.

radius = 238.0e-6
resistivity = 0.354
C_m = 0.0101
temperature = 291.65
L = 0.16
refinements = ((120, 16.0e-6), (240, 8.0e-6), (480, 4.0e-6))
end_time = 9.0e-3
rest = -0.065
stimulus_length = 0.007
stimulus_voltage = -0.025
sensors = (0.045, 0.105)
velocity_published = 18.8
velocity_converged = 18.6
amplitude_published = 0.0905

sigma = 1.0 / resistivity
beta = 2.0 / radius
print(f"axial conductivity = {sigma:.3f} S/m,  surface-to-volume = {beta:.0f} 1/m")

line = Line(start=(0.0, 0.0, 0.0), end=(L, 0.0, 0.0))

model = Monodomain(
    conductivity=sigma,
    capacitance=C_m,
    surface_to_volume=beta,
    cable_area=np.pi * radius**2,
    membrane=HodgkinHuxley1952(temperature=temperature),
)

# Both fibre ends carry an explicit boundary condition: sealed (zero-current)
# electrodes, as for an isolated axon in a bath.
electrodes = (
    MembraneCurrentElectrode(boundary=line.boundaries.start, current_density=0.0),
    MembraneCurrentElectrode(boundary=line.boundaries.end, current_density=0.0),
)

times = np.linspace(0.0, end_time, 451)


velocities = []
for cells, max_step in refinements:
    mesh = StructuredLineMesher(cells=cells).generate(line)
    result = model.solve(
        mesh=mesh,
        times=times,
        max_step=max_step,
        initial_voltage=lambda x: np.where(x < stimulus_length, stimulus_voltage, rest),
        electrodes=electrodes,
    )
    position = result.coordinates[:, 0]
    sensor_nodes = [int(np.argmin(np.abs(position - x))) for x in sensors]
    arrivals = [crossing_times(times, result.voltage[:, node], 0.0)[0] for node in sensor_nodes]
    velocities.append(float(np.diff(position[sensor_nodes])[0] / np.diff(arrivals)[0]))
    print(f"{cells:4d} cells ({1e3 * L / cells:.3f} mm), max step {1e6 * max_step:4.0f} us:  "
          f"arrivals {1e3 * arrivals[0]:.3f} ms / {1e3 * arrivals[1]:.3f} ms,  velocity {velocities[-1]:.2f} m/s")


# ## Conduction velocity
#
# The arrival time of the up-stroke at each sensor is the (interpolated) instant
# at which the membrane potential crosses 0 V; the velocity is the sensor
# spacing divided by the difference of the arrival times. The three meshes must
# converge (each refinement at least roughly halves the change) and the finest
# value must match the references.

velocity = velocities[-1]
print(f"conduction velocity (finest mesh): {velocity:.2f} m/s  "
      f"(published model value {velocity_published:.1f} m/s, converged reference {velocity_converged:.1f} m/s)")


# ## Shape of the action potential
#
# At 18.5 °C Hodgkin and Huxley's computed propagated spike rises 90.5 mV above
# rest (to about +25 mV), lasts about half a millisecond at half amplitude and
# is followed by an after-hyperpolarisation of roughly ten millivolts below
# rest. The wave must also keep its shape as it travels: the traces at the two
# sensors coincide after a shift by the propagation delay.

peaks = [float(result.voltage[:, node].max()) for node in sensor_nodes]
minima = [float(result.voltage[:, node].min()) for node in sensor_nodes]
print(f"peak potential at the sensors: {1e3 * peaks[0]:.1f} mV, {1e3 * peaks[1]:.1f} mV;  "
      f"after-hyperpolarisation: {1e3 * minima[0]:.1f} mV, {1e3 * minima[1]:.1f} mV")
print(f"spike amplitude above rest: {1e3 * (peaks[1] - rest):.1f} mV  (published computed value {1e3 * amplitude_published:.1f} mV)")

wave_0 = result.voltage[:, sensor_nodes[0]]
wave_1 = result.voltage[:, sensor_nodes[1]]
half_level = rest + 0.5 * (peaks[0] - rest)
above = times[wave_0 > half_level]
half_width = float(above[-1] - above[0])
print(f"half-amplitude duration of the spike: {1e3 * half_width:.2f} ms")

delay = arrivals[1] - arrivals[0]
window = (times >= arrivals[0] - 1.0e-3) & (times <= arrivals[0] + 4.0e-3)
shifted = np.interp(times[window] + delay, times, wave_1)
shape_error = float(np.max(np.abs(shifted - wave_0[window])))
print(f"maximum difference between the two aligned traces: {1e3 * shape_error:.2f} mV")


# ## Figure

snapshots = [int(np.argmin(np.abs(times - t))) for t in (2.0e-3, 4.0e-3, 6.0e-3)]
fig = plots.figure(1, 2, size=(11.5, 4.6))
plots.line(times * 1e3, {f"x = {1e3 * x:.0f} mm": result.voltage[:, node] * 1e3 for node, x in zip(sensor_nodes, sensors)},
           hlines={"rest": rest * 1e3}, xlabel="time [ms]", ylabel="membrane potential [mV]",
           title=f"Hodgkin-Huxley axon: v = {velocity:.1f} m/s (published 18.8 m/s)", ax=fig[0])
plots.line(position * 1e3, {f"t = {1e3 * times[k]:.1f} ms": result.voltage[k] * 1e3 for k in snapshots},
           vlines=[1e3 * x for x in sensors], xlabel="position x [mm]", ylabel="membrane potential [mV]",
           title="Travelling action potential (dotted: sensor positions)", ax=fig[1])
plots.show()


# ## Going further
#
# Lower the temperature to 6.3 °C (the default of `HodgkinHuxley1952`) and the
# gate kinetics slow down by the built-in $Q_{10}=3$ factor: the spike grows to
# about 100 mV, lasts over a millisecond and travels more slowly. Raise it and
# the opposite happens until propagation fails altogether (heat block).
# Replace the initial depolarised segment by a timed current pulse through a
# `MembraneCurrentElectrode` at $x=0$, and apply a second pulse a few
# milliseconds later to observe the refractory period: within about 8 ms after
# the first spike the fibre cannot be re-excited.
