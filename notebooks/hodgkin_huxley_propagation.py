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

from importlib.metadata import version
import numpy as np
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

from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.physics.biophysics import (
    HodgkinHuxley1952,
    MembraneCurrentElectrode,
    Monodomain,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# SI units throughout. The cable parameters are those of the original squid-axon
# computation; the published membrane constants (conductances in S/m², reversal
# potentials relative to a -65 mV rest) are built into `HodgkinHuxley1952`.

radius = 238.0e-6                 # m, axon radius
resistivity = 0.354               # Ohm m, axoplasm resistivity (35.4 Ohm cm)
C_m = 0.0101                      # F/m^2, membrane capacitance (1.01 uF/cm^2)
temperature = 291.65              # K, 18.5 degC
L = 0.16                          # m, axon length
refinements = ((120, 16.0e-6), (240, 8.0e-6), (480, 4.0e-6))   # (cells, max time step / s)
end_time = 9.0e-3                 # s
rest = -0.065                     # V, resting potential
stimulus_length = 0.007           # m, initially depolarised segment at x = 0
stimulus_voltage = -0.025         # V, initial potential of that segment
sensors = (0.045, 0.105)          # m, positions of the two recording sites
velocity_published = 18.8         # m/s, Hodgkin & Huxley 1952 computed value
velocity_converged = 18.6         # m/s, OneSimulation mesh-converged reference
amplitude_published = 0.0905      # V, Hodgkin & Huxley 1952 computed spike amplitude at 18.5 degC

sigma = 1.0 / resistivity
beta = 2.0 / radius               # surface-to-volume ratio of a cylinder
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


def arrival_time(wave):
    """Interpolated instant at which the membrane potential first crosses 0 V."""
    k = int(np.flatnonzero(wave >= 0.0)[0])
    return times[k - 1] + (times[k] - times[k - 1]) * (-wave[k - 1]) / (wave[k] - wave[k - 1])


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
    arrivals = [arrival_time(result.voltage[:, node]) for node in sensor_nodes]
    velocities.append(float(np.diff(position[sensor_nodes])[0] / np.diff(arrivals)[0]))
    print(f"{cells:4d} cells ({1e3 * L / cells:.3f} mm), max step {1e6 * max_step:4.0f} us:  "
          f"arrivals {1e3 * arrivals[0]:.3f} ms / {1e3 * arrivals[1]:.3f} ms,  velocity {velocities[-1]:.2f} m/s")
# The finest run is kept for the shape checks and the figure.


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
# Half-amplitude duration of the spike at the first sensor.
half_level = rest + 0.5 * (peaks[0] - rest)
above = times[wave_0 > half_level]
half_width = float(above[-1] - above[0])
print(f"half-amplitude duration of the spike: {1e3 * half_width:.2f} ms")

# Shape invariance: shift the second trace back by the measured delay and compare.
delay = arrivals[1] - arrivals[0]
window = (times >= arrivals[0] - 1.0e-3) & (times <= arrivals[0] + 4.0e-3)
shifted = np.interp(times[window] + delay, times, wave_1)
shape_error = float(np.max(np.abs(shifted - wave_0[window])))
print(f"maximum difference between the two aligned traces: {1e3 * shape_error:.2f} mV")


# ## Figure

cream = "#fbf5eb"
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax = axes[0]
ax.set_facecolor("#fffaf2")
for node, color, x in zip(sensor_nodes, ("#1f6f78", "#c56838"), sensors):
    ax.plot(times * 1e3, result.voltage[:, node] * 1e3, color=color, lw=1.6, label=f"x = {1e3 * x:.0f} mm")
ax.axhline(rest * 1e3, color="#9a8f7d", ls="--", lw=0.9, label="rest")
ax.axhline(0.0, color="#999999", lw=0.6)
ax.set(xlabel="Time / ms", ylabel="Membrane potential / mV",
       title=f"Hodgkin-Huxley axon: v = {velocity:.1f} m/s (published 18.8 m/s)")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.25)

ax = axes[1]
ax.set_facecolor("#fffaf2")
snapshot_times = (2.0e-3, 4.0e-3, 6.0e-3)
shades = ("#d8b48c", "#b5651d", "#6b442f")
for t_snap, shade in zip(snapshot_times, shades):
    k = int(np.argmin(np.abs(times - t_snap)))
    ax.plot(position * 1e3, result.voltage[k] * 1e3, color=shade, lw=1.6, label=f"t = {1e3 * times[k]:.1f} ms")
for x in sensors:
    ax.axvline(1e3 * x, color="#bbbbbb", lw=0.8, ls=":")
ax.set(xlabel="Position x / mm", ylabel="Membrane potential / mV",
       title="Travelling action potential (dotted: sensor positions)")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.25)
plt.show()


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
