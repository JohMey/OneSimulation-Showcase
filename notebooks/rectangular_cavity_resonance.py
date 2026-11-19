"""OneSimulation example: resonances of a rectangular PEC cavity.
Requires the installed, activated OneSimulation runtime.

A short current pulse rings up a closed air-filled cavity; the Yee time-domain
solver propagates the free ring-down and the spectrum of a probe signal is
checked against the closed-form resonance frequencies f_mnp.
"""

# # Resonances of a rectangular PEC cavity
#
# A closed, air-filled box of size $a\times b\times d$ bounded on all six faces
# by perfect electric conductors is excited by a short current pulse. After the
# pulse has ended the field rings freely; the electric field at a probe point is
# recorded and Fourier transformed, and the spectral peaks are the cavity
# resonances.
#
# The time-domain Maxwell equations are advanced with OneSimulation's Yee
# scheme (electric field on edges, magnetic flux on faces, second-order
# leapfrog in time):
#
# $$\varepsilon_0\,\partial_t\mathbf{E}=\nabla\times\mathbf{H}-\mathbf{J},\qquad
#   \mu_0\,\partial_t\mathbf{H}=-\nabla\times\mathbf{E}.$$
#
# **Reference.** The resonances of a rectangular cavity are
#
# $$f_{mnp}=\frac{c}{2}\sqrt{\Big(\frac{m}{a}\Big)^2+\Big(\frac{n}{b}\Big)^2+\Big(\frac{p}{d}\Big)^2},$$
#
# where at most one index may vanish. With $b<a<d$ the fundamental is TE$_{101}$,
# followed by TE$_{102}$ and TE$_{201}$. We check the first resonances of the
# spectrum against these closed forms and confirm that the discrete scheme
# conserves the electromagnetic energy during the ring-down.

from math import pi, sqrt

import numpy as np
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.numerics.discretization import StaggeredLocation
from onesim.physics.electromagnetics import DEFAULT_ELECTROMAGNETICS_PROVIDER_REGISTRY
from onesim.visualization import plots, theme
from scipy.signal import find_peaks

theme.set()


# ## Parameters
#
# The cavity is 50 mm x 20 mm x 60 mm, so TE101 resonates at about 3.90 GHz,
# TE102 at 5.83 GHz and TE201 at 6.50 GHz. The grid spacing of 2.5 mm resolves
# the shortest wavelength of interest (46 mm at 6.5 GHz) with 18 cells.

a = 0.050
b = 0.020
d = 0.060
cells = (20, 8, 24)
c0 = 1.0 / sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)


def f_mnp(m, n, p):
    return 0.5 * c0 * sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2)


modes = sorted(
    (f_mnp(m, n, p), f"{m}{n}{p}")
    for m in range(3)
    for n in range(2)
    for p in range(4)
    if sum(index > 0 for index in (m, n, p)) >= 2
)
reference = modes[:3]
for f_ref, label in reference:
    print(f"f_{label} = {f_ref / 1e9:.4f} GHz")

node_axes = tuple(
    np.linspace(0.0, extent, count + 1) for extent, count in zip((a, b, d), cells)
)
time_step = 4e-12
pulse_duration = 0.5e-9
sample_interval = 20e-12
samples = 2000


def yee_solver(final_time):
    return DEFAULT_ELECTROMAGNETICS_PROVIDER_REGISTRY.create(
        "electromagnetics.maxwell_time_domain",
        node_axes=node_axes,
        periodic=(False, False, False),
        permittivity=VACUUM_PERMITTIVITY,
        permeability=VACUUM_PERMEABILITY,
        final_time=final_time,
        time_step=time_step,
    )


# ## Excitation and ring-down
#
# A Gaussian-derivative current pulse (centre 0.2 ns, width 40 ps, spectrum
# peaking near 4 GHz) is injected as J_y in a small blob off the symmetry planes
# of the cavity so that several low modes are excited. The field state after the
# pulse is then advanced in short segments of 20 ps and E_y is sampled at a probe
# point after each segment.

pulse_centre, pulse_width = 0.2e-9, 0.04e-9
source_x, source_z = 0.3 * a, 0.3 * d


def current_density_y(time, points):
    envelope = (
        -(time - pulse_centre)
        / pulse_width
        * np.exp(-0.5 * ((time - pulse_centre) / pulse_width) ** 2)
    )
    blob = np.exp(
        -((points[..., 0] - source_x) ** 2 + (points[..., 2] - source_z) ** 2)
        / 0.004**2
    )
    return envelope * blob


pulse = yee_solver(pulse_duration)
state = pulse.run(current_density=(0.0, current_density_y, 0.0))
print(f"Pulse: {state.step_count} steps, injected energy {state.source_work:.3e} J")

segment = yee_solver(sample_interval)
ey_axes = segment.grid.axes(StaggeredLocation("edge", 1))
probe = tuple(
    int(np.argmin(np.abs(axis - target)))
    for axis, target in zip(ey_axes, (0.65 * a, 0.5 * b, 0.7 * d))
)
print(
    "Probe at x = {:.1f} mm, y = {:.2f} mm, z = {:.1f} mm".format(
        *(1e3 * axis[index] for axis, index in zip(ey_axes, probe))
    )
)

electric, magnetic, charge = state.electric, state.magnetic, state.charge_density
signal = np.empty(samples)
energy_start = None
for i in range(samples):
    state = segment.run(
        initial_electric=electric, initial_magnetic=magnetic, initial_charge=charge
    )
    electric, magnetic, charge = state.electric, state.magnetic, state.charge_density
    signal[i] = electric[1][probe]
    energy_start = state.initial_energy if energy_start is None else energy_start
energy_drift = abs(state.final_energy / energy_start - 1.0)
times = pulse_duration + sample_interval * np.arange(1, samples + 1)
print(
    f"Ring-down: {samples} samples over {sample_interval * samples * 1e9:.0f} ns, relative energy drift {energy_drift:.1e}"
)


# ## Comparison with the closed-form resonances
#
# The ring-down is Hann-windowed and zero-padded; the spectral peaks below
# 7 GHz are compared with f_101, f_102 and f_201 in order of frequency.

padding = 16 * samples
spectrum = np.abs(np.fft.rfft(signal * np.hanning(samples), padding))
frequencies = np.fft.rfftfreq(padding, sample_interval)
spectrum /= spectrum.max()
band = frequencies < 7e9
peaks, _ = find_peaks(spectrum[band], prominence=0.05)
f_peaks = frequencies[band][peaks]
print("Spectral peaks below 7 GHz:", ", ".join(f"{f / 1e9:.4f} GHz" for f in f_peaks))

errors = []
for (f_ref, label), f_num in zip(reference, f_peaks):
    errors.append(abs(f_num / f_ref - 1.0))
    print(
        f"TE{label}: computed {f_num / 1e9:.4f} GHz, reference {f_ref / 1e9:.4f} GHz, deviation {100 * errors[-1]:.2f} %"
    )


# ## Figure

show = frequencies < 9e9
fig = plots.figure(1, 2, size=(12, 4.8))
plots.line(
    times * 1e9,
    signal / np.max(np.abs(signal)),
    xlimits=(0, 8),
    xlabel="time [ns]",
    ylabel="E_y at the probe / max",
    title="Ring-down of the cavity after the current pulse",
    ax=fig[0],
)
plots.line(
    frequencies[show] / 1e9,
    {"OneSimulation probe spectrum": spectrum[show]},
    vlines={f"closed form {label}": f_ref / 1e9 for f_ref, label in modes[:6]},
    xlimits=(2, 9),
    ylimits=(0, 1.1),
    xlabel="frequency [GHz]",
    ylabel="|FFT| / max",
    title="Resonance spectrum and closed-form mode frequencies",
    ax=fig[1],
)
plots.show()


# ## Going further
#
# Move the source off the mid-plane in y to excite the TE011 and TE111 modes as
# well, halve the grid spacing to shrink the small numerical red shift of the
# peaks, or fill the cavity with a dielectric (`permittivity=eps_r*eps_0`) and
# watch every resonance scale with $1/\sqrt{\varepsilon_r}$.
