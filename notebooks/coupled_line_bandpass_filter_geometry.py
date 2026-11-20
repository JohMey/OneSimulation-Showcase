"""Geometry of the OneSimulation example "Microstrip coupled-line bandpass filter".

Builds with the CAD objects of OneSimulation a third-order, 0.5 dB equal-ripple coupled-line bandpass
filter for 5 GHz with 10 % bandwidth (Pozar, Microwave Engineering, §8.8) on a 0.813 mm substrate of
relative permittivity 3.55. The even- and odd-mode impedances of the four quarter-wave coupled
sections follow from the prototype values; the widths and gaps are found from the coupled
microstrip formulas of Garg and Bahl, the section lengths from the mean of the even- and odd-mode
effective permittivities, shortened by the open-end extension of Hammerstad. 50-ohm lines of 5 mm lead from the ends to the two sides of a box of air
5 mm high. Seven STEP files: coupled_line_bandpass_filter.step (faces port_1, port_2, ground below,
open for the lid and the long sides), coupled_line_bandpass_filter_substrate.step and
coupled_line_bandpass_filter_copper_0.step ... _4.step (thin layers of air in the shape of the five
pieces of copper: the input line with the first quarter wave, the three half-wave resonators and the
output line with the last quarter wave; their faces on the substrate are the copper).
"""

import numpy as np
from onesim.cad.d3 import Box
from scipy.optimize import brentq, fsolve
from scipy.special import ellipk

height, permittivity, frequency, bandwidth, lid, feed, layer = (
    0.813e-3,
    3.55,
    5e9,
    0.10,
    5e-3,
    5e-3,
    0.4e-3,
)
eps0, light, z0 = 8.8541878128e-12, 299792458.0, 50.0
prototype = (1.5963, 1.0967, 1.5963, 1.0)


def single(width, er):
    """Hammerstad-Jensen characteristic impedance and effective permittivity of a microstrip."""
    u = width / height
    a = (
        1
        + np.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49
        + np.log(1 + (u / 18.1) ** 3) / 18.7
    )
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    effective = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    f = 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))
    return (
        376.730313668
        / (2 * np.pi)
        * np.log(f / u + np.sqrt(1 + 4 / u**2))
        / np.sqrt(effective),
        effective,
    )


def capacitances(width, gap, er):
    """Even- and odd-mode capacitances per length of coupled microstrips (Garg and Bahl 1979)."""
    impedance, effective = single(width, er)
    plate = eps0 * er * width / height
    fringe = 0.5 * (np.sqrt(effective) / (light * impedance) - plate)
    a = np.exp(-0.1 * np.exp(2.33 - 2.53 * width / height))
    inner = (
        fringe
        / (1 + a * height / gap * np.tanh(8 * gap / height))
        * np.sqrt(er / effective)
    )
    k = (gap / height) / (gap / height + 2 * width / height)
    air = eps0 * ellipk(1 - k**2) / ellipk(k**2)
    dielectric = eps0 * er / np.pi * np.log(
        1 / np.tanh(np.pi * gap / (4 * height))
    ) + 0.65 * fringe * (0.02 * np.sqrt(er) / (gap / height) + 1 - er**-2)
    return plate + fringe + inner, plate + fringe + air + dielectric


def modes(width, gap):
    """Even and odd impedances and effective permittivities."""
    even, odd = capacitances(width, gap, permittivity)
    even0, odd0 = capacitances(width, gap, 1.0)
    return (
        1 / (light * np.sqrt(even * even0)),
        1 / (light * np.sqrt(odd * odd0)),
        even / even0,
        odd / odd0,
    )


inverters = [np.sqrt(np.pi * bandwidth / (2 * prototype[0]))]
inverters += [
    np.pi * bandwidth / (2 * np.sqrt(prototype[n - 1] * prototype[n])) for n in (1, 2)
]
inverters += [np.sqrt(np.pi * bandwidth / (2 * prototype[2] * prototype[3]))]
sections = []
for j in inverters:
    target = (z0 * (1 + j + j**2), z0 * (1 - j + j**2))
    width, gap = np.exp(
        fsolve(
            lambda v: np.array(modes(*np.exp(v))[:2]) - target, np.log([1.5e-3, 0.3e-3])
        )
    )
    _, _, even, odd = modes(width, gap)
    effective = single(width, permittivity)[1]
    extension = (
        0.412
        * height
        * (effective + 0.3)
        * (width / height + 0.264)
        / ((effective - 0.258) * (width / height + 0.8))
    )
    sections.append(
        (
            width,
            gap,
            light / frequency / np.sqrt((even + odd) / 2) / 4 - extension,
            target,
        )
    )
line = brentq(lambda w: single(w, permittivity)[0] - z0, 0.05 * height, 20 * height)
strip = lambda x, y, dx, dy: Box(
    length=dx, width=dy, height=layer, origin=(x, y, height)
)
# Along x the four sections follow each other; the upper line of one section and the lower line of the
# next form one half-wave resonator on a common centre line. Every separate piece of copper is a
# region of its own.
pieces, x, y = [strip(0.0, -line / 2, feed + 1e-4, line)], feed, 0.0
for width, gap, length, _ in sections:
    pieces[-1] = pieces[-1] | strip(x, y - width / 2, length, width)
    pieces.append(strip(x, y + gap + width / 2, length, width))
    x, y = x + length, y + gap + width
pieces[-1] = pieces[-1] | strip(x - 1e-4, y - line / 2, feed + 1e-4, line)
low, high = -7e-3, y + 7e-3
box = Box(length=x + feed, width=high - low, height=lid, origin=(0.0, low, 0.0))
faces = box.boundaries
box.grouped(
    port_1=faces.x_min,
    port_2=faces.x_max,
    ground=faces.z_min,
    open=faces.z_max | faces.y_min | faces.y_max,
).save("coupled_line_bandpass_filter.step")
Box(length=x + feed, width=high - low, height=height, origin=(0.0, low, 0.0)).save(
    "coupled_line_bandpass_filter_substrate.step"
)
for index, piece in enumerate(pieces):
    piece.save(f"coupled_line_bandpass_filter_copper_{index}.step")
print(f"50-ohm line {line * 1e3:.3f} mm")
for n, (width, gap, length, (even, odd)) in enumerate(sections, start=1):
    print(
        f"section {n}: Z0e {even:.2f} ohm, Z0o {odd:.2f} ohm -> width {width * 1e3:.3f} mm, gap {gap * 1e3:.3f} mm, length {length * 1e3:.3f} mm"
    )
