"""OneSimulation example: Buckley-Leverett displacement of oil by water in a porous column.
Requires the installed, activated OneSimulation runtime.

Water is injected at one end of an oil-filled porous column at a constant rate. The two-phase
Darcy model of OneSimulation computes the saturation front, the water breakthrough at the outlet
and the oil recovery; all three are compared with the Buckley-Leverett solution and Welge's
tangent construction, on three grids.
"""

# # Buckley-Leverett waterflood
#
# When water displaces oil in a reservoir rock, both phases flow at once, each through the part
# of the pores it occupies. With the water saturation $S$, the porosity $\phi$ and the total
# Darcy velocity $u$, mass conservation of the water in one dimension reads
#
# $$\phi\,\frac{\partial S}{\partial t}+u\,\frac{\partial f(S)}{\partial x}=0,\qquad
#   f(S)=\frac{k_{rw}(S)/\mu_w}{k_{rw}(S)/\mu_w+k_{ro}(S)/\mu_o} ,$$
#
# where the fractional flow $f$ follows from the relative permeabilities, here the Corey laws
# $k_{rw}=S^2$, $k_{ro}=(1-S)^2$. Capillary pressure and gravity are neglected.
#
# **Setup.** A column of length $L=100$ m, porosity 0.2, permeability 100 mD, water viscosity
# 1 mPa s, oil viscosity 5 mPa s, initially full of oil. Water is injected at $x=0$ with the
# Darcy velocity $u=10^{-6}$ m/s (8.6 cm per day); the outlet produces whatever arrives.
#
# | end | condition |
# |---|---|
# | $x=0$ (inlet) | total inflow $u$, pure water |
# | $x=L$ (outlet) | pressure 0, outflow of both phases |
#
# **Reference.** S.E. Buckley and M.C. Leverett, Trans. AIME 146, 1942, 107-116: each saturation
# moves with the speed $u f'(S)/\phi$, and the multivalued part is replaced by a shock at the
# saturation $S_f$ where the tangent from the initial state touches $f$ (H.J. Welge, Trans. AIME
# 195, 1952, 91-98):
#
# $$f'(S_f)=\frac{f(S_f)}{S_f},\qquad
#   t_\mathrm{bt}=\frac{\phi L}{u\,f'(S_f)},\qquad
#   \bar S=S_2+\big(1-f(S_2)\big)\,Q\quad\text{with}\quad f'(S_2)=\frac1Q ,$$
#
# with $Q$ the injected pore volumes and $\bar S$ the mean water saturation, equal to the oil
# recovered per pore volume.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.physics.geotechnics import (
    ImmiscibleReservoir,
    ReservoirFiniteVolume,
    ReservoirFluxBoundary,
    ReservoirGeometryFactory,
    ReservoirPressureBoundary,
)
from onesim.visualization import plots, theme

theme.set()

# ## Model
#
# The column is a line of finite volumes with a cross-section of 1 m²; the outlet is held at zero
# pressure. Three grids of 50, 100 and 200 cells; the results are written every 0.05 injected
# pore volumes up to 2.

length, porosity, velocity = 100.0, 0.2, 1e-6
rock = ImmiscibleReservoir(
    porosity=porosity,
    permeability=1e-13,
    water_viscosity=1e-3,
    oil_viscosity=5e-3,
    water_density=1000.0,
    oil_density=850.0,
    corey_exponent=2.0,
    cross_section=1.0,
)
pore_volume_time = porosity * length / velocity
injected = np.linspace(0.0, 2.0, 41)
columns = [50, 100, 200]
column = Line(start=(0.0, 0.0, 0.0), end=(length, 0.0, 0.0))
meshes = {n: StructuredLineMesher(cells=n).generate(column) for n in columns}
results = {
    n: rock.solve(
        mesh=mesh,
        water_saturation=0.0,
        times=injected * pore_volume_time,
        max_step=pore_volume_time / n,
        spatial=ReservoirFiniteVolume(geometry_factory=ReservoirGeometryFactory()),
        flux_boundaries=(
            ReservoirFluxBoundary(
                boundary=column.boundaries.start, flux=-velocity, water_saturation=1.0
            ),
        ),
        pressure_boundaries=(
            ReservoirPressureBoundary(boundary=column.boundaries.end, pressure=0.0),
        ),
    )
    for n, mesh in meshes.items()
}

# ## Reference: Buckley-Leverett and Welge
#
# The fractional flow on a fine saturation grid, the front saturation where the chord $f(S)/S$ from
# the initial state is steepest (the tangent condition),
# the saturation profile $x(S)=u t f'(S)/\phi$ behind the front, and the outlet saturation, water
# cut and recovery after breakthrough.

fraction = lambda s: s**2 / (s**2 + (1 - s) ** 2 / 5)
s = np.linspace(0.0, 1.0, 200001)
slope = np.gradient(fraction(s), s)
front = s[1:][np.argmax(fraction(s[1:]) / s[1:])]
behind = s >= front
breakthrough = 1 / np.interp(front, s, slope)
profile = lambda q, x: np.interp(
    x / length / q, slope[behind][::-1], s[behind][::-1], right=0.0
)
outlet = lambda q: np.interp(1 / q, slope[behind][::-1], s[behind][::-1], left=1.0)
recovery = lambda q: np.where(
    q < breakthrough, q, outlet(q) + (1 - fraction(outlet(q))) * q
)

# ## Comparison with the reference
#
# The mean saturation in the column is the oil recovered; the breakthrough is the first output
# with water at the outlet cell; the profile error is the pore-volume weighted mean deviation at
# 0.4 injected pore volumes. Last, the mass balance of both phases: the change of the mass in the
# column against what entered and left through the ends.

print(
    f"front saturation S_f = {front:.4f}, breakthrough after {breakthrough:.4f} pore volumes = {breakthrough * pore_volume_time / 86400:.0f} days"
)
print(
    f"{'cells':>6s} {'profile error at Q = 0.4':>25s} {'breakthrough [PV]':>18s} {'recovery at Q = 1':>18s} {'Welge':>7s} {'recovery at Q = 2':>18s} {'Welge':>7s}"
)
errors = []
for n, result in results.items():
    x = (meshes[n].faces[:-1] + meshes[n].faces[1:]) / 2
    mean = result.water_saturation @ result.pore_volume / result.pore_volume.sum()
    errors.append(
        result.pore_volume
        @ np.abs(result.water_saturation[8] - profile(0.4, x))
        / result.pore_volume.sum()
    )
    first = injected[np.argmax(result.water_saturation[:, -1] > 0.05)]
    print(
        f"{n:6d} {errors[-1]:25.4f} {first:18.2f} {mean[20]:18.4f} {recovery(1.0):7.4f} {mean[40]:18.4f} {recovery(2.0):7.4f}"
    )
print(
    f"profile error ratio per halving of the cells: {errors[0] / errors[1]:.2f}, {errors[1] / errors[2]:.2f}"
)
print(
    "largest mass-balance error of the phases over the run [kg]: "
    + ", ".join(
        f"{np.abs(result.phase_mass - result.phase_mass[0] - (result.well_phase_volume.sum(axis=1) - result.boundary_phase_volume) * (1000.0, 850.0)).max():.1e} ({n} cells)"
        for n, result in results.items()
    )
)

# ## Figure
#
# Saturation profiles at three times before breakthrough, the oil recovery and the water cut at
# the outlet over the injected pore volumes.

x = np.linspace(0.0, length, 2001)
fine = results[200]
centres = (meshes[200].faces[:-1] + meshes[200].faces[1:]) / 2
order = np.argsort(centres)
fig = plots.figure(1, 3, size=(18, 5.5))
plots.line(
    y={
        f"Q = {q}": (centres[order], fine.water_saturation[k, order])
        for q, k in ((0.2, 4), (0.4, 8), (0.6, 12))
    },
    reference={f"Q = {q}": (x, profile(q, x)) for q in (0.2, 0.4, 0.6)},
    ylimits=(0, 1),
    xlabel="x [m]",
    ylabel="water saturation S",
    title="Saturation profiles, 200 cells",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    points={
        f"{n} cells": (
            injected,
            result.water_saturation @ result.pore_volume / result.pore_volume.sum(),
        )
        for n, result in results.items()
    },
    reference={
        "Welge": (np.linspace(0.01, 2, 400), recovery(np.linspace(0.01, 2, 400)))
    },
    vlines={"breakthrough": breakthrough},
    xlabel="injected pore volumes Q",
    ylabel="oil recovered [pore volumes]",
    title="Oil recovery",
    legend="lower right",
    ax=fig[1],
)
plots.line(
    y={
        f"{n} cells": (injected, fraction(result.water_saturation[:, -1]))
        for n, result in results.items()
    },
    reference={
        "Welge": (
            np.linspace(0.01, 2, 400),
            np.where(
                np.linspace(0.01, 2, 400) < breakthrough,
                0.0,
                fraction(outlet(np.linspace(0.01, 2, 400))),
            ),
        )
    },
    xlabel="injected pore volumes Q",
    ylabel="water cut f at the outlet",
    title="Water cut",
    legend="lower right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Raise the oil viscosity: the front saturation drops and water breaks through early, the
# classical argument for polymer flooding. Add `capillary_pressure_scale` to smear the front, or
# turn the column upright and add gravity.
