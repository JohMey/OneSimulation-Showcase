"""OneSimulation example: band gaps of a one-dimensional photonic crystal for three material contrasts.
Requires the installed, activated OneSimulation runtime.

One period of a multilayer stack, read from a STEP file, is solved on its 2D cross-section with
Bloch-periodic sides; the eigenfrequency solver returns the photonic bands for a uniform medium,
a weak and a strong index contrast. The bands are compared with the exact dispersion relation of
the periodic multilayer.
"""

# # Band gaps of a one-dimensional photonic crystal
#
# A stack of alternating layers with permittivities $\varepsilon_1$ and $\varepsilon_2$, each half a
# period $a$ thick, is periodic along $x$ and uniform along $y$. Waves travelling along $x$ obey
# Bloch's theorem, $E_z(x+a)=e^{\mathrm ik_xa}E_z(x)$, and their frequencies form bands $\omega_n(k_x)$.
# Wherever the layers differ, gaps open at the edge of the Brillouin zone, $k_x=\pi/a$: frequencies in
# the gap cannot propagate, and the stack is a perfect mirror there. Three cases (Joannopoulos et al.,
# *Photonic Crystals: Molding the Flow of Light*, 2nd ed., 2008, ch. 4, fig. 2):
#
# | case | $\varepsilon_1$ | $\varepsilon_2$ |
# |---|---|---|
# | uniform | 13 | 13 |
# | weak contrast | 13 | 12 |
# | strong contrast | 13 | 1 |
#
# OneSimulation solves the time-harmonic Maxwell equation for $E_z$ on the cross-section of one
# period and finds its eigenfrequencies with `electromagnetic_modes`:
#
# | sides | condition |
# |---|---|
# | `x_min` ↔ `x_max` | Bloch periodic with $k_x$ |
# | `y_min` ↔ `y_max` | periodic ($k_y=0$) |
#
# The unit cell comes from `one_dimensional_photonic_crystal_bands.step`, the first layer from
# `one_dimensional_photonic_crystal_bands_layer.step`; the script
# `one_dimensional_photonic_crystal_bands_geometry.py` builds both with the CAD objects of
# OneSimulation.
#
# **Reference.** The transfer matrix of one period gives the exact dispersion relation of the
# multilayer (Yeh, *Optical Waves in Layered Media*, 1988, ch. 6), with $k_i=n_i\omega/c$ and
# $d_1=d_2=a/2$:
#
# $$\cos k_xa=\cos k_1d_1\cos k_2d_2-\frac12\Big(\frac{n_1}{n_2}+\frac{n_2}{n_1}\Big)\sin k_1d_1\sin k_2d_2 .$$
#
# Where the right-hand side exceeds 1 in magnitude, there is a gap.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import (
    IsotropicElectromagneticMaterial,
    RegionwiseElectromagneticMaterial,
)
from onesim.meshing import PeriodicPair, TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    BlochPeriodicBoundary,
    FrequencyDomainMaxwell,
    electromagnetic_modes,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

light = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)

# ## Geometry and mesh
#
# `section` cuts the plates at mid-thickness. Quadratic triangles of $a/40$ follow the layer
# interface; the periodic sides get the same nodes on both sides.

plate, layer_plate = load("one_dimensional_photonic_crystal_bands.step"), load(
    "one_dimensional_photonic_crystal_bands_layer.step"
)
cell = plate.section((0, 0, 1), origin=(0, 0, plate.size[2] / 2))
layer = layer_plate.section((0, 0, 1), origin=(0, 0, plate.size[2] / 2)).named("layer")
a = cell.size[0]
sides = cell.boundaries
mesh = TriangularMesher(
    target_size=a / 40,
    embedded_regions=(layer,),
    hard_sizes={side: a / 40 for side in ("x_min", "x_max", "y_min", "y_max")},
).generate(cell)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(
    [plate.named("unit cell"), layer_plate.named("layer 1")],
    camera=plots.Camera(elevation=55, azimuth=-70),
    title="One period of the stack (STEP)",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Cross-section: {mesh.cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Bands
#
# For every case and eleven Bloch wavenumbers from the centre to the edge of the zone, the four
# lowest eigenfrequencies; the static solution at $k_x=0$ is left out.

cases = {
    "uniform": (13.0, 13.0),
    "weak contrast": (13.0, 12.0),
    "strong contrast": (13.0, 1.0),
}
material = lambda e1, e2: RegionwiseElectromagneticMaterial(
    {"layer": IsotropicElectromagneticMaterial(relative_permittivity=e1)},
    default=IsotropicElectromagneticMaterial(relative_permittivity=e2),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)),
    linear_solver=ComplexSparseDirect(),
)
wavenumbers = np.linspace(0.0, np.pi / a, 11)
bands = {}
for name, (e1, e2) in cases.items():
    rows = []
    for k in wavenumbers:
        m = FrequencyDomainMaxwell(
            domain=cell.domain(),
            frequency=0.2 * light / a,
            material=material(e1, e2),
            polarization="Ez",
            phasor_convention="exp(-i*omega*t)",
            amplitude_convention="peak",
        )
        result = Simulation(
            problem=Problem(
                name="Unit cell",
                geometry=cell,
                physics=(m,),
                boundary_conditions=(
                    BlochPeriodicBoundary(
                        physics=m,
                        pair=PeriodicPair(sides.x_min, sides.x_max),
                        wavevector=(k, 0.0, 0.0),
                    ),
                    BlochPeriodicBoundary(
                        physics=m,
                        pair=PeriodicPair(sides.y_min, sides.y_max),
                        wavevector=(k, 0.0, 0.0),
                    ),
                ),
            ),
            mesh=mesh,
            numerics=numerics,
        ).solve_stationary()
        found = (
            np.sort(
                electromagnetic_modes(
                    result, m, mode_count=6, target_frequency=0.25 * light / a
                ).frequencies.real
            )
            * a
            / light
        )
        rows.append(found[found > 1e-3][:4])
    bands[name] = np.array(rows)

# ## Comparison with the exact dispersion relation
#
# For every computed pair $(k_x,\omega)$ the mismatch in the dispersion relation, divided by its
# slope in $\omega$, is the distance to the exact band (one Newton step); where two exact bands touch
# the slope vanishes and the point is left out. The gaps follow from where the right-hand side
# leaves $[-1, 1]$.

rhs = lambda nu, e1, e2: (
    np.cos(np.pi * nu * np.sqrt(e1)) * np.cos(np.pi * nu * np.sqrt(e2))
    - 0.5
    * (np.sqrt(e1 / e2) + np.sqrt(e2 / e1))
    * np.sin(np.pi * nu * np.sqrt(e1))
    * np.sin(np.pi * nu * np.sqrt(e2))
)
grid = np.linspace(1e-4, 0.45, 45001)
mismatch = lambda nu, k, e1, e2: rhs(nu, e1, e2) - np.cos(k * a)
slope = (
    lambda nu, k, e1, e2: (
        mismatch(nu + 1e-7, k, e1, e2) - mismatch(nu - 1e-7, k, e1, e2)
    )
    / 2e-7
)
first_gap = lambda e1, e2: (
    lambda idx: grid[
        idx[: np.argmax(np.diff(idx) > 1) + 1] if np.any(np.diff(idx) > 1) else idx
    ]
)(np.flatnonzero(np.abs(rhs(grid, e1, e2)) > 1))
print(
    "case              largest relative frequency error   first gap (FEM)       first gap (exact)     gap / midgap (FEM)"
)
for name, (e1, e2) in cases.items():
    k = wavenumbers[:, None] * np.ones_like(bands[name])
    steep = np.abs(slope(bands[name], k, e1, e2)) > 1e-2
    error = np.max(
        np.abs(mismatch(bands[name], k, e1, e2) / slope(bands[name], k, e1, e2))[steep]
        / bands[name][steep]
    )
    exact = first_gap(e1, e2)
    computed = (bands[name][-1, 0], bands[name][-1, 1])
    print(
        f"{name:16s} {error:30.1e}   {computed[0]:.4f} - {computed[1]:.4f}       "
        + (
            f"{exact.min():.4f} - {exact.max():.4f}"
            if exact.size
            else "none           "
        )
        + f"       {2 * (computed[1] - computed[0]) / (computed[1] + computed[0]):9.2%}"
    )

# ## Figure
#
# The bands of the three cases (markers) on the exact dispersion relation (lines), in the
# normalized frequency $\omega a/2\pi c$; the gaps lie between the bands at the zone edge.

curves = {
    name: (
        lambda e1, e2: (
            np.arccos(np.clip(rhs(grid, e1, e2), -1, 1)) / np.pi,
            np.where(np.abs(rhs(grid, e1, e2)) <= 1, grid, np.nan),
        )
    )(*cases[name])
    for name in cases
}
fig = plots.figure(1, 3, size=(20, 6))
for panel, name in enumerate(cases):
    plots.line(
        points={
            "FEM bands": (np.repeat(wavenumbers * a / np.pi, 4), bands[name].ravel())
        },
        reference={"exact dispersion relation": curves[name]},
        ylimits=(0, 0.45),
        xlabel="k_x a / pi",
        ylabel="omega a / 2 pi c",
        title=f"{name}: epsilon {cases[name][0]:.0f} / {cases[name][1]:.0f}",
        legend="upper left",
        ax=fig[panel],
    )
plots.show()

# ## Going further
#
# Give the layers different thicknesses in `one_dimensional_photonic_crystal_bands_geometry.py`: the
# quarter-wave stack $n_1d_1=n_2d_2$ has the widest gap for a given contrast. A nonzero $k_y$ in the
# second Bloch condition gives the bands of oblique propagation, and `polarization="Hz"` those of
# the other polarization.
