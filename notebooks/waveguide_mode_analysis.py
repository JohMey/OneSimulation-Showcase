"""OneSimulation example: mode analysis of a rectangular waveguide cross-section.
Requires the installed, activated OneSimulation runtime.

The first eight TE and TM eigenmodes of a WR-90 waveguide are computed on its meshed cross-section,
read from a STEP file. Cutoff frequencies and mode patterns are checked against the closed-form
modes of the rectangular guide, with a mesh study of the fundamental mode.
"""

# # Mode analysis of a rectangular waveguide cross-section
#
# A hollow metal waveguide carries transverse electric (TE) and transverse magnetic (TM) modes.
# Each mode has a pattern over the cross-section and a cutoff frequency below which it cannot
# propagate. Mode analysis is the 2D eigenvalue problem on the cross-section $S$: for TE modes the
# axial magnetic field $H_z$, for TM modes the axial electric field $E_z$, satisfies
#
# $$\nabla_t^2\psi+k_c^2\,\psi=0\ \text{in }S,\qquad
#   \partial_n H_z=0\ \text{(TE)},\quad E_z=0\ \text{(TM) on the wall},\qquad f_c=\frac{c\,k_c}{2\pi},$$
#
# and the transverse electric field follows from the gradient of $\psi$.
#
# OneSimulation solves this eigenproblem with linear triangles on the **port
# cross-section** of a waveguide model: `NumericalWaveguideMode("TE", order=n)` binds the $n$-th TE
# eigenmode computed on the meshed port face. Here the face belongs to a 1.5 mm long piece of a WR-90
# guide (inner size $a\times b=22.86\times10.16$ mm, the standard X-band guide) read from the CAD
# file `waveguide_mode_analysis.step`, written by `waveguide_mode_analysis_geometry.py` with the CAD
# objects of OneSimulation. Each mode is bound in a driven solve at 25 GHz, above all cutoffs
# considered.
#
# **Reference** (D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley 2012, §3.3; R. E. Collin,
# *Foundations for Microwave Engineering*, 2nd ed., §3.12). For the TE$_{mn}$ and TM$_{mn}$ modes
#
# $$f_{c,mn}=\frac c2\sqrt{\Big(\frac ma\Big)^2+\Big(\frac nb\Big)^2},\qquad
#   H_z\propto\cos\frac{m\pi x}a\cos\frac{n\pi y}b\ \text{(TE)},\qquad
#   E_z\propto\sin\frac{m\pi x}a\sin\frac{n\pi y}b\ \text{(TM)},$$
#
# with $m,n\ge0$, not both zero, for TE and $m,n\ge1$ for TM. In WR-90 the TE$_{10}$ mode propagates
# alone between 6.56 and 13.12 GHz.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import IsotropicElectromagneticMaterial
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell,
    NumericalWaveguideMode,
    PerfectElectricConductor,
    WaveguidePort,
)
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The mode solve uses the triangles of the port face. Three meshes are built for the mesh study;
# the finest, 0.5 mm, resolves the eighth mode with about eleven triangles per half wavelength.

guide = load("waveguide_mode_analysis.step")
faces = guide.boundaries
a, b, length = guide.size
sizes = (2.0e-3, 1.0e-3, 0.5e-3)
meshes = {size: TetrahedralMesher(target_size=size).generate(guide) for size in sizes}

fig = plots.figure(1, 2, size=(13, 5))
plots.geometry(
    guide,
    selection={"port cross-section": faces.port, "short": faces.short},
    labels=False,
    camera=plots.Camera(elevation=-30, azimuth=-55),
    title="Piece of the WR-90 guide",
    ax=fig[0],
)
plots.mesh(
    meshes[sizes[-1]],
    groups=("all",),
    legend=False,
    camera=plots.Camera(elevation=-30, azimuth=-55),
    title=f"Finest mesh: {meshes[sizes[-1]].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# One driven solve per requested mode; only its port eigenmode is used here. The bound mode
# exposes its cutoff frequency and its transverse electric field on the cross-section.

frequency = 25e9
model = FrequencyDomainMaxwell(
    domain=guide.volume,
    frequency=frequency,
    material=IsotropicElectromagneticMaterial(),
    phasor_convention="exp(-i*omega*t)",
    amplitude_convention="peak",
)
numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=1)), linear_solver=ComplexSparseDirect()
)
port_mode = (
    lambda family, order, mesh: Simulation(
        problem=Problem(
            name="WR-90 port",
            geometry=guide,
            physics=(model,),
            boundary_conditions=(
                WaveguidePort(
                    boundary=faces.port,
                    physics=model,
                    mode=NumericalWaveguideMode(family, order=order),
                    incident_power=1.0,
                ),
                PerfectElectricConductor(
                    boundary=faces.walls | faces.short, physics=model
                ),
            ),
        ),
        mesh=mesh,
        numerics=numerics,
    )
    .solve_stationary()
    .operators[model.model_id]
    .ports[0][1]
)
requested = {
    "TE10": ("TE", 1),
    "TE20": ("TE", 2),
    "TE01": ("TE", 3),
    "TE11": ("TE", 4),
    "TE30": ("TE", 5),
    "TE21": ("TE", 6),
    "TM11": ("TM", 1),
    "TM21": ("TM", 2),
}
modes = {
    name: port_mode(family, order, meshes[sizes[-1]])
    for name, (family, order) in requested.items()
}
fundamental = [
    port_mode("TE", 1, meshes[size]).cutoff_frequency for size in sizes[:-1]
] + [modes["TE10"].cutoff_frequency]

# ## Comparison with the closed forms

c0 = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
cutoff = lambda m, n: c0 / 2 * np.hypot(m / a, n / b)
print("mode    computed [GHz]   closed form [GHz]   deviation")
for name, mode in modes.items():
    exact = cutoff(int(name[2]), int(name[3]))
    print(
        f"{name:6s} {mode.cutoff_frequency / 1e9:15.4f} {exact / 1e9:19.4f} {100 * (mode.cutoff_frequency / exact - 1):+10.3f} %"
    )
errors = [abs(value / cutoff(1, 0) - 1) for value in fundamental]
study = convergence(fundamental, sizes)
print(
    "TE10 mesh study: "
    + ", ".join(
        f"h = {1e3 * h:.1f} mm: {100 * error:.4f} %" for h, error in zip(sizes, errors)
    )
    + f"; observed order {study.order:.2f}"
)

x = np.linspace(0.02 * a, 0.98 * a, 25)
profile = modes["TE10"].electric_field(
    np.column_stack((x, np.full_like(x, b / 2), np.zeros_like(x)))
)
shape = np.abs(profile[:, 1]) / np.abs(profile[:, 1]).max()
print(
    f"TE10 pattern against sin(pi x / a): largest deviation {np.abs(shape - np.sin(np.pi * x / a)).max():.4f} of the peak"
)

# ## Figure
#
# The transverse electric field strength of the eight modes on the cross-section, each scaled to
# its maximum, and the mesh study of the fundamental cutoff.

gx, gy = np.linspace(0.002 * a, 0.998 * a, 91), np.linspace(0.002 * b, 0.998 * b, 41)
grid = np.column_stack(
    [value.ravel() for value in np.meshgrid(gx, gy, indexing="ij")]
    + [np.zeros(gx.size * gy.size)]
)
strength = {
    name: np.linalg.norm(mode.electric_field(grid), axis=1).reshape(gx.size, gy.size)
    for name, mode in modes.items()
}
fig = plots.figure(3, 3, size=(16, 9.5))
for index, (name, values) in enumerate(strength.items()):
    plots.heatmap(
        1e3 * gx,
        1e3 * gy,
        values / values.max(),
        limits=(0, 1),
        label="|E_t| / max",
        xlabel="x [mm]",
        ylabel="y [mm]",
        title=f"{name}: {modes[name].cutoff_frequency / 1e9:.3f} GHz",
        ax=fig[index],
    )
plots.convergence(
    list(sizes),
    {"TE10 cutoff": errors},
    order=2,
    xlabel="mesh size h [m]",
    ylabel="relative deviation",
    title="Mesh study of the TE10 cutoff",
    ax=fig[8],
)
plots.show()

# ## Going further
#
# Change the cross-section in `waveguide_mode_analysis_geometry.py` (for example WR-75 with
# 19.05 x 9.525 mm) and write the STEP file again, or replace the box by a ridged or elliptical
# profile, for which no closed form exists: the numerical port mode is computed the same way and
# feeds the 3D field of a longer guide directly.
