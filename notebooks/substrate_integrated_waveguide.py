"""OneSimulation example: a microstrip-fed substrate integrated waveguide in 3D.
Requires the installed, activated OneSimulation runtime.

Two rows of plated vias between the copper planes of a circuit board form a waveguide in the
substrate. Fed by microstrip lines through tapers, it passes signals only above the cutoff of its
TE10 mode. The board is read from STEP files and solved in 3D with Nédélec elements, the copper as a
conducting sheet, the vias as holes with conducting walls and TEM ports on the feed lines; the
cutoff is compared with the equivalent-width formula of the SIW.
"""


# # Substrate integrated waveguide
#
# A substrate integrated waveguide (SIW) is a rectangular waveguide built into a printed circuit
# board: the top and bottom copper are its broad walls, two rows of plated vias its narrow walls.
# Like a hollow waveguide it is a high-pass filter: below the cutoff of its TE$_{10}$ mode nothing
# passes. Here the vias have a diameter of 1.0 mm and a pitch of 1.8 mm, the rows are 11.81 mm apart
# on a 0.787 mm substrate of relative permittivity 2.2; 50 Ω microstrip lines feed the guide through
# tapers that turn the microstrip field into the TE$_{10}$ field.
#
# OneSimulation solves the time-harmonic Maxwell equation
#
# $$\nabla\times\big(\mu_r^{-1}\nabla\times\mathbf E\big)-k_0^2\,\varepsilon_r\,\mathbf E=0$$
#
# with first-order Nédélec elements. The top copper is a sheet without thickness (`region_interface`
# of a thin layer of air shaped like it, lying over the via holes); the vias are holes in the
# substrate with perfectly conducting walls. The feed lines end in `TEMPort`s that compute their
# quasi-TEM mode from the electrostatic potential of the cross-section; port 1 is fed and
# `scattering_parameters` gives reflection and transmission in the impedance of the feed lines.
#
# | faces | condition |
# |---|---|
# | `port_1`, `port_2` | TEM port, signal conductor: the copper |
# | `ground`, `vias`, copper | perfect electric conductor |
# | `open` (lid and long sides) | impedance boundary $Y_0=\sqrt{\varepsilon_0/\mu_0}$ |
#
# **Reference** (Y. Cassivi, L. Perregrini, P. Arcioni, M. Bressan, K. Wu, G. Conciauro, *Dispersion
# characteristics of substrate integrated rectangular waveguide*, IEEE MWCL 12 (2002) 333-335;
# D. Deslandes, K. Wu, IEEE MWCL 11 (2001) 68-70). For $p/d<2$ and $d/w<0.2$ the SIW behaves like a
# rectangular waveguide of width
#
# $$w_{\rm eff}=w-\frac{d^2}{0.95\,p},\qquad f_c=\frac{c}{2\,w_{\rm eff}\sqrt{\varepsilon_r}},
#   \qquad \beta=\sqrt{\varepsilon_r k_0^2-(\pi/w_{\rm eff})^2},$$
#
# 11.225 mm and 9.003 GHz here. Above the cutoff the TE$_{10}$ wave travels with the propagation
# constant $\beta$ of that waveguide; below it the wave decays and the transmission collapses.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.exchange import load
from onesim.materials import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.materials.electromagnetic import IsotropicElectromagneticMaterial, RegionwiseElectromagneticMaterial
from onesim.meshing import LocalRefinement, TetrahedralMesher, region_interface
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (FrequencyDomainMaxwell, ImpedanceBoundary, PerfectElectricConductor, TEMPort,
                                             scattering_parameters)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

light, eta0 = 1 / np.sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY), np.sqrt(VACUUM_PERMEABILITY / VACUUM_PERMITTIVITY)
rows, diameter, pitch, permittivity, height = 11.81e-3, 1.0e-3, 1.8e-3, 2.2, 0.787e-3

# ## Geometry and mesh
#
# `load` reads the air with the vias removed, the substrate and the thin layer shaped like the top
# copper; substrate and layer become regions of one mesh. Tetrahedra of 2 mm follow
# the copper, 2.5 mm the substrate, 0.5 mm the vias and 4 mm the air.

air = load("substrate_integrated_waveguide.step")
substrate = load("substrate_integrated_waveguide_substrate.step").named("substrate")
copper = load("substrate_integrated_waveguide_copper.step").named("copper")
faces = air.boundaries
mesh = TetrahedralMesher(target_size=4e-3, embedded_regions=(substrate, copper), refinements=(
    LocalRefinement(region=copper, target_size=2.0e-3), LocalRefinement(region=substrate, target_size=2.5e-3),
    LocalRefinement(region=faces.vias, target_size=0.5e-3))).generate(air)
sheet = region_interface(mesh, copper, substrate)

view = plots.Camera(elevation=40, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry([copper, load("substrate_integrated_waveguide_vias.step")], labels=False, camera=view, length_unit="mm",
               title="Top copper and vias (STEP)", ax=fig[0])
plots.mesh(mesh, groups=("all",), legend=False, camera=view, length_unit="mm", title=f"{mesh.cell_count} tetrahedra", ax=fig[1])
plots.show()

# ## Model
#
# Port 1 supplies 1 W. One frequency is below the predicted cutoff and three are above it.
# The final solve also supplies the field for the phase measurement at 12 GHz.

material = RegionwiseElectromagneticMaterial({"substrate": IsotropicElectromagneticMaterial(relative_permittivity=permittivity)},
                                             default=IsotropicElectromagneticMaterial())
frequencies = np.array([8.0e9, 9.3e9, 10.5e9, 12.0e9])
numerics = Numerics(spatial=FiniteElement(element=Nedelec(order=1)), linear_solver=ComplexSparseDirect())
sweep = {}
for frequency in frequencies:
    m = FrequencyDomainMaxwell(domain=air.volume, frequency=frequency, material=material,
                               phasor_convention="exp(-i*omega*t)", amplitude_convention="peak")
    boundaries = (TEMPort(physics=m, boundary=faces.port_1, signal=sheet, incident_power=1.0),
                  TEMPort(physics=m, boundary=faces.port_2, signal=sheet),
                  PerfectElectricConductor(physics=m, boundary=faces.ground | faces.vias | sheet),
                  ImpedanceBoundary(physics=m, boundary=faces.open, admittance=1 / eta0))
    problem = Problem(name="Via-fenced waveguide", geometry=air, physics=(m,), boundary_conditions=boundaries)
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    sweep[frequency] = scattering_parameters(result, m)

# ## Results
#
# Transmission and reflection over the band against the cutoff of the equivalent waveguide; the
# power that is neither reflected nor transmitted leaves through the open top of the box, radiated by
# the feed lines and tapers. At 12 GHz the reflected wave is weak, so the field in the
# middle of the guide is one travelling wave: the slope of its phase along the guide is the
# propagation constant, compared with the equivalent waveguide and with a guide as wide as the via
# rows.

dB = lambda x: 20 * np.log10(max(abs(x), 1e-6))
effective = rows - diameter**2 / (0.95 * pitch)
cutoff = light / (2 * effective * np.sqrt(permittivity))
print(f"equivalent width {effective * 1e3:.3f} mm, TE10 cutoff {cutoff / 1e9:.3f} GHz")
print("f [GHz]   |S21| [dB]   |S11| [dB]   |S11|^2 + |S21|^2")
for f, S in sweep.items():
    print(f"{f / 1e9:6.2f} {dB(S[(1, 0)]):12.2f} {dB(S[(0, 0)]):12.2f} {abs(S[(0, 0)]) ** 2 + abs(S[(1, 0)]) ** 2:18.4f}")
print(f"below the cutoff ({frequencies[0] / 1e9:.1f} GHz) |S21| = {dB(sweep[frequencies[0]][(1, 0)]):.1f} dB; "
      f"above it ({frequencies[-2] / 1e9:.1f} and {frequencies[-1] / 1e9:.1f} GHz) {dB(sweep[frequencies[-2]][(1, 0)]):.2f} and {dB(sweep[frequencies[-1]][(1, 0)]):.2f} dB")
x = np.linspace(-8e-3, 8e-3, 81)
along = result.field(m.electric_field).evaluate(np.column_stack((x, 0 * x, 0 * x + height / 2)))[:, 2]
phase = np.unwrap(np.angle(along))
k0 = 2 * np.pi * 12e9 / light
beta = {"equivalent width": np.sqrt(permittivity * k0**2 - (np.pi / effective) ** 2), "width of the via rows": np.sqrt(permittivity * k0**2 - (np.pi / rows) ** 2)}
measured = abs(np.polyfit(x, phase, 1)[0])
print(f"propagation constant at 12 GHz from the phase along the guide: {measured:.1f} rad/m; equivalent waveguide {beta['equivalent width']:.1f} rad/m "
      f"({measured / beta['equivalent width'] - 1:+.1%}), guide as wide as the via rows {beta['width of the via rows']:.1f} rad/m ({measured / beta['width of the via rows'] - 1:+.1%})")

# ## Figure
#
# Left: transmission and reflection over the band, with the cutoff of the equivalent waveguide.
# Middle: the phase of the vertical field along the middle of the guide at 12 GHz against the two
# waveguide widths. Right: the electric field in the middle of the substrate at 12 GHz with port 1
# fed: the microstrip field spreads in the taper into the half sine of the TE$_{10}$ mode between the
# via rows.


fig = plots.figure(1, 3, size=(24, 6.5))
plots.line(frequencies / 1e9, {"|S21|": [dB(S[(1, 0)]) for S in sweep.values()], "|S11|": [dB(S[(0, 0)]) for S in sweep.values()]}, markers=True,
           vlines={"TE10 cutoff, equivalent width": cutoff / 1e9}, xlabel="frequency [GHz]", ylabel="[dB]", ylimits=(-60, 2),
           title="S-parameters of the SIW", legend="lower right", ax=fig[0])
plots.line(points={"computed": (x * 1e3, phase - phase[0])}, reference={f"{name}": (x * 1e3, np.sign(phase[-1] - phase[0]) * value * (x - x[0])) for name, value in beta.items()},
           xlabel="x [mm]", ylabel="phase of E_z [rad]", title="Phase along the guide, 12 GHz", legend="upper left", ax=fig[1])
plots.section(result.field(m.electric_field), plane=Plane(origin=(0.0, 0.0, 0.5 * height), normal=(0.0, 0.0, 1.0)),
              component=lambda E, p: np.linalg.norm(np.abs(E), axis=1), limits="robust", subdivisions=1, length_unit="mm",
              label="|E| [V/m]", title="Electric field in the substrate, 12 GHz", ax=fig[2])
plots.show()

# ## Going further
#
# Change the via pitch or the row spacing in `substrate_integrated_waveguide_geometry.py`: the cutoff
# follows the equivalent width. With a pitch above twice the diameter the via walls start to leak.
