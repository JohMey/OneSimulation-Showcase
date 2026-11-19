"""OneSimulation example: waterflood of an oil reservoir with two horizontal wells.
Requires the installed, activated OneSimulation runtime.

Water is injected through a horizontal well near the base of an oil reservoir and oil is produced
through a horizontal well under the cap rock. The reservoir is read from a STEP file and meshed
with hexahedra; the 3D two-phase Darcy model with Peaceman well connections gives the oil rate,
the water cut and the recovery over ten years. The volume balance of both phases is checked,
and the breakthrough and recovery on two grids.
"""

# # Waterflood with two horizontal wells
#
# Horizontal wells drain a thin oil reservoir along their whole length. In a waterflood a second
# horizontal well near the base injects water, which pushes the oil up towards the producer; the
# heavier water stays below the oil, so the displacement is stabilised by gravity. Until water
# reaches the producer the oil rate equals the injection rate; after breakthrough the well
# produces more and more water.
#
# The model is incompressible immiscible two-phase flow in the 3D rock,
#
# $$\phi\,\frac{\partial S_\alpha}{\partial t}+\nabla\cdot\mathbf u_\alpha=q_\alpha,\qquad
#   \mathbf u_\alpha=-\frac{k\,k_{r\alpha}(S_w)}{\mu_\alpha}\big(\nabla p-\rho_\alpha\mathbf g\big),\qquad\alpha=w,o ,$$
#
# with Corey relative permeabilities $k_{rw}=S_w^2$, $k_{ro}=(1-S_w)^2$. Each well is connected to
# the grid cells it passes through by the well index of Peaceman (Soc. Petrol. Eng. J. 23, 1983,
# 531-543), $\mathrm{WI}=2\pi k\,\Delta x/\ln(r_e/r_w)$, with the equivalent radius $r_e$ of the cell.
#
# **Setup.** A reservoir 600 m × 300 m × 30 m, porosity 0.25, permeability 100 mD, water
# viscosity 0.5 mPa s and density 1000 kg/m³, oil viscosity 2 mPa s and density 850 kg/m³,
# initially full of oil. Both wells run 400 m along the middle of the reservoir, radius 0.1 m;
# the injector 4.5 m above the base injects 500 m³ of water per day, the producer 4.5 m below the cap
# rock is held at a bottom-hole pressure of 20 MPa. The reservoir comes from the CAD file
# `horizontal_wells_waterflood.step`, written by `horizontal_wells_waterflood_geometry.py` with the
# CAD objects of OneSimulation; all its faces are sealed.
#
# **Checks.** For incompressible fluids the volumes must balance exactly: injected water equals
# produced fluid, and the oil produced equals the oil that left the pores; before breakthrough the
# oil rate equals the injection rate. The breakthrough time and the recovery are compared on two
# grids.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import CadHexahedralMesher
from onesim.physics.geotechnics import (
    ImmiscibleReservoir,
    ReservoirPerforation,
    ReservoirWell,
)
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the reservoir block; it is meshed with mapped hexahedra, 30 m × 33 m × 5 m and,
# for the comparison, 50 m × 60 m × 7.5 m.

reservoir = load("horizontal_wells_waterflood.step")
length, width, thickness = reservoir.size
grids = {"fine": (20, 9, 6), "coarse": (12, 5, 4)}
meshes = {
    name: CadHexahedralMesher(cells=cells).generate(reservoir)
    for name, cells in grids.items()
}

view = plots.Camera(elevation=28, azimuth=-60)
fig = plots.figure(1, 2, size=(15, 5.5))
plots.geometry(
    reservoir,
    selection={
        "base": reservoir.boundaries.z_min,
        "cap rock": reservoir.boundaries.z_max,
    },
    labels=False,
    camera=view,
    title="Reservoir block",
    ax=fig[0],
)
plots.mesh(
    meshes["fine"],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{meshes['fine'].cell_count} hexahedra",
    ax=fig[1],
)
plots.show()

# ## Wells
#
# A well perforates every cell whose centre lies on its trajectory, within half a cell; the
# Peaceman index of each perforation uses the cell size across the well.

day, year = 86400.0, 365.25 * 86400.0
rock = ImmiscibleReservoir(
    porosity=0.25,
    permeability=1e-13,
    water_viscosity=0.5e-3,
    oil_viscosity=2e-3,
    water_density=1000.0,
    oil_density=850.0,
)
wells = {}
for name, (nx, ny, nz) in grids.items():
    size = np.array([length / nx, width / ny, thickness / nz])
    centres = np.asarray(meshes[name].coordinates)[np.asarray(meshes[name].cells)].mean(
        axis=1
    )
    along = lambda height: np.flatnonzero(
        (np.abs(centres[:, 0] - length / 2) < 200.0)
        & (np.abs(centres[:, 1] - width / 2) < size[1] / 2)
        & (np.abs(centres[:, 2] - height) < size[2] / 2)
    )
    perforate = lambda cells: tuple(
        ReservoirPerforation.peaceman(
            cell=int(cell),
            transverse_lengths=(size[1], size[2]),
            transverse_permeabilities=(1e-13, 1e-13),
            completion_length=size[0],
            radius=0.1,
        )
        for cell in cells
    )
    wells[name] = (
        ReservoirWell(
            name="injector",
            perforations=perforate(along(4.5)),
            rate=500.0 / day,
            pressure_limit=40e6,
        ),
        ReservoirWell(
            name="producer",
            perforations=perforate(along(thickness - 4.5)),
            pressure=20e6,
        ),
    )

# ## Model
#
# Ten years with output every two months; gravity points down.

months = np.linspace(0.0, 10 * year, 61)
results = {
    name: rock.solve(
        mesh=meshes[name],
        water_saturation=0.0,
        times=months,
        max_step=10 * day,
        wells=wells[name],
        gravity=(0.0, 0.0, -9.81),
    )
    for name in grids
}

# ## Results
#
# Cumulative volumes from the well records (positive into the reservoir), the volume balance, the
# rates from the cumulative volumes, breakthrough (water cut above 1 %) and recovery.

print(
    f"{'grid':7s} {'cells':>6s} {'perforations':>13s} {'water balance [m³]':>19s} {'oil balance [m³]':>17s}"
    f" {'oil rate before bt / injection':>31s} {'breakthrough [years]':>21s} {'recovery after 10 years [%]':>28s}"
)
rates = {}
for name, result in results.items():
    pore_volume = result.pore_volume.sum()
    in_place = result.water_saturation @ result.pore_volume
    injected, produced_water, produced_oil = (
        result.well_phase_volume[:, 0, 0],
        -result.well_phase_volume[:, 1, 0],
        -result.well_phase_volume[:, 1, 1],
    )
    oil_rate, water_rate = np.diff(produced_oil) / np.diff(months), np.diff(
        produced_water
    ) / np.diff(months)
    cut = water_rate / (water_rate + oil_rate)
    rates[name] = (oil_rate, cut)
    breakthrough = months[1:][np.argmax(cut > 0.01)] / year
    print(
        f"{name:7s} {meshes[name].cell_count:6d} {len(wells[name][0].perforations):6d} + {len(wells[name][1].perforations):<4d}"
        f" {np.abs(injected - produced_water - in_place).max():19.2e} {np.abs(produced_oil - in_place).max():17.2e}"
        f" {oil_rate[0] * day / 500.0:31.6f} {breakthrough:21.2f} {100 * produced_oil[-1] / pore_volume:28.2f}"
    )

# ## Figure
#
# The water saturation in the vertical section through the wells after two and six years, and the
# oil rate and water cut at the producer on both grids.

fine = results["fine"]
section = np.abs(fine.cell_centers[:, 1] - width / 2) < width / 30
x, z = fine.cell_centers[section, 0], fine.cell_centers[section, 2]
order = np.lexsort((x, z))
shape = (grids["fine"][2], grids["fine"][0])
mid_months = (months[1:] + months[:-1]) / 2 / year
fig = plots.figure(2, 2, size=(16, 9))
for panel, years in ((fig[0], 2), (fig[1], 6)):
    plots.heatmap(
        np.unique(x),
        np.unique(z),
        fine.water_saturation[6 * years, section][order].reshape(shape).T,
        limits=(0, 1),
        aspect=5.0,
        label="water saturation",
        xlabel="x [m]",
        ylabel="z [m]",
        title=f"Section through the wells after {years} years",
        ax=panel,
    )
plots.line(
    y={
        f"{name} grid": (mid_months, oil_rate * day)
        for name, (oil_rate, cut) in rates.items()
    },
    hlines={"injection rate": 500.0},
    ylimits=(0, 550),
    xlabel="time [years]",
    ylabel="oil rate [m³/day]",
    title="Oil production",
    legend="lower left",
    ax=fig[2],
)
plots.line(
    y={
        f"{name} grid": (mid_months, 100 * cut)
        for name, (oil_rate, cut) in rates.items()
    },
    ylimits=(0, 100),
    xlabel="time [years]",
    ylabel="water cut [%]",
    title="Water cut at the producer",
    legend="upper left",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Lengthen the wells, place them side by side instead of one above the other, or lower the
# vertical permeability with a permeability tensor: the breakthrough moves and the recovery at
# breakthrough changes. Switch off gravity to see the water slump less favourably.
