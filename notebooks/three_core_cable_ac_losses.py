"""OneSimulation example: AC resistance and sheath losses of a three-core power cable (2D).
Requires the installed, activated OneSimulation runtime.

The cross-section of a three-core 220 kV cable, read from STEP files, carries balanced 50 Hz phase
currents. The eddy-current model resolves the skin and proximity effects in the copper conductors
and the currents induced in the lead sheaths bonded at both ends. The AC resistance and the sheath
loss factor are compared with the formulas of IEC 60287-1-1.
"""

# # AC resistance and sheath losses of a three-core power cable
#
# A submarine HVAC cable carries three phases in one jacket: copper conductors of 500 mm² (radius
# 12.6 mm) insulated with XLPE, each core in a lead sheath (45 to 48 mm radius), the cores touching in a
# triangle with centre spacing $s=101$ mm. At 50 Hz the skin depth of copper (about 10 mm) is
# comparable with the conductor radius, so the current crowds towards the surface (skin effect) and
# towards the sides away from the neighbours (proximity effect); both raise the resistance above its
# DC value. The magnetic field of the phases also drives currents in the sheaths, which are bonded
# to each other and to earth at both ends of the cable and therefore carry circulating currents.
#
# OneSimulation solves the 2D eddy-current problem $\nabla\times(\mu_0^{-1}\nabla\times A_z)+
# \mathrm j\omega\sigma A_z=J_z$ with `EddyCurrents`: the conductors are massive coils carrying the
# balanced currents $\hat I e^{-\mathrm j2\pi k/3}$, $\hat I=1000$ A; the sheaths are massive
# conductors with zero voltage per unit length (bonded at both ends).
#
# | region | conductivity |
# |---|---|
# | copper at 90 °C | 45.5 MS/m |
# | lead at 80 °C | 3.77 MS/m |
# | insulation, filler, surroundings | 0 |
#
# | boundary | condition |
# |---|---|
# | outer circle, $r=400$ mm | magnetic insulation, $A_z=0$ |
#
# The surroundings come from `three_core_cable_ac_losses.step`, the conductors and sheaths from one
# file each, `three_core_cable_ac_losses_conductor_a.step` to `..._sheath_c.step`; the script
# `three_core_cable_ac_losses_geometry.py` builds them with the CAD objects of OneSimulation.
#
# **Reference** (IEC 60287-1-1:2006, §2.1.2 and §2.3.3; Anders, *Rating of Electric Power Cables*,
# 1997, ch. 6). With the DC resistance $R'$ per metre and $x^2=8\pi f\cdot10^{-7}/R'$:
#
# $$y_s=\frac{x^4}{192+0.8x^4},\qquad
#   y_p=\frac{x^4}{192+0.8x^4}\Big(\frac{d_c}{s}\Big)^2\Big[0.312\Big(\frac{d_c}{s}\Big)^2+\frac{1.18}{\frac{x^4}{192+0.8x^4}+0.27}\Big],\qquad
#   R_\mathrm{ac}=R'(1+y_s+y_p),$$
#
# and for sheaths bonded at both ends, with the sheath resistance $R_s$ per metre, the mean sheath
# diameter $d$ and $X=2\omega\cdot10^{-7}\ln(2s/d)$, the loss factor
# $\lambda_1=\dfrac{R_s}{R_\mathrm{ac}}\,\dfrac{1}{1+(R_s/X)^2}$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability, regionwise
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.magnetostatics import (
    Coil,
    EddyCurrents,
    MagneticInsulation,
    joule_losses,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `section` cuts the plates at mid-depth. Quadratic triangles of 1.5 mm in the conductors (seven per
# skin depth), 1 mm in the sheaths and up to 40 mm far away; a second mesh with half the sizes in the
# metal checks the losses.

plates = {"surroundings": load("three_core_cable_ac_losses.step")} | {
    f"{kind} {k}": load(f"three_core_cable_ac_losses_{kind}_{k}.step")
    for kind in ("conductor", "sheath")
    for k in "abc"
}
cut = lambda part: part.section(
    (0, 0, 1), origin=(0, 0, 0.001), planar_thickness=1.0
)
disc = cut(plates["surroundings"])
regions = {
    name: cut(part).named(name.replace(" ", "_"))
    for name, part in plates.items()
    if name != "surroundings"
}
meshes = {
    scale: TriangularMesher(
        target_size=0.04,
        embedded_regions=tuple(regions.values()),
        grading=0.25,
        region_sizes={
            region: (0.0015 if "conductor" in name else 0.001) * scale
            for name, region in regions.items()
        },
    ).generate(disc)
    for scale in (1.0, 0.5)
}

fig = plots.figure(1, 2, size=(14, 6.5))
plots.geometry(
    [part.named(name) for name, part in plates.items() if name != "surroundings"],
    camera=plots.Camera(elevation=60, azimuth=-90),
    title="Conductors and lead sheaths (STEP)",
    ax=fig[0],
)
plots.mesh(
    meshes[1.0],
    groups=("all",),
    legend=False,
    title=f"{meshes[1.0].cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Massive coils for the three phases and the three sheaths; the conductivity is given region by
# region.

current, frequency = 1000.0, 50.0
sigma = regionwise(
    tuple(
        (region, 4.55e7 if "conductor" in name else 3.77e6)
        for name, region in regions.items()
    ),
    default=0.0,
    unit="S/m",
)
coils = tuple(
    Coil(
        regions[f"conductor {k}"],
        conductor="massive",
        current=current * np.exp(-2j * np.pi * n / 3),
    )
    for n, k in enumerate("abc")
) + tuple(Coil(regions[f"sheath {k}"], conductor="massive", voltage=0.0) for k in "abc")
solve = lambda mesh: (
    lambda m: (
        Simulation(
            problem=Problem(
                name="Three-core cable",
                geometry=disc,
                physics=(m,),
                boundary_conditions=(
                    MagneticInsulation(boundary=disc.boundaries.all, physics=m),
                ),
            ),
            mesh=mesh,
            numerics=Numerics(
                spatial=FiniteElement(element=Lagrange(order=2), quadrature_order=4),
                linear_solver=ComplexSparseDirect(),
            ),
        ).solve_stationary(),
        m,
    )
)(
    EddyCurrents(
        domain=disc.domain(),
        permeability=VacuumPermeability(),
        conductivity=sigma,
        frequency=frequency,
        coils=coils,
    )
)
runs = {scale: solve(mesh) for scale, mesh in meshes.items()}

# ## Results
#
# The conductor losses give the AC resistance per phase, $R_\mathrm{ac}=P_c/(3\hat I^2/2)$; the
# sheath losses give the loss factor $\lambda_1=P_s/P_c$.

area, sheath_area = np.pi * 0.0126**2, np.pi * (0.048**2 - 0.045**2)
dc = 1 / (4.55e7 * area)
rs = 1 / (3.77e6 * sheath_area)
x4 = (8 * np.pi * frequency * 1e-7 / dc) ** 2
f4 = x4 / (192 + 0.8 * x4)
ys, yp = f4, f4 * (0.0252 / 0.101) ** 2 * (
    0.312 * (0.0252 / 0.101) ** 2 + 1.18 / (f4 + 0.27)
)
rac_iec = dc * (1 + ys + yp)
reactance = 2 * 2 * np.pi * frequency * 1e-7 * np.log(2 * 0.101 / 0.0930)
lambda_iec = rs / rac_iec / (1 + (rs / reactance) ** 2)
print(
    f"DC resistance R' = {1e6 * dc:.3f} uOhm/m; IEC 60287: y_s = {ys:.4f}, y_p = {yp:.4f}, R_ac = {1e6 * rac_iec:.3f} uOhm/m; sheath loss factor {lambda_iec:.4f}"
)
print(
    "mesh                 conductor losses [W/m]   R_ac [uOhm/m]   R_ac / R'   sheath losses [W/m]   loss factor lambda_1"
)
for scale, (result, m) in runs.items():
    conductor_loss = sum(
        joule_losses(result, m, regions[f"conductor {k}"]) for k in "abc"
    )
    sheath_loss = sum(joule_losses(result, m, regions[f"sheath {k}"]) for k in "abc")
    rac = conductor_loss / (3 * current**2 / 2)
    print(
        f"{'coarse' if scale == 1.0 else 'fine':8s} {meshes[scale].cell_count:7d} cells {conductor_loss:18.3f} {1e6 * rac:15.3f} {rac / dc:11.4f} {sheath_loss:19.3f} {sheath_loss / conductor_loss:20.4f}"
    )
print(
    f"fine mesh against IEC 60287: R_ac {rac / rac_iec - 1:+.1%}, loss factor {sheath_loss / conductor_loss / lambda_iec - 1:+.1%}"
)

# ## Figure
#
# The magnitude of the current density at the instant of the peak current in phase a, on the fine
# mesh: the current crowds to the outer sides of the conductors, and the sheaths carry the induced
# circulating currents.

result, m = runs[0.5]
x, y = np.asarray(meshes[0.5].coordinates).T
values = (
    np.abs(result.field(m.current_density).evaluate(np.column_stack((x, y)))[:, 2])
    / 1e6
)
fig = plots.figure(1, 2, size=(16, 6.5))
plots.heatmap(
    1e3 * x,
    1e3 * y,
    values,
    triangles=np.asarray(meshes[0.5].cells),
    limits=(0, np.percentile(values, 99.9)),
    label="|J_z| [A/mm^2]",
    xlimits=(-110, 110),
    ylimits=(-95, 120),
    xlabel="x [mm]",
    ylabel="y [mm]",
    title="Current density, both-ends bonded sheaths",
    ax=fig[0],
)
radii = np.linspace(-0.0125, 0.0125, 101)
centre = np.array((0.0, 0.101 / np.sqrt(3)))
plots.line(
    1e3 * radii,
    {
        "across conductor a, towards the cable axis": np.abs(
            result.field(m.current_density).evaluate(
                centre + np.column_stack((0 * radii, radii))
            )[:, 2]
        )
        / 1e6
    },
    hlines={"DC current density": current / area / 1e6},
    ylimits=(0, None),
    xlabel="position across the conductor [mm]",
    ylabel="|J_z| [A/mm^2]",
    title="Skin and proximity effect in conductor a",
    legend="upper left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Replace `voltage=0.0` by `current=0.0` for the sheaths to bond them at one end only: the circulating
# currents vanish and only the much smaller eddy-current losses remain. Adding the steel wire armour
# with a permeability and conductivity of its own raises both conductor and sheath losses.
