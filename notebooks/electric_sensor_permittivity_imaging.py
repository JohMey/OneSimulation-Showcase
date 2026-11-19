"""OneSimulation example: electric sensor - permittivity sweep over two objects in a box (3D electrostatics).
Requires the installed, activated OneSimulation runtime.

A voltage between the top and bottom electrodes of a box drives a field through two dielectric
objects standing on the bottom electrode. Two charge probes on the top electrode, one above each
object, read the surface charge. A batch of 3D solutions over the permittivities of both objects
shows how each probe images its own object, how much it sees of the other, and how a permittivity
is read back from a probe signal; every capacitance is checked against the variational bounds.
"""

# # Electric sensor: permittivity sweep over two objects in a box
#
# A capacitive sensor tells materials apart by their permittivity. The box is 5 m long, 3 m wide
# and $h=1$ m high, with an electrode on its top held at $V_0=1$ V and a grounded electrode on its
# bottom; the side walls are insulating. On the bottom electrode stand a rectangular bar of
# $1.0\times0.6$ m, 0.8 m high, centred at $x=y=1.5$ m, and a disc of radius 0.4 m, 0.5 m high,
# centred at $x=3.5$ m, $y=1.5$ m. The potential follows from
# $\nabla\cdot(\varepsilon_0\varepsilon_r\nabla V)=0$, with $V$ and $D_n$ continuous on the surfaces
# of the objects. Two probes on the top electrode, at $(1.5, 1.5, 1.0)$ m above the bar and at
# $(3.5, 1.5, 1.0)$ m above the disc, read the surface charge $\sigma=-\varepsilon_0E_z$.
#
# The batch sweeps the relative permittivity of the bar over 1, 2, 4 and 8 with the disc at 1 and
# at 3. One more solution, with the bar at $\varepsilon_r=3$, plays an unknown object whose
# permittivity is read back from the bar probe.
#
# The air box comes from the CAD file `electric_sensor_permittivity_imaging.step` with the face
# groups `top`, `bottom` and `walls`, the objects from `electric_sensor_permittivity_imaging_bar.step`
# and `..._disc.step`. The script `electric_sensor_permittivity_imaging_geometry.py` builds them
# with the CAD objects of OneSimulation.
#
# | face | condition |
# |---|---|
# | `top` | $V=V_0$ |
# | `bottom` | $V=0$ |
# | `walls` | $\mathbf D\cdot\mathbf n=0$ |
#
# **Reference.** With both objects at $\varepsilon_r=1$ the box is an ideal parallel-plate
# capacitor, $\sigma=\varepsilon_0V_0/h$ and $C=\varepsilon_0A/h$ exactly. For the other cases the
# two variational principles of electrostatics bound the capacitance from both sides (J. D.
# Jackson, *Classical Electrodynamics*, 3rd ed., Sec. 1.12 and the capacitance bounds in the
# problems of Chapter 1): Dirichlet's principle with the uniform field as trial potential gives
# $C_\mathrm{up}=\varepsilon_0\big(Ah+\sum_i(\varepsilon_{r,i}-1)V_i\big)/h^2$, Thomson's principle
# with a vertical displacement that is constant in each column gives
# $C_\mathrm{low}=\varepsilon_0\int_A\mathrm dA/(h-t+t/\varepsilon_r)$, with the object height $t$ in
# the column. The same series column, $\sigma_\mathrm{col}=\varepsilon_0V_0/(h-t+t/\varepsilon_r)$,
# is the charge a probe would read above a laterally infinite layer; above an object of finite
# size the fringing field lowers it.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.exchange import load
from onesim.materials import VacuumPermittivity
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential,
    PoissonEquation,
    ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the box with its named faces and the two objects; the tetrahedral mesher embeds
# the objects so that their surfaces run along faces of the mesh. Cells of 0.15 m in the air,
# 0.1 m in the objects. All cases of the batch share this mesh.

box = load("electric_sensor_permittivity_imaging.step")
faces = box.boundaries
bar, disc = (
    load(f"electric_sensor_permittivity_imaging_{name}.step")
    for name in ("bar", "disc")
)
length, width, height = box.size
mesh = TetrahedralMesher(
    target_size=0.15,
    embedded_regions=(bar, disc),
    refinements=(
        LocalRefinement(region=bar, target_size=0.1),
        LocalRefinement(region=disc, target_size=0.1),
    ),
).generate(box)

view = plots.Camera(elevation=30, azimuth=-60)
fig = plots.figure(1, 2, size=(15, 6))
plots.geometry(
    [bar.named("bar"), disc.named("disc")],
    camera=view,
    title="The two objects on the bottom electrode",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Air box 5 m x 3 m x 1 m: {mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Batch of solutions
#
# `regionwise` gives the objects their permittivity, with point tests from the CAD bounds of the
# bar and the disc. One problem per pair of permittivities, all on the same mesh with quadratic
# elements.

voltage, eps0 = 1.0, float(VacuumPermittivity().evaluate())
(bx0, by0, _), (bx1, by1, bar_height) = bar.bounding_box
(dx0, dy0, _), (dx1, dy1, disc_height) = disc.bounding_box
radius, centre = (dx1 - dx0) / 2, ((dx0 + dx1) / 2, (dy0 + dy1) / 2)
in_bar = (
    lambda p: (p[:, 0] >= bx0)
    & (p[:, 0] <= bx1)
    & (p[:, 1] >= by0)
    & (p[:, 1] <= by1)
    & (p[:, 2] <= bar_height)
)
in_disc = lambda p: (np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) <= radius) & (
    p[:, 2] <= disc_height
)
cases = [
    (eps_bar, eps_disc) for eps_disc in (1.0, 3.0) for eps_bar in (1.0, 2.0, 4.0, 8.0)
] + [(3.0, 1.0)]
models = {
    case: PoissonEquation(
        domain=box.volume,
        charge_density=0.0,
        permittivity=regionwise(
            ((in_bar, case[0] * eps0), (in_disc, case[1] * eps0)),
            default=eps0,
            unit="F/m",
        ),
    )
    for case in cases
}
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
results = {
    case: Simulation(
        problem=Problem(
            name="Electric sensor",
            geometry=box,
            physics=(model,),
            boundary_conditions=(
                FixedPotential(boundary=faces.top, physics=model, value=voltage),
                FixedPotential(boundary=faces.bottom, physics=model, value=0.0),
                ZeroNormalElectricDisplacement(boundary=faces.walls, physics=model),
            ),
        ),
        mesh=mesh,
        numerics=numerics,
    ).solve_stationary()
    for case, model in models.items()
}

# ## Results
#
# For every case: the two probe charges, the capacitance and its place between the bounds. Then
# the sensitivity of each probe to its own object and to the other one, and the permittivity of
# the "unknown" bar read back from the bar probe by interpolating the sweep with the disc at 1.

probes = np.array([[1.5, 1.5, height - 1e-6], [centre[0], centre[1], height - 1e-6]])
sigma = {
    case: -eps0 * result.field(models[case].electric_field).evaluate(probes)[:, 2]
    for case, result in results.items()
}
capacitance = {
    case: result.field(models[case].potential).reaction(faces.top) / voltage
    for case, result in results.items()
}
area, footprints = length * width, ((bx1 - bx0) * (by1 - by0), np.pi * radius**2)
column = lambda eps, t: eps0 * voltage / (height - t + t / eps)
bounds = {
    case: (
        sum(
            f * column(e, t)
            for f, e, t in zip(footprints, case, (bar_height, disc_height))
        )
        / voltage
        + eps0 * (area - sum(footprints)) / height,
        eps0
        / height**2
        * (
            area * height
            + (case[0] - 1) * footprints[0] * bar_height
            + (case[1] - 1) * footprints[1] * disc_height
        ),
    )
    for case in cases
}
print(
    f"parallel plates: σ = {1e12 * eps0 * voltage / height:.4f} pC/m^2, C = {1e12 * eps0 * area / height:.4f} pF"
)
print(
    "eps_bar eps_disc   σ bar probe   series col.   σ disc probe   C [pF]   C_low [pF]   C_up [pF]   between the bounds"
)
for case in cases[:-1]:
    low, up = bounds[case]
    place = (
        "exact" if up - low < 1e-18 else f"{(capacitance[case] - low) / (up - low):.3f}"
    )
    print(
        f"{case[0]:6.0f} {case[1]:8.0f} {1e12 * sigma[case][0]:13.4f} {1e12 * column(case[0], bar_height):13.4f}"
        f" {1e12 * sigma[case][1]:14.4f} {1e12 * capacitance[case]:8.4f} {1e12 * low:12.4f} {1e12 * up:11.4f} {place:>20s}"
    )
print(
    f"empty box (both eps_r = 1): σ deviates by {100 * np.max(np.abs(sigma[1.0, 1.0] / (eps0 * voltage / height) - 1)):.1e} %, "
    f"C by {100 * (capacitance[1.0, 1.0] / (eps0 * area / height) - 1):+.1e} %"
)
own = sigma[8.0, 1.0][0] - sigma[1.0, 1.0][0]
cross = sigma[1.0, 3.0][0] - sigma[1.0, 1.0][0]
print(
    f"bar probe: +{1e12 * own:.4f} pC/m^2 from eps_bar 1 -> 8, {1e12 * cross:+.4f} pC/m^2 from eps_disc 1 -> 3 "
    f"(crosstalk {100 * cross / own:.2f} % of the own signal)"
)
print(
    f"disc probe: {1e12 * (sigma[1.0, 3.0][1] - sigma[1.0, 1.0][1]):+.4f} pC/m^2 from eps_disc 1 -> 3, "
    f"{1e12 * (sigma[8.0, 1.0][1] - sigma[1.0, 1.0][1]):+.4f} pC/m^2 from eps_bar 1 -> 8"
)
sweep = [(1.0, 1.0), (2.0, 1.0), (4.0, 1.0), (8.0, 1.0)]
read_back = np.interp(
    sigma[3.0, 1.0][0], [sigma[case][0] for case in sweep], [case[0] for case in sweep]
)
print(
    f"unknown bar: probe reads {1e12 * sigma[3.0, 1.0][0]:.4f} pC/m^2 -> eps_r = {read_back:.2f} read back from the sweep (true 3)"
)

# ## Figure
#
# The surface charge on the top electrode for the bar at $\varepsilon_r=8$ and the disc at 3, the
# field strength in the section through both objects, the surface charge along the middle line
# $y=1.5$ m for the bar sweep, and both probe signals over the permittivity of the bar against the
# series-column value.

field = results[8.0, 3.0].field(models[8.0, 3.0].electric_field)
x = np.linspace(0.01, length - 0.01, 400)
line = np.column_stack((x, np.full_like(x, 1.5), np.full_like(x, height - 1e-6)))
eps_axis = np.linspace(1.0, 8.0, 71)
fig = plots.figure(2, 2, size=(15, 11))
plots.surface(
    field,
    boundary=faces.top,
    component=lambda E, position: -1e12 * eps0 * E[:, 2],
    limits=(8.0, 22.0),
    contours=15,
    subdivisions=2,
    label="σ [pC/m²]",
    title="Surface charge on the top electrode (bar 8, disc 3)",
    ax=fig[0],
)
plots.section(
    field,
    plane=Plane(origin=(0.0, 1.5, 0.0), normal=(0.0, 1.0, 0.0)),
    component=lambda E, position: np.linalg.norm(E, axis=1),
    limits=(0.0, 3.0),
    contours=13,
    subdivisions=2,
    label="|E| [V/m]",
    xlabel="x [m]",
    ylabel="z [m]",
    title="Field strength in the section y = 1.5 m (bar 8, disc 3)",
    ax=fig[1],
)
plots.line(
    x,
    {
        f"bar eps_r = {case[0]:.0f}": -1e12
        * eps0
        * results[case].field(models[case].electric_field).evaluate(line)[:, 2]
        for case in sweep
    },
    xlabel="x [m]",
    ylabel="σ on the top electrode [pC/m²]",
    title="Bar sweep along y = 1.5 m (disc at 1)",
    legend="upper right",
    ax=fig[2],
)
plots.line(
    points={
        "bar probe, disc at 1": (
            [c[0] for c in sweep],
            [1e12 * sigma[c][0] for c in sweep],
        ),
        "bar probe, disc at 3": (
            [c[0] for c in cases[4:8]],
            [1e12 * sigma[c][0] for c in cases[4:8]],
        ),
        "disc probe, disc at 3": (
            [c[0] for c in cases[4:8]],
            [1e12 * sigma[c][1] for c in cases[4:8]],
        ),
    },
    reference={
        "series column above the bar": (eps_axis, 1e12 * column(eps_axis, bar_height))
    },
    xlabel="relative permittivity of the bar",
    ylabel="probe charge [pC/m²]",
    title="Probe signals of the batch",
    legend="upper left",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Add the height of the bar as a second sweep parameter by writing one STEP file per height in
# `electric_sensor_permittivity_imaging_geometry.py`: tall objects of low permittivity and low
# objects of high permittivity can give the same probe charge, and a second probe position is
# needed to tell them apart.
