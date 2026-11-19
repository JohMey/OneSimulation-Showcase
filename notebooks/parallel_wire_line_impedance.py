"""OneSimulation example: characteristic impedance of a parallel-wire transmission line.
Requires the installed, activated OneSimulation runtime.

The air around two parallel wires is read from a STEP file and cut to its cross-section. The
electrostatic field gives the capacitance per metre from the charge on the wires and from the field
energy; with the inductance of the TEM wave this yields the characteristic impedance, which is
checked together with the potential against the closed-form line-charge solution.
"""

# # Characteristic impedance of a parallel-wire line
#
# Two parallel round wires of radius $a$ with centre spacing $D$ in air form a balanced
# transmission line (twin lead). Unlike a coaxial cable, the fields are not enclosed: they reach
# into the unbounded surroundings and decay away from the line. For the TEM wave the transverse
# electric field solves Laplace's equation of the cross-section,
#
# $$\nabla\cdot(\varepsilon_0\nabla\phi)=0,\qquad \phi=\pm V/2\ \text{on the two wires},$$
#
# and the magnetic field is the electric field turned by 90° and divided by $\eta_0=\sqrt{\mu_0/\varepsilon_0}$.
# The capacitance per metre follows from the charge on one wire, $C'=Q'/V$, or from the field
# energy $W'=\tfrac12C'V^2$; the inductance from $L'C'=\mu_0\varepsilon_0$; and the
# characteristic impedance is $Z_0=\sqrt{L'/C'}$.
#
# | quantity | value |
# |---|---|
# | wire radius $a$ | 0.5 mm |
# | centre spacing $D$ | 5 mm |
# | surroundings | air, truncated at radii of 25, 50 and 125 mm, grounded |
#
# The air comes from the CAD files `parallel_wire_line_impedance.step` (125 mm), `..._r50.step` and
# `..._r25.step` with the faces `left.outer_wall`, `right.outer_wall` and `outer_wall`, written by
# `parallel_wire_line_impedance_geometry.py` with the CAD objects of OneSimulation. The model
# lives on its cross-section. With antisymmetric wire voltages the potential far away vanishes, so
# grounding the far boundary changes the result only by the order of $(D/R)^2$.
#
# **Reference** (D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley 2012, Table 2.1;
# S. Ramo, J. R. Whinnery, T. Van Duzer, *Fields and Waves in Communication Electronics*, 3rd ed.,
# Wiley 1994, §1.16). The field outside the wires is that of two line charges at $x=\pm s$ with
# $s=\sqrt{D^2/4-a^2}$; the wire surfaces are equipotential circles of this pair, and
#
# $$C'=\frac{\pi\varepsilon_0}{\operatorname{arccosh}(D/2a)},\qquad
#   Z_0=\frac{\eta_0}{\pi}\operatorname{arccosh}\frac{D}{2a},\qquad
#   \phi(x,0)=\frac{V}{2\operatorname{arccosh}(D/2a)}\ln\left|\frac{x+s}{x-s}\right| .$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability, VacuumPermittivity
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation
from onesim.results.comparisons import convergence
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the air pieces; `section` cuts each at half length. Quadratic triangles grow from
# the wires to the far boundary. The figure shows the smallest piece, in which the wires are visible.

parts = {
    radius: load(f"parallel_wire_line_impedance{suffix}.step")
    for radius, suffix in ((0.125, ""), (0.050, "_r50"), (0.025, "_r25"))
}
sections = {
    radius: part.section((0, 0, 1), origin=(0, 0, part.size[2] / 2))
    for radius, part in parts.items()
}
surface = {face.id: face for face in parts[0.125].faces}
a, spacing = (
    surface["right.outer_wall"].radius,
    surface["right.outer_wall"].center[0] - surface["left.outer_wall"].center[0],
)
meshes = {
    radius: TriangularMesher(
        target_size=radius / 12,
        grading=0.2,
        local_sizes={"left.outer_wall": 0.08e-3, "right.outer_wall": 0.08e-3},
        size_field=lambda points, radius=radius: np.clip(
            0.15 * np.hypot(points[:, 0], points[:, 1]), 0.2e-3, radius / 12
        ),
    ).generate(section)
    for radius, section in sections.items()
}

fig = plots.figure(1, 2, size=(13, 5.5))
plots.geometry(
    parts[0.025],
    selection={
        "wires": parts[0.025].boundaries["left.outer_wall"]
        | parts[0.025].boundaries["right.outer_wall"]
    },
    labels=False,
    camera=plots.Camera(elevation=35, azimuth=-60),
    title="Air around the line, 25 mm radius",
    ax=fig[0],
)
plots.mesh(
    meshes[0.025],
    groups=("all",),
    legend=False,
    title=f"{meshes[0.025].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# One electrostatic solve per air cylinder with $+V/2$ and $-V/2$ on the wires and the far
# boundary grounded; the reaction of the potential on the right wire is its charge per metre.

eps0, mu0, voltage = (
    float(VacuumPermittivity().evaluate()),
    float(VacuumPermeability().evaluate()),
    1.0,
)
models = {
    radius: PoissonEquation(
        domain=section.domain(), permittivity=VacuumPermittivity(), charge_density=0.0
    )
    for radius, section in sections.items()
}
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
results = {
    radius: Simulation(
        problem=Problem(
            name="Parallel-wire line",
            geometry=section,
            physics=(models[radius],),
            boundary_conditions=(
                FixedPotential(
                    boundary=section.boundaries["right.outer_wall"],
                    physics=models[radius],
                    value=voltage / 2,
                ),
                FixedPotential(
                    boundary=section.boundaries["left.outer_wall"],
                    physics=models[radius],
                    value=-voltage / 2,
                ),
                FixedPotential(
                    boundary=section.boundaries.outer_wall,
                    physics=models[radius],
                    value=0.0,
                ),
            ),
        ),
        mesh=meshes[radius],
        numerics=numerics,
    ).solve_stationary()
    for radius, section in sections.items()
}

# ## Line constants and impedance
#
# The grounded far boundary pulls a little extra charge onto the wires; the effect falls off as
# $(D/R)^2$, so the three air cylinders extrapolate to the unbounded line.

closed_capacitance = np.pi * eps0 / np.arccosh(spacing / (2 * a))
closed_impedance = np.sqrt(mu0 / eps0) / np.pi * np.arccosh(spacing / (2 * a))
energy = lambda E, position: 0.5 * eps0 * np.sum(E**2, axis=1)
charge = {
    radius: result.field(models[radius].potential).reaction(
        sections[radius].boundaries["right.outer_wall"]
    )
    / voltage
    for radius, result in results.items()
}
print(
    "air radius   triangles   C' from Q [pF/m]   C' from W [pF/m]   Z0 [ohm]   deviation of Z0"
)
for radius, result in results.items():
    from_energy = (
        2
        * result.field(models[radius].electric_field).integrate(component=energy).value
        / voltage**2
    )
    impedance = np.sqrt(mu0 * eps0) / charge[radius]
    print(
        f"{1e3 * radius:7.0f} mm {meshes[radius].cell_count:11d} {1e12 * charge[radius]:18.4f} {1e12 * from_energy:18.4f} {impedance:10.4f} "
        f"{100 * (impedance / closed_impedance - 1):+13.4f} %"
    )
study = convergence(list(charge.values()), [spacing / radius for radius in charge])
extrapolated = np.sqrt(mu0 * eps0) / study.extrapolated
print(
    f"unbounded line (Richardson extrapolation in D/R): C' = {1e12 * study.extrapolated:.4f} pF/m, Z0 = {extrapolated:.4f} ohm "
    f"({100 * (extrapolated / closed_impedance - 1):+.4f} %); observed order {study.order:.2f}"
)
print(
    f"closed form: C' = {1e12 * closed_capacitance:.4f} pF/m, L' = {1e9 * mu0 * eps0 / closed_capacitance:.2f} nH/m, Z0 = {closed_impedance:.4f} ohm"
)

s = np.sqrt(spacing**2 / 4 - a**2)
line_charge = (
    lambda x: voltage
    / (2 * np.arccosh(spacing / (2 * a)))
    * np.log(np.abs((x + s) / (x - s)))
)
x = np.concatenate(
    (
        np.linspace(-20e-3, -spacing / 2 - a * 1.02, 30),
        np.linspace(-spacing / 2 + a * 1.02, spacing / 2 - a * 1.02, 15),
        np.linspace(spacing / 2 + a * 1.02, 20e-3, 30),
    )
)
potential = (
    results[0.125]
    .field(models[0.125].potential)
    .evaluate(np.column_stack((x, np.zeros_like(x))))
    .ravel()
)
print(
    f"potential on the x axis against the line-charge solution: largest deviation {100 * np.abs(potential - line_charge(x)).max() / voltage:.3f} % of V"
)

# ## Figure

fine = np.concatenate(
    (
        np.linspace(-20e-3, -spacing / 2 - a, 200),
        np.linspace(-spacing / 2 + a, spacing / 2 - a, 100),
        np.linspace(spacing / 2 + a, 20e-3, 200),
    )
)
fig = plots.figure(1, 2, size=(14, 5.2))
plots.surface(
    results[0.125].field(models[0.125].potential),
    limits=(-0.5, 0.5),
    contours=21,
    subdivisions=2,
    label="potential [V]",
    title="Equipotentials around the wires",
    xlimits=(-10e-3, 10e-3),
    ylimits=(-7e-3, 7e-3),
    xlabel="x [m]",
    ylabel="y [m]",
    ax=fig[0],
)
plots.line(
    points={"simulation": (1e3 * x, potential)},
    reference={"line-charge pair": (1e3 * fine, line_charge(fine))},
    vlines=[
        1e3 * (-spacing / 2 - a),
        1e3 * (-spacing / 2 + a),
        1e3 * (spacing / 2 - a),
        1e3 * (spacing / 2 + a),
    ],
    xlabel="x on the line through both wires [mm]",
    ylabel="potential [V]",
    title="Potential through the wires",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Move the wires apart in `parallel_wire_line_impedance_geometry.py` and write the STEP file
# again: the impedance grows with the logarithm of the spacing. Put the line above a grounded
# plane (a cut through the air cylinder) and compare the impedance with the method of images.
