"""OneSimulation example: characteristic impedance of a coaxial cable from its cross-section.
Requires the installed, activated OneSimulation runtime.

The dielectric of a concentric and of an eccentric coaxial cable is read from STEP files and cut to
its cross-section. The electrostatic field gives the capacitance per metre from the conductor
charge and from the field energy; with the inductance of the TEM wave this yields the
characteristic impedance, which is checked against the closed forms for both cables.
"""

# # Characteristic impedance of a coaxial cable
#
# A coaxial cable guides a TEM wave between an inner conductor of radius $a$ and a shield of
# inner radius $b$, separated by a dielectric of relative permittivity $\varepsilon_r$. For a TEM
# wave the transverse fields are static fields of the cross-section: the electric field solves
# Laplace's equation for the potential difference $V$ between the conductors,
#
# $$\nabla\cdot(\varepsilon\nabla\phi)=0,\qquad \phi=V\ \text{on the inner conductor},\quad
#   \phi=0\ \text{on the shield},$$
#
# and the magnetic field is the electric field turned by 90° and divided by the wave impedance
# $\eta=\sqrt{\mu/\varepsilon}$. The line constants per metre therefore follow from one 2D field
# solve: the capacitance $C'=Q'/V$ from the charge on the inner conductor (or from the field
# energy, $W'=\tfrac12C'V^2$), and the inductance from $L'C'=\mu\varepsilon$. The characteristic
# impedance is
#
# $$Z_0=\sqrt{\frac{L'}{C'}}=\frac{\sqrt{\mu\varepsilon}}{C'} .$$
#
# The cable is a common 50 Ω type with a polyethylene dielectric:
#
# | quantity | value |
# |---|---|
# | inner conductor radius $a$ | 0.45 mm |
# | shield inner radius $b$ | 1.475 mm |
# | dielectric | polyethylene, $\varepsilon_r=2.25$ |
# | offset of the inner conductor (second cable) | $d=0.40$ mm |
#
# The dielectric comes from the CAD files `coaxial_cable_impedance.step` (faces `inner_wall`,
# `outer_wall`) and `coaxial_cable_impedance_eccentric.step` (faces `conductor.outer_wall`,
# `outer_wall`), written by `coaxial_cable_impedance_geometry.py` with the CAD objects of
# OneSimulation. The model lives on their cross-sections.
#
# **Reference** (D. M. Pozar, *Microwave Engineering*, 4th ed., Wiley 2012, §2.2 and §3.5;
# S. Ramo, J. R. Whinnery, T. Van Duzer, *Fields and Waves in Communication Electronics*, 3rd ed.,
# Wiley 1994, §5.11). For the concentric and for the eccentric cable
#
# $$C'=\frac{2\pi\varepsilon}{\ln(b/a)},\qquad
#   C'_{\rm ecc}=\frac{2\pi\varepsilon}{\operatorname{arccosh}\dfrac{a^2+b^2-d^2}{2ab}},\qquad
#   Z_0=\frac{\eta}{2\pi}\ln\frac ba .$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability, VacuumPermittivity
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the dielectric; `section` cuts it at half length. Quadratic triangles are three
# times finer on the inner conductor than on the shield.

parts = {
    "concentric": load("coaxial_cable_impedance.step"),
    "eccentric": load("coaxial_cable_impedance_eccentric.step"),
}
conductor = {"concentric": "inner_wall", "eccentric": "conductor.outer_wall"}
sections = {
    name: part.section((0, 0, 1), origin=(0, 0, part.size[2] / 2))
    for name, part in parts.items()
}
a, b, offset = 0.45e-3, parts["concentric"].size[0] / 2, 0.40e-3
meshes = {
    name: TriangularMesher(
        target_size=0.12e-3, local_sizes={conductor[name]: 0.04e-3}
    ).generate(section)
    for name, section in sections.items()
}

fig = plots.figure(2, 2, size=(12, 11))
for row, name in enumerate(parts):
    plots.geometry(
        parts[name],
        selection={
            "inner conductor": parts[name].boundaries[conductor[name]],
            "shield": parts[name].boundaries.outer_wall,
        },
        labels=False,
        camera=plots.Camera(elevation=35, azimuth=-60),
        title=f"Dielectric of the {name} cable",
        ax=fig[2 * row],
    )
    plots.mesh(
        meshes[name],
        groups=("all",),
        legend=False,
        title=f"{meshes[name].cell_count} triangles",
        ax=fig[2 * row + 1],
    )
plots.show()

# ## Model
#
# One electrostatic solve per cable with 1 V on the inner conductor and the shield grounded. The
# reaction of the potential on the inner conductor is its charge per metre.

eps, mu, voltage = (
    2.25 * float(VacuumPermittivity().evaluate()),
    float(VacuumPermeability().evaluate()),
    1.0,
)
models = {
    name: PoissonEquation(
        domain=section.domain(),
        permittivity=regionwise((), default=eps, unit="F/m"),
        charge_density=0.0,
    )
    for name, section in sections.items()
}
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
results = {
    name: Simulation(
        problem=Problem(
            name=f"{name} coaxial cable",
            geometry=section,
            physics=(models[name],),
            boundary_conditions=(
                FixedPotential(
                    boundary=section.boundaries[conductor[name]],
                    physics=models[name],
                    value=voltage,
                ),
                FixedPotential(
                    boundary=section.boundaries.outer_wall,
                    physics=models[name],
                    value=0.0,
                ),
            ),
        ),
        mesh=meshes[name],
        numerics=numerics,
    ).solve_stationary()
    for name, section in sections.items()
}

# ## Line constants and impedance
#
# Capacitance from the charge and from the field energy, inductance from $L'C'=\mu\varepsilon$,
# and the impedance, each against the closed form.

closed_form = {
    "concentric": 2 * np.pi * eps / np.log(b / a),
    "eccentric": 2 * np.pi * eps / np.arccosh((a**2 + b**2 - offset**2) / (2 * a * b)),
}
energy = lambda E, position: 0.5 * eps * np.sum(E**2, axis=1)
print(
    "cable        C' from Q [pF/m]   C' from W [pF/m]   closed form [pF/m]   L' [nH/m]    Z0 [ohm]   closed form [ohm]   deviation"
)
for name, result in results.items():
    from_charge = (
        result.field(models[name].potential).reaction(
            sections[name].boundaries[conductor[name]]
        )
        / voltage
    )
    from_energy = (
        2
        * result.field(models[name].electric_field).integrate(component=energy).value
        / voltage**2
    )
    inductance, impedance = mu * eps / from_charge, np.sqrt(mu * eps) / from_charge
    exact = np.sqrt(mu * eps) / closed_form[name]
    print(
        f"{name:11s} {1e12 * from_charge:17.4f} {1e12 * from_energy:18.4f} {1e12 * closed_form[name]:20.4f} {1e9 * inductance:11.3f} "
        f"{impedance:11.4f} {exact:19.4f} {100 * (impedance / exact - 1):+10.4f} %"
    )

# ## Figure
#
# The field strength in both cross-sections on a common scale, and the radial field of the
# concentric cable against $E_r=V/(r\ln(b/a))$.

strength = lambda E, position: 1e-3 * np.linalg.norm(E[:, :2], axis=1)
r = np.linspace(a * 1.01, b * 0.99, 30)
fine = np.linspace(a, b, 200)
radial = strength(
    results["concentric"]
    .field(models["concentric"].electric_field)
    .evaluate(np.column_stack((r, np.zeros_like(r)))),
    None,
)
fig = plots.figure(1, 3, size=(19, 5))
for index, name in enumerate(results):
    plots.surface(
        results[name].field(models[name].electric_field),
        component=strength,
        limits=(0, 3),
        contours=13,
        subdivisions=2,
        label="|E| [kV/m] at 1 V",
        title=f"Field strength, {name} cable",
        xlabel="x [m]",
        ylabel="y [m]",
        ax=fig[index],
    )
plots.line(
    points={"simulation": (1e3 * r, radial)},
    reference={
        "V / (r ln(b/a))": (1e3 * fine, 1e-3 * voltage / (fine * np.log(b / a)))
    },
    xlabel="radius r [mm]",
    ylabel="|E| [kV/m] at 1 V",
    title="Radial field of the concentric cable",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the radii or the offset in `coaxial_cable_impedance_geometry.py` and write the STEP files
# again: a 75 Ω cable needs $b/a\approx6.5$ with polyethylene. Give part of the dielectric a
# different permittivity with `regionwise` (a foam-filled cable) and the simple relation
# $L'C'=\mu\varepsilon$ no longer holds; the impedance then needs the inductance of the air-filled
# line, $Z_0=1/(c\sqrt{C'C'_{\rm air}})$.
