"""OneSimulation example: eddy currents in a conducting cylinder inside a long coil (2D-axisymmetric).
Requires the installed, activated OneSimulation runtime.

An alternating coil current induces eddy currents in a metal cylinder; they push the field out of
the core (skin effect) and heat it. The meridian section is cut from STEP parts and solved with
the time-harmonic eddy-current model; field profile and loss are checked against the Bessel
solution over three frequencies, and a coil wound from one massive conductor shows the skin
effect in the coil itself.
"""

# # Eddy currents in a cylinder inside a long coil
#
# A long coil (inner radius 15 mm, outer radius 17 mm, 1000 turns per metre) carries an
# alternating current of 2 A amplitude. Inside sits a cylinder of radius $a=10$ mm with the
# conductivity $\sigma=3.5\cdot10^7$ S/m (aluminium). The time-harmonic field induces azimuthal
# eddy currents that oppose the applied field, so the field decays towards the axis within the
# skin depth $\delta=\sqrt{2/(\omega\mu_0\sigma)}$, and the currents heat the cylinder.
#
# OneSimulation solves the time-harmonic eddy-current equation for the azimuthal vector potential
#
# $$\nabla\times\big(\mu_0^{-1}\nabla\times\mathbf A\big)+\mathrm j\omega\sigma\mathbf A=\mathbf J_s$$
#
# on the meridian half-plane with quadratic elements (`EddyCurrents` with the revolved space). The
# coil is long, so a 10 mm slice with field-normal conditions on its top and bottom stands for it.
# The domain is the meridian section of `eddy_currents_cylinder_in_coil.step`; the cylinder and the
# winding are the sections of `_cylinder.step` and `_winding.step`, which the mesh follows. The
# script `eddy_currents_cylinder_in_coil_geometry.py` builds them with the CAD objects of
# OneSimulation.
#
# | edge | condition |
# |---|---|
# | axis $r=0$ | axis of the revolved space |
# | top, bottom and $r=20$ mm | field normal, $\mathbf n\times\mathbf H=0$ (`MagneticSymmetry`) |
#
# **Reference.** Inside an infinitely long cylinder in the uniform applied field $B_0=\mu_0nI$ (which
# falls linearly to zero across the winding) the field and the loss per metre are (Jackson, *Classical Electrodynamics*, 3rd ed., Sec. 5.18;
# Lammeraner and Štafl, *Eddy Currents*, 1966)
#
# $$B_z(r)=B_0\frac{J_0(kr)}{J_0(ka)},\qquad
#   \frac{P}{\ell}=\int_0^a\frac{|nI\,k\,J_1(kr)/J_0(ka)|^2}{2\sigma}\,2\pi r\,\mathrm dr,\qquad k=\sqrt{-\mathrm j\omega\mu_0\sigma}.$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability, regionwise
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, RevolvedH1SpaceFactory
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.magnetostatics import (
    Coil,
    EddyCurrents,
    MagneticSymmetry,
    coil_impedance,
    joule_losses,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.integrate import quad
from scipy.special import jv

theme.set()

# ## Geometry and mesh
#
# `section` cuts the parts in the plane $y=0$; the section coordinates are $(r, z)$. The mesh follows
# the surfaces of the cylinder and the winding and is fine enough for the smallest skin depth.

meridian = lambda name: load(f"eddy_currents_cylinder_in_coil{name}.step").section(
    (0, 1, 0)
)
domain, cylinder, winding = meridian(""), meridian("_cylinder"), meridian("_winding")
axis = domain.select(
    edges=lambda edge: all(abs(point[0]) < 1e-12 for point in edge.points)
)
height, radius = domain.size[1], cylinder.size[0]
inner, outer = winding.bounding_box[0][0], winding.bounding_box[1][0]
mesh = TriangularMesher(
    target_size=0.0004,
    embedded_regions=(cylinder.named("cylinder"), winding.named("winding")),
).generate(domain)

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [
        load(f"eddy_currents_cylinder_in_coil{name}.step").named(label)
        for name, label in (("_cylinder", "cylinder"), ("_winding", "winding"))
    ],
    camera=plots.Camera(elevation=30, azimuth=-60),
    title="Slice of the cylinder and the coil (x > 0)",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Meridian section: {mesh.cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The winding is a stranded coil of $n\,h=10$ turns carrying 2 A; three frequencies put the skin
# depth above, near and below the radius. A second set of runs feeds the same winding as one massive
# copper conductor ($5.8\cdot10^7$ S/m) carrying the same ampere-turns, 20 A.

sigma, per_metre, current = 3.5e7, 1000, 2.0
turns = int(round(per_metre * height))
mu0 = float(VacuumPermeability().evaluate())
frequencies = (50.0, 500.0, 5000.0)
numerics = Numerics(
    spatial=FiniteElement(
        element=Lagrange(order=2), quadrature_order=4, provider=RevolvedH1SpaceFactory()
    ),
    linear_solver=ComplexSparseDirect(),
)
cases = {
    (f, kind): (
        (
            Coil(winding, turns=turns, current=current)
            if kind == "stranded"
            else Coil(winding, current=turns * current, conductor="massive")
        ),
        regionwise(
            (
                {"cylinder": sigma}
                if kind == "stranded"
                else {"cylinder": sigma, "winding": 5.8e7}
            ),
            default=0.0,
            unit="S/m",
        ),
    )
    for f in frequencies
    for kind in ("stranded", "massive")
}
models = {
    key: EddyCurrents(
        domain=domain.domain(),
        permeability=VacuumPermeability(),
        conductivity=conductivity,
        frequency=key[0],
        coils=(coil,),
    )
    for key, (coil, conductivity) in cases.items()
}
results = {
    key: Simulation(
        problem=Problem(
            name="Cylinder in a coil",
            geometry=domain,
            physics=(model,),
            boundary_conditions=(
                MagneticSymmetry(boundary=domain.boundaries.all - axis, physics=model),
            ),
        ),
        mesh=mesh,
        numerics=numerics,
    ).solve_stationary()
    for key, model in models.items()
}

# ## Results
#
# Field on the mid-height line and loss of the cylinder against the Bessel solution; for the
# massive coil the loss in the winding and the power balance of its terminals.

r = np.linspace(0.0005, 0.0195, 39)
line = np.column_stack((r, np.full_like(r, height / 2)))
k_of = lambda f: np.sqrt(-1j * 2 * np.pi * f * mu0 * sigma)
bessel_field = (
    lambda f, r: mu0
    * per_metre
    * current
    * np.where(
        r < radius,
        jv(0, k_of(f) * np.minimum(r, radius)) / jv(0, k_of(f) * radius),
        np.clip((outer - r) / (outer - inner), 0.0, 1.0),
    )
)
print(
    "f [Hz]   skin depth [mm]   largest |B| deviation   loss [W/m]   Bessel [W/m]   deviation"
)
for f in frequencies:
    key = (f, "stranded")
    computed = np.abs(
        results[key].field(models[key].magnetic_flux_density).evaluate(line)[:, 2]
    )
    exact_loss = quad(
        lambda s: abs(
            per_metre * current * k_of(f) * jv(1, k_of(f) * s) / jv(0, k_of(f) * radius)
        )
        ** 2
        / (2 * sigma)
        * 2
        * np.pi
        * s,
        0,
        radius,
        limit=200,
    )[0]
    loss = joule_losses(results[key], models[key], cylinder) / height
    print(
        f"{f:6.0f} {1e3 * np.sqrt(2 / (2 * np.pi * f * mu0 * sigma)):17.2f} "
        f"{100 * np.max(np.abs(computed - np.abs(bessel_field(f, r)))) / (mu0 * per_metre * current):18.3f} % of B0 "
        f"{loss:12.5f} {exact_loss:14.5f} {100 * (loss / exact_loss - 1):+10.3f} %"
    )
print(
    "f [Hz]   massive coil: loss in the winding / in the cylinder [W/m]   1/2 Re(V I*) / total loss"
)
for f in frequencies:
    key = (f, "massive")
    coil = cases[key][0]
    total = joule_losses(results[key], models[key])
    terminal = (
        0.5
        * (
            coil_impedance(results[key], models[key], coil) * abs(turns * current) ** 2
        ).real
    )
    print(
        f"{f:6.0f} {joule_losses(results[key], models[key], winding) / height:24.4f} / {joule_losses(results[key], models[key], cylinder) / height:8.4f}"
        f" {terminal / total:30.6f}"
    )

# ## Figure
#
# The amplitude of the eddy-current density in the meridian plane at 5 kHz, and the field on the
# mid-height line for the three frequencies against the Bessel solution.

fine = np.linspace(0.0, 0.02, 400)
fig = plots.figure(1, 2, size=(15, 6))
plots.surface(
    results[5000.0, "stranded"].field(models[5000.0, "stranded"].current_density),
    component=lambda J, position: 1e-6 * np.sqrt(np.sum(np.abs(J) ** 2, axis=1)),
    limits=(0, 2.0),
    contours=11,
    subdivisions=2,
    label="|J| [A/mm²]",
    title="Eddy-current density at 5 kHz",
    xlabel="r [m]",
    ylabel="z [m]",
    ax=fig[0],
)
plots.line(
    points={
        f"{f:.0f} Hz": (
            1e3 * r[::2],
            1e3
            * np.abs(
                results[f, "stranded"]
                .field(models[f, "stranded"].magnetic_flux_density)
                .evaluate(line[::2])[:, 2]
            ),
        )
        for f in frequencies
    },
    reference={
        f"Bessel, {f:.0f} Hz": (1e3 * fine, 1e3 * np.abs(bessel_field(f, fine)))
        for f in frequencies
    },
    vlines=[1e3 * radius],
    xlabel="r [mm]",
    ylabel="|B_z| [mT]",
    title="Field amplitude across the cylinder",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Make the cylinder of steel (a relative permeability in `permeability`) and watch the field
# concentrate in a much thinner skin, or couple the loss density to heat conduction as in the
# example *Inductive heating of a copper cylinder*.
