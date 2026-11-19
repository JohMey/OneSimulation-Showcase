"""OneSimulation example: induction currents from a circular coil in a copper ring (2D-axisymmetric).
Requires the installed, activated OneSimulation runtime.

An alternating or switched coil current induces currents in a nearby short-circuited copper ring.
The meridian section is cut from STEP parts and solved with the eddy-current model, time-harmonic
over four decades of frequency and transient after switching on. The ring current is checked
against the circuit model of two coupled loops, which holds as long as the skin depth exceeds the
ring section.
"""

# # Induction currents from a circular coil in a copper ring
#
# A circular coil of 100 turns (winding section 10 mm × 10 mm, mean radius 45 mm) sits 20 mm above a
# coaxial copper ring of square section $s=6$ mm (mean radius $R=43$ mm, $\sigma=5.8\cdot10^7$ S/m)
# whose ends are short-circuited. A changing coil current induces a voltage in the ring, and the
# ring current flows so as to oppose the change of flux (Lenz's rule).
#
# OneSimulation solves the eddy-current equation for the azimuthal vector potential on the
# meridian half-plane with quadratic elements (`EddyCurrents` with the revolved space):
#
# $$\nabla\times\big(\mu_0^{-1}\nabla\times\mathbf A\big)+\sigma\,\partial_t\mathbf A=\mathbf J_s .$$
#
# The coil is a stranded winding with a prescribed current; the ring is a massive conductor with
# zero terminal voltage, so its current and its distribution over the section follow from the field
# (`Coil(..., voltage=0.0, conductor="massive")`). Two analyses are made: time-harmonic at 10 Hz to
# 10 kHz, and transient after the coil current is switched on to 1 A.
#
# The domain is the meridian section of `induction_currents_coil_and_ring.step`, a cylinder of air of
# 0.3 m radius; coil and ring are the sections of `_coil.step` and `_ring.step`, which the mesh
# follows. The script `induction_currents_coil_and_ring_geometry.py` builds them with the CAD
# objects of OneSimulation. The far boundary carries $A_\varphi=0$.
#
# **Reference.** Coil and ring as two coupled loops (Grover, *Inductance Calculations*, 1946): the
# mutual inductance of coaxial circles of radii $r_1$, $r_2$ at the distance $d$ is
# $M=\mu_0\sqrt{r_1r_2}\,[(2/m-m)K(m)-(2/m)E(m)]$ with $m^2=4r_1r_2/((r_1+r_2)^2+d^2)$; the ring has
# $L_2=\mu_0R[\ln(8R/g)-2]$ with the geometric mean distance $g=0.447s$ of its square section, and
# $R_2=2\pi R/(\sigma s^2)$. Then
#
# $$I_2=-\frac{\mathrm j\omega MNI_1}{R_2+\mathrm j\omega L_2}\quad\text{(harmonic)},\qquad
#   i_2(t)=-\frac{MNI_1}{L_2}\,e^{-t/\tau},\ \tau=\frac{L_2}{R_2}\quad\text{(switch-on)}.$$
#
# The circuit model assumes a uniform current over the ring section; once the skin depth falls
# below the section size the field computation departs from it.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability, regionwise
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, RevolvedH1SpaceFactory
from onesim.numerics.linear import ComplexSparseDirect, SparseDirect
from onesim.numerics.time import BDF
from onesim.physics.magnetostatics import (
    Coil,
    EddyCurrents,
    MagneticInsulation,
    coil_current,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.special import ellipe, ellipk

theme.set()

# ## Geometry and mesh
#
# `section` cuts the parts in the plane $y=0$; the section coordinates are $(r, z)$. The mesh follows
# coil and ring, is fine in the ring and grows towards the far boundary.

meridian = lambda name: load(f"induction_currents_coil_and_ring{name}.step").section(
    (0, 1, 0)
)
domain, winding, ring = meridian(""), meridian("_coil"), meridian("_ring")
axis = domain.select(
    edges=lambda edge: all(abs(point[0]) < 1e-12 for point in edge.points)
)
mesh = TriangularMesher(
    target_size=0.03,
    grading=0.25,
    embedded_regions=(winding.named("coil"), ring.named("ring")),
    region_sizes=(("coil", 0.002), ("ring", 0.0003)),
).generate(domain)

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [
        load(f"induction_currents_coil_and_ring{name}.step").named(label)
        for name, label in (("_coil", "coil"), ("_ring", "copper ring"))
    ],
    camera=plots.Camera(elevation=25, azimuth=-60),
    title="Coil and ring (x > 0)",
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

# ## Time-harmonic analysis
#
# Coil current 1 A (peak) with 100 turns; one solve per frequency.

sigma, turns, current = 5.8e7, 100, 1.0
mu0 = float(VacuumPermeability().evaluate())
conductivity = regionwise({"ring": sigma}, default=0.0, unit="S/m")
revolved = FiniteElement(
    element=Lagrange(order=2), quadrature_order=4, provider=RevolvedH1SpaceFactory()
)
frequencies = (10.0, 100.0, 1000.0, 10000.0)
rings = {f: Coil(ring, voltage=0.0, conductor="massive") for f in frequencies}
harmonic = {
    f: EddyCurrents(
        domain=domain.domain(),
        permeability=VacuumPermeability(),
        conductivity=conductivity,
        frequency=f,
        coils=(Coil(winding, turns=turns, current=current), rings[f]),
    )
    for f in frequencies
}
ring_current = {
    f: coil_current(
        Simulation(
            problem=Problem(
                name="Coil and ring",
                geometry=domain,
                physics=(model,),
                boundary_conditions=(
                    MagneticInsulation(
                        boundary=domain.boundaries.all - axis, physics=model
                    ),
                ),
            ),
            mesh=mesh,
            numerics=Numerics(spatial=revolved, linear_solver=ComplexSparseDirect()),
        ).solve_stationary(),
        model,
        rings[f],
    )
    for f, model in harmonic.items()
}

# ## Transient analysis
#
# The coil current is switched on to 1 A at $t=0$; second-order BDF steps of $\tau/20$ up to $4\tau$.

r1, r2, s, d = 0.045, 0.043, 0.006, 0.020
m2 = 4 * r1 * r2 / ((r1 + r2) ** 2 + d**2)
M = (
    mu0
    * np.sqrt(r1 * r2)
    * ((2 / np.sqrt(m2) - np.sqrt(m2)) * ellipk(m2) - 2 / np.sqrt(m2) * ellipe(m2))
)
L2 = mu0 * r2 * (np.log(8 * r2 / (0.447 * s)) - 2)
R2 = 2 * np.pi * r2 / (sigma * s**2)
tau = L2 / R2
switched_ring = Coil(ring, voltage=0.0, conductor="massive")
transient = EddyCurrents(
    domain=domain.domain(),
    permeability=VacuumPermeability(),
    conductivity=conductivity,
    coils=(Coil(winding, turns=turns, current=current), switched_ring),
)
history = Simulation(
    problem=Problem(
        name="Coil switch-on",
        geometry=domain,
        physics=(transient,),
        boundary_conditions=(
            MagneticInsulation(
                boundary=domain.boundaries.all - axis, physics=transient
            ),
        ),
    ),
    mesh=mesh,
    numerics=Numerics(
        spatial=revolved, linear_solver=SparseDirect(), time_integrator=BDF(order=2)
    ),
).run(start_time=0.0, end_time=4 * tau, time_step=tau / 20)
times = np.array(history.times)
induced = np.array(
    [coil_current(history, transient, switched_ring, time=t) for t in times]
)

# ## Results

circuit = (
    lambda f: -1j * 2 * np.pi * f * M * turns * current / (R2 + 1j * 2 * np.pi * f * L2)
)
print(
    f"circuit model: M = {1e9 * M:.2f} nH, L2 = {1e9 * L2:.2f} nH, R2 = {1e6 * R2:.2f} µOhm, tau = {1e3 * tau:.3f} ms"
)
print(
    "f [Hz]   skin depth [mm]   |I2| simulation [A]   circuit [A]   deviation   phase simulation / circuit [deg]"
)
for f in frequencies:
    print(
        f"{f:7.0f} {1e3 * np.sqrt(2 / (2 * np.pi * f * mu0 * sigma)):15.2f} {abs(ring_current[f]):20.4f} {abs(circuit(f)):13.4f} "
        f"{100 * (abs(ring_current[f]) / abs(circuit(f)) - 1):+10.2f} % {np.degrees(np.angle(ring_current[f])):12.1f} / {np.degrees(np.angle(circuit(f))):.1f}"
    )
late = times > 0.5 * tau
fitted = -np.polyfit(times[late], np.log(np.abs(induced[late])), 1)[0] ** -1
print(
    f"transient: ring current after the first step {induced[1]:.4f} A (circuit {-M * turns * current / L2:.4f} A); "
    f"decay time {1e3 * fitted:.3f} ms, circuit {1e3 * tau:.3f} ms ({100 * (fitted / tau - 1):+.1f} %)"
)

# ## Figure
#
# The induced ring current over frequency against the circuit model, and its decay after the
# switch-on.

fine = np.logspace(0.5, 4.5, 200)
fig = plots.figure(1, 2, size=(15, 6))
plots.line(
    points={
        "simulation": (
            np.array(frequencies),
            np.array([abs(ring_current[f]) for f in frequencies]),
        )
    },
    reference={"coupled loops": (fine, np.abs(circuit(fine)))},
    scale="loglog",
    xlabel="frequency [Hz]",
    ylabel="|ring current| [A]",
    title="Induced ring current (coil 1 A, 100 turns)",
    ax=fig[0],
)
plots.line(
    points={"simulation": (1e3 * times[1:], -induced[1:])},
    reference={
        "coupled loops": (1e3 * times, M * turns * current / L2 * np.exp(-times / tau))
    },
    xlabel="time [ms]",
    ylabel="-ring current [A]",
    title="After switching on the coil",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Feed the coil by a voltage instead of a current to see the reaction of the ring on the coil
# current, replace the ring by a solid disc, or make it of steel to add the magnetic pull of the
# ring towards the coil.
