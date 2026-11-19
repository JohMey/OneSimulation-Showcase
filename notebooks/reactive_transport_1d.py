"""OneSimulation example: steady 1D reactive transport (advection-diffusion-reaction).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the exact closed-form steady
solution of the advection-diffusion-reaction equation.
"""

# # Reactive transport in a 1D reactor
#
# A dissolved species flows through a thin tubular reactor at a constant speed
# $u$, spreads by diffusion with coefficient $D$, and is consumed by a
# first-order reaction of rate $k$. At the inlet ($x=0$) its concentration is
# held at $C_0$; at the outlet ($x=L$) it is held at $C_L$.
#
# The steady advection-diffusion-reaction balance is
#
# $$u\,\frac{dC}{dx}=D\,\frac{d^2C}{dx^2}-k\,C .$$
#
# OneSimulation integrates the transient transport in time on a real 1D line mesh
# with continuous finite elements until only the steady profile remains.
#
# **Reference.** The steady equation is a linear, constant-coefficient ODE. With
# $C=e^{m x}$ the characteristic equation $D m^2-u m-k=0$ gives the two roots
#
# $$m_\pm=\frac{u\pm\sqrt{u^2+4Dk}}{2D},$$
#
# so the exact solution is $C(x)=A\,e^{m_+x}+B\,e^{m_-x}$, with $A$ and $B$ fixed
# by $C(0)=C_0$ and $C(L)=C_L$. The dimensionless **Damkoehler number**
# $\mathrm{Da}=kL/u$ measures reaction against advection, and the **Peclet
# number** $\mathrm{Pe}=uL/D$ advection against diffusion. We compare the
# computed steady profile with this closed form.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, Nitsche
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.chemistry import SpeciesTransport, TransportSpecies
from onesim.simulation import PrescribedValueBoundary, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()


# ## Parameters
#
# The values are freely adjustable and in SI units. They give a moderately
# reaction- and advection-dominated reactor.

L = 1.0
cells = 200
u = 0.2
D = 0.01
k = 1.0
C0 = 1.0
CL = 0.0

Da = k * L / u
Pe = u * L / D
print(f"Damkoehler Da = {Da:.2f},  Peclet Pe = {Pe:.2f}")

disc = np.sqrt(u * u + 4.0 * D * k)
m_plus = (u + disc) / (2.0 * D)
m_minus = (u - disc) / (2.0 * D)
coeffs = np.linalg.solve(
    np.array([[1.0, 1.0], [np.exp(m_plus * L), np.exp(m_minus * L)]]),
    np.array([C0, CL]),
)


def reference(x):
    return coeffs[0] * np.exp(m_plus * x) + coeffs[1] * np.exp(m_minus * x)


line = Line(start=(0.0, 0.0, 0.0), end=(L, 0.0, 0.0))
mesh = StructuredLineMesher(cells=cells).generate(line)

species = (TransportSpecies("reactive", diffusivity=D, decay_rate=k),)
model = SpeciesTransport(domain=line.line, velocity=u, species=species)

# Both ends carry an explicit boundary condition: the concentration is prescribed
# at the inlet and at the outlet.
boundary_conditions = (
    PrescribedValueBoundary(
        physics=model,
        boundary=line.boundaries.start,
        value=lambda time, coordinate: {"reactive": C0},
    ),
    PrescribedValueBoundary(
        physics=model,
        boundary=line.boundaries.end,
        value=lambda time, coordinate: {"reactive": CL},
    ),
)

problem = Problem(
    name="steady reactive transport in a 1D reactor",
    geometry=line,
    physics=(model,),
    boundary_conditions=boundary_conditions,
    initial_conditions=(InitialValue(model.concentration("reactive"), 0.0),),
)

# The reaction fixes the slowest transient time scale to ~1/k, so a few reaction
# times are enough to reach steady state.
end_time = 15.0 / k
time_step = end_time / 300.0
print(f"settling run: end time = {end_time:.1f} s,  time step = {time_step:.3f} s")

result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(
            element=Lagrange(2), quadrature_order=5, boundary_enforcement=Nitsche()
        ),
        linear_solver=SparseDirect(),
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(),
            jacobian="analytic",
            absolute_tolerance=1e-12,
            relative_tolerance=1e-11,
        ),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
    ),
).run(start_time=0.0, end_time=end_time, time_step=time_step)


# ## Comparison with the closed-form steady profile
#
# The last stored time is deep in the steady regime. We compare the computed
# concentration with the exact two-exponential solution.

field = result.field(model.concentration("reactive"))
points = np.asarray(field.space.dof_coordinates)
x = points[:, 0]
order = np.argsort(x)
x = x[order]
c_num = np.asarray(field.evaluate(points, time=float(result.times[-1]))).reshape(-1)[
    order
]

c_ref = reference(x)
max_abs_error = float(np.max(np.abs(c_num - c_ref)))
rel_error = max_abs_error / C0
print(
    f"maximum deviation from the closed form: {max_abs_error:.3e} mol/m^3  ({100*rel_error:.3f} % of C0)"
)

previous = np.asarray(field.evaluate(points, time=float(result.times[-2]))).reshape(-1)[
    order
]
drift = float(np.max(np.abs(c_num - previous)))
print(f"change over the last stored interval: {drift:.3e} mol/m^3")


# ## Figure

xs = np.linspace(0.0, L, 400)
plots.line(
    points={"OneSimulation": (x[::5], c_num[::5])},
    reference={"closed form": (xs, reference(xs))},
    xlabel="position x [m]",
    ylabel="concentration [mol/m^3]",
    title=f"Steady reactive transport (Da = {Da:.1f}, Pe = {Pe:.0f})",
)
plots.show()


# ## Going further
#
# Raise the reaction rate $k$ (larger Damkoehler number) and the species is
# consumed closer to the inlet. Shrink the diffusivity $D$ (larger Peclet number)
# and the profile sharpens into a nearly pure advection-reaction exponential
# $C(x)\approx C_0\,e^{-kx/u}$ over the interior.
