"""OneSimulation example: steady reaction-diffusion in a catalyst slab.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the exact closed-form steady
profile and the analytic effectiveness factor of a first-order catalyst slab.
"""


# # Reaction-diffusion in a catalyst slab (Thiele modulus)
#
# A reactant diffuses into a flat porous catalyst slab and is consumed there by a
# first-order surface reaction. The slab has half-thickness $L$: its open face at
# $x=0$ is held at the external concentration $C_s$, and its centre plane (or
# sealed back) at $x=L$ carries no flux by symmetry. With diffusivity $D$, rate
# coefficient $k$ and no convection the steady balance is
#
# $$D\,\frac{d^2C}{dx^2}=k\,C,\qquad C(0)=C_s,\quad
#   \left.\frac{dC}{dx}\right|_{x=L}=0.$$
#
# OneSimulation integrates the transient species balance in time on a real 1D
# line mesh with continuous finite elements (Lagrange order 2, surface value
# enforced weakly with Nitsche) until only the steady profile remains.
#
# **Reference.** Introducing the dimensionless **Thiele modulus**
# $\phi=L\sqrt{k/D}$, the linear constant-coefficient ODE with a sealed back has
# the exact solution
#
# $$\frac{C(x)}{C_s}=\frac{\cosh\!\big(\phi\,(1-x/L)\big)}{\cosh\phi}.$$
#
# The **effectiveness factor** — the actual conversion divided by the conversion
# the slab would reach at the surface concentration everywhere — follows as
#
# $$\eta=\frac{1}{L\,C_s}\int_0^L C\,dx=\frac{\tanh\phi}{\phi}.$$
#
# We compare the computed steady profile and effectiveness factor with these
# closed forms.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})

from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, Nitsche
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.chemistry import SpeciesTransport, TransportSpecies
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation.problem_definition.boundaries import (
    OutwardFluxBoundary, PrescribedValueBoundary,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The values are freely adjustable and in SI units. They give a moderate Thiele
# modulus $\phi=2$, where diffusion limits the reaction appreciably but the
# interior is not fully starved.

L = 1.0               # m, slab half-thickness (open face to sealed centre)
cells = 200           # mesh cells
D = 0.1               # m^2/s, effective pore diffusivity
k = 0.4               # 1/s, first-order rate coefficient
C_s = 1.0             # mol/m^3, external (surface) concentration

phi = L * np.sqrt(k / D)              # Thiele modulus
eta_ref = np.tanh(phi) / phi         # analytic effectiveness factor
print(f"Thiele modulus phi = {phi:.3f},  analytic effectiveness eta = {eta_ref:.4f}")


def reference(x):
    return C_s * np.cosh(phi * (1.0 - x / L)) / np.cosh(phi)


line = Line(start=(0.0, 0.0, 0.0), end=(L, 0.0, 0.0))
mesh = StructuredLineMesher(cells=cells).generate(line)
print("Mesh:", mesh.cell_count, "line cells")

species = (TransportSpecies("reactant", diffusivity=D, decay_rate=k),)
model = SpeciesTransport(domain=line.line, velocity=0.0, species=species)

# Both ends carry an explicit boundary condition: the surface concentration is
# prescribed at the open face, and the sealed centre plane carries zero flux.
boundary_conditions = (
    PrescribedValueBoundary(
        physics=model,
        boundary=line.boundaries.start,
        value=lambda time, coordinate: {"reactant": C_s},
    ),
    OutwardFluxBoundary(
        physics=model,
        boundary=line.boundaries.end,
        flux=lambda **kw: np.zeros(1),
        jacobian=lambda **kw: np.zeros((1, 1)),
    ),
)

problem = Problem(
    name="steady reaction-diffusion in a catalyst slab",
    geometry=line,
    physics=(model,),
    boundary_conditions=boundary_conditions,
    initial_conditions=(InitialValue(model.concentration("reactant"), 0.0),),
)

# The slowest transient mode decays at a rate of order D (pi/2L)^2 + k, so a few
# tens of those times are enough to settle onto the steady profile.
settling_rate = D * (np.pi / (2.0 * L)) ** 2 + k
end_time = 40.0 / settling_rate
time_step = end_time / 400.0
print(f"settling run: end time = {end_time:.1f} s,  time step = {time_step:.3f} s")

result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(2), quadrature_order=5,
                              boundary_enforcement=Nitsche()),
        linear_solver=SparseDirect(),
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(), jacobian="analytic",
            absolute_tolerance=1e-12, relative_tolerance=1e-11,
        ),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
    ),
).run(start_time=0.0, end_time=end_time, time_step=time_step)


# ## Comparison with the closed-form steady profile
#
# The last stored time is deep in the steady regime. We compare the computed
# concentration with the exact hyperbolic-cosine profile.

field = result.field(model.concentration("reactant"))
points = np.asarray(field.space.dof_coordinates)
x = points[:, 0]
order = np.argsort(x)
x = x[order]
c_num = np.asarray(field.evaluate(points, time=float(result.times[-1]))).reshape(-1)[order]

c_ref = reference(x)
max_abs_error = float(np.max(np.abs(c_num - c_ref)))
rel_error = max_abs_error / C_s
print(f"maximum deviation from the closed form: {max_abs_error:.3e} mol/m^3  ({100*rel_error:.3f} % of C_s)")

# Confirm steady state has been reached: compare with the profile one step back.
previous = np.asarray(field.evaluate(points, time=float(result.times[-2]))).reshape(-1)[order]
drift = float(np.max(np.abs(c_num - previous)))
print(f"change over the last stored interval: {drift:.3e} mol/m^3")

# ## Effectiveness factor
#
# The effectiveness factor is the slab-averaged concentration divided by C_s. We
# integrate the computed profile with the trapezoidal rule and compare with the
# analytic tanh(phi)/phi.
eta_num = float(np.trapezoid(c_num, x) / (L * C_s))
eta_err = abs(eta_num - eta_ref)
print(f"effectiveness factor: numeric {eta_num:.4f}, analytic {eta_ref:.4f}, error {eta_err:.3e}")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
xs = np.linspace(0.0, L, 400)
ax.plot(xs, reference(xs), color="#214D43", lw=1.8, label="closed form")
ax.plot(x[::5], c_num[::5], "o", color="#336B57", ms=3, label="OneSimulation")
ax.set(xlabel="Position x / m", ylabel="Concentration / (mol/m^3)",
       title=f"Catalyst slab (Thiele phi = {phi:.1f}, eta = {eta_ref:.3f})"
             "\nline: closed form, dots: OneSimulation")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Raise the rate coefficient $k$ (larger Thiele modulus) and the reactant is
# consumed in a thin skin near the open face, so the effectiveness factor falls
# toward the strong-diffusion-limit $\eta\approx1/\phi$. Lower $k$ (small $\phi$)
# and the slab fills almost uniformly with $\eta\to1$, the reaction-limited
# regime.
