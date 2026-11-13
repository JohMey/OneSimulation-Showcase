"""OneSimulation example: transient diffusion of a species out of a slab.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the known closed-form series solution.
"""


# # Transient diffusion of a species into a slab
#
# A slab of thickness $L$ starts at a uniform concentration $C_i$. From $t=0$ its
# open face at $x=0$ is held at a higher reservoir concentration $C_s$, while the
# back face at $x=L$ is sealed. The species then diffuses inward and the slab fills
# up. With a single diffusing species and no flow this is Fick's second law:
#
# $$\frac{\partial C}{\partial t}=D\,\frac{\partial^2 C}{\partial x^2},\qquad
#   C(x,0)=C_i,\quad C(0,t)=C_s,\quad \left.\frac{\partial C}{\partial x}\right|_{x=L}=0.$$
#
# OneSimulation solves it on a real 1D line with finite elements (Lagrange order 2,
# surface value enforced weakly with Nitsche) and a BDF2 time step.
#
# **Reference.** With $\theta=(C-C_s)/(C_i-C_s)$ the sealed back face selects the
# odd quarter-wave modes, giving the classical series
#
# $$\theta(x,t)=\sum_{n=1}^{\infty}\frac{4}{(2n-1)\pi}\,
#   \sin\!\Big(\frac{(2n-1)\pi x}{2L}\Big)\,
#   \exp\!\Big[-\Big(\frac{(2n-1)\pi}{2L}\Big)^2 D t\Big].$$
#
# At early times the front near the open face is the error-function profile
# $\theta\approx\operatorname{erf}\!\big(x/\sqrt{4Dt}\big)$. We check the computed
# concentration against the series.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

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
# The values are freely adjustable. $D$ and the characteristic time $L^2/D$ set
# how fast the slab empties.

L = 0.01              # m, slab thickness (x direction)
D = 1.0e-9            # m^2/s, diffusivity
C_i = 0.2             # mol/m^3, uniform initial concentration
C_s = 1.0             # mol/m^3, reservoir concentration at the open face from t=0 on
tau = L * L / D
end_time = 0.4 * tau
time_step = end_time / 60.0
print(f"D = {D:.1e} m^2/s,  characteristic time L^2/D = {tau:.0f} s,  end time = {end_time:.0f} s")

geometry = Line(start=(0.0, 0.0, 0.0), end=(L, 0.0, 0.0))
mesh = StructuredLineMesher(cells=160).generate(geometry)
print("Mesh:", mesh.cell_count, "line cells")

model = SpeciesTransport(
    domain=geometry.line,
    velocity=0.0,
    species=(TransportSpecies("A", diffusivity=D),),
)
problem = Problem(
    name="Species diffusion out of a slab",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        # open face at x=0 held at the surface concentration (weakly, Nitsche)
        PrescribedValueBoundary(
            physics=model, boundary=geometry.boundaries.start,
            value=lambda time, coordinate: {"A": C_s},
        ),
        # sealed back face at x=L: zero diffusive flux
        OutwardFluxBoundary(
            physics=model, boundary=geometry.boundaries.end,
            flux=lambda **kw: np.zeros(1), jacobian=lambda **kw: np.zeros((1, 1)),
        ),
    ),
    initial_conditions=(InitialValue(model.concentration("A"), C_i),),
)
simulation = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2), boundary_enforcement=Nitsche()),
        nonlinear_solver=Newton(linear_solver=SparseDirect(), jacobian="analytic"),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
    ),
)
simulation.validate().require_valid()
result = simulation.run(start_time=0.0, end_time=end_time, time_step=time_step)


# ## Comparison with the series solution
#
# We evaluate the computed concentration along the line at several times and
# compare it with the truncated quarter-wave series (401 terms).

def analytic(x, t, terms=401):
    theta = np.zeros_like(x, dtype=float)
    for n in range(1, terms + 1):
        m = 2 * n - 1
        theta += (4.0 / (m * np.pi)) * np.sin(m * np.pi * x / (2 * L)) * np.exp(
            -((m * np.pi / (2 * L)) ** 2) * D * t)
    return C_s + (C_i - C_s) * theta

# The sudden surface condition creates a corner singularity at (x=0, t=0); right
# at the open face at the earliest times neither the series nor the mesh is
# meaningfully resolved. The comparison stays in the interior and starts at the
# second stored time.
sample_x = np.linspace(0.06 * L, 0.98 * L, 40)
stored_times = np.asarray([float(t) for t in result.times])
positive = stored_times[stored_times > 0.0]
probe_times = positive[np.linspace(1, len(positive) - 1, 6).round().astype(int)]
max_abs_error = 0.0
profiles = {}
print("Error per time (mol/m^3):")
for t in probe_times:
    pts = np.column_stack((sample_x, np.zeros_like(sample_x), np.zeros_like(sample_x)))
    num = np.asarray(result.field(model.concentration("A")).evaluate(pts, time=float(t))).reshape(-1)
    ana = analytic(sample_x, float(t))
    err = float(np.max(np.abs(num - ana)))
    print(f"  t={t:8.0f} s  max|dC|={err:7.4f}")
    profiles[float(t)] = (num, ana)
    max_abs_error = max(max_abs_error, err)

rel_error = max_abs_error / abs(C_i - C_s)
print(f"Maximum absolute error against the series: {max_abs_error:.4f} mol/m^3  ({100*rel_error:.2f} % of Cs-Ci)")
assert rel_error < 0.05, "Deviation larger than the example tolerance (5 % of Cs-Ci)."
# Physical sanity: the sealed back face fills monotonically and never overshoots.
back = np.array([[L, 0.0, 0.0]])
back_C = [float(np.asarray(result.field(model.concentration("A")).evaluate(back, time=float(t))).reshape(-1)[0]) for t in probe_times]
assert all(b >= a - 1e-9 for a, b in zip(back_C, back_C[1:])), "The sealed face should only fill."
assert back_C[0] > C_i and back_C[-1] < C_s and back_C[-1] > back_C[0]


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(probe_times)))
for (t, (num, ana)), c in zip(profiles.items(), colors):
    ax.plot(sample_x * 1000, ana, color=c, lw=1.6)
    ax.plot(sample_x * 1000, num, "o", color=c, ms=3)
    ax.text(sample_x[-1] * 1000, ana[-1], f" t={t/3600:.1f}h", color=c, fontsize=8, va="center")
ax.set(xlabel="Position x / mm", ylabel="Concentration / (mol/m^3)",
       title="Diffusing slab — lines: series solution, dots: OneSimulation FEM")
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Refine the mesh or halve the time step and watch the error against the series
# shrink. Open the back face as well (a second fixed-concentration boundary) to
# recover the two-sided slab; the reference then becomes the half-wave series
# $\sum_{n\,\mathrm{odd}}(4/n\pi)\sin(n\pi x/L)\exp[-(n\pi/L)^2 D t]$.
