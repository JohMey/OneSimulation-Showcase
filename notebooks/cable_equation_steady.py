"""OneSimulation example: the steady passive cable (electrotonic decay).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the known closed-form steady-state
solution of the linear cable equation.
"""


# # Electrotonic decay along a passive cable
#
# A long, thin passive nerve fibre (a "cable") is held at a fixed membrane
# potential $V_0$ at one end ($x=0$). The other end is **sealed** (no current
# leaves it). Away from the clamped end the potential decays smoothly because
# current leaks steadily across the membrane while it spreads along the fibre.
#
# The passive membrane is modelled as a monodomain cable with a single linear
# **leak channel** of conductance $g_L$ and reversal potential $E_L$:
#
# $$C_m\,\frac{\partial V}{\partial t}
#    = \frac{\sigma}{\beta}\,\frac{\partial^2 V}{\partial x^2}
#      - g_L\,(V-E_L),$$
#
# with axial conductivity $\sigma$, surface-to-volume ratio $\beta$ and membrane
# capacitance $C_m$. OneSimulation integrates this in time on a real 1D line
# mesh until the transient has died out and only the steady profile remains.
#
# **Reference.** In steady state ($\partial_t V=0$) the deviation
# $v=V-E_L$ obeys $\lambda^2 v''-v=0$ with the **space constant**
#
# $$\lambda=\sqrt{\dfrac{\sigma}{\beta\,g_L}}\;=\;\sqrt{\dfrac{r_m}{r_i}}.$$
#
# For a clamp $v(0)=v_0$ at $x=0$ and a sealed end $v'(L)=0$ at $x=L$ the
# closed-form solution is the hyperbolic-cosine profile
#
# $$v(x)=v_0\,\frac{\cosh\!\big((L-x)/\lambda\big)}{\cosh(L/\lambda)}.$$
#
# We check the computed steady profile against this form and also recover the
# space constant $\lambda$ from a fit of the numerical solution.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.physics.biophysics import (
    MembraneChannel,
    MembraneCurrentElectrode,
    MembraneKinetics,
    MembraneVoltageClamp,
    Monodomain,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The values are freely adjustable and in SI units. They are chosen so that the
# cable is several space constants long, which makes the electrotonic decay
# clearly visible while the sealed end still has a small, non-zero potential.

L = 0.06              # m, cable length
cells = 240           # mesh cells along the fibre
sigma = 1.0           # S/m, axial (intracellular) conductivity
beta = 2000.0         # 1/m, membrane surface-to-volume ratio
C_m = 0.01            # F/m^2, membrane capacitance
g_L = 2.0             # S/m^2, passive leak conductance
E_L = -0.065          # V, leak reversal (resting) potential
delta = 0.025         # V, clamp depolarisation above rest
V0 = E_L + delta      # V, clamped potential at x = 0

lam = np.sqrt(sigma / (beta * g_L))   # space constant
print(f"space constant lambda = {lam*1000:.3f} mm,  L/lambda = {L/lam:.2f}")

# Characteristic times: membrane charging C_m/g_L and axial spreading L^2 * beta
# * C_m / sigma. The run must be long enough for both to settle.
D = sigma / (beta * C_m)
tau = max(C_m / g_L, L * L / D)
end_time = 25.0 * tau
print(f"diffusivity = {D:.3e} m^2/s,  settling time ~ {tau:.3f} s,  end time = {end_time:.3f} s")

line = Line(start=(0.0, 0.0, 0.0), end=(L, 0.0, 0.0))
mesh = StructuredLineMesher(cells=cells).generate(line)

# A single linear leak channel makes the membrane passive.
model = Monodomain(
    conductivity=sigma,
    capacitance=C_m,
    surface_to_volume=beta,
    membrane=MembraneKinetics(
        channels=(MembraneChannel(name="leak", conductance=g_L, reversal_potential=E_L),),
    ),
)

# Both ends carry an explicit boundary condition: a voltage clamp at x = 0 and a
# sealed (zero-current) electrode at x = L.
electrodes = (
    MembraneVoltageClamp(boundary=line.boundaries.start, voltage=lambda t: V0),
    MembraneCurrentElectrode(boundary=line.boundaries.end, current_density=0.0),
)

times = np.linspace(0.0, end_time, 60)
result = model.solve(
    mesh=mesh,
    times=times,
    max_step=tau / 50.0,
    initial_voltage=E_L,
    electrodes=electrodes,
)
print("Mesh nodes:", result.voltage.shape[1], " stored times:", len(result.times))


# ## Comparison with the closed-form steady profile
#
# The last stored time is deep in the steady regime. We compare the computed
# deviation from rest with the hyperbolic-cosine reference and also fit the space
# constant.

x = result.coordinates[:, 0]
order = np.argsort(x)
x = x[order]
v_num = result.voltage[-1][order] - E_L


def reference(xx):
    return delta * np.cosh((L - xx) / lam) / np.cosh(L / lam)


v_ref = reference(x)
max_abs_error = float(np.max(np.abs(v_num - v_ref)))
rel_error = max_abs_error / delta
print(f"maximum deviation from the cosh reference: {max_abs_error*1000:.4f} mV  ({100*rel_error:.2f} % of the clamp step)")
assert rel_error < 0.03, "Steady profile deviates more than the example tolerance (3 %)."

# Confirm the run really reached steady state: the last two stored profiles agree.
drift = float(np.max(np.abs(result.voltage[-1] - result.voltage[-2])))
print(f"change over the last stored interval: {drift*1e6:.3f} uV")
assert drift < 1e-4 * delta, "The transient has not settled to the steady state."

# Recover the space constant from the numerical profile. Over the interior, away
# from the sealed end, the decay is essentially a pure exponential e^{-x/lambda}.
interior = x < 0.6 * L
slope = np.polyfit(x[interior], np.log(v_num[interior]), 1)[0]
lam_fit = -1.0 / slope
print(f"fitted space constant: {lam_fit*1000:.3f} mm  (reference {lam*1000:.3f} mm)")
assert abs(lam_fit - lam) / lam < 0.03, "Fitted space constant is off by more than 3 %."

# Physical sanity: the profile decays monotonically from the clamp to the sealed
# end, where it is still clearly positive, and stays below the clamp step.
assert np.all(np.diff(v_num) <= 1e-9), "The potential should decay monotonically."
assert 0.0 < v_num[-1] < delta
assert abs(v_num[0] - delta) < 1e-3 * delta


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
xs = np.linspace(0.0, L, 400)
ax.plot(xs * 1000, reference(xs) * 1000, color="#8c5a2b", lw=1.8, label="cable equation (cosh)")
ax.plot(x * 1000, v_num * 1000, "o", color="#c8873f", ms=3, label="OneSimulation")
ax.axhline(0.0, color="#999999", lw=0.8)
ax.annotate(f"$\\lambda$ = {lam*1000:.2f} mm", xy=(lam * 1000, reference(lam) * 1000),
            xytext=(lam * 1000 + 6, delta * 1000 * 0.6),
            arrowprops=dict(arrowstyle="->", color="#555555"), color="#555555", fontsize=9)
ax.set(xlabel="Position x / mm", ylabel="Depolarisation V - E_L / mV",
       title="Steady passive cable — line: cable equation, dots: OneSimulation")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Lengthen the cable and the sealed-end potential drops toward zero, recovering
# the semi-infinite result $v(x)=v_0 e^{-x/\lambda}$. Raise the leak conductance
# $g_L$ (a leakier membrane) and the space constant $\lambda$ shrinks, so the
# signal decays over a shorter distance.
