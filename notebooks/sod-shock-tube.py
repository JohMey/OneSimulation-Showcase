"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # Sod-Stoßrohr: WENO5 und RK4 über die direkte Python-API
#
# Dieses Beispiel formuliert die eindimensionalen kompressiblen Euler-Gleichungen direkt über `Problem` und `Simulation`. Es importiert **keinen vorgefertigten Sod-Fall und keine analytische Lösung**. Voraussetzung: aktivierte OneSimulation Runtime Alpha **0.2.1a4**, NumPy und Matplotlib.
#
# Normierte Anfangswerte links/rechts: Dichte1/0,125; Druck1/0,1; Geschwindigkeit0; γ=1,4. Ausgewertet wird t=0,2 auf x∈[0,1] mit transmissiven Enden. WENO5-Jiang–Shu rekonstruiert komponentenweise gespaltene Flüsse; die hohe räumliche Ordnung gilt in glatten Bereichen. Diskontinuitäten werden durch die nichtlinearen WENO-Gewichte behandelt. Dies ist keine allgemeine Positivitätsgarantie oder experimentelle Validierung.

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import (
    FiniteDifference, GlobalLaxFriedrichsSplit, WENO5Flux, WENOFluxDifference,
)
from onesim.numerics.time import ClassicalRungeKutta4, TimeStepLimits
from onesim.physics.hydrodynamics import (
    CompressibleEuler, IdealGasEquationOfState, TransmissiveBoundary,
)
from onesim.results.storage import MemoryOutputSink, Output, ResultRetentionPolicy
from onesim.simulation import ExecutionOptions, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

POINTS = 1600
DT = 0.00005
OUTPUT = Path('ergebnisse_sod')


# ## Modell und tatsächliche Rechnung
#
# Das Netz enthält1.600 äquidistante Punktwerte. Klassisches RK4 integriert in der Zeit; eine explizite CFL-Grenze schützt die Zeitschrittwahl. `DT` ist eine obere Vorgabe und kann durch die Runtime verkleinert werden. Die Endfelder werden behalten;21 Ausgabestände dienen zusätzlichen Positivitätsprüfungen.

# Physik und Anfangswerte: links (rho,p,u)=(1,1,0), rechts (0.125,0.1,0).
gamma, end_time = 1.4, 0.2
geometry = Line(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
gas = CompressibleEuler(
    domain=geometry.line, equation_of_state=IdealGasEquationOfState(gamma=gamma),
)
problem = Problem(
    name='Sod: direkte Euler-API', geometry=geometry, physics=(gas,),
    initial_conditions=(
        InitialValue(gas.density, lambda x: np.where(x < 0.5, 1.0, 0.125)),
        InitialValue(gas.momentum, 0.0),
        InitialValue(gas.total_energy, lambda x: np.where(x < 0.5, 1.0, 0.1) / (gamma - 1)),
    ),
    boundary_conditions=(TransmissiveBoundary(physics=gas, boundary=geometry.boundaries.all),),
)
mesh = StructuredLineMesher(cells=POINTS).generate(geometry)
numerics = Numerics(
    spatial=FiniteDifference(gradient=WENOFluxDifference(
        flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux(),
    )),
    time_integrator=ClassicalRungeKutta4(),
    step_limits=TimeStepLimits(cfl_limit=0.4, stability_safety_factor=0.4),
)
simulation = Simulation(problem=problem, mesh=mesh, numerics=numerics)
simulation.validate().require_valid()
snapshots = MemoryOutputSink()
started = time.perf_counter()
result = simulation.run(
    start_time=0.0, end_time=end_time, time_step=DT,
    execution_options=ExecutionOptions(result_retention=ResultRetentionPolicy.final_only()),
    output=Output(fields=(gas.density, gas.pressure), interval=0.01, sink=snapshots),
)
elapsed = time.perf_counter() - started


# ## Positivität und Erhaltungsbilanzen
#
# Dichte und Druck müssen in den gespeicherten Feldern positiv bleiben. Masse und Gesamtenergie bleiben bis t=0,2 praktisch unverändert, da die Störungen die offenen Ränder noch nicht erreicht haben. Der Gesamtimpuls wächst dagegen wegen des Druckunterschieds an den Enden: Die geprüfte Bilanz berücksichtigt diesen Randbeitrag. Die CSV enthält ausschließlich numerisch berechnete Werte.

# Numerische Kontrollen aus den tatsächlich berechneten Feldern.
x = np.asarray(mesh.cell_centers[:, 0])
rho, velocity, pressure, energy, momentum = [
    np.asarray(result.field(field).values)
    for field in (gas.density, gas.velocity, gas.pressure, gas.total_energy, gas.momentum)
]
assert np.isfinite([rho, velocity, pressure, energy, momentum]).all()
assert rho.min() > 0 and pressure.min() > 0
assert all(np.min(field.values) > 0 for snap in snapshots.snapshots for field in snap.fields.values())
mass_error = abs(float(rho @ mesh.cell_volumes) - 0.5625)
energy_error = abs(float(energy @ mesh.cell_volumes) - 1.375)
# Offene Enden: der Druckunterschied bewirkt den Impulszuwachs (1-0.1)*t.
momentum_balance_error = abs(float(momentum @ mesh.cell_volumes) - (1.0 - 0.1) * end_time)
assert max(mass_error, energy_error, momentum_balance_error) < 1e-9
verification = dict(
    passed=True, points=POINTS, requested_dt=DT, final_time=end_time,
    solve_seconds=elapsed, minimum_density=float(rho.min()), minimum_pressure=float(pressure.min()),
    mass_absolute_error=mass_error, energy_absolute_error=energy_error,
    momentum_boundary_balance_absolute_error=momentum_balance_error,
    positive_saved_snapshots=len(snapshots.snapshots),
    final_accepted_step=snapshots.snapshots[-1].accepted_step,
    spatial='WENO5-JS finite differences, componentwise global LF flux splitting',
    time_integrator='classical RK4', analytic_solution_used=False,
)
OUTPUT.mkdir(parents=True, exist_ok=True)
np.savetxt(OUTPUT / 'sod_felder.csv', np.column_stack((x, rho, velocity, pressure, energy, momentum)),
           delimiter=',', header='x,density,velocity,pressure,total_energy,momentum', comments='')
(OUTPUT / 'pruefung.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')


# ## Felder zeichnen
#
# Die durchgezogenen Kurven stammen ausschließlich aus der Rechnung. Gestrichelt sind die vorgegebenen Anfangswerte dargestellt, keine analytische Vergleichslösung. Für eine Netz-/Zeitstudie kann das gleiche Modell mit anderen `POINTS`-/`DT`-Werten erneut ausgeführt werden; Abstände zwischen numerischen Lösungen sind keine exakten Fehler.

# Nur numerische Felder; die gestrichelte Linie zeigt den vorgegebenen Anfangszustand.
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='#f6f1e7')
for ax, values, initial, title in zip(axes.flat, (rho, velocity, pressure, energy),
        (np.where(x < 0.5, 1.0, 0.125), np.zeros_like(x), np.where(x < 0.5, 1.0, 0.1),
         np.where(x < 0.5, 2.5, 0.25)), ('Dichte ρ', 'Geschwindigkeit u', 'Druck p', 'Gesamtenergiedichte E')):
    ax.set_facecolor('#fffdf7')
    ax.plot(x, initial, '--', color='#b4aa98', lw=1.1, label='Anfangswerte t = 0')
    ax.plot(x, values, color='#39715e', lw=2, label=f'Numerik t = {end_time:g}')
    ax.set(xlabel='Position x', ylabel=title, xlim=(0, 1), title=title)
    ax.grid(alpha=0.16)
axes[0, 0].legend(frameon=False, fontsize=10)
fig.suptitle('Eine Drucksprungstelle wird zur Stoßwelle.', fontsize=26, color='#203f36', x=.07, ha='left')
point_label = f'{POINTS:,}'.replace(',', '.')
fig.text(.07, .915, f'Sod-Stoßrohr · {point_label} Stützstellen · WENO5 + RK4 · t = 0,2', color='#39715e', fontsize=14)
fig.text(.07, .025, 'Direkte Python-API · dimensionslos · keine analytische Kurve · keine experimentelle Validierung', color='#6d746c')
fig.subplots_adjust(top=.84, bottom=.10, left=.08, right=.96, hspace=.34, wspace=.24)
fig.savefig(OUTPUT / 'sod_weno5.png', dpi=300)
fig.savefig(OUTPUT / 'sod_weno5.svg')
plt.close(fig)
print(json.dumps(verification, ensure_ascii=False))
