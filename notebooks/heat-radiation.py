"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # Wärmestrahlung mit OneSimulation
#
# Dieses Beispiel berechnet die Abkühlung eines Körpers mit **einer einheitlichen Temperatur** über die installierte OneSimulation-API. Voraussetzung: aktivierte Runtime Alpha 0.2.1a4 und die in `README_DE.md` beschriebene Python-Umgebung. Führe alle Zellen der Reihe nach aus. Die Aktivierung wird hier nicht erneut gestartet; kein Kauf- oder Lizenzschlüssel gehört in dieses Notebook.
#
# Das Modell lautet
#
# $$C\,\frac{dT}{dt}=-\varepsilon\,\sigma\,A\,(T^4-T_\mathrm{U}^4).$$
#
# $C$ ist die gesamte Wärmekapazität in J/K, $A$ die abstrahlende Fläche in m², $\varepsilon$ der Emissionsgrad und $\sigma$ die Stefan-Boltzmann-Konstante. Für den mathematischen Prüffall setzen wir **$T_\mathrm{U}=0$ K**. Das ist eine idealisierte Referenz und keine realistische Umgebungsannahme. Innere Temperaturgradienten, Konvektion und weitere Wärmequellen werden nicht modelliert.

from pathlib import Path
from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

runtime_version = version("onesimulation")
print("OneSimulation:", runtime_version)
assert runtime_version == "0.2.1a4", "Für diese Beispiele ist Runtime Alpha 0.2.1a4 vorgesehen."

from onesim.physics.heat_transfer import RadiationCooling, ThermalGreyDiffuseRadiationBoundary


# ## Parameter und Modell
#
# Die Parameter können verändert werden. Die folgende Referenzformel bleibt nur bei konstanter Wärmekapazität, konstanter Fläche, konstantem Emissionsgrad und $T_\mathrm{U}=0$ K gültig. Die hier eingebauten Fehlertoleranzen gelten für die mitgelieferte Parameterauswahl.

heat_capacity = 100.0          # J/K; gesamte Wärmekapazität
area = 0.1                     # m²
emissivity = 0.8               # dimensionslos, 0 bis 1
initial_temperature = 600.0    # K
ambient_temperature = 0.0      # K; idealisierter analytischer Prüffall
end_time = 20.0                # s
time_step = 0.5                # s

assert heat_capacity > 0 and area > 0 and initial_temperature > 0
assert 0 < emissivity <= 1 and ambient_temperature == 0

domain = LumpedDomain()
radiation = ThermalGreyDiffuseRadiationBoundary(
    condition_id="notebook-radiation",
    emissivity=emissivity,
    ambient_temperature=ambient_temperature,
)
model = RadiationCooling(
    domain=domain.system,
    heat_capacity=heat_capacity,
    radiating_area=area,
    radiation=radiation,
)
simulation = Simulation(
    problem=Problem(
        name="Wärmestrahlung: analytischer Referenzfall",
        geometry=domain,
        physics=(model,),
        initial_conditions=(InitialValue(model.temperature, initial_temperature),),
    ),
    mesh=LumpedMesh(domain),
    numerics=Numerics(
        spatial=LumpedParameters(),
        time_integrator=ClassicalRungeKutta4(),
    ),
)
simulation.validate().require_valid()

sink = MemoryOutputSink()
result = simulation.run(
    start_time=0.0,
    end_time=end_time,
    time_step=time_step,
    output=Output(
        fields=(model.temperature, model.radiated_power),
        interval=1.0,
        sink=sink,
    ),
)
times = np.asarray([float(s.time) for s in sink.snapshots])
temperatures = np.asarray([
    float(np.asarray(s.fields[model.temperature.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])
powers = np.asarray([
    float(np.asarray(s.fields[model.radiated_power.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])
assert len(times) > 1 and np.isclose(times[-1], end_time)


# ## Vergleich mit der geschlossenen Lösung
#
# Für $T_\mathrm{U}=0$ K folgt durch Trennung der Variablen
#
# $$T(t)=\left(T_0^{-3}+\frac{3\varepsilon\sigma A}{C}\,t\right)^{-1/3}.$$
#
# Wir vergleichen sämtliche gespeicherten Zeitpunkte. Eine bestandene Prüfung zeigt die Übereinstimmung dieses numerischen Beispiels mit dieser Gleichung, keine experimentelle Validierung eines realen Systems.

sigma = 5.670374419e-8  # W/(m² K⁴)
reference_temperature = (
    initial_temperature**(-3) + 3 * emissivity * sigma * area * times / heat_capacity
) ** (-1 / 3)
max_temperature_error = float(np.max(np.abs(temperatures - reference_temperature)))
assert max_temperature_error < 1e-6, "Referenzfehler größer als die Beispieltoleranz."
assert np.all(np.diff(temperatures) < 0), "Der Körper muss in diesem Beispiel abkühlen."
assert np.all(powers >= 0)
print(f"Endtemperatur: {temperatures[-1]:.8f} K")
print(f"Maximaler absoluter Referenzfehler: {max_temperature_error:.3e} K")

output_dir = Path("ergebnisse_onesimulation")
output_dir.mkdir(exist_ok=True)
np.savetxt(
    output_dir / "waermestrahlung.csv",
    np.column_stack((times, temperatures, reference_temperature, powers)),
    delimiter=",",
    header="time_s,temperature_K,reference_temperature_K,radiated_power_W",
    comments="",
)
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
ax.set_facecolor("#fffaf2")
ax.plot(times, reference_temperature, color="#6b442f", label="Analytische Referenz")
ax.plot(times, temperatures, "o", color="#c56838", markersize=4, label="OneSimulation RK4")
ax.set(xlabel="Zeit / s", ylabel="Temperatur / K", title="Idealisierte Strahlungskühlung")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(output_dir / "waermestrahlung.svg")
plt.show()
print("CSV und SVG gespeichert in:", output_dir)


# ## Weiterarbeiten
#
# Halbiere die Schrittweite und vergleiche den Referenzfehler. Ändere anschließend einzeln Fläche, Wärmekapazität oder Emissionsgrad und erkläre die Wirkung anhand der Gleichung. Bei anderen Umgebungsbedingungen ist die hier verwendete Referenzformel nicht mehr gültig; ersetze sie vor einer solchen Untersuchung durch eine geeignete Vergleichslösung. Wiederholtes Ausführen überschreibt die beiden gleichnamigen Ergebnisdateien.
