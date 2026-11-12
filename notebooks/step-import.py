"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # 09 · CAD aus STEP: Hülse unter Zug
#
# **OneSimulation Runtime Alpha 0.2.1a4 · STEP-Import · Tetraedernetz · lineare Elastizität · Q2-FEM**
#
# Dieses Beispiel nimmt eine Geometrie aus einer **STEP-Datei**, vernetzt sie mit
# `TetrahedralMesher` und löst darauf eine stationäre lineare Elastizitätsrechnung.
# Gezeigt wird der vollständige Weg: Datei lesen, Flächen des gelesenen Körpers
# inspizieren, Flächen benennen, vernetzen, lösen, prüfen und nach ParaView
# exportieren.
#
# Gerechnet wird eine Hülse mit 12 mm Innen- und 18 mm Außenradius und 100 mm
# Länge aus austenitischem Edelstahl 1.4301 (AISI 304). Die untere Stirnfläche ist
# vollständig eingespannt, auf die obere wirkt eine gleichmäßige Zugspannung von
# 40 MPa; Innen- und Außenwand sind lastfrei.
#
# **Eigene CAD-Datei verwenden:** In der ersten Codezelle steht
# `STEP_DATEI = None`. Trägt man dort den Pfad der eigenen `.step`- oder
# `.stp`-Datei ein, rechnet das Notebook mit dieser Geometrie. Dann sind
# `FLAECHEN_GESAMT` und `FLAECHENGRUPPEN` an die Ausgabe der Flächentabelle
# anzupassen und die analytische Vergleichsrechnung weiter unten gilt nicht mehr.
# Ohne Eintrag schreibt das Notebook die Beispielhülse selbst als STEP-Datei, damit
# es ohne Fremddaten und ohne Internetzugang vollständig durchläuft.
#
# Voraussetzung: das regulär installierte und aktivierte Runtime-Paket, NumPy,
# Matplotlib und Plotly. Alle Codezellen in Reihenfolge ausführen.

from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad import CadGeometryBuilder
from onesim.cad.d3 import AnnularCylinder, ImportedSolid
from onesim.cad.exchange import survey_step
from onesim.materials.library import standard_material
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacement, LinearElasticity, Traction, TractionFree,
)
from onesim.simulation import Problem, Simulation

assert version("onesimulation") == "0.2.1a4"
STEP_DATEI = None  # Pfad der eigenen STEP-Datei eintragen, um sie zu rechnen.
PROFILE = "SCHNELL"  # Alternativ bewusst "HOCHAUFGELOEST" wählen.
PROFILES = {"SCHNELL": {"target_size": 0.006}, "HOCHAUFGELOEST": {"target_size": 0.003}}
config = PROFILES[PROFILE]
output_dir = Path("ergebnisse_onesimulation")
output_dir.mkdir(exist_ok=True)

INNENRADIUS, AUSSENRADIUS, LAENGE = 0.012, 0.018, 0.100  # m
ZUGSPANNUNG = 40.0e6  # Pa, gleichmäßig auf der oberen Stirnfläche
WERKSTOFF = standard_material("stainless_steel_304")

if STEP_DATEI is None:
    # Keine Fremddaten: der CAD-Kern schreibt die Beispielgeometrie selbst als
    # STEP-Datei. Gelesen wird danach ausschließlich diese Datei.
    entwurf = AnnularCylinder(inner_radius=INNENRADIUS,
                              outer_radius=AUSSENRADIUS, length=LAENGE)
    gebaut = CadGeometryBuilder().build(entwurf)
    step_pfad = output_dir / "huelse.step"
    gebaut.kernel.export_step(gebaut.reference, step_pfad)
    volumen_erwartet = entwurf.measure
else:
    step_pfad = Path(STEP_DATEI)
    volumen_erwartet = None
assert step_pfad.is_file() and step_pfad.stat().st_size > 0, step_pfad
print(f"STEP-Datei: {step_pfad} ({step_pfad.stat().st_size / 1024:.1f} KiB)")


# ## Die gelesenen Flächen ansehen, bevor Ränder benannt werden
#
# Eine STEP-Datei trägt keine Randbedingungsnamen. `survey_step` liest die Datei
# einmal mit dem CAD-Kern und berichtet, wie viele Flächen der Körper tatsächlich
# hat, wie groß jede ist und wohin ihre äußere Normale zeigt. Erst daraus werden
# die Flächen benannt.
#
# Die Hülse zeigt, warum dieser Schritt nötig ist: Innen- und Außenwand haben
# dieselbe radiale Normale und sind nur über die Fläche zu unterscheiden
# (2 π r L). Eine Benennung nach Koordinatenrichtung allein würde sie verwechseln.

uebersicht = survey_step(step_pfad)
print(f"Flächen: {uebersicht.face_count}; Volumen: {uebersicht.volume:.6e} m^3; "
      f"Einheit: {uebersicht.unit}")
print(f"Umschließender Kasten: "
      + ", ".join(f"{wert:+.4f}" for wert in uebersicht.bounds))
print()
print(f"{'Index':>5}  {'Name':<10} {'Fläche / m^2':>14}  Normale")
for flaeche in uebersicht:
    normale = ", ".join(f"{wert:+.3f}" for wert in flaeche.normal)
    print(f"{flaeche.index:>5}  {flaeche.member:<10} {flaeche.area:>14.6e}  ({normale})")

if volumen_erwartet is not None:
    # Der Import darf das Bauteilvolumen nicht verändern.
    volumen_abweichung = abs(uebersicht.volume / volumen_erwartet - 1.0)
    assert volumen_abweichung < 1e-9, volumen_abweichung
    print(f"\nRelative Volumenabweichung gegen den Entwurf: {volumen_abweichung:.3e}")


# ## Flächen benennen, vernetzen und lösen
#
# `ImportedSolid` ist die Geometrieerklärung für eine Datei. Sie nennt den Pfad,
# die Längeneinheit, die **erwartete Flächenzahl** und benannte Gruppen von
# Flächenindizes. Weicht die Flächenzahl der Datei davon ab, bricht die Rechnung
# ab, anstatt die Randbedingungen stillschweigend auf andere Flächen zu legen.
#
# Ab diesem Punkt unterscheidet sich der Ablauf nicht mehr von einer im Code
# erzeugten Geometrie: dieselbe `TetrahedralMesher`-, `Problem`- und
# `Simulation`-Schnittstelle.

FLAECHEN_GESAMT = 4
FLAECHENGRUPPEN = (
    ("innenwand", (0,)),
    ("einspannung", (1,)),
    ("aussenwand", (2,)),
    ("zugflaeche", (3,)),
)
koerper = ImportedSolid(path=str(step_pfad), unit="m",
                        face_count=FLAECHEN_GESAMT, face_groups=FLAECHENGRUPPEN)
werkstoff = WERKSTOFF.create()
model = LinearElasticity(domain=koerper.volume, material=werkstoff)
problem = Problem(
    name="Aus STEP gelesene Hülse unter gleichmäßigem Zug",
    geometry=koerper, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=koerper.boundaries.einspannung,
                          physics=model, value=(0.0, 0.0, 0.0)),
        Traction(boundary=koerper.boundaries.zugflaeche, physics=model,
                 value=(0.0, 0.0, ZUGSPANNUNG)),
        TractionFree(boundary=koerper.boundaries.innenwand
                     | koerper.boundaries.aussenwand, physics=model),
    ),
)

gestartet = perf_counter()
mesh = TetrahedralMesher(target_size=config["target_size"]).generate(koerper)
netz_sekunden = perf_counter() - gestartet
gestartet = perf_counter()
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)),
                      linear_solver=SparseDirect()),
).solve_stationary()
loesung_sekunden = perf_counter() - gestartet
displacement = result.field(model.displacement)
print(f"Tetraeder: {mesh.cell_count}; Netzknoten: {len(mesh.coordinates)}; "
      f"FEM-Knoten: {len(displacement.values)}")
print(f"Vernetzen: {netz_sekunden:.2f} s; Lösen: {loesung_sekunden:.2f} s")


# ## Numerische Prüfungen und Ergebnisdaten
#
# Geprüft werden die festgehaltene Einspannfläche, die tatsächlich integrierte
# Gesamtlast, die Netzabdeckung aller benannten Flächen und der Vergleich mit der
# Stabdehnung.
#
# Die Referenz ist die einachsige Stabformel \(u_z(L)=\sigma L/E\) mit
# \(\sigma = 40\,\text{MPa}\). Sie unterstellt eine querdehnungsfreie
# Einspannung und gilt deshalb nur näherungsweise: die vollständig festgehaltene
# untere Stirnfläche unterdrückt die Querkontraktion über etwa eine
# Durchmesserlänge und macht die Hülse dort steifer.

flaechen_facetten = {
    name: len(mesh.facets(getattr(koerper.boundaries, name)))
    for name, _ in FLAECHENGRUPPEN
}
facetten_gesamt = len(mesh.facets(koerper.boundaries.all))
assert sum(flaechen_facetten.values()) == facetten_gesamt
print("Randfacetten je benannter Fläche:", flaechen_facetten)

mittelradius = 0.5 * (INNENRADIUS + AUSSENRADIUS)
winkel = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
stirn_punkte = np.column_stack((
    mittelradius * np.cos(winkel), mittelradius * np.sin(winkel),
    np.full_like(winkel, LAENGE),
))
stirn_u = np.asarray(displacement.evaluate(stirn_punkte))
einspann_punkte = np.column_stack((
    mittelradius * np.cos(winkel), mittelradius * np.sin(winkel),
    np.zeros_like(winkel),
))
einspann_max = float(np.max(np.abs(displacement.evaluate(einspann_punkte))))

z = np.linspace(0.0, LAENGE, 61)
achse = np.column_stack((np.full_like(z, mittelradius), np.zeros_like(z), z))
achse_u = np.asarray(displacement.evaluate(achse))
youngs_modulus = float(WERKSTOFF.properties["youngs_modulus"])
stab_referenz = ZUGSPANNUNG * z / youngs_modulus
dehnung_berechnet = float(stirn_u[:, 2].mean())
dehnung_referenz = ZUGSPANNUNG * LAENGE / youngs_modulus
dehnung_abweichung = abs(dehnung_berechnet / dehnung_referenz - 1.0)

operator = result.operators[model.model_id]
_, _, knotenlasten = operator.conditions(0.0)
integrierte_last = np.asarray(knotenlasten).reshape(-1, 3).sum(axis=0)
querschnitt = np.pi * (AUSSENRADIUS**2 - INNENRADIUS**2)
sollkraft = ZUGSPANNUNG * querschnitt
# Integriert wird über die dreieckig facettierte Stirnfläche. Deren Flächeninhalt
# ist geringfügig kleiner als der des exakten Kreisrings, die Zugkraft deshalb
# ebenfalls. Der Unterschied ist Geometriediskretisierung, kein Bilanzfehler.
last_quer = float(np.linalg.norm(integrierte_last[:2]) / sollkraft)
last_abweichung = float(abs(integrierte_last[2] / sollkraft - 1.0))

mises = np.asarray(result.field(model.von_mises_stress).evaluate(
    np.array([[mittelradius, 0.0, LAENGE / 2]])))
mises_abweichung = float(abs(mises[0] / ZUGSPANNUNG - 1.0))

assert np.isfinite(achse_u).all() and np.isfinite(mises).all()
assert einspann_max < 1e-12
assert last_quer < 1e-12
assert last_abweichung < 0.01
assert float(np.ptp(stirn_u[:, 2])) / abs(dehnung_berechnet) < 0.01
assert dehnung_abweichung < 0.03
assert mises_abweichung < 0.02

print(f"Verlängerung: {dehnung_berechnet * 1e6:.3f} um; "
      f"Stabformel: {dehnung_referenz * 1e6:.3f} um "
      f"(Abweichung {100 * dehnung_abweichung:.2f} %)")
print(f"Integrierte Zugkraft: {integrierte_last[2]:.6f} N; "
      f"exakter Kreisring: {sollkraft:.6f} N "
      f"(Facettierung {100 * last_abweichung:.3f} %)")
print(f"von-Mises-Spannung auf halber Höhe: {mises[0] / 1e6:.3f} MPa "
      f"(Abweichung {100 * mises_abweichung:.2f} %)")

csv_pfad = output_dir / "step_import_huelse.csv"
with csv_pfad.open("w", encoding="utf-8", newline="") as datei:
    schreiber = csv.writer(datei)
    schreiber.writerow(["z_m", "ux_m", "uy_m", "uz_m", "stabformel_uz_m"])
    schreiber.writerows(np.column_stack((z, achse_u, stab_referenz)))


# ## Ergebnisse darstellen und weitergeben
#
# Links die berechnete Verlängerung über der Höhe gegen die Stabformel, rechts
# dasselbe Spannungsfeld als Bibliotheksgrafik aus `onesim.visualization`.
# `write_for_paraview` schreibt die genannten Felder zusätzlich auf das gerechnete
# Netz, lesbar in ParaView und VisIt. Die Werte werden an den geschriebenen
# Netzknoten ausgewertet; die quadratische Lösung erscheint dort als lineare
# Ansicht auf denselben Knoten.

import onesim.visualization as visualization
from onesim.results.storage import write_for_paraview

cream, ink, green, copper = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "text.color": ink, "axes.labelcolor": ink,
                     "xtick.color": ink, "ytick.color": ink,
                     "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=cream)
ax.set_facecolor(cream)
ax.plot(z * 1000, achse_u[:, 2] * 1e6, color=green, linewidth=2.4,
        label="OneSimulation, Q2-FEM auf dem STEP-Körper")
ax.plot(z * 1000, stab_referenz * 1e6, color=copper, linewidth=1.6,
        linestyle="--", label=r"Stabformel $u_z=\sigma z/E$")
ax.set_xlabel("Höhe z / mm")
ax.set_ylabel(r"Verlängerung $u_z$ / µm")
ax.set_title("Aus STEP gelesene Hülse unter 40 MPa Zug", color=ink)
ax.legend(frameon=False)
ax.grid(True, color=ink, alpha=0.12)
fig.tight_layout()
png_pfad = output_dir / "step_import_huelse.png"
svg_pfad = output_dir / "step_import_huelse.svg"
fig.savefig(png_pfad, dpi=200)
fig.savefig(svg_pfad)

PARAVIEW_FELDER = (model.displacement, model.von_mises_stress, model.stress)
paraview_dateien = [
    write_for_paraview(result, mesh, PARAVIEW_FELDER,
                       output_dir / f"step_import_huelse.{suffix}")
    for suffix in ("vtk", "xdmf")
]
for pfad in paraview_dateien:
    assert pfad.is_file() and pfad.stat().st_size > 1000, pfad
    print(f"ParaView-Datei: {pfad} ({pfad.stat().st_size / 1024:.1f} KiB)")

figure = visualization.result(result.result_set, model.von_mises_stress.field_id,
                              title="von-Mises-Spannung (Pa)")
assert figure.data, "Die Bibliotheksgrafik enthält keine Spur"
figure


# ## Aussagekraft und Grenzen
#
# Gelesen wird eine STEP-Datei mit dem mitgelieferten CAD-Kern; andere
# Austauschformate und STEP-Dateien mit fehlerhafter Topologie sind damit nicht
# zugesichert. Flächen werden über ihren **Index in der Flächenreihenfolge des
# Kerns** benannt. Diese Reihenfolge ist für eine gegebene Datei reproduzierbar,
# aber sie ist keine vom Format garantierte Eigenschaft: nach einer Änderung der
# CAD-Datei ist die Flächentabelle erneut zu prüfen. Die erklärte Flächenzahl ist
# genau dafür eine Abbruchbedingung.
#
# Gerechnet wird eine stationäre, isotrope, linear-elastische Aufgabe mit kleinen
# Dehnungen, ohne Eigengewicht, Kontakt, Plastizität oder Temperatur. Die
# Werkstoffwerte sind nominale Handbuchwerte bei 20 °C und keine abgenommenen
# Werkstoffdaten; vor einem Nachweis sind sie gegen das Datenblatt der
# tatsächlichen Charge zu prüfen. Die Stabformel ist eine unabhängige
# Plausibilitätsreferenz, kein exaktes dreidimensionales Vergleichsproblem. Ein
# Netz ist keine Konvergenzstudie. Es liegt keine experimentelle oder
# vergleichende COMSOL-Validierung vor.

verification = {
    "passed": True,
    "profile": PROFILE,
    "runtime_version": version("onesimulation"),
    "example": "linear elastic tension on a solid imported from STEP",
    "step_file": str(step_pfad),
    "step_file_bytes": int(step_pfad.stat().st_size),
    "imported_face_count": int(uebersicht.face_count),
    "declared_face_count": FLAECHEN_GESAMT,
    "imported_volume_m3": float(uebersicht.volume),
    "imported_bounds_m": [float(wert) for wert in uebersicht.bounds],
    "face_group_boundary_facets": flaechen_facetten,
    "boundary_facets_total": int(facetten_gesamt),
    "mesh_cells": int(mesh.cell_count),
    "mesh_nodes": int(len(mesh.coordinates)),
    "fem_nodes": int(len(displacement.values)),
    "material": WERKSTOFF.name,
    "material_quality": WERKSTOFF.quality,
    "youngs_modulus_pa": youngs_modulus,
    "applied_tension_pa": ZUGSPANNUNG,
    "cross_section_m2": float(querschnitt),
    "computed_elongation_m": dehnung_berechnet,
    "bar_formula_elongation_m": dehnung_referenz,
    "elongation_relative_difference": dehnung_abweichung,
    "integrated_force_n": integrierte_last.tolist(),
    "target_force_n": float(sollkraft),
    "axial_load_relative_difference": last_abweichung,
    "transverse_load_relative_error": last_quer,
    "midheight_von_mises_pa": float(mises[0]),
    "midheight_von_mises_relative_difference": mises_abweichung,
    "mesh_seconds": float(netz_sekunden),
    "solve_seconds": float(loesung_sekunden),
    "output_files": [str(csv_pfad), str(png_pfad), str(svg_pfad),
                     *(str(pfad) for pfad in paraview_dateien)],
    "limitations": [
        "STEP read with the bundled CAD kernel; other exchange formats are not qualified",
        "Boundaries are named by kernel face index; recheck the face table after a CAD change",
        "Nominal handbook material values at 20 degC; no certified material data",
        "Small-strain stationary isotropic linear elasticity; no contact, plasticity or thermal load",
        "Bar formula is an approximate reference; the clamped end suppresses lateral contraction",
        "One mesh is not a convergence study",
        "The loaded end face is triangulated, so the integrated force is slightly below the exact annulus",
        "ParaView export samples the quadratic solution at the written mesh vertices",
        "No experimental or COMSOL validation",
    ],
}
assert all(Path(pfad).stat().st_size > 0 for pfad in verification["output_files"])
json_pfad = output_dir / "step_import_huelse_verifikation.json"
json_pfad.write_text(json.dumps(verification, indent=2, ensure_ascii=False),
                     encoding="utf-8")
print(json.dumps(verification, indent=2, ensure_ascii=False))
