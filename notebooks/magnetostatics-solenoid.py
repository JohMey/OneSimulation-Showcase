"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # Magnetostatik: das dreidimensionale Feld einer Spule
#
# **OneSimulation Runtime Alpha 0.2.1a4 · 3D-FEM · Nédélec-Elemente und Coulomb-Eichung**
#
# Eine homogenisierte Ringspule (Innenradius 20 mm, Außenradius 30 mm, Länge 20 mm)
# führt 1 A bei 10 Windungen. Ihr Strom wird als azimutale Volumenstromdichte auf einem
# explizit vernetzten Wicklungsgebiet modelliert. OneSimulation löst
# \(\nabla\times(\mu^{-1}\nabla\times\mathbf A)=\mathbf J\) mit schwacher Coulomb-Eichung
# und rekonstruiert \(\mathbf B=\nabla\times\mathbf A\), \(\mathbf H=\mathbf B/\mu\).
# Dies ist eine tatsächliche räumliche FEM-Rechnung, keine als Solver ausgegebene
# analytische Spulenformel.
#
# Voraussetzung ist das regulär installierte und aktivierte Runtime-Paket 0.2.1a4
# mit NumPy und Matplotlib. Alle Zellen der Reihe nach ausführen; Jupyter ist optional.
# Keine Internetdaten oder Materialkataloge sind erforderlich.
#
# **Auflösung wählen:** Die erste Codezelle enthält `PROFILE = "SCHNELL"`.
# Das schnelle Standardprofil behält seine bisherigen Netzeinstellungen bei.
# `PROFILE = "HOCHAUFGELOEST"` erzeugt und löst ein echtes feineres Tetraedernetz
# mit 6 mm Zielgröße im Rechenraum und 2,5 mm im Wicklungsgebiet.
# Die tatsächliche Elementzahl wird ausgegeben. Dafür werden mehrere GiB freier
# Arbeitsspeicher und mehrere Minuten Rechenzeit benötigt. Um den Speicherbedarf
# zu begrenzen, erfolgt der zusätzliche Stromverdopplungstest nur im schnellen
# Profil; die Eichungs-, Materialgesetz- und Quellenprüfungen laufen in beiden.
#
# **Modellgrenzen:** Vakuumpermeabilität, lineare stationäre Rechnung, homogenisierte
# Wicklung ohne Einzelwindungen oder Zuleitungen. Der endliche Zylinder (Radius 50 mm,
# Länge 80 mm) erhält außen magnetische Isolation; diese Randbedingung beeinflusst
# das Feld. Die Rechnung repräsentiert deshalb keine unendlich ausgedehnte freie
# Umgebung und keine experimentell validierte reale Spule. Die gezeichneten
# Kupferringe illustrieren die Windungen, nicht zusätzliche separat berechnete Drähte.

from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

from onesim.cad.d3 import Cylinder
from onesim.materials import VacuumPermeability
from onesim.meshing import TetrahedralMesher, LocalRefinement
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    CoulombGauge, MagneticInsulation, MagneticVectorPotential,
    Magnetostatics, SolenoidCurrent,
)
from onesim.simulation import Problem, Simulation

assert version("onesimulation") == "0.2.1a4"
PROFILE = "SCHNELL"  # Alternativ bewusst "HOCHAUFGELOEST" wählen.
PROFILES = {
    "SCHNELL": {"target_size": 0.025, "coil_size": 0.012, "grid": 61, "dpi": 180},
    "HOCHAUFGELOEST": {"target_size": 0.006, "coil_size": 0.0025, "grid": 241, "dpi": 300},
}
config = PROFILES[PROFILE]
output_dir = Path("ergebnisse_onesimulation")
output_dir.mkdir(exist_ok=True)
started = perf_counter()
geometry = Cylinder(radius=0.05, length=0.08)
coil = SolenoidCurrent(inner_radius=0.02, outer_radius=0.03, length=0.02,
                       center=(0, 0, 0.04), turns=10, current=1.0)
mesh = TetrahedralMesher(target_size=config["target_size"], embedded_regions=(coil.support,),
    refinements=(LocalRefinement(region=coil.support, target_size=config["coil_size"]),)).generate(geometry)
mesh_seconds = perf_counter() - started

def solve_magnetic(source):
    started = perf_counter()
    model = Magnetostatics(domain=geometry.volume, permeability=VacuumPermeability(),
        formulation=MagneticVectorPotential(gauge=CoulombGauge()), current_sources=(source,))
    result = Simulation(
        problem=Problem(name="FEM-Magnetfeld einer homogenisierten Ringspule",
            geometry=geometry, physics=(model,), boundary_conditions=(
                MagneticInsulation(boundary=geometry.boundaries.all, physics=model),)),
        mesh=mesh,
        numerics=Numerics(spatial=FiniteElement(element=Nedelec(order=1)),
                          linear_solver=SparseDirect()),
    ).solve_stationary()
    return model, result, perf_counter() - started

model, result, solve_seconds = solve_magnetic(coil)
if PROFILE == "SCHNELL":
    model2, result2, doubled_solve_seconds = solve_magnetic(replace(coil, current=2.0))
else:
    # Die Stromverdopplung wird im schnellen Profil tatsächlich gelöst und geprüft.
    # Das große Profil hält für geringeren Speicherbedarf nur eine FEM-Lösung vor.
    model2, result2, doubled_solve_seconds = None, None, None
B_field = result.field(model.magnetic_flux_density)
H_field = result.field(model.magnetic_field_strength)
operator = result.operators[model.model_id]


# ## Das gelöste Feld prüfen
#
# Geprüft werden die tatsächlich aufgelöste Stromregion, die Amperewindungen
# \(NI=10\,\mathrm A\), die diskrete Coulomb-Eichbedingung, \(\mathbf B=\mu\mathbf H\)
# und eine zweite Lösung mit doppeltem Strom. Im linearen Modell muss diese das
# doppelte Magnetfeld liefern. Dies sind Konsistenzprüfungen, keine externe
# Genauigkeitsvalidierung oder abgeschlossene Netzkonvergenzstudie.

check_points = np.array([[0,0,0.04], [0.01,0.005,0.045], [-0.01,-0.005,0.035],
                         [0.035,0,0.06], [-0.035,0,0.02]])
B_check = np.asarray(B_field.evaluate(check_points))
H_check = np.asarray(H_field.evaluate(check_points))
if result2 is not None:
    B_double = np.asarray(result2.field(model2.magnetic_flux_density).evaluate(check_points))
mu = float(VacuumPermeability().evaluate())
linearity_relative_error = (float(np.linalg.norm(B_double - 2*B_check) / np.linalg.norm(2*B_check))
                            if result2 is not None else None)
constitutive_relative_error = float(np.linalg.norm(B_check - mu*H_check) / np.linalg.norm(B_check))
coefficients = np.asarray(result.field(model.vector_potential).coefficients)
n = operator.space.dof_count
gauge_matrix = operator.last_weak_system.matrix.tocsr()[n:, :n]
gauge_residual_norm = float(np.linalg.norm(gauge_matrix @ coefficients))
source_cell_count = len(mesh.region_cells(coil.support))
assert operator.source_cells[0][1] == frozenset(mesh.region_cells(coil.support))
ampere_turns = float(coil.current_density([[0.025,0,0.04]])[0,1] * 0.01 * 0.02)
assert 0 < source_cell_count < mesh.cell_count
assert np.isfinite(B_check).all() and B_check[0,2] > 0
assert abs(ampere_turns - 10.0) < 1e-12
assert linearity_relative_error is None or linearity_relative_error < 1e-8
assert constitutive_relative_error < 1e-12
assert gauge_residual_norm < 1e-10
assert not result.result_set.field(model.vector_potential.field_id).metadata["penalty"]

# Ein tatsächlicher Schnitt durch das räumlich berechnete Feld (y=0).
x = np.linspace(-0.045, 0.045, config["grid"])
z = np.linspace(0.004, 0.076, config["grid"])
xx, zz = np.meshgrid(x, z)
points = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
B = np.asarray(B_field.evaluate(points)).reshape(zz.shape + (3,))
B_magnitude = np.linalg.norm(B, axis=2)
assert np.isfinite(B).all() and np.max(B_magnitude) > 0
csv_path = output_dir / "magnetostatik_spule_3d.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["x_m", "y_m", "z_m", "Bx_T", "By_T", "Bz_T", "B_magnitude_T"])
    writer.writerows(np.column_stack((points, B.reshape(-1,3), B_magnitude.ravel())))
npz_path = output_dir / "magnetostatik_spule_3d_felddaten.npz"
np.savez_compressed(npz_path, x_m=x, z_m=z, B_tesla=B,
    mesh_coordinates_m=mesh.coordinates, mesh_cells=mesh.cells,
    source_cell_indices=np.asarray(sorted(mesh.region_cells(coil.support)), dtype=int))


# ## Aus dem 3D-Feld eine lesbare Abbildung erzeugen
#
# Die räumlichen Pfeile zeigen die berechnete Richtung auf der Mittelebene;
# ihre Längen sind zur Lesbarkeit einheitlich, die Farben geben den Betrag von
# \(\mathbf B\) an. Rechts werden dieselben FEM-Abtastwerte als Schnitt mit
# interpolierten Feldlinien der **in die Ebene projizierten** Komponenten dargestellt.
# Bei Nédélec-Elementen erster Ordnung kann das rekonstruierte Feld zellweise
# springen. Die Abbildung ist eine stationäre Lösung, keine zeitliche Animation.

cream, ink, green, copper = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "text.color":ink,
    "axes.labelcolor":ink, "xtick.color":ink, "ytick.color":ink, "svg.fonttype":"none"})
cmap = LinearSegmentedColormap.from_list("onesimulation_magnetic",
    ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
norm = Normalize(0, float(B_magnitude.max()*1e6))
fig = plt.figure(figsize=(16,9), facecolor=cream)
fig.text(.045,.937,"ONESIMULATION   /   MAGNETOSTATIK", fontsize=11, color=green, weight="bold")
fig.text(.045,.875,"Strom formt ein Magnetfeld.", fontsize=32, weight="bold")
fig.text(.045,.828,"3D-FEM · Nédélec-Elemente · gemischte Coulomb-Eichung · tatsächliches Lösungsfeld",fontsize=12)

ax = fig.add_axes([.025,.19,.53,.60], projection="3d", facecolor=cream)
theta = np.linspace(0,2*np.pi,181)
for zi in np.linspace(30,50,10):
    ax.plot(25*np.cos(theta),25*np.sin(theta),np.full_like(theta,zi),color=copper,linewidth=1.8,alpha=.75)
for zi in (0,80):
    ax.plot(50*np.cos(theta),50*np.sin(theta),np.full_like(theta,zi),color="#8B9A8D",linewidth=.7,linestyle="--")
for theta_value in np.linspace(0,2*np.pi,8,endpoint=False):
    ax.plot([50*np.cos(theta_value)]*2,[50*np.sin(theta_value)]*2,[0,80],color="#8B9A8D",linewidth=.5,linestyle="--")
stride = max(1, config["grid"] // 15)
take = (slice(None,None,stride),slice(None,None,stride))
vectors = B[take].reshape(-1,3)
positions = points.reshape(zz.shape+(3,))[take].reshape(-1,3)*1000
magnitudes = np.linalg.norm(vectors,axis=1)
directions = vectors / np.maximum(magnitudes[:,None],np.finfo(float).tiny)
colors = cmap(norm(magnitudes*1e6))
ax.quiver(positions[:,0],positions[:,1],positions[:,2],directions[:,0],directions[:,1],directions[:,2],
    length=6.0,normalize=False,color=colors,linewidth=1.15,arrow_length_ratio=.28)
ax.set_xlim(-52,52); ax.set_ylim(-52,52); ax.set_zlim(-2,82)
ax.set_box_aspect((104,104,84),zoom=1.05)
ax.view_init(elev=20,azim=-62)
ax.set_proj_type("ortho")
ax.set_axis_off()
for artist in ax.collections:
    artist.set_clip_on(False)
fig.text(.075,.20,"10 Windungen · 1 A · homogenisierte Wicklung",fontsize=11,color=green)
fig.text(.075,.168,"Pfeile: Richtung bei gleicher Länge; Farbe: Betrag von B",fontsize=9)

plane = fig.add_axes([.615,.29,.27,.45], facecolor=cream)
plane.pcolormesh(x*1000,z*1000,B_magnitude*1e6,cmap=cmap,norm=norm,
                 shading="gouraud" if PROFILE == "HOCHAUFGELOEST" else "auto",rasterized=True)
plane.streamplot(x*1000,z*1000,B[:,:,0],B[:,:,2],color="#FAF7ED",
                 density=1.65 if PROFILE == "HOCHAUFGELOEST" else 1.15,
                 linewidth=.7,arrowsize=.8,broken_streamlines=True)
for low_x in (-30,20):
    plane.add_patch(Rectangle((low_x,30),10,20,facecolor="none",edgecolor=copper,linewidth=2))
plane.set_xlabel("x [mm]",fontsize=10); plane.set_ylabel("z [mm]",fontsize=10)
plane.set_title("FEM-Schnitt y = 0 · interpolierte Darstellung",fontsize=11,pad=12)
plane.set_aspect("equal")
plane.set_xlim(x[0]*1000,x[-1]*1000)
plane.set_ylim(z[0]*1000,z[-1]*1000)
plane.tick_params(labelsize=9,length=3)
for spine in plane.spines.values(): spine.set_color("#A8B6A7")
cax=fig.add_axes([.91,.32,.014,.39])
bar=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax)
bar.set_label("Magnetische Flussdichte |B| [µT]",fontsize=10)
bar.outline.set_visible(False); bar.ax.tick_params(length=0,labelsize=9)
fig.text(.615,.205,f"{B_check[0,2]*1e6:.1f} µT",fontsize=26,weight="bold")
fig.text(.615,.168,"berechnete axiale Flussdichte im Zentrum",fontsize=10)
fig.text(.615,.125,f"{mesh.cell_count:,} Tetraeder · {n:,} Potential- + {len(operator.gauge_dofs):,} Eichfreiheitsgrade".replace(",","."),fontsize=10)
fig.add_artist(plt.Line2D([.045,.96],[.10,.10],color="#D5D7C8",linewidth=1))
fig.text(.045,.067,"Endlicher Rechenraum mit magnetischer Isolation. Kupferringe schematisch; Feld aus der 3D-FEM.",fontsize=10)
fig.text(.045,.036,"Illustratives lineares Modell · endlicher Rechenraum · keine experimentelle Validierung",fontsize=9,color="#596B61")
png_path=output_dir/"magnetostatik_spule_3d.png"
svg_path=png_path.with_suffix(".svg")
fig.savefig(png_path,dpi=config["dpi"],facecolor=cream)
fig.savefig(svg_path,facecolor=cream)
plt.show()


# ## Ergebnisse weiterverwenden
#
# `onesim.visualization.result` zeichnet ein berechnetes Feld direkt auf seinem
# eigenen Träger. Das ist eine Zeile anstelle der von Hand geschriebenen
# Abtastung und Einfärbung weiter oben; die gestaltete Abbildung bleibt davon
# unberührt, hier steht der kurze Weg daneben.
#
# `onesim.results.storage.write_for_paraview` schreibt ausdrücklich genannte
# Felder auf genau das Netz, auf dem gerechnet wurde, als `.vtk` und `.xdmf`.
# Beide Dateien öffnet ParaView oder VisIt ohne Zwischenschritt.
#
# Grenzen dieses Exports: die Werte werden an den geschriebenen Netzknoten
# ausgewertet. Eine quadratische oder Nédélec-Lösung erscheint dort als lineare
# Ansicht auf denselben Knoten, nicht als die Lösung selbst; bei Bedarf das Netz
# verfeinern. Geschrieben wird nur, was aufgezählt ist — es gibt keine
# stillschweigende Feldauswahl. Es wird nichts neu gerechnet.

import onesim.visualization as visualization
from onesim.results.storage import write_for_paraview

PARAVIEW_FELDER = (model.magnetic_flux_density, model.magnetic_field_strength)
figure = visualization.result(
    result.result_set, model.magnetic_flux_density.field_id, title="Magnetische Flussdichte (T)",
)
assert figure.data, "Die Bibliotheksgrafik enthält keine Spur"

paraview_dateien = [
    write_for_paraview(result, mesh, PARAVIEW_FELDER,
                       output_dir / f"magnetostatik_spule_3d.{suffix}")
    for suffix in ("vtk", "xdmf")
]
for pfad in paraview_dateien:
    assert pfad.is_file() and pfad.stat().st_size > 1000, pfad
    print(f"ParaView-Datei: {pfad} ({pfad.stat().st_size / 1024:.1f} KiB)")
figure

verification = {
    "passed": True, "runtime_version":version("onesimulation"),
    "profile": PROFILE,
    "example":"3D FEM homogenized solenoid with weak Coulomb gauge",
    "tetrahedra":int(mesh.cell_count), "winding_tetrahedra":source_cell_count,
    "potential_dofs":int(n), "gauge_dofs":len(operator.gauge_dofs),
    "center_B_tesla":B_check[0].tolist(), "ampere_turns_a":ampere_turns,
    "current_doubling_relative_error":linearity_relative_error,
    "current_doubling_checked_in_this_profile": result2 is not None,
    "B_equals_mu_H_relative_error":constitutive_relative_error,
    "weak_coulomb_residual_norm":gauge_residual_norm,
    "mesh_seconds":mesh_seconds,"solve_seconds":solve_seconds,
    "doubled_current_solve_seconds":doubled_solve_seconds,
    "output_files":[str(csv_path),str(png_path),str(svg_path),str(npz_path)],
    "limitations":["Stationary linear vacuum-permeability model",
        "Homogenized winding; illustrated copper turns are schematic",
        "Finite cylinder with magnetic insulation, not unbounded free space",
        "First-order Nedelec mesh; no completed convergence or experimental validation",
        "Equal-length direction arrows; interpolated projected field lines",
        "No COMSOL accuracy or performance comparison"],
}
(output_dir/"magnetostatik_spule_3d_pruefung.json").write_text(
    json.dumps(verification,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
print(json.dumps(verification,indent=2,ensure_ascii=False,allow_nan=False))
