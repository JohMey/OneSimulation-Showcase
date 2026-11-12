"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # 04 · Wärmetransport im Umlauf
#
# Dieses vollständige Anwendungsbeispiel berechnet ein räumliches Temperaturfeld mit **OneSimulation Runtime 0.2.1a4**. Zwei stationäre Wärmequellen liegen in einem 120 × 80 × 2 mm großen Rechengebiet. Wir vergleichen Wärmeleitung ohne Umlauf mit Wärmeleitung und einem vorgegebenen, divergenzfreien Geschwindigkeitsfeld.
#
# **Was tatsächlich gelöst wird:** Die stationäre Advektions-Diffusionsgleichung für die Temperatur, mit dem Finite-Volumen-Verfahren der installierten Runtime. Die Strömung ist eine analytische Eingabe. Es werden keine Navier–Stokes-Gleichungen, kein Auftrieb und keine experimentellen Messdaten berechnet. Das Modell ist in Dickenrichtung invariant; eine Hexaederzellschicht bildet den 2D-Fall ab.
#
# Voraussetzung: Das zu deinem System passende Paket ist installiert und regulär aktiviert. Verwende denselben CPython-3.13-Interpreter wie bei der Installation. Das Beispiel benötigt NumPy, SciPy und Matplotlib, die als Paketabhängigkeiten installiert werden. Kein GUI und kein zusätzlicher Dienst werden gestartet. Es entstehen zwei PNG-/SVG-Abbildungen und ein Prüfbericht im Ordner `ergebnisse_onesimulation`.
#
# Die folgenden Zellen sind Anwendungs- und Darstellungsbeispiele. Sie enthalten keine Implementierung des OneSimulation-Solvers. Eine erste Ausführung dauert abhängig vom Rechner ungefähr eine bis mehrere Minuten.

import json
import platform
import sys
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fvm import (
    CentralDifference, FiniteVolume, LeastSquaresGradient, LinearUpwind,
)
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatTransfer, HeatConduction, FixedTemperature, HeatFlux,
)
from onesim.simulation import Problem, Simulation


# ## Modell und Randbedingungen
#
# Wir verwenden konstante, ausdrücklich illustrative Koeffizienten: $\rho=1000\,\mathrm{kg/m^3}$, $c_p=1000\,\mathrm{J/(kg\,K)}$ und $k=2\,\mathrm{W/(m\,K)}$. Das sind keine aus einem Materialkatalog entnommenen Stoffdaten. Der Materialcontainer `IsotropicSolid` liefert hier ausschließlich die thermischen Koeffizienten; es wird keine Festkörpermechanik gerechnet.
#
# Die vier Seiten stehen auf 20 °C, Vorder- und Rückseite sind wärmeisoliert. Zwei positive Gaußquellen liefern Wärme. Das vorgegebene Geschwindigkeitsfeld ist tangential zu allen Seiten und hat maximal 0,6 mm/s. Daher verlässt keine Wärme das Gebiet durch advektiven Transport über den Rand; im stationären Zustand muss die integrierte Wärmeleitung über den Rand der Quellenleistung entsprechen.
#
# Die unabhängige Quellenintegration unten nutzt die analytischen Gaußintegrale, **keinen zweiten Wärmesolver**.

LENGTH, WIDTH, THICKNESS = 0.120, 0.080, 0.002  # m
AMBIENT_K = 293.15
DENSITY = 1000.0       # kg/m^3; illustrative constant coefficients
HEAT_CAPACITY = 1000.0 # J/(kg K)
CONDUCTIVITY = 2.0     # W/(m K)
SPEED = 0.0006         # m/s, maximum x component of prescribed velocity
SOURCES = (
    {"x": 0.036, "y": 0.024, "sigma": 0.008, "amplitude": 1.0e6},
    {"x": 0.078, "y": 0.054, "sigma": 0.006, "amplitude": 0.8e6},
)
PROFILE = "SCHNELL"  # Change to "HOCHAUFGELOEST" for a genuinely finer solve.
PROFILES = {
    "SCHNELL": {"fine_cells": (72, 48, 1), "coarse_cells": (36, 24, 1)},
    "HOCHAUFGELOEST": {"fine_cells": (144, 96, 1), "coarse_cells": (72, 48, 1)},
}
assert PROFILE in PROFILES
FINE_CELLS = PROFILES[PROFILE]["fine_cells"]
COARSE_CELLS = PROFILES[PROFILE]["coarse_cells"]
OUTPUT = Path("ergebnisse_onesimulation")
OUTPUT.mkdir(exist_ok=True)

def circulating_velocity(x, y, z):
    """Divergence-free prescribed 2D flow, tangent to all lateral walls."""
    return np.column_stack((
        SPEED * np.sin(np.pi*x/LENGTH) * np.cos(np.pi*y/WIDTH),
        -SPEED * WIDTH/LENGTH * np.cos(np.pi*x/LENGTH) * np.sin(np.pi*y/WIDTH),
        np.zeros_like(x),
    ))

def volumetric_heating(x, y, z):
    """Two smooth, positive, stationary heat sources; units W/m^3."""
    return sum(s["amplitude"] * np.exp(
        -((x-s["x"])**2 + (y-s["y"])**2)/(2*s["sigma"]**2)
    ) for s in SOURCES)

def analytic_heating_power():
    """Independent integral of the two Gaussian source functions, in W."""
    total = 0.0
    for s in SOURCES:
        sigma = s["sigma"]
        ix = sigma*np.sqrt(np.pi/2) * (
            erf((LENGTH-s["x"])/(np.sqrt(2)*sigma))
            + erf(s["x"]/(np.sqrt(2)*sigma)))
        iy = sigma*np.sqrt(np.pi/2) * (
            erf((WIDTH-s["y"])/(np.sqrt(2)*sigma))
            + erf(s["y"]/(np.sqrt(2)*sigma)))
        total += s["amplitude"]*ix*iy*THICKNESS
    return float(total)


# ## Drei echte Solverläufe und numerische Prüfungen
#
# Das Profil wählst du in der Parameterzelle über `PROFILE`:
#
# - `SCHNELL` ist die Voreinstellung: 72 × 48 × 1 = **3.456 Kontrollvolumen** je Hauptrechnung; Netzvergleich mit 36 × 24 × 1 Zellen.
# - `HOCHAUFGELOEST`: 144 × 96 × 1 = **13.824 Kontrollvolumen**, also viermal so viele tatsächlich berechnete Zellen; Netzvergleich mit 72 × 48 × 1 Zellen. Plane hierfür mehr Rechenzeit und Arbeitsspeicher ein.
#
# Beide Profile lösen dieselben physikalischen Gleichungen. Die höhere Auflösung entsteht durch feinere Solvernetze, nicht durch das Hochskalieren eines alten Bildes. Alle drei Fälle prüfen endliche Werte, die untere Temperaturgrenze und die Wärmebilanz aus den tatsächlich ausgeführten Randflüssen. Zellwerte und der bereits gebundene native Least-Squares-Gradient werden direkt aus dem FV-Ergebnis gelesen; das vermeidet unnötige räumliche Suchen für bereits bekannte Zellmittelpunkte.
#
# Der Vergleich zweier Netze ist ein Sensitivitätscheck; er ist **kein Beweis für Netzunabhängigkeit oder die Genauigkeit gegenüber einer unbekannten exakten Lösung**. Das lineare Upwind-Verfahren besitzt numerische Diffusion. Die Python-Assertions brechen bei Überschreitung der dokumentierten Grenzen ab.

def solve_heat(cells, *, circulation):
    started = time.perf_counter()
    print(f"FV solve: cells={cells}, circulation={circulation}", flush=True)
    geometry = Box(length=LENGTH, width=WIDTH, height=THICKNESS)
    # IsotropicSolid is used only as a carrier for constant thermal properties.
    # No solid mechanics equation or empirical fluid-material model is invoked.
    material = IsotropicSolid(
        density=DENSITY, specific_heat_capacity=HEAT_CAPACITY,
        thermal_conductivity=CONDUCTIVITY,
        youngs_modulus=1.0e9, poisson_ratio=0.3,
        thermal_expansion_coefficient=0.0,
    )
    common = dict(domain=geometry.volume, material=material,
                  heating=volumetric_heating)
    model = (ConvectiveHeatTransfer(velocity=circulating_velocity, **common)
             if circulation else HeatConduction(**common))
    walls = (geometry.boundaries.x_min | geometry.boundaries.x_max
             | geometry.boundaries.y_min | geometry.boundaries.y_max)
    caps = geometry.boundaries.z_min | geometry.boundaries.z_max
    problem = Problem(
        name="Wärmetransport im Umlauf" if circulation else "Wärmeleitung ohne Umlauf",
        geometry=geometry, physics=(model,),
        boundary_conditions=(
            FixedTemperature(physics=model, boundary=walls, value=AMBIENT_K),
            HeatFlux(physics=model, boundary=caps, value=0.0),
        ),
    )
    mesh = StructuredHexahedralMesher(cells=cells).generate(geometry)
    mesh_seconds = time.perf_counter()-started
    print(f"Mesh complete: {int(np.prod(cells))} cells in {mesh_seconds:.2f}s",flush=True)
    spatial_kwargs = dict(gradient=LeastSquaresGradient(), diffusion=CentralDifference())
    if circulation:
        spatial_kwargs["convection"] = LinearUpwind()
    result = Simulation(
        problem=problem, mesh=mesh,
        numerics=Numerics(spatial=FiniteVolume(**spatial_kwargs),
                          linear_solver=SparseDirect()),
    ).solve_stationary()
    stationary_seconds = time.perf_counter()-started
    print(f"Stationary solution complete at {stationary_seconds:.2f}s",flush=True)
    temperature_field = result.field(model.temperature)
    points = np.asarray(temperature_field.space.dof_coordinates)
    np.testing.assert_allclose(points, mesh.coordinates[mesh.cells].mean(axis=1),atol=1e-13,rtol=0)
    temperature = np.asarray(temperature_field.values)
    # Read native FV cell values and the solver's bound native gradient directly.
    # Coordinates are already known cell centers; no repeated point-location or
    # alternative finite-difference solver is needed. For constant k this is the
    # same Fourier field returned by result.field(model.heat_flux).evaluate(points).
    operator = result.operators[model.model_id]
    heat_flux = -CONDUCTIVITY*np.stack(operator.gradient.gradient(temperature),axis=1)
    # These are the conservative face fluxes of the executed FV equations.
    # Summing cell balances cancels internal faces exactly. Insulated cap faces
    # contribute zero; no repeated interpolation over thousands of faces is needed.
    face_power = operator.conservative_face_fluxes(result.field(model.temperature).values)
    outward_power = float(np.sum(operator.geometry.face_incidence @ face_power))
    coordinates_x = np.unique(np.round(points[:, 0], 13))
    coordinates_y = np.unique(np.round(points[:, 1], 13))
    order = np.lexsort((points[:, 0], points[:, 1]))
    shape = (len(coordinates_y), len(coordinates_x))
    assert shape == (cells[1], cells[0])
    temperature_grid = temperature[order].reshape(shape)
    flux_grid = heat_flux[order].reshape((*shape, 3))
    assert np.isfinite(temperature_grid).all() and np.isfinite(flux_grid).all()
    assert float(temperature.min()) >= AMBIENT_K - 1e-8
    power = analytic_heating_power()
    balance_relative = abs(outward_power - power) / power
    print(f"Completed in {time.perf_counter()-started:.2f}s; power={outward_power:.9g} W; relative balance={balance_relative:.3g}", flush=True)
    assert balance_relative < 2e-4, (outward_power, power, balance_relative)
    return dict(
        x=coordinates_x, y=coordinates_y, temperature=temperature_grid,
        flux=flux_grid,
        metrics={
            "circulation": circulation, "cells": list(cells),
            "control_volumes": int(np.prod(cells)),
            "solver_seconds": float(time.perf_counter()-started),
            "mesh_seconds": float(mesh_seconds),
            "stationary_solution_seconds_including_mesh": float(stationary_seconds),
            "temperature_min_K": float(temperature.min()),
            "temperature_max_K": float(temperature.max()),
            "source_power_W": power, "outward_boundary_power_W": outward_power,
            "relative_heat_balance_error": balance_relative,
            "maximum_conductive_flux_W_m2": float(np.linalg.norm(heat_flux, axis=1).max()),
        },
    )

total_started = time.perf_counter()
baseline = solve_heat(FINE_CELLS, circulation=False)
circulating = solve_heat(FINE_CELLS, circulation=True)
coarse = solve_heat(COARSE_CELLS, circulation=True)
fine_at_coarse = RegularGridInterpolator(
    (circulating["y"], circulating["x"]), circulating["temperature"]
)
coarse_x, coarse_y = np.meshgrid(coarse["x"], coarse["y"])
refinement_difference = fine_at_coarse(np.column_stack((
    coarse_y.ravel(), coarse_x.ravel()))).reshape(coarse["temperature"].shape) - coarse["temperature"]
refinement_rms = float(np.sqrt(np.mean(refinement_difference**2)))
refinement_max = float(np.max(np.abs(refinement_difference)))
temperature_scale = float(np.max(circulating["temperature"])-AMBIENT_K)
assert refinement_rms/temperature_scale < 0.06
assert refinement_max/temperature_scale < 0.15
assert np.max(np.abs(baseline["temperature"] - circulating["temperature"])) > 5.0
assert np.max(np.abs(circulating["flux"][..., 2])) < 1e-6


# ## Felder darstellen
#
# Die Temperaturbilder verwenden dieselbe Farbskala. Die Linien mit Pfeilen im zweiten Temperaturbild zeigen die **vorgegebene Geschwindigkeit**, nicht ein separat gelöstes Strömungsmodell. Das Differenzbild zeigt die Änderung der gelösten Temperatur. Die weitere Abbildung verwendet den von OneSimulation ausgegebenen Fourier-Wärmestrom $\mathbf q=-k\nabla T$.
#
# PNG ist für Präsentationen geeignet; SVG hält Beschriftungen und Konturen als Vektoren vor. Die farbigen Zellfelder sind in den SVG-Dateien gerastert, damit die Dateien handlich bleiben. Geometrie, Werte und Beschriftungen werden unverändert aus derselben Berechnung erzeugt.

CREAM, INK, TEAL, COPPER = "#faf5eb", "#193d3e", "#146f73", "#b3613d"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.edgecolor": "#b7b6a8",
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM,
    "svg.fonttype": "none", "axes.titleweight": "bold",
})
thermal_cmap = LinearSegmentedColormap.from_list(
    "cream_copper", ["#eef1de", "#eed9a6", "#e6a065", "#bd5b37", "#762e2e"])
difference_cmap = LinearSegmentedColormap.from_list(
    "teal_cream_copper", ["#104f54", "#5e9996", CREAM, "#d2996e", "#a4472d"])
temperature_max_C = max(float(baseline["temperature"].max()),
                        float(circulating["temperature"].max())) - 273.15
temperature_limit = 5*np.ceil(temperature_max_C/5)
norm = Normalize(20, temperature_limit)
xmm, ymm = circulating["x"]*1000, circulating["y"]*1000
XX, YY = np.meshgrid(xmm, ymm)
VV = circulating_velocity(XX.ravel()/1000, YY.ravel()/1000,
                          np.zeros(XX.size)).reshape((*XX.shape, 3))
fig = plt.figure(figsize=(16, 11.1))
fig.text(.06, .958, "ONESIMULATION  /  BERECHNETES ANWENDUNGSBEISPIEL", color=TEAL,
         fontsize=10.3, weight="bold", ha="left")
fig.text(.06, .902, "Wie Umlauf Wärme verteilt", fontsize=31, weight="bold")
fig.text(.06, .864, "Zwei Wärmequellen. Dieselben Randbedingungen. Ein verändertes Transportfeld.",
         fontsize=12.4, color="#586b68")

def style_field(ax, *, title, subtitle):
    ax.set_aspect("equal")
    ax.set(xlim=(0, LENGTH*1000), ylim=(0, WIDTH*1000), xlabel="x [mm]", ylabel="y [mm]")
    ax.set_xticks([0, 30, 60, 90, 120]); ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(length=3, labelsize=9)
    ax.set_title(title, loc="left", fontsize=14, pad=27)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=9.3, color="#657975")
    for index, source in enumerate(SOURCES):
        xy=(1000*source["x"], 1000*source["y"])
        ax.add_patch(Circle(xy, radius=2.1, fill=False, linewidth=1.15,
                            edgecolor="#fffaf1", zorder=9))
        ax.text(xy[0]+4, xy[1]+2.7, chr(65+index), color="#fffaf1", weight="bold",
                fontsize=10, zorder=10,
                bbox=dict(boxstyle="round,pad=.15", facecolor=INK, alpha=.7, edgecolor="none"))

axes = [fig.add_axes([.06, .505, .394, .286]), fig.add_axes([.535, .505, .394, .286])]
for ax, case, label, sub in zip(
    axes, [baseline, circulating], ["01  Wärmeleitung", "02  Wärmeleitung + Advektion"],
    ["Ohne Umlauf · gleiche Wärmequellen", "Vorgegebener Umlauf · Pfeile zeigen seine Richtung"],
):
    field_C = case["temperature"]-273.15
    image=ax.pcolormesh(xmm, ymm, field_C, shading="nearest", cmap=thermal_cmap, norm=norm, rasterized=True)
    lines=ax.contour(xmm, ymm, field_C, levels=np.arange(30, temperature_limit, 10),
                    colors="#fff9ee", linewidths=.6, alpha=.75)
    ax.clabel(lines, inline=True, fontsize=7.5, fmt="%.0f°")
    style_field(ax, title=label, subtitle=sub)
stagnant = np.linalg.norm(VV[..., :2], axis=2) < 0.00004
axes[1].streamplot(xmm, ymm, np.ma.masked_where(stagnant,VV[...,0]),
                   np.ma.masked_where(stagnant,VV[...,1]), density=.85, color=INK,
                   linewidth=.7, arrowsize=.8, minlength=.2, maxlength=5, zorder=7)
colorbar=fig.colorbar(image, cax=fig.add_axes([.275, .446, .45, .014]), orientation="horizontal")
colorbar.set_label("Temperatur [°C] · gemeinsame Farbskala", size=10, labelpad=6)
colorbar.ax.tick_params(labelsize=8.5, length=3)

delta=circulating["temperature"]-baseline["temperature"]
delta_limit=5*np.ceil(np.max(np.abs(delta))/5)
ax_delta=fig.add_axes([.06, .17, .285, .195])
delta_image=ax_delta.pcolormesh(xmm, ymm, delta, shading="nearest", cmap=difference_cmap,
                               norm=TwoSlopeNorm(vmin=-delta_limit, vcenter=0, vmax=delta_limit), rasterized=True)
ax_delta.set_aspect("equal")
ax_delta.set(xlim=(0,120),ylim=(0,80),xlabel="x [mm]",ylabel="y [mm]")
ax_delta.set_xticks([0,30,60,90,120]); ax_delta.set_yticks([0,40,80])
ax_delta.tick_params(labelsize=8.5,length=3)
ax_delta.set_title("Wo sich die Temperatur ändert",loc="left",fontsize=11.5,pad=8)
cb=fig.colorbar(delta_image,cax=fig.add_axes([.06,.103,.285,.011]),orientation="horizontal")
cb.set_label("Mit Umlauf minus ohne Umlauf [K]",fontsize=9,labelpad=4)
cb.ax.tick_params(labelsize=8,length=2)

peak_drop=float(baseline["temperature"].max()-circulating["temperature"].max())
fig.text(.405,.325,f"{peak_drop:.1f} K".replace(".",","),fontsize=32,weight="bold",color=TEAL)
fig.text(.405,.294,"niedrigere Spitzentemperatur",fontsize=11.5,weight="bold")
fig.text(.405,.270,"in dieser stationären Modellrechnung",fontsize=10,color="#657975")
fig.text(.405,.222,"120 × 80 mm",fontsize=20,weight="bold")
fig.text(.405,.195,"2 mm Dicke · in z invariant",fontsize=10,color="#657975")
fig.text(.685,.325,f"{int(np.prod(FINE_CELLS)):,}".replace(",", "."),fontsize=30,weight="bold",color=COPPER)
fig.text(.685,.294,"Finite-Volumen-Zellen",fontsize=11.5,weight="bold")
fig.text(.685,.270,"OneSimulation Runtime 0.2.1a4",fontsize=10,color="#657975")
fig.text(.685,.222,f"{analytic_heating_power():.3f} W".replace(".",","),fontsize=20,weight="bold")
fig.text(.685,.195,"zugeführte Wärmeleistung",fontsize=10,color="#657975")
fig.text(.405,.137,"Vier Seiten: 20 °C · Deckflächen: wärmeisoliert\n"
         "Umlauf: maximal 0,6 mm/s · konstante Stoffwerte",fontsize=10,color="#586b68",linespacing=1.65)
fig.text(.06,.036,"Echte Solver-Ausgabe, kein Messdatensatz. Vorgegebene Strömung; keine Navier–Stokes- oder Auftriebsrechnung.\n"
         "Darstellung der Zellwerte mit Konturlinien. Netzvergleich und Wärmebilanz sind im ausführbaren Beispiel dokumentiert.",
         fontsize=9.1,color="#657975",linespacing=1.5)
png_path=OUTPUT/"04_waermetransport_umlauf.png"
svg_path=OUTPUT/"04_waermetransport_umlauf.svg"
fig.savefig(png_path,dpi=250)
fig.savefig(svg_path)
plt.close(fig)

# A second, genuinely derived physical field: the native Fourier heat flux.
flux_kW = circulating["flux"][..., :2]/1000
flux_magnitude = np.linalg.norm(flux_kW, axis=2)
flux_cmap = LinearSegmentedColormap.from_list(
    "cream_teal", ["#f0efda", "#c7dac3", "#83b9ae", "#31858b", "#164c58"])
fig_flux = plt.figure(figsize=(12, 8.7))
fig_flux.text(.075,.947,"ONESIMULATION  /  ABGELEITETES PHYSIKFELD",fontsize=10,weight="bold",color=TEAL)
fig_flux.text(.075,.893,"Wohin die Wärme fließt",fontsize=27,weight="bold")
fig_flux.text(.075,.852,"Fourier-Wärmestrom aus dem gelösten Temperaturfeld mit vorgegebenem Umlauf",fontsize=10.7,color="#586b68")
ax_flux=fig_flux.add_axes([.075,.20,.72,.57])
flux_image=ax_flux.pcolormesh(xmm,ymm,flux_magnitude,shading="nearest",cmap=flux_cmap,rasterized=True)
ax_flux.streamplot(xmm,ymm,flux_kW[...,0],flux_kW[...,1],density=1.15,color="#fffaf0",
                   linewidth=.9,arrowsize=.9,minlength=.12)
temperature_contours=ax_flux.contour(xmm,ymm,circulating["temperature"]-273.15,
                                    levels=[30,40,50,60],colors=COPPER,linewidths=.8,alpha=.85)
ax_flux.clabel(temperature_contours,fmt="%.0f °C",fontsize=8.3,inline=True)
ax_flux.set_aspect("equal")
ax_flux.set(xlim=(0,120),ylim=(0,80),xlabel="x [mm]",ylabel="y [mm]")
ax_flux.set_xticks([0,30,60,90,120]); ax_flux.set_yticks([0,20,40,60,80])
for index, source in enumerate(SOURCES):
    ax_flux.add_patch(Circle((1000*source["x"],1000*source["y"]),2.2,
                            facecolor=CREAM,edgecolor=COPPER,linewidth=1.3,zorder=8))
    ax_flux.text(1000*source["x"],1000*source["y"],chr(65+index),
                 ha="center",va="center",fontsize=8,weight="bold",zorder=9)
flux_bar=fig_flux.colorbar(flux_image,cax=fig_flux.add_axes([.837,.27,.021,.43]))
flux_bar.set_label("Betrag des Wärmestroms [kW/m²]",size=10,labelpad=10)
fig_flux.text(.075,.10,"Farbe: Betrag des berechneten Wärmestroms   ·   Weiße Pfeile: Richtung von −k∇T\n"
              "Kupferfarbene Linien: Temperatur   ·   A / B: Zentren der Wärmequellen",fontsize=10,linespacing=1.8,color="#586b68")
fig_flux.text(.075,.045,f"Stationäre Finite-Volumen-Rechnung · {FINE_CELLS[0]} × {FINE_CELLS[1]} × 1 Zellen · OneSimulation Runtime 0.2.1a4\n"
              "Illustratives Modell mit konstanten Stoffwerten; keine Messdaten und keine experimentelle Validierung.",
              fontsize=8.8,linespacing=1.6,color="#657975")
flux_png_path=OUTPUT/"04_fourier_waermestrom.png"
flux_svg_path=OUTPUT/"04_fourier_waermestrom.svg"
fig_flux.savefig(flux_png_path,dpi=320)
fig_flux.savefig(flux_svg_path)
plt.close(fig_flux)


# ## Maschinenlesbare Verifikation
#
# `verification` enthält Laufzeiten, Netzgrößen, Temperaturen, Quellen- und Randflussleistung sowie den Netzvergleich. Der Bericht dokumentiert diese Rechnung und ihre Grenzen; er behauptet keine wissenschaftliche oder experimentelle Validierung.

verification = {
    "passed": True,
    "example": "stationary_2d_invariant_convective_heat",
    "profile": PROFILE,
    "runtime_version": version("onesimulation"),
    "host": {"system": platform.system(), "machine": platform.machine(),
             "python": platform.python_version()},
    "equation": "rho*cp*v.grad(T)-div(k*grad(T))=Q",
    "input_velocity": "prescribed analytic divergence-free vortex; momentum not solved",
    "length_m": LENGTH, "width_m": WIDTH, "thickness_m": THICKNESS,
    "thermal_properties": {"density_kg_m3": DENSITY,
        "heat_capacity_J_kgK": HEAT_CAPACITY, "conductivity_W_mK": CONDUCTIVITY},
    "boundary_conditions": "four lateral walls 293.15 K; top and bottom insulated",
    "baseline": baseline["metrics"], "circulating": circulating["metrics"],
    "coarse_circulating": coarse["metrics"],
    "refinement_comparison": {"coarse_to_fine_axis_factor": FINE_CELLS[0]/COARSE_CELLS[0],
        "rms_difference_K": refinement_rms, "maximum_difference_K": refinement_max,
        "rms_relative_to_temperature_rise": refinement_rms/temperature_scale,
        "interpretation": "two-grid sensitivity check; not exact-solution error or mesh-independence proof"},
    "peak_temperature_reduction_K": peak_drop,
    "total_seconds_including_plot": float(time.perf_counter()-total_started),
    "artifacts": [str(png_path),str(svg_path),str(flux_png_path),str(flux_svg_path)],
    "limitations": ["illustrative constant coefficients, no material catalogue",
        "prescribed laminar circulation, no coupled momentum or buoyancy solution",
        "stationary solution, not a measured or experimentally validated process",
        "one cell through insulated thickness; z-invariant model",
        "linear-upwind spatial discretization can introduce numerical diffusion"],
}
assert version("onesimulation") == "0.2.1a4"
assert all(Path(path).stat().st_size > 1000 for path in verification["artifacts"])
(OUTPUT/"04_verifikation.json").write_text(json.dumps(verification,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(verification,indent=2,ensure_ascii=False))
