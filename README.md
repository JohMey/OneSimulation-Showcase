# OneSimulation-Showcase
Public PDF releases of OneSimulation – A Computational Atlas

## Free books and custom equations

The updated English **Showcase (136 pages)** and **Cookbook (433 pages)**
include a fully worked, user-defined reaction-diffusion equation with direct
Python API calls, numerical fields and a three-mesh convergence study.

[Download both free PDFs, editable LaTeX sources and the executed notebook](https://github.com/JohMey/OneSimulation-Showcase/releases/tag/custom-equations-2026-11-11).

The example uses a stationary 2D coefficient-form equation and OneSimulation's
CG/Jacobi solve. It is checked against an independently derived exact solution;
its execution scope and the broader operator-extension interfaces are explained
in both books. Running the notebook requires a compatible OneSimulation installation.

## Python API demonstration: Sod shock tube

See the computed shock, contact discontinuity and rarefaction in a
**6,400-point WENO5 + RK4 calculation**, alongside a numerical refinement study
and a space–time density plot from 401 saved states.

[Open the free high-resolution preview and application script](https://github.com/JohMey/OneSimulation-Showcase/releases/tag/sod-weno5-2026-11-10).
The script defines the model directly through the Python API and requires an
installed, activated OneSimulation Runtime.

[Explore the OneSimulation Runtime monthly subscription](https://johmey.gumroad.com/l/onesimulation-runtime-abo).

## Hydrodynamics: vertical shear instability

Explore a **512 × 512-cell stratified disk calculation over 15 orbits**,
with 1,049,600 numerical unknowns, a 4,800 × 3,000-pixel figure and a
32-state animation. The preview includes conservation checks and a
last-orbit time-step comparison; it does not claim full grid convergence
or a saturated three-dimensional turbulent disk.

[Open the computed VSI figures and numerical checks](https://github.com/JohMey/OneSimulation-Showcase/releases/tag/vsi-512-2026-11-11).

## Computed field details

Inspect original figures up to **6,400 × 4,440 pixels**: heat transport with
221,184 finite volumes, a three-dimensional cantilever with 73,728 Q1 elements,
and a magnetostatic coil calculation with 77,302 tetrahedra. Image dimensions
and solver resolution are reported separately. The accompanying numerical
checks include measured mesh differences, conservation checks and model limits;
the magnetic example does not establish mesh independence.

[Open the original field figures and numerical checks](https://github.com/JohMey/OneSimulation-Showcase/releases/tag/field-details-2026-11-11).
