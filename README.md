# OneSimulation

**A whole simulation lab in one Python package:** CAD, meshing, 16 physics modules, FEM, FVM, FDM and DG, implicit and explicit solvers, multiphysics coupling, plotting and storage behind one API.

**[Documentation](https://johmey.github.io/OneSimulation-Showcase/)** · **[Quickstart](https://johmey.github.io/OneSimulation-Showcase/quickstart.html)** · **[Examples](https://johmey.github.io/OneSimulation-Showcase/examples/index.html)** · **[API reference](https://johmey.github.io/OneSimulation-Showcase/reference/a-physics-api-reference/index.html)** · **[Plans](https://johmey.gumroad.com/l/onesimulation-runtime-abo)** · **[Free for students and courses](https://johmey.gumroad.com/l/onesimulation-education)**

<p>
<a href="https://johmey.github.io/OneSimulation-Showcase/examples/magnetostatics.html"><img src="https://johmey.github.io/OneSimulation-Showcase/thumbs/hero-magnetostatics.jpg" width="24%" alt="Computed magnetic field of a coil"></a>
<a href="https://johmey.github.io/OneSimulation-Showcase/examples/heat_transfer.html"><img src="https://johmey.github.io/OneSimulation-Showcase/thumbs/hero-heat_transfer.jpg" width="24%" alt="Computed heat transport with circulation"></a>
<a href="https://johmey.github.io/OneSimulation-Showcase/examples/magnetohydrodynamics.html"><img src="https://johmey.github.io/OneSimulation-Showcase/thumbs/hero-magnetohydrodynamics.jpg" width="24%" alt="Computed resistive MHD field in 3D"></a>
<a href="https://johmey.github.io/OneSimulation-Showcase/examples/multiphysics.html"><img src="https://johmey.github.io/OneSimulation-Showcase/thumbs/hero-multiphysics.jpg" width="24%" alt="Computed Joule heating around a hole in a copper plate"></a>
</p>

Every picture is a real calculation. The [example gallery](https://johmey.github.io/OneSimulation-Showcase/examples/index.html) has one page for each of the sixteen physics modules, with the model, its resolution and its limits.

## One model, five discretizations

The physics is declared once; the numerical method is an argument.

```python
for spatial in (FiniteElement(element=Lagrange(order=1)),
                FiniteElement(element=Lagrange(order=2)),
                FiniteVolume(gradient=LeastSquaresGradient(), diffusion=CentralDifference()),
                FiniteDifference(gradient=FDGradient(), diffusion=CentralLaplacian()),
                FiniteDifference(gradient=PolynomialDifference(accuracy_order=4),
                                 diffusion=PolynomialLaplacian(accuracy_order=4))):
    result = Simulation(problem=problem, mesh=mesh,
                        numerics=Numerics(spatial=spatial, linear_solver=SparseDirect())
                        ).solve_stationary()
```

| Method | RMS error against the exact solution | Time |
| --- | --- | --- |
| FEM, linear elements | 2.8 × 10⁻³ | 14 s |
| FEM, quadratic elements | 6.3 × 10⁻⁷ | 86 s |
| FVM | 5.7 × 10⁻⁴ | 6 s |
| FDM, 2nd order | 1.1 × 10⁻³ | 6 s |
| FDM, 4th order | 1.7 × 10⁻⁸ | 7 s |

Same 3D Poisson problem, same mesh of 16 × 16 × 16 hexahedra. [The full example with the model code](https://johmey.github.io/OneSimulation-Showcase/examples/method-swap.html).

## Install

OneSimulation is delivered as a binary wheel for CPython 3.13 on macOS (Apple Silicon), Linux (x86_64) and Windows (AMD64).

```bash
python3.13 -m venv .venv-onesimulation
.venv-onesimulation/bin/python -m pip install './onesimulation-0.2.1a2-cp313-cp313-macosx_11_0_arm64.whl[meshing,compute,visualization]'
.venv-onesimulation/bin/onesimulation-license activate-gumroad
```

The [Quickstart](https://johmey.github.io/OneSimulation-Showcase/quickstart.html) has the commands for all three platforms and a first 3D model.

## Getting it

- **Free for students and course instructors**, after a short check: [apply for Education access](https://johmey.gumroad.com/l/onesimulation-education).
- **7-day free trial** on every paid plan, from €19 a month: [plans](https://johmey.gumroad.com/l/onesimulation-runtime-abo).
- **Rather have it built for you?** [Order a simulation notebook](https://johmey.gumroad.com/l/onesimulation-notebook-on-order) at a fixed price.

OneSimulation is an alpha and a closed binary package. The results shown are numerically verified against references ([verification](https://johmey.github.io/OneSimulation-Showcase/verification.html)); they are not experimentally validated.

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
