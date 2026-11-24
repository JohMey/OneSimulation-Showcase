# OneSimulation

**API-first · AI-first · fully flexible**

Every part of OneSimulation is a Python API: geometry, mesh, physics, numerics and results. The included MCP server opens the same studies to Codex and Claude Code, so your assistant builds, runs and reads simulations on your machine. Nothing is fixed: 16 physics modules, finite elements, finite volumes, finite differences and discontinuous Galerkin, implicit and explicit time integrators, direct and Krylov solvers, multiphysics couplings and equations of your own.

<p>
<a href="https://onesimulation.onrender.com/blog/halbach_rotor_field/"><img src="https://onesimulation.onrender.com/assets/hero-3d/halbach_rotor_field.webp" width="24%" alt="Static field of a Halbach rotor, computed with OneSimulation"></a>
<a href="https://onesimulation.onrender.com/blog/turbine_stator_blade_thermal_stress/"><img src="https://onesimulation.onrender.com/assets/hero-3d/turbine_stator_blade_thermal_stress.webp" width="24%" alt="Thermal stress in a cooled turbine stator blade, computed with OneSimulation"></a>
<a href="https://onesimulation.onrender.com/blog/tesla_valve_diodicity/"><img src="https://onesimulation.onrender.com/assets/hero-3d/tesla_valve_diodicity.webp" width="24%" alt="Flow through a Tesla valve, computed with OneSimulation"></a>
<a href="https://onesimulation.onrender.com/blog/double_slit_diffraction/"><img src="https://onesimulation.onrender.com/assets/hero-3d/double_slit_diffraction.webp" width="24%" alt="Double-slit diffraction of sound, computed with OneSimulation"></a>
</p>

| | |
| --- | --- |
| **[Website](https://onesimulation.onrender.com/)** | What OneSimulation does: 377 worked examples with reference checks, [one model through every method](https://onesimulation.onrender.com/tutorials/sod-shock-tube/), tutorials and courses. |
| **[Documentation](https://johmey.github.io/OneSimulation-Showcase/)** | How to use it: [quickstart](https://johmey.github.io/OneSimulation-Showcase/quickstart.html), MCP setup for Codex and Claude Code, guide, [API reference](https://johmey.github.io/OneSimulation-Showcase/reference/a-physics-api-reference/index.html), [licence and terms](https://johmey.github.io/OneSimulation-Showcase/plans.html). |
| **[Plans and trial](https://johmey.gumroad.com/l/onesimulation-runtime-abo?section=subscribe)** | 7-day trial, then from €19 per month on Gumroad. [Start the Focus trial](https://johmey.gumroad.com/l/onesimulation-runtime-abo?variant=Focus+%E2%80%93+Structures+%26+Mechanics&monthly=true&wanted=true&utm_source=github&utm_medium=repository&utm_campaign=engineering_pilot&utm_content=github_readme). Free for students and teachers: [apply for the Education plan](https://onesimulation-contact.yummy-bee-0467.chatgpt.site/). |

Runtime Alpha 0.2.1a8 runs on macOS (Apple Silicon), Ubuntu 22.04 (x86-64) and Windows x64 with CPython 3.13. OneSimulation is
an alpha and a closed binary package. Its results are numerically verified against references;
they are not experimentally validated.

## Use it from Claude Code or Codex

This repository is also a plugin marketplace. The plugin adds a skill that connects your
assistant to the MCP server of your runtime download and guides it through a study: building
the model, running it and checking the result against a reference.

```sh
claude plugin marketplace add JohMey/OneSimulation-Showcase
claude plugin install onesimulation@onesimulation
```

```sh
codex plugin marketplace add JohMey/OneSimulation-Showcase
codex plugin add onesimulation@onesimulation
```

## Downloads in this repository

- [Sod shock tube, 6,400-point WENO5 + RK4 preview and script](https://github.com/JohMey/OneSimulation-Showcase/releases/tag/sod-weno5-2026-11-10)
- [Vertical shear instability, 512 × 512 cells over 15 orbits](https://github.com/JohMey/OneSimulation-Showcase/releases/tag/vsi-512-2026-11-11)
- [Original field figures up to 6,400 × 4,440 pixels with numerical checks](https://github.com/JohMey/OneSimulation-Showcase/releases/tag/field-details-2026-11-11)

Questions before you subscribe: [contact form](https://onesimulation-contact.yummy-bee-0467.chatgpt.site/).
