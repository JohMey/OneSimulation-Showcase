---
name: simulate
description: Set up, run and check physics simulations with OneSimulation, a Python library with a local MCP server. Use when the user asks to simulate or compute heat transfer, solid mechanics, fluid flow, electromagnetics, acoustics, chemistry, plasma or coupled physics, wants to compare numerical methods (FEM, FVM, FDM, DG), or mentions OneSimulation.
---

# Simulate with OneSimulation

Every part of OneSimulation is a Python API: geometry, mesh, physics, numerics and results. The included MCP server opens the same studies to Codex and Claude Code, so your assistant builds, runs and reads simulations on your machine. Nothing is fixed: 16 physics modules, finite elements, finite volumes, finite differences and discontinuous Galerkin, implicit and explicit time integrators, direct and Krylov solvers, multiphysics couplings and equations of your own.

There are two ways to work, and they combine: the MCP tools build and run studies directly, and
the Python API covers everything else.

## 1. Find out what is installed

1. If the tools of an MCP server named `onesimulation-runtime` are available (for example
   `list_projects` and `create_simulation`), continue with section 2.
2. Otherwise check for the library in the user's Python environment: `python -c "import onesim"`.
   If it imports, register the MCP server (below) or continue with section 3.
3. If neither is there, the user needs the OneSimulation Runtime. Say so and give these links:
   - [Plans and seven-day trial](https://johmey.gumroad.com/l/onesimulation-runtime-abo?section=subscribe), from €19 per month
   - [Free Education plan for students and teachers](https://onesimulation-contact.yummy-bee-0467.chatgpt.site/)
   - [Download, install and activate](https://johmey.github.io/OneSimulation-Showcase/quickstart.html)

   OneSimulation 1.0 runs on CPython 3.13 for macOS (Apple Silicon), Ubuntu 22.04 (x86-64), Windows x64. It is delivered as a wheel in
   the customer download, package build 0.2.1a8. OneSimulation is not published on PyPI or any other package
   index: never install a package of that name from an index.

### Register the MCP server

The server is the executable `onesimulation-mcp` in the environment where the wheel was installed
with the `agent` extra. Use absolute paths. The workspace is the folder that holds the user's
OneSimulation projects; a new empty folder works.

Claude Code, run in the folder where Claude Code is started:

```sh
claude mcp add onesimulation-runtime --scope local --transport stdio \
  --env 'ONESIMULATION_WORKSPACE=/absolute/path/to/workspace' \
  -- "/absolute/path/to/.venv-onesimulation/bin/onesimulation-mcp"
```

Codex:

```sh
codex mcp add onesimulation-runtime \
  --env 'ONESIMULATION_WORKSPACE=/absolute/path/to/workspace' \
  -- "/absolute/path/to/.venv-onesimulation/bin/onesimulation-mcp"
```

The tools appear in a new session.

## 2. Work through the MCP tools

| Tool | Use |
| --- | --- |
| `list_projects` | Projects in the workspace |
| `list_project_studies` | Saved studies of one project |
| `import_project_bundle` | Import a local `.onesim` project folder, given by absolute path |
| `create_simulation` | Build and save a new model with its stationary study |
| `get_simulation_definition` | Read back geometry, mesh, material and boundary data of a created model |
| `start_project_simulation` | Start a saved study; returns the run ID |
| `get_simulation_status` | Status, progress and diagnostics of a run |
| `get_simulation_results` | Result fields of a finished run and the path of its HDF5 file |
| `validate_project` | Check a project document |
| `solve_heat_1d` | Steady heat conduction through a wall, solved at once |

`create_simulation` builds steady heat conduction on a 2D rectangular plate or a 3D box (fixed
temperatures in kelvins, the remaining sides insulated) and linear elasticity on a 3D box (fixed
faces, uniform tractions in pascals). All inputs are in SI units. Keep the returned `project_id`
and `study_id`, start the study, poll its status and read the fields when it has finished.

A saved project of any physics runs the same way: import it, list its studies, start one. For a
new model outside the two cases above, write Python (section 3) and run it in the user's
environment.

## 3. Work with the Python API

1. Find the closest worked example. [https://onesimulation.onrender.com/llms.txt](https://onesimulation.onrender.com/llms.txt) lists all
   377 examples by subject with a one-line summary and a link. Each example
   page states its model, its reference check and offers the notebook and the script for download.
2. Start from that script. Change geometry, material, loads and mesh for the user's case.
3. Take class and function names from the example or from the
   [API reference](https://johmey.github.io/OneSimulation-Showcase/reference/a-physics-api-reference/index.html). Do not guess them.
4. The numerical method is a choice, not a given: finite elements, finite volumes, finite
   differences and discontinuous Galerkin, implicit and explicit time integrators, direct and
   Krylov solvers. [One shock tube through every method](https://onesimulation.onrender.com/tutorials/sod-shock-tube/) shows which combinations
   suit a problem with shocks.

More: [tutorials](https://onesimulation.onrender.com/tutorials/), [documentation](https://johmey.github.io/OneSimulation-Showcase/).

## 4. Check the result before reporting it

- Compare a central quantity with a reference: an analytical solution, a balance (heat, mass,
  force) or the same run on a coarser mesh.
- Report the number of mesh nodes or elements and the deviation from the reference.
- State units with every number.
- OneSimulation's results are numerically verified against references. Do not call them
  experimentally validated.
