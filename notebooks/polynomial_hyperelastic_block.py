"""OneSimulation example: a user-defined polynomial hyperelastic rubber in three load cases.
Requires the installed, activated OneSimulation runtime.

A rubber block is read from a STEP file and stretched in uniaxial tension, equibiaxial tension
and pure shear. The material is defined by its strain energy as a polynomial in the invariants,
once with two terms (Mooney-Rivlin) and once with five; the 3D finite-strain solution is compared
with the closed-form nominal stresses of the incompressible material in all six combinations.
"""

# # Polynomial hyperelastic rubber in three load cases
#
# A hyperelastic material is defined completely by its strain energy density. For rubber the
# generalised Mooney-Rivlin form, a polynomial in the isochoric invariants $\bar I_1,\bar I_2$, is
# the most common choice:
#
# $$W=\sum_{i+j\ge1}c_{ij}\,(\bar I_1-3)^i(\bar I_2-3)^j+\frac K2(J-1)^2 .$$
#
# In OneSimulation such a material is declared by its list of terms $(i,j,c_{ij})$; stress and
# tangent follow from the energy. Here two models are compared:
#
# | model | $c_{10}$ | $c_{01}$ | $c_{20}$ | $c_{11}$ | $c_{02}$ |
# |---|---|---|---|---|---|
# | Mooney-Rivlin | 0.16 MPa | 0.01 MPa | | | |
# | five-term polynomial | 0.16 MPa | 0.01 MPa | 2 kPa | 0.5 kPa | 0.1 kPa |
#
# with a bulk modulus $K=10^4$ times the shear modulus, so the rubber is nearly incompressible.
#
# **Setup.** A cube of 20 mm side from the CAD file `polynomial_hyperelastic_block.step`, written
# by `polynomial_hyperelastic_block_geometry.py` with the CAD objects of OneSimulation. The faces
# `x_min`, `y_min`, `z_min` are smooth supports (rollers); the load cases move the opposite faces:
#
# | load case | `x_max` | `y_max` | `z_max` | stretches |
# |---|---|---|---|---|
# | uniaxial tension | $u_x=(\lambda-1)L$ | free | free | 1 to 3 |
# | equibiaxial tension | $u_x=(\lambda-1)L$ | $u_y=(\lambda-1)L$ | free | 1 to 2 |
# | pure shear | $u_x=(\lambda-1)L$ | $u_y=0$ | free | 1 to 2.5 |
#
# **Reference.** For the incompressible material with $W_1=\partial W/\partial\bar I_1$,
# $W_2=\partial W/\partial\bar I_2$ the nominal stress in the loaded direction is
# (e.g. R.W. Ogden, *Non-linear Elastic Deformations*, Dover 1997)
#
# $$P_\mathrm{uni}=2\Big(\lambda-\frac1{\lambda^2}\Big)\Big(W_1+\frac{W_2}{\lambda}\Big),\quad
#   P_\mathrm{bi}=2\Big(\lambda-\frac1{\lambda^5}\Big)\big(W_1+\lambda^2W_2\big),\quad
#   P_\mathrm{shear}=2\Big(\lambda-\frac1{\lambda^3}\Big)\big(W_1+W_2\big).$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials.solids import IsochoricInvariantPolynomial, QuadraticVolumeLaw
from onesim.meshing import CadHexahedralMesher
from onesim.numerics.fem import TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import DampedNewton
from onesim.physics.solid_mechanics import FiniteStrainSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the block with its named faces. With smooth supports every load case is a
# homogeneous deformation, which a coarse mesh of trilinear hexahedra represents exactly.

block = load("polynomial_hyperelastic_block.step")
faces = block.boundaries
side = block.size[0]
mesh = CadHexahedralMesher(cells=(3, 3, 3)).generate(block)

view = plots.Camera(elevation=25, azimuth=-55)
fig = plots.figure(1, 2, size=(12, 5.5))
plots.geometry(
    block,
    selection={
        "rollers": faces.x_min | faces.y_min | faces.z_min,
        "pulled": faces.x_max | faces.y_max,
    },
    labels=False,
    camera=view,
    title="Rubber block",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} hexahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The two materials are lists of polynomial terms; each of the six runs ramps its stretch in
# twelve steps. The nominal stress is the reaction on `x_max` over the original face area.

models = {
    "Mooney-Rivlin": ((1, 0, 0.16e6), (0, 1, 0.01e6)),
    "five-term polynomial": (
        (1, 0, 0.16e6),
        (0, 1, 0.01e6),
        (2, 0, 2e3),
        (1, 1, 0.5e3),
        (0, 2, 0.1e3),
    ),
}
cases = {
    "uniaxial tension": (3.0, ()),
    "equibiaxial tension": (2.0, ((faces.y_max, "y", lambda t: t),)),
    "pure shear": (2.5, ((faces.y_max, "y", 0.0),)),
}
supports = (
    (faces.x_min, "x", 0.0),
    (faces.y_min, "y", 0.0),
    (faces.z_min, "z", 0.0),
    (faces.x_max, "x", lambda t: t),
)
stretch = {case: np.linspace(1.0, top, 13) for case, (top, _) in cases.items()}
results = {
    (model, case): FiniteStrainSolid(
        shear_modulus=2 * (terms[0][2] + terms[1][2]),
        bulk_modulus=2e4 * (terms[0][2] + terms[1][2]),
    ).solve(
        mesh=mesh,
        times=(stretch[case] - 1) * side,
        quadrature_order=3,
        displacements=supports + extra,
        nonlinear=DampedNewton(
            linear_solver=SparseDirect(),
            absolute_tolerance=1e-11,
            relative_tolerance=1e-10,
            line_search_merit="newton_correction",
        ),
        space_factory=TaylorHood(),
        elastic_response=IsochoricInvariantPolynomial(terms),
        volume_law=QuadraticVolumeLaw(2e4 * (terms[0][2] + terms[1][2])),
    )
    for model, terms in models.items()
    for case, (top, extra) in cases.items()
}
nominal = {
    key: result.reactions[
        :, np.isclose(result.displacement_space.dof_coordinates[:, 0], side), 0
    ].sum(axis=1)
    / side**2
    for key, result in results.items()
}

# ## Reference
#
# The closed-form nominal stress from the derivatives of the polynomial, with the invariants of
# each load case.


def reference(terms, case, l):
    """Nominal stress of the incompressible polynomial material."""
    i1, i2 = {
        "uniaxial tension": (l**2 + 2 / l, 2 * l + l**-2),
        "equibiaxial tension": (2 * l**2 + l**-4, l**4 + 2 * l**-2),
        "pure shear": (l**2 + 1 + l**-2, l**2 + 1 + l**-2),
    }[case]
    w1 = sum(c * i * (i1 - 3) ** max(i - 1, 0) * (i2 - 3) ** j for i, j, c in terms)
    w2 = sum(c * j * (i1 - 3) ** i * (i2 - 3) ** max(j - 1, 0) for i, j, c in terms)
    return {
        "uniaxial tension": 2 * (l - l**-2) * (w1 + w2 / l),
        "equibiaxial tension": 2 * (l - l**-5) * (w1 + l**2 * w2),
        "pure shear": 2 * (l - l**-3) * (w1 + w2),
    }[case]


# ## Comparison with the reference
#
# The largest deviation of the computed nominal stress from the closed form, relative to the
# stress at the largest stretch, and the volume change, which shows how close to incompressible
# the rubber stays.

print(
    f"{'model':22s} {'load case':20s} {'P at the end [MPa]':>19s} {'closed form':>12s} {'largest deviation':>18s} {'largest |J - 1|':>16s}"
)
for (model, case), values in nominal.items():
    exact = reference(models[model], case, stretch[case])
    print(
        f"{model:22s} {case:20s} {values[-1] / 1e6:19.4f} {exact[-1] / 1e6:12.4f} {100 * np.abs(values - exact).max() / exact[-1]:17.3f} %"
        f" {np.abs(results[model, case].jacobian - 1).max():16.1e}"
    )

# ## Figure
#
# Nominal stress over stretch for the three load cases and both models: simulation as markers,
# closed form dashed.

fine = {case: np.linspace(1.0, values[-1], 200) for case, values in stretch.items()}
fig = plots.figure(1, 3, size=(18, 5.5))
for k, case in enumerate(cases):
    plots.line(
        points={model: (stretch[case], nominal[model, case] / 1e6) for model in models},
        reference={
            model: (fine[case], reference(terms, case, fine[case]) / 1e6)
            for model, terms in models.items()
        },
        xlabel="stretch λ",
        ylabel="nominal stress P [MPa]",
        title=case.capitalize(),
        legend="upper left",
        ax=fig[k],
    )
plots.show()

# ## Going further
#
# Fit the coefficients to measurements with the study service of OneSimulation, as in the
# Mooney-Rivlin example; check the Drucker stability of a fitted polynomial by running large
# stretches in all three load cases. Replace the block by a real part from a STEP file: the
# material declaration stays the same.
