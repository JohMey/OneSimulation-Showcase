"""OneSimulation example: a ship hull lofted through its stations.
Requires the installed, activated OneSimulation runtime.

The Wigley hull, the standard mathematical hull of ship hydrodynamics, is built with the CAD
objects of OneSimulation: every station is an exact profile with parabolic sides, and a loft runs
a smooth surface through eleven stations. The underwater body is cut off at the waterline and
meshed.

Displacement volume, waterplane area and centre of buoyancy are compared with the closed-form
integrals of the hull equation.
"""

# # Ship hull geometry
#
# A hull is drawn as a set of cross-sections, the stations, along its length; the shell runs
# smoothly through them. `Loft(sections=[(profile, position), ...])` does exactly that: with more
# than two sections the surface is a smooth interpolation through all of them.
#
# The Wigley hull of length $L$, beam $B$ and draft $T$ has the half-breadth
#
# $$y(x,z) = \frac{B}{2}\Big(1-\big(\tfrac{2x}{L}\big)^2\Big)\Big(1-\big(\tfrac{z}{T}\big)^2\Big),
#   \qquad -\tfrac{L}{2}\le x\le \tfrac{L}{2},\ -T\le z\le 0 .$$
#
# At every station the side $y(z)$ is a parabola, which is exactly a quadratic Bézier curve: from
# the waterline point $(b,0)$ with control point $(b,-T/2)$ to the keel point $(0,-T)$, where $b$ is
# the half-breadth of the station. Above the waterline the sides rise vertically to the deck at a
# freeboard of $F$. The stem and the stern are cut at $|x| = 0.45L$, where the hull still has a
# tenth of its beam.
#
# **Reference.** Integrating the hull equation between the end stations $\pm x_e$,
#
# $$\nabla = \frac{4}{3} B T \Big(x_e - \frac{4x_e^3}{3L^2}\Big),\qquad
#   A_{WP} = 2B\Big(x_e - \frac{4x_e^3}{3L^2}\Big),\qquad z_B = -\frac{3}{8}T ,$$
#
# displacement volume, waterplane area and height of the centre of buoyancy (W. C. S. Wigley,
# *A comparison of experiment and calculated wave-profiles and wave-resistances for a form having
# parabolic waterlines*, Proc. Royal Society A 144 (1934) 144-159).

import numpy as np
from onesim.cad.d2 import Path, PlanarRegion, Profile, Segment
from onesim.cad.d3 import Box, Loft
from onesim.meshing import OctreeMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## Stations and loft
#
# A model hull of 4 m length, 0.4 m beam, 0.25 m draft and 0.1 m freeboard, eleven stations from
# stern to stem. Each station is a closed profile in its own plane: deck line, side, keel, side.

hull_length, beam, draft, freeboard = 4.0, 0.4, 0.25, 0.1
end = 0.45 * hull_length
stations = np.linspace(-end, end, 11)
half_breadth = lambda x: beam / 2 * (1 - (2 * x / hull_length) ** 2)
sections = [
    (
        Profile(
            Path(
                (b, freeboard),
                (
                    Segment((b, 0.0)),
                    Segment((0.0, -draft), control=((b, -draft / 2),)),
                    Segment((-b, 0.0), control=((-b, -draft / 2),)),
                    Segment((-b, freeboard)),
                    Segment((b, freeboard)),
                ),
            )
        ),
        x,
    )
    for x, b in zip(stations, half_breadth(stations))
]
hull = Loft(sections=sections, axis=(1.0, 0.0, 0.0))
print(
    f"hull: {len(hull.faces)} faces, bounding box {np.round(hull.bounding_box[0], 4)} to {np.round(hull.bounding_box[1], 4)} m"
)

# ## The underwater body
#
# The part of the hull below the waterline $z=0$ is the hull less a box above it. It is filled with
# tetrahedra cut from a grid of cubes of 30 mm at the shell, which grow to 60 mm inside; the grid
# follows a fine triangulation of the lofted shell (`faceted()`).

underwater = hull - Box(
    length=hull_length,
    width=beam,
    height=freeboard,
    origin=(-hull_length / 2, -beam / 2, 0.0),
)
mesh = OctreeMesher(target_size=0.03, coarsening=1).generate(underwater.faceted())

view = plots.Camera(elevation=35, azimuth=-50)
fig = plots.figure(1, 2, size=(17, 6))
plots.geometry(
    hull,
    selection={
        "shell": hull.boundaries.side,
        "transom and stem": hull.boundaries.bottom | hull.boundaries.top,
    },
    labels=False,
    camera=view,
    title="Wigley hull lofted through 11 stations",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"Underwater body: {mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Results
#
# Displacement of the CAD body and of its mesh, the waterplane area (the section of the hull at
# the waterline) and the centre of buoyancy against the closed-form integrals; the station areas
# are exact, so Simpson's rule over them is a second, discrete check of the loft.

integral = end - 4 * end**3 / (3 * hull_length**2)
displacement = 4 / 3 * beam * draft * integral
station_areas = (
    np.array([profile.area for profile, _ in sections])
    - 2 * half_breadth(stations) * freeboard
)
simpson = (
    (stations[1] - stations[0])
    / 3
    * (
        station_areas[0]
        + 4 * station_areas[1:-1:2].sum()
        + 2 * station_areas[2:-1:2].sum()
        + station_areas[-1]
    )
)
waterplane = hull.section((0, 0, 1), (0.0, 0.0, 0.0))
print(
    f"displacement: CAD {underwater.measure * 1e3:.4f} l, mesh {quality(mesh)['size'].sum() * 1e3:.4f} l, Simpson over the stations "
    f"{simpson * 1e3:.4f} l, closed form {displacement * 1e3:.4f} l ({underwater.measure / displacement - 1:+.2e})"
)
print(
    f"waterplane area: CAD {waterplane.measure:.6f} m2, closed form {2 * beam * integral:.6f} m2 ({waterplane.measure / (2 * beam * integral) - 1:+.2e})"
)
print(
    f"centre of buoyancy: x {underwater.center_of_mass[0]:+.5f} m (0 by symmetry), z {underwater.center_of_mass[2]:+.5f} m, "
    f"closed form {-3 / 8 * draft:+.5f} m"
)
print(
    f"block coefficient over the cut length: {underwater.measure / (2 * end * beam * draft):.4f}"
)

# ## Figure
#
# The waterplane with its two parabolic waterlines, the station areas along the hull against the
# closed form, and the midship station with its two parabolic sides.

x = np.linspace(-end, end, 101)
fig = plots.figure(1, 3, size=(20, 6))
plots.geometry(waterplane, title="Waterplane: section at z = 0", ax=fig[0])
plots.line(
    points={"station profiles": (stations, station_areas)},
    reference={"closed form 4/3 b(x) T": (x, 4 / 3 * half_breadth(x) * draft)},
    xlabel="x [m]",
    ylabel="underwater section area [m2]",
    title="Sectional area curve",
    ax=fig[1],
)
plots.geometry(
    PlanarRegion(profile=sections[5][0]),
    title="Midship station: deck line, sides, keel",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Mesh the water around the hull (`Box(...) - underwater`) for a flow model, thicken the shell
# with `hull.surface(hull.boundaries.side).thickened(0.004)` for a structural model, or change the
# stations to the lines of a real ship.
