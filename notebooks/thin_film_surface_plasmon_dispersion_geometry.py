"""Geometry of the OneSimulation example "Surface plasmons of a thin silver film in glass".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, a strip of glass
1 um long and 6 um high (thin_film_surface_plasmon_dispersion.step; x_min and x_max are the
waveguide ports, y_min and y_max the far walls of the cladding), the 40 nm silver film along its
middle (thin_film_surface_plasmon_dispersion_film.step) and a 240 nm band of glass around the film
that is meshed finely (thin_film_surface_plasmon_dispersion_band.step).
"""

from onesim.cad.d3 import Box

length, height, film, band, depth = 1.0, 6.0, 0.04, 0.24, 0.1
Box(length=length, width=height, height=depth, origin=(0, -height / 2, 0)).save(
    "thin_film_surface_plasmon_dispersion.step"
)
Box(length=length, width=film, height=depth, origin=(0, -film / 2, 0)).save(
    "thin_film_surface_plasmon_dispersion_film.step"
)
Box(length=length, width=band, height=depth, origin=(0, -band / 2, 0)).save(
    "thin_film_surface_plasmon_dispersion_band.step"
)
