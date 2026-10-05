"""Mirror substrate visualization in the existing optical system renderer."""

from __future__ import annotations

import pytest
import vtk

from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture, RectangularAperture
from optiland.visualization.system.mirror import Mirror3D
from optiland.visualization.system.system import OpticalSystem


def _backing_bounds(mirror: Mirror3D) -> tuple[float, ...]:
    """Render one mirror and return its substrate actor bounds."""
    renderer = vtk.vtkRenderer()
    mirror.plot(renderer)
    actors = renderer.GetActors()
    assert actors.GetNumberOfItems() == 2
    actors.InitTraversal()
    actors.GetNextActor()  # Reflective face.
    backing = actors.GetNextActor()
    backing.GetMapper().Update()
    assert backing.GetMapper().GetInput().GetNumberOfCells() > 0
    return backing.GetBounds()


def test_3d_backing_follows_mirror_path(set_test_backend) -> None:
    """A return mirror puts its substrate opposite the first mirror's."""
    optic = Optic()
    optic.surfaces.add(index=0, z=0)
    optic.surfaces.add(
        index=1,
        z=10,
        material="mirror",
        mirror_thickness=2,
        aperture=RadialAperture(2),
    )
    optic.surfaces.add(
        index=2,
        z=0,
        material="mirror",
        mirror_thickness=3,
        aperture=RadialAperture(2),
    )
    optic.surfaces.add(index=3, z=10)
    optic.wavelengths.add(value=0.55, is_primary=True)

    class Rays:
        r_extent = [2] * 4

    system = OpticalSystem(optic, Rays(), projection="3d")
    system._identify_components()
    mirrors = [
        component
        for component in system.components
        if isinstance(component, Mirror3D)
    ]
    assert [mirror.backing_sign for mirror in mirrors] == [1, -1]
    assert _backing_bounds(mirrors[0])[4:] == pytest.approx((10, 12))
    assert _backing_bounds(mirrors[1])[4:] == pytest.approx((-3, 0))


def test_3d_rectangular_mirror_backing(set_test_backend) -> None:
    """A clipped mirror has a finite back and side wall in both backends."""
    optic = Optic()
    optic.surfaces.add(index=0)
    optic.surfaces.add(
        index=1,
        z=10,
        material="mirror",
        mirror_thickness=2,
        aperture=RectangularAperture(-2, 2, -1, 1),
    )
    bounds = _backing_bounds(Mirror3D(optic.surfaces[1], 2))
    assert bounds == pytest.approx((-2, 2, -1, 1, 10, 12))
