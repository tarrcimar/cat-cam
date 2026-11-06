import numpy as np
import pytest

from catcam import mouth3d


@pytest.mark.parametrize("animal", mouth3d.ANIMALS)
def test_a_modelled_mouth_has_both_halves(animal):
    parts = mouth3d.build(np.ones(3), cavity=False, animal=animal)
    n = len(parts.pos)
    assert parts.normal.shape == parts.colour.shape == (n, 3)
    assert len(parts.jaw) == len(parts.light) == n
    assert parts.faces.min() >= 0 and parts.faces.max() < n
    # Upper teeth and gum on the skull, the rest on the jaw; only the tongue's body in
    # between, staying a little with the skull as the jaw drops, never less than half.
    assert np.all((parts.jaw == 0) | (parts.jaw >= 0.5)) and parts.jaw.max() == 1
    between = (parts.jaw > 0) & (parts.jaw < 1)
    assert 0 < between.mean() < 0.5
    skull, jaw = parts.pos[parts.jaw == 0], parts.pos[parts.jaw > 0]
    assert skull[:, 1].mean() > jaw[:, 1].mean()
    # The tongue's tip stays with the jaw, behind the lower teeth: what lags is further back.
    assert parts.pos[between, 2].max() < parts.pos[parts.jaw == 1, 2].max()
    # A few thousand triangles, inside the mouth: within its width, behind the lips.
    assert 1000 < len(parts.faces) < 10000
    assert np.abs(parts.pos[:, 0]).max() < 0.6
    assert parts.pos[:, 2].max() < 0.05 and parts.pos[:, 2].min() > -1.2
    assert np.abs(parts.pos[:, 1]).max() < 0.4
    assert np.allclose(np.linalg.norm(parts.normal, axis=1), 1, atol=0.01)
    assert np.all((parts.colour >= 0) & (parts.colour <= 1)) and np.all(parts.light > 0)


def test_the_lip_colour_tints_the_soft_parts_not_the_teeth():
    plain = mouth3d.build(np.ones(3), cavity=False)
    red = mouth3d.build(np.array([1.6, 0.7, 0.7]), cavity=False)
    changed = np.any(np.abs(red.colour - plain.colour) > 1e-6, axis=1)
    assert 0 < changed.mean() < 1  # gums, palate and tongue; the teeth stay white
    assert np.all(red.colour[changed, 0] > plain.colour[changed, 0])


def test_the_cavity_is_only_added_when_asked():
    without = mouth3d.build(np.ones(3), cavity=False)
    with_cavity = mouth3d.build(np.ones(3), cavity=True)
    assert len(with_cavity.pos) > len(without.pos)


@pytest.mark.parametrize("animal", mouth3d.ANIMALS)
def test_a_head_with_its_own_inside_leaves_out_the_palate(animal):
    """Its inside has a palate of its own (see catcam.head3d): the modelled one goes,
    the teeth, gums and tongue stay."""
    whole = mouth3d.load(animal, np.ones(3))
    parts = mouth3d.build(np.ones(3), cavity=False, animal=animal)
    palate = np.any([np.all(np.abs(np.rint(whole.colour * 255) - rgb) <= 1, axis=1)
                     for rgb in mouth3d.PALATE_COLOURS], axis=0)
    assert palate[whole.faces].all(axis=1).sum() > 100
    assert not palate[parts.faces].any()
    assert len(parts.faces) == np.count_nonzero(~palate[whole.faces].any(axis=1))


def test_dogs_and_cats_differ():
    cat, dog = (mouth3d.build(np.ones(3), cavity=False, animal=a) for a in ("cat", "dog"))
    assert cat.pos.shape != dog.pos.shape or not np.allclose(cat.pos, dog.pos)
