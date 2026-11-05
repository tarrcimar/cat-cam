import numpy as np
import pytest

from catcam.tongue import CURL, FORWARD, INSET, LENGTH, TONGUE_RANGE, WIDTH, blep, shape

MOUTH, FACE = 100.0, 300.0  # px


def _lips(angle=0.0, z=50.0):
    """A lower inner lip of 11 points, corner to corner, 100 px wide, tilted by angle."""
    xs = np.linspace(-50, 50, 11)
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack([150 + c * xs, 200 + s * xs, np.full(11, z)])


def test_blep_maps_the_tongue_score():
    assert blep({}) == 0.0
    assert blep({"tongueOut": TONGUE_RANGE[0]}) == 0.0
    assert blep({"tongueOut": np.mean(TONGUE_RANGE)}) == pytest.approx(0.5)
    assert blep({"tongueOut": 1.0}) == 1.0


def test_tongue_hangs_below_the_middle_of_the_lower_lip():
    out = shape(_lips(), 0.0, MOUTH, FACE, 1.0)
    assert out[:, 0].min() == pytest.approx(150 - WIDTH / 2 * MOUTH)
    assert out[:, 0].max() == pytest.approx(150 + WIDTH / 2 * MOUTH)
    assert out[:, 1].min() == pytest.approx(200)  # the root is on the lip (the mouth is closed)
    assert out[:, 1].max() == pytest.approx(200 + LENGTH * MOUTH)
    # In front of the lips, the tip further than the root.
    assert out[:, 2].max() == pytest.approx(50 - FORWARD * FACE)
    assert out[:, 2].min() == pytest.approx(50 - (FORWARD + CURL) * FACE)


def test_tongue_length_follows_the_amount():
    half = shape(_lips(), 0.0, MOUTH, FACE, 0.5)
    assert half[:, 1].max() == pytest.approx(200 + 0.5 * LENGTH * MOUTH)
    none = shape(_lips(), 0.0, MOUTH, FACE, 0.0)
    np.testing.assert_allclose(none[:, 1], 200)


def test_tongue_follows_a_tilted_mouth():
    angle = 0.3
    out = shape(_lips(angle), 0.0, MOUTH, FACE, 1.0)
    straight = shape(_lips(), 0.0, MOUTH, FACE, 1.0)
    # Same shape, rotated about the lip middle by the tilt.
    c, s = np.cos(angle), np.sin(angle)
    rel = straight[:, :2] - (150, 200)
    expected = np.column_stack([150 + c * rel[:, 0] - s * rel[:, 1], 200 + s * rel[:, 0] + c * rel[:, 1]])
    np.testing.assert_allclose(out[:, :2], expected, atol=1e-6)


def test_reversed_corners_give_the_same_tongue():
    out = shape(_lips(), 0.0, MOUTH, FACE, 1.0)
    flipped = shape(_lips()[::-1], 0.0, MOUTH, FACE, 1.0)
    np.testing.assert_allclose(sorted(map(tuple, out)), sorted(map(tuple, flipped)))


def test_open_mouth_roots_the_tongue_inside_it():
    open_wide = shape(_lips(), 60.0, MOUTH, FACE, 1.0)
    assert open_wide[:, 1].min() == pytest.approx(200 - INSET * MOUTH)
    ajar = shape(_lips(), 3.0, MOUTH, FACE, 1.0)
    assert ajar[:, 1].min() == pytest.approx(197)  # never above the upper lip


@pytest.fixture(scope="module")
def meshes(real_cats):
    from catcam.cat_mesh import CatMesh, Renderer

    renderer = Renderer((640, 360))
    return [(cat, CatMesh(renderer, cat.sprite, cat.points, (0, 0, 0))) for cat in real_cats]


def _pink(bgr) -> bool:
    b, g, r = (int(v) for v in bgr)
    return r > b + 30 and r > g + 30


@pytest.mark.gl
def test_every_cat_can_stick_its_tongue_out(meshes):
    from catcam.head_pose import to_matrix

    for cat, mesh in meshes:
        mouth = mesh.mouth_size * mesh.face_width * mesh.scale  # screen px
        for vec in [(0, 0, 0), (0, 0.4, 0), (-0.25, 0, 0.2)]:
            rotation = to_matrix(np.array(vec, float))
            still = np.zeros((478, 3))
            rest = mesh.render(cat.sprite, still, rotation)
            out = mesh.render(cat.sprite, still, rotation, tongue=1.0)
            ndc = mesh._project(mesh._shape(still), rotation)[14]  # the lower lip's middle
            lip = np.array([(ndc[0] + 1) / 2 * 640, (1 - ndc[1]) / 2 * 360])
            x, y = int(lip[0]), int(lip[1] + 0.2 * mouth)
            if vec == (0, 0, 0):
                x += int(0.08 * mouth)  # off the groove
                assert _pink(out[y, x]), (cat.name, out[y, x])
                assert np.abs(out[y, x].astype(int) - rest[y, x]).max() > 40, cat.name
            # Only the area under the mouth changes.
            changed = np.any(out != rest, axis=2)
            ys, xs = np.nonzero(changed)
            assert changed.any(), cat.name
            assert ys.min() > lip[1] - 0.2 * mouth, (cat.name, vec)
            assert ys.max() < lip[1] + (LENGTH + 0.3) * mouth, (cat.name, vec)
            assert np.ptp(xs) < 1.5 * mouth, (cat.name, vec)


@pytest.mark.gl
def test_a_tongue_kept_in_changes_nothing(meshes):
    for cat, mesh in meshes:
        before = mesh.render(cat.sprite, np.zeros((478, 3)), np.eye(3))
        after = mesh.render(cat.sprite, np.zeros((478, 3)), np.eye(3), tongue=0.0)
        np.testing.assert_array_equal(before, after)
