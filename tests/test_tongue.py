import cv2
import numpy as np
import pytest

from catcam.mouth_fit import LOWER_INNER_LIP, UPPER_INNER_LIP
from catcam.tongue import CHEEKS, CURL, FORWARD, INSET, LENGTH, OPENING_RANGE, WIDTH, seen, shape

MOUTH, FACE = 100.0, 300.0  # px


def _lips(angle=0.0, z=50.0):
    """A lower inner lip of 11 points, corner to corner, 100 px wide, tilted by angle."""
    xs = np.linspace(-50, 50, 11)
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack([150 + c * xs, 200 + s * xs, np.full(11, z)])


def _webcam(gap: float, inside, skin=(150, 180, 220)):
    """A face-coloured frame with a 120 px wide mouth opened by `gap` px, filled
    with the BGR colour `inside`, and the landmarks that go with it."""
    frame = np.empty((480, 640, 3), np.uint8)
    frame[:] = skin
    face = np.zeros((478, 3))
    face[:, 0] = np.linspace(170, 470, 478)  # 300 px wide
    face[:, 1] = 240
    xs = np.linspace(260, 380, 11)
    bulge = np.sin(np.linspace(0, np.pi, 11))
    face[UPPER_INNER_LIP, 0] = face[LOWER_INNER_LIP, 0] = xs
    face[UPPER_INNER_LIP, 1] = 300 - gap / 2 * bulge
    face[LOWER_INNER_LIP, 1] = 300 + gap / 2 * bulge
    face[CHEEKS, :2] = [(220, 230), (420, 230), (230, 260), (410, 260), (240, 280), (400, 280)]
    polygon = np.vstack([face[UPPER_INNER_LIP, :2], face[LOWER_INNER_LIP, :2][::-1]])
    cv2.fillPoly(frame, [polygon.astype(np.int32)], inside)
    return frame, face


TONGUE_PINK, THROAT, TEETH = (120, 90, 230), (20, 10, 30), (235, 240, 245)


def test_a_tongue_is_seen_filling_an_open_mouth():
    wide = OPENING_RANGE[1] * 300 * 1.5
    assert seen(*_webcam(wide, TONGUE_PINK)) == 1.0
    assert seen(*_webcam(wide, THROAT)) == 0.0
    assert seen(*_webcam(wide, TEETH)) == 0.0
    # Half open: the tongue is part of the way out.
    assert 0.0 < seen(*_webcam(np.mean(OPENING_RANGE) * 300, TONGUE_PINK)) < 0.8


def test_a_closed_mouth_shows_no_tongue():
    assert seen(*_webcam(0.0, TONGUE_PINK)) == 0.0
    assert seen(*_webcam(OPENING_RANGE[0] * 300 * 0.9, TONGUE_PINK)) == 0.0


def test_the_tongue_must_be_brighter_than_the_dark_of_the_mouth():
    wide = OPENING_RANGE[1] * 300 * 1.5
    dim_red = (20, 15, 60)  # a red but dark inside, as in a shadowed mouth
    assert seen(*_webcam(wide, dim_red)) == 0.0
    # ...relative to the skin: in a dark picture the same tongue still counts.
    assert seen(*_webcam(wide, (40, 30, 90), skin=(50, 60, 80))) == 1.0


def test_a_mouth_at_the_edge_of_the_picture_is_fine():
    frame, face = _webcam(40.0, TONGUE_PINK)
    face[:, 0] -= 300  # the mouth hangs off the left edge
    assert 0.0 <= seen(frame, face) <= 1.0
    face[:, 0] -= 400  # entirely outside
    assert seen(frame, face) == 0.0


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
