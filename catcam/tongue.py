"""A blep: the cat's tongue poking out between its lips.

Stick your tongue out and the cat's comes out too. It emerges from the lower
lip, slides out over it with a rounded tip and curls a little toward the
viewer, so it swings against the lips when the head turns. It is a small
strip of triangles hung from the lower inner lip of the head mesh, coloured
like the tongue inside the mouth (and tinted by the cat's lips), so every cat
has one. Nothing but your own tongue moves it.

Sizes are in mouth widths (the cat's closed mouth, corner to corner), like the
teeth, so a big-mouthed cat gets a bigger tongue.
"""

import numpy as np

from catcam.mouth_fit import LOWER_INNER_LIP

WIDTH = 0.5  # across
LENGTH = 0.5  # how far the tip reaches below the lower lip when fully out
INSET = 0.08  # the root sits this far up inside the mouth (at most the lip gap), hiding the join
FORWARD = 0.05  # in face widths: the tongue lies this far in front of the lips, over the lower lip
CURL = 0.04  # in face widths: the tip comes this much further toward the viewer
LIP_SHADOW = 0.12  # a mouth open less than this still shades the tongue where it leaves the lips
TONGUE_RANGE = (0.15, 0.6)  # tongueOut score mapped to 0..1 out

ROWS, COLS = 4, 3  # vertices along and across the tongue
_v, _u = np.meshgrid(np.linspace(0, 1, ROWS), np.linspace(-1, 1, COLS), indexing="ij")
GRID = np.column_stack([_u.ravel(), _v.ravel()])  # (ROWS * COLS, 2): u across (-1..1), v root (0) to tip (1)
MIDDLE = LOWER_INNER_LIP.index(14)


def _triangles() -> np.ndarray:
    quads = []
    for r in range(ROWS - 1):
        for c in range(COLS - 1):
            a = r * COLS + c
            quads += [(a, a + 1, a + COLS + 1), (a, a + COLS + 1, a + COLS)]
    return np.array(quads)


TRIANGLES = _triangles()

VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    uv = in_uv;
}
"""
# `size` is (half width, length) in mouth widths; uv.x runs -1..1 across the
# tongue and uv.y 0..1 from the root to the tip.
FRAGMENT_SHADER = """
#version 330
uniform vec2 size;
uniform vec3 lip_tint;   // colour of the cat's lips (RGB, mean 1), see cat_mesh
uniform float shade;     // lighting of a surface facing the viewer, for this head rotation
uniform float lip_shadow;  // 0..1 how much the upper lip shades the root (the mouth is nearly closed)
in vec2 uv;
out vec4 color;

const vec3 TONGUE = vec3(0.78, 0.37, 0.41);
const vec3 EDGE = vec3(0.5, 0.17, 0.21);

void main() {
    vec2 p = vec2(uv.x * size.x, uv.y * size.y);  // mouth widths, y down from the root
    float r = size.x;
    // The outline: straight sides and a round tip. When the tongue is barely
    // out only the tip of that round end shows, so it emerges as a small bump.
    float d = length(vec2(abs(p.x), max(p.y - (size.y - r), 0.0))) - r;
    float aa = fwidth(d) + 1e-4;
    float cover = 1.0 - smoothstep(-aa, aa, d);
    if (cover <= 0.0) discard;

    float inside = clamp(-d / r, 0.0, 1.0);  // 0 at the edge, 1 along the middle
    vec3 c = mix(EDGE, TONGUE, smoothstep(0.0, 0.4, inside)) * mix(vec3(1.0), lip_tint, 0.3);
    float toward_tip = smoothstep(size.y - r, size.y, p.y);
    c *= 1.0 - 0.25 * exp(-pow(p.x / 0.03, 2.0)) * (1.0 - toward_tip);  // the groove
    vec2 sheen = vec2((p.x - 0.35 * r) / (0.45 * r), (p.y - 0.45 * size.y) / (0.35 * size.y));
    c += 0.12 * exp(-dot(sheen, sheen));  // wet
    c *= mix(1.0, mix(0.55, 1.0, smoothstep(0.0, 0.07, p.y)), lip_shadow);  // under the upper lip
    color = vec4(c.bgr * shade, cover);  // the pipeline is BGR
}
"""


def _ramp(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def blep(scores: dict) -> float:
    """How far out your tongue is (0..1) from the face blendshape scores."""
    return _ramp(scores.get("tongueOut", 0.0), *TONGUE_RANGE)


def shape(lips: np.ndarray, gap: float, mouth_width: float, face_width: float, amount: float) -> np.ndarray:
    """The tongue's vertices (ROWS * COLS, 3), in the order of GRID, hanging
    from the lower inner lip.

    lips: the lower inner lip (11, 3) corner to corner, in the mesh's px
    (z toward the viewer is negative); gap: how far the lips are apart at the
    middle, px; mouth_width: the closed mouth corner to corner, px; amount:
    0..1 how far the tongue is out.
    """
    across = lips[-1, :2] - lips[0, :2]
    across /= np.linalg.norm(across) + 1e-9
    if across[0] < 0:
        across = -across
    down = np.array([-across[1], across[0]])
    root = lips[MIDDLE].copy()
    root[:2] -= down * min(max(gap, 0.0), INSET * mouth_width)
    root[2] -= FORWARD * face_width
    u, v = GRID[:, 0], GRID[:, 1]
    xy = root[:2] + np.outer(u * WIDTH / 2 * mouth_width, across)
    xy += np.outer(v * amount * LENGTH * mouth_width, down)
    z = root[2] - v * amount * CURL * face_width
    return np.column_stack([xy, z])
