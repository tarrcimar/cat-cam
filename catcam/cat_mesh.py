"""3D cat head mesh rendered with OpenGL.

Vertices are the cat's 478 face landmarks (with depth), points along the cat's
outline and a grid of "skull" points filling the fur in between. Face
vertices are moved by your expression each frame, the whole head rotates with
yours, and the gap between the lips is filled with a drawn mouth (teeth, dark
inside, tongue). When you stick your tongue out, a tongue hangs out over the
lower lip (see catcam.tongue). The texture is the cat sprite, so at rest the
mesh reproduces the photo.

Sizes are in face widths, so any sprite resolution behaves the same.
"""

import cv2
import moderngl
import numpy as np
from scipy.spatial import Delaunay

from catcam import tongue as blep
from catcam.cat_face import EYES
from catcam.ears import bend, find_ears
from catcam.lighting import shading, vertex_normals
from catcam.mouth_fit import LOWER_INNER_LIP, OUTER_LIPS, UPPER_INNER_LIP, snap_mouth

OUTLINE_POINTS = 90
SKULL_DEPTH = 0.15  # head outline sits this far (in face widths) behind the nearest face points
OUTLINE_DEPTH_SMOOTHING = 9  # outline points averaged for a smooth silhouette depth
CAMERA_DISTANCE = 4.0  # perspective strength, in face widths (bigger = flatter)
DEPTH_RANGE = 2.0  # face widths in front of / behind the pivot that fit in the GL depth range
MSAA_SAMPLES = 4
TEETH_SHOW = (0.01, 0.035)  # lip gap over which the teeth fade in
PROBE_OPENING = 0.18  # used once to find the triangles spanning the mouth
PROBE_TOLERANCE = 0.005
# Fur-follows-face: neighbourhood size and falloff.
FUR_FOLLOW = (0.1, 0.3)
SKULL_GRID = 0.07  # spacing of the skull points
SKULL_MARGIN = 0.03  # skull points keep this far from the face and the outline
LIP_SEAM = 0.0015  # half the gap between the lips of the closed mouth
# Cats with a smaller mouth than this (corner to corner, face widths) get theirs
# enlarged, as if liquified: lips, opening and teeth grow together. The area
# around the mouth follows with a falloff (in mouth half-widths: fully inside
# MOUTH_GROW[0], not at all beyond [1]; half as far up and down).
MOUTH_MIN = 0.33
MOUTH_MAX_GROW = 1.3  # enlarged by at most this, more would distort the face around it
MOUTH_GROW = (1.3, 2.5)
# Framing: ear tips and chin as fractions of the screen height.
TOP_MARGIN, CHIN_LINE = 0.04, 0.93  # room above for perked ears

# Around each eye the face moves rigidly: a person's eyelids travel further
# than these squinting cats' eyes are tall, so copying them would fold the
# mesh, and the few triangles inside an eye would turn tracking noise into
# zigzags. The eyelids are drawn closed in the texture instead. Ellipse around
# the eye, in eye widths, inside which the face is rigid, with a soft edge from
# EYE_RIGID[0] to [1] of its radius; the eye follows the average of the points
# around it (EYE_ANCHOR band, not above the eye: brows move on their own).
EYE_ELLIPSE = (0.75, 0.5)
EYE_RIGID = (0.55, 1.0)
EYE_ANCHOR = (0.9, 1.5)
EYE_CORNERS = ((33, 133), (263, 362))  # image-left eye, image-right eye
# Each face point's movement is averaged with its neighbours' (except on the
# lips, which must stay crisp to open), which evens out tracking noise.
SMOOTHING = 0.035  # face widths

# Outer and inner lip points whose midpoints lie on the lips.
LIP_SAMPLES = [(0, 13), (17, 14), (39, 81), (269, 311), (84, 87), (314, 317)]
FACING_VIEWER = np.array([[0.0, 0.0, -1.0]])  # the normal of a tongue hanging out, unrotated

FACE_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec2 in_uv;
in float in_shade;
out vec2 uv;
out float shade;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    uv = in_uv;
    shade = in_shade;
}
"""
FACE_FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
in vec2 uv;
in float shade;
out vec4 color;
void main() {
    vec4 c = texture(tex, uv);
    if (c.a < 0.02) discard;
    color = vec4(c.rgb * shade, c.a);
}
"""
# The inside of the mouth, drawn in the gap between the inner lips. Per vertex:
# s runs 0..1 from one mouth corner to the other, d is the distance below the
# upper lip and g the lip gap there, both in face widths. Everything inside is
# sized in "mouth widths" (the cat's closed mouth, corner to corner), so teeth
# keep their size however wide the mouth opens. Teeth, tongue and throat lie
# behind the lips and slide against them when the head turns (parallax).
MOUTH_VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec3 in_mouth;
in float in_shade;
out vec3 m;
out float shade;
void main() {
    // Slightly in front of the lips, so no skin stretched across the opening
    // can show through; the nose etc. are far enough forward to still cover it.
    gl_Position = vec4(in_pos.xy, in_pos.z - 0.01, 1.0);
    m = in_mouth;
    shade = in_shade;
}
"""
MOUTH_FRAGMENT_SHADER = """
#version 330
uniform float teeth_amount;  // 0 while the lips touch, so a closed mouth stays a dark line
uniform float mouth_size;    // the cat's mouth width at rest, face widths
uniform float mouth_span;    // current mouth width, face widths
uniform vec2 turn;           // image shift per face width of depth (from the head rotation)
uniform vec3 lip_tint;       // colour of the cat's lips (RGB, mean 1): gums, tongue and teeth take it on
in vec3 m;
in float shade;
out vec4 color;

const float TEETH_DEPTH = 0.015, LOWER_TEETH_DEPTH = 0.03, TONGUE_DEPTH = 0.06, THROAT_DEPTH = 0.14;
const float ARCH = 0.55;  // dental arch radius, mouth widths
// A cat's teeth from the middle outward: three small incisors, the fang,
// premolars, molar.
const int FANG = 3;
const float UPPER_W[6] = float[6](0.07, 0.065, 0.07, 0.12, 0.14, 0.15);
const float UPPER_L[6] = float[6](0.1, 0.095, 0.1, 0.3, 0.13, 0.1);
const float LOWER_W[6] = float[6](0.06, 0.055, 0.06, 0.11, 0.14, 0.15);
const float LOWER_L[6] = float[6](0.055, 0.05, 0.055, 0.17, 0.07, 0.05);

const vec3 ENAMEL = vec3(0.86, 0.83, 0.75);
const vec3 ENAMEL_ROOT = vec3(0.72, 0.64, 0.5);
const vec3 ENAMEL_TIP = vec3(0.7, 0.72, 0.76);
const vec3 GUM = vec3(0.78, 0.38, 0.42);
const vec3 TONGUE = vec3(0.74, 0.34, 0.38);
const vec3 MUCOSA = vec3(0.42, 0.11, 0.13);
const vec3 THROAT = vec3(0.05, 0.008, 0.015);
const vec3 TOOTH_SHADOW = vec3(0.3, 0.13, 0.11);  // teeth in shadow pick up the mouth's red

// Position of a layer lying `depth` behind the lips, as seen at this pixel:
// x across the mouth and y down from the upper lip, in mouth widths.
vec2 layer(float depth) {
    float x = (m.x - 0.5) * mouth_span - turn.x * depth;
    return vec2(x, m.y - turn.y * depth) / mouth_size;
}

// A row of teeth around the dental arch. `down` is the distance from the gum
// edge toward the tips (mouth widths). Returns (coverage, shade); `rgb` gets
// the tooth colour.
vec2 tooth_row(float x, float down, const float W[6], const float L[6], float point, out vec3 rgb) {
    float theta = asin(clamp(x / ARCH, -0.999, 0.999));
    float arc = abs(theta) * ARCH, start = 0.0;
    int k = 0;
    float w = W[0];
    for (int i = 0; i < 6; i++) {
        k = i; w = W[i];
        if (arc < start + w || i == 5) break;
        start += w;
    }
    float q = clamp((arc - start) / w, 0.0, 1.0);  // 0 = side toward the middle
    float side = abs(2.0 * q - 1.0);
    // The fang tapers to a point, slightly curved; the others are rounded off.
    float len = L[k] * (k == FANG ? 1.0 - point * pow(side, 1.4)
                                  : 1.0 - 0.2 * pow(smoothstep(0.5, 1.0, side), 2.0));
    float aa = fwidth(down) + 1e-4;
    float cover = 1.0 - smoothstep(len - aa, len + aa, down);
    // Dark slits between teeth, wider toward the tips.
    float slit = 0.025 + 0.06 * smoothstep(0.6 * len, len, down);
    float qa = fwidth(q) + 1e-4;
    cover *= smoothstep(0.0, slit * 0.5 + qa, min(q, 1.0 - q));
    float v = clamp(down / len, 0.0, 1.0);
    rgb = mix(ENAMEL_ROOT, ENAMEL * pow(lip_tint, vec3(0.15)), smoothstep(0.0, 0.35, v));
    rgb = mix(rgb, ENAMEL_TIP, smoothstep(0.65, 1.0, v));
    float facing = cos(theta);
    float shade = (0.25 + 0.75 * pow(facing, 1.5)) * (0.8 + 0.2 * sqrt(sin(3.14159 * q)));
    rgb = mix(TOOTH_SHADOW, rgb, clamp(shade, 0.0, 1.0));
    rgb += 0.1 * exp(-pow((q - 0.42) / 0.12, 2.0)) * smoothstep(0.15, 0.4, v)
           * (1.0 - smoothstep(0.6, 0.9, v)) * pow(facing, 4.0);  // shine
    return vec2(cover, shade);
}

void main() {
    float gap = m.z / mouth_size;
    // Throat: dark in the middle and back, reddish toward the lips and corners.
    vec2 t = layer(THROAT_DEPTH);
    float into = clamp(min(t.y, gap - t.y) / max(0.5 * gap, 1e-4), 0.0, 1.0);
    float centre = 1.0 - smoothstep(0.15, 0.5, abs(t.x));
    vec3 gum_color = GUM * mix(vec3(1.0), lip_tint, 0.5);
    vec3 mucosa = MUCOSA * mix(vec3(1.0), lip_tint, 0.3);
    vec3 c = mix(mucosa, THROAT, smoothstep(0.0, 1.0, into) * mix(0.55, 1.0, centre));

    // Tongue, rising from the floor of the mouth.
    vec2 g = layer(TONGUE_DEPTH);
    float rise = gap - g.y;  // height above the lower lip
    float height = min(0.4 * gap, 0.16) * sqrt(max(1.0 - pow(g.x / 0.36, 2.0), 0.0));
    float aa = fwidth(rise) + 1e-4;
    float on_tongue = 1.0 - smoothstep(height - aa, height + aa, rise);
    float groove = 1.0 - 0.3 * exp(-pow(g.x / 0.035, 2.0));
    float top = smoothstep(0.0, max(height, 1e-4), rise);  // 0 at the front, 1 at its back edge
    float sides = mix(0.7, 1.0, 1.0 - smoothstep(0.2, 0.36, abs(g.x)));
    vec3 tongue = TONGUE * mix(vec3(1.0), lip_tint, 0.3) * groove * mix(1.05, 0.55, top * top) * sides;
    c = mix(c, tongue, on_tongue);

    // Lower teeth, mostly hidden behind the lower lip.
    vec3 rgb;
    vec2 lo = layer(LOWER_TEETH_DEPTH);
    vec2 lower = tooth_row(lo.x, gap - lo.y, LOWER_W, LOWER_L, 0.8, rgb);
    c = mix(c, mix(TOOTH_SHADOW, rgb, 0.75), lower.x * teeth_amount);

    // Upper teeth, with the gum line just showing between them.
    vec2 up = layer(TEETH_DEPTH);
    vec2 upper = tooth_row(up.x, up.y, UPPER_W, UPPER_L, 0.85, rgb);
    c = mix(c, rgb, upper.x * teeth_amount);
    float gum = 1.0 - smoothstep(0.012, 0.02, up.y);
    c = mix(c, gum_color * 0.8, gum * teeth_amount);

    // The lips shade the inside right next to them.
    float lip = m.y / mouth_size, rim = gap - lip;
    c *= mix(0.45, 1.0, smoothstep(0.0, 0.05, lip)) * mix(0.6, 1.0, smoothstep(0.0, 0.03, rim));
    c *= mix(0.55, 1.0, smoothstep(0.0, 0.12, 0.5 - abs(m.x - 0.5)));  // corners
    color = vec4(c.bgr * shade, 1.0);  // the pipeline is BGR
}
"""


def _lip_tint(sprite: np.ndarray, face: np.ndarray) -> np.ndarray:
    """The cat's lip colour as RGB with mean 1."""
    samples = []
    for i, j in LIP_SAMPLES:
        x, y = ((face[i, :2] + face[j, :2]) / 2).astype(int)
        samples.append(sprite[y - 2:y + 3, x - 2:x + 3, :3].reshape(-1, 3))
    rgb = np.median(np.concatenate(samples), axis=0)[::-1].astype(float) + 1
    return rgb / rgb.mean()


def _outline(alpha: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        (alpha > 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=len)[:, 0].astype(np.float64) + 0.5
    step = len(contour) / OUTLINE_POINTS
    return contour[(np.arange(OUTLINE_POINTS) * step).astype(int)]


def _closed_mouth(face: np.ndarray, face_width: float) -> np.ndarray:
    """The face with its lips together, so at rest no mouth opening shows.

    (The fitted inner lips of a cat rarely touch exactly.) They stay a hair
    apart, as the mesh needs distinct vertices.
    """
    face = face.copy()
    upper, lower = UPPER_INNER_LIP[1:-1], LOWER_INNER_LIP[1:-1]
    middle = (face[upper] + face[lower]) / 2
    face[upper] = middle - (0, LIP_SEAM * face_width, 0)
    face[lower] = middle + (0, LIP_SEAM * face_width, 0)
    return face


def _outline_depth(outline_xy: np.ndarray, face: np.ndarray, face_width: float) -> np.ndarray:
    """Depth of the head outline: behind the face points next to it.

    Beside the cheeks that is behind the face's sides (in front of them the mesh
    would fold over when turning); above the forehead it is much less deep,
    or the forehead would slide over the top of the head when nodding.
    """
    nearest = np.linalg.norm(outline_xy[:, None] - face[None, :, :2], axis=2).argmin(axis=1)
    z = face[nearest, 2] + SKULL_DEPTH * face_width
    k = OUTLINE_DEPTH_SMOOTHING
    padded = np.concatenate([z[-k:], z, z[:k]])  # the outline is closed
    return np.convolve(padded, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


def _bigger_mouth(points: np.ndarray, face: np.ndarray, face_width: float) -> np.ndarray:
    """The points with the mouth region scaled up to at least MOUTH_MIN wide."""
    corners = face[[78, 308]]
    width = np.linalg.norm(corners[0, :2] - corners[1, :2])
    grow = min(MOUTH_MIN * face_width / width, MOUTH_MAX_GROW)
    if grow <= 1:
        return points
    centre = (corners.mean(axis=0) + face[[13, 14]].mean(axis=0)) / 2
    d = (points[:, :2] - centre[:2]) / (np.array([1.0, 0.5]) * width / 2)
    t = np.clip((np.linalg.norm(d, axis=1) - MOUTH_GROW[0]) / (MOUTH_GROW[1] - MOUTH_GROW[0]), 0, 1)
    weight = 1 - t * t * (3 - 2 * t)
    out = points.copy()
    out[:, :2] += (grow - 1) * weight[:, None] * (points[:, :2] - centre[:2])
    return out


def _skull_points(alpha: np.ndarray, face: np.ndarray, face_width: float, outline: np.ndarray):
    """Grid points in the fur around the face, with depth easing from the face
    edge back to the outline, so the head bends smoothly when it turns."""
    mask = (alpha > 20).astype(np.uint8)
    to_outline = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    hull = cv2.convexHull(face[:, :2].astype(np.float32))
    step = SKULL_GRID * face_width
    margin = SKULL_MARGIN * face_width
    ys, xs = np.mgrid[step / 2:alpha.shape[0]:step, step / 2:alpha.shape[1]:step]
    grid = np.column_stack([xs.ravel(), ys.ravel()])
    d_outline = to_outline[grid[:, 1].astype(int), grid[:, 0].astype(int)]
    d_face = -np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in grid])
    keep = (d_outline > margin) & (d_face > margin)
    grid, d_outline, d_face = grid[keep], d_outline[keep], d_face[keep]
    near_face = np.linalg.norm(grid[:, None] - face[None, :, :2], axis=2).argmin(axis=1)
    near_outline = np.linalg.norm(grid[:, None] - outline[None, :, :2], axis=2).argmin(axis=1)
    t = d_face / (d_face + d_outline)
    z = (1 - t) * face[near_face, 2] + t * outline[near_outline, 2]
    return np.column_stack([grid, z])


def _eye_weights(face: np.ndarray):
    """Per eye: how rigidly each face point moves with the eye (2, 478), and
    the averaging weights of the points the eye follows (2, 478)."""
    rigid = np.zeros((len(EYE_CORNERS), len(face)))
    anchor = np.zeros_like(rigid)
    for eye, corners in enumerate(EYE_CORNERS):
        a, b = face[list(corners), :2]
        width = np.linalg.norm(a - b)
        d = (face[:, :2] - (a + b) / 2) / (np.array(EYE_ELLIPSE) * width)
        r = np.linalg.norm(d, axis=1)
        t = np.clip((r - EYE_RIGID[0]) / (EYE_RIGID[1] - EYE_RIGID[0]), 0, 1)
        rigid[eye] = 1 - t * t * (3 - 2 * t)  # smoothstep falloff
        ring = (r > EYE_ANCHOR[0]) & (r < EYE_ANCHOR[1]) & (d[:, 1] > -0.4)
        ring[list(corners)] = True
        anchor[eye] = ring / ring.sum()
    for eye, (upper, lower) in enumerate(EYES):
        rigid[eye, upper + lower] = 1  # the lids themselves always
    return rigid, anchor


def _smoothing(face: np.ndarray, face_width: float) -> np.ndarray:
    """(478, 478) matrix averaging each point's movement with its neighbours'."""
    lips = np.zeros(len(face), bool)
    outline = face[OUTER_LIPS, :2].astype(np.float32)
    for i, (x, y) in enumerate(face[:, :2]):
        lips[i] = cv2.pointPolygonTest(outline, (float(x), float(y)), True) > -0.01 * face_width
    dist = np.linalg.norm(face[:, None, :2] - face[None, :, :2], axis=2)
    weights = np.exp(-((dist / (SMOOTHING * face_width)) ** 2)) * ~lips[None, :]
    weights[lips] = 0
    weights[lips, lips] = 1  # lips keep their own movement
    return weights / weights.sum(axis=1, keepdims=True)


def _face_triangles(points_2d: np.ndarray, face_width: float) -> np.ndarray:
    """Delaunay triangles, minus the ones bridging the lips (the mouth opening).

    The mouth is opened artificially (everything below the upper lip moves
    down) and triangulated open, so the triangles follow the lips; the ones
    inside the opening are then dropped.
    """
    upper = points_2d[UPPER_INNER_LIP]
    order = np.argsort(upper[:, 0])
    lip_line = np.interp(points_2d[:, 0], upper[order, 0], upper[order, 1])
    opened = points_2d.copy()
    below = points_2d[:, 1] > lip_line + 0.5 * PROBE_TOLERANCE * face_width
    below[LOWER_INNER_LIP[1:-1]] = True
    below[UPPER_INNER_LIP] = False
    opened[below, 1] += PROBE_OPENING * face_width

    triangles = Delaunay(opened).simplices
    mouth = np.vstack([opened[UPPER_INNER_LIP], opened[LOWER_INNER_LIP][::-1]]).astype(np.float32)
    corners = opened[triangles]  # (n, 3, 2)
    probes = np.concatenate([
        corners.mean(axis=1, keepdims=True),
        (corners + np.roll(corners, 1, axis=1)) / 2,  # edge midpoints
    ], axis=1)
    tolerance = PROBE_TOLERANCE * face_width
    inside = np.array([
        any(cv2.pointPolygonTest(mouth, (float(x), float(y)), True) > tolerance for x, y in tri)
        for tri in probes
    ])
    # Only triangles made of lip points span the opening. Others reaching into
    # it (near the corners, where the lips are thin) stay, so no hole opens
    # beside the mouth; the mouth is drawn in front of them anyway.
    on_lips = np.isin(triangles, UPPER_INNER_LIP + LOWER_INNER_LIP).all(axis=1)
    return triangles[~(inside & on_lips)]


def _mouth_triangles() -> np.ndarray:
    """Strip of triangles between the inner lip contours."""
    u, lo = UPPER_INNER_LIP, LOWER_INNER_LIP
    quads = [((u[i], u[i + 1], lo[i + 1]), (u[i], lo[i + 1], lo[i])) for i in range(len(u) - 1)]
    return np.array(quads).reshape(-1, 3)


class Renderer:
    """OpenGL context, shaders and frame buffers, shared by all cats."""

    def __init__(self, out_size):
        self.out_w, self.out_h = out_size
        self.ctx = moderngl.create_standalone_context(require=330)
        self.face_prog = self.ctx.program(
            vertex_shader=FACE_VERTEX_SHADER, fragment_shader=FACE_FRAGMENT_SHADER
        )
        self.mouth_prog = self.ctx.program(
            vertex_shader=MOUTH_VERTEX_SHADER, fragment_shader=MOUTH_FRAGMENT_SHADER
        )
        self.tongue_prog = self.ctx.program(
            vertex_shader=blep.VERTEX_SHADER, fragment_shader=blep.FRAGMENT_SHADER
        )
        size = (self.out_w, self.out_h)
        self.msaa = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(size, 4, samples=MSAA_SAMPLES)],
            depth_attachment=self.ctx.depth_renderbuffer(size, samples=MSAA_SAMPLES),
        )
        self.resolved = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(size, 4)])


class CatMesh:
    def __init__(self, renderer: Renderer, sprite: np.ndarray, landmarks: np.ndarray, bg_color):
        """sprite: BGRA, landmarks: (478, 3) sprite px with depth, bg_color: BGR."""
        self.renderer = renderer
        self.out_w, self.out_h = renderer.out_w, renderer.out_h
        self.bg = tuple(np.asarray(bg_color) / 255) + (1.0,)
        tex_h, tex_w = sprite.shape[:2]

        self.face_count = len(landmarks)
        self.face_width = np.ptp(landmarks[:, 0])
        landmarks = snap_mouth(sprite, landmarks, self.face_width)
        face = _closed_mouth(landmarks, self.face_width)
        outline_xy = _outline(sprite[..., 3])
        outline = np.column_stack([outline_xy, _outline_depth(outline_xy, face, self.face_width)])
        skull = _skull_points(sprite[..., 3], face, self.face_width, outline)

        photo = np.vstack([face, skull, outline])  # where each vertex is in the texture
        self.rest = _bigger_mouth(photo, face, self.face_width)
        face = self.rest[: self.face_count]
        self.ears = find_ears(self.rest, outline_xy, self.face_count, self.face_width)
        # The fur follows the nearby face points (e.g. cheeks widening in a
        # smile) with a falloff, so the face edge can't fold over the fur.
        fur = self.rest[self.face_count:, :2]
        dist = np.linalg.norm(fur[:, None] - face[None, :, :2], axis=2)
        weights = np.exp(-((dist / (FUR_FOLLOW[0] * self.face_width)) ** 2))
        weights /= weights.sum(axis=1, keepdims=True) + 1e-9
        falloff = np.exp(-((dist.min(axis=1) / (FUR_FOLLOW[1] * self.face_width)) ** 2))
        self.fur_follow = weights * falloff[:, None]  # (fur points, face points)
        self.eye_rigid, self.eye_anchor = _eye_weights(face)
        self.smoothing = _smoothing(face, self.face_width)
        # Rotate around the middle of the face so the head turns in place.
        self.pivot = np.array([face[:, 0].mean(), face[1, 1], np.median(face[:, 2])])
        self.focal = CAMERA_DISTANCE * self.face_width

        top, chin = outline_xy[:, 1].min(), face[:, 1].max()
        self.scale = (CHIN_LINE - TOP_MARGIN) * self.out_h / (chin - top)
        self.offset = np.array([
            self.out_w / 2 - self.pivot[0] * self.scale,
            TOP_MARGIN * self.out_h - top * self.scale,
        ])

        self.normals = vertex_normals(self.rest, self.face_width)

        ctx = renderer.ctx
        self.vbo = ctx.buffer(reserve=len(self.rest) * 3 * 4, dynamic=True)
        self.shade_vbo = ctx.buffer(reserve=len(self.rest) * 4, dynamic=True)
        uv = (photo[:, :2] / (tex_w, tex_h)).astype("f4")
        triangles = _face_triangles(photo[:, :2], self.face_width)
        self.face_vao = ctx.vertex_array(
            renderer.face_prog,
            [(self.vbo, "3f", "in_pos"), (ctx.buffer(uv.tobytes()), "2f", "in_uv"),
             (self.shade_vbo, "1f", "in_shade")],
            ctx.buffer(triangles.astype("i4").tobytes()),
        )
        self.lip_tint = _lip_tint(sprite, landmarks)
        self.mouth_size = np.linalg.norm(face[78, :2] - face[308, :2]) / self.face_width
        self.mouth = np.zeros((len(self.rest), 3), "f4")  # per vertex: s, d, g (see shader)
        across = np.linspace(0, 1, len(UPPER_INNER_LIP))
        self.mouth[UPPER_INNER_LIP, 0] = self.mouth[LOWER_INNER_LIP, 0] = across
        self.mouth_vbo = ctx.buffer(self.mouth.tobytes(), dynamic=True)
        self.mouth_vao = ctx.vertex_array(
            renderer.mouth_prog,
            [(self.vbo, "3f", "in_pos"), (self.mouth_vbo, "3f", "in_mouth"),
             (self.shade_vbo, "1f", "in_shade")],
            ctx.buffer(_mouth_triangles().astype("i4").tobytes()),
        )
        self.tongue_vbo = ctx.buffer(reserve=len(blep.GRID) * 3 * 4, dynamic=True)
        self.tongue_vao = ctx.vertex_array(
            renderer.tongue_prog,
            [(self.tongue_vbo, "3f", "in_pos"),
             (ctx.buffer(blep.GRID.astype("f4").tobytes()), "2f", "in_uv")],
            ctx.buffer(blep.TRIANGLES.astype("i4").tobytes()),
        )

        self.texture = ctx.texture((tex_w, tex_h), 4)
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._uploaded = None  # sprite currently in the texture

    def face_on_screen(self):
        """Screen position of the cat's face center and its width in screen px."""
        return self.offset + self.pivot[:2] * self.scale, self.face_width * self.scale

    def _steady_eyes(self, offsets: np.ndarray) -> np.ndarray:
        """Offsets smoothed between neighbours, with the eye regions moving rigidly."""
        out = self.smoothing @ offsets
        for rigid, anchor in zip(self.eye_rigid, self.eye_anchor):
            out += rigid[:, None] * (anchor @ out - out)
        return out

    def _shape(self, offsets: np.ndarray, ears=(0.0, 0.0)) -> np.ndarray:
        """The mesh deformed by the face offsets and ear angles, unrotated."""
        offsets = self._steady_eyes(offsets)
        shaped = self.rest.copy()
        shaped[: self.face_count] += offsets
        shaped[self.face_count:] += self.fur_follow @ offsets
        return bend(shaped, self.ears, ears)

    def _project(self, shaped: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Rotate the shaped mesh and project it to NDC."""
        pos = (shaped - self.pivot) @ rotation.T + self.pivot
        # Perspective relative to the unrotated pose, so at rest the mesh matches the photo.
        depth = lambda z: self.focal / (self.focal + z - self.pivot[2])
        persp = (depth(pos[:, 2]) / depth(shaped[:, 2]))[:, None]
        xy = self.pivot[:2] + (pos[:, :2] - self.pivot[:2]) * persp
        screen = self.offset + xy * self.scale
        return np.column_stack([
            2 * screen[:, 0] / self.out_w - 1,
            1 - 2 * screen[:, 1] / self.out_h,
            (pos[:, 2] - self.pivot[2]) / (DEPTH_RANGE * self.face_width),
        ]).astype("f4")

    def _update_mouth(self, shaped: np.ndarray, rotation: np.ndarray):
        upper, lower = shaped[UPPER_INNER_LIP], shaped[LOWER_INNER_LIP]
        gaps = np.maximum(lower[:, 1] - upper[:, 1], 0) / self.face_width
        self.mouth[UPPER_INNER_LIP, 1] = 0
        self.mouth[LOWER_INNER_LIP, 1] = gaps
        self.mouth[UPPER_INNER_LIP, 2] = self.mouth[LOWER_INNER_LIP, 2] = gaps
        self.mouth_vbo.write(self.mouth.tobytes())
        prog = self.renderer.mouth_prog
        gap = gaps[1:-1].mean()
        teeth = (gap - TEETH_SHOW[0]) / (TEETH_SHOW[1] - TEETH_SHOW[0])
        prog["teeth_amount"].value = float(np.clip(teeth, 0, 1))
        prog["mouth_size"].value = float(self.mouth_size)
        prog["lip_tint"].value = tuple(float(v) for v in self.lip_tint)
        prog["mouth_span"].value = float(np.linalg.norm(upper[-1, :2] - upper[0, :2]) / self.face_width)
        # A point one face width deeper moves by rotation[:2, 2] (image x, y).
        prog["turn"].value = tuple(float(v) for v in rotation[:2, 2])

    def _draw_tongue(self, shaped: np.ndarray, rotation: np.ndarray, amount: float):
        """The tongue hanging `amount` (0..1) out of the shaped mouth, drawn
        over the face (so it must come after it)."""
        gap = float(shaped[14, 1] - shaped[13, 1])
        mouth_width = self.mouth_size * self.face_width
        vertices = blep.shape(shaped[LOWER_INNER_LIP], gap, mouth_width, self.face_width, amount)
        self.tongue_vbo.write(self._project(vertices, rotation).tobytes())
        prog = self.renderer.tongue_prog
        prog["size"].value = (blep.WIDTH / 2, amount * blep.LENGTH)
        prog["lip_tint"].value = tuple(float(v) for v in self.lip_tint)
        prog["shade"].value = float(shading(FACING_VIEWER, rotation)[0])
        prog["lip_shadow"].value = float(np.clip(1 - gap / (blep.LIP_SHADOW * mouth_width), 0, 1))
        self.tongue_vao.render()

    def render(self, sprite: np.ndarray, offsets: np.ndarray, rotation: np.ndarray,
               ears=(0.0, 0.0), tongue: float = 0.0) -> np.ndarray:
        """Draw the cat with face offsets (478, 3), head rotation, ear angles
        (radians, image-left and image-right ear, positive = flattened) and how
        far its tongue is out (0..1); returns BGR."""
        if sprite is not self._uploaded:  # unchanged while the eyes are open
            self.texture.write(np.ascontiguousarray(sprite).tobytes())
            self.texture.build_mipmaps()
            self._uploaded = sprite
        shaped = self._shape(offsets, ears)
        self.vbo.write(self._project(shaped, rotation).tobytes())
        self.shade_vbo.write(shading(self.normals, rotation).astype("f4").tobytes())
        r = self.renderer
        self._update_mouth(shaped, rotation)
        r.msaa.use()
        r.ctx.clear(*self.bg, depth=1.0)
        r.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        r.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.mouth_vao.render()
        self.texture.use(0)
        self.face_vao.render()
        if tongue > 0.01:
            self._draw_tongue(shaped, rotation, tongue)
        r.ctx.copy_framebuffer(r.resolved, r.msaa)

        data = r.resolved.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(self.out_h, self.out_w, 3)[::-1]
