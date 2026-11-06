"""A cat drawn as a real 3D head model instead of the warped photo.

The head is a textured mesh (assets/cats/<Name>/head3d.npz + head3d.png),
made from the cat's portrait by an image-to-3D model and imported with
catcam.head3d_import, which also records where every vertex sits in the
portrait. That is what lets it borrow what the 2D cat does: each vertex
moves with the face landmarks around it (expressions, the jaw), the eye areas
of the texture blink with the portrait's eyelids, and the ears bend about the
same hinges. The lips part over a real 3D mouth
(catcam.mouth3d) whose jaw swings open with your lower lip. Head3D stands in
for CatMesh: same render() and face_on_screen().
"""

from typing import NamedTuple

import cv2
import moderngl
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from catcam import mouth3d
from catcam.cat_mesh import CHIN_LINE, TOP_MARGIN, CatMesh, Renderer, _outline
from catcam.ears import bend, find_ears
from catcam.lighting import LIGHT, SHADE_RANGE, STRENGTH
from catcam.mouth_fit import LOWER_INNER_LIP, OUTER_LIPS, UPPER_INNER_LIP
from catcam.paths import ASSETS

FOV = 25  # vertical field of view of the virtual camera, degrees
FACE_BAND = (0.35, 0.75)  # the face's width is measured between these heights (0 = chin, 1 = ear tips)
# How vertices follow the face landmarks (face widths): each takes a Gaussian-weighted
# mix of its nearest landmarks' movement, fading out away from the face.
NEAREST = 8
FOLLOW = 0.06
FALLOFF = 0.25  # wide, so the whole lower jaw and chin move together instead of tearing
# Near the mouth, vertices only follow landmarks on their own side of the lips, so the
# lips part instead of stretching: this far past the mouth corners and above/below the lip line.
MOUTH_ZONE = (0.06, 0.15)
SIDES_FADE = 0.4  # with its own jaw, the outer share of that zone over which the rule fades out
# Face widths: only skin this close to a face landmark (in 3D) counts as the lips; the back
# of the head lines up with the mouth in the portrait too.
ON_FACE = 0.1
DEPTH_SPAN = 1.5  # model units in front of / behind the head kept in the depth range
# Image axes (x right, y down, z away from the camera) <-> model axes (y up, z toward it).
FLIP = np.diag([1.0, -1.0, -1.0])
MOUTH_CORNERS = (UPPER_INNER_LIP[0], UPPER_INNER_LIP[-1])
LIP_MIDDLE = (13, 14)  # where the upper and lower inner lips meet in the middle
LOWER_LIP = LOWER_INNER_LIP[1:-1] + OUTER_LIPS[12:]  # its landmarks, without the corners
JAW_MAX = 1.0  # radians, the most the jaw tilts; beyond that it just drops with the lip
BREATH_RISE = 0.008  # head heights a full breath lifts the head by
# A head with its own jaw (see catcam.head3d_import) opens it about a hinge this
# far from the middle of its lips (face widths: right, up, out of the mouth), where
# a jaw joint is, at most OWN_JAW_MAX radians. In face widths, not mouth widths: a
# small mouth doesn't make a short jaw, and about a hinge just behind the lips the
# chin would swing back into the throat instead of down.
JAW_HINGE = (0.0, 0.04, -0.55)
OWN_JAW_MAX = 0.5
OWN_GAPE_MAX = 1.0  # mouth widths: the most its lips part, so a small mouth opens round, not in a slot
# A cat's upper lip hangs from its nose: whatever yours does (rising as you smile or
# open wide), each point of it moves up or down from where the nose goes by at most
# UPPER_LIP_TRAVEL of how far it is from the nose, so it never rides up into the nose
# nor drags it down.
UPPER_LIP = UPPER_INNER_LIP + OUTER_LIPS[:11]
NOSE = [1, 2, 4, 5, 98, 327, 195, 197, 48, 278]  # landmarks around the nose
UPPER_LIP_TRAVEL = 0.5
MOUTH_BOX = 0.3  # how far round the jaw (shares of its size on screen) the head's inside may show
# A head's own inside is drawn as a mouth cavity (see CAVITY_SHADING): its centre and radii
# in the mouth frame (mouth widths: right, up, out of the mouth; up: to its roof, just above
# the teeth, where the palate is), how far down its floor is with the mouth closed, ...
OWN_CAVITY = ((0.0, -0.03, -0.45), (0.75, 0.12, 0.6))
OWN_FLOOR = 0.4
OWN_LINING = (0.8, 0.52, 0.56)  # ...the RGB (0..1) of its lining, dusky pink, tinted by the lip colour...
OWN_THROAT = (0.42, 0.2, 0.25)  # ...and of its throat, a deeper red (the light falling off does the rest)...
OWN_THROAT_DEPTH = 3.0  # ...which goes on down this much further back (mouth widths), so is lit as that deep
LINING_TINT = 0.5  # how much of the lip colour the lining takes on (like the gums, tools/mouth_model.py)
LIPS_SHADOW = 0.08  # mouth widths: how far inside the lips the dark line where they meet the mouth reaches
PIGMENT = {"dog": 0.45}  # how much of the lining is blotched dark, by animal (cats: none)
# How the inside of the mouth (the cavity, teeth, gums and tongue alike, see MOUTH_LIGHT)
# is lit, to match the soft photo light baked into the heads: mostly light from all round
# that only gets in through the opening between the lips, so it falls off deeper inside
# and under the lips, plus a little of the head's own light (catcam.lighting) where it
# shines in through the opening.
INSIDE_OPENING = 0.4  # mouth widths: half the width of the opening the light gets in through
INSIDE_AMBIENT = 1.3  # brightness of a wall that sees all the light getting in from all round...
INSIDE_BOUNCE = 0.3  # ...the share of that reaching even the deepest corners, off the lining...
INSIDE_BOUNCE_GAPE = 0.3  # ...once the lips are this far apart (mouth widths; less gets in before)...
INSIDE_KEY = 0.25  # ...and of one lit square on by the head's light through the opening
INSIDE_SHEEN = 0.15  # how bright the wet glints are where that light glances back off it
LID_REACH = 2  # portrait px: texels sampled this close to where the lids are drawn take part in blinks

# The head is shaped here, on the GPU, as Head3D._shape does it on the CPU: with a few
# hundred thousand vertices that would take most of a frame. Each vertex follows its
# NEAREST (= 8, two ivec4) landmarks, bends with the ears and swings with the jaw.
VERTEX_SHADER = """
#version 330
uniform mat4 proj;
uniform mat3 rot;
uniform vec3 shift;
uniform vec3 light;
uniform float strength;
uniform vec2 shade_range;
uniform sampler2D moves;  // the landmarks' movement (portrait px), one per texel
uniform mat3 to_model;    // a movement in portrait px as one of the model
uniform vec2 ear_hinge[2];
uniform float ear_turn[2];  // radians, signed as each ear turns outward
uniform mat4 jaw;           // model -> model for the head's own jaw
uniform vec3 tie_shift;     // its move, less its turn of the upper lip's
in vec3 in_pos;  // at rest
in vec3 in_normal;
in vec2 in_uv;
in vec2 in_photo;  // where it is in the portrait (px)
in uvec4 in_near0;
in uvec4 in_near1;
in vec4 in_follow0;  // how much it moves with each of those landmarks
in vec4 in_follow1;
in vec2 in_ear;      // how much of each ear's turn it takes
in float in_jaw;     // how much it moves with the jaw
in float in_tie;     // on the jaw's side of the lip cut: how far it follows the jaw from its twin
out vec2 uv;
out float shade;
vec3 landmark(uint i, int row) {
    return texelFetch(moves, ivec2(int(i), row), 0).xyz;
}
void main() {
    vec3 move = vec3(0.0);
    for (int k = 0; k < 4; k++)
        move += in_follow0[k] * landmark(in_near0[k], 0) + in_follow1[k] * landmark(in_near1[k], 0);
    // Ears turn about their hinges in the portrait (see catcam.ears.bend), one after the other.
    vec2 here = in_photo + move.xy;
    vec2 bent = here;
    for (int e = 0; e < 2; e++) {
        if (in_ear[e] > 0.0 && ear_turn[e] != 0.0) {
            float phi = ear_turn[e] * in_ear[e];
            float c = cos(phi), s = sin(phi);
            vec2 rel = bent - ear_hinge[e];
            bent = ear_hinge[e] + vec2(c * rel.x - s * rel.y, s * rel.x + c * rel.y);
        }
    }
    move.xy += bent - here;
    vec3 pos = in_pos + to_model * move;
    if (in_jaw != 0.0) {
        // The jaw swings rigidly, with what the lips do beyond that (see Head3D._beyond).
        vec3 beyond = vec3(0.0);
        for (int k = 0; k < 4; k++)
            beyond += in_follow0[k] * landmark(in_near0[k], 1) + in_follow1[k] * landmark(in_near1[k], 1);
        pos += in_jaw * ((jaw * vec4(in_pos + to_model * beyond, 1.0)).xyz - pos);
    }
    if (in_tie >= 0.0)
        pos += in_tie * (mat3(jaw) * pos + tie_shift - pos);
    gl_Position = proj * vec4(rot * pos + shift, 1.0);
    uv = in_uv;
    // The texture already has the light of the photo it came from baked in; only the
    // change in facing since then brightens or darkens it (see catcam.lighting).
    float before = clamp(dot(in_normal, light), 0.0, 1.0);
    float after = clamp(dot(rot * in_normal, light), 0.0, 1.0);
    shade = clamp(1.0 + strength * (after - before), shade_range.x, shade_range.y);
}
"""
# Whether screen px p is inside the opening between the lips (a polygon of OPENING
# points, crossing test).
OPENING = 22
IN_OPENING = """
uniform vec2 opening[22];  // the gap between the lips on screen (px), around its edge
bool in_opening(vec2 p) {
    bool inside = false;
    for (int i = 0, j = 21; i < 22; j = i++) {
        vec2 a = opening[i], b = opening[j];
        if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}
"""
# How light gets into the mouth, for the cavity and the teeth, gums and tongue alike: the
# light from all round reaches a point inside only through the opening between the lips,
# as much as it sees of it (the solid angle of the opening, taken as a rectangle in the
# lips' plane in the mouth frame, from the upper lip down to the lower), so it falls off
# with depth, into the throat, the back corners and under the lips. Some of it, bounced
# off the walls and through the flesh, reaches everywhere, red with the lining's colour,
# as in a real mouth, whose shadows deepen into red rather than grey. The head's light
# lights what it reaches through the opening and glints off it, wet.
MOUTH_LIGHT = """
uniform mat4 to_mouth;    // camera space -> mouth frame (mouth widths, see catcam.mouth3d)
uniform float gape;       // mouth widths: how far the jaw's front dropped
uniform vec3 key;         // toward the head's light (camera space, catcam.lighting)
uniform float opening_half;  // mouth widths: half the opening's width (INSIDE_OPENING)
uniform vec4 inside_light;   // INSIDE_AMBIENT, INSIDE_KEY, INSIDE_SHEEN, INSIDE_BOUNCE
uniform vec3 bounce;         // RGB: the colour of the light bounced off the lining
uniform float bounce_gape;   // mouth widths: INSIDE_BOUNCE_GAPE
// One corner's share of the solid angle a rectangle is seen in, d in front of it.
float corner(vec2 c, float d) {
    return atan(c.x * c.y, d * sqrt(dot(c, c) + d * d));
}
// The light (RGB) at q (mouth frame) on a surface facing n (both in the mouth frame), and
// its wet glint (to be added) seen from the camera.
vec4 mouth_light(vec3 q, vec3 n) {
    float d = max(-q.z, 0.005);
    float open = max(gape, 0.03);
    vec2 lo = vec2(-opening_half, -open) - q.xy, hi = vec2(opening_half, 0.0) - q.xy;
    float seen = (corner(hi, d) - corner(vec2(lo.x, hi.y), d) - corner(vec2(hi.x, lo.y), d)
                  + corner(lo, d)) / 6.2831853;  // of all round; at most half, right at the lips
    vec3 to_gap = normalize(vec3(-q.x, -0.5 * open - q.y, d));
    float facing = 0.35 + 0.65 * max(dot(n, to_gap), 0.0);
    float direct = pow(min(2.0 * seen, 1.0), 0.4) * facing;
    float bounced = inside_light.w * min(open / bounce_gape, 1.0);  // as much as gets in
    vec3 lit = inside_light.x * ((1.0 - inside_light.w) * direct + bounced * bounce);
    // The head's light, where its ray from q leaves through the opening (soft edged, the
    // further away the softer).
    vec3 l = normalize(mat3(to_mouth) * key);
    float t = d / max(l.z, 0.05);
    vec2 at = q.xy + t * l.xy;
    float inside = min(min(at.x + opening_half, opening_half - at.x), min(at.y + open, -at.y));
    float soft = 0.02 + 0.1 * t;
    float through = smoothstep(-soft, soft, inside);
    lit += inside_light.y * through * max(dot(n, l), 0.0);
    vec3 v = normalize((to_mouth * vec4(0.0, 0.0, 0.0, 1.0)).xyz - q);
    float glint = inside_light.z * through * pow(max(dot(n, normalize(l + v)), 0.0), 120.0);
    return vec4(lit, glint);
}
"""
# A head's own inside, seen between its parted lips, is drawn as the mouth cavity behind
# them: the view ray through each pixel is followed into an egg-shaped hollow in the mouth
# frame (OWN_CAVITY; its lower half deepens as the jaw drops) and coloured where it hits
# its far wall, lit as MOUTH_LIGHT says: a pink lining with the palate's ridges across its
# roof and fine folds down the cheeks, a deeper red, smooth throat at the back, darker
# below where the tongue shades it. Only a backdrop: it is never drawn outside the head's
# own inside, and the teeth and tongue go in front.
CAVITY_SHADING = """
uniform vec2 focal;       // the projection's x and y scale
uniform vec2 screen;      // px
uniform vec3 cavity_centre;  // the cavity (mouth frame)...
uniform vec3 cavity_radii;   // ...its y: up to the roof...
uniform float cavity_floor;  // ...and down to the floor with the mouth closed (further as the jaw drops)
uniform float rim;        // px: how far inside the lips the dark line where they meet the mouth reaches
uniform vec3 lining;      // RGB: the lining...
uniform vec3 throat;      // ...and the throat
uniform float pigment;    // how much the lining is blotched dark (a dog's)
uniform float throat_depth;  // mouth widths: how much further back the throat goes

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1) * 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float noise(vec3 x) {  // smooth value noise, 0..1
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    vec2 e = vec2(1.0, 0.0);
    vec4 front = vec4(hash(i), hash(i + e.xyy), hash(i + e.yxy), hash(i + e.xxy));
    vec4 back = vec4(hash(i + e.yyx), hash(i + e.xyx), hash(i + e.yxx), hash(i + e.xxx));
    vec4 z = mix(front, back, f.z);  // the cell's four corners, at x's depth
    vec2 y = mix(z.xy, z.zw, f.y);
    return mix(y.x, y.y, f.x);
}
// px from p to the edge of the opening between the lips.
float to_lips(vec2 p) {
    float near = 1e9;
    for (int i = 0, j = 21; i < 22; j = i++) {
        vec2 a = opening[j], ab = opening[i] - a;
        float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
        near = min(near, length(p - a - t * ab));
    }
    return near;
}
// RGB of the cavity's far wall seen at screen px p; gap: p is between the lips.
vec3 cavity(vec2 p, bool gap) {
    vec3 o = (to_mouth * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 d = mat3(to_mouth) * vec3((p / screen * 2.0 - 1.0) / focal, -1.0);
    vec3 q = vec3(0.0), u = vec3(0.0, 0.0, -1.0), n = vec3(0.0, 0.0, 1.0);
    for (int lower = 0; lower < 2; lower++) {
        vec3 r = vec3(cavity_radii.x, lower == 1 ? cavity_floor + gape : cavity_radii.y, cavity_radii.z);
        vec3 oo = (o - cavity_centre) / r, dd = d / r;
        float a = dot(dd, dd), b = dot(oo, dd), c = dot(oo, oo) - 1.0;
        float disc = b * b - a * c;
        if (disc < 0.0) continue;
        vec3 hit = o + (-b + sqrt(disc)) / a * d;  // where the ray leaves it
        if ((hit.y < cavity_centre.y) == (lower == 1)) {
            q = hit;
            u = (hit - cavity_centre) / r;
            n = -normalize(u / r);  // facing in
            break;
        }
    }
    // Turned off the grid the noise is made on, so it shows no rows.
    vec3 m = mat3(0.8, 0.36, -0.48, -0.6, 0.48, -0.64, 0.0, 0.8, 0.6) * q;
    float coarse = noise(m * 9.0), fine = noise(m * 37.0);
    vec3 c = lining * (0.9 + 0.14 * coarse + 0.06 * fine);
    // The palate's ridges, arching back across the roof from behind the teeth to the soft
    // palate: pale crests, darker furrows, and the seam down its middle.
    float roof = smoothstep(0.2, 0.6, u.y) * smoothstep(-0.7, -0.5, q.z) * smoothstep(0.75, 0.4, abs(u.x));
    float ridge = sin((q.z + 1.2 * q.x * q.x) * 80.0);
    c *= 1.0 + roof * (0.22 * smoothstep(0.2, 0.9, ridge) - 0.2 * smoothstep(0.2, 0.9, -ridge)
                       - 0.18 * (1.0 - smoothstep(0.0, 0.02, abs(q.x))));
    // The cheeks' lining runs back from the mouth's corners in fine folds.
    float cheek = smoothstep(0.5, 0.85, abs(u.x));
    float fold = sin(q.y * 110.0 + 9.0 * noise(vec3(q.x * 6.0, q.y * 7.0, q.z * 12.0)));
    c *= 1.0 - 0.08 * cheek * smoothstep(0.7, 1.0, fold) * noise(q * 14.0);
    // Deep at the back (u.z = -1), halfway down, the throat (more of it as the jaw drops).
    float throat_y = cavity_centre.y - 0.2 - 0.5 * gape;
    float deep = smoothstep(0.45, 0.95, -u.z) * exp(-pow((q.y - throat_y) / (0.3 + 0.5 * gape), 2.0)
                                                   - pow(u.x / 0.6, 2.0));
    c = mix(c, throat, deep);
    c = mix(c, c * 0.3, pigment * smoothstep(0.5, 0.7, noise(m * 4.0 + 7.0)));
    // The lining is soft and a little uneven, which breaks up its light and its wet glints.
    vec3 bump = vec3(noise(m * 70.0), noise(m * 70.0 + 17.0), noise(m * 70.0 + 31.0)) - 0.5;
    // The throat goes on down beyond the cavity: lit as if that much deeper.
    vec4 lighting = mouth_light(q - vec3(0.0, 0.0, throat_depth * deep), normalize(n + 0.5 * bump));
    vec3 lit = lighting.rgb * mix(0.5, 1.0, smoothstep(-0.9, -0.1, u.y));  // the floor, in the tongue's shade
    float glint = lighting.a * (1.0 - deep);
    // A dark line where the lips meet the mouth; back faces round them are in their shadow.
    float shadow = gap ? mix(0.4, 1.0, smoothstep(0.0, rim, to_lips(p))) : 0.35;
    return (c * lit + glint) * shadow;
}
"""
FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
uniform bool seam;       // drawing the skin across the lips, which gives way where they part
uniform bool own_mouth;  // nothing added inside: the head's own inside shows between the lips, as
uniform vec4 mouth_box;  // the mouth cavity: its back faces around the mouth (mouth_box, screen
uniform sampler2D skin;  // px) with no outside of the head behind them (skin's depth) too
uniform bool outside_only;  // drawing the skin's depth: the opening between the lips has none
in vec2 uv;
in float shade;
out vec4 color;
""" + IN_OPENING + MOUTH_LIGHT + CAVITY_SHADING + """
void main() {
    gl_FragDepth = gl_FragCoord.z;
    vec2 p = gl_FragCoord.xy;
    bool near_mouth = all(greaterThan(p, mouth_box.xy)) && all(lessThan(p, mouth_box.zw));
    // Between its cut lips everything of the head is its inside (whatever the model has
    // there was never seen); round them, so are back faces with no outside behind them.
    // Elsewhere a back face is the back of a tuft of fur, drawn like the rest.
    bool gap = in_opening(p);
    if (outside_only && gap) discard;
    bool inner = gap || !gl_FrontFacing && near_mouth && texelFetch(skin, ivec2(p), 0).r == 1.0;
    if (own_mouth && inner) {
        color = vec4(cavity(p, gap).bgr, 1.0);  // BGR like the app
        // Between the lips it is only the backdrop: the teeth and tongue go in front of it.
        if (gap) gl_FragDepth = 0.99999;
        return;
    }
    if (seam && in_opening(gl_FragCoord.xy)) discard;
    color = vec4(texture(tex, uv).rgb * shade, 1.0);
}
"""
# The mouth parts (catcam.mouth3d): the jaw half swings about the hinge, the rest
# moves with the upper lip; lit as the cavity behind them is (MOUTH_LIGHT), on top of the
# shadows the model has baked in where its parts shade each other.
MOUTH_VERTEX_SHADER = """
#version 330
uniform mat4 proj;
uniform mat3 rot;
uniform vec3 shift;
uniform mat4 skull;  // mouth frame -> model, moving with the upper lip
uniform mat4 jaw;    // the same with the jaw opened
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_colour;
in float in_jaw;
in float in_light;
out vec3 colour;
out vec3 normal;
out vec3 eye;  // camera space
out float light;
void main() {
    vec3 p = mix((skull * vec4(in_pos, 1.0)).xyz, (jaw * vec4(in_pos, 1.0)).xyz, in_jaw);
    vec3 n = mix(mat3(skull) * in_normal, mat3(jaw) * in_normal, in_jaw);
    eye = rot * p + shift;
    gl_Position = proj * vec4(eye, 1.0);
    normal = rot * n;
    colour = in_colour;
    light = in_light;
}
"""
MOUTH_FRAGMENT_SHADER = """
#version 330
in vec3 colour;
in vec3 normal;
in vec3 eye;
in float light;
uniform bool clip;        // only seen through the opening between the lips...
uniform sampler2D skin;   // ...or else never in front of the head's outside (its depth)
out vec4 color;
""" + IN_OPENING + MOUTH_LIGHT + """
void main() {
    // Only seen through the mouth: parts that would poke out through the chin or
    // cheeks (which slope back faster than the mouth inside) stay hidden. A head with
    // its own mouth really parts its lips: there, what pokes out in front of its skin
    // goes, and its skin hides the rest.
    if (clip) {
        if (!in_opening(gl_FragCoord.xy)) discard;
    } else {
        float outside = texelFetch(skin, ivec2(gl_FragCoord.xy), 0).r;
        if (outside < 1.0 && gl_FragCoord.z < outside) discard;
    }
    vec3 n = normalize(mat3(to_mouth) * normal);
    vec4 lighting = mouth_light((to_mouth * vec4(eye, 1.0)).xyz, gl_FrontFacing ? n : -n);
    color = vec4((colour * light * lighting.rgb + lighting.a).bgr, 1.0);  // BGR like the app
}
"""


def files(name: str):
    return ASSETS / name / "head3d.npz", ASSETS / name / "head3d.png"


def has_head(name: str) -> bool:
    return all(f.exists() for f in files(name))


def animal(name: str) -> str:
    """What the cat is ("cat", "dog"): its 3D head says (head3d_import --animal); a cat
    without one is a cat."""
    mesh_file = files(name)[0]
    if not mesh_file.exists():
        return "cat"
    with np.load(mesh_file) as mesh:
        return str(mesh["animal"]) if "animal" in mesh.files else "cat"


def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])


def sample(image: np.ndarray, px: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear colours of image at px (n, 2); remap wants 2D maps under 32k a side."""
    n = len(px)
    if n == 0:
        return np.zeros((0, image.shape[2]), np.float32)
    pad = -n % width
    maps = [np.pad(c.astype(np.float32), (0, pad)).reshape(-1, width) for c in px.T]
    return cv2.remap(image, *maps, cv2.INTER_LINEAR).reshape(-1, image.shape[2])[:n]


def lip_line(points):
    """y of the line between the lips (portrait px) at given x."""
    seam = (points[UPPER_INNER_LIP, :2] + points[LOWER_INNER_LIP, :2]) / 2
    order = np.argsort(seam[:, 0])
    return lambda x: np.interp(x, seam[order, 0], seam[order, 1])


def lip_sides(photo_px, points, face_width):
    """For each vertex: whether it is near the mouth, below the line between the lips,
    and between the mouth corners (all in the portrait), and for each landmark whether
    it belongs below the line."""
    seam = (points[UPPER_INNER_LIP, :2] + points[LOWER_INNER_LIP, :2]) / 2
    line_y = lip_line(points)
    left, right = seam[:, 0].min(), seam[:, 0].max()
    reach_x, reach_y = (r * face_width for r in MOUTH_ZONE)
    x, y = photo_px[:, 0], photo_px[:, 1]
    close = np.abs(y - line_y(x)) < reach_y
    near_mouth = close & (x > left - reach_x) & (x < right + reach_x)
    inside = close & (x > left) & (x < right)
    below = points[:, 1] > line_y(points[:, 0])
    below[UPPER_INNER_LIP], below[LOWER_INNER_LIP] = False, True
    return near_mouth, y > line_y(x), inside, below


def follow_weights(vertices, photo_px, landmarks, points, face_model, face_width,
                   on_jaw=None) -> csr_matrix:
    """(vertices, landmarks) weights: how each vertex moves with the face landmarks.
    Closeness is measured on the model (landmarks: their spots on it; face_model: a
    face width in model units), so the back of the head stays put; which side of the
    lips a vertex is on is decided in the portrait (photo_px, points, face_width px),
    or for a head with its own jaw by whether it is on the jaw (on_jaw)."""
    tree = cKDTree(landmarks)
    dist, nearest = tree.query(vertices, k=NEAREST)
    weights = np.exp(-((dist / (FOLLOW * face_model)) ** 2))
    # Lips: keep each vertex to the landmarks on its side of the line between them.
    near_mouth, vertex_below, _, below = lip_sides(photo_px, points, face_width)
    if on_jaw is None:
        weights[near_mouth[:, None] & (below[nearest] != vertex_below[:, None])] = 0
    else:
        # By the lips, the jaw follows only what is below them and the face only what is
        # above, so the lips part. Away from them that fades: all of a vertex's nearest
        # landmarks can be on the other side (beside the chin), and following none of
        # them it would stay put while its neighbours move, tearing the skin between.
        seam = (points[UPPER_INNER_LIP, :2] + points[LOWER_INNER_LIP, :2]) / 2
        reach_x, reach_y = (r * face_width for r in MOUTH_ZONE)
        x, y = photo_px[:, 0], photo_px[:, 1]
        past = np.maximum(seam[:, 0].min() - x, x - seam[:, 0].max()).clip(0) / reach_x
        off = np.maximum(np.abs(y - lip_line(points)(x)) / reach_y, past)
        by_lips = 1 - np.clip((off - (1 - SIDES_FADE)) / SIDES_FADE, 0, 1)
        weights *= np.where(below[nearest] != on_jaw[:, None], 1 - by_lips[:, None], 1)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.where(total > 0, weights / np.maximum(total, 1e-12), 0)
    falloff = np.exp(-((dist[:, 0] / (FALLOFF * face_model)) ** 2))
    weights *= falloff[:, None]
    rows = np.repeat(np.arange(len(photo_px)), NEAREST)
    return csr_matrix((weights.ravel(), (rows, nearest.ravel())), shape=(len(photo_px), len(points)))


class _Part(NamedTuple):
    """Some of a head's vertices, with what Head3D._shape needs to shape them."""
    rows: np.ndarray | slice  # them, and any face-side twins they are tied to
    follow: csr_matrix  # their rows of the follow weights
    ears: list  # the ears, with weights for the face landmarks and these vertices only
    face: np.ndarray  # tied twins (positions in rows): the face's side of the lip cut...
    chin: np.ndarray  # ...and the jaw's
    back: np.ndarray | slice  # where in rows the vertices asked for are


def _padded(weights: csr_matrix, width: int):
    """The (rows, width) column indices (uint16) and values (float32) of a sparse matrix
    with at most width entries a row, padded with index 0 and weight 0."""
    counts = np.diff(weights.indptr)
    if counts.max(initial=0) > width:
        raise ValueError(f"more than {width} entries in a row")
    rows = np.repeat(np.arange(weights.shape[0]), counts)
    slot = np.arange(weights.nnz) - np.repeat(weights.indptr[:-1], counts)
    index = np.zeros((weights.shape[0], width), np.uint16)
    value = np.zeros((weights.shape[0], width), np.float32)
    index[rows, slot], value[rows, slot] = weights.indices, weights.data
    return index, value


def lip_seam(faces, photo_px, points, face_width, on_face) -> np.ndarray:
    """Which faces bridge the lips between the mouth corners: they give way where the
    mouth opens. on_face: which vertices are on the face (not behind it)."""
    _, vertex_below, inside, _ = lip_sides(photo_px, points, face_width)
    sides = vertex_below[faces]
    lips = (inside & on_face)[faces].all(axis=1)
    return lips & (sides.any(axis=1) != sides.all(axis=1))


def mouth_frame(landmarks, facing):
    """Origin (where the lips meet in the middle), axes (columns: toward the image-right
    corner, up, out of the mouth) and width of the mouth, in model units; facing: the
    direction the face looks in, in the model."""
    left, right = landmarks[MOUTH_CORNERS[0]], landmarks[MOUTH_CORNERS[1]]
    origin = (landmarks[LIP_MIDDLE[0]] + landmarks[LIP_MIDDLE[1]]) / 2
    x = (right - left) / np.linalg.norm(right - left)
    z = facing - (facing @ x) * x
    z /= np.linalg.norm(z)
    return origin, np.column_stack([x, np.cross(z, x), z]), np.linalg.norm(right - left)


class Head3D:
    def __init__(self, renderer: Renderer, name: str, flat: CatMesh, sprite: np.ndarray,
                 points: np.ndarray, bg_color, lids: tuple[int, int, int, int] | None = None):
        """flat: the same cat's CatMesh, whose expression smoothing and mouth this
        head shares; sprite: its open-eyed portrait (BGRA); points: its landmarks;
        bg_color: BGR; lids: the part of the portrait blinking redraws (x0, y0, x1, y1,
        see CatFace.lid_box), None if not known."""
        mesh_file, texture_file = files(name)
        mesh = np.load(mesh_file)
        self.renderer, self.flat, self.open_sprite = renderer, flat, sprite
        self.out_w, self.out_h = renderer.out_w, renderer.out_h
        self.bg = tuple(np.asarray(bg_color) / 255) + (0.0,)  # alpha 0: not the animal

        centre = (mesh["vertices"].min(0) + mesh["vertices"].max(0)) / 2
        self.rest = mesh["vertices"] - centre
        self.landmarks = mesh["landmarks"] - centre
        self.photo_px = mesh["photo_px"]
        # A movement in portrait px (x right, y down, z away) as a movement of the model.
        self.to_model = (mesh["cam_rot"].T @ np.diag([1, -1, -1]) / mesh["cam_scale"]).astype(np.float64)
        self.to_px = np.linalg.inv(self.to_model)
        face_width = np.ptp(points[:, 0])
        self.rigid = lip_sides(self.photo_px, points, face_width)[3]  # below the lips...
        self.rigid[LOWER_LIP] = False  # ...but for the lower lip
        # How far each point of the upper lip is from the nose (portrait px, see UPPER_LIP_TRAVEL).
        self.lip_travel = UPPER_LIP_TRAVEL * cKDTree(points[NOSE, :2]).query(points[UPPER_LIP, :2])[0]
        jaw = mesh["jaw_weight"] if "jaw_weight" in mesh.files else np.zeros(0)
        self.jaw_weight = jaw.astype(np.float64)[:, None] if len(jaw) else None
        self.follow = follow_weights(self.rest, self.photo_px, self.landmarks, points,
                                     face_width / float(mesh["cam_scale"]), face_width,
                                     None if self.jaw_weight is None else jaw > 0.5)
        outline = _outline(sprite[..., 3])
        self.ears = find_ears(np.vstack([points[:, :2], self.photo_px]), outline, len(points), face_width)
        self.face_count = len(points)

        # Frame it like CatMesh: top of the ears at TOP_MARGIN, chin at CHIN_LINE.
        height = np.ptp(self.rest[:, 1])
        half = np.tan(np.radians(FOV) / 2)
        distance = height / ((CHIN_LINE - TOP_MARGIN) * 2 * half)
        middle_ndc = 1 - (TOP_MARGIN + CHIN_LINE)
        self.shift = np.array([0.0, middle_ndc * distance * half, -distance])
        self.lift = BREATH_RISE * height  # model units, at a full breath
        self.moved = self.shift  # where the head is this frame (it rises as it breathes)
        self.proj = perspective(FOV, self.out_w / self.out_h, distance - DEPTH_SPAN, distance + DEPTH_SPAN)
        px_per_unit = self.out_h / (2 * half * distance)
        lo, hi = self.rest[:, 1].min() + np.array(FACE_BAND) * height
        band = self.rest[(self.rest[:, 1] > lo) & (self.rest[:, 1] < hi)]
        self.face_width = np.ptp((band if len(band) else self.rest)[:, 0]) * px_per_unit
        self.face_centre = np.array([self.out_w / 2, (1 - middle_ndc) / 2 * self.out_h])

        ctx = renderer.ctx
        self.prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.prog["proj"].write(self.proj.astype("f4").T.tobytes())
        self.prog["shift"].value = tuple(self.shift)
        self.prog["light"].value = tuple(LIGHT @ FLIP)
        self.prog["strength"].value = STRENGTH
        self.prog["shade_range"].value = SHADE_RANGE
        self.prog["to_model"].write(self.to_model.astype("f4").T.tobytes())
        self.prog["moves"].value = 2
        self.moves = ctx.texture((len(points), 2), 3, dtype="f4")  # rows: the landmarks' moves, and _beyond
        self.moves.filter = (moderngl.NEAREST, moderngl.NEAREST)
        face_model = self.face_model = face_width / float(mesh["cam_scale"])
        on_face = cKDTree(self.landmarks).query(self.rest)[0] < ON_FACE * face_model
        self.lip_cut = mesh["lip_cut"] if "lip_cut" in mesh.files else None
        self.everything = self._part(slice(None))
        near, follow = _padded(self.follow, NEAREST)
        ear = np.zeros((len(self.rest), 2), "f4")
        for i, e in enumerate(self.ears):
            ear[:, i] = e.weights[self.face_count:]
        jaw = np.zeros(len(self.rest)) if self.jaw_weight is None else self.jaw_weight[:, 0]
        shaping = [self.rest.astype("f4"), self.photo_px.astype("f4"), near, follow, ear, jaw.astype("f4")]
        # A vertex on the jaw's side of the lip cut is shaped as its twin on the face's side,
        # then tied to it by its own jaw weight (-1: not tied).
        tie = np.full(len(self.rest), -1, "f4")
        face, chin = self.everything.face, self.everything.chin
        if len(chin):
            tie[chin] = jaw[chin]
            for values in shaping:
                values[chin] = values[face]
        attributes = [(ctx.buffer(shaping[0]), "3f", "in_pos"),
                      (ctx.buffer(mesh["normals"].astype("f4").tobytes()), "3f", "in_normal"),
                      (ctx.buffer(mesh["uv"].astype("f4").tobytes()), "2f", "in_uv"),
                      (ctx.buffer(shaping[1]), "2f", "in_photo"),
                      (ctx.buffer(shaping[2]), "4u2 4u2", "in_near0", "in_near1"),
                      (ctx.buffer(shaping[3]), "4f 4f", "in_follow0", "in_follow1"),
                      (ctx.buffer(shaping[4]), "2f", "in_ear"),
                      (ctx.buffer(shaping[5]), "1f", "in_jaw"),
                      (ctx.buffer(tie), "1f", "in_tie")]
        if self.jaw_weight is None:
            seam = lip_seam(mesh["faces"], self.photo_px, points, face_width, on_face)
        else:  # the jaw is cut free already (see catcam.head3d_import.cut_jaw)
            seam = np.zeros(len(mesh["faces"]), bool)
        self.vao, self.seam_vao = (
            ctx.vertex_array(self.prog, attributes, ctx.buffer(mesh["faces"][part].astype("i4").tobytes()))
            if part.any() else None for part in (~seam, seam))

        # The mouth: real teeth, tongue and cavity behind lips that part, unless the
        # head keeps its own (imported with --own-mouth): then only its inside shows.
        self.own_mouth = bool(mesh["own_mouth"]) if "own_mouth" in mesh.files else False
        self.prog["own_mouth"].value = self.own_mouth
        # The portrait's camera looks straight at the face: its axis is where the face looks.
        origin, axes, width = mouth_frame(self.landmarks, mesh["cam_rot"][2].astype(np.float64))
        if "slit_middle" in mesh.files:  # its own lips: the mouth sits where they really meet
            origin, width = mesh["slit_middle"] - centre, float(mesh["slit_width"])
        self.mouth_origin, self.mouth_axes, self.mouth_width = origin, axes, width
        # The portrait's lips rarely meet exactly; the mouth opens as far as they part
        # from there, so a closed mouth shows no gap.
        half_gap = (self.landmarks[LOWER_INNER_LIP] - self.landmarks[UPPER_INNER_LIP]) / 2
        self.lips_meet = np.vstack([half_gap, -half_gap[::-1]])
        # Cat teeth unless the head says it is another animal (head3d_import --animal).
        animal = str(mesh["animal"]) if "animal" in mesh.files else "cat"
        parts = mouth3d.build(flat.lip_tint, cavity=not self.own_mouth, animal=animal)
        self.mouth_prog = ctx.program(vertex_shader=MOUTH_VERTEX_SHADER,
                                      fragment_shader=MOUTH_FRAGMENT_SHADER)
        self.mouth_prog["clip"].value = not self.own_mouth
        # The lining of the mouth takes on some of the lips' colour, as the gums do.
        lining = np.multiply(OWN_LINING, 1 + LINING_TINT * (np.asarray(flat.lip_tint, float) - 1))
        for prog in (self.prog, self.mouth_prog):  # the inside of the mouth is lit alike (MOUTH_LIGHT)
            prog["key"].value = tuple(LIGHT @ FLIP)
            prog["opening_half"].value = INSIDE_OPENING
            prog["inside_light"].value = (INSIDE_AMBIENT, INSIDE_KEY, INSIDE_SHEEN, INSIDE_BOUNCE)
            prog["bounce"].value = tuple(lining / lining.max())  # its hue, at full brightness
            prog["bounce_gape"].value = INSIDE_BOUNCE_GAPE
        if self.own_mouth:  # the depth of the head's outside, for the mouth to stay within
            self.skin = ctx.depth_texture((self.out_w, self.out_h))
            self.skin.compare_func = ""
            self.skin_fbo = ctx.framebuffer(depth_attachment=self.skin)
            self.mouth_prog["skin"].value = 1
            self.prog["skin"].value = 1
            self.on_jaw = self._part(np.nonzero(self.jaw_weight[:, 0] > 0.5)[0])
            # Its inside is the mouth cavity.
            self.prog["lining"].value = tuple(lining)
            self.prog["throat"].value = OWN_THROAT
            self.prog["throat_depth"].value = OWN_THROAT_DEPTH
            self.prog["pigment"].value = PIGMENT.get(animal, 0.0)
            self.prog["cavity_centre"].value, self.prog["cavity_radii"].value = OWN_CAVITY
            self.prog["cavity_floor"].value = OWN_FLOOR
            self.prog["focal"].value = (self.proj[0, 0], self.proj[1, 1])
            self.prog["screen"].value = (self.out_w, self.out_h)
            self.prog["rim"].value = LIPS_SHADOW * width * px_per_unit
        if self.lip_cut is not None and len(self.lip_cut):
            self.cut = self._part(self.lip_cut.ravel())
        self.mouth_prog["proj"].write(self.proj.astype("f4").T.tobytes())
        self.mouth_prog["shift"].value = tuple(self.shift)
        data = np.column_stack([parts.pos, parts.normal, parts.colour, parts.jaw, parts.light]).astype("f4")
        self.mouth_vao = ctx.vertex_array(
            self.mouth_prog,
            [(ctx.buffer(data.tobytes()), "3f 3f 3f 1f 1f", "in_pos", "in_normal", "in_colour", "in_jaw",
              "in_light")],
            ctx.buffer(parts.faces.astype("i4").tobytes()))

        # Blinks: the eye areas were painted from the portrait, so they repaint from its
        # eyelid-drawn versions. Only the texels taken from where the lids are drawn can
        # change, so only those are repainted, in the texture as uploaded (bottom row
        # first, as OpenGL has it, and with a 4th channel, which uploads several times
        # faster), and only the rows they lie in go up again.
        self.base = cv2.imread(str(texture_file))
        self.painted = np.ascontiguousarray(cv2.cvtColor(self.base, cv2.COLOR_BGR2BGRA)[::-1])
        tex_h, tex_w = self.base.shape[:2]
        eye_index, eye_px = mesh["eye_index"], mesh["eye_px"].astype(np.float32)
        if lids is None:
            lids = (0, 0, *sprite.shape[1::-1])
        x0, y0, x1, y1 = lids
        lo, hi = np.array([x0, y0]) - LID_REACH, np.array([x1, y1]) + LID_REACH
        lid = np.all((eye_px >= lo) & (eye_px < hi), axis=1)
        rows, cols = np.divmod(eye_index[lid], tex_w)
        self.lid_texels = (tex_h - 1 - rows) * tex_w + cols
        # The portrait around the lids, and where in it each of their texels comes from.
        h, w = sprite.shape[:2]
        crop = max(x0 - 2 * LID_REACH, 0), max(y0 - 2 * LID_REACH, 0)
        self.lid_crop = np.s_[crop[1]:min(y1 + 2 * LID_REACH, h), crop[0]:min(x1 + 2 * LID_REACH, w), :3]
        self.lid_px = eye_px[lid] - np.array(crop, np.float32)
        self.lid_weight = mesh["eye_weight"][lid, None]
        self.lid_open = sample(sprite[..., :3].astype(np.float32), eye_px[lid])
        self.lid_base = self.base.reshape(-1, 3)[eye_index[lid]]
        self.lid_rows = (tex_h - 1 - rows.max(), tex_h - rows.min()) if lid.any() else None
        self.texture = ctx.texture((tex_w, tex_h), 4, self.painted)  # BGR(A) like the rest of the app
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.anisotropy = 8.0
        self.texture.build_mipmaps()
        self._uploaded = sprite

    def face_on_screen(self):
        """Screen position of the cat's face center and its width in screen px."""
        return self.face_centre, self.face_width

    def _upload(self, sprite: np.ndarray):
        if self.lid_rows is not None:
            texels = self.painted.reshape(-1, 4)
            if sprite is self.open_sprite:
                texels[self.lid_texels, :3] = self.lid_base
            else:
                closed = sample(np.ascontiguousarray(sprite[self.lid_crop]), self.lid_px).astype(np.float32)
                eyes = self.lid_base + self.lid_weight * (closed - self.lid_open)
                texels[self.lid_texels, :3] = np.clip(eyes, 0, 255).astype(np.uint8)
            y0, y1 = self.lid_rows
            self.texture.write(self.painted[y0:y1], viewport=(0, y0, self.painted.shape[1], y1 - y0))
            self.texture.build_mipmaps()
        self._uploaded = sprite

    def _part(self, rows) -> "_Part":
        """What _shape needs to shape only the vertices at rows (an index array, or
        slice(None) for all): with any of them on the jaw's side of the lip cut, the face
        side's twins they are tied to as well."""
        face = chin = np.zeros(0, int)
        if self.lip_cut is not None and len(self.lip_cut) and self.jaw_weight is not None:
            face, chin = self.lip_cut.T
        if isinstance(rows, slice):
            return _Part(rows, self.follow, self.ears, face, chin, rows)
        tied = np.isin(chin, rows)
        need = np.unique(np.concatenate([rows, face[tied]]))
        face, chin = face[tied], chin[tied]
        ears = [e._replace(weights=np.concatenate([e.weights[:self.face_count],
                                                   e.weights[self.face_count:][need]]))
                for e in self.ears]
        at = lambda i: np.searchsorted(need, i)  # noqa: E731
        return _Part(need, self.follow[need], ears, at(face), at(chin), at(rows))

    def _shape(self, landmark_moves: np.ndarray, ears, part: "_Part | None" = None) -> np.ndarray:
        """Rest vertices (all, or a _part of them) moved by the landmarks' movement
        (portrait px) and the ear angles, and swung open with the jaw if it is marked.
        The vertex shader does the same for drawing; this is for the few the mouth needs."""
        part = self.everything if part is None else part
        rows, follow = part.rows, part.follow
        rest = self.rest[rows]
        moves = follow @ landmark_moves
        if any(ears):
            here = np.vstack([np.zeros((self.face_count, 2)), self.photo_px[rows] + moves[:, :2]])
            moves[:, :2] += (bend(here, part.ears, ears) - here)[self.face_count:]
        shaped = rest + moves @ self.to_model.T
        if self.jaw_weight is None:
            return shaped[part.back]
        # The jaw swings rigidly (the chin with it), but the skin on it still does what
        # your lips do beyond that (a smile pulling the corners up and out), as the face
        # does, so the two sides of the cut between the lips stay together.
        jaw = self._own_jaw(landmark_moves)
        beyond = follow @ self._beyond(landmark_moves, jaw)
        swung = (rest + beyond @ self.to_model.T) @ jaw[:3, :3].T + jaw[:3, 3]
        jaw_weight = self.jaw_weight[rows]
        shaped = shaped + jaw_weight * (swung - shaped)
        if len(part.face):
            # Where they were cut apart the lips part only as far as the jaw swings.
            face, chin = part.face, part.chin
            turn, upper = jaw[:3, :3], landmark_moves[LIP_MIDDLE[0]] @ self.to_model.T
            swung = shaped[face] @ turn.T + jaw[:3, 3] - upper @ turn.T
            shaped[chin] = shaped[face] + jaw_weight[chin] * (swung - shaped[face])
        return shaped[part.back]

    def _upper_lip_kept(self, landmark_moves: np.ndarray) -> np.ndarray:
        """landmark_moves (portrait px) with the upper lip kept clear of the nose (see
        UPPER_LIP_TRAVEL)."""
        nose = landmark_moves[NOSE[1], 1]
        moves = landmark_moves.copy()
        moves[UPPER_LIP, 1] = np.clip(moves[UPPER_LIP, 1], nose - self.lip_travel, nose + self.lip_travel)
        return moves

    def _with_the_jaw(self, landmark_moves: np.ndarray) -> np.ndarray:
        """landmark_moves (portrait px) with what is below the lips going where the jaw
        takes it: the chin as the jaw swings, the lower lip no lower than that. Your chin
        drops further than the jaw opens (OWN_GAPE_MAX, OWN_JAW_MAX); skin following it
        past there would tear away from the jaw, or slide down off it at the corners."""
        jaw = self._own_jaw(landmark_moves)
        swing = (self.landmarks @ jaw[:3, :3].T + jaw[:3, 3] - self.landmarks) @ self.to_px.T
        moves = landmark_moves.copy()
        moves[self.rigid] = swing[self.rigid]
        moves[LOWER_LIP, 1] = np.minimum(moves[LOWER_LIP, 1], swing[LOWER_LIP, 1])
        return moves

    def _beyond(self, landmark_moves: np.ndarray, jaw: np.ndarray) -> np.ndarray:
        """What the landmarks do beyond swinging with the head's own jaw (portrait px)."""
        swing = (self.landmarks @ jaw[:3, :3].T + jaw[:3, 3] - self.landmarks) @ self.to_px.T
        beyond = np.where(self.rigid[:, None], 0.0, landmark_moves - swing)
        beyond[:, 1] = np.minimum(beyond[:, 1], 0)  # dropping further than the jaw goes would only stretch it
        return beyond

    def _shape_on_gpu(self, landmark_moves: np.ndarray, ears):
        """Hands the vertex shader what it shapes the head with this frame (see _shape)."""
        jaw, beyond = np.eye(4), np.zeros_like(landmark_moves)
        if self.jaw_weight is not None:
            jaw = self._own_jaw(landmark_moves)
            beyond = self._beyond(landmark_moves, jaw)
            upper = landmark_moves[LIP_MIDDLE[0]] @ self.to_model.T
            self.prog["tie_shift"].value = tuple(jaw[:3, 3] - jaw[:3, :3] @ upper)
        self.moves.write(np.stack([landmark_moves, beyond]).astype("f4").tobytes())
        self.moves.use(2)
        hinges, turns = [(0.0, 0.0)] * 2, [0.0] * 2
        for i, ear in enumerate(self.ears):
            hinges[i] = tuple(ear.hinge)
            turns[i] = ear.side * ears[0 if ear.side < 0 else 1]
        self.prog["ear_hinge"].value = hinges
        self.prog["ear_turn"].value = turns
        self.prog["jaw"].write(jaw.astype("f4").T.tobytes())

    def _own_jaw(self, landmark_moves: np.ndarray) -> np.ndarray:
        """Model -> model (4x4) for the head's own jaw: turned about its hinge as far
        as your lower lip has dropped, and moved with the upper lip."""
        moved = landmark_moves[list(LIP_MIDDLE)] @ self.to_model.T
        drop = (moved[1] - moved[0]) @ self.mouth_axes[:, 1]  # the lower lip's, down is negative
        hinge = np.array(JAW_HINGE) * self.face_model
        gape = min(-drop, OWN_GAPE_MAX * self.mouth_width)
        angle = min(np.arcsin(np.clip(gape / np.linalg.norm(hinge[1:]), 0, 1)), OWN_JAW_MAX)
        c, s = np.cos(angle), np.sin(angle)
        turn = self.mouth_axes @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ self.mouth_axes.T
        pivot = self.mouth_origin + self.mouth_axes @ hinge
        jaw = np.eye(4)
        jaw[:3, :3] = turn
        jaw[:3, 3] = pivot - turn @ pivot + moved[0]
        return jaw

    def _mouth_matrices(self, landmark_moves: np.ndarray):
        """Mouth frame -> model, for the parts moving with the upper lip and with the jaw."""
        moved = landmark_moves[list(LIP_MIDDLE)] @ self.to_model.T
        follow = moved[0]  # everything moves with the upper lip
        drop = (moved[1] - moved[0]) @ self.mouth_axes[:, 1]  # the lower lip's, down is negative
        hinge = np.array(mouth3d.HINGE)
        reach = np.linalg.norm(hinge[1:]) * self.mouth_width  # hinge to the front teeth
        angle = min(np.arcsin(np.clip(-drop / reach, 0, 1)), JAW_MAX)
        skull = np.eye(4)
        skull[:3, :3] = self.mouth_axes * self.mouth_width
        skull[:3, 3] = self.mouth_origin + follow
        if self.jaw_weight is not None:  # the parts in the jaw go with the head's own
            at_rest = skull.copy()
            at_rest[:3, 3] = self.mouth_origin
            return skull, self._own_jaw(landmark_moves) @ at_rest
        c, s = np.cos(angle), np.sin(angle)
        swing = np.eye(4)
        swing[1:3, 1:3] = [[c, -s], [s, c]]
        to_hinge, back = np.eye(4), np.eye(4)
        to_hinge[:3, 3], back[:3, 3] = -hinge, hinge
        jaw = skull @ back @ swing @ to_hinge
        # Pinned to the lower lip: its front teeth stay right behind it however wide it opens.
        lower_lip = self.landmarks[LIP_MIDDLE[1]] + moved[1]
        jaw[:3, 3] += lower_lip - jaw[:3, 3]
        return skull, jaw

    def _mouth_box(self, jaw: np.ndarray, model_rot: np.ndarray):
        """Screen box (gl_FragCoord px: x0, y0, x1, y1) around the jaw (model points),
        reaching MOUTH_BOX of its size further each way."""
        clip = np.column_stack([jaw @ model_rot.T + self.moved, np.ones(len(jaw))]) @ self.proj.T
        screen = (clip[:, :2] / clip[:, 3:] + 1) / 2 * (self.out_w, self.out_h)
        lo, hi = screen.min(axis=0), screen.max(axis=0)
        grow = MOUTH_BOX * (hi - lo)
        return (*(lo - grow), *(hi + grow))

    def _lip_gap(self, pts: np.ndarray, model_rot: np.ndarray) -> list:
        """The opening between a head's own cut lips on screen (gl_FragCoord px): the
        face's side of the cut left to right, then the jaw's back, each smoothed into
        half of the opening's points. Closed, the two sides lie on each other.
        pts: the lip cut's vertices (lip_cut, raveled), shaped."""
        clip = np.column_stack([pts @ model_rot.T + self.moved, np.ones(len(pts))]) @ self.proj.T
        screen = (clip[:, :2] / clip[:, 3:] + 1) / 2 * (self.out_w, self.out_h)
        upper, lower = screen[0::2], screen[1::2]
        half = OPENING // 2
        bins = np.linspace(upper[:, 0].min(), upper[:, 0].max(), half + 1)
        centres = (bins[1:] + bins[:-1]) / 2

        def edge(side):
            which = np.clip(np.digitize(side[:, 0], bins) - 1, 0, half - 1)
            count = np.bincount(which, minlength=half)
            mean = np.bincount(which, side[:, 1], half) / np.maximum(count, 1)
            return np.interp(centres, centres[count > 0], mean[count > 0])

        top, bottom = edge(upper), edge(lower)
        return list(zip(centres, top)) + list(zip(centres[::-1], bottom[::-1]))

    def _draw_mouth(self, landmark_moves: np.ndarray, model_rot: np.ndarray):
        if not self.own_mouth:
            # The gap between the lips on screen: upper lip left to right, lower lip back.
            ring = UPPER_INNER_LIP + LOWER_INNER_LIP[::-1]
            lips = self.landmarks[ring] + self.lips_meet + landmark_moves[ring] @ self.to_model.T
            clip = np.column_stack([lips @ model_rot.T + self.moved, np.ones(len(ring))]) @ self.proj.T
            ndc = clip[:, :2] / clip[:, 3:]
            screen = (ndc + 1) / 2 * (self.out_w, self.out_h)  # gl_FragCoord: px from the bottom left
            opening = [tuple(p) for p in screen]
            self.prog["opening"].value = opening
            self.mouth_prog["opening"].value = opening
        skull, jaw = self._mouth_matrices(landmark_moves)
        self.mouth_prog["rot"].write(model_rot.astype("f4").T.tobytes())
        self.mouth_prog["skull"].write(skull.astype("f4").T.tobytes())
        self.mouth_prog["jaw"].write(jaw.astype("f4").T.tobytes())
        # Where the inside of the mouth is, for its light, and the cavity a head with its own
        # mouth sees its inside as: the mouth frame.
        view = np.eye(4)
        view[:3, :3], view[:3, 3] = model_rot.T, -model_rot.T @ self.moved
        from_skull = np.linalg.inv(skull)
        gape = max(-(from_skull @ jaw)[1, 3], 0.0)  # how far the jaw's front dropped
        for prog in (self.prog, self.mouth_prog):
            prog["to_mouth"].write((from_skull @ view).astype("f4").T.tobytes())
            prog["gape"].value = gape
        self.mouth_vao.render()

    def render(self, sprite: np.ndarray, offsets: np.ndarray, rotation: np.ndarray,
               ears=(0.0, 0.0), breath: float = 0.0) -> np.ndarray:
        """Draw the head with face offsets (478, 3) portrait px, head rotation (image
        axes, as for CatMesh), ear angles and breath (0..1, how full its lungs are,
        see catcam.idle); returns BGR. sprite: the portrait with its eyelids drawn
        as blinking (or the open-eyed one)."""
        if sprite is not self._uploaded:
            self._upload(sprite)
        landmark_moves = self.flat._steady_eyes(offsets)
        if self.jaw_weight is not None:
            landmark_moves = self._with_the_jaw(self._upper_lip_kept(landmark_moves))
        self._shape_on_gpu(landmark_moves, ears)
        model_rot = FLIP @ rotation @ FLIP
        self.prog["rot"].write(model_rot.astype("f4").T.tobytes())
        self.moved = self.shift + (0.0, self.lift * breath, 0.0)
        self.prog["shift"].value = tuple(self.moved)
        self.mouth_prog["shift"].value = tuple(self.moved)
        r = self.renderer
        r.ctx.enable(moderngl.DEPTH_TEST)
        r.ctx.disable(moderngl.BLEND)
        self.texture.use(0)
        if self.own_mouth:
            if self.lip_cut is not None and len(self.lip_cut):
                self.prog["opening"].value = self._lip_gap(self._shape(landmark_moves, ears, self.cut),
                                                           model_rot)
            self.skin_fbo.use()
            self.skin_fbo.clear(depth=1.0)
            r.ctx.enable(moderngl.CULL_FACE)
            self.prog["seam"].value = False
            self.prog["outside_only"].value = True
            self.vao.render()
            self.prog["outside_only"].value = False
            r.ctx.disable(moderngl.CULL_FACE)
            self.skin.use(1)
            self.prog["mouth_box"].value = self._mouth_box(self._shape(landmark_moves, ears, self.on_jaw),
                                                           model_rot)
        r.msaa.use()
        r.ctx.clear(*self.bg, depth=1.0)
        self._draw_mouth(landmark_moves, model_rot)
        self.prog["seam"].value = False
        self.vao.render()
        if self.seam_vao is not None:
            self.prog["seam"].value = True
            self.seam_vao.render()
        r.ctx.copy_framebuffer(r.resolved, r.msaa)
        return r.read()
