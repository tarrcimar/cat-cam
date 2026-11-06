"""Models the inside of a mouth in Blender, once for a cat and once for a dog:
teeth with real crowns, the gums they sit in, a palate with its ridges and a
tongue. No lips or skin: the heads have their own. Run it headless:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python tools/mouth_model.py -- [--preview DIR]

It writes assets/mouth/cat.npz and assets/mouth/dog.npz for catcam.mouth3d: per
vertex position, normal, colour, jaw weight, how much the cat's lip colour tints
it, and light (ambient occlusion from ray casts, darker deeper in the mouth).
--preview also renders each set from the front, the side and from below with
the jaw open, to judge the shapes.

Everything is built in catcam.mouth3d's mouth frame, in mouth widths: origin
where the lips meet in the middle, x toward the image-right corner, y up, z out
of the mouth. Blender's axes are used as they are, so "up" is y here.
"""

import argparse
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "mouth"

# The dental arch, in mouth widths: z of the front teeth, and z at x = ±ARCH_HALF on the
# way back (a parabola, continued past there for the cheek teeth). Wide, so the cheek teeth
# reach back toward the corners of the mouth as a cat's do, not huddle in its middle.
ARCH_FRONT, ARCH_BACK, ARCH_HALF = -0.075, -0.36, 0.48
LOWER_ARCH = (0.9, -0.03)  # the lower arch is this much narrower and sits this far back, inside the upper
# Where the teeth come out of the gums. Just past where the lips meet, so with the mouth
# open a strip of gum shows under the upper lip and above the lower one, the crowns whole
# below it, instead of only the tips of the teeth peeking out from behind the lips.
UPPER_GUM_Y, LOWER_GUM_Y = -0.014, -0.012
ROOT_DEPTH = 0.02  # teeth start this far inside the gum, so nothing shows where they come out
FIRST_GAP = 0.004  # between the two middle incisors

# Teeth on each side, from the middle back: (kind, width along the arch, thickness across
# it, crown length, gap before it), mouth widths. Cats: small incisors, a gap the lower
# fang fits in, rounded fangs about as far apart as the nose is wide, a gap, then
# blade-like premolars ending in the carnassial. Dogs: bigger incisors, the outer one nearly
# a little fang, stout rounded fangs and a row of spaced premolars. Friendly, not fierce:
# the fangs are long enough to read on a video call, not daggers.
TEETH = {
    "cat": {
        "upper": [("incisor", 0.036, 0.022, 0.045, 0), ("incisor", 0.038, 0.023, 0.048, 0.001),
                  ("incisor", 0.044, 0.028, 0.06, 0.002), ("canine", 0.07, 0.06, 0.14, 0.07),
                  ("premolar", 0.032, 0.022, 0.035, 0.035), ("premolar", 0.06, 0.027, 0.065, 0.006),
                  ("carnassial", 0.1, 0.032, 0.078, 0.006)],
        "lower": [("incisor", 0.03, 0.018, 0.036, 0), ("incisor", 0.031, 0.019, 0.039, 0.001),
                  ("incisor", 0.034, 0.021, 0.045, 0.001), ("canine", 0.06, 0.053, 0.112, 0.03),
                  ("premolar", 0.052, 0.023, 0.058, 0.045), ("premolar", 0.058, 0.026, 0.064, 0.006),
                  ("molar", 0.08, 0.028, 0.068, 0.006)],
    },
    "dog": {
        "upper": [("incisor", 0.038, 0.026, 0.056, 0), ("incisor", 0.04, 0.027, 0.06, 0.001),
                  ("incisor", 0.05, 0.036, 0.078, 0.003), ("canine", 0.08, 0.07, 0.155, 0.032),
                  ("premolar", 0.05, 0.026, 0.05, 0.03), ("premolar", 0.06, 0.028, 0.06, 0.012),
                  ("carnassial", 0.1, 0.034, 0.085, 0.01)],
        "lower": [("incisor", 0.032, 0.022, 0.05, 0), ("incisor", 0.033, 0.023, 0.053, 0.001),
                  ("incisor", 0.037, 0.026, 0.058, 0.001), ("canine", 0.07, 0.062, 0.13, 0.02),
                  ("premolar", 0.045, 0.024, 0.045, 0.035), ("premolar", 0.05, 0.025, 0.052, 0.012),
                  ("premolar", 0.056, 0.027, 0.058, 0.012)],
    },
}
CANINE_CURVE = {"cat": 0.08, "dog": 0.06}  # fang tips bend back into the mouth by this share of their length
# No two teeth alike: each tooth's length, width and shade differ from the table by up to
# these shares, left and right independently, so a row doesn't read as a picket fence.
VARY_LENGTH, VARY_WIDTH, VARY_SHADE = 0.08, 0.05, 0.04
VARY_LEAN = 0.1  # radians: and each leans to one side or the other along the arch by up to this

# Gums: the crest wraps each tooth this much beyond its thickness, and dips between teeth
# (and rises over each tooth's middle) by SCALLOP; heights of the band into the lip on the
# outside and toward the palate or floor inside.
GUM_COVER, SCALLOP = 0.014, 0.009
GUM_OUTSIDE = {False: 0.05, True: 0.03}  # upper, lower: the lower lip covers less of its gum
GUM_INSIDE = 0.035
# Across the band, from the lip side over the crest to the inside: (out, into, on the crest)
# as shares of (half its thickness, its height on that side, the crest's scallop).
GUM_SECTION = [(1.1, 1, 0), (1.15, 0.5, 0), (1.0, 0.12, 0.5), (0.7, 0, 1), (-0.7, 0, 1),
               (-1.05, 0.45, 0), (-1.25, 1, 0)]
GUM_PAST = 0.03  # the band runs on this far past the last tooth

# Palate: a vault inside the upper arch, from behind the incisors to PALATE_BACK, with
# transverse ridges (rugae) bowed back toward the sides, split by the midline.
PALATE_BACK = -0.85
PALATE_VAULT = 0.03  # how much higher its middle is than its edges
RUGAE = {"cat": 6, "dog": 7}  # how many
RUGA_SPACING, RUGA_RADIUS, RUGA_SWEEP = 0.055, 0.007, 0.9  # z step, thickness, how far back they bow

# Tongue: (tip z, back z, top y of its middle, its widest half width, thickness, how much
# its middle rises above its edges, groove depth and half width). It fills the lower arch,
# its tip just behind the lower incisors and its sides TONGUE_CLEAR inside the arch (the
# gums and teeth are between), up to its widest, narrowing by TONGUE_NARROW toward its
# root; from a thin spatula at the tip (TONGUE_TIP_THICK of its thickness) it thickens into
# its body. A dog's is longer, flatter and wider, with a deeper groove.
TONGUE = {
    "cat": (-0.112, -0.85, -0.045, 0.28, 0.11, 0.02, 0.013, 0.03),
    "dog": (-0.108, -0.9, -0.045, 0.32, 0.1, 0.014, 0.016, 0.04),
}
TONGUE_CLEAR, TONGUE_NARROW, TONGUE_TIP_THICK = 0.04, 0.3, 0.22
TONGUE_ROUND = 0.12  # the share of its length that rounds off into the tip
TONGUE_DIP = 0.008  # mouth widths: how much lower its tip lies than the top of its body
# Mouth widths: its back half curves down this far toward the throat, so it fades into the
# dark there instead of ending in an edge.
TONGUE_FALL = 0.12
TONGUE_RINGS, TONGUE_SIDES = 34, 36  # its mesh: cross-sections along it, points around each
# As the jaw drops the tongue doesn't just fall with it: its body rises and comes forward
# a little, staying with the skull by up to this share at its back (none at its tip, which
# stays behind the lower incisors).
TONGUE_LIFT = 0.12

# Colours (RGB 0..1), and how much the cat's lip colour tints each (0 = not at all, 1 = fully).
# Enamel is warmer where it comes out of the gum, ivory over the crown, and a little
# translucent (greyer) over its last TIP_GLASS share toward the tip.
ENAMEL_NECK, ENAMEL, ENAMEL_TIP = (0.9, 0.81, 0.68), (0.98, 0.96, 0.91), (0.88, 0.88, 0.87)
TIP_GLASS = 0.2
GUM, GUM_MARGIN = (0.7, 0.36, 0.41), (0.93, 0.62, 0.64)  # deep, and the pale pink crest at the teeth
MARGIN_BAND = 0.03  # mouth widths: how far from the crest the gum turns from pale to deep
# Dogs' gums are often pigmented: this share of them blotched this dark (RGB 0..1), the
# blotches about this big (mouth widths). The crest at the teeth stays pink.
GUM_PIGMENT = {"cat": 0.0, "dog": 0.6}
PIGMENT_COLOUR, PIGMENT_SIZE = (0.2, 0.14, 0.15), 0.05
PALATE, RUGA = (0.55, 0.27, 0.3), (0.72, 0.44, 0.45)
# A warm (salmon) pink, redder at the tip; the middle is paler where a cat's rough
# papillae are, speckled by PAPILLAE (shares of its brightness): cats' rough, dogs' smooth.
TONGUE_COLOUR = {"cat": (0.94, 0.5, 0.53), "dog": (0.95, 0.44, 0.5)}
TONGUE_TIP = {"cat": (0.97, 0.48, 0.52), "dog": (0.98, 0.44, 0.5)}
TONGUE_MIDDLE = {"cat": (0.93, 0.6, 0.62), "dog": (0.93, 0.5, 0.55)}
# Toward the throat it turns the colour of the mouth's inside (RGB 0..1; the cavity the
# head draws), over this span (mouth widths) around where it starts to fall away.
TONGUE_BACK, TONGUE_BACK_SPAN = (0.45, 0.14, 0.16), 0.4
PAPILLAE = {"cat": 0.16, "dog": 0.05}
SOFT_TINT = 0.5  # gums and palate take half the lip colour: it's strong
TONGUE_TINT = 0.3  # the tongue less: it is pinker than any lip

# Light: ambient occlusion from AO_RAYS rays over each vertex's hemisphere, blocked by
# anything within AO_REACH (mouth widths) in the same half of the mouth, and in the jaw by
# a floor under the tongue (FLOOR_Y; not drawn) so its edges and the gums' inside are dark;
# AO_POWER > 1 deepens the shadow in the creases. The throat end darkens by up to DEEP
# over DEEP_SPAN behind the front teeth. Teeth are shaded where they come out of the gum,
# by CROWN_SHADOW right there, fading over CROWN_SHADOW_SPAN (mouth widths).
AO_RAYS, AO_REACH, AO_FLOOR, AO_POWER = 96, 0.15, 0.25, 1.4
# Rays start this far off the surface (mouth widths): any closer and on a curved surface
# they hit the next row of faces, which streaks it.
AO_START = 0.002
FLOOR_Y = -0.16
DEEP, DEEP_SPAN = 0.7, 0.8
CROWN_SHADOW, CROWN_SHADOW_SPAN = 0.4, 0.009
# Teeth take only this share of their ambient occlusion: enamel is bright and lets light
# through, and small lower teeth by the tongue would otherwise turn grey.
ENAMEL_AO = 0.6


def _gauss(u, centre, width):
    return np.exp(-(((u - centre) / width) ** 2))


class Arch:
    """The dental arch: points, tangent and outward direction at a signed distance s
    along it from the middle (negative to the image-left)."""

    def __init__(self, lower: bool):
        scale, back = LOWER_ARCH if lower else (1.0, 0.0)
        x = np.linspace(0, 0.7, 2000)
        z = ARCH_FRONT + (ARCH_BACK - ARCH_FRONT) * (x / ARCH_HALF) ** 2
        self.x, self.z = x * scale, z + back
        self.s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.z)))])

    def point(self, s):
        s = np.asarray(s, float)
        a = np.abs(s)
        return np.stack([np.sign(s) * np.interp(a, self.s, self.x), np.zeros_like(s),
                         np.interp(a, self.s, self.z)], axis=-1)

    def frame(self, s):
        """(point, along (toward larger s), out (away from the mouth's inside))."""
        p = self.point(s)
        along = self.point(np.asarray(s) + 1e-4) - self.point(np.asarray(s) - 1e-4)
        along /= np.linalg.norm(along, axis=-1, keepdims=True)
        out = np.stack([-along[..., 2], np.zeros_like(along[..., 0]), along[..., 0]], axis=-1)
        return p, along, out


def place_teeth(animal: str, lower: bool):
    """Each tooth of one jaw as (kind, s of its middle, width, thickness, length, shade,
    lean), both sides, each a little different (VARY_*): the same every run."""
    rng = np.random.default_rng([list(TEETH).index(animal), int(lower)])
    teeth = []
    for side in (1, -1):
        s = FIRST_GAP / 2
        for kind, width, thick, length, gap in TEETH[animal]["lower" if lower else "upper"]:
            dl, dw, shade, lean = rng.uniform(-1, 1, 4) * (VARY_LENGTH, VARY_WIDTH, VARY_SHADE, VARY_LEAN)
            w = width * (1 + dw)
            s += gap + w / 2
            teeth.append((kind, side * s, w, thick * (1 + dw), length * (1 + dl), 1 + shade, lean))
            s += w / 2
    return teeth


def _grid_faces(rows: int, cols: int, wrap: bool = False):
    faces = []
    for r in range(rows - 1):
        for c in range(cols if wrap else cols - 1):
            a, b = r * cols + c, r * cols + (c + 1) % cols
            faces += [(a, b, a + cols), (b, b + cols, a + cols)]
    return faces


def _ring_tooth(c, along, out, up, width, thick, length, kind, curve):
    """An incisor (a chisel: flat across the arch, thinning to its edge) or a canine (a
    curved cone) from its rings of cross-sections, root in the gum."""
    sides = 10
    if kind == "canine":  # rings closer together toward the tip, so it rounds off smoothly
        ts = [-ROOT_DEPTH / length, 0, 0.15, 0.35, 0.55, 0.72, 0.84, 0.92, 0.97]
        # A rounded cone: it tapers, then closes in a blunt dome rather than a point.
        size = lambda t: (1 - 0.5 * t) * np.sqrt(max(1 - t ** 6, 0.0))  # noqa: E731
        across, power, bend = lambda t: 1.0, 2.0, curve  # noqa: E731
    else:
        ts = [-ROOT_DEPTH / length, 0, 0.4, 0.75, 0.92]
        size = lambda t: (1 + 0.08 * t) * np.sqrt(max(1 - t ** 6, 0.0)) * (1 - 0.15 * t)  # noqa: E731
        across, power, bend = lambda t: 1 - 0.6 * t, 3.0, 0.12  # noqa: E731
    angle = 2 * np.pi * np.arange(sides) / sides
    ca, sa = np.cos(angle), np.sin(angle)
    ring_a = np.sign(ca) * np.abs(ca) ** (2 / power)
    ring_b = np.sign(sa) * np.abs(sa) ** (2 / power)
    verts = []
    for t in ts:
        tc = max(t, 0)
        centre = c + up * length * t - out * bend * length * tc ** 2
        a, b = width / 2 * size(tc), thick / 2 * size(tc) * across(tc)
        verts += list(centre + np.outer(ring_a * a, along) + np.outer(ring_b * b, out))
    verts.append(c + up * length - out * bend * length)
    faces = _grid_faces(len(ts), sides, wrap=True)
    tip, last = len(verts) - 1, (len(ts) - 1) * sides
    faces += [(last + k, last + (k + 1) % sides, tip) for k in range(sides)]
    return np.array(verts), np.array(faces)


def _blade_profile(kind):
    """Crown height (share of its length) along a cheek tooth, u from -1 (front) to 1."""
    taper = lambda u: (1 - u ** 2) ** 0.4  # noqa: E731
    if kind == "carnassial":  # the upper shearing tooth: a small front cusp, a tall one, a blade
        return lambda u: taper(u) * (0.15 + np.maximum.reduce(
            [0.4 * _gauss(u, -0.62, 0.2), 0.85 * _gauss(u, -0.12, 0.3), 0.55 * _gauss(u, 0.5, 0.3)]))
    if kind == "molar":  # the lower one: two cusps with a notch between
        return lambda u: taper(u) * (0.15 + np.maximum(0.8 * _gauss(u, -0.35, 0.3),
                                                       0.85 * _gauss(u, 0.32, 0.3)))
    return lambda u: taper(u) * np.minimum(0.2 + 0.8 * _gauss(u, -0.1, 0.42) + 0.3 * _gauss(u, 0.6, 0.18), 1)


def _blade_tooth(c, along, out, up, width, thick, length, kind):
    """A cheek tooth: a blade along the arch, thin at its cusps, from slices across it."""
    us, phis = np.linspace(-1, 1, 9), np.linspace(0, np.pi, 7)
    height = length * _blade_profile(kind)(us)
    envelope = (1 - us ** 2) ** 0.35
    verts = [c + along * width / 2 * u + out * thick / 2 * e * np.cos(p)
             + up * (-ROOT_DEPTH + (h + ROOT_DEPTH) * np.sin(p) ** 0.7)
             for u, h, e in zip(us, height, envelope) for p in phis]
    return np.array(verts), np.array(_grid_faces(len(us), len(phis)))


def teeth(animal: str, lower: bool):
    """(verts, faces, base, up, length, shade) of every tooth in one jaw."""
    arch = Arch(lower)
    gum_y = LOWER_GUM_Y if lower else UPPER_GUM_Y
    out = []
    for kind, s, width, thick, length, shade, lean in place_teeth(animal, lower):
        point, along, outward = arch.frame(s)
        along = along * np.sign(s)  # from the middle toward the back, on either side
        base = point + [0, gum_y, 0]
        up = np.array([0, 1.0 if lower else -1.0, 0])
        if kind in ("incisor", "canine"):
            up = up * np.cos(lean) + along * np.sin(lean)
            v, f = _ring_tooth(base, along, outward, up, width, thick, length, kind, CANINE_CURVE[animal])
        else:
            v, f = _blade_tooth(base, along, outward, up, width, thick, length, kind)
        out.append((v, f, base, up, length, shade))
    return out


def gum(animal: str, lower: bool):
    """The band of gum along one arch the teeth come out of: a U-shaped section from the
    lip side over the crest to the inside, scalloped around each tooth."""
    arch = Arch(lower)
    placed = place_teeth(animal, lower)
    end = max(abs(s) + w / 2 for _, s, w, *_ in placed) + GUM_PAST
    stations = {0.0, end, -end}  # each tooth's middle and edges, so the scallops fall on teeth
    for _, s, w, *_ in placed:
        stations |= {s - w / 2, s, s + w / 2}
    ss = np.array(sorted(stations))
    near = [min(placed, key=lambda t: abs(t[1] - s)) for s in ss]
    half = np.array([t[3] / 2 for t in near]) + GUM_COVER
    crest = np.array([SCALLOP if abs(s - t[1]) < 1e-6 else -SCALLOP * 0.5 for s, t in zip(ss, near)])
    crest[[0, -1]] = 0
    into = np.array([0, -1.0 if lower else 1.0, 0])  # into the gum, away from the teeth's tips
    gum_y = LOWER_GUM_Y if lower else UPPER_GUM_Y
    verts, margin = [], []
    for s, h, m in zip(ss, half, crest):
        p, _, out = arch.frame(s)
        for o, height, on_crest in GUM_SECTION:
            side = GUM_OUTSIDE[lower] if o > 0 else GUM_INSIDE
            verts.append(p + [0, gum_y, 0] + out * o * h + into * (height * side + on_crest * m))
            margin.append(on_crest)
    faces = np.array(_grid_faces(len(ss), len(GUM_SECTION)))
    return np.array(verts), faces, np.array(margin, float), -into


def _inner_edge(gum_verts):
    """The upper gum's inner top edge on the right, from the back to the front."""
    edge = gum_verts[len(GUM_SECTION) - 1::len(GUM_SECTION)]
    edge = edge[edge[:, 0] >= 0]
    return edge[np.argsort(edge[:, 2])]


def palate_y(x, z, front, edge):
    """Height of the palate at (x, z): its edge on the gum, vaulting up in the middle
    once past the front teeth. edge: the gum's inner edge, from _inner_edge."""
    half = np.interp(z, edge[:, 2], edge[:, 0]) * 1.04
    across = np.clip(x / max(half, 1e-6), -1, 1)
    return UPPER_GUM_Y + GUM_INSIDE + 0.003 + PALATE_VAULT * (1 - across ** 2) * min((front - z) / 0.15, 1)


def palate(gum_verts):
    """A vault filling the upper arch from its inner gum edge back to the throat."""
    edge = _inner_edge(gum_verts)
    front = edge[-1, 2]
    rows = front + (PALATE_BACK - front) * np.linspace(0, 1, 13) ** 1.4
    verts = []
    for z in rows:
        half = np.interp(z, edge[:, 2], edge[:, 0]) * 1.04
        verts += [(x, palate_y(x, z, front, edge), z) for x in np.linspace(-half, half, 11)]
    return np.array(verts), np.array(_grid_faces(len(rows), 11))


def rugae(animal: str, gum_verts):
    """Ridges across the palate, as Blender curves with a round bevel, two per row (they
    part at the midline) and bowing back toward the sides."""
    edge = _inner_edge(gum_verts)
    front = edge[-1, 2]
    objects = []
    for k in range(RUGAE[animal]):
        z0 = front - 0.05 - k * RUGA_SPACING
        half = np.interp(z0, edge[:, 2], edge[:, 0])
        for side in (1, -1):
            xs = side * np.linspace(0.025, 0.85 * half, 6)
            zs = z0 - RUGA_SWEEP * xs ** 2
            ys = [palate_y(x, z, front, edge) - 0.3 * RUGA_RADIUS for x, z in zip(xs, zs)]
            curve = bpy.data.curves.new(f"ruga{k}{side}", "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth, curve.bevel_resolution, curve.use_fill_caps = RUGA_RADIUS, 0, True
            spline = curve.splines.new("POLY")
            spline.points.add(len(xs) - 1)
            for point, x, y, z in zip(spline.points, xs, ys, zs):
                point.co = (x, y, z, 1)
            # Thinner toward the sides, where they fade into the gum.
            for i, point in enumerate(spline.points):
                point.radius = 1.0 - 0.5 * i / (len(xs) - 1)
            ob = bpy.data.objects.new(curve.name, curve)
            bpy.context.collection.objects.link(ob)
            objects.append(ob)
    return objects


def tongue(animal: str):
    """The tongue, from cross-sections along it (u: 0 at the tip, 1 at the back): in plan
    it fills the lower arch behind a round tip; a thin spatula at the tip, thick in its
    body, its top domed with a groove down the middle and round at the edges."""
    tip, back, top, widest, thick, dome, groove, groove_half = TONGUE[animal]
    u = np.linspace(0, 1, TONGUE_RINGS)[:, None] ** 1.3  # closer together toward the tip, where it shows
    theta = 2 * np.pi * np.arange(TONGUE_SIDES) / TONGUE_SIDES
    c, s = np.cos(theta)[None], np.sin(theta)[None]
    # An ellipse around, a little fuller at the top and bottom: round edges, no walls.
    across, up = c, np.sign(s) * np.abs(s) ** 0.8
    # Closing round at both ends (the back is never seen, but closed it shades right).
    ends = (np.sqrt(1 - (1 - np.clip(u / TONGUE_ROUND, 0, 1)) ** 2)
            * np.sqrt(1 - np.clip((u - 0.9) / 0.1, 0, 1) ** 2))
    z = tip + (back - tip) * u
    arch = Arch(lower=True)
    inside = np.interp(z, arch.z[::-1], arch.x[::-1]) - TONGUE_CLEAR
    root = 1 - TONGUE_NARROW * np.clip((u - 0.5) / 0.5, 0, 1) ** 1.5
    half = np.minimum(inside, widest) * ends * root
    body = np.clip(u / 0.4, 0, 1)
    thickness = thick * (TONGUE_TIP_THICK + (1 - TONGUE_TIP_THICK) * body * (2 - body)) * np.sqrt(ends)
    x = half * across
    rel = x / np.maximum(half, 1e-6)
    top_side = np.clip(up, 0, 1)
    # Its top: the middle humps up behind the tip (which lies TONGUE_DIP lower), falling
    # away to the edges and a little toward the throat; the groove fades out at the tip.
    hump = np.clip(u / 0.35, 0, 1) ** 1.5 * (1 - 0.6 * np.clip((u - 0.6) / 0.4, 0, 1))
    y = top - thickness / 2 + thickness / 2 * up
    y += top_side * (dome * (1 - rel ** 2) * hump - TONGUE_DIP * (1 - hump))
    y -= top_side * groove * np.exp(-((x / groove_half) ** 2)) * np.clip(u / 0.15, 0, 1)
    y -= TONGUE_FALL * np.clip((u - 0.45) / 0.55, 0, 1) ** 2
    verts = np.stack([x, y, np.broadcast_to(z, x.shape)], axis=-1).reshape(-1, 3)
    faces = np.array(_grid_faces(TONGUE_RINGS, TONGUE_SIDES, wrap=True))
    middle = top - thick / 2
    faces = _orient(verts, faces, lambda f: f * [1, 1, 0] - [0, middle, 0])  # away from its middle line
    ob = _object("tongue", verts, faces)
    return ob


def _orient(verts, faces, want):
    """Flip all faces if on the whole their normals point away from want(face centres)."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    n = np.cross(b - a, c - a)
    return faces if (n * want((a + b + c) / 3)).sum() >= 0 else faces[:, ::-1]


def _object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-7)
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    return ob


def _arrays(ob):
    """Final vertices, smooth vertex normals and triangles of an object, modifiers and
    curve bevels applied."""
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = ob.evaluated_get(dg)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    n = len(mesh.vertices)
    verts, normals = np.zeros(n * 3), np.zeros(n * 3)
    mesh.vertices.foreach_get("co", verts)
    mesh.vertices.foreach_get("normal", normals)
    tris = np.zeros(len(mesh.loop_triangles) * 3, np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    evaluated.to_mesh_clear()
    return verts.reshape(-1, 3), normals.reshape(-1, 3), tris.reshape(-1, 3)


def _speckle(count):
    """Fixed speckle in -1..1, one per vertex. Random, not from the positions: a hash of
    those lines up with the mesh's regular rows into stripes."""
    return np.random.default_rng(count).uniform(-1, 1, count)


def _blotches(pos, size):
    """Smooth fixed blotches in 0..1, about size across, from a few crossed waves."""
    k = pos / size
    waves = (np.sin(1.7 * k[:, 0] + 0.3 * k[:, 1] + 2.1 * k[:, 2])
             + np.sin(-0.6 * k[:, 0] + 1.9 * k[:, 1] + 1.3 * k[:, 2] + 1.3)
             + np.sin(1.1 * k[:, 0] - 1.4 * k[:, 2] + 2.7))
    return (waves / 3 + 1) / 2


def _mix(a, b, t):
    """Per vertex blend of colours a and b (each RGB or per vertex) by t (0..1)."""
    t = np.asarray(t, float)[:, None]
    return (1 - t) * np.asarray(a, float) + t * np.asarray(b, float)


def enamel_colour(pos, base, up, length, shade):
    """Warm at the neck, ivory over the crown, a little glassy toward the tip."""
    h = np.clip((pos - base) @ up / length, 0, 1)
    colour = _mix(ENAMEL_NECK, ENAMEL, np.clip(h / 0.35, 0, 1) ** 0.8)
    colour = _mix(colour, ENAMEL_TIP, np.clip((h - 1 + TIP_GLASS) / TIP_GLASS, 0, 1) ** 1.5)
    return colour * shade


def gum_colour(animal, pos, gum_y):
    """Pale pink at the crest, deeper away from it; blotched dark if the animal's are."""
    t = np.clip(np.abs(pos[:, 1] - gum_y) / MARGIN_BAND, 0, 1)
    colour = _mix(GUM_MARGIN, GUM, t)
    if GUM_PIGMENT[animal]:
        b = _blotches(pos, PIGMENT_SIZE)
        threshold = 1 - GUM_PIGMENT[animal]
        dark = np.clip((b - threshold + 0.08) / 0.16, 0, 1) * np.clip(t * 6, 0, 1)
        colour = _mix(colour, PIGMENT_COLOUR, dark)
    return colour


def tongue_colour(animal, pos, n):
    """Redder toward the tip, paler and speckled with papillae over the middle of its top,
    dark where it falls away into the throat."""
    tip, back, *_ = TONGUE[animal]
    from_tip = np.clip((tip - pos[:, 2]) / 0.12, 0, 1)
    colour = _mix(TONGUE_TIP[animal], TONGUE_COLOUR[animal], from_tip)
    on_top = np.clip(n[:, 1] * 1.5, 0, 1)
    middle = np.exp(-((pos[:, 0] / 0.13) ** 2)) * on_top * from_tip
    colour = _mix(colour, TONGUE_MIDDLE[animal], 0.5 * middle)
    speckle = 1 + PAPILLAE[animal] * _speckle(len(pos)) * (0.4 + 0.6 * on_top)
    falls = tip + 0.45 * (back - tip)  # see tongue()
    colour = _mix(colour * speckle[:, None], TONGUE_BACK,
                  np.clip((falls + 0.5 * TONGUE_BACK_SPAN - pos[:, 2]) / TONGUE_BACK_SPAN, 0, 1))
    return colour


def tongue_lift(animal, pos):
    """The tongue's jaw weight: 1 at its tip, less toward its back (TONGUE_LIFT)."""
    tip, back, *_ = TONGUE[animal]
    return 1 - TONGUE_LIFT * np.clip((tip - pos[:, 2]) / (tip - back), 0, 1) ** 0.8


def build(animal: str):
    """All the parts of one set, as lists of per vertex arrays (pos, normal, colour, jaw,
    tint) and triangles, each part tagged with its half of the mouth."""
    parts = []

    def add(ob, colour, jaw, tint, crown=None):
        """jaw: a weight, or a function of the positions giving one per vertex. crown: for
        a tooth, (base, up): how high above the gum each vertex is is kept for its light."""
        v, n, f = _arrays(ob)
        weight = jaw(v) if callable(jaw) else np.full(len(v), float(jaw))
        height = np.full(len(v), np.inf) if crown is None else (v - crown[0]) @ crown[1]
        parts.append(dict(pos=v, normal=n, faces=f, colour=colour(v, n), jaw=weight,
                          tint=np.full(len(v), tint), crown=height))

    for lower in (False, True):
        for i, (v, f, base, up, length, shade) in enumerate(teeth(animal, lower)):
            axis = base
            f = _orient(v, f, lambda c, axis=axis, up=up: (c - axis) - np.outer((c - axis) @ up, up))
            ob = _object(f"tooth{int(lower)}{i}", v, f)
            ob.data.shade_smooth()
            add(ob, lambda pos, n, b=base, u=up, ln=length, sh=shade: enamel_colour(pos, b, u, ln, sh),
                lower, 0.0, crown=(base, up))

        v, f, margin, away = gum(animal, lower)
        f = _orient(v, f, lambda c, away=away: np.tile(away, (len(c), 1)))
        ob = _object(f"gum{int(lower)}", v, f)
        ob.data.shade_smooth()
        gum_y = LOWER_GUM_Y if lower else UPPER_GUM_Y
        add(ob, lambda pos, n, gy=gum_y: gum_colour(animal, pos, gy), lower, SOFT_TINT)
        if not lower:
            gum_verts = v
            pv, pf = palate(v)
            pf = _orient(pv, pf, lambda c: np.tile([0, -1.0, 0], (len(c), 1)))
            ob = _object("palate", pv, pf)
            ob.data.shade_smooth()
            add(ob, lambda pos, n: np.tile(PALATE, (len(pos), 1)), 0, SOFT_TINT)
            for ob in rugae(animal, gum_verts):
                add(ob, lambda pos, n: np.tile(RUGA, (len(pos), 1)), 0, SOFT_TINT)

    ob = tongue(animal)
    ob.data.shade_smooth()
    add(ob, lambda pos, n: tongue_colour(animal, pos, n), lambda pos: tongue_lift(animal, pos), TONGUE_TINT)
    return parts


def bake_light(parts):
    """Ambient occlusion per vertex by ray casts against its own half of the mouth (the
    jaw is open whenever any of it shows) and, in the jaw, a floor under the tongue;
    darker toward the throat, and on the teeth right by the gum."""
    k = np.arange(AO_RAYS) + 0.5
    r, a = np.sqrt(k / AO_RAYS), k * np.pi * (3 - np.sqrt(5))  # cosine-weighted, spread evenly
    local = np.column_stack([r * np.cos(a), r * np.sin(a), np.sqrt(1 - r ** 2)])
    floor_z = ARCH_FRONT + LOWER_ARCH[1]
    floor = ([(-0.7, FLOOR_Y, floor_z), (0.7, FLOOR_Y, floor_z), (0.7, FLOOR_Y, -1.2), (-0.7, FLOOR_Y, -1.2)],
             [(0, 1, 2, 3)])
    for on_jaw in (False, True):
        group = [p for p in parts if (p["jaw"].max() > 0.5) == on_jaw]
        offsets = np.cumsum([0] + [len(p["pos"]) for p in group])
        verts = [tuple(v) for p in group for v in p["pos"]]
        polygons = [tuple(int(i) for i in f + o) for p, o in zip(group, offsets) for f in p["faces"]]
        if on_jaw:
            polygons.append(tuple(len(verts) + i for i in floor[1][0]))
            verts += floor[0]
        tree = BVHTree.FromPolygons(verts, polygons)
        for p in group:
            light = np.empty(len(p["pos"]))
            for i, (pos, n) in enumerate(zip(p["pos"], p["normal"])):
                t1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
                t1 /= np.linalg.norm(t1)
                dirs = local @ np.array([t1, np.cross(n, t1), n])
                start = pos + n * AO_START
                hits = sum(tree.ray_cast(start, d, AO_REACH)[0] is not None for d in dirs)
                light[i] = AO_FLOOR + (1 - AO_FLOOR) * (1 - hits / AO_RAYS) ** AO_POWER
            enamel = np.isfinite(p["crown"])
            by_gum = 1 - CROWN_SHADOW * np.exp(-np.maximum(p["crown"], 0) / CROWN_SHADOW_SPAN)
            light = np.where(enamel, (1 - ENAMEL_AO * (1 - light)) * by_gum, light)
            deep = np.clip((ARCH_FRONT - p["pos"][:, 2]) / DEEP_SPAN, 0, 1)
            p["light"] = light * (1 - DEEP * deep ** 0.8)


def save(parts, path: Path):
    keys = ("pos", "normal", "colour", "jaw", "tint", "light")
    merged = {key: np.concatenate([p[key] for p in parts]) for key in keys}
    offsets = np.cumsum([0] + [len(p["pos"]) for p in parts[:-1]])
    faces = np.concatenate([p["faces"] + o for p, o in zip(parts, offsets)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pos=merged["pos"].astype("f4"), normal=merged["normal"].astype("f2"),
                        colour=np.round(np.clip(merged["colour"], 0, 1) * 255).astype("u1"),
                        jaw=merged["jaw"].astype("f2"), tint=merged["tint"].astype("f2"),
                        light=merged["light"].astype("f2"), faces=faces.astype("u2"))
    print(f"{path.name}: {len(merged['pos'])} vertices, {len(faces)} triangles, "
          f"{path.stat().st_size // 1024} KB")
    return merged, faces


def _clear():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for curve in list(bpy.data.curves):
        bpy.data.curves.remove(curve)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def preview(merged, faces, folder: Path, animal: str):
    """Renders the set as built (its colour times its light) from the front, the side,
    and from below in front with the jaw open, with Blender's workbench."""
    _clear()
    views = {  # name: (jaw angle, camera position, rotation (x, y), scale)
        "front": (0.3, (0, -0.1, 3), (0, 0), 0.9),
        "side": (0.3, (3, -0.15, -0.4), (0, np.pi / 2), 1.2),
        "below": (0.3, (0.9, -1.6, 2.4), (np.radians(30), np.radians(20)), 1.0),
    }
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = 900, 700
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "VERTEX"
    scene.view_settings.view_transform = "Standard"
    scene.world.color = (0.12, 0.12, 0.14)
    hinge = np.array([0.0, 0.1, -1.35])
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    for name, (angle, position, (rx, ry), scale) in views.items():
        c, s = np.cos(angle), np.sin(angle)
        turn = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        jaw = merged["jaw"][:, None]  # as the app does it: a blend of the closed and the swung
        pos = _mix(merged["pos"], (merged["pos"] - hinge) @ turn.T + hinge, merged["jaw"])
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([tuple(v) for v in pos], [], [tuple(int(i) for i in f) for f in faces])
        # Lit roughly like the app: its colour, its light, facing a light near the camera.
        normal = merged["normal"].astype(float)
        normal = (1 - jaw) * normal + jaw * normal @ turn.T
        facing = np.clip(normal @ (np.array(position) / np.linalg.norm(position)), 0, 1)
        shade = merged["colour"] * (merged["light"] * (0.55 + 0.45 * facing))[:, None]
        colours = mesh.color_attributes.new("colour", "FLOAT_COLOR", "POINT")
        colours.data.foreach_set("color", np.column_stack([np.clip(shade, 0, 1), np.ones(len(pos))]).ravel())
        mesh.color_attributes.active_color = colours
        ob = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(ob)
        camera.location, camera.rotation_euler = position, (rx, ry, 0)
        camera.data.ortho_scale = scale
        scene.render.filepath = str(folder / f"{animal}_{name}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(ob)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preview", type=Path, help="also render the sets into this folder")
    args = parser.parse_args(argv)
    for animal in TEETH:
        _clear()
        parts = build(animal)
        bake_light(parts)
        merged, faces = save(parts, OUT / f"{animal}.npz")
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            preview(merged, faces, args.preview, animal)


if __name__ == "__main__":
    main()
