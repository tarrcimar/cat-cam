"""Cat face filter for Zoom.

Pick a cat, then it copies you live: it turns, nods and tilts with your head,
copies your whole face (brows, eyes, cheeks, lips, jaw) and your hands become
paws (3D or photo ones), or stay your own hands, tinted like its fur. Only the
rendered cat goes out to the virtual camera, and in the "Hands" mode your hands
cut out of the webcam picture; no other webcam pixels are ever sent, so your
face is never visible.

Run:  uv run cat-cam      then pick "OBS Virtual Camera" in Zoom.
      uv run cat-cam --camera 1   to use another webcam (listed at the start).
Keys in the preview window: c = change cat, n = "this is my neutral face"
(recalibrate), q = quit; the switches' keys are shown next to them (h: 3D paws,
photo paws or your real hands).
"""

import argparse
import gc
import os
import platform
import plistlib
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision

from catcam import cats as cat_assets
from catcam import glass, paw3d
from catcam.background import Background, Chooser, behind
from catcam.blackhole import VoiceSetup
from catcam.cat_face import EYES, CatFace
from catcam.cat_mesh import CatMesh, Renderer
from catcam.ears import EarMotion
from catcam.face_motion import ExpressionTransfer, RestingScores
from catcam.fx import Effects, terminal_name
from catcam.fx import load as load_effects
from catcam.head3d import Head3D, animal, has_head
from catcam.head_pose import (
    OneEuroFilter,
    facing_camera,
    relative_rotation,
    rotation_of,
    to_matrix,
    to_vector,
)
from catcam.idle import Idle
from catcam.paths import FACE_MODEL, HAND_MODEL, LOG_FILE
from catcam.paw3d import PawColours, paw_colours
from catcam.paws import Paws, paw_set, paw_style
from catcam.picker import SIZE as PREVIEW_SIZE
from catcam.picker import Picker
from catcam.reactions import Reactions, reaction_for
from catcam.real_hands import RealHands, Segmenter, cut_all
from catcam.tail import Tail, TailHand, TailSkin, tail_skin
from catcam.voice import Sounds, VirtualMic, Voice
from catcam.webcams import choose, webcams

OUT_W, OUT_H, FPS = 1280, 720, 30
WINDOW = "Cat Cam"
# Switches in a glass pill in the preview's top left corner: (name, label, key).
# While "Tail" is on, a hand alone in view lifts a tail instead of being a paw
# (two hands are two paws); "3D head" shows the cat's 3D head, if it has one;
# "Background" asks for a picture or video to put behind the animal; "Voice"
# (on to begin with) turns your words into the animal's sounds, off lets the
# call hear your own voice. "Paws" is a three-way choice instead of a switch: a click
# or its key goes on to the next of PAW_MODES, its label says which it is.
TOGGLES = (("tail", "Tail", "t"), ("paws", "Paws", "h"), ("head3d", "3D head", "d"),
           ("background", "Background", "b"), ("voice", "Voice", "v"))
# (mode, label): 3D paws, the photo paws (or drawn ones, for a cat without photos), or
# your own hands cut out of the webcam picture (see catcam.real_hands). Cats without 3D
# paws leave out the first.
PAW_MODES = (("3d", "3D paws"), ("photo", "Photo paws"), ("hands", "Real hands"))
# Keys shown in the bar in the bottom left corner (the switches show their own).
KEY_HINTS = (("c", "Change cat"), ("1-8", "Reactions"), ("n", "Neutral face"), ("q", "Quit"))
# The preview's controls are small and keep to the corners, so the animal has
# the window: the switches top left, your animal's name and the keys bottom
# left, the voice setup (catcam.blackhole) bottom right and "Looking for your
# face" top right. All in preview px: the window is a fixed 960 x 540
# (catcam.picker.SIZE), so these are fixed shares of it.
MARGIN = 10  # the controls' distance from the window's edges
PILL_H = 30  # height of the switches' pill (each switch is clickable over its full height)
LABEL = 12  # text size of the switches' labels and of the corner notes
KEYCAP = (15, 10)  # height of a key, and its letter's size
BAR_H = 26  # height of the bar with the name and keys, and of the corner notes
NAME, HINT = 13, 11  # text sizes of your animal's name and of what the keys do

EXPRESSION_GAIN = 1.3  # exaggerate your expressions on the cat (1 = 1:1)
# The cat's lids close as fast as your eyes (a blink can be a single frame)
# but open again over this many s, which also softens tracking flicker.
BLINK_OPEN = 0.06
# One Euro filters (see catcam.head_pose): lower MIN_CUTOFF (Hz) = less jitter
# when still, higher BETA = less lag when moving fast; moving slower than FLOOR
# is taken for tracking noise. Head speeds are in radians per second; expression
# speeds in face widths per second, for each face point on its own. Tuned with
# tools/tracking_eval.py.
ROTATION_MIN_CUTOFF, ROTATION_BETA, ROTATION_FLOOR = 0.1, 12.0, 0.15
EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA, EXPRESSION_FLOOR = 0.7, 20.0, 0.06
SPEED_CUTOFF = (3.0, 2.0)  # Hz, how quickly the head's and the face's speed estimates follow

# How far an eye's blink score rises above its resting score (see
# catcam.face_motion.RestingScores), mapped to 0..1 eye closedness...
BLINK_RANGE = (0.2, 0.45)
# ...and, as some people's blink score stays low, also how much the eye
# opening shrank compared to your neutral face.
SHUT_RANGE = (0.5, 0.9)
# The neutral face only adapts once none of these has been more than RELAXED_BELOW
# above its resting score (the mouth's: MOUTH_RELAXED_BELOW) for CALM_FOR s, and
# you are not talking. Between syllables your mouth is often still part open, and
# taken for your neutral face that made the animal's mouth open less as you talked.
EXPRESSIVE = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
    "mouthStretchLeft", "mouthStretchRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "eyeBlinkLeft", "eyeBlinkRight", "cheekPuff",
]
RELAXED_BELOW = 0.25
MOUTH = ("jawOpen", "mouthFunnel", "mouthPucker")  # the expressive scores your mouth opening moves
MOUTH_RELAXED_BELOW = 0.06
CALM_FOR = 0.6  # s
# How sure the face detector must be to (re)find a face. MediaPipe's 0.5 lets a
# face partly out of the picture or in a dim room go unfound for seconds.
FACE_DETECTION_CONFIDENCE = 0.3
# Losing your face (a hand in front of it, turning away, bad light), the
# animal keeps its last pose for KEEP_POSE s, then relaxes toward rest (facing
# the viewer, eyes open, a neutral face) over about RELAX s. When your face is
# back it eases from where it was to you over EASE_BACK s instead of jumping.
KEEP_POSE, RELAX, EASE_BACK = 0.5, 1.0, 0.3
NO_FACE_HINT_AFTER = 1.0  # s without a face before the preview says so
WEBCAM_TIMEOUT = 5.0  # s of failed webcam reads before giving up
FIRST_FRAME_TIMEOUT = 60.0
STOP_TIMEOUT = 2.0  # s to wait for the tracker's thread to finish its frame when quitting
ENTER, ESC = 13, 27


@dataclass(frozen=True)
class Cat:
    """Everything needed to draw one cat."""
    name: str
    points: np.ndarray
    eyelids: CatFace
    mesh: CatMesh | Head3D
    head3d: Head3D | None  # the real 3D head, if this cat has one
    paw: object  # 3D paws (catcam.paw3d), or else {(view, hand): PawSprite} (catcam.paws.paw_set)
    photo_paws: dict  # {(view, hand): PawSprite}, the photo paws (drawn ones without photos)
    colours: PawColours  # its fur's colours, which tint your real hands
    tail: TailSkin | None
    sounds: Sounds  # what your syllables become, see catcam.voice; empty = your own voice
    sprite: np.ndarray
    background: np.ndarray


def load_cats(renderer: Renderer) -> list[Cat]:
    names = cat_assets.available()
    missing = cat_assets.unprepared()
    if missing:
        print(f"Not prepared yet (skipped): {', '.join(missing)}. "
              "Run: uv run --group prepare python -m catcam.prepare")
    if not names:
        sys.exit("No cats found. Put portraits in all_cats/ and run: "
                 "uv run --group prepare python -m catcam.prepare")
    cats = []
    for name in names:
        a = cat_assets.load(name)
        mesh = CatMesh(renderer, a.sprite, a.points, a.background, snap=not a.mouth_set)
        eyelids = CatFace(a.sprite, a.points[:, :2])
        photo_paws = paw_set(a.paws, paw_style(a.sprite, a.points))
        cats.append(Cat(
            name=name,
            points=a.points,
            eyelids=eyelids,
            mesh=mesh,
            head3d=Head3D(renderer, name, mesh, a.sprite, a.points, a.background, eyelids.lid_box())
            if has_head(name) else None,
            paw=paw3d.for_animal(renderer, animal(name), a.sprite, a.points) or photo_paws,
            photo_paws=photo_paws,
            colours=paw_colours(a.sprite, a.points),
            tail=tail_skin(a.paws),
            sounds=a.sounds,
            sprite=a.sprite,
            background=a.background,
        ))
    return cats


def shown(cat: Cat, head3d: bool) -> Cat:
    """The cat as drawn: with its 3D head while that switch is on, if it has one."""
    return replace(cat, mesh=cat.head3d) if head3d and cat.head3d is not None else cat


def landmarks_px(landmarks, width, height):
    """(478, 3) pixel coords; z uses the x scale, as MediaPipe defines it."""
    return np.array([(p.x * width, p.y * height, p.z * width) for p in landmarks])


def ramp(value, lo, hi):
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def eye_openness(face: np.ndarray) -> np.ndarray:
    """Height of each eye's opening relative to its width (image-left, image-right)."""
    openness = []
    for upper, lower in EYES:
        middle = slice(2, -2)  # skip the corners
        height = np.mean(face[lower[middle], 1] - face[upper[middle], 1])
        openness.append(height / (np.linalg.norm(face[upper[0], :2] - face[upper[-1], :2]) + 1e-6))
    return np.array(openness)


def blinks(scores, face=None, neutral=None, rest=None):
    """Eye closedness (image-left, image-right), each 0..1; rest: the resting
    scores (see catcam.face_motion.RestingScores), else 0."""
    rest = rest or {}
    # MediaPipe names eyes from the person's view: their right eye is on the
    # image's left.
    closed = np.array([ramp(scores[name] - rest.get(name, 0.0), *BLINK_RANGE)
                       for name in ("eyeBlinkRight", "eyeBlinkLeft")])
    if face is not None and neutral is not None:
        shrunk = 1 - eye_openness(face) / np.maximum(eye_openness(neutral), 1e-6)
        closed = np.maximum(closed, [ramp(v, *SHUT_RANGE) for v in shrunk])
    return closed


class FaceState:
    """Your smoothed head pose and expression, independent of the cat."""

    def __init__(self):
        self.rotation_filter = OneEuroFilter(ROTATION_MIN_CUTOFF, ROTATION_BETA, SPEED_CUTOFF[0],
                                             ROTATION_FLOOR)
        self.expression_filter = OneEuroFilter(EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA, SPEED_CUTOFF[1],
                                               EXPRESSION_FLOOR, per_point=True)
        self.transfer = ExpressionTransfer()
        self.resting = RestingScores()
        self.calm_for = 0.0  # s your face has been calm and quiet for (see CALM_FOR)
        self.neutral_pose = None  # the pose that faces the viewer; None = looking into the camera
        self.take_pose = False
        self.head = np.zeros(3)  # rotation vector; 0 = facing the viewer
        self.blink = np.zeros(2)
        self.ear_motion = EarMotion()
        self.ears = np.zeros(2)  # ear angles, see catcam.ears
        self.idle = Idle()  # what the animal does on its own while you are still
        self.offsets = np.zeros((478, 3))  # in face widths
        self.face = None  # last seen face landmarks in px
        self.last_seen = None
        self.missing_for = 0.0  # s since the face was last seen
        self.easing = None  # (head, offsets, blink, s since) shown when the face came back
        # The last frame's head, offsets and blink before smoothing (for tools/tracking_eval.py).
        self.measured = (self.head, self.offsets, self.blink)

    def reset_neutral(self):
        """Your next face and pose become the neutral ones."""
        self.transfer.reset()
        self.take_pose = True

    def update(self, result, cam_size, dt: float, now: float, talking: bool = False):
        """result: the face tracker's; talking: whether you are speaking (see catcam.voice)."""
        if not result.face_landmarks:
            self._without_face(dt)
            self.ears = self.ear_motion.update(None, dt, now)
            self.idle.update(self.head, None, talking, dt, now)
            return
        if self.missing_for > 0 and self.last_seen is not None:
            # Back after a while: start the filters afresh and ease over from what was shown.
            self.rotation_filter.reset()
            self.expression_filter.reset()
            self.easing = (self.head, self.offsets, self.blink, 0.0)
        self.missing_for = 0.0
        self.face = landmarks_px(result.face_landmarks[0], *cam_size)
        self.last_seen = now
        matrix = result.facial_transformation_matrixes[0]
        pose = rotation_of(matrix)
        if self.take_pose:
            # Your pose when you asked for it (e.g. looking down at the screen
            # past the camera) is "facing the viewer" for the cat from now on.
            self.neutral_pose, self.take_pose = pose, False
        reference = facing_camera(matrix) if self.neutral_pose is None else self.neutral_pose
        head = to_vector(relative_rotation(pose, reference))
        self.head = self.rotation_filter(head, dt)
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        self.ears = self.ear_motion.update(scores, dt, now)
        rest = self.resting.update(scores, dt)
        closed = blinks(scores, self.face, self.transfer.neutral, rest)
        opening = 1 - np.exp(-dt / BLINK_OPEN)
        self.blink = np.where(closed > self.blink, closed, self.blink + opening * (closed - self.blink))
        calm = not talking and all(
            scores[name] - rest[name] < (MOUTH_RELAXED_BELOW if name in MOUTH else RELAXED_BELOW)
            for name in EXPRESSIVE)
        self.calm_for = self.calm_for + dt if calm else 0.0
        relaxed = self.calm_for >= CALM_FOR
        offsets = EXPRESSION_GAIN * self.transfer(self.face, relaxed)
        self.offsets = self.expression_filter(offsets, dt)
        self.measured = (head, offsets, closed)
        if self.easing is not None:
            self._ease_back(dt)
        self.idle.update(self.head, scores, talking, dt, now)

    def _without_face(self, dt: float):
        """Keep the last pose for a moment, then relax toward rest. The eyes open
        at once: lids left half shut while nothing is tracked look broken."""
        self.missing_for += dt
        self.blink = self.blink * np.exp(-dt / BLINK_OPEN)
        if self.missing_for > KEEP_POSE:
            keep = np.exp(-dt / RELAX)
            self.head, self.offsets = keep * self.head, keep * self.offsets

    def _ease_back(self, dt: float):
        """Blend from what was shown when the face came back to the live pose."""
        head, offsets, blink, since = self.easing
        since += dt
        t = min(since / EASE_BACK, 1.0)
        live = t * t * (3 - 2 * t)  # eases in and out
        self.head = head + live * (self.head - head)
        self.offsets = offsets + live * (self.offsets - offsets)
        self.blink = blink + live * (self.blink - blink)
        self.easing = (head, offsets, blink, since) if t < 1 else None


def has_paw3d(cat: Cat) -> bool:
    return not isinstance(cat.paw, dict)


def render(cat: Cat, state: FaceState, paws: Paws, tail: Tail, tail_hand: str | None,
           background: np.ndarray | None = None, paw_mode: str = "3d",
           real_hands: RealHands | None = None) -> np.ndarray:
    """tail_hand: the hand ("Left", "Right") that lifts the tail instead of
    being a paw, if any. The tail is drawn while it sinks away, too.
    background: a picture (BGR, output size) to put behind the animal.
    paw_mode: see PAW_MODES; "hands" draws real_hands' cut-outs in place of paws."""
    sprite = cat.eyelids.render(blink=tuple(state.blink))
    offsets = state.transfer.to_cat(state.offsets, cat.points)
    rotation = to_matrix(state.head + state.idle.head)
    renderer = cat.mesh.renderer
    renderer.keep_coverage = background is not None
    if isinstance(cat.mesh, Head3D):
        frame = cat.mesh.render(sprite, offsets, rotation, state.ears, state.idle.breath)
    else:
        frame = cat.mesh.render(sprite, offsets, rotation, state.ears)
    if background is not None:
        frame = behind(frame, renderer.coverage, background)
    hidden = ()
    if cat.tail is not None:
        frame = tail.draw(frame, cat.tail)
        hidden = (tail_hand,) if tail_hand else ()
    if paw_mode == "hands":
        return real_hands.draw(frame, paws.slots, paws.scale, hidden) if real_hands is not None else frame
    return paws.draw(frame, cat.photo_paws if paw_mode == "photo" else cat.paw, hidden=hidden)


def paw_modes(paw3d: bool = True) -> list[tuple[str, str]]:
    """The paw modes (mode, label) to choose from; paw3d: whether the cat has 3D paws."""
    return [(mode, label) for mode, label in PAW_MODES if paw3d or mode != "3d"]


def _label_width(name: str, label: str) -> int:
    """A switch's label's width; the paws' is as wide as its widest mode's, so it doesn't jump."""
    labels = [mode_label for _, mode_label in PAW_MODES] if name == "paws" else [label]
    return max(glass.text_width(text, LABEL, 500) for text in labels)


def toggle_rects(head3d: bool = True) -> dict[str, tuple[int, int, int, int]]:
    """Each switch's area in the pill (x, y, w, h, preview px); head3d: whether the
    cat has a 3D head, else its switch is left out."""
    rects, x = {}, MARGIN + 4
    for name, label, key in TOGGLES:
        if name == "head3d" and not head3d:
            continue
        w = 8 + glass.SWITCH[0] + 6 + _label_width(name, label) + 5 \
            + glass.keycap_width(key, *KEYCAP) + 8
        rects[name] = (x, MARGIN, w, PILL_H)
        x += w
    return rects


def toggle_at(x: int, y: int, head3d: bool) -> str | None:
    """The switch a click at preview px (x, y) lands on, if any."""
    for name, (rx, ry, w, h) in toggle_rects(head3d).items():
        if rx <= x < rx + w and ry <= y < ry + h:
            return name
    return None


def _paw_dots(view: np.ndarray, x: int, y: int, mode: str, paw3d: bool):
    """In the paws' switch's place: a dot per mode to choose from, the current one honey."""
    modes = [m for m, _ in paw_modes(paw3d)]
    w, h = glass.SWITCH
    gap = 9
    left = x + (w - gap * (len(modes) - 1)) // 2
    for i, m in enumerate(modes):
        glass.dot(view, left + i * gap, y + h // 2, 4 if m == mode else 2,
                  glass.HONEY if m == mode else glass.MUTED)


def _switches(view: np.ndarray, states: dict, paw3d: bool = True) -> tuple[int, int, int, int]:
    """The pill of switches in the top left corner; returns its (x, y, w, h). states: each
    switch's state, "paws" the paw mode (see PAW_MODES)."""
    rects = toggle_rects("head3d" in states)
    right = max(rx + w for rx, _, w, _ in rects.values()) + 4
    glass.panel(view, MARGIN, MARGIN, right - MARGIN, PILL_H, PILL_H // 2)
    cap_h, cap_font = KEYCAP
    for name, label, key in TOGGLES:
        if name not in rects:
            continue
        rx, ry, _, h = rects[name]
        if name == "paws":
            _paw_dots(view, rx + 8, ry + (h - glass.SWITCH[1]) // 2, states[name], paw3d)
            label = dict(PAW_MODES)[states[name]]
        else:
            glass.switch(view, rx + 8, ry + (h - glass.SWITCH[1]) // 2, states[name])
        tx = rx + 8 + glass.SWITCH[0] + 6
        glass.text(view, label, tx, ry + h / 2 + LABEL * 0.36, LABEL, weight=500)
        glass.keycap(view, tx + _label_width(name, label) + 5, ry + (h - cap_h) // 2, key,
                     cap_h, cap_font)
    return MARGIN, MARGIN, right - MARGIN, PILL_H


def _capsule(view: np.ndarray, name: str) -> tuple[int, int, int, int]:
    """The bar in the bottom left corner: who you are, and the keys; returns its (x, y, w, h)."""
    cap_h, cap_font = KEYCAP
    name_w = glass.text_width(name, NAME, 600, "rounded")
    hint_w = [glass.keycap_width(key, cap_h, cap_font) + 4 + glass.text_width(label, HINT)
              for key, label in KEY_HINTS]
    pad, gap = BAR_H // 2, 10  # room at the ends, and between the hints
    width = pad + name_w + gap + 1 + gap + sum(hint_w) + gap * (len(KEY_HINTS) - 1) + pad
    x, y = MARGIN, PREVIEW_SIZE[1] - MARGIN - BAR_H
    glass.panel(view, x, y, width, BAR_H, BAR_H // 2)
    glass.text(view, name, x + pad, y + BAR_H / 2 + NAME * 0.36, NAME, weight=600, family="rounded")
    divider = x + pad + name_w + gap
    glass.paint(view, np.ones((BAR_H - 12, 1), np.float32), divider, y + 6, glass.WHITE, 0.3)
    hx = divider + 1 + gap
    for (key, label), hw in zip(KEY_HINTS, hint_w):
        cap_w = glass.keycap(view, hx, y + (BAR_H - cap_h) // 2, key, cap_h, cap_font)
        glass.text(view, label, hx + cap_w + 4, y + BAR_H / 2 + HINT * 0.36, HINT, glass.MUTED)
        hx += hw + gap
    return x, y, width, BAR_H


def _looking(view: np.ndarray) -> tuple[int, int, int, int]:
    """"Looking for your face" in the top right corner, level with the switches;
    returns its (x, y, w, h)."""
    top = MARGIN + (PILL_H - BAR_H) // 2
    return glass.chip(view, "Looking for your face", PREVIEW_SIZE[0] - MARGIN, top, BAR_H, LABEL)


def _message(view: np.ndarray, lines: list[str], y: int):
    """A glass panel with lines of text, centred across the window at y."""
    w, _ = PREVIEW_SIZE
    width = max(glass.text_width(line, 15, 500) for line in lines) + 48
    height = 24 + 22 * len(lines)
    x = (w - width) // 2
    glass.panel(view, x, y, width, height, min(height // 2, 22))
    for i, line in enumerate(lines):
        glass.text(view, line, w / 2, y + 12 + 16 + 22 * i, 15, weight=500, align="center")


def preview(out: np.ndarray, cat: Cat, looking: bool, notice: list[str] | None = None,
            tail: bool = False, head3d: bool | None = None, background: bool = False,
            voice: bool = True, voice_setup: VoiceSetup | None = None, paw_mode: str = "3d",
            paw3d: bool = True) -> np.ndarray:
    """Mirrored like a mirror, with the switches and hints on glass, none of which
    reach the call. head3d: whether the 3D head is on; None leaves its switch out
    (the cat has none). voice_setup: the "Get the voice" button, in the bottom
    right corner, for a Mac without BlackHole. paw_mode: the paws shown (see PAW_MODES);
    paw3d: whether the cat has 3D paws to choose."""
    view = cv2.flip(cv2.resize(out, PREVIEW_SIZE, interpolation=cv2.INTER_AREA), 1)
    states = ({"tail": tail, "paws": paw_mode} | ({} if head3d is None else {"head3d": head3d})
              | {"background": background, "voice": voice})
    width, height = PREVIEW_SIZE
    _switches(view, states, paw3d)
    if voice_setup is not None:
        voice_setup.draw(view, width - MARGIN, height - MARGIN)
    _capsule(view, cat.name)
    if looking:
        _looking(view)
    if notice:
        _message(view, notice, PREVIEW_SIZE[1] // 2 - 40)
    return view


def create_trackers():
    video = vision.RunningMode.VIDEO
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=video,
        min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=video,
        num_hands=2,
    ))
    return face, hands


def pick_webcam(asked: int | None) -> int:
    """The webcam's number: the one asked for, else the best one there is. Lists them
    all, as the numbers to ask for and, in the app's log, to see which one was used."""
    cams = webcams()
    chosen = choose(cams)
    index = asked if asked is not None else chosen.index if chosen is not None else 0
    for cam in cams:
        print(f"{'*' if cam.index == index else ' '} camera {cam.index}: {cam.name} ({cam.model})")
    return index


def open_webcam(index: int):
    # macOS asks for camera permission asynchronously; keep retrying while the
    # permission dialog is open.
    for _ in range(30):
        webcam = cv2.VideoCapture(index)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, OUT_W)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, OUT_H)
            return webcam
        webcam.release()
        time.sleep(1)
    sys.exit(f"Cannot open webcam {index}. Grant camera access to {terminal_name()} in "
             "System Settings > Privacy & Security > Camera, or pick another one with --camera.")


def open_virtual_camera():
    try:
        return pyvirtualcam.Camera(OUT_W, OUT_H, FPS, fmt=pyvirtualcam.PixelFormat.BGR)
    except RuntimeError as e:
        sys.exit(f"{e}\n\nOne-time setup: open OBS, click 'Start Virtual Camera', "
                 "approve the system extension, click 'Stop Virtual Camera', quit OBS. "
                 "Keep OBS closed while this runs.")


def window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # macOS raises once the window is gone
        return True


class Mouse:
    """Collects mouse events from OpenCV's callback: clicks on the picker's
    cards, and where the last click was (for the preview's switches)."""

    def __init__(self, picker: Picker):
        self.picker = picker
        self.clicked = None
        self.position = None

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.picker.hover = self.picker.hit(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.clicked = self.picker.hit(x, y)
            self.position = (x, y)

    def take_click(self) -> int | None:
        """The card clicked since the last take, if any."""
        clicked, self.clicked, self.position = self.clicked, None, None
        return clicked

    def take_position(self) -> tuple[int, int] | None:
        """Where the last click since the last take was, if any."""
        position, self.clicked, self.position = self.position, None, None
        return position


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cat face filter for Zoom.")
    parser.add_argument("--camera", type=int, default=None, metavar="N",
                        help="webcam to use, by number (default: the Mac's own camera; "
                             "the numbers are listed at the start)")
    parser.add_argument("--mic", default=None, metavar="NAME",
                        help="microphone, by name or number (default: the system's; "
                             "list them: python -m catcam.voice --list)")
    parser.add_argument("--background", default=None, metavar="FILE",
                        help="a picture or video to put behind the animal")
    parser.add_argument("--check", action="store_true",
                        help="load and draw every cat once, without camera, then quit")
    return parser.parse_args(argv)


class Controls:
    """Which cat is streamed and whether the picker is open, driven by keys and clicks."""

    def __init__(self, picker: Picker, mouse: Mouse, voice_setup: VoiceSetup | None = None):
        self.picker, self.mouse = picker, mouse
        self.voice_setup = voice_setup  # the "Get the voice" button, if BlackHole isn't set up
        self.current, self.picking, self.picked_once = 0, True, False
        self.tail = False  # the tail switch
        self.head3d = True  # the 3D head switch
        self.background = False  # the background switch
        self.voice = True  # the voice switch: on = the animal's sounds, off = your own voice
        self.paws = "3d"  # the paws chosen, see PAW_MODES
        self.has_head3d = True  # whether the current cat has a 3D head (its switch shows)
        self.has_paw3d = True  # whether the current cat has 3D paws (that choice shows)
        self.reaction = None  # a reaction asked for by its number key, see take_reaction
        self.quit = False

    def paw_mode(self) -> str:
        """The paws shown: the ones chosen, the photo paws standing in for 3D paws the cat hasn't got."""
        return "photo" if self.paws == "3d" and not self.has_paw3d else self.paws

    def _next_paws(self):
        modes = [mode for mode, _ in paw_modes(self.has_paw3d)]
        self.paws = modes[(modes.index(self.paw_mode()) + 1) % len(modes)]

    def take_reaction(self) -> str | None:
        """The reaction asked for since the last take, if any."""
        reaction, self.reaction = self.reaction, None
        return reaction

    def handle(self, key: int, state: FaceState):
        """Number keys pick a cat in the picker, and are reactions in the call."""
        if key == ord("q"):
            self.quit = True
            return
        choice = None
        if self.picking:
            choice = self.picker.key_choice(key)
            if key == ENTER and self.picker.hover is not None:
                choice = self.picker.hover
            clicked = self.mouse.take_click()
            if clicked is not None:
                choice = clicked
            if key == ESC and self.picked_once:
                self.picking = False
        else:
            click = self.mouse.take_position()
            hit = toggle_at(*click, self.has_head3d) if click is not None else None
            if click is not None and self.voice_setup is not None:
                self.voice_setup.click(*click)
            if key == ord("h") or hit == "paws":
                self._next_paws()
            if key == ord("t") or hit == "tail":
                self.tail = not self.tail
            if key == ord("d") or hit == "head3d":
                self.head3d = not self.head3d
            if key == ord("b") or hit == "background":
                self.background = not self.background
            if key == ord("v") or hit == "voice":
                self.voice = not self.voice
            if key == ord("c"):
                self.picking = True
                self.picker.hover = None
            elif key == ord("n"):
                state.reset_neutral()
            self.reaction = reaction_for(key) or self.reaction
        if choice is not None:
            self.current, self.picking, self.picked_once = choice, False, True


def frames(webcam):
    """Yields (frame, time) from the webcam; exits if it stops sending."""
    # The first frame can take long: macOS may still be asking for permission.
    last_frame, timeout = time.monotonic(), FIRST_FRAME_TIMEOUT
    while True:
        ok, frame = webcam.read()
        now = time.monotonic()
        if ok:
            last_frame, timeout = now, WEBCAM_TIMEOUT
            yield frame, now
        elif now - last_frame > timeout:
            sys.exit("The webcam stopped sending images (is another app using it?).")
        else:
            time.sleep(0.01)


class Tracked(NamedTuple):
    """One webcam frame, tracked."""
    time: float  # when it was read (time.monotonic)
    size: tuple[int, int]  # the frame's width and height
    face: object  # MediaPipe's FaceLandmarkerResult
    hands: object  # and HandLandmarkerResult
    # Your hands cut out of the frame (catcam.real_hands.Cut or None, in the order of the
    # hand tracker's hands) while they are shown, else None.
    cuts: list | None = None


class Tracker:
    """Reads the webcam and finds your face and hands on a thread of its own, so one
    frame is tracked while the one before is drawn. Only the newest frame is kept:
    if drawing falls behind, frames are skipped rather than queued up.

    While cut_for is set (to the animal's colours), your hands are cut out of each
    frame too (see catcam.real_hands): the segmenter works on the frame alongside the
    face and hand trackers, so it adds little to the time a frame takes."""

    def __init__(self, webcam, trackers, segmenter: Segmenter | None = None):
        self.webcam = webcam
        self.face_tracker, self.hand_tracker = trackers
        self.segmenter = segmenter
        self.cut_for: PawColours | None = None
        self._ready = threading.Condition()
        self._newest: Tracked | None = None
        self._error: BaseException | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="tracker", daemon=True)

    def start(self) -> "Tracker":
        self._thread.start()
        return self

    def track(self, frame: np.ndarray, timestamp: int):
        """MediaPipe's face and hand results for a frame (BGR) at timestamp (ms)."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return (self.face_tracker.detect_for_video(image, timestamp),
                self.hand_tracker.detect_for_video(image, timestamp))

    def cut(self, frame: np.ndarray, segmenting: Future, hands, colours: PawColours) -> list:
        """The hands (a HandLandmarkerResult) cut out of frame (BGR), once the segmenter has
        found its skin (segmenting: see Segmenter.start)."""
        return cut_all(frame, segmenting.result(), hands, colours)

    def _run(self):
        start = time.monotonic()
        try:
            for frame, now in frames(self.webcam):
                if self._stopping:
                    return
                timestamp = int((now - start) * 1000)
                colours = self.cut_for  # read once: the main thread sets it
                segmenting = (self.segmenter.start(frame, timestamp)
                              if colours is not None and self.segmenter is not None else None)
                face, hands = self.track(frame, timestamp)
                cuts = self.cut(frame, segmenting, hands, colours) if segmenting is not None else None
                with self._ready:
                    self._newest = Tracked(now, (frame.shape[1], frame.shape[0]), face, hands, cuts)
                    self._ready.notify()
        except BaseException as e:  # e.g. the webcam stopping (SystemExit): the main thread raises it
            with self._ready:
                self._error = e
                self._ready.notify()

    def next(self) -> Tracked:
        """The newest tracked frame not handed out yet; waits for one."""
        with self._ready:
            while self._newest is None and self._error is None:
                self._ready.wait()
            if self._newest is None:
                raise self._error
            tracked, self._newest = self._newest, None
            return tracked

    def stop(self):
        self._stopping = True
        self._thread.join(timeout=STOP_TIMEOUT)
        if self.segmenter is not None and not self._thread.is_alive():
            self.segmenter.close()


def stream(cats: list[Cat], controls: Controls, effects: Effects, voice: Voice, tracker: Tracker, vcam,
           background: Background | None = None):
    state, paws, tail, tail_hands = FaceState(), Paws(OUT_W), Tail((OUT_W, OUT_H)), TailHand()
    chooser, reactions, real_hands = Chooser(), Reactions(), RealHands()
    controls.background = background is not None
    last = time.monotonic()
    notice_until = 0.0
    app = terminal_name()
    restart = "quit Cat Cam and open it again" if app == "Cat Cam" else f"quit and reopen {app}"
    notice = [f"Allow {app} under Input Monitoring (opened in System Settings),",
              f"then {restart}" + ("." if app == "Cat Cam" else " and start Cat Cam again.")]
    while True:
        tracked = tracker.next()
        now, cam_size, hands = tracked.time, tracked.size, tracked.hands
        dt, last = max(now - last, 1e-3), now
        state.update(tracked.face, cam_size, dt, now, voice.syllables.speaking)
        cat = shown(cats[controls.current], controls.head3d)
        controls.has_head3d, controls.has_paw3d = cat.head3d is not None, has_paw3d(cat)
        paw_mode = controls.paw_mode()
        # The next frames' hands are cut out only while they are shown.
        tracker.cut_for = cat.colours if paw_mode == "hands" else None
        voice.sounds = cat.sounds if controls.voice else Sounds()
        background = _follow_switch(controls, chooser, background)
        paws.update(hands, state.face, cam_size, dt, *cat.mesh.face_on_screen())
        real_hands.update(tracked.cuts, paws.slots)
        tail_hand = tail_hands.update(paws.slots) if controls.tail and cat.tail is not None else None
        hand = paws.slots.get(tail_hand, {}) if tail_hand else {}
        tail.update(hand.get("params"), hand.get("presence", 0.0), dt, *cat.mesh.face_on_screen())

        special = effects.update(now, cat.name)
        picture = background.frame(now) if background is not None else None
        out = special if special is not None else render(cat, state, paws, tail, tail_hand, picture,
                                                         paw_mode, real_hands)
        out = reactions.draw(out, now)
        vcam.send(out)
        if controls.picking:
            cv2.imshow(WINDOW, controls.picker.draw(controls.current if controls.picked_once else None, out))
        else:
            looking = state.last_seen is None or now - state.last_seen > NO_FACE_HINT_AFTER
            cv2.imshow(WINDOW, preview(out, cat, looking, notice if now < notice_until else None,
                                       tail=controls.tail,
                                       head3d=controls.head3d if cat.head3d is not None else None,
                                       background=controls.background, voice=controls.voice,
                                       voice_setup=controls.voice_setup, paw_mode=paw_mode,
                                       paw3d=controls.has_paw3d))
        key = cv2.waitKey(1) & 0xFF
        if effects.needs_permission(key, now):
            notice_until = now + 10
        controls.handle(key, state)
        if (reaction := controls.take_reaction()) is not None:
            reactions.play(reaction, now)
        if controls.quit or window_closed():
            return


def _follow_switch(controls: Controls, chooser: Chooser, background: Background | None) -> Background | None:
    """The background as the switch wants it: switched on, it asks for a picture or
    video (the switch goes back off if none is chosen); switched off, it goes."""
    if not controls.background:
        if background is not None:
            background.close()
        return None
    if background is not None:
        return background
    if chooser.idle:
        chooser.start()
        return None
    path = chooser.answer()
    if path is None:  # still choosing
        return None
    if path:
        try:
            return Background(path, (OUT_W, OUT_H))
        except ValueError as e:
            print(e)
    controls.background = False
    return None


def check(cats: list[Cat]):
    """Draws every cat once at rest, with paws, as a quick test of the install."""
    for cat in cats + [shown(c, True) for c in cats if c.head3d is not None]:
        out = render(cat, FaceState(), Paws(OUT_W), Tail((OUT_W, OUT_H)), tail_hand="Right")
        if out.shape != (OUT_H, OUT_W, 3) or not out.any():
            sys.exit(f"Drawing {cat.name} failed.")
    load_effects()  # its data file must load too
    Segmenter().close()  # and the segmenter's model, for real hands
    import sounddevice  # noqa: F401  (imported only when needed, so check here they're there)
    if sys.platform == "darwin":
        import AVFoundation  # noqa: F401
        import Quartz  # noqa: F401
    print(f"OK: {len(cats)} cat{'' if len(cats) == 1 else 's'}")


def alert(message: str):
    """A dialog for the standalone app, which has no terminal to print to."""
    quoted = message.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display alert "Cat Cam" message "{quoted}"'], check=False)


def log_to_file():
    """The standalone app has no terminal, so everything it prints goes to the log
    file instead, along with what its libraries print (OpenCV says there why a
    camera won't open). Each start begins a new log. Started from a terminal (as the
    build's --check is), it prints there as usual."""
    if os.isatty(2):
        return
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", buffering=1)  # stays open while the app runs
    for fd in (1, 2):  # the libraries' C code writes to these
        os.dup2(log.fileno(), fd)
    sys.stdout = sys.stderr = log
    info = Path(sys.executable).parents[1] / "Info.plist"  # Cat Cam.app/Contents/
    version = plistlib.loads(info.read_bytes()).get("CFBundleShortVersionString", "?")
    print(f"Cat Cam {version} on macOS {platform.mac_ver()[0]} ({platform.machine()}), "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")


def main(argv=None):
    """Runs Cat Cam; the standalone app shows errors in a dialog and logs everything."""
    if not getattr(sys, "frozen", False):
        return run(argv)
    log_to_file()
    try:
        return run(argv)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code)
            alert(f"{e.code}\n\nDetails are in {LOG_FILE}.")
        raise
    except Exception:
        traceback.print_exc()
        alert(f"Cat Cam stopped because of an error. Details are in {LOG_FILE}.")
        raise


def settle():
    """Everything loaded so far stays for good, so the garbage collector stops looking
    through it: with the 130K objects of the libraries and cats, each of its full
    collections otherwise stops a frame for 10-25 ms."""
    gc.collect()
    gc.freeze()


def run(argv=None):
    args = parse_args(argv)
    print("Loading cats...")
    renderer = Renderer((OUT_W, OUT_H))
    cats = load_cats(renderer)
    if args.check:
        return check(cats)
    picker = Picker([(c.name, c.sprite, c.background) for c in cats])
    trackers = create_trackers()
    vcam = open_virtual_camera()
    webcam = open_webcam(pick_webcam(args.camera))

    background = None
    if args.background:
        try:
            background = Background(args.background, (OUT_W, OUT_H))
        except ValueError as e:
            sys.exit(str(e))
    effects = Effects(load_effects(), (OUT_W, OUT_H))
    effects.start()
    voice = Voice()
    mic = VirtualMic(voice, args.mic)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    mouse = Mouse(picker)
    cv2.setMouseCallback(WINDOW, mouse)
    print(f"Streaming to '{vcam.device}'. Select it as the camera in Zoom.")
    if mic.start():
        print(f"Listening to '{mic.heard}', sending your voice to '{mic.device}'. "
              "Select it as the microphone in Zoom.")
    voice_setup = VoiceSetup.for_this_mac()
    print("Pick a cat in the window. Keep a relaxed face for a second at the start.")
    settle()
    tracker = Tracker(webcam, trackers, Segmenter()).start()
    try:
        with vcam:
            stream(cats, Controls(picker, mouse, voice_setup), effects, voice, tracker, vcam, background)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        mic.stop()
        effects.stop()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
