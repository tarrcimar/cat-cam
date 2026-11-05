"""Cat face filter for Zoom.

Pick a cat, then it copies you live: it turns, nods and tilts with your head,
copies your whole face (brows, eyes, cheeks, lips, jaw, tongue) and your hands
become paws. Only the rendered cat goes out to the virtual camera; no webcam
pixels are ever sent, so your face is never visible.

Run:  uv run cat-cam      then pick "OBS Virtual Camera" in Zoom.
      uv run cat-cam --camera 1   to use another webcam.
Keys in the preview window: c = change cat, n = "this is my neutral face"
(recalibrate), t = tail, q = quit.
"""

import argparse
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision

from catcam import cats as cat_assets
from catcam.cat_face import EYES, CatFace
from catcam.cat_mesh import CatMesh, Renderer
from catcam.ears import EarMotion
from catcam.face_motion import ExpressionTransfer
from catcam.fx import Effects, terminal_name
from catcam.fx import load as load_effects
from catcam.head_pose import OneEuroFilter, relative_rotation, rotation_of, to_matrix, to_vector
from catcam.paths import FACE_MODEL, HAND_MODEL, LOG_FILE
from catcam.paws import Paws, paw_set, paw_style
from catcam.picker import SIZE as PREVIEW_SIZE
from catcam.picker import Picker, put_text
from catcam.tail import TAIL_HAND, Tail, TailSkin, tail_skin
from catcam.tongue import seen

OUT_W, OUT_H, FPS = 1280, 720, 30
WINDOW = "Cat Cam"
# The "Tail" checkbox in the preview's top left corner (x, y, size in preview
# px); while it is ticked one of your hands lifts a tail instead of a paw.
TAIL_BOX = (16, 14, 22)
TAIL_LABEL = "Tail (t)"

EXPRESSION_GAIN = 1.3  # exaggerate your expressions on the cat (1 = 1:1)
BLINK_SMOOTHING = 0.7  # 0..1, higher = snappier; kept snappy so blinks show
TONGUE_SMOOTHING = 0.2  # 0..1 per frame: the tongue slides out and in; a one-frame glitch barely shows
# One Euro filters: lower MIN_CUTOFF = less jitter when still, higher BETA =
# less lag when moving fast. Expression speeds are in face widths per second.
ROTATION_MIN_CUTOFF, ROTATION_BETA = 0.8, 3.5
EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA = 1.5, 30.0

# Blendshape score range mapped to 0..1 eye closedness...
BLINK_RANGE = (0.3, 0.6)
# ...and, as some people's blink score stays low, also how much the eye
# opening shrank compared to your neutral face.
SHUT_RANGE = (0.35, 0.75)
# The neutral face only adapts while all of these are low.
EXPRESSIVE = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
    "mouthStretchLeft", "mouthStretchRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "eyeBlinkLeft", "eyeBlinkRight", "cheekPuff",
]  # ...and while no tongue is seen
RELAXED_BELOW = 0.25
NO_FACE_HINT_AFTER = 1.0  # s without a face before the preview says so
WEBCAM_TIMEOUT = 5.0  # s of failed webcam reads before giving up
FIRST_FRAME_TIMEOUT = 60.0
ENTER, ESC = 13, 27


@dataclass(frozen=True)
class Cat:
    """Everything needed to draw one cat."""
    name: str
    points: np.ndarray
    eyelids: CatFace
    mesh: CatMesh
    paw: dict  # {(view, hand): PawSprite}, see catcam.paws.paw_set
    tail: TailSkin | None
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
        cats.append(Cat(
            name=name,
            points=a.points,
            eyelids=CatFace(a.sprite, a.points[:, :2]),
            mesh=CatMesh(renderer, a.sprite, a.points, a.background),
            paw=paw_set(a.paws, paw_style(a.sprite, a.points)),
            tail=tail_skin(a.paws),
            sprite=a.sprite,
            background=a.background,
        ))
    return cats


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


def blinks(scores, face=None, neutral=None):
    """Eye closedness (image-left, image-right), each 0..1."""
    # MediaPipe names eyes from the person's view: their right eye is on the
    # image's left.
    closed = np.array([
        ramp(scores["eyeBlinkRight"], *BLINK_RANGE),
        ramp(scores["eyeBlinkLeft"], *BLINK_RANGE),
    ])
    if face is not None and neutral is not None:
        shrunk = 1 - eye_openness(face) / np.maximum(eye_openness(neutral), 1e-6)
        closed = np.maximum(closed, [ramp(v, *SHUT_RANGE) for v in shrunk])
    return closed


class FaceState:
    """Your smoothed head pose and expression, independent of the cat."""

    def __init__(self):
        self.rotation_filter = OneEuroFilter(ROTATION_MIN_CUTOFF, ROTATION_BETA)
        self.expression_filter = OneEuroFilter(EXPRESSION_MIN_CUTOFF, EXPRESSION_BETA)
        self.transfer = ExpressionTransfer()
        self.neutral_pose = None
        self.head = np.zeros(3)  # rotation vector; 0 = facing the viewer
        self.blink = np.zeros(2)
        self.ear_motion = EarMotion()
        self.ears = np.zeros(2)  # ear angles, see catcam.ears
        self.tongue = 0.0  # 0..1 how far the cat's tongue is out, see catcam.tongue
        self.tongue_seen = 0.0  # ...as your face last asked for
        self.offsets = np.zeros((478, 3))  # in face widths
        self.face = None  # last seen face landmarks in px
        self.last_seen = None

    def reset_neutral(self):
        """Your next face and pose become the neutral ones."""
        self.transfer.reset()
        self.neutral_pose = None

    def _slide_tongue(self):
        self.tongue += TONGUE_SMOOTHING * (self.tongue_seen - self.tongue)

    def update(self, result, cam_size, dt: float, now: float, frame: np.ndarray | None = None):
        """result: the face landmarker's; frame: the webcam picture (BGR) it
        came from, for what the landmarks alone don't tell (the tongue)."""
        if not result.face_landmarks:  # no face -> keep the last pose and expression
            self.ears = self.ear_motion.update(None, dt, now)
            self._slide_tongue()
            return
        self.face = landmarks_px(result.face_landmarks[0], *cam_size)
        self.last_seen = now
        pose = rotation_of(result.facial_transformation_matrixes[0])
        if self.neutral_pose is None:
            # The way you sit at the start (e.g. looking down at the screen
            # past the camera) is "facing the viewer" for the cat.
            self.neutral_pose = pose
        self.head = self.rotation_filter(to_vector(relative_rotation(pose, self.neutral_pose)), dt)
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        self.ears = self.ear_motion.update(scores, dt, now)
        closed = blinks(scores, self.face, self.transfer.neutral)
        self.blink = self.blink + BLINK_SMOOTHING * (closed - self.blink)
        if frame is not None:  # the face model has no tongue score: look for the tongue itself
            self.tongue_seen = seen(frame, self.face)
        self._slide_tongue()
        relaxed = max(scores[name] for name in EXPRESSIVE) < RELAXED_BELOW and self.tongue_seen < 0.1
        self.offsets = self.expression_filter(EXPRESSION_GAIN * self.transfer(self.face, relaxed), dt)


def render(cat: Cat, state: FaceState, paws: Paws, tail: Tail, tail_on: bool) -> np.ndarray:
    """tail_on: the tail hand moves the tail (still sinking out of view for a
    moment after it is switched off) instead of a paw."""
    sprite = cat.eyelids.render(blink=tuple(state.blink))
    offsets = state.transfer.to_cat(state.offsets, cat.points)
    frame = cat.mesh.render(sprite, offsets, to_matrix(state.head), state.ears, state.tongue)
    if cat.tail is None:
        return paws.draw(frame, cat.paw)
    hidden = (TAIL_HAND,) if tail_on else ()
    return paws.draw(tail.draw(frame, cat.tail), cat.paw, hidden=hidden)


def draw_tail_box(view: np.ndarray, checked: bool):
    """The tail checkbox, drawn into the preview (in place)."""
    x, y, size = TAIL_BOX
    label_w = cv2.getTextSize(TAIL_LABEL, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0][0]
    shade = view[y - 6:y + size + 6, x - 6:x + size + label_w + 18]
    shade[:] = (shade * 0.45).astype(np.uint8)
    cv2.rectangle(view, (x, y), (x + size, y + size), (240, 240, 240), 2, cv2.LINE_AA)
    if checked:
        tick = np.array([(x + 5, y + 11), (x + 9, y + 16), (x + 17, y + 6)], np.int32)
        cv2.polylines(view, [tick], False, (64, 170, 255), 3, cv2.LINE_AA)
    cv2.putText(view, TAIL_LABEL, (x + size + 10, y + size - 5), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)


def on_tail_box(x: int, y: int) -> bool:
    """Whether a click at preview px (x, y) hits the tail checkbox or its label."""
    bx, by, size = TAIL_BOX
    label_w = cv2.getTextSize(TAIL_LABEL, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0][0]
    return bx - 6 <= x <= bx + size + label_w + 12 and by - 6 <= y <= by + size + 6


def preview(out: np.ndarray, cat: Cat, looking: bool, notice: list[str] | None = None,
            tail: bool = False) -> np.ndarray:
    """Mirrored like a mirror, with hints and the tail checkbox, none of which reach the call."""
    view = cv2.flip(cv2.resize(out, PREVIEW_SIZE, interpolation=cv2.INTER_AREA), 1)
    w, h = PREVIEW_SIZE
    bar = view[h - 34:]
    bar[:] = (bar * 0.45).astype(np.uint8)
    put_text(view, f"{cat.name}   -   c: change cat   t: tail   n: reset neutral face   q: quit",
             w / 2, h - 12, 0.5, (240, 240, 240))
    draw_tail_box(view, tail)
    if looking:
        put_text(view, "Looking for your face...", w / 2, 36, 0.8, (255, 255, 255), 2)
    if notice:
        box = view[h // 2 - 50:h // 2 + 22 * len(notice) - 10]
        box[:] = (box * 0.3).astype(np.uint8)
        for i, line in enumerate(notice):
            put_text(view, line, w / 2, h // 2 - 18 + 30 * i, 0.6, (255, 255, 255), 1)
    return view


def create_trackers():
    video = vision.RunningMode.VIDEO
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=video,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=video,
        num_hands=2,
    ))
    return face, hands


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
    cards, and where the last click was (for the preview's checkbox)."""

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
    parser.add_argument("--camera", type=int, default=0, metavar="N",
                        help="webcam to use (default 0; try 1 if the wrong camera opens)")
    parser.add_argument("--check", action="store_true",
                        help="load and draw every cat once, without camera, then quit")
    return parser.parse_args(argv)


class Controls:
    """Which cat is streamed and whether the picker is open, driven by keys and clicks."""

    def __init__(self, picker: Picker, mouse: Mouse):
        self.picker, self.mouse = picker, mouse
        self.current, self.picking, self.picked_once = 0, True, False
        self.tail = False  # the tail checkbox
        self.quit = False

    def handle(self, key: int, state: FaceState):
        if key == ord("q"):
            self.quit = True
            return
        choice = self.picker.key_choice(key)
        if self.picking:
            if key == ENTER and self.picker.hover is not None:
                choice = self.picker.hover
            clicked = self.mouse.take_click()
            if clicked is not None:
                choice = clicked
            if key == ESC and self.picked_once:
                self.picking = False
        else:
            click = self.mouse.take_position()
            if key == ord("t") or (click is not None and on_tail_box(*click)):
                self.tail = not self.tail
            if key == ord("c"):
                self.picking = True
                self.picker.hover = None
            elif key == ord("n"):
                state.reset_neutral()
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


def stream(cats: list[Cat], controls: Controls, effects: Effects, trackers, vcam, webcam):
    face_tracker, hand_tracker = trackers
    state, paws, tail = FaceState(), Paws(OUT_W), Tail((OUT_W, OUT_H))
    start = last = time.monotonic()
    notice_until = 0.0
    app = terminal_name()
    restart = "quit Cat Cam and open it again" if app == "Cat Cam" else f"quit and reopen {app}"
    notice = [f"Allow {app} under Input Monitoring (opened in System Settings),",
              f"then {restart}" + ("." if app == "Cat Cam" else " and start Cat Cam again.")]
    for frame, now in frames(webcam):
        dt, last = max(now - last, 1e-3), now
        cam_size = (frame.shape[1], frame.shape[0])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int((now - start) * 1000)
        state.update(face_tracker.detect_for_video(image, timestamp), cam_size, dt, now, frame)
        hands = hand_tracker.detect_for_video(image, timestamp)
        cat = cats[controls.current]
        paws.update(hands, state.face, cam_size, dt, *cat.mesh.face_on_screen())
        hand = paws.slots.get(TAIL_HAND, {}) if controls.tail and cat.tail is not None else {}
        tail.update(hand.get("params"), hand.get("presence", 0.0), dt, *cat.mesh.face_on_screen())

        special = effects.update(now, cat.name)
        out = special if special is not None else render(cat, state, paws, tail, controls.tail)
        vcam.send(out)
        if controls.picking:
            cv2.imshow(WINDOW, controls.picker.draw(controls.current if controls.picked_once else None))
        else:
            looking = state.last_seen is None or now - state.last_seen > NO_FACE_HINT_AFTER
            cv2.imshow(WINDOW, preview(out, cat, looking, notice if now < notice_until else None,
                                       tail=controls.tail))
        key = cv2.waitKey(1) & 0xFF
        if effects.needs_permission(key, now):
            notice_until = now + 10
        controls.handle(key, state)
        if controls.quit or window_closed():
            return


def check(cats: list[Cat]):
    """Draws every cat once at rest, with paws, as a quick test of the install."""
    for cat in cats:
        out = render(cat, FaceState(), Paws(OUT_W), Tail((OUT_W, OUT_H)), tail_on=True)
        if out.shape != (OUT_H, OUT_W, 3) or not out.any():
            sys.exit(f"Drawing {cat.name} failed.")
    load_effects()  # its data file must load too
    import sounddevice  # noqa: F401  (imported only when needed, so check here they're there)
    if sys.platform == "darwin":
        import Quartz  # noqa: F401
    print(f"OK: {len(cats)} cats")


def alert(message: str):
    """A dialog for the standalone app, which has no terminal to print to."""
    quoted = message.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display alert "Cat Cam" message "{quoted}"'], check=False)


def main(argv=None):
    """Runs Cat Cam; the standalone app shows errors in a dialog and a log file."""
    if not getattr(sys, "frozen", False):
        return run(argv)
    try:
        return run(argv)
    except SystemExit as e:
        if isinstance(e.code, str):
            alert(e.code)
        raise
    except Exception:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text(traceback.format_exc())
        alert(f"Cat Cam stopped because of an error. Details are in {LOG_FILE}.")
        raise


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
    webcam = open_webcam(args.camera)

    effects = Effects(load_effects(), (OUT_W, OUT_H))
    effects.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    mouse = Mouse(picker)
    cv2.setMouseCallback(WINDOW, mouse)
    print(f"Streaming to '{vcam.device}'. Select it as the camera in Zoom.\n"
          "Pick a cat in the window. Keep a relaxed face for a second at the start.")
    try:
        with vcam:
            stream(cats, Controls(picker, mouse), effects, trackers, vcam, webcam)
    except KeyboardInterrupt:
        pass
    finally:
        effects.stop()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
