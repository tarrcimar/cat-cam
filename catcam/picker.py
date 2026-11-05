"""The cat picker, drawn into the preview window.

A grid of cards, one per cat. Hovering highlights a card; clicking it or
pressing its number picks the cat. It is only an image plus hit-testing, so
the app keeps streaming while it is shown.
"""

import math

import cv2
import numpy as np

SIZE = (960, 540)  # also the size of the live preview, so the window keeps its size
COLUMNS = 3
MARGIN, GAP = 40, 18
HEADER, FOOTER = 92, 44
NAME_BAR = 38
RADIUS = 14
FONT = cv2.FONT_HERSHEY_DUPLEX

BACKGROUND = (34, 30, 28)
CARD_BORDER = (70, 64, 60)
ACCENT = (64, 170, 255)  # BGR orange
TEXT = (245, 245, 245)
MUTED = (165, 160, 155)


def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Anti-aliased rounded rectangle, float 0..1."""
    ss = 4
    mask = np.zeros((h * ss, w * ss), np.uint8)
    r = radius * ss
    cv2.rectangle(mask, (r, 0), (w * ss - r, h * ss), 255, -1)
    cv2.rectangle(mask, (0, r), (w * ss, h * ss - r), 255, -1)
    for cx in (r, w * ss - r - 1):
        for cy in (r, h * ss - r - 1):
            cv2.circle(mask, (cx, cy), r, 255, -1)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def _blend(canvas: np.ndarray, image: np.ndarray, alpha: np.ndarray, x: int, y: int):
    h, w = image.shape[:2]
    roi = canvas[y:y + h, x:x + w]
    a = alpha[..., None]
    canvas[y:y + h, x:x + w] = (image * a + roi * (1 - a)).astype(np.uint8)


def put_text(canvas, text, center_x, baseline_y, scale, color, thickness=1):
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(canvas, text, (int(center_x - w / 2), int(baseline_y)), FONT, scale,
                color, thickness, cv2.LINE_AA)


def _thumbnail(sprite: np.ndarray, background, w: int, h: int) -> np.ndarray:
    """The cat on its own backdrop, cropped to the cat and fitted into w x h."""
    ys, xs = np.nonzero(sprite[..., 3] > 20)
    cat = sprite[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min(w / cat.shape[1], h / cat.shape[0]) * 0.92
    cat = cv2.resize(cat, (max(int(cat.shape[1] * scale), 1), max(int(cat.shape[0] * scale), 1)),
                     interpolation=cv2.INTER_AREA)
    thumb = np.empty((h, w, 3), np.uint8)
    thumb[:] = background
    x, y = (w - cat.shape[1]) // 2, h - cat.shape[0]  # sits on the name bar
    _blend(thumb, cat[..., :3], cat[..., 3].astype(np.float32) / 255, x, y)
    return thumb


class Picker:
    def __init__(self, cats: list[tuple[str, np.ndarray, np.ndarray]]):
        """cats: (name, BGRA sprite, BGR background) per cat."""
        self.names = [name for name, _, _ in cats]
        width, height = SIZE
        # More columns for many cats, so the cards keep room for the cat.
        columns = min(len(cats), max(COLUMNS, math.ceil(math.sqrt(len(cats) * 1.5))))
        rows = math.ceil(len(cats) / columns)
        card_w = (width - 2 * MARGIN - (columns - 1) * GAP) // columns
        card_h = (height - HEADER - FOOTER - (rows - 1) * GAP) // rows
        self.cards = []  # (x, y, w, h) per cat
        for i in range(len(cats)):
            row, col = divmod(i, columns)
            in_row = min(columns, len(cats) - row * columns)  # centre a short last row
            x0 = (width - in_row * card_w - (in_row - 1) * GAP) // 2
            self.cards.append((x0 + col * (card_w + GAP), HEADER + row * (card_h + GAP), card_w, card_h))
        self.mask = _rounded_mask(card_w, card_h, RADIUS)
        self.border = _rounded_mask(card_w + 6, card_h + 6, RADIUS + 3)
        self.thumbs = [_thumbnail(s, bg, card_w, card_h - NAME_BAR) for _, s, bg in cats]
        self.hover = None

    def hit(self, x: int, y: int) -> int | None:
        for i, (cx, cy, w, h) in enumerate(self.cards):
            if cx <= x < cx + w and cy <= y < cy + h:
                return i
        return None

    def key_choice(self, key: int) -> int | None:
        """Cat picked by a number key (1-9), if any."""
        index = key - ord("1")
        return index if 0 <= index < min(len(self.names), 9) else None

    def draw(self, current: int | None) -> np.ndarray:
        """current: index of the cat being streamed, marked on its card."""
        width, height = SIZE
        canvas = np.empty((height, width, 3), np.uint8)
        canvas[:] = BACKGROUND
        put_text(canvas, "Choose your cat", width / 2, 50, 1.1, TEXT)
        hint = "click a cat or press its number" if len(self.names) > 1 else "click the cat"
        if current is not None:
            hint += "  -  Esc to go back"
        put_text(canvas, hint, width / 2, 78, 0.5, MUTED)

        for i, ((x, y, w, h), thumb, name) in enumerate(zip(self.cards, self.thumbs, self.names)):
            hovered, live = i == self.hover, i == current
            if hovered or live:
                ring = np.empty((h + 6, w + 6, 3), np.uint8)
                ring[:] = ACCENT if hovered else CARD_BORDER
                _blend(canvas, ring, self.border, x - 3, y - 3)
            card = np.empty((h, w, 3), np.uint8)
            card[:] = (24, 22, 20)
            card[: h - NAME_BAR] = thumb if hovered else (thumb * 0.88).astype(np.uint8)
            label = f"{i + 1}  {name}" if i < 9 else name
            put_text(card, label, w / 2, h - 13, 0.62, ACCENT if hovered else TEXT)
            if live:
                cv2.circle(card, (w - 16, 16), 6, (90, 200, 90), -1, cv2.LINE_AA)
            _blend(canvas, card, self.mask, x, y)

        put_text(canvas, "In the call: c = change cat   t = tail   n = reset neutral face   q = quit",
                 width / 2, height - 16, 0.45, MUTED)
        return canvas
