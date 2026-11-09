# Cat Cam

Be a cat, or Woofgang the golden retriever, in your Zoom calls. Pick one of
eight: Gingersnap, Grinbert, Grumbledore, Grumpkin, Meringue, Pleadwell,
Wrinklestiltskin or Woofgang. Each is a real 3D head that turns, nods and tilts
with your head, copies your whole face (brows, blinks, cheeks, smile) and opens
its own mouth with yours, its jaw swinging down to show teeth and tongue, while
the side of its head that turns away from the light falls into shadow. Your
hands become its paws, real 3D ones in its own colours (a dog's for Woofgang),
that turn with your hands and curl toe by toe with your fingers: the toe beans
when you show your palm, the fur when you show the back of your hand. Or keep
your own hands, cut out of the webcam picture and washed in its fur colours. When you
talk, the call hears your animal instead of your voice: Woofgang barks, and
every cat has a voice of its own, from Grinbert's chirpy trills to
Grumbledore's slow old grumbles, all real animals. When you sit still, your animal
breathes and looks around on its own. And you can put any picture or video
behind it.

Your own face is never sent anywhere. Only the drawn cat goes out to the
virtual camera (with your hands, cut out of the picture, if you choose "Real
hands"), and everything runs locally on your computer.

![The cat picker](docs/picker.jpg)

![Rest, smile, and a smile with the head turned](docs/expressions.jpg)

## Easy install (Mac with Apple Silicon)

No programming tools needed: download **Cat-Cam-<version>.dmg** from the
[Releases page](https://github.com/tarrcimar/cat-cam/releases), open it and
follow *Read me first.txt* inside. In short: drag Cat Cam to Applications,
set up OBS's virtual camera once (step 2 below), and the first time you open
Cat Cam allow it under *System Settings → Privacy & Security → Open Anyway*.

The rest of this page is for running it from the code instead.

## What you need

- **macOS**. It is developed and tested on Apple Silicon. Windows should work
  too (pyvirtualcam and OpenGL both support it) but is untested.
- A webcam.
- [**uv**](https://docs.astral.sh/uv/), which installs Python and all
  libraries for you.
- [**OBS Studio**](https://obsproject.com). Only its virtual camera driver is
  used; OBS itself stays closed.

## One-time setup

1. **Install uv**:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install OBS and enable its virtual camera.** Run
   `brew install --cask obs` or download it from obsproject.com. Then:
   1. Open OBS and click **Start Virtual Camera**.
   2. When macOS asks, allow the system extension. You may need to approve it
      under *System Settings → General → Login Items & Extensions*.
   3. Click **Stop Virtual Camera** and **quit OBS**.

   You only do this once. Afterwards, keep OBS closed while Cat Cam runs, or
   it takes the virtual camera for itself.

3. **Allow keyboard access for your terminal app.** A few keyboard
   shortcuts also work while Zoom is in front, which macOS only allows with
   *Input Monitoring*. macOS usually does **not** ask for it by itself:
   1. Open *System Settings → Privacy & Security → Input Monitoring*.
   2. If your terminal app (Terminal, iTerm, Warp, ...) isn't in the list,
      click **+** at the bottom, choose it from *Applications*, and add it.
   3. Switch it on.
   4. **Quit and reopen the terminal app** (⌘Q). The permission only takes
      effect after a restart.

4. **Optional: its voice.** For the call to hear barks and meows instead of
   your voice, Cat Cam needs the free **BlackHole 2ch** virtual microphone,
   made by [Existential Audio](https://existential.audio/blackhole/). No
   Terminal needed:
   1. Start Cat Cam (see below). Without BlackHole, its window shows a
      **Get the voice** button in its bottom right corner. Click it: Cat Cam
      downloads BlackHole's installer from its makers' site, checks it is
      really theirs, and opens it in Apple's installer.
   2. Click through the installer. It asks for your Mac's password, as it
      installs an audio driver.
   3. **Restart the Mac**, so it loads the new microphone.
   4. In Zoom, choose **BlackHole 2ch** as the microphone (*Settings → Audio →
      Microphone*).

   Or get it from the website instead: go to
   [existential.audio/blackhole](https://existential.audio/blackhole/), choose
   **BlackHole 2ch** and enter your email address (the download link arrives
   by email), then do steps 2 to 4 with the downloaded `BlackHole2ch-….pkg`.
   With Homebrew: `brew install blackhole-2ch`.

   Cat Cam listens to your real microphone and sends the animal's sounds to
   BlackHole. Without BlackHole everything else works, and the call hears
   your real voice.

5. **Get the code**:

   ```sh
   git clone https://github.com/tarrcimar/cat-cam.git
   cd cat-cam
   ```

## Run it

```sh
uv run cat-cam
```

The first run downloads the libraries, which takes a minute. macOS then asks
whether your terminal may use the camera; allow it. If you said no, enable it
under *System Settings → Privacy & Security → Camera* and run again.

A window opens with the cat picker. Click a cat, or press its number.

In Zoom, go to *Settings → Video → Camera* and choose **OBS Virtual Camera**.
Do the same in any other app (Meet, Teams, FaceTime, ...). If the camera is
not listed, restart Zoom while Cat Cam is running.

Keep a relaxed face for the first second: the cat compares your face against
how it looks at the start. Looking into the camera is looking at the viewer,
wherever you sit. If the cat looks grumpy while you are relaxed, or you'd
rather it faced the viewer while you look elsewhere (at the screen, say),
press `n` while doing that.

Raise your hands into the camera's view for paws. Turn a hand around to
show the other side of the paw; tilt or turn it and the paw does too. Each toe
follows one of your fingers: curl a finger and its toe curls in, make a fist
and they all do.
**Paws** in the top left corner of the window (or `h`) goes from the
**3D paws** to the **photo paws** (or drawn ones, for a cat without paw photos)
to **Real hands** and round again: your own hands, cut out of the webcam picture
(nothing else of you: not your face, hair or clothes) and drawn where the paws
would be, as big next to the animal as they are next to your face, with a light
wash of its fur colours and a fine grain of fur. They are still clearly your
hands. Bare forearms show a little way down, sleeves don't. In a dim room the
cut-out gets patchy, so light your hands well.
Turn on **Tail** in the top left corner of the window (or press `t`) for the
cat's tail: while only one of your hands is in view, it lifts the tip of
the tail instead of being a paw, and the rest of the tail rises from below,
trails behind, curls and swishes on its own. Show both hands and they are
two paws again. Turn it off for paws only.

The **3D head** switch next to it (or `d`) is on to begin with: turn it off
for the flat photo head, which is lighter on older Macs. (Cats without a 3D
head, see [docs/3d-heads.md](docs/3d-heads.md), don't have the switch.)

Turn on **Background** (or press `b`) and pick a picture or a video: it goes
behind the animal, in place of the plain backdrop. A video plays on a loop.
Turn it off and on again to pick another.

When you talk (and BlackHole is set up, see above), your animal talks along,
as loud as you are: Woofgang barks with almost every syllable, and a cat
answers with a whole meow now and then, a few times a sentence. Grinbert
trills, Pleadwell mews like a kitten, Meringue meows long and bored, Grumpkin
grumbles for its food, Wrinklestiltskin yowls with a rasp, Gingersnap goes
"mrrp?" and Grumbledore grumbles slow and old. They are recordings of real
cats and a real dog, played as they are.
Turn off **Voice** (or press `v`) for the call to hear your own voice
instead, and on again for the animal's. When you sit still and quiet for a
few seconds, your animal breathes, sways and looks around on its own, until
you move or talk again.

Press `1` to `8` (in the Cat Cam window, while the picker is closed) for an
emoji reaction beside your animal, which the call sees too: 1 🫵 points at
the viewer, 2 🙌, 3 🙏, 4 👎, 5 👏, 6 🤌, 7 🤚 and 8 👋 waves. Each moves
the way its gesture does and is gone after a couple of seconds; press it
again to start it over. They are drawn with the Mac's own emoji font, so
without it (not a Mac) the keys do nothing.

### Keys (in the Cat Cam window)

| Key        | What it does                                              |
|------------|-----------------------------------------------------------|
| `1`, `2`, ... | in the picker: pick that cat                            |
| `1` to `8` | in the call: an emoji reaction (see above)                |
| `c`        | open the cat picker (the call keeps showing your cat)     |
| `Esc`      | close the picker                                          |
| `t`        | tail on/off (same as the Tail switch)                     |
| `h`        | paws: 3D paws, photo paws or your real hands, in turn     |
| `d`        | 3D head on/off, for cats that have one                    |
| `b`        | background on/off (on asks for a picture or video)        |
| `v`        | voice on/off: the animal's sounds, or your own voice      |
| `n`        | "this is my neutral face, looking at the viewer": recalibrate face and head pose |
| `q`        | quit (closing the window or Ctrl+C also works)            |

The preview in the window is mirrored, like a mirror. The call gets the
unmirrored picture, just as a normal camera would send.

### Options

```sh
uv run cat-cam --camera 1             # use another webcam (they are listed, numbered, at the start)
uv run cat-cam --mic "USB"            # listen to another microphone (part of its name)
uv run cat-cam --background beach.jpg # start with this picture or video behind the animal
```

`uv run python -m catcam.voice --list` lists the microphones;
`uv run python -m catcam.voice` lets you hear the barks in your headphones
(`--cat Grumbledore` for another animal's voice).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open webcam` | Allow camera access for your terminal (see above). Cat Cam uses the Mac's own camera if it has one; for another, pick its number from the list printed at the start with `--camera`. |
| Error about the virtual camera | Do the OBS setup above, and make sure OBS is **not** running. |
| Zoom shows a black picture | OBS is probably running; quit it and restart Cat Cam. |
| Zoom does not list "OBS Virtual Camera" | Restart Zoom while Cat Cam is running. |
| The cat looks off while you are relaxed | Press `n` with a neutral face. |
| "Looking for your face..." | Face the camera and make sure your face is lit. |
| The call hears your real voice | Choose **BlackHole 2ch** as the microphone in Zoom. If Cat Cam shows a **Get the voice** button, BlackHole isn't installed yet: click it (see above). |
| "Restart your Mac to finish setting up the voice" | BlackHole is installed, but macOS hasn't loaded it yet: restart the Mac (or run `sudo killall coreaudiod`), then restart Cat Cam. |
| No barks or meows at all | Check the **Voice** switch is on (`v`), and allow microphone access for your terminal under *System Settings → Privacy & Security → Microphone*. |

## Add your own cat

1. Put a portrait in `all_cats/<Name>.png`. It works best when the face looks
   straight at the camera with the mouth closed, against a plain background.
2. Prepare it. This cuts the cat out and finds its face, and takes about
   10 seconds per cat:

   ```sh
   uv run --group prepare python -m catcam.prepare
   ```

   The results go to `assets/cats/<Name>/`. Already prepared cats are skipped;
   add `--force` to redo them, or name cats to do only those.
   Optionally add paw photos next to it: `all_cats/<Name>PawsFront.png`
   (the front legs seen from the front, as if the cat stood facing you) and
   `all_cats/<Name>PawsBack.png` (the soles). In each, put the left paw on
   the left and the right paw on the right, separated by a vertical purple
   line. `prepare` cuts them out too; a cat without paw photos gets drawn
   paws in its colours. These photo paws show when **Paws** is set to
   them, or if the 3D paws (`assets/paws/`) are missing; the tail always
   takes its skin from them.
   If the mouth ends up in the wrong place (it can on faces that aren't
   cat-like, such as a dog's), write where the lips meet in
   `all_cats/<Name>Mouth.txt` and prepare it again with `--force`; see
   [docs/3d-heads.md](docs/3d-heads.md#how-the-eyes-and-mouth-work).
3. Run `uv run cat-cam`. The new cat appears in the picker.
4. Optionally give it a 3D head, generated from the same portrait on your
   Mac or with Meshy: see [docs/3d-heads.md](docs/3d-heads.md). A cat that
   only exists as a Meshy model gets its portrait rendered from the model.
5. Optionally give it a voice: a few short sounds (`.ogg`, `.wav`, `.flac`
   or `.mp3`) in `assets/cats/<Name>/voice/`, each a single real meow (or
   bark) with a little quiet around it, and a `voice.toml` next to them for
   how they follow your voice (copy one from another cat).
   `tools/cat_voices.py` cuts them out of CC0 recordings from Freesound. Without
   sounds, the call hears your own voice.

## Tuning

These constants sit at the top of their files:

| Constant | File | Effect |
|---|---|---|
| `EXPRESSION_GAIN` | `catcam/app.py` | how exaggerated the cat's expressions are (1 = same as you) |
| `EXPRESSION_MIN_CUTOFF`, `EXPRESSION_BETA`, `EXPRESSION_FLOOR` | `catcam/app.py` | face smoothing: lower cutoff means less jitter, higher beta means less lag, and moves slower than the floor count as tracking noise (measure with `tools/tracking_eval.py`) |
| `ROTATION_MIN_CUTOFF`, `ROTATION_BETA`, `ROTATION_FLOOR` | `catcam/app.py` | the same for head movement |
| `BLINK_RANGE`, `BLINK_OPEN` | `catcam/app.py` | how far your eyes' blink scores must rise above their resting level before the cat blinks, and how quickly its lids open again |
| `FLATTEN`, `PERK`, `TWITCH_EVERY` | `catcam/ears.py` | how far the ears move, and how often they twitch |
| `STRENGTH`, `LIGHT` | `catcam/lighting.py` | how much turning the head changes its shading, and where the light comes from |
| `PAW_SIZE` | `catcam/paws.py` | paw size relative to your hand |
| `WASH`, `GRAIN`, `MAX_SCALE` | `catcam/real_hands.py` | how far your real hands take on the animal's fur colour, how strong its fur grain is, and how much bigger than in the webcam picture they may be drawn |
| `SKIN`, `REACH`, `FOREARM` | `catcam/real_hands.py` | how sure the segmenter must be that a pixel is skin, and how far round your hand and down your forearm the cut-out reaches |
| `HAND_DEPTH` | `catcam/paws.py` | how much of your hand's tilt toward or away from the camera the 3D paws copy |
| `CURL_ANGLE`, `RAISE` | `catcam/paw3d.py` | how far a toe curls with its finger, and how much taller a finger raised out of a fist stands |
| `SOCK_SHARE`, `TABBY`, `AMBIENT` | `catcam/paw3d.py` | how much lighter a paw is than the arm, how finely patterned fur must be for stripes, and how flat the paws' light is |
| `LENGTH`, `SWISH`, `DRAG`, `SWITCH_FRAMES` | `catcam/tail.py` | how long the tail is, how much it sways by itself, how quickly it calms down, and how long a change between one and two hands must last before the tail follows |
| `TOP_MARGIN`, `CHIN_LINE` | `catcam/cat_mesh.py` | how much of the screen the cat fills |
| `JAW_HINGE`, `OWN_JAW_MAX` | `catcam/head3d.py` | where a 3D head's own jaw swings from, and how far it opens |
| `STILL_AFTER`, `BREATHING`, `SWAY`, `LOOK` | `catcam/idle.py` | how long you must be still before he acts on his own, his breathing, his sway, and how far he looks around |
| `ABOVE_NOISE`, `GAIN`, `PITCH_LIMIT` | `catcam/voice.py` | how loud you must talk for your animal to answer, how loud it answers, and how far its sounds' pitch may move from the recording |
| `pitch_follow`, `jitter`, `gap` | `assets/cats/<Name>/voice/voice.toml` | per animal: how much its sounds' pitch follows yours, how much it varies, and how long it stays quiet after a sound before the next |
| `SIZE`, `CENTRE`, `DURATION` | `catcam/reactions.py` | how big the emoji reactions are, where they sit and how long they stay |

## How it works

1. MediaPipe tracks 478 points on your face plus your hands, from the webcam.
2. Each cat has the same 478 points, found once by `catcam.prepare`. The
   cats have human-like faces, so the human face model fits them.
3. Your points are aligned to your neutral face, which removes head position,
   rotation and distance. What remains is your expression. That is mapped onto
   the cat's points (`catcam/face_motion.py`).
4. The cat photo is a textured 3D mesh. It is deformed by your expression,
   rotated by your head pose and drawn with OpenGL, shaded by how much each
   part turned toward or away from the light (`catcam/lighting.py`). The
   mouth's inside and cat teeth are drawn in the gap between the lips,
   coloured to match the cat's lips (`catcam/cat_mesh.py`).
5. Blinks lower the cat's eyelids in its texture (`catcam/cat_face.py`), the
   ears bend about a hinge below their tips (`catcam/ears.py`), and paws are
   drawn where your hands are (`catcam/paws.py`). They are 3D models
   (`catcam/paw3d.py`, a cat's and a dog's, made from Meshy models by
   `tools/paw_model.py`, which finds their pads and rigs their toes) turned
   like your hand, each toe bent about its knuckle as far as its finger is
   curled. They have no texture of their own: each animal colours them from
   its portrait, the fur from its cheeks and chin (striped for a tabby), the
   pads from its nose. A tail (`catcam/tail.py`) is
   a chain of points whose tip is pulled toward your hand; the rest is
   simulated with momentum, even bending and a bend limit, and wrapped in the
   skin of the cat's paw photo. With **Real hands**, MediaPipe's multiclass
   selfie segmenter tells the skin of your hands and arms from your face,
   hair, clothes and background, on its own thread alongside the trackers;
   only that skin near a tracked hand is cut out, given a wash of the
   animal's fur colours and drawn where the paw would be
   (`catcam/real_hands.py`).
6. A 3D head (`catcam/head3d.py`) is tied to the photo cat's face points, so
   it takes on the same expressions. One with its own mouth, like all of these,
   has its lower jaw found and cut free when it is imported: it grows over the
   model from the lower lip, the upper face from the upper lip, and they meet
   in the slit between the lips (`catcam/head3d_import.py`). The jaw then swings
   about its hinge as far as your lower lip drops, showing teeth, gums and a
   tongue modelled in Blender, a cat's or a dog's (`catcam/mouth3d.py`,
   `tools/mouth_model.py`), in front of a mouth cavity the shader traces
   behind them: a wet pink lining with a ridged palate and a deeper red
   throat. All of it is lit as the heads are, by soft light that only gets in
   through the opening between the lips, so it darkens with depth.
7. While you are still and quiet, `catcam/idle.py` adds breathing, a sway and
   glances to the head. A background picture or video goes wherever the
   animal isn't (`catcam/background.py`). Emoji reactions are drawn on
   top, from the Mac's emoji font (`catcam/reactions.py`).
8. The frame goes to the OBS virtual camera through pyvirtualcam. Your
   microphone goes through `catcam/voice.py`, which finds the syllables in
   your loudness and starts one of the animal's sounds (a bark, a meow, a
   trill...) with a syllable, plays it to its end and waits a moment before
   the next, out to the BlackHole virtual microphone. An animal's
   `voice/voice.toml` says how its sounds follow your voice.

## Development

Want to contribute or make a release? See [CLAUDE.md](CLAUDE.md).

```sh
uv run pytest          # tests, including render checks on every cat
uv run ruff check .    # lint
uv run cat-cam --check # load and draw every cat once, without a camera
```

To build the standalone app and its disk image (`dist/Cat-Cam-<version>.dmg`)
on an Apple Silicon Mac:

```sh
packaging/build_mac.sh
```

```
catcam/
  app.py           main loop: webcam -> tracking -> render -> virtual camera
  webcams.py       which webcam to use (the Mac's own, never OBS's virtual camera)
  picker.py        the cat picker
  cats.py          finding and loading prepared cats
  cat_mesh.py      the 3D head mesh and its OpenGL rendering
  cat_face.py      eyelids
  ears.py          ear movement
  lighting.py      shading as the head turns
  face_motion.py   expression transfer
  head_pose.py     head rotation and the One Euro smoothing filter
  idle.py          what the animal does by itself while you are still
  voice.py         your voice as barks or meows, out to the virtual microphone
  blackhole.py     the "Get the voice" button: fetches, checks and opens BlackHole's installer
  background.py    a picture or video behind the animal
  reactions.py     emoji reactions on the number keys
  paws.py          paws: following your hands, and the photo paws
  paw3d.py         3D paws, coloured like each animal (models made by tools/paw_model.py)
  real_hands.py    your real hands cut out of the webcam picture, tinted like the animal
  tail.py          the tail: a chain of points simulated like a real tail
  paw_photos.py    turns paw photos into paw sprites (used by prepare)
  prepare.py       turns all_cats/*.png into assets/cats/*
  landmark_fit.py  finds a cat's face points (used by prepare)
  head3d.py        3D heads, behind the 3D head switch
  glass.py         the window's glass panels, switches and text
  mouth3d.py       a 3D head's teeth, tongue and jaw (sets made by tools/mouth_model.py)
  head3d_import.py turns a generated .glb head into a cat's 3D head
all_cats/          source portraits
assets/cats/       prepared cats (cutout + face points, 3D heads), used at runtime
assets/mouth/      3D teeth, gums, palate and tongue, a cat's and a dog's
assets/paws/       3D paws, a cat's and a dog's
docs/3d-heads.md   how to make a 3D head
tools/             3D helpers: TRELLIS.2 install and fixes, render sheets of heads and paws, the
                   mouth and paw models;
                   tracking_eval.py measures face tracking on recorded webcam clips;
                   frame_bench.py measures frame times, every animal, no one at the camera;
                   cat_voices.py makes the animals' voices from CC0 recordings
models/            MediaPipe face and hand models, and the selfie segmenter
packaging/         the standalone macOS app: PyInstaller spec, icon, build script
```

The MediaPipe models in `models/` are by Google and licensed under Apache 2.0:
the face and hand landmarkers, and the multiclass selfie segmenter
(`selfie_multiclass_256x256.tflite`, from
`https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/`,
see its [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf)).
The animals' voices (`assets/cats/<Name>/voice/*.ogg`) are made by
`tools/cat_voices.py` from these Freesound recordings, all CC0 (public
domain), cut into single meows (or barks) and otherwise left as recorded:

| Animal | Recordings |
|--------|------------|
| Woofgang | ["Animal - Dog, Retriever (Teen)"](https://freesound.org/s/146343/) by colinpoh |
| Grinbert | ["Cat trilling"](https://freesound.org/s/200336/) by jsbarrett |
| Pleadwell | ["Kitten Meowing.wav"](https://freesound.org/s/196251/) by lolamadeus, ["kittens meowing"](https://freesound.org/s/149191/) by videog |
| Meringue | ["Cat meowing x5"](https://freesound.org/s/214759/) by peridactyloptrix |
| Grumpkin | ["Cat Wants Food"](https://freesound.org/s/120160/) by Oneirophile |
| Wrinklestiltskin | ["Cat meowing raspy"](https://freesound.org/s/757185/) by HenKonen |
| Gingersnap | ["AnnoyedCatMeowsGruntsVowels.wav"](https://freesound.org/s/541264/) by sweet_niche |
| Grumbledore | ["Old cat meowing"](https://freesound.org/s/338819/) by cabled_mess |
