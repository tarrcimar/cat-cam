# Making a 3D head for a cat

Each cat can have a real 3D head next to its photo head (the **3D head**
switch in the app, or `d`). It turns properly, and when the mouth opens there
are teeth, a tongue and a jaw inside. The head is generated from the cat's
portrait by [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), an
image-to-3D model, running locally on an Apple Silicon Mac. This page is the
recipe the first heads were made with, including what we tried that didn't work.

Everything after generation is automatic. The eyes, the mouth, expressions,
blinks, ears and shading all come from the import and the app; there is
nothing to draw or rig by hand.

TRELLIS.2 is free and local. [Meshy](https://www.meshy.ai) (paid, online)
makes more detailed heads, and the same import takes them; see
[Heads from Meshy](#heads-from-meshy).

## What you need

- An Apple Silicon Mac. The heads were made on an M5 MacBook Pro with 32 GB;
  generating one needs about 6 GB of memory.
- About 35 GB of free disk: about 30 GB of model weights (downloaded on the
  first run) plus a 1 GB Python environment.
- **Xcode** from the App Store (the full app, not only the command line
  tools), opened once. Then, in Terminal:

  ```sh
  sudo xcode-select -s /Applications/Xcode.app
  sudo xcodebuild -license accept
  xcodebuild -downloadComponent MetalToolchain
  ```

  Without the Metal toolchain the texture is baked by a slower, worse
  fallback.
- A free [Hugging Face](https://huggingface.co) account:
  - Request access to [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0),
    the background remover TRELLIS uses. Approval is usually instant.
  - Then log in: `uvx --from huggingface_hub hf auth login`.
  - You don't need access to Meta's DINOv3; see step 3 of the install.

## Install TRELLIS.2 (once)

We use [trellis-mac](https://github.com/shivampkumar/trellis-mac), a port
of TRELLIS.2 to Apple's Metal. Out of the box it doesn't build with today's
Xcode and torch, and its textures come out with flaws. `tools/trellis/`
has the exact versions and fixes the heads were made with.

1. Get trellis-mac and its parts at the tested versions, with the fixes
   applied. Run this from the Cat Cam folder; it is safe to run again:

   ```sh
   tools/trellis/install.sh ~/trellis-mac
   ```

   It does four things:
   - Builds the Metal parts as C++20.
   - Fetches the Eigen library that one of them needs.
   - Adds a `TRELLIS_REMESH` switch to `generate.py`.
   - Fixes the texture baker: black specks, dark seams along texture edges,
     and a texture that came out upside down.
2. Run trellis-mac's own setup, which installs and builds everything:

   ```sh
   cd ~/trellis-mac
   bash setup.sh
   ```

3. Make the DINOv3 image encoder TRELLIS needs. Meta's copy is gated behind
   an access request that can take days. This converts timm's public copy of
   the same weights instead, and checks the two give the same results:

   ```sh
   .venv/bin/python convert_dinov3.py dinov3-vitl16
   ```

## Make a head

1. **Generate** it from the cat's portrait, the same `all_cats/<Name>.png`
   the photo cat uses. TRELLIS removes the background itself.

   ```sh
   cd ~/trellis-mac
   TRELLIS_REMESH=1 .venv/bin/python run_local_dino.py \
       ~/path/to/cat-cam/all_cats/<Name>.png --output out/<name> --texture-size 2048
   ```

   The first run downloads the model weights. After that a head takes about
   6 minutes. The result is `out/<name>.glb`, which you can look at in
   Preview or any 3D viewer.
2. **Import** it into Cat Cam, from the Cat Cam folder:

   ```sh
   uv run --with trimesh --with networkx python -m catcam.head3d_import ~/trellis-mac/out/<name>.glb <Name>
   ```

   This writes `assets/cats/<Name>/head3d.npz` and `head3d.png`. It prints
   two numbers worth reading:

   ```
   Woofgang: portrait outline overlap 0.961
     features were [ 7. 42.] px off; moved the camera
   ```

   - **Outline overlap:** how well the head's outline, seen face-on, matches
     the portrait. 0.95 or more is good. Well under 0.9 means the head came
     out a different shape from the photo; generate it again with another
     `--seed`.
   - **Features off:** how far the eyes and nose had to be nudged to land on
     the portrait's. Up to a few dozen pixels is normal; the import corrects it.
3. **Look at it** without a camera. This renders the photo head and the 3D
   head side by side: at rest, mouth half and wide open, eyes closed, ears
   flat, and turned with the mouth open:

   ```sh
   uv run python tools/head3d_sheet.py <Name> -o sheet.jpg
   ```

   Then try it for real: `uv run cat-cam`, pick the cat, and press `d`.
4. **Commit** `assets/cats/<Name>/head3d.npz` and `head3d.png`. They are
   about 15 MB per cat.

## Heads from Meshy

Woofgang's head and the seven cats' were made with Meshy's Image to 3D
instead of TRELLIS:

1. **Settings:**
   - Multi-view on, with the front portrait as the main view and the
     profiles and back as Left, Back and Right. "Left" faces the image's
     left.
   - Meshy 7.1 Flagship, High Detail, Ultra 2K.
   - Texture on, with the Meshy 7 texture model at 2K and no PBR maps.
   - Pose off, Image Enhancement off, licence Private.
2. **Remesh it in Meshy** to 300,000 triangles: Target Polycount Fixed,
   Custom 300K, Topology Triangle. Meshy's heads have millions of
   triangles, far too many to move every frame, and Meshy bakes the texture
   onto the remeshed head properly. Download that as GLB.

   Or reduce a downloaded copy yourself (the download stays as it is). This
   works on smooth heads; on very furry ones the texture's many small pieces
   stop it well short of 300,000, and Meshy's remesh is the way:

   ```sh
   uv run --no-project --with pymeshlab --with trimesh --with pillow python - <head.glb> <small.glb> <<'PY'
   import sys, tempfile, pymeshlab, trimesh
   head = next(iter(trimesh.load(sys.argv[1]).geometry.values()))
   ms = pymeshlab.MeshSet()
   ms.add_mesh(pymeshlab.Mesh(vertex_matrix=head.vertices, face_matrix=head.faces,
                              v_tex_coords_matrix=head.visual.uv))
   ms.compute_texcoord_transfer_vertex_to_wedge()
   ms.meshing_decimation_quadric_edge_collapse_with_texture(
       targetfacenum=300000, preserveboundary=True, preservenormal=True)
   with tempfile.TemporaryDirectory() as tmp:  # an .obj keeps the texture coordinates per corner
       ms.save_current_mesh(f"{tmp}/small.obj")
       small = trimesh.load(f"{tmp}/small.obj", process=False, force="mesh")
   small.visual = trimesh.visual.TextureVisuals(
       uv=small.visual.uv, material=trimesh.visual.material.PBRMaterial(
           baseColorTexture=head.visual.material.baseColorTexture))
   small.export(sys.argv[2])
   PY
   ```
3. **No portrait?** A cat made from pictures that aren't a straight,
   closed-mouth portrait gets one from its head: render it face-on, unlit,
   on the usual backdrop, then prepare the cat from it as any other
   (README, "Add your own cat"):

   ```sh
   /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
       --python tools/head_portrait.py -- <small.glb> all_cats/<Name>.png
   ```
4. **Import** it as above. Meshy's heads come with a real mouth, closed, with
   a slit between the lips. Add `--own-mouth` to keep it instead of adding
   one (see below), and `--animal dog` for a dog:

   ```sh
   uv run --with trimesh --with networkx python -m catcam.head3d_import <small.glb> <Name> --own-mouth
   ```

## How the eyes and mouth work

**Eyes.** TRELLIS's eyes come out blurry and empty-looking. The import finds
the head's eyes and paints the portrait's real eyes over them in the texture,
lined up and with soft edges. Blinks reuse the photo cat's eyelids: when they
close, the app repaints those same texture spots from the eyelid drawing.
The knobs are at the top of `catcam/head3d_import.py`:
- `EYE_GROW`: how far past the eye the paint reaches.
- `EYE_FEATHER`: how soft its edge is.
- `EYE_SEARCH`: how far to look for the model's own eye.

**Mouth.** The generated head's mouth is closed and sealed, so nothing about
it is generated. When the app loads a 3D head it does three things:
- **Lip cut:** it splits the lips along the cat's mouth line, so they can
  part.
- **Inside of the mouth:** `catcam/mouth3d.py` puts a mouth in real
  geometry inside: incisors, fangs, premolars and the cheek teeth behind them;
  the gums they sit in; a palate with its ridges; a tongue with a groove down
  its middle; and a dark mouth cavity that fades toward the throat. The gums,
  palate and tongue take on the cat's own lip colour.
- **Jaw:** the lower teeth and tongue sit on a jaw that swings open about a
  hinge behind the mouth and stays with the lower lip. The mouth is only
  drawn through the lip opening, so it can't show through the cheeks or chin.

The teeth, gums, palate and tongue are modelled in Blender, one set for cats
and one for dogs (`assets/mouth/cat.npz`, `dog.npz`), and sized to each
head's lips. Cats get the cat set unless the head says otherwise: import it
with `--animal dog` for a dog's teeth. To change a set, edit the constants at
the top of `tools/mouth_model.py` (in mouth widths: each tooth's size and
place, the gums, the palate's ridges, the tongue, the colours) and run it in
Blender without its window; it rewrites both sets, and `--preview DIR` also
renders each from the front, the side and from below:

```sh
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
    --python tools/mouth_model.py -- --preview /tmp/mouth
```

The hinge and the mouth cavity are in `catcam/mouth3d.py`.

**Its own mouth instead** (`--own-mouth`): the head keeps its own lips and
inside, so no lip cut or mouth cavity mesh is added. The import finds the
head's own lower jaw on the model and cuts it free, and the app swings it open
with your mouth, with the same teeth, gums and tongue inside. All eight
shipped heads are like that:
- **Where the lips meet:** where the upper lip ends. The face fit's line
  between the lips lands on the lower lip when it misses, so the import looks
  on the model, seen from the portrait's camera, for the upper lip's lower
  edge above it: where the surface falls back steeply going down, an upper
  lip hanging over the lower one, or over a slit left open behind it (many of
  Meshy's heads have one, with teeth painted on its back). Where the model
  shows no such edge, the fit's line stays.
- **Finding the jaw:** the jaw spreads over the model's surface from its lower
  lip and chin, the rest of the face from its upper lip and nose, and the jaw
  ends where the two meet. Both also start right by the lips, a little below
  and above the line where they meet, and going into a fold (a valley in the
  surface) costs extra, so they meet in the slit between the lips rather
  than in a deeper fold nearby (a cat's chin, a grin's lower lip). The jaw
  also starts from everything hidden behind that line: the underside of the
  upper lip and the inside of an open slit go with it, so the opening starts
  right at the upper lip's edge and nothing of the lower lip stays behind on
  the face. Bumps of fur and pores don't count as folds; loose tufts of fur
  go with the surface nearest them. The jaw also takes in the skin under the
  chin, less and less of it further from the lips.
- **Cutting it free:** along the lips, the faces across that line are split
  along it, and every point on it is made twice, once for each side, so the
  lips part cleanly. Further along the jaw's edge (down the sides of the
  chin, under it) the skin stretches instead, as a real one does: the jaw
  itself moves whole, and the skin round it follows it less and less over a
  band past its edge, narrow beside the mouth and wide round the chin
  (`JAW_SOFT_NEAR`, `JAW_SOFT`), so the chin swings down in one piece. The
  cut ends soon past the mouth corners (`CORNER_CUT`), so no slit opens down
  the cheeks.
- **Its inside:** between the parted lips the model's own surface there was
  never seen, so its texture is guesswork; the app draws it, and the head's
  back faces just round it, as a mouth cavity instead. For each of those
  pixels the shader follows the ray from the camera into an egg-shaped hollow
  behind the teeth (`OWN_CAVITY` in `catcam/head3d.py`, in mouth widths: its
  roof just above the teeth, where the palate is, its floor `OWN_FLOOR` down
  and dropping further as the jaw opens) and colours the wall it hits: a
  dusky pink lining (`OWN_LINING`, taking on half the lip colour like the
  gums, `LINING_TINT`) with a soft mottle, ridges across the palate and fine
  folds down the cheeks; a deeper red throat (`OWN_THROAT`) halfway down the
  back wall, going on further back (`OWN_THROAT_DEPTH`); a floor in the
  tongue's shadow; and a dark line where the lips meet it (`LIPS_SHADOW`). A
  dog's lining has dark pigment blotches (`PIGMENT`).
  Because it is traced, it turns with the head (the cheek wall shows as you
  turn, the palate as you tilt your head back), yet it is only a backdrop:
  never drawn outside the head's own inside, so it can't poke through the
  cheeks or chin, and the teeth and tongue are always in front of it. The
  modelled palate is left out for these heads, as the cavity's roof is the
  palate.
- **Its light:** the cavity, teeth, gums and tongue share one light
  (`MOUTH_LIGHT` in `catcam/head3d.py`), to match the soft photo light baked
  into the heads: mostly light from all round that gets in only through the
  opening between the lips, as much as each point sees of it, so the inside
  darkens with depth, into the throat and under the upper lip, and brightens
  as the jaw drops (`INSIDE_OPENING`, `INSIDE_AMBIENT`); some of it bounced
  round the walls, red with the lining's colour (`INSIDE_BOUNCE`); and a
  little of the head's own light shining in through the opening, glinting
  off the wet lining in small broken glints (`INSIDE_KEY`, `INSIDE_SHEEN`).
- **The mouth inside** is sized and placed by the slit: the import saves where
  the lips meet in the middle (the upper lip's rim there, so the teeth hang
  right behind it) and how wide the slit is. Teeth or tongue that
  would poke out through the skin anywhere are not drawn there.
- **Opening it:** the jaw swings about a hinge well behind the mouth, where
  a jaw joint is (`JAW_HINGE` in `catcam/head3d.py`, in face widths, so a
  small mouth still has a long jaw), as far as your lower lip drops, up to a
  gape of `OWN_GAPE_MAX` mouth widths (but at least `OWN_GAPE_MIN` face
  widths, so a small mouth still opens wide) and `OWN_JAW_MAX`. Everything below the
  lips goes where the jaw takes it, the lower lip no lower: your chin drops
  further than the jaw opens, and skin following it there would tear off
  the jaw. That swing reaches the skin through the jaw weights alone: skin
  beside the chin that also followed the chin's face points down dropped
  with the jaw where it should stay, and the fur where it gave way was drawn
  out into pale streaks. The upper lip hangs from the nose: as yours rises (opening wide,
  smiling), the animal's moves at most `UPPER_LIP_TRAVEL` of its distance
  from the nose, so the opening never rides up into the nose. Toward the mouth corners less and
  less of the jaw swings (`CORNER_START`, `CORNER_DEPTH` in
  `catcam/head3d_import.py`), so the lips stay together there and the
  opening is lens-shaped rather than a slot. Beyond the swing the lower lip
  still does what your lips do (a smile pulls both corners up and out), and
  the two sides of the cut part only as far as the jaw swings, so a smile
  with the mouth closed shows no crack. Its rising off the jaw fades out as
  the mouth opens (`LIP_RISE_OPEN`): your lip corners drop less than your
  chin, but the animal's lips already meet at their corners, and lifted off
  the jaw as well, the lower lip drew the fur below it out into streaks.
- **The lips' edge:** the opening is not the cut's raw zigzag. Each lip's
  edge is a smooth curve fitted to the cut (`OPENING`, `OPENING_SMOOTH` in
  `catcam/head3d.py`), pushed out just far enough to cover the whole gap the
  cut leaves, with rounded corners. Where a sliver of the jaw runs up from
  the lips (Gingersnap's, up the groove under her nose), it stays with the
  face when the head loads (`SLIVER_*`), so the opening has no spike up
  toward the nose. Below the cut many heads have a recess
  before the lower lip (a slit left open, sometimes with teeth painted in
  it); as the mouth opens the opening reaches down to the lower lip's rim
  (`RIM_*`: the top of the lower lip's inner side, where the surface stops
  coming forward), so that recess becomes part of the mouth instead of
  dropping with the jaw like a flap. Inside the edge the lips roll inward: a
  thin band of their skin (`LIP_BAND`, coloured from the texture next to the
  cut, so a black-lipped cat keeps black lips) turns into the mouth's darker
  lining and shades the teeth just behind it; the upper lip's is thinnest, so
  the gum shows just inside it. Outside the edge the lip's colour covers the
  fur for a px (`LIP_OUTLINE`), then gives way to it over a px or two
  instead of ending in a hard line.
- **Teeth** are short and rounded on purpose: a friendly cartoon, not a
  predator (`tools/mouth_model.py`).

The import prints how many vertices move with the jaw. If the jaw takes a
piece of the upper lip, or leaves part of the lower lip behind, the knobs are
at the top of `catcam/head3d_import.py`: `JAW_SEEDS` and `FACE_SEEDS` (the
face points each side starts from), `LIP_SEED_GAPS` (how close to the lips
they also start), `LIP_EDGE_SEARCH` and `LIP_EDGE_STEEPER` (how far above
the fit's line to look for the upper lip's edge, and how steep it must be),
`BEHIND_LIPS` (how deep behind that edge the jaw starts from what is hidden),
`JAW_FOLD` (how strongly folds divide them), `CUT_REACH` (how far from the
lips, seen from the front, the jaw is cut rather than stretched) and
`JAW_REACH` (how far down the throat the jaw's pull fades out). If the
mouth is in the wrong place altogether, or too narrow, the face fit is off:
place the lips by hand (below), as for Woofgang, Pleadwell and Grinbert.

**The mouth line** is where the lips meet in the portrait, found when the cat
is prepared. The fit is made for cats; on other faces it can land elsewhere.
A golden retriever's lands on its nose, for example, because its mouth is a
smile whose corners are its highest points. Then write the line yourself in
`all_cats/<Name>Mouth.txt`: one `x, y` per line, in portrait pixels, from
one mouth corner to the other. Then prepare the cat again with
`--force`, and import its head again. `all_cats/WoofgangMouth.txt` is an
example. If the fit put the whole lower face too low (the nose below your
line), the face below the eyes is lifted to it first.

**Everything else** comes from the photo cat:
- **Expressions:** the head is tied to the photo cat's face points, so every
  expression moves both heads alike.
- **Ears:** they bend the same way on both heads.
- **Shading:** turning the head shades it as it does the photo cat.

## What didn't work

So you don't have to try it again:

- **Several photos of the cat** (profiles, three-quarter views, the back).
  TRELLIS.2 makes a head from a single image. We projected the other views
  onto the head as a texture, and the result showed ghosting and seams.
  TRELLIS's own texture, with the real eyes painted in, looked better. The
  front portrait is all you need.
- **No remesh.** Without `TRELLIS_REMESH=1` the head has holes, flaky
  patches of skin, and places you can see through. Remeshing rebuilds it as
  one closed surface.
- **Closing the holes with pymeshfix.** It wrecked the head.
- **A photo of the open mouth pasted inside.** It was too low-res and
  didn't match the 3D lips. The built mouth looks much better.
- **Hunyuan3D 2.** We passed on it without trying: its licence excludes the
  EU, the UK and South Korea. TRELLIS.2 is MIT licensed.

## Licences

- **TRELLIS.2:** MIT.
- **DINOv3 weights:** Meta's
  [DINOv3 licence](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md),
  whichever copy you download.
- **RMBG-2.0:** CC BY-NC 4.0, non-commercial. That's fine for a cat filter
  among friends; selling heads made with it is not.

## Troubleshooting

- **`setup.sh` fails compiling C++.** Run `tools/trellis/install.sh` on that
  folder first. It is safe to run on a folder where `setup.sh` already ran.
- **`xcrun: error` or something about a licence.** Accept Xcode's licence:
  `sudo xcodebuild -license accept`.
- **The texture has black specks, dark seams or is upside down.** The
  trellis-mac fixes weren't applied. Run `install.sh` again and look for
  errors.
- **Access errors for RMBG-2.0.** Request access on its Hugging Face page
  and log in again.
- **The eyes in the 3D head sit off the model's eyes.** The import prints how
  far it moved them. If that's tens of pixels, the portrait and the head
  disagree; try another `--seed`.
