# reelforge

Agentic editor for vertical video. Drop footage in a folder, describe the video
you want in plain English, get a posting-ready Reel.

It is built on one observation: an LLM is very good at deciding _which words
should survive the cut_, and very bad at looking at 30,000 frames. So the agent
reads a compact transcript and writes an edit decision list; deterministic Python
does the frame-accurate work.

```
transcribe → pack → agent reasons → EDL → reframe → render → lint → post
```

## Why this exists

It combines three upstream projects and closes the gap between them.

| Source                                                   | What it contributes                                                         | What it could not do                                                      |
| -------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [video-use](https://github.com/browser-use/video-use)    | Transcript-as-interface, the EDL/render spine, production-correctness rules | No reframing — landscape footage stays landscape                          |
| [OpenMontage](https://github.com/calesthio/OpenMontage)  | Pipeline governance, provider breadth, self-review gates                    | Twelve pipelines and 100+ tools is a lot of ceremony for a 30-second Reel |
| [HyperFrames](https://github.com/heygen-com/hyperframes) | HTML/GSAP motion graphics with deterministic frame capture                  | Composition engine, not an editor — no source footage, no cuts            |

The three things reelforge adds:

**Subject-tracked reframing.** Landscape footage becomes a 9:16 Reel with the
speaker held in frame. On a clip where the subject crosses the frame, a fixed
centre crop loses them entirely in a third of frames; the tracked crop holds them
for all of them. This is the capability none of the three upstream projects have,
and it is the difference between posting to Instagram and posting _at_ it.

**Safe zones as a first-class constraint.** Captions and overlays are positioned
against the region each platform's UI actually leaves free, not against a fixed
margin that happens to work on one device. Change `platform` and everything moves.

**A retention linter.** `reelforge lint` inspects the edit before you render:
hook strength, pace, dead air, shot-length monotony, caption legibility, safe-zone
violations. It reports what to fix while fixing is still cheap.

## Scope — what is and isn't here

|                                       | status                                                                                                                                                                                                           |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **video-use** — the editing half      | Ported and extended, including `timeline_view`. Its automated three-pass self-eval loop is not ported; `review_cuts` gives an agent the same frames to judge from.                                               |
| **OpenMontage** — the generation half | Image, video, speech and music generation behind a provider registry, with offline fallbacks. Not ported: avatar/lipsync, the 12-pipeline system, Backlot UI, Remotion composer, upscaling and face restoration. |
| **HyperFrames**                       | Integrated as a bridge — scaffold, lint and render overlay slots. The 50+ registry blocks and motion-doctrine skills are not wrapped.                                                                            |

Footage in, edited vertical video out is complete and tested. Generation reaches
far enough to build a video from nothing — stills, motion, narration and a music
bed — but it is a thin provider layer, not OpenMontage's scored registry across
a dozen vendors.

## Generation

Four asset kinds behind one interface, cached by request hash so an identical
prompt never bills or waits twice.

```bash
reelforge generate --list-providers
reelforge generate "a lighthouse in fog" -k image
reelforge generate "calm ambient bed" -k music -t 20
reelforge clip .reelforge/assets/image_*/asset.png -t 3.0   # still -> moving shot
```

| provider     | kinds         | needs                                         |
| ------------ | ------------- | --------------------------------------------- |
| `replicate`  | image, video  | `REPLICATE_API_TOKEN` — FLUX, Kling, Veo      |
| `openai`     | image, speech | `OPENAI_API_KEY`                              |
| `elevenlabs` | speech, music | `ELEVENLABS_API_KEY`                          |
| `espeak`     | speech        | nothing — local, robotic, **real durations**  |
| `mock`       | all four      | nothing — deterministic labelled placeholders |

`auto` prefers a credentialled cloud provider and falls back to offline, so a
call always yields a file; the returned provider tells you which ran. The
offline pair is what makes the generate-to-render path developable with nothing
spent — assemble and time a rough cut on placeholders, then swap in real assets.

A generated still needs `still_to_clip` before it belongs on a timeline. A
motionless still in a feed reads as a loading error; the slow push is what makes
it read as a shot.

## Transitions

Cuts are the default. Thirteen seam treatments when a cut isn't enough —
`crossfade`, `dip_black`, `dip_white`, `whip_left`/`whip_right`, `blur`,
`slide_left`/`slide_up`, `zoom`, `pixelize`, `circle`, `dissolve`.

```json
{ "source": "B", "start": 3.0, "end": 7.5, "transition": "crossfade", "transition_duration": 0.4 }
```

The seam belongs to the incoming clip, so `transition` describes how a range
_enters_. **A transition consumes timeline time** — two 3s clips joined by a
0.5s crossfade run 5.5s, not 6s — and every offset accounts for the overlap, so
captions and overlays stay aligned across seams.

## Looking at footage

The transcript answers what was said and when. It cannot tell you whether the
subject left frame or whether a cut flashes.

```bash
reelforge view take-01.mp4 12.0 18.0     # filmstrip + waveform PNG
```

Over MCP this comes back as an image the agent actually sees, and `review_cuts`
renders one per seam of a finished file — a self-review pass before anything is
shown to you.

## Install

```bash
pip install -e ".[all]"        # face tracking + hosted transcription
pip install -e .               # core only; saliency tracking, no ASR
```

Requires `ffmpeg` and `ffprobe` on `PATH`.

The `vision` extra (OpenCV) enables face-aware tracking. Without it the reframer
falls back to a gradient/motion saliency estimate that needs only numpy — decent
on b-roll and screencasts, weaker on faces.

## Use

The fast path — trim dead air and go vertical, no transcript required:

```bash
cd ~/footage/my-reel

reelforge autocut take-01.mp4 --grade punch   # → edl.json
reelforge render edl.json -o final.mp4
```

The full path, when you want the cut chosen on what was actually said:

```bash
reelforge init                          # scan sources, scaffold the project
reelforge transcribe                    # word-level ASR, cached per source
reelforge pack                          # → takes.md, the agent's reading view
#   ... the agent reads takes.md and writes edl.json ...
reelforge lint edl.json                 # retention + correctness report
reelforge render edl.json -q preview    # look at it
reelforge render edl.json -o final.mp4  # post it
```

## Transcription

Two backends, same output shape. `auto` picks hosted when a key is present.

| backend   | needs                    | trade-off                                                                                                        |
| --------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `scribe`  | `ELEVENLABS_API_KEY`     | Verbatim — fillers and stumbles survive as the signal for choosing between takes. Audio is uploaded.             |
| `whisper` | `pip install '.[local]'` | No key, nothing leaves the machine. Slower on CPU, and it normalises fillers away, which weakens take selection. |

```bash
reelforge transcribe --backend whisper --model base
```

Neither is required to edit. `autocut` works from the waveform, and captions
only need a transcript file — drop one at `.reelforge/transcripts/<source>.json`
from any source and everything else works.

## MCP

The pipeline is also an MCP server, so Claude can call it directly instead of
shelling out.

```bash
pip install -e ".[mcp]"
claude mcp add reelforge -- reelforge-mcp
```

Or in `claude_desktop_config.json` / `.mcp.json`:

```json
{ "mcpServers": { "reelforge": { "command": "reelforge-mcp" } } }
```

Then just talk to it: _"make this landscape clip into a Reel."_

| tool                                          | does                                                           |
| --------------------------------------------- | -------------------------------------------------------------- |
| `check_environment`                           | what this machine can actually do                              |
| `probe_media`                                 | geometry, duration, whether it needs reframing                 |
| `list_capabilities`                           | platforms, safe zones, styles, grades, transitions             |
| `autocut`                                     | dead-air trim → EDL, no transcript needed                      |
| `transcribe`                                  | word-level ASR, cached                                         |
| `pack_takes`                                  | **returns the transcript view inline** for cut planning        |
| `write_edl`                                   | validate and save an edit                                      |
| `lint_edl`                                    | **returns the retention report inline**                        |
| `render`                                      | the full render pipeline                                       |
| `timeline_view`                               | **returns a filmstrip + waveform image** — look at the footage |
| `review_cuts`                                 | **one image per seam** of a render, for self-review            |
| `generate_asset`                              | image / video / speech / music from a prompt                   |
| `still_to_clip`                               | generated still → a shot with a slow push                      |
| `list_generation_providers`                   | which providers are usable right now                           |
| `create_overlay_slot` / `render_overlay_slot` | HyperFrames motion graphics                                    |

The reading tools return their content inline rather than writing a file, and
`timeline_view` / `review_cuts` return real images. That is the point of the MCP
surface: an agent that can read the transcript, see the frames and judge its own
cut, rather than one parsing stdout.

Or drive it as a plain skill instead — see [`SKILL.md`](SKILL.md).

## The EDL

The only artifact the agent writes. Everything else is derived.

```json
{
  "version": 2,
  "platform": "reels",
  "reframe": "track",
  "grade": "punch",
  "sources": { "A": "raw/take-01.mp4" },
  "ranges": [
    {
      "source": "A",
      "start": 12.4,
      "end": 17.02,
      "beat": "HOOK",
      "quote": "we tried this for ninety days",
      "reason": "cleanest delivery; the 14.9s take stumbles on 'ninety'"
    }
  ],
  "captions": { "style": "karaoke", "words_per_cue": 4 },
  "overlays": [
    {
      "file": "overlays/stat.webm",
      "start_in_output": 3.2,
      "duration": 4.0,
      "anchor": "center",
      "scale": 0.8,
      "fade_in": 0.2
    }
  ]
}
```

Ranges carry `beat` and `reason` because they are read by humans auditing the
cut, and by the linter when it scores structure.

## Reframing

| mode       | behaviour                                                                             |
| ---------- | ------------------------------------------------------------------------------------- |
| `track`    | Subject-tracked crop. Holds still, glides when the subject genuinely moves.           |
| `static`   | One fixed crop per range, placed where the subject spent most of the take.            |
| `center`   | Plain centre crop.                                                                    |
| `blur_pad` | Whole frame over a blurred backdrop — for screencasts, where cropping severs content. |
| `fit`      | Letterbox. Honest, and occasionally correct.                                          |

`track` does not follow the subject frame by frame; that jitters. It behaves like
an operator on a tripod: hold the frame, ignore drift below a deadband, and when
a move is needed, ease into it under a speed cap.

## Captions

`punch` (2 words, huge), `karaoke` (word-by-word colour), `pop` (word-by-word
scale), `clean` (sentence case), `boxed` (opaque block).

Rendered as ASS so individual words can be styled, positioned from the target's
safe area, and burned _after_ overlays — an overlay composited on top of captions
hides them silently.

## Layout

```
src/reelforge/
  config.py      platform specs + safe zones — the only place constants live
  ffmpeg.py      probing, filter fragments, error handling
  edl.py         schema, validation
  reframe.py     subject tracking + the virtual camera operator
  captions.py    cue building, ASS/SRT emission
  grade.py       colour presets
  render.py      the pipeline
  retention.py   the linter
  overlays.py    HyperFrames bridge
  transcribe.py  word-level ASR
  cli.py
```

## Licence

MIT.
