---
name: reelforge
description: Make vertical video — Instagram Reels, TikToks, Shorts — by conversation. Transcribe, choose takes, cut, join clips with transitions, reframe landscape footage to vertical with subject tracking, caption, overlay motion graphics, generate footage/narration/music when there is none, and render posting-ready. Use when the user has video files to cut for a vertical feed, or asks to make a Reel, Short or TikTok, with or without source footage.
---

# reelforge

## What this is for

The user has footage and wants a vertical post out of it, quickly and repeatedly.
You read a transcript, decide what the video is, write an EDL, and render.

You do not watch the footage frame by frame. You read `takes.md` — a compact,
timestamped, phrase-level transcript — and drill into visuals only when a
specific decision needs it. Thirty thousand frames is not a thing to look at;
twelve kilobytes of text is.

## Principles

1. **Audio is the spine.** Cut points come from word boundaries and silences.
   Every silence in `takes.md` is already a cut candidate.
2. **Ask, propose, confirm, then execute.** Never touch the cut before the user
   has agreed to a plan in plain English. The plan is four to eight sentences,
   not a document.
3. **Look at the material before deciding what it is.** Do not assume talking
   head, tutorial, or promo. Read the transcript, then ask.
4. **The linter is a colleague, not a gate.** Run it before rendering. It will
   tell you the hook is late or a shot runs eleven seconds. Act on it or
   consciously decide not to.
5. **Taste is yours.** Every style, palette, grade and pacing choice below is a
   worked example. The Hard Rules are not.

## Hard rules

Breaking these produces silent, shipped defects.

1. **Never cut inside a word.** Use the timestamps in `takes.md` — they already
   sit on word boundaries.
2. **The first frame is the whole hook.** Speech starts within ~0.5s or the
   opening is spent. Trim dead head every time.
3. **Never reframe blind.** Landscape source in a vertical target always gets a
   `reframe` mode chosen on purpose. The default `track` is right for people;
   `blur_pad` is right for screencasts and anything with edge text.
4. **Captions are burned last.** The renderer handles this. Do not hand-build an
   ffmpeg chain that composites overlays after subtitles.
5. **Never re-transcribe.** Transcripts cache against source content. If you
   think you need a re-run, you are wrong unless the file itself changed.
6. **Outputs stay out of the footage directory root.** Everything derived lives
   in `.reelforge/`; renders go where the user asks.
7. **Confirm before the first render, not after.**
8. **Report what the linter says.** If you ship with warnings outstanding, say
   which and why.

## Two routes

**Fast route — no transcript.** When the ask is "tighten this up" or "make this
vertical", and there is one take rather than several to choose between:

```bash
reelforge autocut clip.mp4 --grade punch
reelforge render edl.json -q preview -o preview.mp4
```

`autocut` finds dead air in the waveform and writes an EDL. No ASR, no network,
no key. Still confirm the plan with the user first — but the plan is one
sentence, not a strategy document.

**Full route — transcript-driven.** When there are multiple takes, when the
structure needs rearranging, or when captions are wanted. This is the rest of
this document.

**Generated route — no footage.** When the user has nothing to cut, generate it:
`reelforge generate` for stills, video, narration and music, then `reelforge
clip` to turn a still into a shot. Check `--list-providers` first and say which
one will run — with no API key the offline providers produce placeholders and
robotic speech, which are right for building and timing a cut and wrong for
anything anyone will watch. Never present placeholder output as finished.

Reach for the fast route when it genuinely fits. Transcribing a single clean
take to remove three pauses is ceremony.

## Looking at the footage

`reelforge view <source> <start> <end>` renders a filmstrip over the waveform.
Over MCP it comes back as an image you can actually see.

Use it at decision points the transcript cannot settle — did the subject stay in
frame, does this cut flash, which take is framed better. Do not scan a whole
timeline with it; it is slow and mostly shows nothing.

Before showing any render to the user, run `review_cuts` on it. One filmstrip
per seam, and you look for: a flash or jump at the boundary, the subject leaving
frame, captions hidden behind an overlay, an overlay showing the wrong frames.
Fix and re-render, up to three passes, then tell the user what you could not
resolve rather than looping.

## The loop

```bash
reelforge doctor                     # once, on a cold start
reelforge init -p reels              # scan sources, scaffold .reelforge/
reelforge transcribe                 # word-level ASR, cached
reelforge pack --intent "<brief>"    # → .reelforge/takes.md
#  read takes.md, converse, agree a plan, write edl.json
reelforge lint edl.json              # fix what it finds
reelforge render edl.json -q preview -o preview.mp4
#  show the user, iterate
reelforge render edl.json -q final -o final.mp4
```

Read `.reelforge/project.md` first if it exists, and summarise the last session
in one sentence before asking whether to continue.

## Step 1 — inventory and read

Run `init`, `transcribe`, `pack`. Then read `takes.md` in full.

While reading, note in one pass: verbal stumbles, repeated takes of the same
line, the strongest single sentence in the material, and anything that would
make a good opening frame. You will need all four in the next step.

## Step 2 — converse

Describe what you actually see, in plain English, in a few sentences. Then ask
questions **shaped by this material** — not a fixed checklist. Typically you
need to establish:

- what the video is for, and who sees it
- target platform and rough length
- which moments must survive
- tone: is this energetic and cut-heavy, or calm and let-it-breathe
- whether they want motion graphics, and whether there is a brand palette

Ask about what is genuinely ambiguous. If the footage is one clean take of
someone explaining a product, do not ask whether it is a montage.

## Step 3 — propose

Four to eight sentences: the shape, which takes you are using, roughly how many
cuts, reframing approach, caption style, grade direction, and an estimated
runtime. **Wait for agreement.**

## Step 4 — write the EDL

```json
{
  "version": 2,
  "platform": "reels",
  "intent": "one line, carried into the linter",
  "reframe": "track",
  "grade": "punch",
  "sources": { "A": "take-01.mp4" },
  "ranges": [
    {
      "source": "A",
      "start": 12.4,
      "end": 17.02,
      "beat": "HOOK",
      "quote": "we tried this for ninety days",
      "reason": "cleanest delivery; the 31.2s take stumbles on 'ninety'"
    }
  ],
  "captions": { "style": "punch" }
}
```

Fill in `beat` and `reason` on every range. They cost nothing, they let the
linter check structure, and they are what makes the cut auditable when the user
asks why you dropped something.

Structural archetypes worth adapting — or ignore them and invent one:

| shape            | beats                                                         |
| ---------------- | ------------------------------------------------------------- |
| Product / launch | HOOK → PROBLEM → SOLUTION → PROOF → CTA                       |
| Story            | HOOK → SETUP → TURN → RESOLUTION                              |
| Tutorial         | HOOK → WHAT YOU NEED → STEPS → RESULT                         |
| Listicle         | HOOK → ITEM ×N → PAYOFF                                       |
| Interview        | HOOK (best line, pulled forward) → CONTEXT → EXCHANGE → CLOSE |

Pulling the strongest line to the front and letting context follow is the single
highest-yield structural move in short form. Chronological order is a default,
not a requirement.

## Step 5 — reframe deliberately

| mode       | use when                                                    |
| ---------- | ----------------------------------------------------------- |
| `track`    | A person is the subject. Default.                           |
| `static`   | The subject barely moves; avoids any pan at all.            |
| `center`   | Composition is already centred and you want zero surprises. |
| `blur_pad` | Screencasts, demos, anything where cropping severs content. |
| `fit`      | You genuinely want letterboxing.                            |

Set it per range when the material changes: a talking-head opener followed by a
screen recording wants `track` then `blur_pad`.

`detector` is `auto` by default — face detection with a saliency fallback. Force
`saliency` for footage with no faces, `face` when you know there is one and want
it prioritised.

## Step 6 — joining clips

Cuts are the default and are usually right. Soften a seam only when it earns it:

| transition                 | when                                                        |
| -------------------------- | ----------------------------------------------------------- |
| `crossfade`                | two angles of one moment, or a gentle time passage          |
| `dip_black`                | a hard break between sections — the strongest punctuation   |
| `dip_white`                | brighter, more energetic; product and reveals               |
| `whip_left` / `whip_right` | fast location or subject change; alternate direction        |
| `blur`                     | shots that share no visual anchor                           |
| `slide_up`                 | forward motion through a sequence; native to vertical feeds |
| `zoom`                     | a reveal or a sharp escalation in energy                    |

```json
{ "source": "B", "start": 3.0, "end": 7.5, "transition": "crossfade" }
```

The seam belongs to the incoming clip. Two things to hold on to: a transition
**consumes timeline time**, so the render is shorter than the sum of the ranges;
and dissolving every cut is the single fastest way to make a Reel look like a
2009 holiday slideshow. Most seams should stay cuts.

## Step 7 — captions

`punch` is the default and is right most of the time. `karaoke` and `pop`
highlight word by word and suit fast, energetic delivery. `clean` and `boxed`
suit explainers and anything an audience will read rather than feel.

Placement is computed from the platform's safe area. Do not hand-position
captions; change `platform` and they move correctly on their own.

## Step 8 — motion graphics (optional)

Overlay slots are HyperFrames compositions rendered to alpha WebM.

```bash
reelforge slot statcard -t 4.0 --brief "count 0 to 340 with a label"
```

That scaffolds a composition already sized to the delivery target with the safe
area pre-inset, and prints a self-contained brief. **Spawn one sub-agent per
slot, in parallel** — never sequentially. Each brief is self-contained because
sub-agents inherit no context.

Then add to the EDL:

```json
"overlays": [
  { "file": ".reelforge/slots/slot_statcard/render.webm",
    "start_in_output": 6.2, "duration": 4.0,
    "anchor": "center", "fade_in": 0.2, "fade_out": 0.3 }
]
```

Timing rule that matters: if the overlay lands on a spoken payoff word, start it
`reveal_duration` seconds _earlier_ so the landing frame coincides with the word.
Without that it reads as disconnected.

## Step 9 — lint, render, verify

`reelforge lint edl.json` before rendering. Fix what it finds or decide not to
and say so.

Render `-q preview` first and actually check it before showing the user. Sample
frames at cut boundaries and confirm: no flash at the seam, the subject is in
frame throughout, captions are legible and clear of the UI, overlays are where
you meant them. `render` runs the linter against the output automatically —
read that report.

If something is wrong: fix, re-render, re-check. Cap at three passes, then tell
the user what you could not resolve rather than looping.

## Step 10 — persist

Append to `.reelforge/project.md`:

```markdown
## Session N — YYYY-MM-DD

**Strategy:** one paragraph
**Decisions:** takes chosen, cuts, reframing, grade, captions — and why
**Outstanding:** anything deferred
```

## Anti-patterns

- Rendering before the user has agreed a plan.
- Reading raw transcript JSON instead of `takes.md`.
- Sampling frames to "understand" the footage. Read the transcript.
- Leaving a landscape source in a vertical target without choosing a reframe mode.
- Hand-writing ffmpeg chains. The renderer's ordering exists for reasons; a
  bespoke chain will composite overlays over captions and hide them.
- Sequential sub-agents for multiple overlay slots.
- Re-transcribing cached sources.
- Shipping with linter warnings you have not mentioned.
- Assuming the video is chronological. The best line usually belongs at the top.
- Captions at the very bottom of frame. The platform UI covers them; the safe
  area exists to stop this and only works if you let it.
- A transition on every seam. Cuts are the default for a reason.
- Presenting placeholder or `espeak` output as finished work. Say which
  provider ran.
- Showing a render you have not run `review_cuts` on.
