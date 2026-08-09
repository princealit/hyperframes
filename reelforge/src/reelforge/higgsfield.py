"""Higgsfield provider — the verified path for talking heads.

The `avatar` module's default stack (Fish for voice, sync.so for lipsync) is
three network stages that must be wired together, and none of it could be
exercised from the environment it was written in. This module is the opposite:
every call here has been run against the live API, and the whole talking-head
job collapses into **two** calls instead of three.

The collapse is the point. Higgsfield's `wan2_7` accepts an `audio_references`
media role alongside a `start_image`, which means "photo + speech -> the person
saying it" is one generation. There is no separate animate stage, because the
model is already generating motion, and no separate lipsync stage, because the
model is already conditioned on the audio. The driver-extension problem that
`avatar.extend_driver` exists to solve does not arise either: the model is
generating the frames, so it generates as many as the audio needs.

    script ──→ [generate_audio: cloned voice] ──┐
                                                 ├──→ [wan2_7] ──→ talking video
    photo ───────────────────────────────────────┘

What this module does NOT do is talk to Higgsfield directly. Higgsfield is
reached over MCP, and reelforge is a library that cannot assume an MCP client is
present. So this is the *planning and assembly* half: it builds the exact tool
calls an agent should make, validates what comes back, and turns the finished
URLs into local files on the timeline. The agent makes the calls.

That split is deliberate. It keeps the credit-spending decisions in the agent's
hands — where the user can see and approve them — while the parts that benefit
from being deterministic and tested (cost arithmetic, duration reconciliation,
download and probe) stay here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .ffmpeg import media_duration, probe

#: Verified live. Costs are credits, read from the API's own `get_cost`
#: preflight rather than a published price list, because the preflight is what
#: the account is actually charged.
TTS_MODEL = "seed_audio"
TALKING_MODEL = "wan2_7"

#: Measured against the live API on 2026-08-09.
COST_TTS = 0.2
COST_TALKING_5S_720P = 7.5

#: `wan2_7` accepts 2-15s. Anything longer has to be generated as segments and
#: joined, because the model will not produce a single clip beyond this.
MAX_SEGMENT_S = 15.0
MIN_SEGMENT_S = 2.0


class HiggsfieldError(RuntimeError):
    pass


@dataclass
class ToolCall:
    """One MCP call for the agent to make, with why it exists."""

    tool: str
    params: dict[str, Any]
    purpose: str
    est_credits: float = 0.0

    def render(self) -> str:
        return f"{self.tool}({json.dumps(self.params, ensure_ascii=False)})"


@dataclass
class TalkingHeadPlan:
    """A costed, ordered plan the agent executes call by call."""

    calls: list[ToolCall] = field(default_factory=list)
    segments: int = 1
    est_credits: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{len(self.calls)} call(s), ~{self.est_credits:.1f} credits"]
        for i, c in enumerate(self.calls, 1):
            lines.append(f"  {i}. {c.tool:<16} {c.purpose}  (~{c.est_credits:.1f})")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def plan_speech(
    text: str,
    voice_id: str,
    *,
    voice_type: Literal["preset", "element"] = "element",
) -> ToolCall:
    """The TTS call.

    `voice_type='element'` selects a voice cloned into the workspace, which is
    the one worth using — a preset is a stranger's voice. Language is not a
    parameter: the model reads the script, so Persian text in Persian script
    produces Persian without a flag and without transliterating into Arabic.
    """
    if not text.strip():
        raise HiggsfieldError("empty script")
    return ToolCall(
        tool="generate_audio",
        params={
            "model": TTS_MODEL,
            "prompt": text,
            "voice_type": voice_type,
            "voice_id": voice_id,
            "use_unlim": False,
        },
        purpose="speak the script in the cloned voice",
        est_credits=COST_TTS,
    )


def plan_talking_head(
    image_media_id: str,
    audio_job_id: str,
    *,
    duration: float = 5.0,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    prompt: str | None = None,
) -> ToolCall:
    """The photo + audio -> talking video call.

    `medias[].value` takes a media UUID *or* a prior job id, which is what makes
    the two-call chain work: the audio generated in step one is referenced
    directly by its job id, with nothing downloaded and re-uploaded in between.

    The default prompt asks for subtle motion over a static background on
    purpose. Large gestures pull the face off-axis and the mouth stops tracking;
    a moving background makes any later loop or extension visibly seam.
    """
    if not (MIN_SEGMENT_S <= duration <= MAX_SEGMENT_S):
        raise HiggsfieldError(
            f"{duration:.1f}s is outside {TALKING_MODEL}'s {MIN_SEGMENT_S}-"
            f"{MAX_SEGMENT_S}s range — split the script into segments"
        )
    return ToolCall(
        tool="generate_video",
        params={
            "model": TALKING_MODEL,
            "prompt": prompt or (
                "The person speaks directly to camera with natural head movement "
                "and subtle breathing, lips synchronized to the spoken audio, "
                "static background, warm even lighting"
            ),
            "duration": int(round(duration)),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "medias": [
                {"role": "start_image", "value": image_media_id},
                {"role": "audio_references", "value": audio_job_id},
            ],
            "use_unlim": False,
        },
        purpose=f"{duration:.0f}s of the subject speaking, {aspect_ratio}",
        est_credits=COST_TALKING_5S_720P * max(1.0, duration / 5.0),
    )


def plan(
    script: str,
    voice_id: str,
    image_media_id: str,
    *,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    words_per_second: float = 2.3,
) -> TalkingHeadPlan:
    """Cost and order the whole job before spending anything.

    The speech length is *estimated* here rather than measured, because the plan
    is built before the audio exists. That estimate only decides how many video
    segments to budget for; the real duration is read back from the generated
    audio, and `reconcile` exists to correct the plan once it is known.

    `words_per_second` defaults to a deliberately unhurried 2.3 — over-estimating
    the duration costs a little more in the quote and under-estimating truncates
    the script, so the error is pushed toward the harmless side.
    """
    words = len(script.split())
    est_seconds = max(MIN_SEGMENT_S, words / words_per_second)

    p = TalkingHeadPlan()
    p.calls.append(plan_speech(script, voice_id))

    if est_seconds <= MAX_SEGMENT_S:
        p.segments = 1
        p.calls.append(plan_talking_head(
            image_media_id, "<audio_job_id from step 1>",
            duration=est_seconds, aspect_ratio=aspect_ratio, resolution=resolution,
        ))
    else:
        # Long scripts have to be cut into segments and joined. Splitting on
        # sentence boundaries rather than on a clock keeps each segment's audio
        # a complete thought, so the joins land in natural pauses.
        p.segments = int(est_seconds // MAX_SEGMENT_S) + 1
        p.notes.append(
            f"~{est_seconds:.0f}s exceeds {TALKING_MODEL}'s {MAX_SEGMENT_S:.0f}s "
            f"ceiling — split into {p.segments} segments on sentence boundaries "
            "and join with reelforge's EDL"
        )
        for i in range(p.segments):
            p.calls.append(plan_talking_head(
                image_media_id, f"<audio_job_id for segment {i + 1}>",
                duration=MAX_SEGMENT_S, aspect_ratio=aspect_ratio,
                resolution=resolution,
            ))

    p.est_credits = sum(c.est_credits for c in p.calls)
    p.notes.append(f"estimated from {words} words at {words_per_second}/s")
    return p


def split_script(script: str, max_seconds: float = MAX_SEGMENT_S,
                 words_per_second: float = 2.3) -> list[str]:
    """Split a long script into segment-sized chunks on sentence boundaries.

    Splitting mid-sentence would put a generation seam inside a clause, where
    any mismatch in head position between segments reads as a glitch. A sentence
    boundary already carries a pause, so the seam hides in it.

    Handles Persian and Arabic punctuation (`؟` `،` `۔`) alongside Latin, since
    the whole reason for choosing these models was non-Latin script support.
    """
    import re

    budget_words = int(max_seconds * words_per_second)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?؟۔])\s+", script) if s.strip()]

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = current + [sentence]
        if sum(len(s.split()) for s in candidate) > budget_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks or [script]


def reconcile(plan_obj: TalkingHeadPlan, actual_audio_seconds: float) -> TalkingHeadPlan:
    """Correct a plan once the real speech duration is known.

    The estimate in `plan` is words-per-second guesswork. Once the audio exists
    its duration is a fact, and a video generated to the guessed length either
    truncates the speech or pads it with silence. This rewrites the video calls
    to the measured duration.
    """
    corrected = TalkingHeadPlan(notes=list(plan_obj.notes))
    corrected.notes.append(
        f"reconciled to measured audio: {actual_audio_seconds:.2f}s"
    )
    speech_calls = [c for c in plan_obj.calls if c.tool == "generate_audio"]
    corrected.calls.extend(speech_calls)

    remaining = actual_audio_seconds
    idx = 0
    while remaining > 0.05:
        seg = min(MAX_SEGMENT_S, max(MIN_SEGMENT_S, remaining))
        video_calls = [c for c in plan_obj.calls if c.tool == "generate_video"]
        template = video_calls[min(idx, len(video_calls) - 1)] if video_calls else None
        if template is None:
            break
        params = {**template.params, "duration": int(round(seg))}
        corrected.calls.append(ToolCall(
            tool="generate_video", params=params,
            purpose=f"segment {idx + 1}: {seg:.0f}s",
            est_credits=COST_TALKING_5S_720P * max(1.0, seg / 5.0),
        ))
        remaining -= seg
        idx += 1

    corrected.segments = idx
    corrected.est_credits = sum(c.est_credits for c in corrected.calls)
    return corrected


def fetch(url: str, out_path: Path, *, timeout: int = 600) -> Path:
    """Download a finished generation to the local timeline.

    Uses curl rather than `requests` because it is the one HTTP dependency
    reelforge can assume everywhere, and this is the only place the library
    itself reaches the network.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl") is None:
        raise HiggsfieldError("curl is required to download generated media")
    result = subprocess.run(
        ["curl", "-fsSL", "--max-time", str(timeout), "-o", str(out_path), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise HiggsfieldError(
            f"download failed (curl {result.returncode}): {result.stderr[:200]}"
        )
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise HiggsfieldError(f"downloaded file is empty or truncated: {out_path}")
    return out_path


def collect(
    result_urls: list[str],
    work_dir: Path,
    *,
    stem: str = "talking",
) -> list[Path]:
    """Download finished segments and verify each is real, playable media.

    Verification is not ceremony: a generation endpoint that returns 200 with an
    error page produces a file that only fails later, during render, where the
    cause is much harder to see.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, url in enumerate(result_urls):
        suffix = ".wav" if url.rsplit(".", 1)[-1].lower() == "wav" else ".mp4"
        path = fetch(url, work_dir / f"{stem}_{i:02d}{suffix}")
        try:
            info = probe(path)
        except Exception as e:  # noqa: BLE001
            raise HiggsfieldError(f"{path.name} is not playable media: {e}") from None
        if info.duration <= 0:
            raise HiggsfieldError(f"{path.name} has zero duration")
        out.append(path)
    return out


def to_edl(
    segments: list[Path],
    project_dir: Path,
    *,
    platform: str = "reels",
) -> dict[str, Any]:
    """Turn finished segments into an EDL, ready for the normal pipeline.

    This is the join back to the rest of reelforge. Once the talking head is an
    EDL it is just footage: caption it, grade it, cut other shots against it,
    lint it, render it. Nothing downstream needs to know it was generated.
    """
    if not segments:
        raise HiggsfieldError("no segments to assemble")

    sources: dict[str, str] = {}
    ranges: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        name = seg.stem
        try:
            rel = seg.resolve().relative_to(project_dir.resolve()).as_posix()
        except ValueError:
            rel = str(seg)
        sources[name] = rel
        ranges.append({
            "source": name,
            "start": 0.0,
            "end": round(media_duration(seg), 3),
            # The first segment carries the hook; everything after is body.
            "beat": "HOOK" if i == 0 else "BODY",
        })

    return {
        "version": 2,
        "platform": platform,
        "sources": sources,
        "ranges": ranges,
    }
