"""MCP server tests — a real client speaking to the server over stdio.

Importing the module and asserting the decorators ran would prove almost
nothing. These spawn `reelforge-mcp` as a subprocess and drive it with the MCP
client SDK, so what is tested is the thing a Claude client actually connects to:
the handshake, the advertised tool schemas, and round-trip calls.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp", reason="mcp SDK not installed")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _params(cwd=None):
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "reelforge.mcp_server"],
        cwd=str(cwd) if cwd else None,
    )


async def _call(session: ClientSession, name: str, **kwargs) -> str:
    result = await session.call_tool(name, kwargs)
    return "\n".join(
        c.text for c in result.content if getattr(c, "type", None) == "text"
    )


async def test_server_handshakes_and_advertises_tools():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.server_info.name == "reelforge"

            tools = {t.name for t in (await session.list_tools()).tools}
            # The pipeline has to be reachable end to end over MCP, not partially.
            assert {
                "probe_media", "list_capabilities", "check_environment",
                "autocut", "transcribe", "pack_takes",
                "write_edl", "lint_edl", "render",
                "create_overlay_slot", "render_overlay_slot",
            } <= tools


async def test_every_tool_has_a_description_and_schema():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                # An agent picks tools by description; a bare name is unusable.
                assert tool.description, f"{tool.name} has no description"
                assert len(tool.description) > 40, f"{tool.name} description is too thin"
                assert tool.input_schema.get("type") == "object", tool.name


async def test_list_capabilities_reports_platforms_and_safe_zones():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            data = json.loads(await _call(session, "list_capabilities"))
            assert "reels" in data["platforms"]
            reels = data["platforms"]["reels"]
            assert reels["canvas"] == "1080x1920"
            # The action rail makes the right inset the larger one.
            assert reels["safe_zone_px"]["right"] > reels["safe_zone_px"]["left"]
            assert "karaoke" in data["caption_styles"]
            assert "track" in data["reframe_modes"]


async def test_check_environment_reports_the_toolchain():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            data = json.loads(await _call(session, "check_environment"))
            assert "ffmpeg" in data
            assert "transcription_backend" in data


async def test_errors_come_back_as_readable_text_not_tracebacks():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            out = await _call(session, "probe_media", path="/nope/missing.mp4")
            assert "not found" in out.lower()
            assert "Traceback" not in out


async def test_write_edl_rejects_malformed_json_clearly():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            out = await _call(session, "write_edl", edl_json="{not json")
            assert "not valid JSON" in out


async def test_pack_takes_explains_itself_when_there_are_no_transcripts(tmp_path):
    async with stdio_client(_params(cwd=tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            out = await _call(session, "pack_takes", directory=str(tmp_path))
            assert "transcribe" in out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
async def test_full_edit_over_mcp(tmp_path):
    """Probe, autocut, lint and render a real clip entirely through MCP calls."""
    source = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#182030:s=1280x720:d=8:r=30",
        "-f", "lavfi", "-i", "sine=frequency=320:duration=8:sample_rate=48000",
        "-filter_complex",
        "[1:a]volume='if(between(t,0,2.5)+between(t,4.5,8),1,0)':eval=frame[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(source),
    ], check=True, capture_output=True)

    async with stdio_client(_params(cwd=tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            d = str(tmp_path)

            probed = json.loads(
                await _call(session, "probe_media", path=str(source), platform="reels")
            )
            assert probed["orientation"] == "landscape"
            assert probed["needs_reframe"] is True

            cut = json.loads(
                await _call(session, "autocut", source="clip.mp4", directory=d,
                            reframe="center")
            )
            assert cut["stats"]["removed_s"] > 1.0
            assert len(cut["ranges"]) >= 2

            report = await _call(session, "lint_edl", edl_path="edl.json", directory=d)
            assert "score" in report

            rendered = json.loads(
                await _call(session, "render", edl_path="edl.json", directory=d,
                            output="out.mp4", quality="draft", captions=False)
            )
            assert rendered["size"].endswith("x1280") or "x" in rendered["size"]
            out_w, out_h = (int(v) for v in rendered["size"].split("x"))
            assert out_h > out_w, "MCP render must produce a vertical file"
            assert (tmp_path / "out.mp4").exists()


# --- the new tool surface ---------------------------------------------------


async def _content(session: ClientSession, name: str, **kwargs):
    return (await session.call_tool(name, kwargs)).content


async def test_new_tools_are_advertised():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            assert {
                "timeline_view", "review_cuts",
                "generate_asset", "still_to_clip", "list_generation_providers",
            } <= tools


async def test_capabilities_include_transitions():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            data = json.loads(await _call(session, "list_capabilities"))
            names = {t["name"] for t in data["transitions"]}
            assert {"cut", "crossfade", "dip_black", "whip_left"} <= names
            for t in data["transitions"]:
                assert t["use"], f"{t['name']} has no stated use"


async def test_generation_providers_report_offline_availability():
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            providers = json.loads(await _call(session, "list_generation_providers"))
            by_name = {p["name"]: p for p in providers}
            # Offline providers must be usable with no key configured.
            assert by_name["mock"]["usable"] is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
async def test_generate_and_clip_over_mcp(tmp_path):
    async with stdio_client(_params(cwd=tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            d = str(tmp_path)

            asset = json.loads(await _call(
                session, "generate_asset", kind="image",
                prompt="a lighthouse in fog", directory=d, provider="mock",
            ))
            assert asset["provider"] == "mock"
            assert Path(asset["path"]).exists()

            clip = json.loads(await _call(
                session, "still_to_clip", image=asset["path"],
                duration=2.0, directory=d,
            ))
            assert Path(clip["path"]).exists()

            # Music is audio-only; its duration must still come back real.
            music = json.loads(await _call(
                session, "generate_asset", kind="music", prompt="ambient bed",
                duration=3.0, directory=d, provider="mock",
            ))
            assert music["duration_s"] > 1.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
async def test_timeline_view_returns_an_actual_image(tmp_path):
    source = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#203040:s=1280x720:d=5:r=30",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=5",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(source),
    ], check=True, capture_output=True)

    async with stdio_client(_params(cwd=tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            content = await _content(
                session, "timeline_view", source="clip.mp4",
                start=0.0, end=4.0, directory=str(tmp_path), frames=6,
            )
            # The point of this tool is that the agent SEES the frames, so it
            # must come back as image content and not a path in a string.
            kinds = {getattr(c, "type", None) for c in content}
            assert "image" in kinds
            image = next(c for c in content if getattr(c, "type", None) == "image")
            assert image.mime_type.startswith("image/")
            assert len(image.data) > 500


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
async def test_review_cuts_returns_one_image_per_seam(tmp_path):
    source = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=#402030:s=1280x720:d=9:r=30",
        "-f", "lavfi", "-i", "sine=frequency=280:duration=9",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(source),
    ], check=True, capture_output=True)

    async with stdio_client(_params(cwd=tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            d = str(tmp_path)
            edl = json.dumps({
                "platform": "reels", "reframe": "center",
                "sources": {"A": "clip.mp4"},
                "ranges": [
                    {"source": "A", "start": 0.0, "end": 2.5, "beat": "HOOK"},
                    {"source": "A", "start": 3.0, "end": 5.5},
                    {"source": "A", "start": 6.0, "end": 8.5},
                ],
                "captions": {"enabled": False},
            })
            await _call(session, "write_edl", edl_json=edl, directory=d)
            await _call(session, "render", edl_path="edl.json", directory=d,
                        output="out.mp4", quality="draft", captions=False)

            content = await _content(session, "review_cuts", rendered="out.mp4",
                                     edl_path="edl.json", directory=d)
            images = [c for c in content if getattr(c, "type", None) == "image"]
            # Three ranges means two seams.
            assert len(images) == 2
