"""Tests for serving the pipeline as a remote connector.

Over stdio the server is launched by the user's own client on the user's own
machine, and the user's filesystem is exactly the right scope. Over HTTP it is a
URL on the internet whose `directory` arguments arrive from the network, and the
same code becomes a filesystem read primitive plus an open ffmpeg runner.

These tests cover the two things that close that gap: the workspace jail and the
bearer token. Both were also exercised against a live server — health open, no
token 401, wrong token 401, correct token 200.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reelforge import mcp_server as srv


@pytest.fixture
def workspace(tmp_path):
    """Serve with a jail, and always lift it afterwards.

    The root is module-global, so a test that failed to reset it would silently
    confine every later test in the session.
    """
    root = tmp_path / "workspace"
    (root / "footage").mkdir(parents=True)
    (root / "footage" / "clip.mp4").write_bytes(b"x" * 2048)
    srv.set_workspace_root(root)
    yield root
    srv.set_workspace_root(None)


# --- the jail ---------------------------------------------------------------


def test_paths_inside_the_workspace_resolve(workspace):
    assert srv._resolve_dir(str(workspace / "footage")) == workspace / "footage"


def test_a_relative_path_is_relative_to_the_workspace_not_the_cwd(workspace):
    """`directory="."` from a phone should mean the workspace, not the server's
    working directory — which is wherever the operator happened to launch it."""
    assert srv._resolve_dir(".") == workspace
    assert srv._resolve_dir("footage") == workspace / "footage"


def test_an_absolute_path_outside_the_workspace_is_refused(workspace):
    with pytest.raises(ValueError, match="outside the workspace"):
        srv._resolve_dir("/etc")


def test_dot_dot_traversal_cannot_escape(workspace):
    """Resolution happens before the check, so `..` is already collapsed.

    Checking the raw string instead would be defeated by exactly this.
    """
    with pytest.raises(ValueError, match="outside the workspace"):
        srv._resolve_dir(str(workspace / ".." / ".."))


def test_a_symlink_pointing_out_cannot_escape(workspace):
    """The other half of resolving first: a symlink is followed before checking."""
    escape = workspace / "escape"
    escape.symlink_to("/etc")
    with pytest.raises(ValueError, match="outside the workspace"):
        srv._resolve_dir(str(escape))


def test_files_are_confined_too(workspace):
    with pytest.raises(ValueError, match="outside the workspace"):
        srv._resolve_file("/etc/hosts")


def test_a_file_inside_the_workspace_resolves(workspace):
    got = srv._resolve_file("footage/clip.mp4")
    assert got == workspace / "footage" / "clip.mp4"


def test_stdio_mode_has_no_jail(tmp_path):
    """On someone's own machine a jail would only get in the way — their footage
    lives wherever they keep it, not in a directory reelforge chose."""
    srv.set_workspace_root(None)
    outside = tmp_path / "anywhere"
    outside.mkdir()
    assert srv._resolve_dir(str(outside)) == outside.resolve()


def test_the_refusal_names_the_workspace(workspace):
    """An agent hitting this needs to know where it is allowed to look."""
    with pytest.raises(ValueError) as exc:
        srv._resolve_dir("/var")
    assert str(workspace) in str(exc.value)


# --- auth -------------------------------------------------------------------


def _middleware_response(token: str, header: str | None, path: str = "/mcp"):
    """Drive the middleware directly, without a live server."""
    import anyio
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    app = Starlette(routes=[
        Route(path, lambda r: JSONResponse({"reached": True}), methods=["GET", "POST"]),
        Route("/health", lambda r: JSONResponse({"unused": True})),
    ])
    app.add_middleware(srv._auth_middleware(token))
    del anyio
    client = TestClient(app)
    headers = {"authorization": header} if header else {}
    return client.get(path, headers=headers)


def test_a_correct_token_is_let_through():
    r = _middleware_response("s" * 32, f"Bearer {'s' * 32}")
    assert r.status_code == 200
    assert r.json() == {"reached": True}


def test_a_missing_token_is_rejected():
    r = _middleware_response("s" * 32, None)
    assert r.status_code == 401


def test_a_wrong_token_is_rejected():
    r = _middleware_response("s" * 32, "Bearer " + "w" * 32)
    assert r.status_code == 401


def test_a_token_without_the_bearer_prefix_is_rejected():
    r = _middleware_response("s" * 32, "s" * 32)
    assert r.status_code == 401


def test_the_bearer_prefix_is_case_insensitive():
    """Clients differ on capitalisation; the token is the secret, not the word."""
    assert _middleware_response("s" * 32, f"bearer {'s' * 32}").status_code == 200


def test_a_token_that_is_a_prefix_of_the_real_one_is_rejected():
    """Guards against a comparison that stops at the first difference."""
    assert _middleware_response("s" * 32, "Bearer " + "s" * 16).status_code == 401


def test_health_needs_no_credential():
    """Otherwise every uptime monitor and load balancer needs the secret just to
    ask whether the port is up."""
    r = _middleware_response("s" * 32, None, path="/health")
    assert r.status_code == 200
    assert r.json()["service"] == "reelforge"


def test_the_rejection_says_what_to_send():
    r = _middleware_response("s" * 32, None)
    assert "Bearer" in r.json()["error"]
