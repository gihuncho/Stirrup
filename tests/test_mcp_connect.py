"""Connecting an MCPToolProvider to several servers at once."""

import os
import signal
import sys
import time
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import anyio
import pytest

from stirrup.tools.mcp import MCPConfig, MCPToolProvider

pytest.importorskip("mcp.server.fastmcp")

#: What the servers in the timing test sleep before serving, on top of the
#: interpreter and FastMCP import they already pay, so that startup dominates.
STARTUP_DELAY_S = 1.0
SERVER_COUNT = 5

#: How long a failing server stays quiet before giving up, rather than failing
#: the moment it is spawned: long enough for the healthy servers alongside it to
#: have finished connecting, which is what puts a half-connected provider and
#: its processes at risk in the first place.
FAILS_AFTER_S = 1.5

#: A server slow enough to still be starting when a caller gives up on it, and
#: how long that caller waits.
NEVER_READY_S = 2.0
GIVES_UP_AFTER_S = 1.0


def _write_server(script_path: Path, name: str, delay: float) -> None:
    """A one-file stdio MCP server, optionally slow to become ready.

    It records its pid beside itself before anything else, so a test can tell
    whether the process outlived the call that spawned it.
    """
    script_path.write_text(
        f"""
import os
import sys
import time
from pathlib import Path

Path(sys.argv[0]).with_suffix(".pid").write_text(str(os.getpid()))

from mcp.server.fastmcp import FastMCP

time.sleep({delay})

mcp = FastMCP("{name}")


@mcp.tool()
def ping_{name}() -> str:
    return "{name}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
""".strip()
    )


def _servers(tmp_path: Path, count: int, delays: Sequence[float] | None = None) -> dict[str, dict[str, object]]:
    servers: dict[str, dict[str, object]] = {}
    for index in range(count):
        name = f"server{index}"
        script = tmp_path / f"{name}.py"
        _write_server(script, name, delays[index] if delays is not None else 0.0)
        servers[name] = {"command": sys.executable, "args": [str(script)]}
    return servers


def _quitter(script_path: Path) -> dict[str, object]:
    """A server that starts, never answers, and exits while the others connect."""
    script_path.write_text(f"import time\n\ntime.sleep({FAILS_AFTER_S})\n")
    return {"command": sys.executable, "args": [str(script_path)]}


def _from_servers(servers: dict[str, dict[str, object]]) -> MCPToolProvider:
    return MCPToolProvider(config=MCPConfig.model_validate({"mcpServers": servers}))


def _provider(
    tmp_path: Path,
    count: int,
    delays: Sequence[float] | None = None,
    server_names: list[str] | None = None,
) -> MCPToolProvider:
    config = MCPConfig.model_validate({"mcpServers": _servers(tmp_path, count, delays)})
    return MCPToolProvider(config=config, server_names=server_names)


def _entered(provider: MCPToolProvider, api: str) -> AbstractAsyncContextManager[Any]:
    """Both ways in: the tool lifecycle an Agent uses, and connect()."""
    return provider.connect() if api == "connect" else provider


def _spawned_pids(tmp_path: Path) -> list[int]:
    """The pids of the servers that got far enough to record one."""
    return [int(pid_file.read_text()) for pid_file in sorted(tmp_path.glob("server*.pid"))]


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


async def test_every_server_is_connected_and_its_tools_kept_apart(tmp_path: Path) -> None:
    provider = _provider(tmp_path, SERVER_COUNT)

    async with provider as tools:
        assert sorted(provider.servers) == [f"server{i}" for i in range(SERVER_COUNT)]
        # Each server's tools stay under its own name rather than being merged.
        for index in range(SERVER_COUNT):
            assert [t["name"] for t in provider.get_tools(f"server{index}")] == [f"ping_server{index}"]
        assert len(tools) == SERVER_COUNT


async def test_tools_are_ordered_by_the_config_not_by_who_answered_first(tmp_path: Path) -> None:
    """The tools go into every request the model is sent, so their order has to
    be a property of the config and not of the race between servers: an order
    that moves between runs is a prompt cache prefix that never hits again, and
    a run nobody can reproduce.

    These servers become ready in the reverse of the order they are configured
    in, so a list filled in as the handshakes land comes out backwards.
    """
    count = 4
    spacing_s = 0.25
    delays = [spacing_s * (count - 1 - index) for index in range(count)]
    provider = _provider(tmp_path, count, delays=delays)

    async with provider as tools:
        assert [tool.name for tool in tools] == [f"server{i}__ping_server{i}" for i in range(count)]
        assert provider.servers == [f"server{i}" for i in range(count)]
        assert list(provider.all_tools) == [f"server{i}" for i in range(count)]


async def test_connecting_does_not_serialise_server_startup(tmp_path: Path) -> None:
    """Servers start up concurrently, so the cost is the slowest one rather
    than the sum.

    Measured against one server on this machine rather than against a number of
    seconds, since most of a server's startup is its interpreter and imports and
    that varies. Waiting each startup out in turn puts N servers at N times one.
    The bound sits well short of that because concurrently is not 1 either: the
    interpreter and imports are CPU, and on a two core runner that part goes
    back to being taken in turns however the connecting is arranged.
    """

    async def connect(count: int) -> float:
        started = time.perf_counter()
        async with _provider(tmp_path / str(count), count, delays=[STARTUP_DELAY_S] * count):
            return time.perf_counter() - started

    (tmp_path / "1").mkdir()
    (tmp_path / str(SERVER_COUNT)).mkdir()
    one = await connect(1)
    many = await connect(SERVER_COUNT)

    in_turn = one * SERVER_COUNT
    assert many < in_turn * 0.6, (
        f"{SERVER_COUNT} servers took {many:.2f}s against {one:.2f}s for one, "
        f"near the {in_turn:.2f}s of connecting to them one at a time; startup looks serialised"
    )


async def test_a_server_that_fails_leaves_no_half_connected_state(tmp_path: Path) -> None:
    """A provider that kept the tools of the servers that did connect would
    offer the model tools it cannot call: the sessions behind them are gone with
    the rest, so calling one only reaches 'Server not connected'.
    """
    servers = _servers(tmp_path, 1)
    servers["quits"] = _quitter(tmp_path / "quits.py")
    provider = _from_servers(servers)

    with pytest.raises(RuntimeError, match="quits"):
        async with provider:
            pass

    assert provider.servers == []
    assert provider.all_tools == {}
    assert provider.get_all_tools() == []


async def test_a_server_that_fails_does_not_leave_the_others_running(tmp_path: Path) -> None:
    """Every transport is entered before any handshake runs, so by the time one
    server fails, all of the others have been spawned. Whatever came up has to
    be taken back down on the way out, or it keeps running with nobody left
    holding a handle to it.
    """
    servers = _servers(tmp_path, 3)
    servers["quits"] = _quitter(tmp_path / "quits.py")
    provider = _from_servers(servers)

    with pytest.raises(RuntimeError, match="quits"):
        async with provider:
            pass

    pids = _spawned_pids(tmp_path)
    survivors = [pid for pid in pids if _running(pid)]
    for pid in survivors:
        os.kill(pid, signal.SIGKILL)

    assert len(pids) == 3, "the healthy servers never started, so this proves nothing"
    assert survivors == []


async def test_giving_up_partway_through_does_not_leave_servers_running(tmp_path: Path) -> None:
    """A caller that stops waiting is served the same way a failure is: whatever
    was spawned comes down before the cancellation is let out.
    """
    provider = _provider(tmp_path, 3, delays=[NEVER_READY_S] * 3)

    with pytest.raises(TimeoutError), anyio.fail_after(GIVES_UP_AFTER_S):
        async with provider:
            pass

    pids = _spawned_pids(tmp_path)
    survivors = [pid for pid in pids if _running(pid)]
    for pid in survivors:
        os.kill(pid, signal.SIGKILL)

    assert len(pids) == 3, "the servers never started, so this proves nothing"
    assert survivors == []


@pytest.mark.parametrize("api", ["lifecycle", "connect"])
async def test_a_server_that_will_not_start_is_named(tmp_path: Path, api: str) -> None:
    """With several connecting together, "it failed" is not enough to act on.

    The name has to survive the way out. Taking the connections back down runs
    each transport's own teardown, and mcp's stdio transport meets a half-open
    connection with errors of its own, one task group deep per server that was
    open - enough to bury the failure that started it all and hand back a nest
    of ExceptionGroups instead. Naming the server is only worth anything if the
    name is still on the exception the caller catches.
    """
    servers = _servers(tmp_path, 2)
    servers["broken"] = {"command": sys.executable, "args": [str(tmp_path / "does-not-exist.py")]}
    provider = _from_servers(servers)

    with pytest.raises(RuntimeError, match="broken") as failure:
        async with _entered(provider, api):
            pass

    # And what actually went wrong is still underneath it.
    assert failure.value.__cause__ is not None


async def test_an_unknown_server_name_still_raises_keyerror(tmp_path: Path) -> None:
    provider = _provider(tmp_path, 1, server_names=["server0", "not-in-config"])

    with pytest.raises(KeyError, match="not-in-config"):
        async with provider:
            pass
