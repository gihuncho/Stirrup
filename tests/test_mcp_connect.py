"""Connecting an MCPToolProvider to several servers at once."""

import sys
import time
from pathlib import Path

import pytest

from stirrup.tools.mcp import MCPConfig, MCPToolProvider

pytest.importorskip("mcp.server.fastmcp")

#: Each server sleeps this long before serving, on top of the interpreter and
#: FastMCP import it already pays, so that startup is the dominant cost.
STARTUP_DELAY_S = 0.4
SERVER_COUNT = 5


def _write_server(script_path: Path, name: str) -> None:
    """A one-file stdio MCP server that is slow to become ready."""
    script_path.write_text(
        f"""
import time

from mcp.server.fastmcp import FastMCP

time.sleep({STARTUP_DELAY_S})

mcp = FastMCP("{name}")


@mcp.tool()
def ping_{name}() -> str:
    return "{name}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
""".strip()
    )


def _servers(tmp_path: Path, count: int) -> dict[str, dict[str, object]]:
    servers: dict[str, dict[str, object]] = {}
    for index in range(count):
        name = f"server{index}"
        script = tmp_path / f"{name}.py"
        _write_server(script, name)
        servers[name] = {"command": sys.executable, "args": [str(script)]}
    return servers


def _provider(tmp_path: Path, count: int, **kwargs: object) -> MCPToolProvider:
    config = MCPConfig.model_validate({"mcpServers": _servers(tmp_path, count)})
    return MCPToolProvider(config=config, **kwargs)  # type: ignore[arg-type]


async def test_every_server_is_connected_and_its_tools_kept_apart(tmp_path: Path) -> None:
    provider = _provider(tmp_path, SERVER_COUNT)

    async with provider as tools:
        assert sorted(provider.servers) == [f"server{i}" for i in range(SERVER_COUNT)]
        # Each server's tools stay under its own name rather than being merged.
        for index in range(SERVER_COUNT):
            assert [t["name"] for t in provider.get_tools(f"server{index}")] == [f"ping_server{index}"]
        assert len(tools) == SERVER_COUNT


async def test_connecting_does_not_serialise_server_startup(tmp_path: Path) -> None:
    """Servers start up concurrently, so the cost is the slowest one rather
    than the sum.

    Measured against one server on this machine rather than against a fixed
    number, since most of a server's startup is its interpreter and imports and
    that varies. Waiting each startup out in turn would put five servers at
    roughly five times one; concurrently it stays close to one.
    """

    async def connect(count: int) -> float:
        started = time.perf_counter()
        async with _provider(tmp_path / str(count), count):
            return time.perf_counter() - started

    (tmp_path / "1").mkdir()
    (tmp_path / str(SERVER_COUNT)).mkdir()
    one = await connect(1)
    many = await connect(SERVER_COUNT)

    assert many < one * 2, (
        f"{SERVER_COUNT} servers took {many:.2f}s against {one:.2f}s for one; startup looks serialised"
    )


async def test_a_server_that_will_not_start_is_named(tmp_path: Path) -> None:
    """With several connecting together, "it failed" is not enough to act on."""
    servers = _servers(tmp_path, 2)
    servers["broken"] = {"command": sys.executable, "args": [str(tmp_path / "does-not-exist.py")]}
    provider = MCPToolProvider(config=MCPConfig.model_validate({"mcpServers": servers}))

    with pytest.raises(RuntimeError, match="broken"):
        async with provider:
            pass


async def test_an_unknown_server_name_still_raises_keyerror(tmp_path: Path) -> None:
    provider = _provider(tmp_path, 1, server_names=["server0", "not-in-config"])

    with pytest.raises(KeyError, match="not-in-config"):
        async with provider:
            pass
