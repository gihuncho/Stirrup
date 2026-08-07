"""MCP (Model Context Protocol) tool provider for connecting to MCP servers.

This module provides MCPToolProvider, a ToolProvider that manages connections to
multiple MCP servers and exposes each MCP tool as a separate Tool object.

Example usage:
    ```python
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools import default_tools

    # With Agent (preferred)
    client = ChatCompletionsClient(model="gpt-5.6-luna", max_tokens=8_192, context_window_tokens=1_000_000)
    agent = Agent(
        client=client,
        name="assistant",
        tools=[*default_tools(), MCPToolProvider.from_config("mcp.json")],
    )
    async with agent.session() as session:
        await session.run("Use MCP tools")

    # Standalone usage
    provider = MCPToolProvider.from_config(Path("mcp.json"))
    async with provider as tools:
        # tools is a list of Tool objects
        pass
    ```

Requires the optional `mcp` dependency:
    pip install stirrup[mcp]
"""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import anyio
from json_schema_to_pydantic import create_model
from pydantic import BaseModel, Field, model_validator

from stirrup.core.models import (
    AudioContentBlock,
    Content,
    ContentBlock,
    ImageContentBlock,
    Tool,
    ToolProvider,
    ToolResult,
    ToolUseCountMetadata,
)

# MCP imports (optional dependency)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import (
        AudioContent as MCPAudioContent,
    )
    from mcp.types import (
        ImageContent as MCPImageContent,
    )
    from mcp.types import (
        TextContent as MCPTextContent,
    )
except ImportError as e:
    raise ImportError(
        "Requires installation of the mcp extra. Install with (for example): `uv pip install stirrup[mcp]` or `uv add stirrup[mcp]`",
    ) from e

# WebSocket client requires additional 'websockets' package
try:
    from mcp.client.websocket import websocket_client
except ImportError:
    websocket_client = None  # ty: ignore[invalid-assignment]


logger = logging.getLogger(__name__)

__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "MCPToolProvider",
    "SseServerConfig",
    "StdioServerConfig",
    "StreamableHttpServerConfig",
    "WebSocketServerConfig",
]


# === Models ===


class StdioServerConfig(BaseModel):
    """Configuration for stdio-based MCP servers (local process)."""

    command: str
    """Command to run the MCP server (e.g., "npx", "python")."""

    args: list[str] = Field(default_factory=list)
    """Arguments to pass to the command."""

    env: dict[str, str] | None = None
    """Environment variables to set for the server process."""

    cwd: str | None = None
    """Working directory for the server process."""

    encoding: str = "utf-8"
    """Text encoding for messages."""


class SseServerConfig(BaseModel):
    """Configuration for SSE-based MCP servers (HTTP GET with Server-Sent Events)."""

    url: str
    """The SSE endpoint URL (must end with /sse)."""

    headers: dict[str, str] | None = None
    """Optional HTTP headers."""

    timeout: float = 5.0
    """HTTP timeout for regular operations (seconds)."""

    sse_read_timeout: float = 300.0
    """Timeout for SSE read operations (seconds)."""


class StreamableHttpServerConfig(BaseModel):
    """Configuration for Streamable HTTP MCP servers (HTTP POST with optional SSE responses)."""

    url: str
    """The endpoint URL."""

    headers: dict[str, str] | None = None
    """Optional HTTP headers."""

    timeout: float = 30.0
    """HTTP timeout (seconds)."""

    sse_read_timeout: float = 300.0
    """SSE read timeout (seconds)."""

    terminate_on_close: bool = True
    """Close session when transport closes."""


class WebSocketServerConfig(BaseModel):
    """Configuration for WebSocket-based MCP servers."""

    url: str
    """The WebSocket URL (must start with ws:// or wss://)."""


# Type alias for the union of all server config types
MCPServerConfig = StdioServerConfig | SseServerConfig | StreamableHttpServerConfig | WebSocketServerConfig


def _infer_server_config(data: dict[str, Any]) -> MCPServerConfig:
    """Infer and instantiate the correct config class from raw data.

    Inference rules:
    - 'command' field present -> StdioServerConfig
    - 'url' starts with ws:// or wss:// -> WebSocketServerConfig
    - 'url' ends with /sse -> SseServerConfig
    - 'url' present (default) -> StreamableHttpServerConfig

    Args:
        data: Raw configuration dictionary.

    Returns:
        Appropriate server config instance.

    Raises:
        ValueError: If neither 'command' nor 'url' is provided.
    """
    if "command" in data:
        return StdioServerConfig(**data)
    if "url" in data:
        url = data["url"]
        if url.startswith(("ws://", "wss://")):
            return WebSocketServerConfig(**data)
        if url.endswith("/sse"):
            return SseServerConfig(**data)
        return StreamableHttpServerConfig(**data)
    raise ValueError("Config must have 'command' (stdio) or 'url' (SSE/HTTP/WebSocket)")


class MCPConfig(BaseModel):
    """Root configuration matching mcp.json format."""

    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")
    """Map of server names to their configurations."""

    @model_validator(mode="before")
    @classmethod
    def _infer_transport_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert raw server configs to appropriate typed instances."""
        if "mcpServers" in data:
            data["mcpServers"] = {
                name: _infer_server_config(config) if isinstance(config, dict) else config
                for name, config in data["mcpServers"].items()
            }
        return data


# === Manager ===


class MCPToolProvider(ToolProvider):
    """MCP tool provider that manages connections to multiple MCP servers.

    MCPToolProvider connects to MCP servers and exposes each server's tools
    as individual Tool objects.

    Usage with Agent (preferred):
        from stirrup.clients.chat_completions_client import ChatCompletionsClient
        from stirrup.tools import default_tools

        client = ChatCompletionsClient(model="gpt-5.6-luna", max_tokens=8_192, context_window_tokens=1_000_000)
        agent = Agent(
            client=client,
            name="assistant",
            tools=[*default_tools(), MCPToolProvider.from_config("mcp.json")],
        )

        async with agent.session(output_dir="./output") as session:
            await session.run("Use MCP tools")

    Standalone usage with connect() context manager:
        provider = MCPToolProvider.from_config(Path("mcp.json"))
        async with provider.connect() as provider:
            tools = provider.get_all_tools()
            # Use tools...
    """

    def __init__(
        self,
        config: MCPConfig,
        server_names: list[str] | None = None,
    ) -> None:
        """Initialize the MCP manager.

        Args:
            config: MCPConfig instance.
            server_names: Which servers to connect to. If None, connects to all servers in config.
        """
        self._config = config
        self._server_names = server_names
        self._servers: dict[str, ClientSession] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}
        self._exit_stack: AsyncExitStack | None = None

    @classmethod
    def from_config(cls, config_path: Path | str, server_names: list[str] | None = None) -> Self:
        """Create an MCPToolProvider from a config file.

        Args:
            config_path: Path to the MCP config file.
            server_names: Which servers to connect to. If None, connects to all servers in config.

        Returns:
            MCPToolProvider instance.
        """
        config = MCPConfig.model_validate_json(Path(config_path).read_text())

        return cls(config=config, server_names=server_names)

    async def _open_sessions(self, stack: AsyncExitStack) -> None:
        """Open a session per configured server and cache the tools each offers.

        Every transport is entered first, and only then are the handshakes run
        together. The order matters: entering a stdio transport just spawns the
        process and returns, so the waiting happens in `initialize`, while that
        process starts up. Interleaving the two means each server's startup is
        paid end to end — connecting to N servers costs N startups rather than
        the slowest one.

        Stdio and Streamable HTTP gain from this. SSE and WebSocket do not:
        their clients finish connecting before they yield their streams
        (`mcp/client/sse.py`, `mcp/client/websocket.py`), so only the round
        trips overlap.

        Only the handshakes run concurrently. Entering and unwinding the exit
        stack stays on this task, which is what anyio's cancel scopes require.

        Either every server connects or none does: nothing reaches the provider
        until all of the handshakes are through, and anything already opened is
        closed here if one of them is not.
        """
        config = self._config
        servers_to_connect = self._server_names or list(config.mcp_servers.keys())

        sessions: dict[str, ClientSession] = {}
        tools: dict[str, list[dict[str, Any]]] = {}
        try:
            for name in servers_to_connect:
                if name not in config.mcp_servers:
                    raise KeyError(f"Server '{name}' not found in config. Available: {list(config.mcp_servers.keys())}")

                read, write = await self._enter_transport(stack, name, config.mcp_servers[name])
                sessions[name] = await stack.enter_async_context(ClientSession(read, write))

            # A task group rather than asyncio.gather: mcp's transports are anyio
            # based and so is this package, and a failure here should cancel the
            # handshakes still in flight rather than leave them running while the
            # exit stack unwinds.
            try:
                async with anyio.create_task_group() as task_group:
                    for name, session in sessions.items():
                        task_group.start_soon(self._handshake, name, session, tools)
            except BaseExceptionGroup as group:
                # One server failing is the ordinary case and reads better as
                # itself than as a group of one, the way it did when the
                # handshakes ran in sequence. Re-raised with the cause it
                # already carries, so the error underneath and its traceback
                # survive the unwrapping, and the group stays reachable as the
                # new exception's context when more than one server failed —
                # keeping the group instead would make the type a caller sees
                # depend on whether two handshakes failed in the same
                # scheduling window.
                failure = group.exceptions[0]
                raise failure from failure.__cause__
        except BaseException:
            # Every transport was entered before any handshake ran, so by the
            # time one fails the rest are already spawned and would outlive this
            # call. Closing a half-open connection throws errors of its own (a
            # transport complaining about the process that has already gone),
            # and those must not stand in for the failure the caller needs to
            # see. The stack keeps closing past them either way, and past a
            # cancellation, so a caller that gives up partway through leaves no
            # servers behind either.
            try:
                await stack.aclose()
            except Exception as cleanup_error:
                # Logged rather than swallowed: a transport complaining about a
                # process that has already gone is noise, but a session that
                # would not terminate is a leak on the far end, and the caller
                # is about to be handed an unrelated error.
                logger.warning("Error while closing MCP connections: %s", cleanup_error)
            raise

        # Recorded in configured order rather than in the order the handshakes
        # happened to finish: `get_all_tools` walks this, so completion order
        # would reshuffle the tools on every request the model is sent.
        for name, session in sessions.items():
            self._servers[name] = session
            self._tools[name] = tools[name]

    async def _handshake(self, name: str, session: ClientSession, tools: dict[str, list[dict[str, Any]]]) -> None:
        """Initialize one session and record the tools it reports in `tools`.

        The server is named in the failure: with several connecting at once,
        the underlying error alone does not say which one it came from.

        Results are collected for the caller instead of being written to the
        provider, which keeps both the ordering and the failure handling in one
        place rather than in whichever task got there first.
        """
        try:
            await session.initialize()
            response = await session.list_tools()
        except Exception as exc:
            raise RuntimeError(f"MCP server '{name}' failed to connect: {exc}") from exc

        tools[name] = [{"name": t.name, "description": t.description, "schema": t.inputSchema} for t in response.tools]

    async def _enter_transport(
        self, stack: AsyncExitStack, name: str, server_config: MCPServerConfig
    ) -> tuple[Any, Any]:
        """Open the transport a server is configured for, and return its streams."""
        match server_config:
            case StdioServerConfig():
                server_params = StdioServerParameters(
                    command=server_config.command,
                    args=server_config.args,
                    env=server_config.env,
                    cwd=server_config.cwd,
                    encoding=server_config.encoding,
                )
                read, write = await stack.enter_async_context(stdio_client(server_params))
            case SseServerConfig():
                read, write = await stack.enter_async_context(
                    sse_client(
                        url=server_config.url,
                        headers=server_config.headers,
                        timeout=server_config.timeout,
                        sse_read_timeout=server_config.sse_read_timeout,
                    )
                )
            case StreamableHttpServerConfig():
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(
                        url=server_config.url,
                        headers=server_config.headers,
                        timeout=server_config.timeout,
                        sse_read_timeout=server_config.sse_read_timeout,
                        terminate_on_close=server_config.terminate_on_close,
                    )
                )
            case WebSocketServerConfig():
                if websocket_client is None:
                    raise ImportError(
                        f"WebSocket transport for server '{name}' requires the 'websockets' package. "
                        "Install with: pip install websockets"
                    )
                read, write = await stack.enter_async_context(websocket_client(url=server_config.url))
            case _:
                raise TypeError(f"Server '{name}' has an unsupported transport: {type(server_config).__name__}")
        return read, write

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Self]:
        """Connect to MCP servers from config file.

        Nothing is left running behind a failure: if any server does not come
        up, the ones that did are closed before the error is raised.

        Yields:
            Self with active connections to specified servers.

        Raises:
            KeyError: If a specified server name doesn't exist in config.
            FileNotFoundError: If a stdio server's command doesn't exist.
            ImportError: If a server needs the optional 'websockets' package and
                it isn't installed.
            TypeError: If a server is configured for an unsupported transport.
            RuntimeError: If a server doesn't complete the MCP handshake. The
                message names the server; the error it came from is the cause.
        """
        async with AsyncExitStack() as stack:
            await self._open_sessions(stack)

            try:
                yield self
            finally:
                self._servers.clear()
                self._tools.clear()

    @property
    def servers(self) -> list[str]:
        """List of connected server names."""
        return list(self._servers.keys())

    def get_tools(self, server: str) -> list[dict[str, Any]]:
        """Get available tools for a specific server.

        Args:
            server: Server name.

        Returns:
            List of tool info dicts with name, description, and schema.
        """
        return self._tools.get(server, [])

    @property
    def all_tools(self) -> dict[str, list[str]]:
        """Get all available tools grouped by server.

        Returns:
            Dict mapping server names to lists of tool names.
        """
        return {server: [t["name"] for t in tools] for server, tools in self._tools.items()}

    def _convert_mcp_content(self, content_blocks: list[Any]) -> Content:
        """Convert MCP content blocks into Stirrup content blocks."""
        content: list[ContentBlock] = []

        for block in content_blocks:
            if isinstance(block, MCPTextContent):
                content.append(block.text)
                continue
            if isinstance(block, MCPImageContent):
                content.append(ImageContentBlock(data=block.data))
                continue
            if isinstance(block, MCPAudioContent):
                content.append(AudioContentBlock(data=block.data))
                continue
            raise TypeError(f"Unsupported MCP content block: {type(block).__name__}")

        if not content:
            return ""
        if len(content) == 1 and isinstance(content[0], str):
            return content[0]
        return content

    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any]) -> Content:
        """Call a tool on a specific MCP server.

        Args:
            server: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool result converted into Stirrup content blocks.

        Raises:
            ValueError: If server is not connected.
        """
        session = self._servers.get(server)
        if session is None:
            raise ValueError(f"Server '{server}' not connected. Available: {self.servers}")

        result = await session.call_tool(tool_name, arguments)
        return self._convert_mcp_content(result.content)

    def get_all_tools(self) -> list[Tool[Any, ToolUseCountMetadata]]:
        """Get individual Tool objects for each tool from all connected MCP servers.

        Each MCP tool is exposed as a separate Tool with its own parameter schema,
        allowing the LLM to see and call each tool directly without routing through
        a unified proxy.

        Tool names are formatted as '{server}__{tool_name}' to ensure uniqueness
        across servers (e.g., 'supabase__query_table').

        Returns:
            List of Tool objects, one for each tool available across all connected servers.
        """
        tools: list[Tool[Any, ToolUseCountMetadata]] = []

        for server_name, server_tools in self._tools.items():
            for tool_info in server_tools:
                mcp_tool_name = tool_info["name"]
                # Create unique tool name with server prefix
                unique_name = f"{server_name}__{mcp_tool_name}"

                # Convert JSON schema to Pydantic model
                params_model = create_model(
                    tool_info.get("schema", {}),
                )

                # Create executor closure - capture server_name and mcp_tool_name
                # using default arguments to avoid late binding issues in the loop
                async def executor(
                    params: BaseModel,
                    _server: str = server_name,
                    _tool: str = mcp_tool_name,
                ) -> ToolResult[ToolUseCountMetadata]:
                    content = await self.call_tool(_server, _tool, params.model_dump())
                    return ToolResult(content=content, metadata=ToolUseCountMetadata())

                tools.append(
                    Tool(
                        name=unique_name,
                        description=tool_info.get("description") or f"Tool '{mcp_tool_name}' from {server_name}",
                        parameters=params_model,
                        executor=executor,
                    )
                )

        return tools

    # Tool lifecycle protocol implementation
    async def __aenter__(self) -> list[Tool[Any, ToolUseCountMetadata]]:
        """Enter async context: connect to MCP servers and return all tools.

        Fails the way `connect` does, leaving no servers running behind it.
        That includes a tool whose schema will not build: it is turned into a
        Tool here, inside the guard, rather than after it.

        Returns:
            List of Tool objects, one for each tool available across all connected servers.
        """
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        try:
            await self._open_sessions(self._exit_stack)
            return self.get_all_tools()
        except BaseException:
            try:
                await self._exit_stack.aclose()
            except Exception as cleanup_error:
                logger.warning("Error while closing MCP connections: %s", cleanup_error)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context: disconnect from MCP servers."""
        self._servers.clear()
        self._tools.clear()
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
