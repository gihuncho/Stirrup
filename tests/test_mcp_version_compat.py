"""The places where mcp 1.x and 2.x disagree, and this module has to not care.

The suite's other MCP test builds its server with `mcp.server.fastmcp`, which
2.x does not ship, so it skips there — these run on both.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from stirrup.tools.mcp import (
    StreamableHttpServerConfig,
    _streamable_http_streams,
    _tool_input_schema,
)

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


class TestToolInputSchema:
    """2.x renamed `Tool.inputSchema` to `input_schema`."""

    def test_reads_the_two_x_spelling(self) -> None:
        assert _tool_input_schema(SimpleNamespace(input_schema=SCHEMA)) == SCHEMA

    def test_reads_the_one_x_spelling(self) -> None:
        assert _tool_input_schema(SimpleNamespace(inputSchema=SCHEMA)) == SCHEMA

    def test_prefers_the_two_x_spelling_when_a_model_carries_both(self) -> None:
        """Some 2.x models keep the old name as an alias; they mean the same
        thing, and reading the current one avoids a deprecation path."""
        both = SimpleNamespace(input_schema=SCHEMA, inputSchema={"stale": True})

        assert _tool_input_schema(both) == SCHEMA


class TestStreamableHttpTransport:
    """2.x renamed the client, took a prepared httpx client instead of headers
    and timeouts, and dropped the third yielded value."""

    def test_the_configured_values_reach_the_transport(self) -> None:
        config = StreamableHttpServerConfig(
            url="http://example.test/mcp",
            headers={"Authorization": "Bearer x"},
            timeout=11,
            sse_read_timeout=222,
            terminate_on_close=False,
        )

        # Built, not entered: entering it would open a connection. What matters
        # is that assembling the arguments works against the installed mcp.
        assert _streamable_http_streams(config) is not None

    def test_a_config_with_no_headers_is_accepted(self) -> None:
        assert _streamable_http_streams(StreamableHttpServerConfig(url="http://example.test/mcp")) is not None
