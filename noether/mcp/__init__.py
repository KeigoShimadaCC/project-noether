"""MCP adapter: Noether as a tool server for MCP-capable agents.

A thin frontend over the same session surface that drives HTTP and CLI
(docs/02_TECH_SPEC.md section 2). Requires the optional [mcp] extra.
"""

from noether.mcp.server import NoetherTools, create_mcp_server

__all__ = ["NoetherTools", "create_mcp_server"]
