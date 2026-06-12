"""HTTP session API (FastAPI): the same surface that drives CLI, web, and MCP.

Requires the optional [server] extra. Import create_app lazily so the core
package works without FastAPI installed.
"""

from noether.server.app import create_app

__all__ = ["create_app"]
