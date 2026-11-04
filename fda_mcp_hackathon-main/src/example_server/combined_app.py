"""Serve Flask UI and MCP tools in one Databricks App."""

import os

import uvicorn
from starlette.middleware.wsgi import WSGIMiddleware

from example_server.app import mcp
from flask_ui.app import app as flask_app

# Build the FastMCP ASGI application.
# MCP and custom health routes are registered already.
app = mcp.http_app(path="/mcp", transport="http")

# Mount Flask after MCP routes so /mcp and /health
# remain handled by FastMCP.
app.mount("/", WSGIMiddleware(flask_app))


def main():
    port = int(os.getenv("DATABRICKS_APP_PORT") or os.getenv("PORT") or "8000")

    uvicorn.run(
        "example_server.combined_app:app",
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":
    main()
