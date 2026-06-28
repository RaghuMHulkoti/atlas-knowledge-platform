"""
Root conftest.py — sets required environment variables before any app
module is imported so that pydantic-settings' Settings() can be
instantiated without a real .env file.

These are **test-only dummy values** and are never used to make real
API calls during unit tests.
"""

import os

# Set required env vars before any app module is imported.
# pydantic-settings reads from os.environ, so patching here is sufficient.
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("CHROMA_API_KEY", "test-chroma-key")
os.environ.setdefault("CHROMA_TENANT", "test-tenant")
os.environ.setdefault("CHROMA_DATABASE", "test-database")
