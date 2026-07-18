"""Unit tests for GitConnector auth handling (offline)."""

import git
from pydantic import SecretStr

from app.infrastructure.connectors.git import connector as gc
from app.infrastructure.connectors.git.connector import (
    GitConnector,
    _looks_like_auth_failure,
)


def test_authenticated_url_unchanged_without_token(monkeypatch):
    monkeypatch.setattr(gc.settings, "GIT_TOKEN", None)
    url = "https://github.com/org/repo.git"
    assert GitConnector()._authenticated_url(url) == url


def test_authenticated_url_embeds_token(monkeypatch):
    monkeypatch.setattr(gc.settings, "GIT_TOKEN", SecretStr("tok123"))
    monkeypatch.setattr(gc.settings, "GIT_USERNAME", None)
    out = GitConnector()._authenticated_url("https://github.com/org/repo.git")
    assert out == "https://tok123@github.com/org/repo.git"


def test_authenticated_url_embeds_user_and_token(monkeypatch):
    monkeypatch.setattr(gc.settings, "GIT_TOKEN", SecretStr("tok123"))
    monkeypatch.setattr(gc.settings, "GIT_USERNAME", "raghu")
    out = GitConnector()._authenticated_url("https://github.com/org/repo.git")
    assert out == "https://raghu:tok123@github.com/org/repo.git"


def test_authenticated_url_ignores_ssh(monkeypatch):
    monkeypatch.setattr(gc.settings, "GIT_TOKEN", SecretStr("tok123"))
    url = "git@github.com:org/repo.git"
    assert GitConnector()._authenticated_url(url) == url


def test_looks_like_auth_failure_detects_missing_username():
    exc = git.GitCommandError(
        ["git", "clone"],
        128,
        b"fatal: could not read Username for 'https://github.com': "
        b"No such device or address",
    )
    assert _looks_like_auth_failure(exc) is True


def test_looks_like_auth_failure_false_for_other_errors():
    exc = git.GitCommandError(["git", "clone"], 128, b"fatal: some disk error")
    assert _looks_like_auth_failure(exc) is False
