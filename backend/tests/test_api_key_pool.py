"""Tests for multi-key failover.

The failure this protects against is a 429 arriving mid-demo. These tests
therefore assert the behaviour that matters: a rate-limited key is skipped, the
request still succeeds on another key, and no error reaches the caller while any
key remains usable.
"""
from __future__ import annotations

import httpx
import pytest

from backend.services.api_key_pool import (
    ApiKeyPool,
    ApiKeyPoolExhausted,
    post_with_failover,
)

URL = "https://example.invalid/v1beta/models/test:generateContent"


def _pool() -> ApiKeyPool:
    return ApiKeyPool([("key-alpha", "key#1"), ("key-beta", "key#2"), ("key-gamma", "key#3")])


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", payload=None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


def _patch_post(monkeypatch, handler) -> list:
    """Record which key each request used and return a scripted response.

    `post_with_failover` imports httpx inside the function, so patching the
    module attribute is what actually intercepts the call.
    """
    seen: list = []

    def fake_post(url, headers=None, json=None, timeout=None):
        key = (headers or {}).get("x-goog-api-key")
        seen.append(key)
        return handler(key)

    monkeypatch.setattr(httpx, "post", fake_post)
    return seen


def test_pool_loads_and_reports_health() -> None:
    pool = _pool()
    assert pool.size == 3
    assert pool.configured
    described = pool.describe()
    assert described["configured_keys"] == 3
    assert described["available_now"] == 3
    # Key material must never appear in the health payload.
    serialised = str(described)
    assert "key-alpha" not in serialised
    assert "key-beta" not in serialised


def test_rotation_spreads_requests_across_keys() -> None:
    pool = _pool()
    used = [pool.next_key().key for _ in range(6)]
    assert set(used) == {"key-alpha", "key-beta", "key-gamma"}
    assert used[:3] != [used[0]] * 3, "the pool must rotate, not pin one key"


def test_rate_limited_key_is_parked_then_skipped() -> None:
    pool = _pool()
    first = pool.next_key()
    pool.record_rate_limited(first, "429 quota exceeded")

    described = pool.describe()
    assert described["available_now"] == 2
    parked = next(item for item in described["keys"] if item["id"] == first.masked)
    assert parked["available"] is False
    assert parked["cooldown_seconds_remaining"] > 0
    assert parked["rate_limited"] == 1

    for _ in range(5):
        assert pool.next_key().key != first.key


def test_failover_retries_on_the_next_key_after_429(monkeypatch) -> None:
    pool = _pool()
    seen = _patch_post(
        monkeypatch,
        lambda key: (
            _FakeResponse(429, "quota exceeded")
            if key == "key-alpha"
            else _FakeResponse(200, payload={"ok": True})
        ),
    )

    body = post_with_failover(URL, {"contents": []}, 5.0, pool=pool)

    assert body == {"ok": True}
    assert "key-alpha" in seen, "the rate-limited key must have been attempted"
    assert len(seen) >= 2, "the request must have failed over to another key"


def test_transient_upstream_500_fails_over_instead_of_giving_up(monkeypatch) -> None:
    """The hosted endpoint intermittently returns 500; that must not end the run."""
    pool = _pool()
    seen = _patch_post(
        monkeypatch,
        lambda key: (
            _FakeResponse(500, '{"error":{"status":"INTERNAL"}}')
            if key == "key-alpha"
            else _FakeResponse(200, payload={"ok": True})
        ),
    )

    assert post_with_failover(URL, {"contents": []}, 5.0, pool=pool) == {"ok": True}
    assert len(seen) >= 2

    # A transient upstream error is not the key's fault, so the key stays usable.
    assert pool.describe()["available_now"] == 3


def test_a_denied_key_is_parked_but_the_pool_still_serves(monkeypatch) -> None:
    """One revoked key must not take the whole pool down."""
    pool = _pool()
    _patch_post(
        monkeypatch,
        lambda key: (
            _FakeResponse(403, "Your project has been denied access.")
            if key == "key-alpha"
            else _FakeResponse(200, payload={"ok": True})
        ),
    )

    assert post_with_failover(URL, {"contents": []}, 5.0, pool=pool) == {"ok": True}
    described = pool.describe()
    assert described["available_now"] == 2, "the denied key must be parked"


def test_exhausted_pool_reports_when_a_key_returns(monkeypatch) -> None:
    pool = _pool()
    _patch_post(monkeypatch, lambda key: _FakeResponse(429, "quota exceeded"))

    with pytest.raises(ApiKeyPoolExhausted) as excinfo:
        post_with_failover(URL, {"contents": []}, 5.0, pool=pool)

    message = str(excinfo.value)
    assert "rate limited" in message
    assert "available in" in message
    assert "key-alpha" not in message, "key material must not appear in errors"


def test_auth_failure_parks_a_key_for_much_longer_than_a_rate_limit() -> None:
    pool = _pool()
    rate_limited = pool.next_key()
    rejected = pool.next_key()
    pool.record_rate_limited(rate_limited)
    pool.record_auth_failure(rejected)

    described = {item["id"]: item for item in pool.describe()["keys"]}
    assert (
        described[rejected.masked]["cooldown_seconds_remaining"]
        > described[rate_limited.masked]["cooldown_seconds_remaining"]
    )


def test_success_clears_a_previous_cooldown() -> None:
    pool = _pool()
    state = pool.next_key()
    pool.record_rate_limited(state)
    assert pool.describe()["available_now"] == 2
    pool.record_success(state)
    assert pool.describe()["available_now"] == 3


def test_slow_upstream_cannot_multiply_the_timeout_across_retries(monkeypatch) -> None:
    """A slow upstream must not turn one timeout into seven.

    Measured before this guard existed: a 20-second extraction timeout became a
    141-second freeze, because every key and every transient retry paid the full
    read timeout and nothing capped the total. The interface showed a spinner the
    whole time while a declared fallback was sitting there able to answer in
    milliseconds.
    """
    elapsed = {"now": 0.0}

    def fake_monotonic() -> float:
        return elapsed["now"]

    monkeypatch.setattr(
        "backend.services.api_key_pool.time.monotonic", fake_monotonic
    )

    def timeout_after(timeout):
        # Simulate an attempt that burns its whole allowance, then fails.
        elapsed["now"] += timeout
        raise httpx.ReadTimeout("The read operation timed out")

    seen: list = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append(timeout)
        return timeout_after(timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(ApiKeyPoolExhausted) as excinfo:
        post_with_failover(
            URL,
            {"contents": []},
            timeout_seconds=20.0,
            pool=_pool(),
            total_deadline_seconds=50.0,
        )

    assert elapsed["now"] <= 50.0, (
        f"total wall clock {elapsed['now']}s exceeded the 50s deadline"
    )
    assert len(seen) < 7, f"attempted {len(seen)} times; the old code attempted 7"
    assert "unavailable" in str(excinfo.value)


def test_transport_failures_consume_the_transient_budget(monkeypatch) -> None:
    """A read timeout is a slow upstream, not a wrong key.

    It used to `continue` without charging any budget, so the loop ran the full
    key count plus every transient retry no matter how hopeless.
    """
    seen: list = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append((headers or {}).get("x-goog-api-key"))
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(ApiKeyPoolExhausted):
        post_with_failover(
            URL,
            {"contents": []},
            timeout_seconds=5.0,
            pool=_pool(),
            total_deadline_seconds=600.0,
        )

    # 3 keys + 4 transient retries was 7; charging transport failures stops it
    # one attempt after the transient budget is spent.
    assert len(seen) <= 5, f"attempted {len(seen)} times"


def test_a_healthy_key_still_answers_within_the_deadline(monkeypatch) -> None:
    """The deadline must not break the ordinary failover it sits on top of."""
    seen = _patch_post(
        monkeypatch,
        lambda key: (
            _FakeResponse(429, "quota exceeded")
            if key == "key-alpha"
            else _FakeResponse(200, payload={"ok": True})
        ),
    )

    body = post_with_failover(
        URL,
        {"contents": []},
        timeout_seconds=20.0,
        pool=_pool(),
    )

    assert body == {"ok": True}
    assert len(seen) == 2
