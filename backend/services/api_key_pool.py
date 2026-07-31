"""Failover and load balancing across multiple Gemma API keys.

A live demo has one failure mode that no amount of engineering elsewhere
protects against: the model call returns 429 halfway through, in front of
judges. A single key makes that a single point of failure, and it is the one
component whose failure is entirely outside our control.

This pool holds several keys and moves to the next one when the current key is
rate-limited or rejected. Two properties matter:

* **Failover is per-request and immediate.** A 429 does not surface to the
  caller while another key is still usable; the request is retried on the next
  key.
* **Cooldowns expire.** A key that returned 429 is parked for a cooldown rather
  than discarded, because rate limits are windows, not verdicts. After the
  window it rejoins the rotation.

Keys are read from the environment and never logged. `describe()` returns
health, never key material.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# How long a key sits out after a rate-limit response, in seconds. Google's
# per-minute quotas recover inside a minute; a shorter park would just burn the
# key again on the next request.
RATE_LIMIT_COOLDOWN_SECONDS = 65.0

# An authentication failure is not a transient window, so the key is parked for
# much longer rather than retried every request.
AUTH_FAILURE_COOLDOWN_SECONDS = 900.0

# Extra attempts granted for 5xx responses, which are upstream faults rather
# than key faults and so must not be paid for out of the per-key budget.
MAX_TRANSIENT_RETRIES = 4
TRANSIENT_RETRY_BACKOFF_SECONDS = 1.5

# A ceiling on the wall-clock cost of one logical call, across every retry.
#
# Without this, retries multiply the per-attempt timeout: three keys plus four
# transient retries is seven attempts, and when the upstream is slow rather than
# down, every one of them burns the full read timeout before failing. A 20s
# extraction timeout became a measured 141-second freeze with the interface
# showing a spinner the whole time. The declared fallback answers in
# milliseconds, so waiting minutes to maybe reach the hosted model is never the
# right trade during a demo.
#
# Expressed as a multiple of the caller's own timeout so a slow call
# (orchestration) gets proportionally more room than a fast one.
#
# Kept low deliberately: one attempt with a generous timeout beats three that get
# cut off. Measured on this endpoint, a 20s per-attempt timeout against an
# upstream that intermittently takes longer produced the worst of both worlds —
# three truncated attempts, 50 seconds spent, and a silent downgrade to the
# deterministic fallback, when a single 45s attempt would have returned a real
# hosted extraction. Retries are for a key that is refused, not for a model that
# is merely thinking.
TOTAL_DEADLINE_MULTIPLIER = float(os.getenv("GEMMA_RETRY_DEADLINE_MULTIPLIER", "1.4"))


@dataclass
class _KeyState:
    key: str
    label: str
    available_at: float = 0.0
    successes: int = 0
    rate_limited: int = 0
    failures: int = 0
    last_error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def masked(self) -> str:
        """A stable identifier for logs that is not the key."""
        if len(self.key) <= 8:
            return "****"
        return f"{self.key[:4]}…{self.key[-4:]}"


def _load_keys_from_env() -> List[tuple]:
    """Collect keys from GEMMA_API_KEY plus GEMMA_API_KEY_2..N.

    Comma-separated values in any of those variables are also accepted, so a
    single variable can carry the whole pool when that is easier to deploy.
    """
    raw: List[tuple] = []
    seen = set()

    def _add(value: str, label: str) -> None:
        for part in value.split(","):
            candidate = part.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                raw.append((candidate, f"{label}#{len(raw) + 1}"))

    primary = os.getenv("GEMMA_API_KEY", "")
    if primary:
        _add(primary, "key")

    index = 2
    while True:
        value = os.getenv(f"GEMMA_API_KEY_{index}", "")
        if not value:
            # Allow one gap so GEMMA_API_KEY_2 missing but _3 present still works.
            if not os.getenv(f"GEMMA_API_KEY_{index + 1}", ""):
                break
            index += 1
            continue
        _add(value, "key")
        index += 1

    return raw


class ApiKeyPool:
    """Round-robin pool of API keys with per-key cooldown."""

    def __init__(self, keys: Optional[List[tuple]] = None) -> None:
        self._lock = threading.Lock()
        self._cursor = 0
        self._explicit = keys is not None
        self._loaded = keys is not None
        self._states: List[_KeyState] = [
            _KeyState(key=key, label=label) for key, label in (keys or [])
        ]

    def _ensure_loaded(self) -> None:
        """Read keys on first use, not at import.

        The module is imported before `.env` is loaded, so reading the
        environment in `__init__` produced an empty pool no matter how many keys
        were configured.
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._states = [
                _KeyState(key=key, label=label) for key, label in _load_keys_from_env()
            ]
            self._loaded = True

    def reload(self) -> None:
        """Re-read keys from the environment, preserving health of known keys."""
        if self._explicit:
            return
        with self._lock:
            existing = {state.key: state for state in self._states}
            self._states = [
                existing.get(key) or _KeyState(key=key, label=label)
                for key, label in _load_keys_from_env()
            ]
            self._cursor = 0
            self._loaded = True

    @property
    def size(self) -> int:
        self._ensure_loaded()
        return len(self._states)

    @property
    def configured(self) -> bool:
        self._ensure_loaded()
        return bool(self._states)

    def _acquire(self, now: float) -> Optional[_KeyState]:
        """Next key that is not cooling down, advancing the rotation cursor.

        Rotating on every acquisition (rather than pinning to the first healthy
        key) spreads load across the pool, so no single key reaches its quota
        while the others sit idle.
        """
        with self._lock:
            count = len(self._states)
            for offset in range(count):
                state = self._states[(self._cursor + offset) % count]
                if state.available_at <= now:
                    self._cursor = (self._cursor + offset + 1) % count
                    return state
            return None

    def next_key(self) -> Optional[_KeyState]:
        self._ensure_loaded()
        return self._acquire(time.monotonic())

    def soonest_availability(self) -> float:
        """Seconds until the first key becomes usable again. 0 if one is ready."""
        now = time.monotonic()
        with self._lock:
            if not self._states:
                return 0.0
            return max(0.0, min(state.available_at for state in self._states) - now)

    # -- outcome reporting -------------------------------------------------

    def record_success(self, state: _KeyState) -> None:
        with state.lock:
            state.successes += 1
            state.available_at = 0.0
            state.last_error = None

    def record_rate_limited(self, state: _KeyState, detail: str = "") -> None:
        with state.lock:
            state.rate_limited += 1
            state.available_at = time.monotonic() + RATE_LIMIT_COOLDOWN_SECONDS
            state.last_error = detail[:200] or "rate limited"

    def record_auth_failure(self, state: _KeyState, detail: str = "") -> None:
        with state.lock:
            state.failures += 1
            state.available_at = time.monotonic() + AUTH_FAILURE_COOLDOWN_SECONDS
            state.last_error = detail[:200] or "authentication rejected"

    def record_failure(self, state: _KeyState, detail: str = "") -> None:
        """A failure that is not the key's fault — do not park it."""
        with state.lock:
            state.failures += 1
            state.last_error = detail[:200] or "request failed"

    def describe(self) -> Dict[str, object]:
        """Pool health for the status endpoint. Never returns key material."""
        self._ensure_loaded()
        now = time.monotonic()
        return {
            "configured_keys": len(self._states),
            "available_now": sum(
                1 for state in self._states if state.available_at <= now
            ),
            "keys": [
                {
                    "label": state.label,
                    "id": state.masked,
                    "available": state.available_at <= now,
                    "cooldown_seconds_remaining": round(
                        max(0.0, state.available_at - now), 1
                    ),
                    "successes": state.successes,
                    "rate_limited": state.rate_limited,
                    "failures": state.failures,
                    "last_error": state.last_error,
                }
                for state in self._states
            ],
        }


gemma_key_pool = ApiKeyPool()


class ApiKeyPoolExhausted(RuntimeError):
    """Every key in the pool is rate-limited, rejected, or absent."""


def post_with_failover(
    url: str,
    payload: Dict[str, object],
    timeout_seconds: float,
    *,
    pool: Optional[ApiKeyPool] = None,
    fallback_key: str = "",
    total_deadline_seconds: Optional[float] = None,
) -> Dict[str, object]:
    """POST to the Gemma endpoint, moving to the next key on a quota response.

    A 429 or 403-with-quota is treated as "this key, right now" rather than
    "this request cannot succeed", so it is retried on the next healthy key
    before any error reaches the caller. Errors that are not the key's fault
    (a malformed request, a server error) are raised immediately — retrying
    them on another key would just multiply the failure.
    """
    import httpx  # imported here so the pool module stays dependency-light

    active = pool or gemma_key_pool
    if not active.configured:
        if not fallback_key:
            raise ApiKeyPoolExhausted("No Gemma API key is configured")
        active = ApiKeyPool([(fallback_key, "env#1")])

    # A 5xx is not attributable to a key, so it must not consume the pool: the
    # endpoint intermittently returns 500 INTERNAL on function-calling requests
    # (observed roughly 1 in 3), and with only two usable keys a per-key budget
    # exhausted after two tries and killed the request outright. Allow extra
    # attempts specifically for transient upstream failures.
    attempts: List[str] = []
    budget = active.size + MAX_TRANSIENT_RETRIES
    transient_used = 0
    started = time.monotonic()
    deadline = started + (
        total_deadline_seconds
        if total_deadline_seconds is not None
        else timeout_seconds * TOTAL_DEADLINE_MULTIPLIER
    )
    for attempt_index in range(budget):
        # Retries are only worth attempting if there is time left to make one.
        # Starting an attempt that cannot finish before the deadline just adds
        # another full timeout to the caller's wait.
        remaining = deadline - time.monotonic()
        if attempt_index and remaining <= 0:
            attempts.append(f"deadline reached after {attempt_index} attempt(s)")
            break
        state = active.next_key()
        if state is None:
            break
        try:
            response = httpx.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": state.key,
                },
                json=payload,
                # Never let a single attempt outlive the overall deadline.
                timeout=max(1.0, min(timeout_seconds, remaining)) if attempt_index else timeout_seconds,
            )
        except httpx.HTTPError as exc:
            # Transport failures are not attributable to the key, but another
            # key is worth one try in case the endpoint is per-project.
            #
            # They do consume the transient budget, though. A read timeout means
            # the upstream is slow, not that this key is wrong, and retrying at
            # the same timeout usually times out again — the failure mode that
            # turned one 20s extraction into seven of them.
            active.record_failure(state, str(exc))
            attempts.append(f"{state.label}: transport error")
            transient_used += 1
            if transient_used > MAX_TRANSIENT_RETRIES:
                break
            continue

        if response.status_code == 429:
            active.record_rate_limited(state, response.text[:200])
            attempts.append(f"{state.label}: rate limited")
            continue
        if response.status_code >= 500:
            # Transient and not the key's fault, so the key is not parked and
            # this retry does not count against the per-key budget.
            active.record_failure(state, response.text[:200])
            attempts.append(f"{state.label}: upstream {response.status_code}")
            transient_used += 1
            if transient_used <= MAX_TRANSIENT_RETRIES:
                # Brief backoff: an immediate retry tends to hit the same
                # unhealthy upstream instance.
                time.sleep(TRANSIENT_RETRY_BACKOFF_SECONDS)
            continue
        if response.status_code in (401, 403):
            body = response.text.lower()
            if "quota" in body or "rate" in body:
                active.record_rate_limited(state, response.text[:200])
                attempts.append(f"{state.label}: quota exceeded")
            else:
                active.record_auth_failure(state, response.text[:200])
                attempts.append(f"{state.label}: key rejected")
            continue

        response.raise_for_status()
        active.record_success(state)
        return response.json()

    wait = active.soonest_availability()
    raise ApiKeyPoolExhausted(
        f"All {active.size} Gemma key(s) are unavailable "
        f"({'; '.join(attempts) or 'all cooling down'}). "
        f"Next key available in {wait:.0f}s."
    )
