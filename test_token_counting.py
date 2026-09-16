"""
Tests for the corrected token counting logic in codex_usage.py.

Covers the three issues identified in the post-review:
  1. High-cache math — estimate_cost_details and estimate_energy must NOT cap
     cached_input_tokens to input_tokens (uncached).
  2. total_tokens on UsageEvent and Aggregate must include cached_input_tokens.
  3. cached_ratio on Aggregate must use total input (uncached + cached) as denominator.
  4. token_usage_record (TUR) handling in parse_session_file:
       a. Missing TUR timestamp — falls back to file mtime, session is not dropped.
       b. Mixed event types — TUR total > delta sum uses TUR; delta > TUR keeps deltas.
  5. token_delta — cached may exceed input; no capping.
"""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from codex_usage import (
    Aggregate,
    UsageEvent,
    estimate_cost,
    estimate_cost_details,
    estimate_energy,
    parse_session_file,
    token_delta,
)


def _make_event(input_tokens: int, cached_input_tokens: int, output_tokens: int) -> UsageEvent:
    return UsageEvent(
        session_id="test",
        session_title=None,
        day="2026-09-16",
        timestamp="2026-09-16T12:00:00+02:00",
        model="gpt-5.4",
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        plan_type=None,
        estimated_energy_wh=0.0,
        estimated_cost_usd=None,
    )


def _make_session_jsonl(lines: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# 1. High-cache math — cost and energy functions
# ---------------------------------------------------------------------------

class TestHighCacheMath(unittest.TestCase):
    """Verify cost/energy are computed correctly when cached >> uncached."""

    MODEL = "gpt-5.4"
    UNCACHED = 10
    CACHED = 5_000
    OUTPUT = 50

    def test_cost_details_does_not_clip_cached(self):
        result = estimate_cost_details(self.MODEL, self.UNCACHED, self.CACHED, self.OUTPUT)
        self.assertIsNotNone(result, "Expected pricing to be available for gpt-5.4")
        cost, is_guess = result  # type: ignore[misc]

        # cost must be > 0 and must reflect the full 5000 cached tokens
        # At gpt-5.4 cached rate (~$0.25/1M), 5000 cached tokens = $0.00000125
        # at uncached rate (~$2.50/1M), 10 uncached = $0.000000025
        # The clipped version would charge only ~10 cached tokens — much smaller.
        min_expected = (self.CACHED * 0.05e-6)  # very conservative lower bound
        self.assertGreater(cost, min_expected,
                           msg=f"Cost {cost} is too low — cached tokens likely being clipped")

    def test_energy_does_not_clip_cached(self):
        energy = estimate_energy(self.MODEL, self.UNCACHED, self.CACHED, self.OUTPUT)
        # With 5000 cached tokens the energy contribution from cache alone must
        # dwarf the uncached contribution (10 tokens). If cached were clipped to
        # min(5000, 10)=10, total energy ≈ energy of 20 tokens — tiny.
        # Unclipped: cache_energy(5000) >> uncached_energy(10).
        # So energy with 5000 cached must exceed energy with 0 cached by a wide margin.
        energy_no_cache = estimate_energy(self.MODEL, self.UNCACHED, 0, self.OUTPUT)
        self.assertGreater(
            energy, energy_no_cache * 3,
            msg=(
                f"Energy {energy:.6f} should be >> energy_no_cache {energy_no_cache:.6f} "
                f"when 5000 cached tokens are present. Likely still being clipped."
            ),
        )

    def test_estimate_cost_symmetric_with_details(self):
        cost = estimate_cost(self.MODEL, self.UNCACHED, self.CACHED, self.OUTPUT)
        details = estimate_cost_details(self.MODEL, self.UNCACHED, self.CACHED, self.OUTPUT)
        self.assertIsNotNone(cost)
        self.assertIsNotNone(details)
        self.assertAlmostEqual(cost, details[0], places=10)  # type: ignore[index]


# ---------------------------------------------------------------------------
# 2. total_tokens includes cached_input_tokens
# ---------------------------------------------------------------------------

class TestTotalTokens(unittest.TestCase):

    def test_usage_event_total_tokens_includes_cached(self):
        event = _make_event(input_tokens=10, cached_input_tokens=5_000, output_tokens=50)
        self.assertEqual(event.total_tokens, 10 + 5_000 + 50)

    def test_aggregate_total_tokens_includes_cached(self):
        agg = Aggregate()
        agg.add(_make_event(10, 5_000, 50))
        agg.add(_make_event(20, 3_000, 100))
        self.assertEqual(agg.total_tokens, (10 + 20) + (5_000 + 3_000) + (50 + 100))

    def test_aggregate_cached_ratio_uses_total_input_denominator(self):
        agg = Aggregate()
        agg.add(_make_event(input_tokens=100, cached_input_tokens=900, output_tokens=0))
        # ratio should be 900 / (100 + 900) = 0.9, not 900 / 100 = 9.0
        self.assertAlmostEqual(agg.cached_ratio, 0.9, places=5)

    def test_aggregate_cached_ratio_zero_when_no_tokens(self):
        agg = Aggregate()
        self.assertEqual(agg.cached_ratio, 0.0)


# ---------------------------------------------------------------------------
# 3. token_delta — cached may exceed uncached input
# ---------------------------------------------------------------------------

class TestTokenDelta(unittest.TestCase):

    def test_no_cap_when_cached_exceeds_uncached(self):
        # Simulate a turn where cached_total >> input_total (common in high-cache sessions)
        input_delta, cached_delta, output_delta, _ = token_delta(
            total_usage={"input_tokens": 10, "cached_input_tokens": 5_000, "output_tokens": 50},
            last_usage={},
            previous_totals=None,
        )
        self.assertEqual(input_delta, 10)
        self.assertEqual(cached_delta, 5_000)   # must NOT be capped to 10
        self.assertEqual(output_delta, 50)

    def test_delta_computation_with_previous_totals(self):
        prev = (100, 200, 50)
        input_delta, cached_delta, output_delta, new_totals = token_delta(
            total_usage={"input_tokens": 110, "cached_input_tokens": 5_200, "output_tokens": 55},
            last_usage={},
            previous_totals=prev,
        )
        self.assertEqual(input_delta, 10)
        self.assertEqual(cached_delta, 5_000)
        self.assertEqual(output_delta, 5)
        self.assertEqual(new_totals, (110, 5_200, 55))

    def test_last_usage_fallback_no_cap(self):
        input_delta, cached_delta, output_delta, totals = token_delta(
            total_usage={},
            last_usage={"input_tokens": 5, "cached_input_tokens": 3_000, "output_tokens": 20},
            previous_totals=None,
        )
        self.assertEqual(input_delta, 5)
        self.assertEqual(cached_delta, 3_000)   # must NOT be capped to 5
        self.assertEqual(output_delta, 20)
        self.assertIsNone(totals)


# ---------------------------------------------------------------------------
# 4a. TUR with missing timestamp — falls back to mtime, session not dropped
# ---------------------------------------------------------------------------

class TestTURMissingTimestamp(unittest.TestCase):

    def test_tur_with_no_timestamp_uses_file_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-09-16T10-00-00-abc123.jsonl"
            # token_usage_record with NO timestamp field
            lines = [
                {"type": "session_meta", "payload": {"id": "abc123"}, "timestamp": ""},
                {
                    "type": "token_usage_record",
                    # no "timestamp" key at all
                    "payload": {
                        "thread_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 500,
                            "output_tokens": 50,
                        }
                    },
                },
            ]
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            # Session must NOT be silently dropped
            self.assertGreater(len(events), 0, "Session with TUR but no timestamp should not be dropped")
            ev = events[0]
            self.assertEqual(ev.input_tokens, 100)
            self.assertEqual(ev.cached_input_tokens, 500)
            self.assertEqual(ev.output_tokens, 50)


# ---------------------------------------------------------------------------
# 4b. Mixed event types — TUR total vs delta sum comparison
# ---------------------------------------------------------------------------

class TestMixedEventTypes(unittest.TestCase):

    def _token_count_event(self, ts: str, total_input: int, total_cached: int, total_output: int) -> dict:
        return {
            "type": "event_msg",
            "timestamp": ts,
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": total_input,
                        "cached_input_tokens": total_cached,
                        "output_tokens": total_output,
                    }
                },
            },
        }

    def _tur_event(self, ts: str, input_tok: int, cached_tok: int, output_tok: int) -> dict:
        return {
            "type": "token_usage_record",
            "timestamp": ts,
            "payload": {
                "thread_token_usage": {
                    "input_tokens": input_tok,
                    "cached_input_tokens": cached_tok,
                    "output_tokens": output_tok,
                }
            },
        }

    def test_tur_greater_than_delta_uses_tur(self):
        """TUR has more total tokens than delta sum → use TUR (authoritative for complete session)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-09-16T10-00-00-aaa111.jsonl"
            lines = [
                # 2 token_count events (delta total = 50+500+10 + 80+1000+20 = 1660)
                self._token_count_event("2026-09-16T10:00:00Z", 50, 500, 10),
                self._token_count_event("2026-09-16T10:05:00Z", 130, 1500, 30),
                # TUR with higher cumulative (2000 total > 1660 delta)
                self._tur_event("2026-09-16T10:10:00Z", 200, 1700, 100),
            ]
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            self.assertEqual(len(events), 1)
            ev = events[0]
            # Should use TUR values
            self.assertEqual(ev.input_tokens, 200)
            self.assertEqual(ev.cached_input_tokens, 1700)
            self.assertEqual(ev.output_tokens, 100)

    def test_delta_greater_than_tur_keeps_deltas(self):
        """Delta sum exceeds TUR total (session continued with token_count after TUR) → keep deltas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-09-16T10-00-00-bbb222.jsonl"
            lines = [
                # TUR mid-session (small cumulative)
                self._tur_event("2026-09-16T10:05:00Z", 50, 200, 20),
                # Additional token_count events AFTER TUR (session continued)
                self._token_count_event("2026-09-16T10:06:00Z", 100, 500, 50),
                self._token_count_event("2026-09-16T10:07:00Z", 300, 2000, 150),
            ]
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            # Delta sum (100+500+50 + 200+1500+100 = 2450) > TUR total (270)
            # → keep the delta events, do not replace with TUR
            total_toks = sum(e.input_tokens + e.cached_input_tokens + e.output_tokens for e in events)
            self.assertGreater(total_toks, 270,
                               "Expected delta sum to be used when it exceeds TUR total")

    def test_tur_only_session_emits_event(self):
        """Session with ONLY token_usage_record events, no token_count at all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-09-16T10-00-00-ccc333.jsonl"
            lines = [
                self._tur_event("2026-09-16T10:00:00Z", 1000, 5000, 200),
                self._tur_event("2026-09-16T10:10:00Z", 1500, 8000, 350),  # last TUR wins
            ]
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            self.assertEqual(len(events), 1)
            ev = events[0]
            self.assertEqual(ev.input_tokens, 1500)
            self.assertEqual(ev.cached_input_tokens, 8000)
            self.assertEqual(ev.output_tokens, 350)


if __name__ == "__main__":
    unittest.main()
