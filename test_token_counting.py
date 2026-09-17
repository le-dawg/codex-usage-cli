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

    def test_legacy_token_delta_uncached_subtraction(self):
        """total_usage['input_tokens'] is total prompt (uncached + cached).
        token_delta must return uncached prompt as input_delta."""
        total_usage_1 = {
            "input_tokens": 100_000,
            "cached_input_tokens": 80_000,
            "output_tokens": 1_000,
        }
        unc, cac, out, prev = token_delta(
            total_usage=total_usage_1,
            last_usage={},
            previous_totals=None,
        )
        self.assertEqual(unc, 20_000)  # 100k total - 80k cached
        self.assertEqual(cac, 80_000)
        self.assertEqual(out, 1_000)
        self.assertEqual(prev, (100_000, 80_000, 1_000))

        # Turn 2: total grows to 150k input (120k cached), 2k output
        total_usage_2 = {
            "input_tokens": 150_000,
            "cached_input_tokens": 120_000,
            "output_tokens": 2_000,
        }
        unc2, cac2, out2, prev2 = token_delta(
            total_usage=total_usage_2,
            last_usage={},
            previous_totals=prev,
        )
        # Delta total prompt = 50k, delta cached = 40k -> uncached = 10k
        self.assertEqual(unc2, 10_000)
        self.assertEqual(cac2, 40_000)
        self.assertEqual(out2, 1_000)
        self.assertEqual(prev2, (150_000, 120_000, 2_000))

    def test_legacy_last_token_usage_uncached_subtraction(self):
        """last_token_usage['input_tokens'] is total prompt (uncached + cached)."""
        last_usage = {
            "input_tokens": 50_000,
            "cached_input_tokens": 45_000,
            "output_tokens": 500,
        }
        unc, cac, out, prev = token_delta(
            total_usage={},
            last_usage=last_usage,
            previous_totals=None,
        )
        self.assertEqual(unc, 5_000)  # 50k - 45k
        self.assertEqual(cac, 45_000)
        self.assertEqual(out, 500)

    def test_no_cap_when_cached_exceeds_uncached(self):
        # Simulate a turn where cached_total >> uncached (common in high-cache sessions)
        # input_tokens is total prompt: 5,010 total - 5,000 cached = 10 uncached
        input_delta, cached_delta, output_delta, _ = token_delta(
            total_usage={"input_tokens": 5_010, "cached_input_tokens": 5_000, "output_tokens": 50},
            last_usage={},
            previous_totals=None,
        )
        self.assertEqual(input_delta, 10)
        self.assertEqual(cached_delta, 5_000)   # must NOT be capped to 10
        self.assertEqual(output_delta, 50)

    def test_delta_computation_with_previous_totals(self):
        prev = (100, 200, 50)
        # Delta total prompt = 5,110 - 100 = 5,010. Delta cached = 5,200 - 200 = 5,000.
        # Uncached delta = 5,010 - 5,000 = 10.
        input_delta, cached_delta, output_delta, new_totals = token_delta(
            total_usage={"input_tokens": 5_110, "cached_input_tokens": 5_200, "output_tokens": 55},
            last_usage={},
            previous_totals=prev,
        )
        self.assertEqual(input_delta, 10)
        self.assertEqual(cached_delta, 5_000)
        self.assertEqual(output_delta, 5)
        self.assertEqual(new_totals, (5_110, 5_200, 55))

    def test_last_usage_fallback_no_cap(self):
        # input_tokens is total prompt: 3,005 total - 3,000 cached = 5 uncached
        input_delta, cached_delta, output_delta, totals = token_delta(
            total_usage={},
            last_usage={"input_tokens": 3_005, "cached_input_tokens": 3_000, "output_tokens": 20},
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
                        "usage": {
                            "input_tokens": 600,
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
            self.assertEqual(ev.input_tokens, 100)  # 600 total - 500 cached
            self.assertEqual(ev.cached_input_tokens, 500)
            self.assertEqual(ev.output_tokens, 50)


# ---------------------------------------------------------------------------
# 4b. Mixed and Per-Turn Event Types
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

    def _tur_turn_event(self, ts: str, input_tok: int, cached_tok: int, output_tok: int) -> dict:
        return {
            "type": "token_usage_record",
            "timestamp": ts,
            "payload": {
                "usage": {
                    "input_tokens": input_tok,
                    "cached_input_tokens": cached_tok,
                    "output_tokens": output_tok,
                }
            },
        }

    def test_tur_file_emits_per_turn_events(self):
        """Files with TUR emit discrete per-turn UsageEvents based on payload['usage']."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-09-16T10-00-00-aaa111.jsonl"
            lines = [
                self._tur_turn_event("2026-09-16T10:00:00Z", 200, 150, 20),
                self._tur_turn_event("2026-09-16T10:05:00Z", 300, 250, 30),
            ]
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            self.assertEqual(len(events), 2)
            # Turn 1: 200 prompt - 150 cached = 50 uncached
            self.assertEqual(events[0].input_tokens, 50)
            self.assertEqual(events[0].cached_input_tokens, 150)
            self.assertEqual(events[0].output_tokens, 20)
            # Turn 2: 300 prompt - 250 cached = 50 uncached
            self.assertEqual(events[1].input_tokens, 50)
            self.assertEqual(events[1].cached_input_tokens, 250)
            self.assertEqual(events[1].output_tokens, 30)

    def test_tur_presence_skips_token_count_double_counting(self):
        """When TUR is present, token_count events are skipped to avoid multi-counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-09-16T10-00-00-bbb222.jsonl"
            lines = [
                self._tur_turn_event("2026-09-16T10:00:00Z", 100, 80, 10),
                self._token_count_event("2026-09-16T10:00:00Z", 100, 80, 10),
            ]
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].input_tokens, 20)  # 100 - 80
            self.assertEqual(events[0].cached_input_tokens, 80)
            self.assertEqual(events[0].output_tokens, 10)

    def test_hybrid_transition_token_count_to_tur(self):
        """When a session begins with legacy token_count events and transitions to TUR,
        legacy turns before TUR emit delta events, and modern turns emit TUR events
        while skipping companion token_count events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-08-25T10-00-00-hybrid.jsonl"
            lines = [
                # Turn 1: legacy token_count only
                self._token_count_event("2026-08-25T10:00:00Z", 50_000, 40_000, 500),
                # Turn 2: legacy token_count only (5 minutes later)
                self._token_count_event("2026-08-25T10:05:00Z", 100_000, 80_000, 1_000),
                # Turn 3: resumed weeks later with TUR + companion token_count
                self._tur_turn_event("2026-09-10T14:00:00Z", 60_000, 50_000, 600),
                self._token_count_event("2026-09-10T14:00:00Z", 160_000, 130_000, 1_600),
            ]
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            # Must have 3 events total (2 from legacy token_count, 1 from TUR)
            self.assertEqual(len(events), 3)

            # Turn 1 (legacy): 50k input - 40k cached = 10k uncached
            self.assertEqual(events[0].input_tokens, 10_000)
            self.assertEqual(events[0].cached_input_tokens, 40_000)
            self.assertEqual(events[0].output_tokens, 500)
            self.assertEqual(events[0].day, "2026-08-25")

            # Turn 2 (legacy delta): (100k - 50k) - (80k - 40k) = 10k uncached
            self.assertEqual(events[1].input_tokens, 10_000)
            self.assertEqual(events[1].cached_input_tokens, 40_000)
            self.assertEqual(events[1].output_tokens, 500)
            self.assertEqual(events[1].day, "2026-08-25")

            # Turn 3 (TUR): 60k input - 50k cached = 10k uncached
            self.assertEqual(events[2].input_tokens, 10_000)
            self.assertEqual(events[2].cached_input_tokens, 50_000)
            self.assertEqual(events[2].output_tokens, 600)
            self.assertEqual(events[2].day, "2026-09-10")



# ---------------------------------------------------------------------------
# 5. Startup history replay burst filtering in subagent rollout files
# ---------------------------------------------------------------------------

class TestSubagentStartupReplay(unittest.TestCase):

    def test_subagent_startup_replay_filtered(self):
        """When a subagent rollout file dumps 10 historical turns at startup within milliseconds,
        it should not re-accumulate the parent's history as new tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-08-25T15-52-34-subagent.jsonl"
            lines = [
                {"type": "session_meta", "payload": {"id": "subagent-1", "parent_id": "parent-1"}},
            ]
            # 10 historical burst events at exact same startup millisecond
            for i in range(1, 11):
                lines.append({
                    "type": "event_msg",
                    "timestamp": "2026-08-25T15:52:34.400Z",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": i * 10_000,
                                "cached_input_tokens": i * 8_000,
                                "output_tokens": i * 100,
                            }
                        }
                    }
                })
            # 1 real turn executed 10 seconds later
            lines.append({
                "type": "event_msg",
                "timestamp": "2026-08-25T15:52:44.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 105_000,
                            "cached_input_tokens": 84_000,
                            "output_tokens": 1_050,
                        }
                    }
                }
            })
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            # Only the single real turn executed post-startup should be recorded
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].input_tokens, 1_000)  # (105k - 100k) - (84k - 80k) = 5k - 4k = 1k
            self.assertEqual(events[0].cached_input_tokens, 4_000)
            self.assertEqual(events[0].output_tokens, 50)

    def test_subagent_startup_replay_only_no_real_turns(self):
        """When a subagent rollout file only contains the startup replay burst,
        zero synthetic events should be emitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-08-25T15-52-34-replay-only.jsonl"
            lines = [
                {"type": "session_meta", "payload": {"id": "subagent-dead", "parent_id": "parent-1"}},
            ]
            for i in range(1, 8):
                lines.append({
                    "type": "event_msg",
                    "timestamp": "2026-08-25T15:52:34.400Z",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": i * 10_000,
                                "cached_input_tokens": i * 8_000,
                                "output_tokens": i * 100,
                            }
                        }
                    }
                })
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            self.assertEqual(len(events), 0)

    def test_fast_session_under_burst_threshold_not_filtered(self):
        """When a normal fast session has <= 5 turns within 1 second,
        it should not be filtered as a replay burst."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollout-2026-08-25T15-52-34-fast.jsonl"
            lines = [
                {"type": "session_meta", "payload": {"id": "fast-1"}},
            ]
            for i in range(1, 4):  # 3 turns
                lines.append({
                    "type": "event_msg",
                    "timestamp": "2026-08-25T15:52:34.400Z",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": i * 1_000,
                                "cached_input_tokens": i * 800,
                                "output_tokens": i * 50,
                            }
                        }
                    }
                })
            _make_session_jsonl(lines, path)

            sid, title, events, _ = parse_session_file(path, {}, with_cost=False)
            # All 3 turns should be preserved
            self.assertEqual(len(events), 3)
            for ev in events:
                self.assertEqual(ev.input_tokens, 200)
                self.assertEqual(ev.cached_input_tokens, 800)
                self.assertEqual(ev.output_tokens, 50)


if __name__ == "__main__":
    unittest.main()


