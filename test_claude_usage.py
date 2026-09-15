import json
import tempfile
import unittest
from pathlib import Path
from codex_usage import (
    build_parser,
    scan_claude_code_sessions,
    DEFAULT_CLAUDE_CONTEXT_THRESHOLD,
    AZURE_GPT54_EU_STD_RATES,
    AZURE_GPT54_EU_LONG_RATES,
)


class TestClaudeUsage(unittest.TestCase):
    def test_cli_flags(self):
        parser = build_parser()

        # Test -claude flag
        args = parser.parse_args(["-claude"])
        self.assertTrue(args.claude)

        # Test --claude flag
        args = parser.parse_args(["--claude"])
        self.assertTrue(args.claude)

        # Test threshold flag
        args = parser.parse_args(["-claude", "--claude-threshold", "250000"])
        self.assertEqual(args.claude_threshold, 250000)

        # Test subcommands inherit -claude
        args_daily = parser.parse_args(["daily", "-claude"])
        self.assertTrue(args_daily.claude)

        args_sessions = parser.parse_args(["sessions", "-claude"])
        self.assertTrue(args_sessions.claude)

    def test_scan_claude_code_sessions_standard_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir) / "projects" / "test-project"
            proj_dir.mkdir(parents=True)
            session_file = proj_dir / "test-session-1.jsonl"

            # 2 assistant messages with duplicate chunks (same message id), standard context (< 272k)
            lines = [
                {
                    "type": "assistant",
                    "timestamp": "2026-09-10T10:00:00Z",
                    "message": {
                        "id": "msg_001",
                        "usage": {
                            "input_tokens": 1000,
                            "cache_creation_input_tokens": 500,
                            "cache_read_input_tokens": 2000,
                            "output_tokens": 300,
                        },
                    },
                },
                # Duplicate chunk for msg_001 (should be skipped)
                {
                    "type": "assistant",
                    "timestamp": "2026-09-10T10:00:01Z",
                    "message": {
                        "id": "msg_001",
                        "usage": {
                            "input_tokens": 1000,
                            "cache_creation_input_tokens": 500,
                            "cache_read_input_tokens": 2000,
                            "output_tokens": 300,
                        },
                    },
                },
                # Second distinct message
                {
                    "type": "assistant",
                    "timestamp": "2026-09-10T10:05:00Z",
                    "message": {
                        "id": "msg_002",
                        "usage": {
                            "input_tokens": 2000,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 1000,
                            "output_tokens": 500,
                        },
                    },
                },
            ]

            with session_file.open("w", encoding="utf-8") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")

            report = scan_claude_code_sessions(Path(tmpdir), None, None)
            self.assertEqual(len(report.sessions), 1)
            session = report.sessions["test-session-1"]

            self.assertEqual(session.session_id, "test-session-1")
            self.assertEqual(session.project_dir, "test-project")
            # msg_001 + msg_002 uncached input: (1000 + 500) + 2000 = 3500
            self.assertEqual(session.uncached_input_tokens, 3500)
            # msg_001 + msg_002 cached input: 2000 + 1000 = 3000
            self.assertEqual(session.cached_input_tokens, 3000)
            # output tokens: 300 + 500 = 800
            self.assertEqual(session.output_tokens, 800)
            self.assertEqual(session.total_tokens, 3500 + 3000 + 800)
            # Both were below 272k tokens
            self.assertEqual(session.long_turns, 0)
            self.assertEqual(session.standard_turns, 2)

            # Standard cost check — Azure GPT-5.4 Global Standard rates
            # Input: 3500 * ($2.50 / 1M) = $0.00875
            # Cached: 3000 * ($0.25 / 1M) = $0.00075
            # Output: 800 * ($15.00 / 1M) = $0.012
            expected_cost = (
                (3500 * AZURE_GPT54_EU_STD_RATES[0])
                + (800 * AZURE_GPT54_EU_STD_RATES[1])
                + (3000 * AZURE_GPT54_EU_STD_RATES[2])
            )
            self.assertAlmostEqual(session.estimated_cost_usd, expected_cost, places=6)

    def test_scan_claude_code_sessions_threshold_adaptation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj_dir = Path(tmpdir) / "projects" / "large-project"
            proj_dir.mkdir(parents=True)
            session_file = proj_dir / "large-session.jsonl"

            threshold = DEFAULT_CLAUDE_CONTEXT_THRESHOLD  # 272_000
            # Turn 1: 280k context (> 272k) -> long tier rate
            # Turn 2: 100k context (<= 272k) -> standard rate
            lines = [
                {
                    "type": "assistant",
                    "timestamp": "2026-09-12T12:00:00Z",
                    "message": {
                        "id": "msg_long",
                        "usage": {
                            "input_tokens": 100_000,
                            "cache_creation_input_tokens": 30_000,
                            "cache_read_input_tokens": 150_000,  # context = 100k + 30k + 150k = 280k > 272k
                            "output_tokens": 1_000,
                        },
                    },
                },
                {
                    "type": "assistant",
                    "timestamp": "2026-09-12T12:05:00Z",
                    "message": {
                        "id": "msg_std",
                        "usage": {
                            "input_tokens": 20_000,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 80_000,  # context = 100k <= 272k
                            "output_tokens": 1_000,
                        },
                    },
                },
            ]

            with session_file.open("w", encoding="utf-8") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")

            report = scan_claude_code_sessions(Path(tmpdir), None, None)
            session = report.sessions["large-session"]

            self.assertEqual(session.long_turns, 1)
            self.assertEqual(session.standard_turns, 1)

            # Turn 1 cost (long context rates)
            turn1_cost = (
                (130_000 * AZURE_GPT54_EU_LONG_RATES[0])
                + (1_000 * AZURE_GPT54_EU_LONG_RATES[1])
                + (150_000 * AZURE_GPT54_EU_LONG_RATES[2])
            )

            # Turn 2 cost (standard context rates)
            turn2_cost = (
                (20_000 * AZURE_GPT54_EU_STD_RATES[0])
                + (1_000 * AZURE_GPT54_EU_STD_RATES[1])
                + (80_000 * AZURE_GPT54_EU_STD_RATES[2])
            )

            expected_total_cost = turn1_cost + turn2_cost
            self.assertAlmostEqual(session.estimated_cost_usd, expected_total_cost, places=6)


if __name__ == "__main__":
    unittest.main()
