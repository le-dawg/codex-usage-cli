from pathlib import Path
import unittest

import codex_usage as cu


class PricingResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_pricing = dict(cu.PRICING)

    def tearDown(self) -> None:
        cu.PRICING.clear()
        cu.PRICING.update(self.original_pricing)

    def set_pricing(self, pricing: dict[str, tuple[float, float, float | None]]) -> None:
        cu.PRICING.clear()
        cu.PRICING.update(pricing)

    def test_exact_gpt6_astra_is_exact_not_guess(self) -> None:
        rate = (3.0e-6, 1.8e-5, 3.0e-7)
        self.set_pricing({"gpt-6-astra": rate})
        resolved = cu.resolve_pricing("gpt-6-astra")
        self.assertEqual(resolved, (rate, False))

    def test_alias_variant_resolves_to_gpt6_astra_pricing(self) -> None:
        rate = (3.0e-6, 1.8e-5, 3.0e-7)
        self.set_pricing({"gpt-6-astra": rate})
        resolved = cu.resolve_pricing("openai/gpt6_astra-2026-09-10-high")
        self.assertEqual(resolved, (rate, False))

    def test_gpt6_family_fallback_is_guess_when_exact_missing(self) -> None:
        rate = (2.5e-6, 1.5e-5, 2.5e-7)
        self.set_pricing({"gpt-6": rate})
        resolved = cu.resolve_pricing("gpt-6-astra-2026-09-10-high")
        self.assertEqual(resolved, (rate, True))

    def test_unknown_model_stays_unresolved(self) -> None:
        self.set_pricing({"gpt-6-astra": (3.0e-6, 1.8e-5, 3.0e-7)})
        self.assertIsNone(cu.resolve_pricing("unknown-custom-model"))

    def test_existing_gpt4_5_behavior_regression_guard(self) -> None:
        rate = (1.75e-6, 1.4e-5, 1.75e-7)
        self.set_pricing({"gpt-5.2-codex": rate, "gpt-5.2": rate})
        self.assertEqual(cu.resolve_pricing("openai/gpt-5.2-codex"), (rate, False))
        self.assertIsNone(cu.resolve_pricing("gpt-4o"))

    def test_models_and_sessions_cost_columns_show_non_dash_for_resolved_gpt6(self) -> None:
        rate = (3.0e-6, 1.8e-5, 3.0e-7)
        self.set_pricing({"gpt-6-astra": rate})
        normalized = cu.normalize_model("openai/gpt6_astra-2026-09-10-high")
        estimated = cu.estimate_cost_details(normalized, input_tokens=1000, cached_input_tokens=100, output_tokens=500)
        self.assertIsNotNone(estimated)
        cost_value, is_guess = estimated  # type: ignore[misc]
        event = cu.UsageEvent(
            session_id="11111111-1111-1111-1111-111111111111",
            session_title="title",
            day="2026-09-10",
            timestamp="2026-09-10T12:00:00+00:00",
            model=normalized,
            input_tokens=1000,
            cached_input_tokens=100,
            output_tokens=500,
            plan_type=None,
            estimated_energy_wh=0.0,
            estimated_cost_usd=cost_value,
            estimated_cost_is_guess=is_guess,
        )
        report = cu.build_report(
            Path("."),
            None,
            None,
            [event],
            cu.ScanDiagnostics(parsed_events=1),
            {"source": "builtin"},
        )
        model_rows = cu.build_model_rows(report, with_cost=True, limit=10, unicode_ok=False)
        session_rows = cu.build_session_rows(report, with_cost=True, limit=10, unicode_ok=False, censored=False)
        self.assertNotEqual(model_rows[0][-1], "-")
        self.assertNotEqual(session_rows[0][-1], "-")


if __name__ == "__main__":
    unittest.main()
