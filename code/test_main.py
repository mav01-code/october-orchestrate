import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from gemini_interpreter import GeminiInterpreter, MODEL_DEFAULT, safe_excerpt


class ContractTests(unittest.TestCase):
    def test_authorization_multilingual_and_non_deletion(self):
        self.assertTrue(main.authorizes_deletion("Please erase my account and remove its records."))
        self.assertTrue(main.authorizes_deletion("Intenté eliminar mi cuenta y borrar mis datos."))
        self.assertTrue(main.authorizes_deletion("このアカウントの削除を依頼します"))
        self.assertFalse(main.authorizes_deletion("Please provide a copy of my data for access."))
        self.assertFalse(main.authorizes_deletion("Please correct the delivery address on my order."))

    def test_phase_order_and_step_numbers(self):
        steps = main.number_steps(main.sort_phase([
            main.make_step("FINDING", "z", "r2", "REPORT_ACTIVE_DATA_FLOW", "ACTIVE", "e1"),
            main.make_step("FINDING", "a", "r1", "REPORT_REMAINING_RECORD", "PRESENT", "e1"),
        ]))
        self.assertEqual([1, 2], [s["step_no"] for s in steps])
        self.assertEqual("REPORT_REMAINING_RECORD", steps[0]["operation"])

    def test_timestamp_handles_zulu_and_rejects_bad(self):
        self.assertIsNotNone(main.parse_time("2026-09-15T12:00:00Z"))
        self.assertIsNone(main.parse_time("not-a-time"))

    def test_validator_rejects_nonconsecutive_steps(self):
        request = [{"case_id": "case_1"}]
        result = [{"case_id": "case_1", "deletion_status": "INCOMPLETE", "steps": [
            {"step_no": 2, "phase": "FINDING", "system_id": "x", "resource_id": "r",
             "operation": "REPORT_REMAINING_RECORD", "state": "PRESENT", "evidence_id": "e1"}
        ]}]
        with self.assertRaises(ValueError):
            main.validate_result(request, result, {"e1"})

    def test_validator_rejects_invalid_operation_state(self):
        request = [{"case_id": "case_1"}]
        result = [{"case_id": "case_1", "deletion_status": "INCOMPLETE", "steps": [
            {"step_no": 1, "phase": "FINDING", "system_id": "integration_service", "resource_id": "data_flow_1",
             "operation": "REPORT_ACTIVE_DATA_FLOW", "state": "PRESENT", "evidence_id": "e1"}
        ]}]
        with self.assertRaises(ValueError):
            main.validate_result(request, result, {"e1"})

    def test_validator_rejects_complete_without_verification(self):
        request = [{"case_id": "case_1"}]
        result = [{"case_id": "case_1", "deletion_status": "COMPLETE", "steps": []}]
        with self.assertRaises(ValueError):
            main.validate_result(request, result, {"e1"})

    def test_validator_rejects_incomplete_without_followup(self):
        request = [{"case_id": "case_1"}]
        result = [{"case_id": "case_1", "deletion_status": "INCOMPLETE", "steps": [
            {"step_no": 1, "phase": "FINDING", "system_id": "account_service", "resource_id": "account_1",
             "operation": "REPORT_REMAINING_RECORD", "state": "PRESENT", "evidence_id": "e1"}
        ]}]
        with self.assertRaises(ValueError):
            main.validate_result(request, result, {"e1"})

    def test_validator_rejects_unsorted_cases(self):
        requests = [{"case_id": "case_2"}, {"case_id": "case_1"}]
        results = [
            {"case_id": "case_2", "deletion_status": "NOT_APPLICABLE", "steps": []},
            {"case_id": "case_1", "deletion_status": "NOT_APPLICABLE", "steps": []},
        ]
        with self.assertRaises(ValueError):
            main.validate_result(requests, results, set())

    def test_sample_run_emits_valid_rows(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "code" / "evaluation") as td:
            out = Path(td) / "sample.csv"
            rows = main.run(repo / "dataset", out, repo / "dataset" / "sample_deletion_requests.csv", 32)
            self.assertEqual(32, len(rows))
            with out.open(encoding="utf-8", newline="") as fh:
                import csv
                saved = list(csv.DictReader(fh))
            self.assertEqual(32, len(saved))
            for row in saved:
                steps = json.loads(row["steps_json"])
                self.assertEqual(list(range(1, len(steps) + 1)), [s["step_no"] for s in steps])

    def test_full_run_confirmation_applies_to_injected_interpreters(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "code" / "evaluation") as td:
            temp_dir = Path(td)
            request_file = temp_dir / "requests.csv"
            all_requests = main.load_csv(repo / "dataset" / "deletion_requests.csv",
                                         {"case_id", "email", "requested_at", "audit_at", "request_text"})
            identities = main.load_csv(repo / "dataset" / "user_identities.csv",
                                       {"user_id", "name", "email", "phone", "email_hash", "updated_at"})
            audits = main.load_csv(repo / "dataset" / "user_audit_log.csv",
                                   {"audit_id", "user_id", "event_time", "field_name", "old_value", "new_value"})
            observations, evidence_meta = main.evidence_observations(repo / "dataset")
            candidate_request = next(
                r for r in all_requests
                if main.gemini_candidates_for_case(r, observations, identities, audits, evidence_meta)[0]
            )
            source = all_requests[:32]
            if candidate_request not in source:
                source = source[:32] + [candidate_request]
            else:
                source = source[:31] + [candidate_request]
            import csv
            with request_file.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(source[0]))
                writer.writeheader()
                writer.writerows(source)

            class FakeInterpreter:
                stopped = False
                calls = 0
                usage = SimpleNamespace(successful_calls=0, failed_calls=0, cache_hits=0,
                                         prompt_tokens=0, output_tokens=0,
                                         estimated_prompt_tokens=0, stopped_for_quota=False)
                model = MODEL_DEFAULT
                error_types = {}
                def classify(self, candidates):
                    self.calls += 1
                    return []

            fake = FakeInterpreter()
            with patch.dict(os.environ, {"GEMINI_FULL_RUN_CONFIRMED": ""}):
                main.run(repo / "dataset", temp_dir / "output.csv", request_file, 33,
                         gemini_interpreter=fake, gemini_enabled=True)
            self.assertEqual(0, fake.calls)

    def test_synthetic_historical_email_finds_related_record(self):
        request = {"case_id": "synthetic_history", "email": "old@example.test",
                   "audit_at": "2026-09-10T00:00:00Z", "request_text": "Delete my account"}
        identities = [{"user_id": "user_900", "name": "Test Person", "email": "new@example.test",
                       "phone": "5550100100", "email_hash": "hash", "updated_at": "2026-09-01T00:00:00Z"}]
        audits = [{"audit_id": "audit_900", "user_id": "user_900", "event_time": "2026-08-01T00:00:00Z",
                   "field_name": "email", "old_value": "old@example.test", "new_value": "new@example.test"}]
        obs = [main.Observation("search_service", "search_900", "SEARCH_INDEX_ENTRY", "PRESENT", "evidence_900",
                                "2026-09-02T00:00:00Z", {"email": "old@example.test"}, 2)]
        result = main.build_case(request, obs, identities, audits,
                                 {"evidence_900": {"captured_at": "2026-09-02T00:00:00Z"}},
                                 {"SEARCH_INDEX_ENTRY": "DELETE"})
        self.assertEqual("INCOMPLETE", result["deletion_status"])
        self.assertEqual("REPORT_REMAINING_RECORD", result["steps"][0]["operation"])
        self.assertEqual("evidence_900", result["steps"][0]["evidence_id"])

    def test_synthetic_active_flow_is_disabled_before_record_cleanup(self):
        request = {"case_id": "synthetic_flow", "email": "person@example.test",
                   "audit_at": "2026-09-10T00:00:00Z", "request_text": "Erase my account"}
        identities = [{"user_id": "user_901", "name": "Test Person", "email": "person@example.test",
                       "phone": "5550100101", "email_hash": "hash", "updated_at": "2026-09-01T00:00:00Z"}]
        obs = [main.Observation("integration_service", "data_flow_901", "DATA_FLOW", "ACTIVE", "evidence_901",
                                "2026-09-02T00:00:00Z", {"user_id": "user_901", "enabled": True}, 2)]
        result = main.build_case(request, obs, identities, [],
                                 {"evidence_901": {"captured_at": "2026-09-02T00:00:00Z"}}, {})
        self.assertEqual(["REPORT_ACTIVE_DATA_FLOW", "DISABLE_DATA_FLOW", "VERIFY_DATA_FLOW_DISABLED"],
                         [step["operation"] for step in result["steps"]])

    def test_synthetic_retained_tombstone_requires_redaction(self):
        request = {"case_id": "synthetic_tombstone", "email": "person@example.test",
                   "audit_at": "2026-09-10T00:00:00Z", "request_text": "Delete all my personal data"}
        identities = [{"user_id": "user_902", "name": "Test Person", "email": "person@example.test",
                       "phone": "5550100102", "email_hash": "hash", "updated_at": "2026-09-01T00:00:00Z"}]
        obs = [main.Observation("identity_service", "tombstone_902", "ACCOUNT_TOMBSTONE", "IDENTIFIABLE", "evidence_902",
                                "2026-09-02T00:00:00Z", {"user_id": "user_902", "email": "person@example.test"}, 2)]
        result = main.build_case(request, obs, identities, [],
                                 {"evidence_902": {"captured_at": "2026-09-02T00:00:00Z"}},
                                 {"ACCOUNT_TOMBSTONE": "REDACT"})
        self.assertEqual(["REPORT_EXPOSED_PII", "REDACT_RECORD", "VERIFY_RECORD_REDACTED"],
                         [step["operation"] for step in result["steps"]])

    def test_identifier_free_readback_joins_through_verified_resource_link(self):
        request = {"case_id": "synthetic_bridge", "email": "person@example.test",
                   "audit_at": "2026-09-10T00:00:00Z", "request_text": "Erase my account"}
        identities = [{"user_id": "user_903", "name": "Test Person", "email": "person@example.test",
                       "phone": "5550100103", "email_hash": "hash", "updated_at": "2026-09-01T00:00:00Z"}]
        obs = [
            main.Observation("analytics_platform", "analytics_903", "ANALYTICS_PROFILE", "IDENTIFIABLE", "evidence_link",
                            "2026-09-03T00:00:00Z", {"user_id": "user_903", "email": "person@example.test"}, 2),
            main.Observation("analytics_platform", "analytics_903", "ANALYTICS_PROFILE", "ANONYMIZED", "evidence_readback",
                            "2026-09-04T00:00:00Z", {"audience_cohort": "general_shopper"}, 3),
        ]
        evidence = {"evidence_link": {"captured_at": "2026-09-03T00:00:00Z"},
                    "evidence_readback": {"captured_at": "2026-09-04T00:00:00Z"}}
        result = main.build_case(request, obs, identities, [], evidence, {"ANALYTICS_PROFILE": "ANONYMIZE"})
        self.assertEqual("COMPLETE", result["deletion_status"])
        self.assertEqual("evidence_readback", result["steps"][0]["evidence_id"])
        self.assertEqual("VERIFY_RECORD_ANONYMIZED", result["steps"][0]["operation"])


class GeminiSafetyTests(unittest.TestCase):
    def test_model_is_fixed_to_verified_free_tier_allowlist(self):
        self.assertEqual("gemini-3.5-flash-lite", MODEL_DEFAULT)
        with self.assertRaises(ValueError):
            GeminiInterpreter(object(), "paid-model")

    def test_redaction_removes_direct_identifiers_from_context(self):
        excerpt = safe_excerpt({"message_text": "For Alice Example at alice@example.test call +1 555 111 2222"},
                              {"alice example", "alice@example.test", "+1 555 111 2222"})
        self.assertNotIn("Alice Example", excerpt)
        self.assertNotIn("alice@example.test", excerpt)
        self.assertNotIn("555 111 2222", excerpt)

    def test_structured_claim_is_provenance_checked_and_cached(self):
        response_text = json.dumps({"claims": [{"candidate_ref": "C001", "category": "USER_PROFILE",
                                               "state": "PRESENT", "basis": "found", "uncertain": False}]})
        response = SimpleNamespace(text=response_text, usage_metadata=SimpleNamespace(prompt_token_count=31,
                                                                                        candidates_token_count=12))
        class FakeModels:
            calls = 0
            def generate_content(self, **kwargs):
                self.calls += 1
                return response
        class FakeClient:
            models = FakeModels()
        interpreter = GeminiInterpreter(FakeClient(), config_factory=lambda **kwargs: kwargs)
        candidates = [{"candidate_ref": "C001", "excerpt": '{"result":"found"}'}]
        self.assertEqual("USER_PROFILE", interpreter.classify(candidates)[0]["category"])
        self.assertEqual("USER_PROFILE", interpreter.classify(candidates)[0]["category"])
        self.assertEqual(1, interpreter.client.models.calls)
        self.assertEqual(1, interpreter.usage.cache_hits)
        self.assertEqual(31, interpreter.usage.prompt_tokens)

    def test_structured_claim_cannot_invent_reference_or_quote(self):
        response = SimpleNamespace(text=json.dumps({"claims": [{"candidate_ref": "C999", "category": "USER_PROFILE",
                                                                   "state": "PRESENT", "basis": "made up", "uncertain": False}]}),
                                   usage_metadata=None)
        fake = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
        interpreter = GeminiInterpreter(fake, config_factory=lambda **kwargs: kwargs)
        self.assertEqual([], interpreter.classify([{"candidate_ref": "C001", "excerpt": '{"result":"found"}'}]))

    def test_quota_error_stops_additional_calls(self):
        class FakeModels:
            calls = 0
            def generate_content(self, **kwargs):
                self.calls += 1
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
        fake = SimpleNamespace(models=FakeModels())
        interpreter = GeminiInterpreter(fake, config_factory=lambda **kwargs: kwargs)
        candidate = [{"candidate_ref": "C001", "excerpt": '{"result":"found"}'}]
        self.assertEqual([], interpreter.classify(candidate))
        self.assertEqual([], interpreter.classify(candidate))
        self.assertEqual(1, fake.models.calls)
        self.assertTrue(interpreter.usage.stopped_for_quota)

    def test_missing_key_fails_closed(self):
        env = {"GEMINI_API_KEY": "", "GEMINI_FREE_TIER_CONFIRMED": "", "GEMINI_DATA_USE_ACK": ""}
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                GeminiInterpreter.from_environment()


if __name__ == "__main__":
    unittest.main()
