"""Compare prediction CSVs with the public answer CSV without modifying truth."""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

FIELDS = ("step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id")


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate case_id in {path}")
    return result


def parse_steps(row: dict[str, str], path: Path, case_id: str) -> list[dict[str, Any]]:
    try:
        steps = json.loads(row["steps_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid steps_json in {path}, {case_id}: {exc}") from exc
    if not isinstance(steps, list):
        raise ValueError(f"steps_json must be an array in {path}, {case_id}")
    return steps


def step_key(step: dict[str, Any]) -> str:
    """Canonical semantic object serialization; array order is evaluated separately."""
    return json.dumps(step, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def evaluate(expected_path: Path, predicted_path: Path, dataset: Path | None = None) -> dict[str, Any]:
    expected, predicted = read_csv(expected_path), read_csv(predicted_path)
    ids = sorted(set(expected) | set(predicted))
    if set(expected) != set(predicted):
        raise ValueError(f"Case coverage differs: missing={sorted(set(expected)-set(predicted))}; extra={sorted(set(predicted)-set(expected))}")

    status_ok = 0
    exact_case = 0
    expected_steps = predicted_steps = matched_steps = 0
    field_matches = field_slots = 0
    expected_counter: collections.Counter[str] = collections.Counter()
    predicted_counter: collections.Counter[str] = collections.Counter()
    evidence_expected = evidence_matched = evidence_valid = evidence_total = 0
    sequence_ok = 0
    per_status: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"cases": 0, "correct": 0})
    per_case_errors: list[dict[str, Any]] = []
    error_counts = collections.Counter()
    missing_by_operation: collections.Counter[str] = collections.Counter()
    extra_by_operation: collections.Counter[str] = collections.Counter()
    valid_refs: set[str] = set()
    if dataset:
        for filename, key in (("evidence_files.csv", "file_id"), ("system_activity.csv", "activity_id"), ("user_audit_log.csv", "audit_id")):
            file_path = dataset / filename
            if file_path.exists():
                with file_path.open(encoding="utf-8-sig", newline="") as f:
                    valid_refs.update(row[key] for row in csv.DictReader(f) if row.get(key))

    for case_id in ids:
        er, pr = expected[case_id], predicted[case_id]
        es, ps = parse_steps(er, expected_path, case_id), parse_steps(pr, predicted_path, case_id)
        s_ok = er.get("deletion_status") == pr.get("deletion_status")
        status_ok += s_ok
        kind = er.get("deletion_status", "<missing>")
        per_status[kind]["cases"] += 1
        per_status[kind]["correct"] += int(s_ok)
        ekeys = [step_key(x) for x in es]
        pkeys = [step_key(x) for x in ps]
        ec, pc = collections.Counter(ekeys), collections.Counter(pkeys)
        overlap = sum((ec & pc).values())
        missing_keys, extra_keys = ec - pc, pc - ec
        missing_by_operation.update(json.loads(key).get("operation", "<missing>") for key, count in missing_keys.items() for _ in range(count))
        extra_by_operation.update(json.loads(key).get("operation", "<missing>") for key, count in extra_keys.items() for _ in range(count))
        error_counts["missing_steps"] += sum(missing_keys.values())
        error_counts["extra_steps"] += sum(extra_keys.values())
        matched_steps += overlap
        expected_steps += len(es)
        predicted_steps += len(ps)
        expected_counter.update(ekeys)
        predicted_counter.update(pkeys)
        slot_count = max(len(es), len(ps))
        field_slots += slot_count * len(FIELDS)
        for index in range(slot_count):
            e = es[index] if index < len(es) else {}
            p = ps[index] if index < len(ps) else {}
            field_matches += sum(index < len(es) and index < len(ps) and e.get(field) == p.get(field) for field in FIELDS)
        case_sequence = s_ok and ekeys == pkeys
        sequence_ok += int(case_sequence)
        exact_case += int(case_sequence)
        ee = [x.get("evidence_id", "") for x in es if x.get("evidence_id")]
        pe = [x.get("evidence_id", "") for x in ps if x.get("evidence_id")]
        evidence_expected += len(ee)
        evidence_total += len(pe)
        evidence_matched += sum((collections.Counter(ee) & collections.Counter(pe)).values())
        if valid_refs:
            evidence_valid += sum(ref in valid_refs for ref in pe)
        if not case_sequence:
            if not s_ok:
                error_counts["status_mismatches"] += 1
            if ec == pc and ekeys != pkeys:
                error_counts["ordering_only_mismatches"] += 1
            for e, p in zip(es, ps):
                if e.get("phase") != p.get("phase"):
                    error_counts["phase_mismatches_at_position"] += 1
                if e.get("operation") != p.get("operation"):
                    error_counts["operation_mismatches_at_position"] += 1
                if e.get("state") != p.get("state"):
                    error_counts["state_or_policy_mismatches_at_position"] += 1
                if e.get("resource_id") != p.get("resource_id") or e.get("system_id") != p.get("system_id"):
                    error_counts["resource_or_system_mismatches_at_position"] += 1
                if e.get("evidence_id") != p.get("evidence_id"):
                    error_counts["evidence_mismatches_at_position"] += 1
            per_case_errors.append({
                "case_id": case_id,
                "expected_status": er.get("deletion_status"), "predicted_status": pr.get("deletion_status"),
                "expected_steps": len(es), "predicted_steps": len(ps),
                "step_object_matches": overlap,
                "missing_steps": sum(missing_keys.values()),
                "extra_steps": sum(extra_keys.values()),
                "sequence_match": False,
            })

    step_precision = matched_steps / predicted_steps if predicted_steps else 0.0
    step_recall = matched_steps / expected_steps if expected_steps else 0.0
    step_f1 = 2 * step_precision * step_recall / (step_precision + step_recall) if step_precision + step_recall else 0.0
    return {
        "cases": len(ids),
        "status_accuracy": {"correct": status_ok, "total": len(ids), "rate": status_ok / len(ids) if ids else 0.0},
        "per_status": dict(per_status),
        "case_exact_match": {"correct": exact_case, "total": len(ids), "rate": exact_case / len(ids) if ids else 0.0},
        "sequence_accuracy": {"correct": sequence_ok, "total": len(ids), "rate": sequence_ok / len(ids) if ids else 0.0},
        "steps": {"expected": expected_steps, "predicted": predicted_steps, "exact_object_matches": matched_steps,
                  "precision": step_precision, "recall": step_recall, "f1": step_f1},
        "field_accuracy": {"matching_fields": field_matches, "slots_including_missing_or_extra_steps": field_slots,
                           "rate": field_matches / field_slots if field_slots else 1.0},
        "evidence": {"expected_citations": evidence_expected, "predicted_citations": evidence_total,
                     "matching_citations": evidence_matched,
                     "citation_precision": evidence_matched / evidence_total if evidence_total else 0.0,
                     "citation_recall": evidence_matched / evidence_expected if evidence_expected else 0.0,
                     "valid_predicted_citations": evidence_valid if valid_refs else None,
                     "known_valid_reference_count": len(valid_refs)},
        "error_breakdown": {"counts": dict(error_counts), "missing_steps_by_operation": dict(missing_by_operation),
                            "extra_steps_by_operation": dict(extra_by_operation),
                            "identity_resolution": "not directly observable from final CSV; inspect resource/system mismatches and case evidence manually"},
        "errors": per_case_errors,
        "method": "Exact step-object multiset for step P/R/F1; status plus full ordered semantic JSON equality for case/sequence exact match; 7 output fields per positional slot, including missing/extra slots, for field accuracy. Positional mismatch counts are diagnostic and may overlap missing/extra counts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    result = evaluate(args.expected, args.predicted, args.dataset)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Status accuracy: {result['status_accuracy']['correct']}/{result['cases']} ({result['status_accuracy']['rate']:.1%})")
        print(f"Case/sequence exact: {result['case_exact_match']['correct']}/{result['cases']} ({result['case_exact_match']['rate']:.1%})")
        print("Step P/R/F1: {precision:.1%}/{recall:.1%}/{f1:.1%} ({exact_object_matches}/{expected} expected objects; {predicted} predicted)".format(**result["steps"]))
        print(f"Field accuracy: {result['field_accuracy']['matching_fields']}/{result['field_accuracy']['slots_including_missing_or_extra_steps']} ({result['field_accuracy']['rate']:.1%})")
        print("Status mismatches:", ", ".join(e["case_id"] for e in result["errors"] if e["expected_status"] != e["predicted_status"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
