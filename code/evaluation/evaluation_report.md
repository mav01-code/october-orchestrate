# Evaluation Report

## Scope and reproducibility

The only available answer-labelled benchmark is the 32-row public sample in `dataset/sample_output.csv`. No hidden labels were available. The 518-row full run can be structurally checked but has no valid full-set accuracy score.

Before edits, the original root `output.csv` was copied to ignored `output.baseline.csv`; a fresh run of the existing implementation produced a byte-identical 518-row result. The final deterministic full run was regenerated after the controlled experiments and compared with that saved baseline. The evaluator never reads sample answers as prediction inputs and never modifies the answer file.

## Baseline: existing implementation

| Metric | Result | Definition |
|---|---:|---|
| Status accuracy | 29/32 (90.6%) | Exact deletion status |
| COMPLETE status accuracy | 7/10 (70.0%) | Correct COMPLETE rows in the public sample |
| INCOMPLETE status accuracy | 22/22 (100%) | Correct INCOMPLETE rows in the public sample |
| Case/sequence exact match | 4/32 (12.5%) | Status and complete ordered step array both equal |
| Step precision | 99/151 (65.6%) | Exact semantic step objects, multiset matched / predicted steps |
| Step recall | 99/193 (51.3%) | Exact semantic step objects, multiset matched / expected steps |
| Step F1 | 57.6% | Harmonic mean of step precision and recall |
| Field accuracy, legacy calculation | 871/1,351 (64.5%) | Positional matching fields / seven fields per expected step; extra slots are excluded |
| Field accuracy, strict calculation | 871/1,365 (63.8%) | Positional field matches / seven fields per max(expected,predicted) step slot per case; missing and extra slots count |
| Evidence citation precision / recall | 54/55 (98.2%) / 54/75 (72.0%) | Reference ID multiset overlap; all 55 predicted references exist in participant-facing IDs |

The reported “approximately 65%” is reproducible as the **legacy field accuracy**, 871 / (193 expected steps × 7 fields) = 64.47%, rounded to 64.5% or 65%. It is not case-level status accuracy. The evaluator now also reports a stricter field score that penalizes extra predicted step slots.

## Controlled comparisons

| Variant | Calls / eligible evidence | Public status | Exact sequence | Step F1 | Field accuracy |
|---|---:|---:|---:|---:|---:|
| A. Existing baseline | 0 | 29/32 (90.6%) | 4/32 (12.5%) | 57.6% | 63.8% strict |
| B. Cross-system duplicate-resource experiment (reverted) | 0 | 29/32 (90.6%) | 4/32 (12.5%) | 57.1% | 63.1% strict |
| C. Gemini-assisted public-sample pilot | 0; sample preflight found 0 eligible groups | Not measured as an LLM variant | Not measured | Not measured | Not measured |
| D. Final deterministic output | 0 | 29/32 (90.6%) | 4/32 (12.5%) | 57.6% | 63.8% strict |

The cross-system experiment treated matching resource IDs in different systems as distinct records. On the public sample it added three extra steps to one case, changing predicted step objects from 151 to 154 and reducing step F1 from 57.6% to 57.1% and strict field accuracy from 63.8% to 63.1%. It was reverted because the available public labels did not support those extra steps. This is evidence that the attempted change degraded measured performance, not a general conclusion that duplicate IDs always denote one record.

The Gemini sample preflight found no candidate groups, so the public labels offered no eligible model-assisted comparison. The optional SDK (`google-genai`) was also not installed in the active Python environment. Although the local key and confirmations were present, no API request was sent. Four candidate groups exist in the unlabeled full request set, but they cannot support a measured accuracy claim; no full-dataset Gemini run was attempted.

## Error analysis

The three status mismatches are `case_01`, `case_02`, and `case_04`. For the first two, the SQL evidence is a time-bounded snapshot and the activity rows identify target resources, but the participant-visible records inspected here do not independently link those target resources to the requested subjects; the activity entries report operations, not outcomes. For the third, the analytics report has a record with null identity fields and the operation activity names a resource but does not link it to the requested subject. The current implementation therefore does not infer the missing relationship from case numbering, sample labels, or workflow claims. This leaves the public sample status errors unresolved pending stronger participant-visible linkage.

The semantic step comparison found 94 missing and 52 extra exact step objects. Most missing objects were verification checks: 33 `VERIFY_RECORD_ABSENT`, 20 `VERIFY_RECORD_PRESENT`, 12 `VERIFY_RECORD_ANONYMIZED`, 9 `VERIFY_RECORD_REDACTED`, 8 `VERIFY_DATA_FLOW_DISABLED`, and 6 `VERIFY_RESTORE_BLOCKED`. This indicates incomplete proof coverage and weak resource matching as the main measured limitations. Positional diagnostics also show 38 operation, 37 state/policy, 48 resource/system, and 1 evidence mismatch; these counts overlap missing/extra objects and are not an exclusive partition.

Identity-resolution errors cannot be isolated mechanically from the final CSV alone. The evaluator reports resource/system mismatches and the public-sample case list for manual evidence review. No hidden-set error claims are made.

## Additional labelled synthetic tests

These are hand-authored unit fixtures, not ground-truth benchmark cases and not included in the sample accuracy denominator:

1. **Historical email:** a request using an old address resolves through an `email` audit event to a present search index record; expected result is an evidence-cited finding plus delete and future verification steps.
2. **Active data flow:** a related enabled integration is reported and paired with `DISABLE_DATA_FLOW` and `VERIFY_DATA_FLOW_DISABLED` in phase order.
3. **Retained account tombstone:** the stable user reference remains, direct email is exposed, and the expected treatment is `REDACT_RECORD` followed by `VERIFY_RECORD_REDACTED`.
4. **Identifier-free readback:** a subject-linked resource key connects an identity-bearing observation to a later anonymized readback; the latest supported state is used for evidence-backed verification.

All four fixtures are tests in `code/test_main.py`; they do not constitute measured accuracy against challenge labels.

## Improvements and remaining limitations

- Added a semantic evaluator for status, exact ordered arrays, exact-step-object precision/recall/F1, strict field accuracy, citation validity/accuracy, and per-case plus operation-level missing/extra counts.
- Added the deterministic resource-key readback join described above in the earlier implementation. It did not change the public or full prediction output for this dataset. The later cross-system duplicate-resource variant was measured and reverted because public-sample metrics degraded.
- Tightened the full-run confirmation gate so it also applies when an interpreter is injected programmatically; a regression test verifies the fake interpreter receives no calls without the confirmation.
- Added an opt-in Gemini interpreter that accepts only schema-constrained claims whose candidate reference and exact quoted basis are locally validated. Deterministic Python remains authoritative for final status and policy steps.
- Gemini context is selected after deterministic identity matching, redacts known aliases and common identifier patterns, omits request text, and allowlists fields. The API path is off by default, bounded to 10 calls (5 by default), one SDK attempt, 8 evidence snippets/call, and stops after any API, parsing, or quota error.
- Added the three required historical-identity, active-flow, and retained-record synthetic tests, plus model validation, redaction, quota-stop, and output contract tests.

Remaining limitations include the weak exact sequence score, missing or unprovable resource links, incomplete verification planning, deterministic intent matching, manual screenshot transcription, and no live Gemini evaluation. With only 32 labelled public cases, the metrics are weak evidence for generalization. Hidden-set accuracy remains unknown.
