# Gemini Usage and Cost Report

## Final run that produced `output.csv`

- Run date: 2026-10-25 (Asia/Calcutta).
- Cases processed: 518; Gemini cases: 0.
- Provider/model actually used: none. `gemini-3.5-flash-lite` is configured as the optional candidate model, but it was not called.
- SDK: `google-genai` is declared in `code/requirements.txt`; it was not installed in the active Python environment (`pip show google-genai` reported no package), so a live pilot could not be executed.
- Successful/failed API calls: 0 / 0. Prompt, output, and total API tokens: 0 / 0 / 0. These zeros describe the absence of API traffic, not a model response with zero token usage.
- Average API tokens per dataset case: 0 / 518 = 0 because no case was sent. Average per API request is not applicable because there were no requests.
- Cache hits, retries, API errors, and quota exhaustion: 0, 0, 0, and not observed, respectively. The application cannot inspect account-specific quota; no quota lookup was made.
- API cost incurred: $0 because no API request was made. Estimated cost for a hypothetical call is not reported. This is not a claim that a configured account or future request is covered by the Free tier. Local compute and electricity are outside this API-usage report.

The key and three confirmation settings were checked without displaying their values: a key was present and each required confirmation equaled `YES`. The final output command did not enable `--gemini-pilot`. The public 32-case sample preflight found zero Gemini-eligible groups, so it could not support a labeled pilot comparison. No full-dataset Gemini pass was attempted. A separate full-input candidate preflight previously identified 7 redacted excerpts in 4 unlabeled groups; this is not a model-call or accuracy result and was not used to send data.

## Free-tier and privacy limitations

Google's official model page currently lists `gemini-3.5-flash-lite` as stable and supporting structured outputs, and the official Standard pricing table lists Free-tier input and output tokens for this model. This verifies published model eligibility only, not this account's project tier or remaining quota. Check the project's tier and quota in AI Studio before any future call; the application cannot verify them.

Google's [Gemini API terms](https://ai.google.dev/gemini-api/terms) state that unpaid-service prompts and responses may be used to improve Google products and may be reviewed by human reviewers, and advise against submitting sensitive, confidential, or personal information. No dataset excerpt was sent during the final run. The optional code applies local redaction but cannot guarantee that all identifying information is removed.

No paid endpoint, billing upgrade, provider fallback, Batch API, caching service, grounding, or search feature was used. No model totals apply because no model was called.
