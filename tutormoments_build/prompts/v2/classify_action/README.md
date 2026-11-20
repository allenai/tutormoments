# classify_action: prediction → gold mapping

`tutormoments_build/v2/evaluate_taxonomy.py` scores this prompt's labels against
the adjudicators' strategy lists. The gold vocabulary and the prompt's vocabulary
are not identical, so the evaluator maps one onto the other. This file records
every change to that mapping and why, so a score can be read against the mapping
it was computed under.

Keep it in sync with `GOLD_ALIASES` and the comparison rules in
`evaluate_taxonomy.py`.

## Current mapping

### Label aliases (`GOLD_ALIASES`)

Applied to **gold only**. Predicted labels are compared as the model emits them.

| Gold label | Scored as | Added | Why |
|---|---|---|---|
| `re_explain_procedure` | `re_explain` | 2026-11-13 | The annotation form splits re-explaining a concept from re-explaining a procedure; the prompt has one `re_explain`. |
| `different_representation` | `diff_rep` | 2026-11-13 | Naming only: same category, shorter name in the prompt. |
| `simpler_language` | `re_explain` | 2026-11-14 | Annotators file "simpler alternative" moves under `re_explain` (e.g. `48a5b4a6`: both adjudicators wrote "uses a simpler comparison" and labelled it `re_explain`). Gold only, so a predicted `simpler_language` never matches and always counts as a false positive. |

### Comparison rules

- **Gold is the union of the adjudicators' labels.** First pass (selector) and
  second pass (reannotator) strategies are not used as gold.
- **A family is compared only where the gold label marks it present.**
  Scaffolding strategies count when `scaffolding_present`, rigor strategies when
  `rigor_present`. These are the same families the prompt's `{action}` slot is
  filled from.
- **Predicted labels the prompt doesn't list are set aside** (`unknown_labels`),
  not counted as false positives.
- **Free-text "other" strategies are not compared.** They have no category.

## Change log

- **2026-11-14:** gold `simpler_language` → `re_explain` (gold only).
- **2026-11-13:** initial aliases `re_explain_procedure` → `re_explain` and
  `different_representation` → `diff_rep`.
