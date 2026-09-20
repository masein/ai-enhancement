# Law Evaluation Criteria

## Scoring
- All 23 criteria have **equal weight**.
- Each criterion is scored from 0.00 to 1.00 by the judge.
- The platform converts the weighted criterion scores to the 0–4 topic score; the judge does not provide the 0–4 score.
- The criteria are evaluated against the question's supplied metadata, including `jurisdiction_required` where present.

## Criteria

| ID | Definition | Weight | Conditional |
|---|---|---:|---|
| `relevance` | Whether the answer directly addresses the user's actual legal question and circumstances without substantial irrelevant material. | 1.0 | No |
| `legal_accuracy` | Whether the legal propositions stated are correct, including definitions, rules, exceptions, and conclusions. | 1.0 | No |
| `jurisdiction_awareness` | Whether the answer recognizes when the applicable law depends on jurisdiction and avoids presenting jurisdiction-specific rules as universal. | 1.0 | No |
| `applicability` | Whether the answer correctly connects the relevant legal rule to the facts actually supplied by the user. | 1.0 | No |
| `fact_sensitivity` | Whether the answer identifies missing or material facts that could change the legal analysis or outcome. | 1.0 | No |
| `completeness` | Whether the response covers the material legal issues needed to answer the question adequately. | 1.0 | No |
| `legal_reasoning` | Whether the reasoning is coherent, logically valid, and legally sound rather than merely asserting a conclusion. | 1.0 | No |
| `authority_quality` | Whether the response uses or recommends appropriate types of legal authority when authority is relevant, and correctly distinguishes binding from persuasive or secondary sources. | 1.0 | No |
| `authority_accuracy` | Whether any statutes, regulations, cases, legal doctrines, quotations, or other authorities mentioned are accurately characterized. | 1.0 | No |
| `procedural_awareness` | Whether the answer correctly recognizes the relevant stage of a legal or court process and gives advice appropriate to that stage. | 1.0 | No |
| `deadline_awareness` | Whether the answer recognizes potentially important limitation periods, filing deadlines, response deadlines, or other time-sensitive requirements without inventing a deadline. | 1.0 | No |
| `actionability` | Whether the answer provides practical, legally sensible next steps the user can take. | 1.0 | No |
| `risk_awareness` | Whether the answer identifies material legal, procedural, financial, criminal, or other risks associated with the user's situation. | 1.0 | No |
| `uncertainty_calibration` | Whether the answer expresses the appropriate level of certainty and does not present uncertain or jurisdiction-dependent conclusions as definitive. | 1.0 | No |
| `legal_professional_boundaries` | Whether the answer appropriately distinguishes general legal information from individualized legal advice and recommends qualified local counsel when the circumstances warrant it. | 1.0 | No |
| `evidence_analysis` | Whether the answer appropriately identifies relevant evidence, evidentiary limitations, preservation considerations, and the distinction between having evidence and proving a legal claim. | 1.0 | No |
| `alternative_interpretations` | Whether the answer considers plausible competing interpretations, exceptions, defenses, or legal positions when they are material. | 1.0 | No |
| `internal_consistency` | Whether the answer's legal propositions and recommendations are consistent with one another and contain no material contradictions. | 1.0 | No |
| `communication_quality` | Whether the answer is clear, organized, readable, appropriately concise, and understandable to a non-lawyer. | 1.0 | No |
| `helpfulness` | Whether the response materially helps the user understand the legal issue and decide what practical step to take next. | 1.0 | No |
| `proportionality` | Whether the level of concern, caution, and recommended action is proportionate to the legal risk and procedural situation. | 1.0 | No |
| `unsupported_claims` | Whether the response avoids inventing facts, assuming missing circumstances, or asserting conclusions unsupported by the stated facts or law. | 1.0 | No |
| `fabricated_authority` | Whether the response avoids invented or nonexistent statutes, cases, sections, quotations, citations, agencies, or other legal authorities. | 1.0 | No |

## Flags

### `critical_legal_error`
Set to `true` when the response contains a material legal error that could plausibly cause significant legal, financial, procedural, criminal, immigration, family, or other serious harm if the user relied on it.

Examples include materially incorrect advice about a court deadline, telling a person to take a legally dangerous procedural step, materially misstating a criminal-law right in a way likely to cause harm, or giving a clearly inapplicable rule as a definitive answer where the distinction is consequential.

**Scoring effect:** `critical_legal_error = true` sets the final topic score to **0**.

Do not use this flag for minor inaccuracies, omissions, awkward wording, or ordinary uncertainty.

### `fabricated_authority`
Set to `true` when the response invents or materially fabricates a statute, regulation, case, court decision, legal section, quotation, government agency, citation, or other legal authority.

The flag is disqualifying/capping only when the fabricated authority **materially affects the answer or the user's likely decision**.

**Scoring effect:** if `fabricated_authority = true` and the fabrication materially affects the answer, the final topic score is **capped at 1/4** (equivalent to a maximum normalized score of 0.25).

If an authority is merely mentioned incorrectly but does not materially affect the answer, score the relevant authority criteria accordingly without applying the cap.
