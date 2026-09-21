# Law Evaluation Criteria

## Task

Score **ONE answer to ONE Law question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, jurisdiction requirement, procedural context, and assumptions.

Accept different legally defensible arguments and conclusions where the authority, facts, and procedural posture genuinely support them. Do **not** require a particular wording, IRAC structure, citation style, or rhetorical format. Do not reward unnecessary length, legalese, case-name dropping, or citations merely because they appear.

The benchmark evaluates **legal reasoning**, not resemblance to a reference answer.

## Core Legal-Evidence Principles

- **Facts are not allegations.** An allegation, assertion, or argument is not automatically an established fact.
- **Rules are jurisdiction-dependent.** Do not import legal rules from another jurisdiction without support.
- **Authority has hierarchy.** Binding authority generally carries different legal weight from persuasive or secondary authority.
- **Holding is not dicta.** Not every statement in a judicial opinion is equally authoritative.
- **Legal analysis is element-based.** Where a claim has multiple elements, each material element must be addressed.
- **Exceptions matter.** A correct general rule can still produce the wrong result if a relevant exception applies.
- **Defenses matter.** Establishing a prima facie claim does not necessarily establish final liability.
- **Procedural posture matters.** Pleading, summary judgment, trial, and appeal can require different treatment of the same factual dispute.
- **Burdens matter.** Who must prove what can be outcome-determinative.
- **Evidence quality matters.** A legal conclusion should reflect the strength and admissibility of supporting evidence where relevant.
- **Statutory text matters.** Do not replace enacted language with generalized policy intuition.
- **Context matters.** Defined terms, surrounding provisions, structure, and purpose may affect meaning.
- **Precedent requires comparison.** Compare cases using legally material facts, governing principles, hierarchy, and procedural posture.
- **Temporal validity matters.** Use the law in effect at the legally relevant time.
- **Liability is not remedy.** Establishing liability does not automatically determine relief, amount, or procedure.
- **Legality is not morality.** Moral approval or disapproval does not by itself establish legal consequence.
- **Policy is not doctrine.** Policy considerations may inform interpretation but do not automatically override controlling authority.
- **Uncertainty should be explicit.** Missing facts, ambiguous text, conflicting authority, or discretionary standards should be identified.
- **Multiple reasonable legal arguments may exist.** Evaluate authority, reasoning, factual application, and counterarguments rather than forcing a predetermined conclusion.
- **No case-name worship.** Knowing a case name without correctly applying its principle should not receive a high score.
- **No citation worship.** A citation helps only if the authority actually supports the proposition and has appropriate legal weight.
- **No disclaimer worship.** Generic caution such as “consult a lawyer” does not compensate for failure to analyze the legal question.

## Critical Error

If `critical_legal_error` applies, **score = 0** regardless of the remaining criteria.

A critical legal error is a materially false central legal claim, jurisdictional assumption, procedural error, authority error, evidentiary error, or element-level mistake that fundamentally invalidates the answer or reverses its substantive legal conclusion.

Examples include:

- inventing a central legal rule;
- applying the wrong jurisdiction without justification where outcome-determinative;
- treating nonbinding authority as controlling where that changes the result;
- fundamentally misidentifying a supplied holding;
- ignoring an explicit statutory exception that reverses the outcome;
- omitting a required element and still concluding liability is established;
- shifting an outcome-determinative burden to the wrong party;
- treating allegations as established facts contrary to procedural posture;
- applying a trial standard at pleading in a way that invalidates the result;
- treating a clearly time-barred claim as timely;
- contradicting an explicit statutory definition;
- treating possession as ownership where the distinction controls;
- treating authentication as proof of truth;
- treating relevance as sufficient for admissibility despite a supplied exclusion rule;
- fundamentally incorrect causation that reverses liability;
- ignoring an expressly supplied complete defense;
- applying the wrong version of law after an effective-date change;
- fabricating controlling authority or procedural facts;
- asserting certainty where supplied authorities genuinely conflict and no controlling authority resolves the issue;
- reaching a legal conclusion that directly contradicts controlling law supplied in the question.

Do **not** classify as critical: minor terminology errors, harmless citation-format mistakes, reasonable differences in emphasis, defensible alternative interpretations, reasonable uncertainty, small non-dispositive omissions, organizational differences, failure to mention a weak secondary argument, minor factual imprecision that does not affect analysis, or a different legally available remedy.

## Evaluation Criteria

All 20 criteria have equal nominal weight: **0.05** each. Conditional criteria are scored only when relevant to the question.

| ID | Criterion | Conditional | Weight | Operational definition |
|---|---|---:|---:|---|
| `relevance` | Relevance | No | 0.05 | Directly addresses the legal question asked, prioritizes legally material issues, and avoids tangents or indiscriminate issue dumping. |
| `legal_accuracy` | Legal Accuracy | No | 0.05 | States and applies the governing legal principles accurately, does not invent law, and avoids material doctrinal, procedural, evidentiary, or interpretive error. |
| `jurisdiction_and_authority` | Jurisdiction and Authority | Yes | 0.05 | When jurisdiction or authority hierarchy matters, identifies the governing system, distinguishes binding from persuasive or secondary authority, and does not import unsupported rules from another jurisdiction. |
| `issue_spotting` | Issue Spotting | Yes | 0.05 | Identifies and prioritizes the material legal issues, threshold questions, defenses, procedural barriers, and secondary issues raised by the facts without listing remote doctrines. |
| `rule_identification` | Rule Identification | Yes | 0.05 | Identifies the correct supplied or otherwise applicable legal rule, framework, elements, factors, exceptions, and burdens necessary to answer the question. |
| `application_to_facts` | Application to Facts | Yes | 0.05 | Connects specific facts to each relevant legal requirement, explains supporting and contrary facts, and does not replace application with conclusory labels. |
| `statutory_interpretation` | Statutory Interpretation | Yes | 0.05 | When statutory or regulatory text is at issue, uses the actual text, defined terms, grammar, context, structure, exceptions, cross-references, and genuine ambiguity without inventing language. |
| `precedent_reasoning` | Precedent Reasoning | Yes | 0.05 | When cases matter, identifies the holding and material facts, distinguishes dicta, assesses hierarchy and procedural posture, and reasons carefully by analogy or distinction. |
| `procedural_reasoning` | Procedural Reasoning | Yes | 0.05 | Applies the standard appropriate to the current procedural stage, respects the roles of judge and factfinder, and correctly handles jurisdiction, timing, preservation, or review where relevant. |
| `element_and_defense_analysis` | Element and Defense Analysis | Yes | 0.05 | Separately addresses every material element, applicable exception, and meaningful defense, and does not treat a prima facie showing as final liability when a defense remains. |
| `evidence_and_fact_reasoning` | Evidence and Fact Reasoning | Yes | 0.05 | Distinguishes allegations, testimony, documents, findings, inferences, assumptions, authentication, admissibility, weight, and established facts as the task requires. |
| `remedies_reasoning` | Remedies Reasoning | Yes | 0.05 | Separates liability from entitlement to relief, correctly applies remedy prerequisites and limitations, and calculates or scopes relief only from supported facts and rules. |
| `counterarguments` | Counterarguments | Yes | 0.05 | Where the issue is genuinely contestable, identifies and fairly analyzes the strongest legally supported competing argument without manufacturing false balance. |
| `assumptions_and_missing_facts` | Assumptions and Missing Facts | Yes | 0.05 | Identifies legally decisive missing facts or rules, labels necessary assumptions, and does not silently fill gaps with unsupported factual or legal premises. |
| `uncertainty_and_qualification` | Uncertainty and Qualification | Yes | 0.05 | Calibrates confidence to the clarity of the law, evidence, and procedural posture; acknowledges ambiguity, conflicting authority, or unresolved facts where they genuinely matter. |
| `legal_source_reasoning` | Legal Source Reasoning | Yes | 0.05 | Evaluates whether cited or supplied sources are primary or secondary, controlling or persuasive, current, jurisdictionally relevant, and actually supportive of the proposition asserted. |
| `completeness` | Completeness | No | 0.05 | Covers all material parts of the requested legal analysis at a depth proportionate to the question, without omitting a dispositive issue or requested subpart. |
| `consistency` | Consistency | No | 0.05 | Maintains internally consistent facts, rules, burdens, dates, calculations, and conclusions throughout the answer. |
| `clarity` | Clarity | No | 0.05 | Presents legal reasoning in a clear, organized, precise, and readable way; terminology serves the analysis rather than substituting for it. |
| `practical_legal_judgment` | Practical Legal Judgment | Yes | 0.05 | When practical judgment is called for, distinguishes stronger from weaker legal positions based on authority and facts, flags consequential deadlines or verification needs, and avoids generic disclaimers in place of analysis. |

## Score Anchors

### 4 — Strong legal reasoning

The answer is legally accurate, directly addresses the question, identifies the governing rule or authority, applies it carefully to the supplied facts, and reaches a well-supported and appropriately qualified conclusion.

Where relevant, it correctly handles jurisdiction, authority hierarchy, elements, exceptions, defenses, procedural posture, burdens, statutory text, precedent, evidence, remedies, counterarguments, and uncertainty. For ambiguous questions, it identifies competing interpretations and explains why one is stronger or why the issue remains unresolved. For fact patterns, it distinguishes established facts from allegations or assumptions. For procedural questions, it applies the standard appropriate to the stage. For statutory questions, it uses the actual supplied language and definitions.

Minor stylistic imperfections or harmless omissions do not reduce a 4.

### 3 — Substantially correct

The answer is substantially correct and useful but contains one meaningful omission, minor doctrinal imprecision, limited factual application, incomplete counterargument, or insufficient qualification. The main legal reasoning and conclusion remain sound.

Typical 3-level weaknesses include one secondary issue omitted, one counterargument underdeveloped, minor ambiguity not fully explored, correct rule with slightly incomplete application, or insufficient discussion of a non-dispositive defense.

### 2 — Partly correct

The answer is directionally useful but has important omissions or reasoning weaknesses. It may identify the correct legal area but miss a material element, state the right rule but apply it superficially, recognize ambiguity but resolve it without adequate support, miss a significant defense, inadequately address procedural posture, or rely too heavily on policy rather than controlling law.

The answer shows meaningful legal understanding but requires substantial improvement.

### 1 — Major noncritical weakness

The answer contains a substantial legal, jurisdictional, procedural, evidentiary, or interpretive misunderstanding, but does not meet the critical-error threshold. It provides only limited useful legal analysis.

### 0 — Fundamental failure

Use 0 for a critical legal error, no answer, an off-topic answer, fabricated controlling authority, or an equivalent fundamental failure.

## Scoring Workflow

1. Check first for `critical_legal_error`. If present, assign **0**.
2. Identify which conditional criteria are actually implicated by the question.
3. Evaluate the answer against the applicable criteria, using the supplied rule, jurisdiction, authority, facts, and procedural posture.
4. Do not penalize a defensible alternative conclusion merely for differing from a reference answer.
5. Do penalize unsupported certainty, invented rules or facts, omitted dispositive elements, ignored exceptions or defenses, and misuse of procedural standards.
6. Assign a single final score from **0–4** using the score anchors above.
