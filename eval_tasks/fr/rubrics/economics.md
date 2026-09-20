# Economics Evaluation Rubric

## Task

Score ONE answer to ONE economics question from 0–4.

The reference/question metadata establishes what concepts, mechanisms, assumptions, distinctions, domain, intent, difficulty, and context matter.

Do not require a particular wording or structure.

Do not reward:
- unnecessary length
- jargon for its own sake
- citations merely because they are present

A short, precise answer can receive 4.

## Critical Error

If `critical_economic_error` applies:

**score = 0**

Otherwise evaluate the applicable criteria.

A critical economic error is a materially false central economic claim, calculation, or causal mechanism that fundamentally misleads the user. Examples include reversing a fundamental economic relationship, a major calculation error that changes the substantive conclusion, claiming correlation establishes causation when causality is central, or describing a policy mechanism in the fundamentally wrong direction.

Do **not** classify as critical:
- minor imprecision
- small arithmetic errors that do not change the conclusion
- reasonable disagreement between economic schools when assumptions are explicitly stated
- legitimate model simplifications
- omissions that do not reverse the answer

## Score Anchors

### 4
The answer is substantively correct, directly answers the question, uses appropriate economic concepts and mechanisms, handles important assumptions and qualifications, and reaches a sound conclusion. It covers the material components needed for the question.

### 3
The answer is substantially correct and useful but has one meaningful omission, minor imprecision, limited reasoning gap, or insufficient qualification.

### 2
The answer is partly correct or directionally useful but leaves important concepts, mechanisms, assumptions, or implications missing or unclear.

### 1
The answer contains a substantial misunderstanding, weak reasoning, or inappropriate model/framework, but does not meet the critical-error threshold.

### 0
Critical economic error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Evaluation Guidance

- Do not require citations unless the question itself requires evidence.
- Citations do not compensate for incorrect reasoning.
- Policy questions do not require the evaluator to prefer a particular policy; evaluate the economic mechanisms, effects, assumptions, trade-offs, and distributional consequences.
- Empirical questions should distinguish association from causal identification.
- Do not penalize an answer simply because it does not use a particular economic model if another valid model is appropriately justified.
- Do not reward verbosity.
- Conditional criteria should be applied only when relevant to the question.
- Evaluate answers against the question's difficulty and intended scope; a concise answer can score 4 if it covers the material components.
- When multiple mechanisms are plausible, reward answers that distinguish them and state what assumptions determine which mechanism dominates.

## Criteria

The benchmark contains 20 equally weighted criteria. Each criterion has weight 0.05. The platform scores each applicable criterion from 0–1 and converts the aggregate to the final 0–4 score.

| ID | Criterion | Conditional | What to look for |
|---|---|---:|---|
| relevance | Relevance | No | Directly answers the task and prioritizes material information. |
| economic_accuracy | Economic Accuracy | No | Correct facts, relationships, calculations, and mechanisms. |
| reasoning | Reasoning | No | Coherent reasoning from assumptions/evidence to conclusion. |
| conceptual_precision | Conceptual Precision | No | Accurate distinctions and terminology. |
| causal_identification | Causal Identification | Yes | Causal claims are identified rather than inferred from association alone. |
| model_selection | Model Selection | Yes | Appropriate model/framework for the setting. |
| assumptions | Assumptions | Yes | Material assumptions are recognized and used correctly. |
| quantitative_correctness | Quantitative Correctness | Yes | Calculations and numerical interpretation are correct. |
| equilibrium_and_dynamics | Equilibrium and Dynamics | Yes | Adjustment, timing, feedback, and short/long-run effects are handled correctly. |
| welfare_analysis | Welfare Analysis | Yes | Efficiency, surplus, deadweight loss, distribution, and externalities are handled when relevant. |
| policy_analysis | Policy Analysis | Yes | Policy mechanisms, incidence, trade-offs, and unintended effects are handled when relevant. |
| empirical_interpretation | Empirical Interpretation | Yes | Estimates, uncertainty, and empirical evidence are interpreted appropriately. |
| contextualisation | Contextualisation | No | Relevant market, institutional, behavioral, and temporal context is recognized. |
| scope_and_qualifications | Scope and Qualifications | No | Important limits, exceptions, and boundary conditions are stated. |
| uncertainty_calibration | Uncertainty Calibration | No | Certainty is proportional to the evidence and assumptions. |
| alternative_explanations | Alternative Explanations | Yes | Plausible competing mechanisms are considered when needed. |
| completeness | Completeness | No | Material requested components are covered. |
| consistency | Consistency | No | Assumptions, calculations, terminology, and conclusions remain internally consistent. |
| clarity | Clarity | No | Clear, understandable, appropriately structured presentation. |
| actionability | Actionability | Yes | Practical implications or next steps are useful when the question calls for them. |

## Final Scoring

If the critical error flag applies, the final score is **0** regardless of other criteria.

Otherwise, score each applicable criterion from 0 to 1, apply its 0.05 weight, and convert the aggregate to the platform's final 0–4 scale.

The evaluator should judge the answer itself, not whether it resembles a reference answer word-for-word.
