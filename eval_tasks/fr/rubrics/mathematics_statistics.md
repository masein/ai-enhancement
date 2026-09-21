# Mathematics & Statistics Evaluation Rubric

## Task

Score **ONE answer to ONE Mathematics & Statistics question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, style, and mathematical/statistical context.

Accept different valid mathematical derivations, proof strategies, approximations, or statistical procedures when they satisfy the stated assumptions and requirements. Do not require a particular wording, notation, or structure.

Do not reward unnecessary length, jargon, formula dumping, or citations. The score should reflect reasoning quality and correctness, not resemblance to a reference answer.

## Critical Error

If `critical_mathematics_statistics_error` applies:

**score = 0**

A critical error is a materially false central mathematical, probabilistic, or statistical claim that fundamentally invalidates the answer or reverses its substantive conclusion. It includes fatal algebra, division by zero, decisive extraneous roots, false theorems, fatal proof steps, conclusion-changing differentiation/integration errors, fundamentally wrong probability or conditional-probability reasoning, unjustified independence that determines the answer, central p-value/confidence-interval misinterpretation, correlation-as-causation, failure-to-reject-as-proof, invalid inferential procedures, unjustified global-optimum claims, finite examples presented as universal proof, or fabricated exact answers to non-identifiable problems.

It does **not** include harmless rounding, minor arithmetic that leaves the conclusion unchanged, notation differences, minor terminology mistakes, a small omitted justification in otherwise sound reasoning, an alternative correct solution, a defensible alternative statistical procedure, clearly labeled reasonable approximation, minor non-material imprecision, or failure to simplify a correct expression.

## Score Anchors

### 4

The answer is mathematically and statistically accurate, directly addresses the question, chooses an appropriate method, performs relevant calculations correctly, and reaches a justified conclusion.

Where assumptions matter, it identifies or respects them. For proof questions, the logical argument establishes the claimed result. For probability questions, conditionality and dependence are handled correctly. For statistical questions, uncertainty and inferential limitations are interpreted appropriately. For applied questions, the mathematical result is translated back into the context correctly.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor mathematical/statistical imprecision, limited reasoning gap, incomplete justification, or insufficient qualification. The main conclusion remains correct.

Examples include an omitted but fairly obvious intermediate step, minor rounding or arithmetic issues that do not affect the conclusion, insufficient discussion of an assumption, or a mostly correct statistical interpretation with a modest qualification missing.

### 2

The answer is partly correct or directionally useful but has important omissions or reasoning weaknesses.

Examples include a correct setup but incomplete solution; a partially correct proof with a nonfatal gap; correct calculations but materially weak interpretation; a statistical procedure used without addressing an important assumption; mixing up related concepts without completely invalidating the answer; or obtaining some correct intermediate results but failing to synthesize them correctly.

### 1

The answer contains a substantial mathematical, probabilistic, or statistical misunderstanding, inappropriate method, major non-critical calculation error, or serious logical gap, but does not meet the critical-error threshold. The response shows limited useful understanding of the problem.

### 0

Critical mathematics/statistics error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criteria

All 20 criteria have equal weight **0.05**. Conditional criteria are applied only when the question or answer calls for them.

| ID | Criterion | Conditional | Weight | Operational definition |
|---|---|---:|---:|---|
| `relevance` | Relevance | No | 0.05 | Addresses the mathematical or statistical task actually asked, uses the supplied information, and avoids substituting a different problem. |
| `mathematical_accuracy` | Mathematical Accuracy | No | 0.05 | Uses mathematically valid definitions, transformations, identities, theorem conditions, and conclusions; respects domains, signs, units, and exact-versus-approximate distinctions. |
| `statistical_accuracy` | Statistical Accuracy | Yes | 0.05 | When statistical claims, data, sampling, inference, or uncertainty are involved, states and interprets them correctly, including distinctions among parameters, statistics, estimators, standard deviations, and standard errors. |
| `logical_reasoning` | Logical Reasoning | No | 0.05 | The chain of reasoning validly supports the conclusion, with no circular step, quantifier error, unjustified implication, or necessary-condition/sufficient-condition confusion. |
| `conceptual_precision` | Conceptual Precision | No | 0.05 | Uses the relevant concepts with sufficient precision to distinguish closely related ideas such as continuity versus differentiability, correlation versus causation, or rank versus matrix size. |
| `quantitative_correctness` | Quantitative Correctness | No | 0.05 | Calculations, algebraic values, probabilities, estimates, units, signs, and rounding are correct enough to support the conclusion; harmless rounding is accepted. |
| `algebraic_reasoning` | Algebraic Reasoning | Yes | 0.05 | When algebra is required, manipulations preserve equivalence or clearly track one-way implications, domain restrictions, zero divisors, inequality direction, and possible extraneous solutions. |
| `proof_reasoning` | Proof Reasoning | Yes | 0.05 | When proof is required, the argument establishes the claim over the stated domain, explicitly uses needed assumptions, and treats examples as evidence rather than universal proof unless a counterexample is sufficient to refute. |
| `probability_reasoning` | Probability Reasoning | Yes | 0.05 | When probability is involved, events, conditioning, complements, dependence, independence, expectations, and distributions are handled according to the stated sample space and assumptions. |
| `statistical_inference` | Statistical Inference | Yes | 0.05 | When inference is required, p-values, confidence intervals, power, errors, estimators, sampling distributions, and multiple-testing issues are interpreted consistently with the inferential framework and procedure assumptions. |
| `modeling_and_translation` | Modeling and Translation | Yes | 0.05 | When a verbal or applied problem must be modeled, defines suitable quantities, translates relationships and constraints correctly, maintains dimensional consistency, and maps mathematical results back to context. |
| `method_selection` | Method Selection | Yes | 0.05 | When more than one technique is plausible, selects a method appropriate to the structure, assumptions, information available, and requested conclusion rather than mechanically applying a familiar formula. |
| `assumptions` | Assumptions | Yes | 0.05 | Identifies, respects, or challenges assumptions that materially affect validity; does not silently add assumptions to force a unique answer when the information is insufficient. |
| `uncertainty_and_interpretation` | Uncertainty and Interpretation | Yes | 0.05 | When uncertainty matters, characterizes sampling, parameter, predictive, numerical, or model uncertainty appropriately and avoids stronger certainty than the evidence supports. |
| `data_interpretation` | Data Interpretation | Yes | 0.05 | When data summaries, tables, plots, or regression outputs are involved, extracts the relevant numerical relationships and interprets them without being misled by outliers, aggregation, scales, or omitted context. |
| `error_analysis` | Error Analysis | Yes | 0.05 | When the task involves critique or approximation, identifies material mathematical, statistical, numerical, measurement, or reasoning errors and explains their effect on the result. |
| `completeness` | Completeness | No | 0.05 | Provides the steps, cases, qualifications, checks, or conclusions needed to answer all material parts of the question without requiring unnecessary verbosity. |
| `consistency` | Consistency | No | 0.05 | Intermediate results, notation, assumptions, signs, units, and final conclusions do not contradict one another. |
| `clarity` | Clarity | No | 0.05 | Presents the reasoning in a readable order with notation and prose precise enough for an evaluator to follow and verify the argument. |
| `result_interpretation` | Result Interpretation | Yes | 0.05 | When interpretation is required, explains what the mathematical or statistical result means in the original context and states material limitations without overclaiming. |

## Evaluation Principles

### Mathematical validity
Every step supporting the central conclusion must be mathematically valid. A correct final number obtained through invalid reasoning should not automatically receive full credit.

### Method selection
The model should identify an appropriate mathematical or statistical method rather than mechanically applying a familiar formula.

### Equivalent methods
Different correct approaches must receive full credit. Do not require the reference solution's exact derivation, notation, or sequence of steps.

### Assumptions
Strong answers identify assumptions when they materially affect the result. Do not invent unstated assumptions when the question is underdetermined.

### Insufficient information
If a problem cannot be uniquely solved from the information supplied, recognizing that insufficiency may be the correct answer. Do not reward fabricated numerical answers.

### Proof
A proof must establish the claimed conclusion for the required domain. Examples alone do not prove a universal statement. A single valid counterexample is sufficient to refute a universal statement.

### Necessary vs sufficient
Do not treat a necessary condition as sufficient unless equivalence has been established.

### Exact vs approximate
Distinguish exact results from numerical approximations. Reasonable rounding is acceptable when appropriate.

### Probability
Probability answers must respect conditionality and dependence. Do not assume independence unless it is given or justified.

### Expected values
An expected value is a long-run or distributional quantity and need not be an outcome that actually occurs.

### Statistical inference
Inference must reflect sampling uncertainty and the assumptions of the procedure used.

### P-values
A p-value is not the probability that the null hypothesis is true.

### Confidence intervals
Confidence intervals must be interpreted according to the framework used. Do not automatically treat a frequentist confidence interval as a posterior probability interval.

### Statistical significance
Statistical significance does not by itself establish practical, scientific, or business importance.

### Causality
Association, regression coefficients, or predictive accuracy do not by themselves establish causation. Strong causal claims require an appropriate identification strategy or assumptions.

### Model assumptions
Results from mathematical or statistical models are conditional on their assumptions. Where material, strong answers should discuss sensitivity to assumption violations.

### Model fit
Good in-sample fit does not guarantee correct specification, causality, or good out-of-sample prediction.

### Uncertainty
Statistical estimates, forecasts, and predictions should communicate material uncertainty when relevant.

### No formula worship
Writing down the correct formula without applying it appropriately should not receive a high score.

### No jargon worship
Using sophisticated mathematical or statistical terminology without correct reasoning should not receive a high score.

### No citation worship
Citations do not compensate for incorrect mathematical or statistical reasoning. External citations should generally be unnecessary for self-contained mathematics questions.
