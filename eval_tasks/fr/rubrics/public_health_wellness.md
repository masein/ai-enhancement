# Public Health & Wellness Evaluation Rubric

## Task

Score **ONE answer to ONE Public Health & Wellness question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, acuity, population, context, and assumptions. Accept different valid epidemiologic interpretations, prevention approaches, program strategies, or wellness recommendations when they are supported by the supplied evidence.

Do not require a particular wording or structure. Do not reward unnecessary length, jargon, or citations. The benchmark evaluates population-health reasoning, prevention, evidence interpretation, risk communication, wellness reasoning, and practical public-health judgment rather than resemblance to a reference answer.

## Critical Error

If `critical_public_health_wellness_error` applies, **score = 0** regardless of the rest of the answer.

A materially false central epidemiologic, quantitative, causal, screening, prevention, risk, or wellness claim fundamentally invalidates the answer or creates a materially unsafe or misleading conclusion.

Examples include:
- Fundamentally confusing incidence and prevalence in a way that changes the conclusion.
- Using a materially wrong denominator that reverses a population comparison.
- Fundamentally misinterpreting sensitivity, specificity, or predictive value.
- Claiming that a positive screening test automatically proves disease.
- Claiming that increased survival time after diagnosis necessarily proves reduced mortality.
- Fundamentally misinterpreting absolute versus relative risk.
- Claiming that correlation alone establishes a public-health cause.
- Making a major quantitative error that changes the substantive conclusion.
- Using fundamentally incorrect outbreak reasoning that drives the central recommendation.
- Recommending an obviously unsafe wellness practice as the central answer.
- Presenting a pseudoscientific mechanism as established fact when it is central to the answer.
- Claiming statistical significance automatically establishes practical or health importance.
- Claiming absence of statistical significance proves no effect.
- Fundamentally misrepresenting a study design in a way that invalidates the inference.
- Ignoring a clearly stated confounder that invalidates the claimed causal conclusion.
- Treating a population-level association as proof of an individual's outcome.
- Fabricating evidence required to support the answer.
- Asserting certainty where the supplied evidence is clearly insufficient.
- Fundamentally misinterpreting hazard, exposure, and risk.
- Recommending an intervention based on a materially wrong effect estimate or denominator.

Do **not** classify the following as critical by themselves:
- Minor arithmetic errors that do not change the conclusion.
- Harmless rounding.
- Minor terminology mistakes.
- Small omissions.
- Defensible differences in public-health strategy.
- Reasonable uncertainty.
- Different valid program designs.
- Legitimate disagreements about policy trade-offs.
- Alternative evidence-consistent, low-risk wellness approaches.
- Minor imprecision that does not materially affect the population-health conclusion.

## Evaluation Criteria

All 20 criteria have equal weight: **0.05** each. Conditional criteria are applied only when relevant to the question.

| ID | Criterion | Weight | Conditional | Operational definition |
|---|---|---:|:---:|---|
| `relevance` | Relevance | 0.05 | No | Directly addresses the question asked, prioritizes the supplied public-health or wellness context, and avoids material digressions that do not help answer the task. |
| `public_health_accuracy` | Public-Health Accuracy | 0.05 | No | Uses correct public-health concepts and distinctions, including incidence versus prevalence, risk versus rate, mortality versus case fatality, prevention versus treatment, surveillance, burden, and population-level interpretation where relevant. |
| `wellness_accuracy` | Wellness Accuracy | 0.05 | No | When wellness guidance is relevant, keeps it evidence-informed, proportionate, feasible, low-risk, and appropriately general; avoids miracle claims, unsupported universal prescriptions, overmedicalization, and pseudoscientific mechanisms. |
| `epidemiologic_reasoning` | Epidemiologic Reasoning | 0.05 | Yes | Correctly identifies the relevant population, time period, exposure, outcome, numerator, denominator, and epidemiologic measure; distinguishes association from causation and population measures from individual conclusions. |
| `conceptual_precision` | Conceptual Precision | 0.05 | No | Uses key concepts with their correct meaning and does not substitute jargon for mechanism; makes material distinctions explicitly when the question depends on them. |
| `quantitative_correctness` | Quantitative Correctness | 0.05 | Yes | When calculations or numeric interpretation are required, chooses the correct measure and denominator, performs arithmetic correctly, preserves units and time frames, and interprets the result consistently with the calculation. |
| `study_design_reasoning` | Study-Design Reasoning | 0.05 | Yes | When study design is relevant, explains why a design fits or fails the research question, including temporal ordering, comparison structure, randomization or lack of it, bias, loss to follow-up, and internal/external validity as applicable. |
| `screening_and_risk_interpretation` | Screening and Risk Interpretation | 0.05 | Yes | When screening or risk is relevant, correctly handles sensitivity, specificity, predictive values, prevalence, false positives/negatives, lead-time bias, length bias, overdiagnosis, baseline risk, and absolute versus relative effects as applicable. |
| `causal_reasoning` | Causal Reasoning | 0.05 | Yes | When causal claims are at issue, assesses temporality, plausible mechanisms, confounding, bias, reverse causation, mediation, effect modification, alternative explanations, consistency, and intervention evidence without treating correlation alone as proof. |
| `population_health_reasoning` | Population-Health Reasoning | 0.05 | Yes | When relevant, distinguishes population effects from individual prognosis, considers exposure prevalence and total burden, recognizes heterogeneity, and explains how small average effects can create large population benefits. |
| `evidence_evaluation` | Evidence Evaluation | 0.05 | Yes | When claims depend on evidence, weighs study design, sample size, measurement quality, replication, conflicts or incentives where relevant, effect size, uncertainty, generalizability, and the difference between plausibility and demonstrated benefit. |
| `health_behavior_and_communication` | Health Behavior and Communication | 0.05 | Yes | When behavior or communication is relevant, accounts for motivation, capability, opportunity, environment, social context, adherence, trust, numeracy, framing, plain language, and the need to inform rather than manipulate. |
| `assumptions` | Assumptions | 0.05 | Yes | Makes material assumptions explicit, does not invent missing facts, and distinguishes conclusions that follow from the prompt from those that require additional information. |
| `uncertainty_calibration` | Uncertainty Calibration | 0.05 | Yes | Matches confidence to the strength and precision of evidence, identifies important unknowns, avoids false certainty or false reassurance, and states what additional evidence would materially change the assessment. |
| `equity_and_context` | Equity and Context | 0.05 | Yes | When relevant, considers subgroup differences, access barriers, differential exposure or vulnerability, intervention reach, local context, and whether average improvement masks stable or widening disparities; separates empirical from normative claims. |
| `systems_and_program_reasoning` | Systems and Program Reasoning | 0.05 | Yes | When interventions or systems are relevant, links identified problem to mechanism and measurable outcome while considering reach, adoption, fidelity, capacity, bottlenecks, incentives, spillovers, scalability, sustainability, and unintended consequences. |
| `completeness` | Completeness | 0.05 | No | Covers the material parts of the task at an appropriate depth, including required calculations, comparisons, qualifications, or decision factors, without requiring exhaustive treatment of irrelevant details. |
| `consistency` | Consistency | 0.05 | No | Maintains internally consistent definitions, calculations, assumptions, causal claims, and conclusions; later statements do not materially contradict earlier reasoning. |
| `clarity` | Clarity | 0.05 | No | Explains the reasoning in understandable, well-organized language, using technical terms only when they improve precision and defining or translating them when needed. |
| `practical_public_health_reasoning` | Practical Public-Health Reasoning | 0.05 | Yes | When action or prioritization is required, proposes feasible steps that match the level of evidence and urgency, protects against material harms, considers implementation constraints, and avoids individualized diagnosis or prescribing unless explicitly appropriate. |

## Score Anchors

### 4

The answer is accurate, directly addresses the question, applies appropriate public-health or wellness reasoning, and reaches a well-supported conclusion.

Where relevant, it correctly handles population versus individual reasoning, incidence and prevalence, denominators, screening, absolute and relative risk, study design, bias and confounding, causal inference, uncertainty, health behavior, equity, program implementation, and evidence quality.

For quantitative questions, calculations and interpretations are correct. For screening questions, it accounts for prevalence, test properties, and relevant biases. For program questions, it considers mechanism, reach, implementation, context, and unintended consequences. For wellness questions, recommendations are evidence-informed, proportionate, feasible, and appropriately qualified. For underdetermined questions, it identifies what cannot be concluded and what additional evidence would be needed.

Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor epidemiologic imprecision, limited reasoning gap, incomplete qualification, or insufficient consideration of an important secondary factor. The main conclusion remains correct.

### 2

The answer is partly correct or directionally useful but has important omissions, weak epidemiologic reasoning, incomplete interpretation, questionable assumptions, poor consideration of bias/confounding, or insufficient risk communication. It shows meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial public-health, wellness, epidemiologic, quantitative, causal, screening, or evidence-evaluation misunderstanding, but does not meet the critical-error threshold. The response provides limited useful understanding.

### 0

Critical Public Health & Wellness error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Important Evaluation Principles

### Population vs individual

Population-level evidence does not automatically determine an individual outcome. Individual anecdotes do not establish population effects.

### Incidence vs prevalence

Incidence measures new occurrence. Prevalence measures existing burden. Do not treat them as interchangeable.

### Association vs causation

Association alone does not establish causation. Strong causal conclusions require appropriate design, evidence, assumptions, and/or mechanism.

### Absolute vs relative risk

Interpret relative effects alongside baseline and absolute risk whenever material.

### Screening

Earlier detection does not automatically improve outcomes. Consider lead-time bias, length bias, overdiagnosis, false positives, false negatives, and downstream consequences.

### Statistical significance

Statistical significance does not by itself establish public-health importance, and lack of significance does not prove no effect.

### Evidence quality

Study design, bias, confounding, sample size, measurement, replication, and generalizability matter.

### Prevention

Preventive interventions can operate at individual, community, environmental, system, and population levels.

### Population benefit

A small effect applied broadly can generate substantial total population benefit.

### Health behavior

Behavior depends on knowledge and motivation, but also capability, opportunity, environment, constraints, and social context.

### Social determinants

Health outcomes are influenced by social and environmental conditions as well as individual behavior.

### Equity

Improving the population average does not necessarily narrow disparities.

### Program effectiveness

Reach, implementation, adherence, fidelity, capacity, and context can determine real-world effectiveness.

### Wellness

General wellness guidance should be evidence-informed, proportionate, feasible, low-risk, and should not overmedicalize ordinary variation.

### Risk communication

Use appropriate denominators, absolute effects, uncertainty, and context; avoid sensationalism.

### Uncertainty

Confidence should reflect the evidence. Incomplete evidence does not justify fabricated certainty.

### Multiple valid interventions

Public-health and wellness problems can have several defensible interventions; evaluate reasoning, evidence, constraints, trade-offs, and implementation.

### No jargon worship

Technical terminology without correct reasoning should not receive a high score.

### No citation worship

Citations do not compensate for incorrect epidemiologic or public-health reasoning.

## Application Notes

- Evaluate the reasoning actually needed by the question; do not penalize omission of concepts that are irrelevant to that question.
- A technically correct term without a correct mechanism should not receive high credit.
- A cited answer can still be wrong; an uncited answer can still be correct when the task does not require citation.
- Do not require current regulations, schedules, policies, or recommendations unless the prompt explicitly supplies the needed jurisdiction and reference frame.
- Do not reward individualized diagnosis or prescribing when the task is about population health, prevention, community health, or general wellness.
- For emergency public-health scenarios, prioritize immediate risk reduction, affected populations, exposure control, communication, surveillance, and explicit uncertainty.
- For resource-allocation or policy scenarios, multiple answers may be defensible. Evaluate objectives, expected benefit, opportunity cost, equity, feasibility, implementation, unintended consequences, autonomy where relevant, and uncertainty.
