# Medicine & Clinical Health Evaluation Rubric

## Task

Score **ONE answer to ONE Medicine & Clinical Health question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, acuity, clinical context, and expected reasoning style.

Accept different valid diagnostic approaches and treatment plans where more than one is medically reasonable. Do not require a particular wording or answer structure. Do not reward unnecessary length, jargon, defensive over-testing, or citations.

For criterion scoring, evaluate only criteria that are applicable to the question. All criteria have equal base weight (`0.05`). If conditional criteria are not applicable, exclude them rather than penalizing the answer; normalize the weighted average across the criteria that do apply. The final holistic score should follow the score anchors below.

## Critical Error

If `critical_medicine_clinical_health_error` applies:

**score = 0**

A critical error is a materially false or unsafe central medical claim that would fundamentally mislead clinical reasoning, miss a time-critical condition, recommend a clearly dangerous intervention, reverse the substantive conclusion, or create a major foreseeable risk.

Examples include: failure to recognize an obvious emergency; dangerous delay in stroke, myocardial infarction, sepsis, anaphylaxis, shock, or severe respiratory failure; a clearly contraindicated treatment; a major medication-dose or unit error; a major anticoagulation error; a fundamentally wrong interpretation of a life-threatening electrolyte, acid-base, ECG, pregnancy, or glycemic emergency; or treating unstable physiology as reassuring.

Do **not** classify as critical: minor arithmetic errors without management impact, small terminology errors, minor omissions, reasonable guideline differences, different valid treatment choices, conservative uncertainty, small safe follow-up differences, reasonable diagnostic alternatives, minor harmless over-testing, stylistic differences, or omitted nonessential detail.

## Score Anchors

### 4

The answer is medically accurate, directly addresses the question, correctly prioritizes the clinical problem, applies appropriate diagnostic and management reasoning, interprets relevant data correctly, recognizes urgent risks, accounts for important patient-specific factors, and reaches a safe and clinically sound conclusion.

For emergency questions, it identifies instability and prioritizes immediate stabilization.

For diagnostic questions, it appropriately weighs likely and dangerous alternatives.

For treatment questions, it considers contraindications, risks, benefits, and monitoring.

For uncertain cases, it calibrates confidence appropriately.

### 3

The answer is substantially correct and clinically useful but contains one meaningful omission, minor medical imprecision, limited reasoning gap, incomplete differential, modestly suboptimal test choice, or insufficient qualification.

The central diagnosis or management approach remains safe and substantially correct.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete clinical reasoning, weak prioritization, significant test-interpretation problems, poor treatment justification, or insufficient recognition of relevant risks.

The answer may require meaningful correction before use.

### 1

The answer contains a substantial medical misunderstanding, poor differential prioritization, major non-critical treatment error, inappropriate investigation strategy, or badly incomplete reasoning, but does not meet the critical-error threshold.

### 0

Critical medicine/clinical health error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question.

A diagnostic question may not require treatment discussion. A treatment question with an established diagnosis may not require a broad differential. A routine prevention question need not include emergency triage. Conditional criteria should not penalize answers for omitting irrelevant material.

However, when a conditional issue is necessary for safe and correct management, failure on that issue should affect the score.

## Evaluation Criteria

| ID | Criterion | Conditional | Definition |
|---|---|---:|---|
| `relevance` | Relevance | No | Directly addresses the clinical question, requested task, and supplied context without drifting into unrelated material or substituting a different problem. |
| `medical_accuracy` | Medical accuracy | No | Uses medically sound facts and distinctions; does not confuse symptoms with diagnoses, risk factors with disease, abnormal tests with confirmed disease, or screening with diagnosis. |
| `clinical_reasoning` | Clinical reasoning | No | Connects history, examination, physiology, tests, time course, and context into a coherent argument. A correct label reached through unsafe or unsupported reasoning should score lower than a justified conclusion. |
| `diagnostic_reasoning` | Diagnostic reasoning | Yes | When diagnosis is part of the task, identifies the leading diagnosis or diagnostic category, uses supporting and opposing evidence, recognizes missing information, and avoids keyword-matching or unjustified certainty. |
| `differential_prioritization` | Differential prioritization | Yes | When alternatives matter, ranks likely and dangerous possibilities and explains which need urgent exclusion; does not substitute an indiscriminate list for prioritization. |
| `triage_and_urgency` | Triage and urgency | Yes | When acuity matters, correctly distinguishes routine, urgent, emergency, and immediate-stabilization needs. Missing an obvious time-critical emergency or delaying stabilization is a major failure and may trigger the critical flag. |
| `management_and_treatment` | Management and treatment | Yes | When management is requested, selects clinically appropriate therapy or supportive care, sequences actions sensibly, allows reasonable alternatives, and does not delay necessary treatment for nonessential testing. |
| `medication_safety` | Medication safety | Yes | When medications are relevant, considers indication, allergy, contraindications, interactions, renal/hepatic function, pregnancy, age, weight, route, monitoring, and adverse effects as applicable. A dangerous contraindicated recommendation should be heavily penalized and may be critical. |
| `quantitative_correctness` | Quantitative correctness | Yes | When calculations are required, uses the correct equation, arithmetic, units, and clinically plausible interpretation. Minor arithmetic slips differ from major dose or unit errors that materially change management. |
| `investigation_selection` | Investigation selection | Yes | When testing is part of the task, chooses investigations that reduce meaningful uncertainty or change management, respects urgency and pretest probability, and avoids unnecessary duplication or delaying stabilization. |
| `test_and_data_interpretation` | Test and data interpretation | Yes | When laboratory, ECG, imaging, physiologic, or diagnostic-test data are supplied, interprets them in context and respects test limitations rather than treating an abnormal or negative result as automatically definitive. |
| `risk_benefit_reasoning` | Risk-benefit reasoning | Yes | When decisions involve trade-offs, compares expected benefit, harm, burden, competing risks, and consequences of action versus inaction rather than optimizing one metric in isolation. |
| `evidence_based_reasoning` | Evidence-based reasoning | Yes | When evidence is relevant, distinguishes association from causation, relative from absolute effect, statistical from clinical significance, surrogate from patient-important outcomes, and trial evidence from patient-specific applicability. |
| `uncertainty_calibration` | Uncertainty calibration | Yes | When evidence is incomplete or conflicting, states what is known, what remains uncertain, and what would reduce uncertainty. Appropriate caution should receive credit; unjustified diagnostic certainty should be penalized. |
| `patient_specific_reasoning` | Patient-specific reasoning | Yes | When relevant, accounts for comorbidities, physiology, function, goals, preferences, adherence, social context, treatment burden, pregnancy, age, and other supplied characteristics that alter the decision. |
| `safety_netting_and_follow_up` | Safety-netting and follow-up | Yes | When ongoing care or outpatient uncertainty is relevant, gives a reassessment plan, monitoring needs, treatment-failure criteria, and warning signs that should trigger escalation. |
| `communication_and_clarity_of_plan` | Communication and clarity of plan | No | Presents the clinical reasoning and plan in a usable sequence, uses plain language when communicating with patients, distinguishes facts from uncertainty, and avoids coercive or needlessly technical framing. |
| `completeness` | Completeness | No | Covers the clinically important parts of the requested task, especially any element necessary for safety, while not requiring exhaustive irrelevant detail. |
| `consistency` | Consistency | No | Maintains internal consistency across diagnosis, interpretation, urgency, treatment, and follow-up; recommendations do not contradict supplied facts or one another. |
| `clarity` | Clarity | No | Explains the answer precisely and intelligibly, with priorities and causal logic easy to follow; jargon or citations do not substitute for reasoning. |

## Diagnostic Answers

Strong diagnostic answers should, where appropriate:

- summarize the clinical problem
- identify the leading diagnosis
- identify dangerous alternatives
- explain supporting and opposing evidence
- identify missing information
- select useful tests
- calibrate certainty

Do not require exhaustive differentials.

## Treatment Answers

Strong treatment answers should, where appropriate:

- identify the treatment objective
- select appropriate therapy
- identify contraindications
- consider interactions
- consider organ function
- consider patient preferences
- explain monitoring
- identify escalation criteria

Do not require unnecessary treatment when observation or supportive care is appropriate.

## Emergency Answers

Strong emergency answers should:

1. recognize instability
2. prioritize immediate threats
3. initiate appropriate stabilization
4. obtain focused diagnostic information
5. begin time-sensitive treatment
6. arrange appropriate disposition

An answer should not receive full credit if it reaches the correct final diagnosis but delays necessary stabilization.

## Investigation Answers

Strong answers should choose tests that:

- address meaningful diagnostic uncertainty
- can change management
- fit the clinical context
- account for pretest probability
- avoid unnecessary duplication

More testing is not automatically better.

## Medication Answers

Strong answers should consider, where relevant:

- indication
- dose when requested
- route
- contraindications
- interactions
- renal function
- hepatic function
- pregnancy
- age
- weight
- monitoring
- adverse effects

A safe alternative medication strategy should receive full credit when clinically reasonable. Do not invent a precise dose when essential dosing information is missing.

## Evidence-Based Answers

Strong answers should distinguish:

- relative effect from absolute effect
- benefit from harm
- association from causation
- statistical significance from clinical importance
- surrogate outcomes from demonstrated patient benefit
- efficacy from effectiveness
- study population from patient-specific applicability

## Patient-Centered Answers

Where relevant, strong answers should include:

- patient goals
- preferences
- function
- burden of treatment
- quality of life
- uncertainty
- shared decision-making

Do not require a preference-sensitive treatment choice to have one universal correct answer.

## Important Evaluation Principles

### Stabilize before perfect diagnosis
In emergencies, immediate stabilization can be more important than complete diagnostic certainty.

### Dangerous diagnoses matter
A less likely but immediately dangerous diagnosis may need urgent exclusion before a more likely benign condition.

### Probability is not certainty
Clinical diagnosis is probabilistic. Strong answers calibrate certainty to the evidence.

### Tests modify probability
Positive and negative tests should be interpreted in the context of pretest probability, sensitivity, specificity, and limitations.

### Abnormal is not automatically disease
Reference-range abnormalities may be incidental, physiologic, transient, clinically insignificant, or secondary to another process.

### Treat the patient, not the number
Laboratory values and physiologic measurements should be interpreted in clinical context.

### Medication safety
Treatment recommendations should consider allergies, contraindications, interactions, renal function, hepatic function, age, pregnancy, weight, and monitoring requirements when relevant.

### Multimorbidity
Optimizing one disease in isolation can harm another condition. Strong answers reconcile competing risks.

### Evidence is patient-specific
Population-level evidence informs but does not replace patient-specific clinical judgment.

### Screening is not diagnosis
A screening result modifies risk; it does not necessarily establish disease.

### Association is not causation
Observational associations do not automatically prove treatment effects or mechanisms.

### Statistical significance is not clinical importance
A statistically detectable effect may be too small to matter clinically.

### Shared decision-making
Preference-sensitive choices should reflect expected benefits, expected harms, uncertainty, patient goals, and treatment burden.

### Follow-up matters
A treatment plan should include reassessment or escalation criteria when clinically important.

### Safety-netting
For uncertain outpatient presentations, strong answers identify warning symptoms or changes requiring urgent reassessment.

### No diagnostic overconfidence
Do not reward answers that confidently diagnose beyond the evidence.

### No guideline worship
Guidelines support clinical reasoning but do not eliminate patient-specific judgment.

### No citation worship
Citations do not compensate for unsafe or incorrect medical reasoning.
