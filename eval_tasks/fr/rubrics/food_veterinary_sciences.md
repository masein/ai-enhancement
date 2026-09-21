# Food & Veterinary Sciences Evaluation Rubric

## Task

Score ONE answer to ONE Food & Veterinary Sciences question from **0–4**.

The question metadata establishes the relevant domain, intent, difficulty, acuity, species or food context, and assumptions. Accept different valid diagnostic approaches, food-processing methods, epidemiologic interpretations, or management strategies when supported by the supplied evidence. Do not require a particular wording or structure. Do not reward unnecessary length, jargon, or citations.

The benchmark evaluates scientific understanding, clinical reasoning, food-safety reasoning, population reasoning, and professional judgment rather than resemblance to a reference answer.

## Critical Error

If `critical_food_veterinary_sciences_error` applies, **score = 0**. Otherwise evaluate the applicable criteria.

A critical error is a materially false central food-science, food-safety, veterinary, epidemiologic, diagnostic, toxicologic, quantitative, or One Health claim that fundamentally invalidates the answer or creates a materially unsafe or misleading conclusion.

Examples include: treating visibly normal food as necessarily safe; confusing infection and intoxication in a way that changes control; treating pasteurization as sterilization when central; a major lethality, dose, concentration, or epidemiologic calculation error that reverses the conclusion; a fundamentally wrong diagnostic-test interpretation; failure to recognize a clear veterinary emergency; describing antimicrobial resistance as the animal becoming genetically resistant; ignoring a decisive species difference; or fabricating key findings.

Do **not** classify as critical: harmless rounding, minor terminology mistakes, small omissions, reasonable uncertainty, defensible alternative differentials, or different valid management/processing strategies that remain scientifically sound.

## Evaluation Criteria

### `relevance` — Relevance (Generally applicable; weight 0.05)

Directly addresses the question asked, prioritizes information that affects the requested conclusion, and avoids tangents that do not help resolve the food, veterinary, population, or One Health problem.

### `food_science_accuracy` — Food-science accuracy (Conditional; weight 0.05)

When food science is relevant, states food chemistry, microbiology, processing, preservation, quality, nutrition, and food-safety principles correctly, including distinctions such as water activity vs moisture, quality vs safety, pasteurization vs sterilization, and hazard vs risk.

### `veterinary_accuracy` — Veterinary accuracy (Conditional; weight 0.05)

When veterinary content is relevant, states species-appropriate anatomy, physiology, pathology, infectious-disease, nutrition, pharmacology, toxicology, reproduction, welfare, and production principles correctly without importing unsupported human assumptions.

### `clinical_reasoning` — Clinical reasoning (Conditional; weight 0.05)

For clinical scenarios, builds an appropriate problem representation, identifies urgent threats, develops plausible differentials, avoids premature closure, and links findings to the next diagnostic or management step.

### `conceptual_precision` — Conceptual precision (Generally applicable; weight 0.05)

Uses key concepts with their correct meaning and keeps important distinctions clear, including sign vs diagnosis, exposure vs disease, prevalence vs incidence, concentration vs dose, detection vs absence, and association vs causation.

### `quantitative_correctness` — Quantitative correctness (Conditional; weight 0.05)

When calculation is required, chooses the correct quantities and denominator, preserves units and dimensions, performs arithmetic correctly, rounds reasonably, and interprets the result in context.

### `diagnostic_reasoning` — Diagnostic reasoning (Conditional; weight 0.05)

When tests or diagnostic evidence are involved, interprets them in context of sensitivity, specificity, prevalence, pre-test probability, false positives/negatives, sampling, and the distinction between screening and confirmation.

### `food_safety_reasoning` — Food-safety reasoning (Conditional; weight 0.05)

When food safety is involved, correctly identifies the relevant hazard, exposure pathway, susceptible population where relevant, control measure, residual risk, and distinction between prevention, detection, and post-process contamination.

### `epidemiologic_reasoning` — Epidemiologic reasoning (Conditional; weight 0.05)

When population data are involved, uses correct epidemiologic measures, denominators, comparison groups, time frames, and causal caution, while considering bias, confounding, clustering, surveillance, and sampling as relevant.

### `mechanistic_explanation` — Mechanistic explanation (Conditional; weight 0.05)

When mechanism is requested or material, explains how and why the relevant chemical, microbial, physiologic, toxicologic, pharmacologic, behavioral, or transmission process leads to the observed outcome rather than merely naming a term.

### `risk_analysis` — Risk analysis (Conditional; weight 0.05)

When risk is central, separates hazard from risk and evaluates exposure, dose, probability, severity, susceptibility, uncertainty, and risk-reduction options in a logically coherent way.

### `species_and_context_awareness` — Species and context awareness (Conditional; weight 0.05)

Recognizes when species, life stage, production system, food matrix, route, environment, or management context materially changes interpretation, and avoids unsupported transfer across species or settings.

### `assumptions` — Assumptions (Conditional; weight 0.05)

Makes necessary assumptions explicit, keeps them consistent with supplied facts, and does not invent missing clinical, laboratory, process, regulatory, or exposure details required to force a conclusion.

### `evidence_and_uncertainty` — Evidence and uncertainty (Conditional; weight 0.05)

Matches confidence to evidence quality, acknowledges conflicting or incomplete data, identifies what information would change the conclusion, and distinguishes established findings from plausible inferences.

### `population_health_reasoning` — Population-health reasoning (Conditional; weight 0.05)

When herd, flock, farm, food-lot, or population questions are involved, reasons at the correct unit of analysis, uses appropriate denominators and comparison groups, and considers transmission, prevention, management, and population consequences.

### `one_health_reasoning` — One Health reasoning (Conditional; weight 0.05)

When applicable, explains concrete pathways linking animal, human, food, and environmental health, including zoonotic transmission, antimicrobial resistance, food-chain exposure, reservoirs, vectors, or environmental pathways rather than merely invoking the label.

### `completeness` — Completeness (Generally applicable; weight 0.05)

Covers the material parts of the task at the level needed for a defensible answer, including requested calculations, comparisons, caveats, alternatives, or decision factors without requiring unnecessary length.

### `consistency` — Consistency (Generally applicable; weight 0.05)

Maintains internally consistent facts, calculations, assumptions, terminology, and conclusions; later claims do not contradict earlier reasoning or supplied evidence.

### `clarity` — Clarity (Generally applicable; weight 0.05)

Communicates the reasoning in a clear, organized, and unambiguous way appropriate to the question, with enough explanation that the evaluator can follow how the conclusion was reached.

### `practical_decision_reasoning` — Practical decision reasoning (Conditional; weight 0.05)

For action-oriented questions, identifies objectives, urgency, evidence, feasible options, likely mechanisms, risks, trade-offs, constraints, information gaps, and follow-up or verification needs without pretending that one strategy is uniquely correct when several are defensible.

## Important Evaluation Principles

### Food safety vs food quality

Food that looks, smells, or tastes acceptable is not necessarily microbiologically safe. Food quality and food safety are related but distinct.

### Hazard vs risk

A hazard is a potential source of harm. Risk depends on probability, exposure, dose, susceptibility, and consequence.

### Infection vs intoxication

Foodborne infection and foodborne intoxication have different mechanisms and may require different controls.

### Process control

A food-safety process must be judged by whether it controls the relevant hazard under actual operating conditions.

### Multiple hurdles

Microbial control may depend on combinations of temperature, pH, water activity, preservatives, atmosphere, and storage time.

### Species matters

Veterinary conclusions may differ materially across species. Do not automatically transfer human or one-species findings to another species.

### Differential diagnosis

A differential diagnosis is not a confirmed diagnosis. Strong clinical answers discuss or prioritize plausible alternatives when appropriate.

### Diagnostic tests

Diagnostic results must be interpreted in context. Sensitivity, specificity, prevalence, pre-test probability, sampling, and confirmatory testing can matter.

### Treatment reasoning

Do not reward unsupported treatment recommendations. When exact dosing is required, supplied information must be used correctly with coherent units.

### Population vs individual

An intervention suitable for one animal may not be the optimal herd/flock strategy, and the reverse can also be true.

### Epidemiology

Association does not by itself establish causation. Denominators, sampling, bias, confounding, time frame, and unit of analysis matter.

### Antimicrobial resistance

Antimicrobial exposure can create selection pressure in microbial populations. Do not describe the animal itself as becoming genetically resistant to the antimicrobial.

### Welfare

Animal welfare includes health, comfort, pain/distress, behavioral needs, and affective state where assessable. Productivity alone is not a complete welfare measure.

### One Health

Animal, human, food, and environmental health can be interconnected. Strong answers explain the pathway rather than merely naming One Health.

### Uncertainty

Clinical, epidemiologic, and food-safety evidence may be incomplete or conflicting. Confidence should match the strength of evidence.

### Multiple valid decisions

Veterinary management and food-processing questions can have more than one defensible solution. Evaluate reasoning, evidence, constraints, risk, and trade-offs rather than requiring one predetermined strategy.

### No jargon worship

Technical terminology without correct scientific reasoning should not receive a high score.

### No citation worship

Citations do not compensate for incorrect food or veterinary reasoning.

## Score Anchors

### 4

The answer is scientifically accurate, directly addresses the question, applies appropriate food-science or veterinary reasoning, and reaches a well-supported conclusion. Where relevant, it correctly handles food-safety mechanisms, clinical findings, differential diagnosis, diagnostic tests, species differences, quantitative calculations, epidemiology, population health, biosecurity, welfare, uncertainty, and One Health interactions. Clinical answers distinguish plausible differentials from confirmed diagnoses and appropriately prioritize urgent issues. Food-safety answers identify the relevant hazard, exposure/control pathway, and risk. Population answers use appropriate denominators and distinguish individual-animal from herd/flock reasoning. Decision answers consider objectives, constraints, risks, trade-offs, and uncertainty. Minor stylistic imperfections or harmless rounding do not reduce a 4.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor scientific imprecision, limited clinical or epidemiologic gap, incomplete qualification, or insufficient discussion of an important secondary factor. The main conclusion remains correct and safe.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete mechanistic reasoning, weak differential diagnosis, incorrect interpretation of some relevant factors, or insufficient risk analysis. It shows meaningful understanding but requires substantial improvement.

### 1

The answer contains a substantial food-science, veterinary, diagnostic, quantitative, epidemiologic, or One Health misunderstanding, but does not meet the critical-error threshold. The response provides limited useful understanding.

### 0

Critical Food & Veterinary Sciences error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Application Notes

- Apply only criteria relevant to the question. Conditional criteria should not penalize an answer when the underlying capability is not called for.
- Correct terminology with incorrect reasoning should not receive high credit.
- Citations do not compensate for incorrect reasoning.
- Do not require current regulations, product labels, withdrawal times, or legal thresholds unless the question supplies them or explicitly makes jurisdiction relevant.
- Where evidence permits multiple defensible decisions, score the quality of reasoning rather than agreement with one preferred strategy.
- For quantitative items, reasonable rounding receives full credit when it does not change the conclusion.
