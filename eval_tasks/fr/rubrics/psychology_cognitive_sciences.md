# Psychology & Cognitive Sciences Evaluation Rubric

## Task

Score **ONE answer to ONE Psychology & Cognitive Sciences question from 0–4**.

The question metadata establishes the relevant domain, intent, difficulty, experimental context, and expected reasoning style.

Accept different valid cognitive mechanisms, theories, computational models, and experimental interpretations when the supplied evidence permits them.

Do not require a particular theoretical school, named effect, neural region, or wording unless specified.

Do not reward unnecessary length, jargon, famous-researcher names, or citations.

## Critical Error

If `critical_psychology_cognitive_sciences_error` applies:

**score = 0**

A critical Psychology & Cognitive Sciences error is a materially false central psychological, cognitive, experimental, psychometric, quantitative, neuroscientific, or causal claim that fundamentally invalidates the answer or reverses the substantive conclusion.

Examples include:

- fundamentally confusing negative reinforcement with punishment when central to the answer;
- claiming extinction necessarily erases the original learned association;
- reversing an encoding-versus-retrieval conclusion through fundamentally incorrect memory reasoning;
- treating memory confidence as equivalent to memory accuracy when central;
- claiming correlation proves a psychological mechanism;
- inferring causality from an observational design despite a clearly supplied confound;
- claiming activation in one region proves that region uniquely implements a cognitive function;
- using reverse inference as definitive proof of a mental state;
- treating fMRI timing as if it had millisecond temporal precision;
- fundamentally confusing reliability with validity;
- treating a highly reliable test as necessarily valid;
- ignoring a clearly dominant speed–accuracy trade-off and interpreting faster responses as better cognition;
- claiming a nonsignificant result proves no effect despite clearly inadequate power;
- treating a significant p-value as proof of theoretical importance;
- treating group-average differences as deterministic for every individual;
- inferring intelligence, mental disorder, or cognitive impairment from sparse anecdotal evidence;
- treating trained-task improvement as proof of broad cognitive transfer;
- materially incorrect base-rate or conditional-probability reasoning that changes the conclusion;
- treating a fitted model parameter as a directly observed psychological process;
- claiming one task uniquely and purely measures one cognitive construct when supplied evidence contradicts that claim.

Do **not** classify as critical: minor arithmetic errors that do not change the conclusion, small terminology mistakes, reasonable differences among cognitive theories, legitimate alternative models, cautious uncertainty, minor omissions, reasonable approximations, defensible interpretations of ambiguous evidence, small disagreements about effect magnitude, or minor historical attribution errors.

## Score Anchors

### 4

The answer is psychologically and cognitively accurate, directly addresses the question, correctly interprets relevant behavioral or neural evidence, applies an appropriate mechanism or model, recognizes important assumptions and measurement limitations, and reaches a sound conclusion. Experimental answers distinguish observation from causal inference; measurement answers distinguish reliability, validity, construct, and error; neuroscience answers avoid unjustified reverse inference; ambiguous evidence is handled with plausible alternatives and calibrated certainty.

### 3

The answer is substantially correct and useful but contains one meaningful omission, minor psychological imprecision, limited reasoning gap, non-central quantitative error, or insufficient qualification regarding evidence, measurement, or alternatives. The central conclusion remains sound.

### 2

The answer is partly correct or directionally useful but has important omissions, incomplete cognitive reasoning, weak experimental interpretation, significant psychometric or quantitative problems, or inadequate consideration of competing mechanisms. Meaningful correction is required.

### 1

The answer contains a substantial psychology/cognitive-science misunderstanding, inappropriate mechanism, major non-critical experimental error, weak interpretation, or badly incomplete analysis, but does not meet the critical-error threshold.

### 0

Critical Psychology & Cognitive Sciences error, no answer, off-topic answer, unrelated fabrication, or equivalent fundamental failure.

## Criterion Application

Evaluate only criteria relevant to the question.

A memory question need not discuss developmental psychology. A psychometric question need not discuss neuroimaging. A qualitative conceptual question need not contain quantitative calculations.

Conditional criteria should not penalize answers for omitting irrelevant material. However, when a conditional issue is necessary for correctness, failure on that issue should affect the score.

## Evaluation Criteria

| ID | Name | Weight | Conditional | Definition |
|---|---|---:|:---:|---|
| `relevance` | Relevance | 0.05 | No | Directly addresses the question asked, uses the supplied evidence, and avoids irrelevant material. The conclusion must answer the stated intent rather than substitute a nearby topic. |
| `psychological_accuracy` | Psychological Accuracy | 0.05 | No | Uses psychological principles correctly; distinguishes behavior from inferred constructs, learning from temporary performance, confidence from accuracy, motivation from performance, trait tendencies from single behaviors, and group-level patterns from individual conclusions. |
| `cognitive_science_accuracy` | Cognitive Science Accuracy | 0.05 | No | Represents cognitive-science concepts and levels of explanation accurately; avoids treating one task as process-pure, one theoretical framework as universally established, or descriptive model fit as proof of psychological truth. |
| `conceptual_precision` | Conceptual Precision | 0.05 | No | Distinguishes construct from operationalization, observation from inference, description from explanation, association from mechanism, and theoretical variables from the measures used to estimate them. |
| `cognitive_mechanism_reasoning` | Cognitive Mechanism Reasoning | 0.05 | Yes | When mechanism is relevant, explains a coherent chain from stimulus/task through representation or process to observable response, states assumptions, and considers alternative mechanisms or discriminating predictions rather than merely naming a construct. |
| `perception_attention_and_memory_reasoning` | Perception, Attention, and Memory Reasoning | 0.05 | Yes | When relevant, correctly separates sensation from perception, attention from awareness, working memory from long-term memory, and encoding from storage and retrieval; incorporates interference, cueing, task demands, and speed-accuracy considerations. |
| `learning_and_decision_reasoning` | Learning and Decision Reasoning | 0.05 | Yes | When relevant, correctly analyzes conditioning, reinforcement, extinction, prediction error, exploration/exploitation, habit versus goal-directed control, risk, utility, framing, and learning versus short-lived performance change. |
| `language_and_reasoning_analysis` | Language and Reasoning Analysis | 0.05 | Yes | When relevant, accurately analyzes lexical access, ambiguity, context, syntax/semantics/pragmatics, deductive versus inductive reasoning, validity versus truth, and probabilistic versus certain conclusions. |
| `developmental_and_individual_difference_reasoning` | Developmental and Individual-Difference Reasoning | 0.05 | Yes | When relevant, accounts for age-appropriate task demands, strategy, prior knowledge, expertise, exposure, variability, and measurement equivalence; avoids deterministic inference from demographic or group averages. |
| `quantitative_correctness` | Quantitative Correctness | 0.05 | Yes | When quantitative reasoning is required, selects the relevant quantities, performs material calculations correctly, interprets probabilities and effect estimates appropriately, and separates minor arithmetic slips from errors that alter the substantive conclusion. |
| `experimental_design_reasoning` | Experimental Design Reasoning | 0.05 | Yes | When relevant, identifies independent/dependent variables, controls, randomization, counterbalancing, order and expectancy effects, task demands, manipulation checks, and the strongest causal conclusion justified by the design. |
| `psychometric_and_measurement_reasoning` | Psychometric and Measurement Reasoning | 0.05 | Yes | When relevant, distinguishes reliability from validity, latent construct from indicator, and test score from trait; recognizes measurement error, range restriction, floor/ceiling effects, norm dependence, response bias, and measurement non-equivalence. |
| `statistical_and_data_interpretation` | Statistical and Data Interpretation | 0.05 | Yes | When relevant, interprets means, variability, correlations, effect sizes, confidence intervals, statistical power, significance, interactions, multiple comparisons, regression, and uncertainty without confusing statistical evidence with theoretical or causal importance. |
| `neuroscience_reasoning` | Neuroscience Reasoning | 0.05 | Yes | When relevant, distinguishes neural correlation, necessity, sufficiency, and causal perturbation; respects temporal/spatial-resolution limits, distributed processing, network effects, lesion limitations, and avoids definitive reverse inference. |
| `computational_and_model_reasoning` | Computational and Model Reasoning | 0.05 | Yes | When relevant, evaluates assumptions, qualitative and quantitative predictions, fit, complexity, parameter identifiability, generalization, and competing models; fitted parameters are treated as model-dependent estimates rather than directly observed psychological entities. |
| `causal_inference_and_alternative_explanations` | Causal Inference and Alternative Explanations | 0.05 | Yes | When relevant, separates association, prediction, explanation, and intervention; identifies confounds, temporal assumptions, regression to the mean, mediation limits, and plausible alternative mechanisms with discriminating evidence. |
| `uncertainty_and_evidence_limits` | Uncertainty and Evidence Limits | 0.05 | Yes | When relevant, calibrates certainty to design and measurement quality, recognizes ambiguity and statistical uncertainty, allows reasonable alternative cognitive models, and states what the supplied evidence cannot establish. |
| `completeness` | Completeness | 0.05 | No | Covers the material reasoning steps needed for a correct answer, including central comparisons, assumptions, requested calculations, qualifications, and alternative explanations when necessary, without requiring irrelevant conditional content. |
| `consistency` | Consistency | 0.05 | No | Maintains internally compatible claims, calculations, terminology, and conclusions; later statements do not contradict earlier reasoning or the supplied data. |
| `clarity` | Clarity | 0.05 | No | Communicates the reasoning in a precise, readable, appropriately structured way, with enough explanation to show why the conclusion follows and without using jargon as a substitute for reasoning. |

## Cognitive-Mechanism Answers

Strong answers should identify, where relevant:

- input or stimulus;
- psychological process;
- relevant representation;
- intermediate mechanism;
- behavioral consequence;
- competing mechanisms;
- discriminating evidence.

Do not reward merely naming a construct.

## Experimental Answers

Strong answers should identify, where appropriate:

- hypothesis;
- independent variable;
- dependent variable;
- experimental control;
- confounders;
- randomization;
- counterbalancing;
- task demands;
- alternative explanations;
- what conclusions the design supports.

Do not claim causal evidence when the design is observational.

## Psychometric Answers

Strong answers should consider, where relevant:

- construct definition;
- reliability;
- validity;
- measurement error;
- norming;
- range;
- floor/ceiling effects;
- response bias or possible bias;
- measurement equivalence when groups are compared.

Do not treat test scores as error-free direct measurements of a latent trait.

## Neuroscience Answers

Strong answers should distinguish:

- behavioral effect;
- neural correlate;
- causal evidence;
- localization;
- network-level interpretation;
- temporal/spatial-resolution limits.

Do not treat brain localization as complete psychological explanation.

## Computational-Model Answers

Strong answers should consider:

- model assumptions;
- predictions;
- fit;
- complexity;
- identifiability;
- alternative models;
- out-of-sample prediction where relevant.

Do not interpret model parameters as literal psychological entities without supporting evidence.

## Quantitative Answers

Strong answers should, where relevant:

- select the relevant measure;
- use consistent units;
- perform calculations correctly;
- recognize speed–accuracy trade-offs;
- interpret probabilities correctly;
- consider measurement error;
- interpret the result psychologically.

Do not require every trivial arithmetic step.

## Data-Interpretation Answers

Strong answers should:

- identify what the data show;
- identify what remains uncertain;
- distinguish descriptive from causal claims;
- compare relevant conditions;
- consider effect magnitude;
- consider measurement quality;
- identify alternative explanations.

## Important Evaluation Principles

### Behavior is evidence, not direct access to mental processes
Psychological mechanisms are inferred from patterns of behavior and other measurements.

### Constructs are not measurements
Working memory, attention, intelligence, motivation, and similar constructs are theoretical variables operationalized imperfectly by tasks and scales.

### Correlation is not causation
Association does not establish psychological mechanism.

### Neural activation is not explanation by itself
A brain-imaging difference does not automatically explain cognition.

### Reverse inference requires caution
Observing activation associated with a task does not prove a particular mental process occurred.

### Reliability is not validity
A measure may produce highly consistent results while measuring the wrong construct.

### Confidence is not accuracy
Confidence and performance should be evaluated separately.

### Learning is not temporary performance
Short-term improvement during practice may not imply durable learning or transfer.

### Training is not broad transfer
Improvement on trained tasks does not automatically generalize to unrelated cognitive abilities.

### Attention is not awareness
These constructs overlap but are not interchangeable.

### One task rarely measures one pure process
Cognitive tasks typically involve multiple abilities and strategies.

### Reaction time requires accuracy context
Faster responses can reflect better processing, a lower response threshold, guessing, or strategic trade-offs.

### Group averages do not determine individuals
Population-level effects should not be treated as deterministic individual characteristics.

### Heritability is not immutability
A heritable psychological trait can still be influenced by environment and development.

### Developmental differences require age-appropriate measurement
Poorer task performance can reflect task demands rather than absence of the underlying competence.

### Statistical significance is not theoretical importance
A small robust effect may be theoretically meaningful; a statistically significant effect may still be practically or theoretically weak.

### Nonsignificance is not proof of no effect
Power and precision matter.

### Models are approximations
Computational and cognitive models should be evaluated by assumptions, fit, prediction, and discriminating evidence.

### Multiple models can fit the same data
Model fit alone does not establish psychological truth.

### Levels of explanation differ
Behavioral, computational, neural, developmental, and functional explanations answer related but non-identical questions.

### Replication requires interpretation
Replication evidence depends on power, population, measurement, procedure, and effect heterogeneity.

### No theory worship
Do not reward allegiance to one psychological school or model.

### No neuroscience worship
Neural measurements do not automatically outrank behavioral evidence.

### No jargon worship
Psychological terminology without valid reasoning should not receive high credit.

### No citation worship
Citations do not compensate for incorrect psychological, methodological, or causal reasoning.
